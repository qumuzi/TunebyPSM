import random
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from _lib.experiment.models import *
import os
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from _lib.experiment.idhp_data import *
from _lib.semi_parametric_estimation.ate import *
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct
from warnings import simplefilter
from _lib.modellearnertutorial.learner.mlearner import sample_random, stepwise_feature_selection
from _lib.semi_parametric_estimation.helpers import *
import warnings
warnings.filterwarnings("ignore")

ac_dict = {'sky': 'Skylake', 'has': 'Haswell', 'stg': 'Storage'}
Outcome = "Throughput"
Step = 1
TuningTime = 3
TestTime = 1

Workloads = ["als","bayes","kmeans","linear","lr","nweight","pagerank","rf","terasort","wordcount"]

def _get_t_flags(flag_list:list):
    ac_list = []
    for ac_0 in flag_list:
        tar_list = flag_list.copy()
        tar_list.remove(ac_0)
        ac_list.append([ac_0,tar_list])
    return ac_list

def _split_output(yt_hat, y_scaler):
    t_len = yt_hat.shape[1]/2
    q_list = []
    g_list = []
    for t_id in np.arange(t_len-1):
        q_list.append(y_scaler.inverse_transform(yt_hat[:, int(t_id)].copy().reshape(1, -1)))
        g_list.append(yt_hat[:, int(t_id+t_len)].copy())
    q_list.append(y_scaler.inverse_transform(yt_hat[:, int(t_len-1)].copy().reshape(1, -1)))
    eps = yt_hat[:, -1][0]

    return q_list, g_list, eps

def _read_data(base_path, tune_id, flag_list:list, t_list:list, len_list:list):
    if len(flag_list) != len(t_list) or len(flag_list) != len(len_list):
        print("Wrong Input in Read Data")
        return 0

    for id in np.arange(len(flag_list)):
        file_path_0 = base_path+"Step"+str(Step)+"_"+flag_list[id]+str(tune_id)+".csv"
        data_0 = np.loadtxt(file_path_0, delimiter=',')[:len_list[id]]
        ids0, y0, x0 = data_0[:, 0], data_0[:, 1][:, None], data_0[:, 2:]
        t0 = np.array([[t_list[id]]]*len_list[id])
        if id == 0:
            x,y,t,ids = x0, y0, t0, ids0
        else:
            x = np.concatenate((x,x0),axis=0)
            y = np.concatenate((y,y0),axis=0)
            t = np.concatenate((t,t0),axis=0)
            ids = np.concatenate((ids,ids0),axis=0)
    return t.reshape(-1, 1), y, x, ids

def _opti_feat(X_id: pd.DataFrame, Y, features, opti, step, n_iter):

    X_opti = X_id[np.append(features,"id")].drop_duplicates(subset=features, keep="first")
    if step*n_iter >= X_opti.shape[0]:
        return X_opti[features].values, Y[X_opti["id"]]
    def _opti_bo(X, Y, n_step, n_iter):
        X_res = Y_res = []

        X_re = X.reset_index(drop=True, inplace=False)
        pool_idx_bo = range(X_re.shape[0])
        seed_idx_bo = np.random.randint(0, high=X_re.shape[0]-1, size=n_step, dtype='l').tolist()
        pool_idx_bo = list(set(pool_idx_bo)-set(seed_idx_bo))

        kernel = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e1))
        model=GaussianProcessRegressor(kernel=kernel, alpha=1)

        for i in range(n_iter):
            ## Fit GP model
            X_bo_tr, Y_bo_tr = X_re.values[seed_idx_bo], Y[seed_idx_bo]
            mean_bo, std_bo = model.fit(X_bo_tr, Y_bo_tr).predict(X_re.values, return_std=True)
            ## Get co reward function
            re_bo = mean_bo
            re_bo_tmp = std_bo.copy()
            re_bo_tmp[seed_idx_bo] = -1000
            new_idx = np.argsort(re_bo_tmp)[-n_step:].tolist()
            ## Update pool indices and seed indices
            pool_idx_bo = list(set(pool_idx_bo)-set(new_idx))
            seed_idx_bo = seed_idx_bo + new_idx
        return X_bo_tr, Y_bo_tr
    if opti == "bo":
        return _opti_bo(X_opti[features], Y[X_opti["id"]], step, n_iter)

def _psi_naive(q_list, g_list, t_flags, truncate_level=0.):
    q_best = 0
    q_best_id = ""
    for q_id in np.arange(len(q_list)):
        q_mean = np.mean(truncate_by_all_g(q_list[q_id], g_list, level=truncate_level))
        #q_mean = np.max(truncate_by_all_g(q_list[q_id], g_list, level=truncate_level))
        if q_mean > q_best:
            q_best = q_mean
            q_best_id = t_flags[q_id]
    return q_best_id


def get_estimate(q_list, g_list, eps, t_flags, truncate_level=0.01):
    """
    getting the back door adjustment & TMLE estimation
    """
    def _reshape(x):
        return list(map(lambda a:a.reshape(-1, 1),x))
    t_best = _psi_naive(_reshape(q_list), _reshape(g_list), t_flags, truncate_level=truncate_level)
    return t_best

def train_and_predict_multidragons(t_train, y_unscaled, x_train, x_test,t_flags, knob_loss=dragonnet_loss_binarycross,
                                   val_split=0.2, batch_size=64):
    verbose = 0
    y_scaler = StandardScaler().fit(y_unscaled)
    y_train = y_scaler.transform(y_unscaled)
    yt_train = np.concatenate([y_train, t_train], 1)

    dragonnet = make_multidragonnet(x_train.shape[1], 0.01, len(t_flags))

    #metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    loss = knob_loss

    import time
    start_time = time.time()

    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]

    sgd_lr = 1e-5
    momentum = 0.9

    dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss)
    #metrics=metrics)
    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    #print("***************************** elapsed_time is: ", elapsed_time)

    #yt_hat_train = dragonnet.predict(x_train)
    yt_hat_test = dragonnet.predict(x_test,verbose=verbose)
    K.clear_session()
    q_list, g_list, eps = _split_output(yt_hat_test, y_scaler)

    t_best = get_estimate(q_list, g_list, eps, t_flags, truncate_level=0.01)
    return t_best


def run_linear_pred(data_base_dir='SingleEnvTrack/generate/als/bo/', source_flag='stg', target_flags=None,
                    workload="als", iter_dict=None):
    TuneAlgo = "rlr"
    simple_errors = []
    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    Y = {}
    Y[source_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[source_flag], workload)
                                 ,index_col=0)[Outcome].values
    Y_best = max(Y[source_flag])
    for tar_flag in target_flags:
        Y[tar_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], workload)
                                  ,index_col=0)[Outcome].values
        Y_best = max((Y_best,max(Y[tar_flag])))

    for idx in np.arange(TuningTime):
        t0, y0, x0_train, ids0 = _read_data(data_base_dir+TuneAlgo+'/', idx, [source_flag],[0],[iter_dict["source"]])
        rlr = Ridge()
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        grid_search = GridSearchCV(rlr, param_grid, n_jobs = 1, verbose=0)
        y_scaler = StandardScaler().fit(y0)
        y_opti_true = y_scaler.transform(y0)
        pred_ex = grid_search.fit(x0_train, y_opti_true).predict(x_test)
        y0_test = y_scaler.inverse_transform(pred_ex.reshape(-1, 1))

        y_best_pred = max(y0_test)
        y_best_test = Y[source_flag][np.argmax(y0_test)]

        for tar_flag in target_flags:

            t1, y1, x1_train, ids1 = _read_data(data_base_dir+TuneAlgo+'/', idx, [tar_flag],[1],[iter_dict["target"]])
            y1_opti_true = y_scaler.transform(y1)
            y0_train = grid_search.predict(x1_train).reshape(-1, 1)
            reg = LinearRegression().fit(y0_train, y1_opti_true)
            pred_ex = reg.predict(y0_test.reshape(-1, 1))
            y1_test = y_scaler.inverse_transform(pred_ex.reshape(-1, 1))
            if y_best_pred < max(y1_test):
                y_best_pred = max(y1_test)
                y_best_test = Y[tar_flag][np.argmax(y1_test)]

        err = (Y_best-y_best_test)/Y_best
        simple_errors.append(err)

    simple_error = np.mean(simple_errors)

    return simple_error

def run_rf_pred(data_base_dir='SingleEnvTrack/generate/als/bo/', source_flag='stg', target_flags=None,
                workload="als", iter_dict=None):

    simple_errors = []

    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    Y = {}
    Y[source_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[source_flag], workload)
                                 ,index_col=0)[Outcome].values
    Y_best = max(Y[source_flag])
    for tar_flag in target_flags:
        Y[tar_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], workload)
                                  ,index_col=0)[Outcome].values
        Y_best = max((Y_best,max(Y[tar_flag])))
    for idx in np.arange(TuningTime):
        t0, y0, x0_train, ids0 = _read_data(data_base_dir, idx, [source_flag],[0],[iter_dict["source"]])
        cart = tree.DecisionTreeClassifier()
        cart.fit(x0_train, y0)
        y0_test = cart.predict(x_test)
        y_best_pred = max(y0_test)
        y_best_test = Y[source_flag][np.argmax(y0_test)]

        for tar_flag in target_flags:

            t1, y1, x1_train, ids1 = _read_data(data_base_dir, idx, [tar_flag],[1],[iter_dict["target"]])
            y0_train = cart.predict(x1_train).reshape(-1, 1)
            rfr = RandomForestRegressor(n_estimators=16).fit(y0_train, y1)
            y1_test = rfr.predict(y0_test.reshape(-1, 1))
            if y_best_pred < max(y1_test):
                y_best_pred = max(y1_test)
                y_best_test = Y[tar_flag][np.argmax(y1_test)]

        err = (Y_best-y_best_test)/Y_best
        simple_errors.append(err)

    simple_error = np.mean(simple_errors)

    return simple_error

def run_gaussian_pred(data_base_dir='SingleEnvTrack/generate/als/bo/', t_flags:list = None,
                      workload="als", iter_dict=None):
    TuneAlgo = "rlr"
    simple_errors = []
    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    Y = {}
    Y_best = 0
    for tar_flag in t_flags:
        Y[tar_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], workload)
                                  ,index_col=0)[Outcome].values
        Y_best = max((Y_best,max(Y[tar_flag])))

    XT = np.concatenate((np.concatenate((np.array([[0]]*len(X)), X),axis=1),
                         np.concatenate((np.array([[1]]*len(X)), X),axis=1),
                         np.concatenate((np.array([[2]]*len(X)), X),axis=1)),axis=0)
    Y_sub = np.concatenate(list(Y.values()),axis=0)

    for idx in np.arange(TuningTime):


        init_idx = np.random.randint(0, high=X.shape[0]-1, size=Step, dtype='l').tolist()
        seed_idx_bo = init_idx
        pool_idx_bo = list(set(range(X.shape[0]))-set(init_idx))

        kernel = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e1))
        model=GaussianProcessRegressor(kernel=kernel, alpha=1)

        for i in range(int(iter_dict["random"]/Step)):
            ## Fit GP model
            X_bo_tr, Y_bo_tr = XT[seed_idx_bo], Y_sub[seed_idx_bo]
            id_bo_tmp = seed_idx_bo.copy()
            #sample_idx = _random_sample(X_sub, 20)
            sample_idx = np.random.randint(0, high=XT.shape[0]-1, size=10, dtype='l').tolist()
            X_sample = XT[sample_idx]
            mean_bo, std_bo = model.fit(X_bo_tr, Y_bo_tr).predict(X_sample, return_std=True)
            ## Get co reward function
            re_bo = mean_bo
            re_bo_tmp = std_bo.copy()
            new_idx = np.array(sample_idx)[np.argsort(re_bo_tmp)[-Step:].tolist()].tolist()
            ## Update pool indices and seed indices
            pool_idx_bo = list(set(pool_idx_bo)-set(new_idx))
            seed_idx_bo = seed_idx_bo + new_idx

        top_idx_bo  = np.argmax(Y_bo_tr)
        y_best_test = Y_sub[id_bo_tmp[top_idx_bo]]
        err = (Y_best-y_best_test)/Y_best
        simple_errors.append(err)
    simple_error = np.mean(simple_errors)

    return simple_error

def run_L2S_pred(data_base_dir='SingleEnvTrack/generate/als/bo/', source_flag='stg', target_flags=None,
                 workload="als", iter_dict=None):
    simple_errors = []
    TuneAlgo = "rlr"
    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    X_id = X.copy()
    X_id["id"] = range(0,len(X_id))
    Y = {}
    Y[source_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[source_flag], workload)
                                 ,index_col=0)[Outcome].values
    Y_best = max(Y[source_flag])
    for tar_flag in target_flags:
        Y[tar_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], workload)
                                  ,index_col=0)[Outcome].values
        Y_best = max((Y_best,max(Y[tar_flag])))
    for idx in np.arange(TuningTime):
        t0, y0, x0_train, ids0 = _read_data(data_base_dir+TuneAlgo+'/', idx, [source_flag],[0],[iter_dict["source"]])
        rlr = Ridge()
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        grid_search = GridSearchCV(rlr, param_grid, n_jobs = 1, verbose=0)

        y_scaler = StandardScaler().fit(y0)
        y_opti_true = y_scaler.transform(y0)
        pred_ex = grid_search.fit(x0_train, y_opti_true).predict(x_test)
        y0_test = y_scaler.inverse_transform(pred_ex.reshape(-1, 1))

        params = X.keys()[stepwise_feature_selection(x0_train, y0,verbose=False)].values#, initial_list=[2,6,3,12,9,15])
        if (len(params) == 0):
            params = X.keys().values
        y_best_pred = max(y0_test)
        y_best_test = Y[source_flag][np.argmax(y0_test)]

        for tar_flag in target_flags:

            x1_train, y1 = _opti_feat(X_id, Y[tar_flag], params, "bo", 1, iter_dict["target"])
            x1_test_feat = X[params].values
            rlr1 = Ridge()
            y1_scaler = StandardScaler().fit(y1.reshape(1, -1))
            y1_opti_true = y1_scaler.transform(y1.reshape(1, -1))
            pred_ex = rlr1.fit(x1_train, y1_opti_true.reshape( -1)).predict(x1_test_feat)
            y1_test = y_scaler.inverse_transform(pred_ex.reshape(-1, 1))
            if y_best_pred < max(y1_test):
                y_best_pred = max(y1_test)
                y_best_test = Y[tar_flag][np.argmax(y1_test)]

        err = (Y_best-y_best_test)/Y_best
        simple_errors.append(err)
    simple_error = np.mean(simple_errors)

    return simple_error

def multidragon_pred(data_base_dir='SingleEnvTrack/generate/als/bo/', t_flags:list = None,
                     knob_loss=dragonnet_loss_binarycross, workload="als", iter_dict=None):
    TuneAlgo = "bo"
    simple_errors = []
    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    Y = {}
    Y_best = 0
    for tar_flag in t_flags:
        Y[tar_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], workload)
                                  ,index_col=0)[Outcome].values
        Y_best = max((Y_best,max(Y[tar_flag])))
    for idx in np.arange(TuningTime):
        t, y, x_train, ids = _read_data(data_base_dir+TuneAlgo+'/', idx,
                                        t_flags, list(np.arange(len(t_flags))), [iter_dict["select"]]*len(t_flags))
        y_best_first = max(y)

        t_best = train_and_predict_multidragons(t, y, x_train, x_test,t_flags,
                                                knob_loss=knob_loss,val_split=0.2, batch_size=64)

        _, y_opti, x_opti, _ = _read_data(data_base_dir+TuneAlgo+'/', idx, [t_best], [0], [iter_dict["select"]+iter_dict["addition"]])
        y_best_pred = max(y_opti)
        y_best = max(y_best_first,y_best_pred)
        err = (Y_best-y_best)/Y_best
        simple_errors.append(err)
    simple_errors_baseline = np.mean(simple_errors)

    return simple_errors_baseline

def aver_pred(data_base_dir, t_flags:list = None, workload="als", iter_dict=None):
    TuneAlgo = "bo"
    iter_full_err = []
    rand_select_err = []
    Y = {}
    Y_best = 0
    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    for tar_flag in t_flags:
        Y[tar_flag] = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], workload)
                                  ,index_col=0)[Outcome].values
        Y_best = max((Y_best,max(Y[tar_flag])))
    for idx in np.arange(TuningTime):

        y_pred = 0
        y_best_rand = []
        for t_flag in t_flags:
            _, y, x_train, _ = _read_data(data_base_dir+TuneAlgo+'/', idx,
                                          [t_flag],[0],[iter_dict["average"]])
            if y_pred < max(y):
                y_pred = max(y)

            _, y, x_train, _ = _read_data(data_base_dir+TuneAlgo+'/', idx,
                                          [t_flag],[0],[iter_dict["random"]])
            y_best_rand.append(max(y))

        iter_full_err.append((Y_best-y_pred)/Y_best)
        rand_select_err.append((Y_best-np.mean(y_best_rand))/Y_best)

    return np.mean(iter_full_err), np.mean(rand_select_err)

def turn_knob(data_base_dir, workload, iter_dict):
    T_flags = _get_t_flags(list(ac_dict.keys()))
    multi_env_algos = ["linear_trans", "gaus_trans", "learn2sample", "multidragon", "full_iter", "random_select"]
    algo_cal_dict = {}
    for algo in multi_env_algos:
        algo_cal_dict[algo] = []

    for test_id in np.arange(TestTime):
        for T_flag in T_flags:
            #linear_transform
            algo_cal_dict["linear_trans"].append(
                run_linear_pred(data_base_dir=data_base_dir, source_flag=T_flag[0], target_flags=T_flag[1],
                                workload=workload, iter_dict=iter_dict))

            #gaussian_process_transform
            algo_cal_dict["gaus_trans"].append(
                run_gaussian_pred(data_base_dir=data_base_dir, t_flags=list(ac_dict.keys()),
                                  workload=workload, iter_dict=iter_dict))
            #learning2sample_transform
            algo_cal_dict["learn2sample"].append(
                run_L2S_pred(data_base_dir=data_base_dir, source_flag=T_flag[0], target_flags=T_flag[1],
                             workload=workload, iter_dict=iter_dict))
            #multidragon_tuning
            algo_cal_dict["multidragon"].append(
                multidragon_pred(data_base_dir=data_base_dir, t_flags=list(ac_dict.keys()), workload=workload,
                                 iter_dict=iter_dict))
            #full_iter, random_select
            res_aver_pred = aver_pred(data_base_dir=data_base_dir, t_flags=list(ac_dict.keys()),workload=workload,
                                      iter_dict=iter_dict)

            algo_cal_dict["full_iter"].append(res_aver_pred[0])
            algo_cal_dict["random_select"].append(res_aver_pred[1])
    res_list = {}
    for algo in multi_env_algos:
        res_list[algo] = np.mean(algo_cal_dict[algo])
    return res_list

def main():
    def _cal_iter(a,b):
        iter_dict = {}
        iter_dict["select"] = a
        iter_dict["addition"] = b
        iter_dict["source"] = a + b
        iter_dict["target"] = a
        iter_dict["average"] = int(a+b/len(list(ac_dict.keys())))
        iter_dict["random"] = int(b+a*len(list(ac_dict.keys())))
        return iter_dict

    for a in np.arange(2,41,1):
        print(a)
        for b in np.arange(3, 31, 3):
            print(b)
            iter_dict = _cal_iter(a,b)
            ResABDict = {}
            for workload in Workloads:
                print(workload)
                ResABDict[workload] = []
                res = turn_knob('SingleEnvTrack/generate/'+workload+'/', workload, iter_dict)
                ResABDict[workload].append(res)
                print(res)

            print(ResABDict)
            F = open("MultipleEnvTrack/RQ1&2/"+str(a)+"_"+str(b)+".txt","w")
            F.write(str(ResABDict))
    print("Calculate Done")


if __name__ == '__main__':
    main()
