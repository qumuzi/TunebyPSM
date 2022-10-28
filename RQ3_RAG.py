
from sklearn.linear_model import Ridge
from _lib.experiment.models import *
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from _lib.experiment.idhp_data import *
from _lib.semi_parametric_estimation.helpers import *
import warnings
warnings.filterwarnings("ignore")

ac_dict = {'sky': 'Skylake', 'has': 'Haswell', 'stg': 'Storage'}
Workloads = ["als","bayes","kmeans","linear","lr","nweight","pagerank","rf","terasort","wordcount"]
Step = 1
TuningTime = 3
BaseAddr = ""
Outcome = "Throughput"

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

def _calculate_eps(yt_hat, y_scaler, t, y):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy().reshape(1, -1))
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy().reshape(1, -1))
    g = yt_hat[:, 2].copy()
    y = y_scaler.inverse_transform(y.copy())
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1 # predictions from unperturbed model

    return np.sum(h*(y-full_q)) / np.sum(np.square(h))

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

def train_and_predict_linear(t_train, y_unscaled, x_train, x_test,t_flags):
    verbose = 0
    len_t = int(len(y_unscaled)/len(t_flags))
    xt_train = np.concatenate([x_train, t_train], 1)
    xt_test = np.concatenate([np.concatenate([x_test, [[0]]*len(x_test)], 1),
                              np.concatenate([x_test, [[1]]*len(x_test)], 1),
                              np.concatenate([x_test, [[2]]*len(x_test)], 1),
                              ], 0)

    rlr = Ridge()


    y_scaler = StandardScaler().fit(y_unscaled)
    y_opti_true = y_scaler.transform(y_unscaled)
    pred_ex = rlr.fit(xt_train, y_opti_true).predict(xt_test)
    y_test = y_scaler.inverse_transform(pred_ex.reshape(-1, 1))

    res = {}
    for t_id in np.arange(len(t_flags)):
        y = y_test[t_id*len_t:(t_id+1)*len_t]
        res[t_flags[t_id]] = np.mean(y)


    return res

def train_and_predict_rf(t_train, y_unscaled, x_train, x_test,t_flags):
    verbose = 0
    len_t = int(len(y_unscaled)/len(t_flags))
    xt_train = np.concatenate([x_train, t_train], 1)
    xt_test = np.concatenate([np.concatenate([x_test, [[0]]*len(x_test)], 1),
                              np.concatenate([x_test, [[1]]*len(x_test)], 1),
                              np.concatenate([x_test, [[2]]*len(x_test)], 1),
                              ], 0)
    from sklearn import tree
    cart = tree.DecisionTreeRegressor()


    y_scaler = StandardScaler().fit(y_unscaled)
    y_opti_true = y_scaler.transform(y_unscaled)
    pred_ex = cart.fit(xt_train, y_opti_true).predict(xt_test)
    y_test = y_scaler.inverse_transform(pred_ex.reshape(-1, 1))

    res = {}
    for t_id in np.arange(len(t_flags)):
        y = y_test[t_id*len_t:(t_id+1)*len_t]
        res[t_flags[t_id]] = np.mean(y)


    return res

def train_and_predict_multidragons(t_train, y_unscaled, x_train, x_test,t_flags,
                                   knob_loss=dragonnet_loss_binarycross,
                                   val_split=0.2, batch_size=64):
    verbose = 0
    y_scaler = StandardScaler().fit(y_unscaled)
    y_train = y_scaler.transform(y_unscaled)
    yt_train = np.concatenate([y_train, t_train], 1)


    sgd_lr = 1e-5
    momentum = 0.9
    sgd = SGD(lr=sgd_lr, momentum=momentum, nesterov=True)
    dragonnet = make_multidragonnet(x_train.shape[1], 0.01, 3)

    loss = knob_loss

    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]

    dragonnet.compile(optimizer=sgd, loss=loss)
    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size, verbose=verbose)

    yt_hat_test = dragonnet.predict(x_test,verbose=verbose)
    K.clear_session()
    q_list, g_list, eps = _split_output(yt_hat_test, y_scaler)

    def _reshape(x):
        return list(map(lambda a:a.reshape(-1, 1),x))
    res = {}
    for q_id in np.arange(len(q_list)):
        res[t_flags[q_id]] = np.mean(truncate_by_all_g(_reshape(q_list)[q_id], _reshape(g_list)))

    return res

def run_test(path_dict, Workload, n_iter_sel, t_flags, knob_loss=dragonnet_loss_binarycross):
    X = pd.read_csv(BaseAddr+'OriginData/perf/config_clean.csv', index_col=0)
    X = (X-X.mean())/X.std()
    x_test = X.values
    Y = {}

    AverPerf = {}
    for tar_flag in t_flags:
        Y[tar_flag] = pd.read_csv(BaseAddr+'OriginData/perf/{}/{}.csv'.format(ac_dict[tar_flag], Workload)
                                  ,index_col=0)[Outcome].values
        AverPerf[tar_flag] = np.mean(Y[tar_flag])

    def _cal_err(input_dict):

        err = 0
        for key1 in list(AverPerf.keys()):
            for key2 in list(AverPerf.keys()):
                if key1 != key2:
                    err += abs(abs(AverPerf[key1]-AverPerf[key2])-abs(input_dict[key1]-input_dict[key2])) \
                           /(2*3*abs(AverPerf[key1]-AverPerf[key2]))
        return err

    cal_tmp = {
        "Aver": [],
        "Linear": [],
        "Cart" : [],
        "MulDra": []
    }
    for idx in np.arange(TuningTime):
        t0, y0, x_train0, ids0 = _read_data(path_dict[0], idx,
                                            [t_flags[0]], [0], [n_iter_sel])
        t1, y1, x_train1, ids1 = _read_data(path_dict[1], idx,
                                            [t_flags[1]], [1], [n_iter_sel])
        t2, y2, x_train2, ids2 = _read_data(path_dict[2], idx,
                                            [t_flags[2]], [2], [n_iter_sel])
        t = np.concatenate((t0,t1,t2),axis=0)
        x_train = np.concatenate((x_train0,x_train1,x_train2),axis=0)
        y = np.concatenate((y0,y1,y2),axis=0)

        aver_perf = {}
        for i in np.arange(len(t_flags)):
            aver_perf[t_flags[i]] = np.mean(y[i*n_iter_sel:(i+1)*n_iter_sel])
        cal_tmp["Aver"].append(_cal_err(aver_perf))

        cal_tmp["MulDra"].append(_cal_err(train_and_predict_multidragons(
            t, y, x_train, x_test,t_flags, knob_loss=knob_loss,val_split=0.2, batch_size=64)))

        cal_tmp["Linear"].append(_cal_err(train_and_predict_linear(t, y, x_train, x_test,t_flags)))

        cal_tmp["Cart"].append(_cal_err(train_and_predict_rf(t, y, x_train, x_test,t_flags)))

    Res = {}
    for key in list(cal_tmp.keys()):
        Res[key] = np.mean(cal_tmp[key])

    return Res


def turn_knob(data_base_dir, workload, a):

    Res =  {
        "Aver": [],
        "Linear": [],
        "Cart" : [],
        "MulDra": []
    }

    for i in np.arange(1):
        path_dict = []
        if i % 2 == 1:
            path_dict.append(data_base_dir+'SingleEnvTrack/directed_low/'+workload+'/bo/')
        else:
            path_dict.append(data_base_dir+'SingleEnvTrack/directed_high/'+workload+'/bo/')
        if i <=3:
            path_dict.append(data_base_dir+'SingleEnvTrack/directed_low/'+workload+'/bo/')
        else:
            path_dict.append(data_base_dir+'SingleEnvTrack/directed_high/'+workload+'/bo/')
        if i < 2 or i > 5 :
            path_dict.append(data_base_dir+'SingleEnvTrack/directed_low/'+workload+'/bo/')
        else:
            path_dict.append(data_base_dir+'SingleEnvTrack/directed_high/'+workload+'/bo/')
        cal_tmp = run_test(path_dict, workload, a, list(ac_dict.keys()))
        for key in list(cal_tmp.keys()):
            Res[key].append(cal_tmp[key])
    for key in list(Res.keys()):
        Res[key] = np.mean(Res[key].copy())
    return Res


def main():
    Res = {}
    for workload in Workloads:
        print(workload)
        for sampled_num in np.arange(3,31,3):
            Res[workload+"+"+str(sampled_num)] = turn_knob(BaseAddr, workload, sampled_num)
    print(Res)
    with open('MultipleEnvTrack/RQ3/RAG.txt', 'w') as f:
        f.write(str(Res))


if __name__ == '__main__':
    main()
