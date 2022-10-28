import numpy as np
import pandas as pd
import random
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct
import warnings
warnings.filterwarnings("ignore")

ac1, ac2, ac3 = 'sky', 'has', 'stg'
ac_list = ['sky', 'has', 'stg']

def cross_mu(q, l):
    q = list(q)
    l = list(l)
    m = random.randint(0, len(q))
    for i in random.sample(range(len(q)), m):
        q[i], l[i] = l[i], q[i]
        q_mu = random.randint(0, len(q)-1)
        q[q_mu] = np.random.uniform(-1, 1, 1).item()
        l_mu = random.randint(0, len(l)-1)
        l[l_mu] = np.random.uniform(-1, 1, 1).item()
    return q, l

def _random_sample(X, n_step):

    X_sample = []
    for i in np.arange(X.shape[1]):
        X_min = min(X.values[:,i])
        X_max = max(X.values[:,i])
        X_sample.append([])
        for step in np.arange(n_step):
            X_sample[i].append(random.uniform(X_min,X_max))
    X_sample = np.array(X_sample).T
    idx_near = []
    for x in X_sample:
        idx_near.append(np.array([np.linalg.norm(a-x) for a in X.values]).argmin())
    return idx_near

def ParameterTune(tunetime,workloads,label,ranges):

    #####################################################
    ########## Read data and simple processing ##########
    #####################################################

    parser = argparse.ArgumentParser(description='Configuration extrapolation with GIL/GIL+.')
    parser.add_argument('--target', type=str, default="has", help='Target hardware')
    parser.add_argument('--workload', type=str, default="als", help='Workload')
    parser.add_argument('--outcome', type=str, default="Throughput", help='Outcome metric')
    parser.add_argument('--n_start', type=int, default=80, help='Start index (default: 200)')
    parser.add_argument('--n_step', type=int, default=1, help='k: number of configurations queried at each round (default: 20)')
    parser.add_argument('--n_iter', type=int, default=400, help='T: number of rounds (default: 20)')

    args = parser.parse_args()

    target     = args.target
    workload   = workloads
    outcome    = args.outcome
    n_start    = args.n_start
    n_step     = args.n_step
    n_iter     = args.n_iter

    print("target:   {}".format(target))
    print("workload: {}".format(workload))
    print("outcome:  {}".format(outcome))
    print("n_start:  {}".format(n_start))
    print("n_step:   {}".format(n_step))
    print("n_iter:   {}".format(n_iter))

    X = pd.read_csv('OriginData/perf/config_clean.csv', index_col=0)
    Xr = X.copy()
    X = (X-X.mean())/X.std()

    ac_dict = {'sky': 'Skylake', 'has': 'Haswell', 'stg': 'Storage'}

    rp_sky = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[ac1], workload), index_col=0)
    rp_has = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[ac2], workload), index_col=0)
    rp_stg = pd.read_csv('OriginData/perf/{}/{}.csv'.format(ac_dict[ac3], workload), index_col=0)

    rp = pd.DataFrame(columns=[ac1, ac2, ac3])
    rp[ac1] = rp_sky[outcome].values
    rp[ac2] = rp_has[outcome].values
    rp[ac3] = rp_stg[outcome].values

    rpn = rp

    ############################# Load LLSM data ############################
    Sn_dict = {}

    S_sky = pd.read_csv('OriginData/llsm/{}/{}.csv'.format(ac_dict['sky'], workload), index_col=0)
    S_has = pd.read_csv('OriginData/llsm/{}/{}.csv'.format(ac_dict['has'], workload), index_col=0)
    S_stg = pd.read_csv('OriginData/llsm/{}/{}.csv'.format(ac_dict['stg'], workload), index_col=0)

    Sn_sky = (S_sky-S_sky.mean())/S_sky.std()
    Sn_dict['sky'] = Sn_sky.copy()

    Sn_has = (S_has-S_has.mean())/S_has.std()
    Sn_dict['has'] = Sn_has.copy()

    Sn_stg = (S_stg-S_stg.mean())/S_stg.std()
    Sn_dict['stg'] = Sn_stg.copy()


    res = {}
    res[ac1], res[ac2], res[ac3] = [], [], []
    Algos = ['bo','rlr']

    GenerateDataDict = {}
    for algo in Algos:
        GenerateDataDict[algo] = {}
    Process = {}
    for ac in ac_list:

        sort_idx_full = rp.sort_values(by=[ac], ascending=True).index
        sort_idx_sub = sort_idx_full

        X_sub = X.iloc[sort_idx_sub]
        Y_sub = rpn.loc[sort_idx_sub, ac]
        U_sub = Sn_dict[ac].iloc[sort_idx_sub]
        X_sub.reset_index(drop=True, inplace=True)
        Y_sub.reset_index(drop=True, inplace=True)
        U_sub.reset_index(drop=True, inplace=True)

        sort_idx_full_tar = rp.sort_values(by=[ac], ascending=True).index

        if ranges > 0:
            sort_idx_sub_tar = sort_idx_full_tar[:ranges]
        elif ranges < 0:
            sort_idx_sub_tar = sort_idx_full_tar[ranges:]
        else:
            sort_idx_sub_tar = sort_idx_full_tar[:]

        init_idx_pool = list(sort_idx_sub_tar)
        init_idx_raw = init_idx_pool[:n_step]

        val_init_idx = rpn.loc[init_idx_raw, ac].values
        init_idx = np.searchsorted(Y_sub, val_init_idx).tolist()

        Process[ac] = []

        ########################## 1. BO ###########################
        pool_idx_bo = range(X_sub.shape[0])
        seed_idx_bo = init_idx
        pool_idx_bo = list(set(pool_idx_bo)-set(init_idx))

        kernel = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e1))
        model=GaussianProcessRegressor(kernel=kernel, alpha=1)
        GenerateDataDict['bo'][ac] = np.concatenate((
            np.array(list(map(lambda id: X[X==X_sub.values[id]].dropna().index.item(), seed_idx_bo))).reshape((n_step,1)),
            Y_sub.values[seed_idx_bo].reshape((n_step,1)),
            X_sub.values[seed_idx_bo]),axis=1)

        for i in range(n_iter):
            ## Fit GP model
            X_bo_tr, Y_bo_tr = X_sub.values[seed_idx_bo], Y_sub.values[seed_idx_bo]
            sample_idx = np.random.randint(0, high=X_sub.shape[0]-1, size=10, dtype='l').tolist()
            X_sample = X_sub.values[sample_idx]
            mean_bo, std_bo = model.fit(X_bo_tr, Y_bo_tr).predict(X_sample, return_std=True)
            ## Get co reward function
            re_bo_tmp = std_bo.copy()
            new_idx = np.array(sample_idx)[np.argsort(re_bo_tmp)[-n_step:].tolist()].tolist()
            Process[ac].append(rp.loc[X[X==X_bo_tr[np.argmax(Y_bo_tr),:]].dropna().index.item(), ac])
            ## Update pool indices and seed indices
            pool_idx_bo = list(set(pool_idx_bo)-set(new_idx))
            seed_idx_bo = seed_idx_bo + new_idx
            GenerateDataDict['bo'][ac] = \
                np.concatenate((GenerateDataDict['bo'][ac],
                                np.concatenate((
                                    np.array(list(map(lambda id: X[X==X_sub.values[id]].dropna().index.item(), new_idx)))
                                    .reshape((n_step,1)),
                                    Y_sub.values[new_idx].reshape((n_step,1)),
                                    X_sub.values[new_idx]),axis=1)
                                ),axis=0)

        ###################### 2. Random sampling with linear regression ######################
        seed_idx_rnd = np.random.randint(0, high=X_sub.shape[0]-1, size=n_step*n_iter, dtype='l').tolist()
        GenerateDataDict['rlr'][ac] = np.concatenate((
            np.array(list(map(lambda id: X[X==X_sub.values[id]].dropna().index.item(), seed_idx_rnd))).reshape((n_step*n_iter,1)),
            Y_sub.values[seed_idx_rnd].reshape((n_step*n_iter,1)),
            X_sub.values[seed_idx_rnd]),axis=1)


    ###################### Save results ######################

    print(GenerateDataDict)
    for algo in Algos:
        for ac in ac_list:
            np.savetxt("SingleEnvTrack2/"+label+"/"
                       +workload+"/"+algo+"/Step"+str(n_step)+"_"+ac+str(tunetime)+".csv", GenerateDataDict[algo][ac],delimiter=",")

    return 1
if __name__ == '__main__':

    TuneTime = 40
    WorkLoads = ["als","bayes","kmeans","linear","lr","nweight","pagerank","rf","terasort","wordcount"]


    for workload in WorkLoads:
        for i_tune in np.arange(TuneTime):
            TuningRes = ParameterTune(i_tune,workload,"generate",0)
    for workload in WorkLoads:
        for i_tune in np.arange(TuneTime):
            TuningRes = ParameterTune(i_tune,workload,"directed_high",-200)
    for workload in WorkLoads:
        for i_tune in np.arange(TuneTime):
            TuningRes = ParameterTune(i_tune,workload,"directed_low",200)

