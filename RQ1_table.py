import numpy as np

Workloads = ["terasort","wordcount", "als","bayes","kmeans","linear","lr","rf","pagerank","nweight",]
distribution_apps = ["linear_trans",'learn2sample','multidragon']
average_apps = ['gaus_trans','full_iter', 'random_select']
CompareApps = ["linear_trans",'learn2sample','random_select','full_iter', 'gaus_trans', 'multidragon',  ]
method_convert = {
    'full_iter' : 'Average Distributing',
    'random_select' : 'Random Selecting',
    'gaus_trans' : 'Environment Serializing',
    'linear_trans' : 'Linear Transfer Learning',
    'learn2sample' : 'Learning to Sample',
    'multidragon' : 'Our Method'
}
rgb_convert = {
    'Average Distributing' : 'red',
    'Random Selecting' : 'yellow',
    'Environment Serializing' : 'blue',
    'Linear Transfer Learning' : 'green',
    'Learning to Sample' : 'brown',
    'Our Method' : 'black'
}
WorkloadDict = {"Micro Benchmarks":["terasort","wordcount"],
                "Machine Learning":["als","bayes","kmeans","linear","rf","lr"],
                "Websearch Benchmarks":["pagerank"],
                "Graph Benchmark":["nweight"]}
iterations = np.arange(9, 151, 3)
select_iter_list = [24, 72]

ResDict = {}
for work in Workloads:
    ResDict[work] = {}
    for app in CompareApps:
        ResDict[work][app] = {}
        for iter in iterations:
            if app in distribution_apps:
                ResDict[work][app][str(iter)] = 1
            else:
                ResDict[work][app][str(iter)] = []



for a in np.arange(2,40,1):
    for b in np.arange(3,30,3):
        File = open("MultipleEnvTrack/RQ1&2/"+str(a)+"_"+str(b)+".txt","r")
        info = eval(File.read())

        iter = 3*a + b

        for app in CompareApps:
            for work in Workloads:
                if app in distribution_apps:
                    ResDict[work][app][str(iter)] = min(ResDict[work][app][str(iter)], info[work][0][app])
                else:
                    ResDict[work][app][str(iter)].append(info[work][0][app])

for app in average_apps:
    for work in Workloads:
        for iter in iterations:
            if len(ResDict[work][app][str(iter)]) == 0:
                ResDict[work][app][str(iter)] = 1
            else:
                ResDict[work][app][str(iter)] = np.mean(ResDict[work][app][str(iter)].copy())

AverDict = {}
for app in CompareApps:
    AverDict[app] = {}
    for iter in iterations:
        cal_list = []
        for work in Workloads:
            cal_list.append(ResDict[work][app][str(iter)])
        AverDict[app][str(iter)] = np.mean(cal_list)

ResDict["aver"] = AverDict

for work in Workloads:
    aver_perf = []
    for app in CompareApps[:-1]:
        aver_perf.append(ResDict[work][app][str(select_iter_list[0])])

for iter in select_iter_list:
    for app in CompareApps:
        str_csv = method_convert[app]
        for label in list(WorkloadDict.keys()):
            cal_tmp = []
            for work in WorkloadDict[label]:
                cal_tmp.append(ResDict[work][app][str(iter)])
            str_csv += " & " +str(round(np.mean(cal_tmp)*100,2)) + "\%"
        str_csv += " & " +str(round(ResDict["aver"][app][str(iter)]*100,2)) + "\% \\\\"
        print(str_csv)
    print("\n")