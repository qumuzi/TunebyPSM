import numpy as np

Workloads = ["als","bayes","kmeans","linear","lr","nweight","pagerank","rf","terasort","wordcount"]
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
select_ratio_list = [0.16,  0.08]

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
#Workloads.append("aver")

MinDict = {}
for work in Workloads:
    MinDict[work] = {}
    for app in CompareApps:
        MinDict[work][app] = min(list(ResDict[work][app].values()))


def _cal_iter(ratio):
    collect_app = []
    for app in CompareApps:
        if_selected = 1
        for work in Workloads:
            if MinDict[work][app] <= ratio:
                if_selected = 0
        if if_selected == 0:
            collect_app.append(app)

    MinIterDict = {}
    for work in Workloads:
        MinIterDict[work] = {}
        for app in collect_app:
            MinIterDict[work][app] = 999
            for iter in iterations:
                if ResDict[work][app][str(iter)] <= ratio:
                    MinIterDict[work][app] = min(iter, MinIterDict[work][app])
    return MinIterDict


#print(ResDict["aver"])
for ratio in select_ratio_list:
    MinIterDict = _cal_iter(ratio)
    for app in CompareApps:
        str_csv = method_convert[app]
        aver_list = []
        total_missed_index = 0
        for label in list(WorkloadDict.keys()):
            label_missed_index = 0
            cal_tmp = []
            for work in WorkloadDict[label]:
                if app in MinIterDict[work] and MinIterDict[work][app] != 999:
                    cal_tmp.append(MinIterDict[work][app])
                    aver_list.append(MinIterDict[work][app])
                else:
                    label_missed_index +=1
                    total_missed_index +=1
            if len(cal_tmp) == 0:
                str_csv +=" & " + "N/A" + " "
            else:
                miss_str = ""
                if label_missed_index > 0:
                    miss_str = "(" + str(int(label_missed_index)) +" workload missed)"
                str_csv +=" & " + str(round(np.mean(cal_tmp),2)) + miss_str +" "
        str_csv += "\\\\"
        print(str_csv)
    print("\n")