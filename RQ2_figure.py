import numpy as np
from matplotlib import pyplot as plt

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

iterations = np.arange(9, 151, 3)
filter_ratio_list = [0.16, 0.08]
file_name = "RQ2"

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

MinDict = {}
for work in Workloads:
    MinDict[work] = {}
    for app in CompareApps:
        MinDict[work][app] = min(list(ResDict[work][app].values()))
print(MinDict)

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
        for app in CompareApps:
            MinIterDict[work][app] = 999
            for iter in iterations:
                if ResDict[work][app][str(iter)] <= ratio:
                    MinIterDict[work][app] = min(iter, MinIterDict[work][app])
    PerfDict = {}
    for app in CompareApps:
        PerfDict[method_convert[app]] = []
        for work in Workloads:
            if MinIterDict[work][app] > 150:
                PerfDict[method_convert[app]].append(0)
            else:
                PerfDict[method_convert[app]].append(MinIterDict[work][app])
    return PerfDict

PerfDict = {}
for ratio in filter_ratio_list:
    PerfDict[str(ratio)] = _cal_iter(ratio)


y_ticks = np.arange(0,161,20)

ind = np.arange(len(Workloads))  # the x locations for the groups
width = 0.15  # the width of the bars

pl = plt.figure(figsize=(30, 10), dpi=40)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
ax1 = pl.add_subplot(2,1,1)
ax2 = pl.add_subplot(2,1,2)

for id,method in enumerate(CompareApps):
    ind_met = ind + (id-len(CompareApps)/2)*width
    rects1 = ax1.bar(ind_met, PerfDict[str(filter_ratio_list[0])][method_convert[method]], width,
                     color=rgb_convert[method_convert[method]], label=method_convert[method])
    rects2 = ax2.bar(ind_met, PerfDict[str(filter_ratio_list[1])][method_convert[method]], width,
                     color=rgb_convert[method_convert[method]], label=method_convert[method])


# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_title('Scores by group and gender')
ax1.set_ylabel('Cost',  fontsize=24)
ax1.set_ylim(0,160)
ax1.set_yticklabels(y_ticks, fontsize=24)

ax2.set_ylabel('Cost',  fontsize=24)
ax2.set_ylim(0,160)
ax2.set_yticklabels(y_ticks, fontsize=24)

ax2.set_xticks(ind)
ax2.set_xticklabels(Workloads, rotation=10, fontsize=24)
ax1.legend( fontsize=16)
ax1.set_xticks([])


plt.savefig("FigureOutput/"+file_name+".png")
plt.show()
