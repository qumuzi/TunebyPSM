
import matplotlib.pyplot as plt
import numpy as np

ranges = np.arange(3,31,3)

Workloads = ["als","bayes","kmeans","linear","lr","nweight"
    ,"pagerank","rf","terasort","wordcount"]

File = open('MultipleEnvTrack/RQ3/RA.txt',"r")
precise_dict = eval(File.read())

AverAccu = {"aver":[]}
LinearAccu = {"aver":[]}
MulDraAccu = {"aver":[]}
CartAccu = {"aver":[]}

for work in Workloads:
    res_list_Aver = []
    res_list_Linear = []
    res_list_MulDra = []
    res_list_Cart = []
    for id in ranges:
        res_list_Aver.append(precise_dict[work+"+"+str(id)]["Aver"])
        res_list_Linear.append(precise_dict[work+"+"+str(id)]["Linear"])
        res_list_MulDra.append(precise_dict[work+"+"+str(id)]["MulDra"])
        res_list_Cart.append(precise_dict[work+"+"+str(id)]["Cart"])
    AverAccu[work] = np.array(res_list_Aver)
    LinearAccu[work] = np.array(res_list_Linear)
    MulDraAccu[work] = np.array(res_list_MulDra)
    CartAccu[work] = np.array(res_list_Cart)

for id in np.arange(len(ranges)):
    aver_list = []
    linear_list = []
    muldra_list = []
    cart_list = []
    for work in Workloads:
        aver_list.append(AverAccu[work][id])
        linear_list.append(LinearAccu[work][id])
        muldra_list.append(MulDraAccu[work][id])
        cart_list.append(CartAccu[work][id])
    AverAccu["aver"].append(np.mean(aver_list))
    LinearAccu["aver"].append(np.mean(linear_list))
    MulDraAccu["aver"].append(np.mean(muldra_list))
    CartAccu["aver"].append(np.mean(cart_list))
Workloads.append("aver")

fig, ax = plt.subplots(figsize=(15, 7.5), dpi=80)
ax.plot(ranges,AverAccu["aver"],label="Simple Average", linewidth=3)
ax.plot(ranges,LinearAccu["aver"],label="Linear Regression", linewidth=3)
ax.plot(ranges,CartAccu["aver"],label="Cart Regression", linewidth=3)
ax.plot(ranges,MulDraAccu["aver"],label="MultiDragon", linewidth=3)
y_ticks = [0.3,0.35,0.4,0.45, 0.5,]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks, fontsize=24)
ax.legend(fontsize=16)
ax.set_ylabel('RA',  fontsize=24)
ax.set_xlabel('Sampled Points',  fontsize=24)
ax.set_xticklabels(ranges, fontsize=24)
ax.set_xticks(ranges)

plt.savefig("FigureOutput/RQ3_RA.png")


