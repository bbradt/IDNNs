import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


def main():
    mypath = "jobs\multiLossErr"
    save_name = "multi"
    plotStyle = ['-', '--', '-.', ':']
    plotColor = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    dirList = [f for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]
    f1 = plt.figure(figsize=(12, 8))
    dirCount = 0
    labeled = False
    fig_strs = ['train_error', 'test_error', 'loss_train', 'loss_test']
    for dirName in dirList:
        d = pkl.load(open(os.path.join(mypath, dirName, 'data.pickle'), 'rb'))
        epochsInds = d['params']['epochsInds']
        fig_data = [np.squeeze(d[fig_str]) for fig_str in fig_strs]
        ax1 = f1.add_subplot(111)
        mean_sample = False if len(fig_data[0].shape) == 1 else True
        if mean_sample:
            fig_data = [np.mean(fig_data_s, axis=0) for fig_data_s in fig_data]
        if labeled:
            ax1.plot(epochsInds, fig_data[i], plotColor[dirCount] + plotStyle[0], linewidth=2, label="(Lr = " + str(d['params']['lr']) + ")")
            for i in range(len(fig_data) - 1):
                ax1.plot(epochsInds, fig_data[i + 1], plotColor[dirCount] + plotStyle[i + 1], linewidth=2)
        else:
            for i in range(len(fig_data)):
                ax1.plot(epochsInds, fig_data[i], plotColor[dirCount] + plotStyle[i], linewidth=2, label=fig_strs[i] + "(Lr = " + str(d['params']['lr']) + ")")
            labeled = True
        ax1.legend(loc='best')
        dirCount += 1
        if dirCount == len(plotColor):
            dirCount = 0
    print("Saving %s" % 'err_' + save_name + '.png')
    f1.savefig('figures/err_' + save_name + '.png', dpi=500, format='png')


if __name__ == '__main__':
    main()
