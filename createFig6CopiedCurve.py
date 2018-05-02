"""Outputs muliple error/loss line plots on one figure. Move the relevant jobs folders into a subfolder of jobs titled multiLossErr.  Only works well for 7 options since after 7, the colors repeat"""

import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',
                        '-d', dest="data", default="jobs/net_batch=512_LastEpochsInds=48_nEpoch=50_nDistSmpls=1_DataName=var_u_layerSizes=10,7,5,4,3_nEpochInds=49_sampleLen=1_nRepeats=5_lr=0.0004",
                        type=str, help='The folder with saved data for the network')
    parser.add_argument('-name',
                        '-n', dest="save_name", default="Fig6Temp",
                        type=str, help='The name to save the image as')
    args = parser.parse_args()
    mypath = args.data
    save_name = args.save_name
    f1 = plt.figure(figsize=(12, 8))
    d = pkl.load(open(os.path.join(mypath, 'data.pickle'), 'rb'))
    data = d['information']
    data = np.squeeze(data)
    ax1 = f1.add_subplot(111)
    xLine = [1.0819, 1.5672, 2.0866, 2.7264, 3.5842, 4.5500, 5.5500, 6.1669, 7.1716, 8.9552, 12.1947]
    yLine = [0.9856, 0.9881, 0.9903, 0.9927, 0.9952, 0.9968, 0.9981, 0.9989, 0.9993, 0.9994, 0.9994]
    ax1.plot(xLine, yLine, 'k', linewidth=2, label="Information Bottelneck Bound")
    if len(data.shape) < 3:
        data = np.array([data])
    print(data)
    Ix = np.empty((data.shape[2], data.shape[0]))
    Iy = np.empty((data.shape[2], data.shape[0]))
    for i in range(data.shape[0]):
        for k in range(data.shape[2]):
            Ix[k][i] = data[i][-1][k]['local_IXT']
            Iy[k][i] = data[i][-1][k]['local_ITY']
    print(Iy)
    IxStd = np.std(Ix, axis=1)
    IyStd = np.std(Iy, axis=1)
    IxMean = np.mean(Ix, axis=1)
    IyMean = np.mean(Iy, axis=1)
    print(IyMean)
    for i in range(data.shape[2]):
        ax1 = f1.add_subplot(111)
        xLine = [IxMean[i], IxMean[i]]
        yLine = [IyMean[i] + IyStd[i], IyMean[i] - IyStd[i]]
        ax1.plot(xLine, yLine, 'b', linewidth=3)
        ax1 = f1.add_subplot(111)
        xLine = [IxMean[i] + IxStd[i], IxMean[i] - IxStd[i]]
        yLine = [IyMean[i], IyMean[i]]
        ax1.plot(xLine, yLine, 'b', linewidth=3)
        if IyStd[i] == 0 and IxStd[i] == 0:
            x1 = f1.add_subplot(111)
            xLine = [IxMean[i] + 0.01, IxMean[i] - 0.01]
            yLine = [IyMean[i], IyMean[i]]
            ax1.plot(xLine, yLine, 'b', linewidth=3)
    ax1.legend(loc='best')
    f1.savefig(save_name + '.png', dpi=500, format='png')


if __name__ == '__main__':
    main()
