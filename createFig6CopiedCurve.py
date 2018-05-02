"""Outputs muliple error/loss line plots on one figure. Move the relevant jobs folders into a subfolder of jobs titled multiLossErr.  Only works well for 7 options since after 7, the colors repeat"""

import os
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import re
from scipy import stats
from scipy.interpolate import spline

DEFAULT_DATADIR = 'jobs'
''' net_sampleLen=1_nDistSmpls=1_layerSizes=10,7,5,4,3_nEpoch=8000_batch=512_nRepeats=10_nEpochInds=274_LastEpochsInds=7999_DataName=var_u_lr=0.004_l1=0.0_l2=0.0_act=0 '''
DEFAULT_FORM = "%s/net_sampleLen=1_nDistSmpls=1_layerSizes=10,7,5,4,3_nEpoch=8000_batch=512_nRepeats=10_nEpochInds=274_LastEpochsInds=7999_DataName=%s_lr=%s_l1=%s_l2=%s_act=%s"
DEFAULT_DNAME = "var_u"
DEFAULT_LR = "0.0004"
DEFAULT_L1 = "0.0"
DEFAULT_L2 = "0.0"
DEFAULT_EN = "0.0"
DEFAULT_ACT = 0

LR_int = [0.0004, 0.004, 0.04]
L1_int = [0.0001, 0.0004, 0.004]
L2_int = [0.0001, 0.0004, 0.004]
EN_int = [0.0001, 0.0004, 0.004]
ACT_int = [6, 5, 4, 3, 0] # 4, 3, 2]
ACT_NAMES = {0:'tanh', 1:'ReLU',2:'log softmax', 3:'elu', 4:'softplus', 5:'softsign', 6:'sigmoid'}
KEYS = ['DIR','DATA', 'LR', '\lambda_1', '\lambda_2',  'activation']
interv = {'LR':LR_int,'\lambda_1': L1_int, '\lambda_2': L2_int, '\lambda': EN_int, 'activation': ACT_int}

colors = ['red', 'blue', 'green', 'magenta', 'cyan', 'coral']
FORM_ARGS = [DEFAULT_DATADIR, DEFAULT_DNAME, DEFAULT_LR, DEFAULT_L1, DEFAULT_L2, DEFAULT_ACT]

def main(key='LR'):
    matplotlib.rc('font', **{'family': 'normal', 'weight':'bold', 'size':18})
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',
                       '-d', dest="data", default=DEFAULT_FORM % (DEFAULT_DATADIR,DEFAULT_DNAME,DEFAULT_LR, DEFAULT_L1, DEFAULT_L2, DEFAULT_ACT),
                        type=str, help='The folder with saved data for the network')
    parser.add_argument('-name',
                        '-n', dest="save_name", default="%s_bottleneck" % key,
                        type=str, help='The name to save the image as')
    parser.add_argument('-param', '-p', dest='param', default='')
    parser.add_argument('-d_name', dest='dataname', default='var_u')
    args = parser.parse_args()
    mypath = args.data
    save_name = args.save_name
    f1 = plt.figure(figsize=(12, 8))
    ax1 = f1.add_subplot(111)
    xLine = [1.0819, 1.5672, 2.0866, 2.7264, 3.5842, 4.5500, 5.5500, 6.1669, 7.1716, 8.9552, 12.1947]
    yLine = [0.9856, 0.9881, 0.9903, 0.9927, 0.9952, 0.9968, 0.9981, 0.9989, 0.9993, 0.9994, 0.9994]
    p0 = ax1.plot(xLine, yLine, 'k', linewidth=5, label="Information Bottelneck Bound")
    ps = []
    interval = interv[key]
    form_args = FORM_ARGS[:]
    for i, LR in enumerate(interval):
        form_args[KEYS.index(key)] = LR
        mypath=DEFAULT_FORM % tuple(form_args)
        p1 = plot_beta_values(mypath, f1=f1, ax1=ax1, color=colors[i])
        ps.append(p1)
    if key == 'activation':
        f1.legend([p0] + ps, ["Information Bottleneck Bound"] + [ "$%s=%s$" % (key, ACT_NAMES[LR]) for LR in interval], loc='lower right')
    else:
        f1.legend([p0] + ps, ["Information Bottleneck Bound"] + [ "$%s=%s$" % (key, LR) for LR in interval], loc='lower right')
    ax1.set_xlabel('$I(X|T)$')
    ax1.set_ylabel('$I(T|Y)$')
    f1.savefig('figures/bottlenecks/' + save_name + '.jpg', dpi=500, format='jpg')
    #ax1.set_xlim([min(xLine), max(xLine)])
    #ax1.set_ylim([min(yLine), max(yLine)])
    plt.show()

def plot_beta_values(mypath, f1=None, ax1=None, color=None, name=''):
    if f1 is None:
        f1 = plt.figure(figsize=(12, 8))
    if ax1 is None:
        ax1 = f1.add_subplot(111)
    print('Loading %s' % mypath)
    d = pkl.load(open(os.path.join(mypath, 'data.pickle'), 'rb'))
    data = d['information']
    data = np.squeeze(data)
    print('Generating subplot')
    print(data.shape)
    if len(data.shape) < 3:
        data = np.array([data])
    Ix = np.empty((data.shape[2], data.shape[0]))
    Iy = np.empty((data.shape[2], data.shape[0]))
    for i in range(data.shape[0]):  # for each repeat
        for k in range(data.shape[2]):  # for each layer
            ixe = [data[i][r][k]['local_IXT'] for r in range(data.shape[1])]
            iye = [data[i][r][k]['local_ITY'] for r in range(data.shape[1])]
            Iy[k][i] = np.max(iye)
            Ix[k][i] = ixe[iye.index(Iy[k][i])]
    xm = []
    ym = []
    for k in range(data.shape[2]):  # for each layer
        print(k, Ix[k], Iy[k])
        Iykm = np.median(Iy[k])
        Ixkm = np.median(Ix[k])
        Iystd = stats.sem(Iy[k])
        xp = [Ixkm for i in range(3)]
        yp = [Iykm-Iystd, Iykm, Iykm+Iystd]
        ax1.plot(xp, yp, color=color)
        xm.append(Ixkm)
        ym.append(Iykm)
    #p = ax1.plot(xm, ym, color=color)
    p = ax1.scatter(xm, ym,marker='+', s=200, color=color, label=name)
    return p


if __name__ == '__main__':
    main('LR')
    main('activation')
