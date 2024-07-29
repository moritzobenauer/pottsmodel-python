import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_analysis import *

data_16x16 = pd.read_csv('16x16_HALF_Working_results_0720/out_final.txt')
data_10x10 = pd.read_csv('10x10GRID/out_final.txt')
data_24x24 = pd.read_csv('24x24GRID/out_final.txt')
data_32x32 = pd.read_csv('32x32GRID/out_final.txt')


data = [data_10x10, data_16x16, data_24x24, data_32x32]
gridsizes = [10,16,24, 32]
latticesize = [x**2 for x in gridsizes]
colors=['orange', 'blue', 'green', 'red']

for color, N, g, d in zip(colors, latticesize, gridsizes, data):
    x,y = MirrorDataAndNormalize(d)
    plt.plot(x*N, y,  'bo-', label=f'{g}x{g}, Flatness: 0.8, HI', alpha=0.6, color=color, markersize=4)


plt.ylabel('ln g(E)')
plt.xlabel('E')
plt.legend(loc='best')
plt.savefig('lnge-versus-e.png')
plt.clf()
temps = np.linspace(0.5, 5., 500)  # T varies from 0.4 to 8

for color, N, g, d in zip(colors, latticesize, gridsizes, data):
    x,y = MirrorDataAndNormalize(d)
    plots = []
    for T in temps:
        plots.append(MLOThermo(T,x,y,N))
    plots = np.array(plots)
    fig = plt.figure(1)
    plt.plot(temps, plots[:, 2], 'ko--', label=f'{g}x{g}, Flatness: 0.8, HI', alpha=0.4, color=color, markersize=4)
    tc = 2*1 / np.log(np.sqrt(2) + 1)
    plt.axvline(tc, color='k')
    plt.xlabel('T')
    plt.ylabel('C(T)/N')
    plt.legend(loc='best')
plt.savefig('C-versus-T.png')
plt.clf()

temps = np.linspace(0.5, 5., 500)  # T varies from 0.4 to 8

for color, N, g, d in zip(colors, latticesize, gridsizes, data):
    x,y = MirrorDataAndNormalize(d)
    plots = []
    for T in temps:
        plots.append(MLOThermo(T,x,y,N))
    plots = np.array(plots)
    fig = plt.figure(1)
    plt.plot(temps, plots[:, 0], 'ko--', label=f'{g}x{g}, Flatness: 0.8, HI', alpha=0.4, color=color, markersize=4)
    plt.xlabel('T')
    plt.ylabel('F(T)/N')
    plt.legend(loc='best')
plt.savefig('F-versus-T.png')
plt.clf()

for color, N, g, d in zip(colors, latticesize, gridsizes, data):
    x,y = MirrorDataAndNormalize(d)
    plots = []
    for T in temps:
        plots.append(MLOThermo(T,x,y,N))
    plots = np.array(plots)
    fig = plt.figure(1)
    plt.plot(temps, plots[:, 1], 'ko--', label=f'{g}x{g}, Flatness: 0.8, HI', alpha=0.4, color=color, markersize=4)
    tc = 2*1 / np.log(np.sqrt(2) + 1)
    plt.axvline(tc, color='k')
    plt.xlabel('T')
    plt.ylabel('U(T)/N')
    plt.legend(loc='best')
plt.savefig('U-versus-T.png')
plt.clf()


for color, N, g, d in zip(colors, latticesize, gridsizes, data):
    x,y = MirrorDataAndNormalize(d)
    plots = []
    for T in temps:
        plots.append(MLOThermo(T,x,y,N))
    plots = np.array(plots)
    fig = plt.figure(1)
    plt.plot(temps, plots[:, 3], 'ko--', label=f'{g}x{g}, Flatness: 0.8, HI', alpha=0.4, color=color, markersize=4)
    tc = 2*1 / np.log(np.sqrt(2) + 1)
    plt.axvline(tc, color='k')
    plt.xlabel('T')
    plt.ylabel('S(T)/N')
    plt.legend(loc='best')
plt.savefig('S-versus-T.png')
plt.clf()
