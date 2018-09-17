import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('../Coffee_TRAIN', sep=',', header=None)
label = data.pop(data.columns[0])

def plot_match(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta)/len(Tb)), 1]) 

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))
    
    print(np.argmax(values))
    print(np.argmin(values))
    
    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g', label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3,3))

    plt.subplot(gs[1])
    plt.plot(Tb)
    

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3,3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

print(data.head())

ts = data[label==0].sample().values.flatten() # pick a random sample from class 0
query = data[label==1].sample().values.flatten()[200:250] # pick a subsequence from a sample of class 1

from owlpy.core import *

Pab, Iab = stamp(ts,query,50)                       # run the STAMP algorithm to compute the Matrix Profile
print(Iab)
print(Pab)
plot_match(ts,query,Pab,Iab,50)

