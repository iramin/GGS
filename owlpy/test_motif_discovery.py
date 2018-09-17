import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('../ECGFiveDays_TRAIN', sep=',', header=None)
label = data.pop(data.columns[0])




def findTopKMotifs(values, k, m):
    kList = list(range(0,k))
    result = list(np.argpartition(values, kth=kList))

    i = 1
    pointer = 0
    print("before")
    print(result)
    while pointer < len(result) - 1:
        i = pointer + 1
        while i < len(result):
            if abs(result[pointer] - result[i]) < m:
                # print("deleting " + str(i) + " = " + str(result[i]))
                del result[i]
            else:
                i=i+1
        pointer=pointer+1
    print("after")
    print(result)
    return result

def remove_overlaps(motifs, m):
    pointer = 0
    while pointer < len(motifs.index) - 1:
        i = pointer + 1
        while i < len(motifs.index):
            if abs(motifs.iloc[pointer].iloc[0] - motifs.iloc[i].iloc[0]) < m:
                motifs.drop(motifs.index[i], inplace=True)
            else:
                i = i + 1
        pointer = pointer + 1
    return motifs

def plot_motif(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8,4))
    plt.subplot(211)
    plt.plot(Ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(Ta)))
    
    print(np.argmax(values))
    print(np.argmin(values))
    print(np.argpartition(values,kth=[0,1,2]))
    print(values)

    showThis = findTopKMotifs(values,3,m)

    counter = 1
    colors=['g','y','b','r','c']

    for item in showThis:
        plt.plot(range(item, item + m), Ta[item:item + m], c=colors[counter % len(colors)],
                 label='Motif #' + str(counter))
        counter=counter+1

    plt.plot(range(np.argmax(values), np.argmax(values) + m), Ta[np.argmax(values):np.argmax(values) + m], c='w',
             label='Top Discord')
    
    # plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g', label='Top Motif')
    # plt.plot(range(549, 549 + m), Ta[549:549 + m], c='y',
    #          label='Second Best Motif')

    
    plt.legend(loc='best')
    plt.title('Time-Series')


    plt.subplot(212)
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

print(data.head())

my_data = pd.read_csv('../test.txt', sep=',', header=None)
print(my_data)
print(my_data.values.flatten())

# ts = data.iloc[:10].values.flatten() # pick a random sample from class 0
ts = my_data.values.flatten()

query = my_data.values.flatten()[1781:2181]

from owlpy.core import *

# Pab, Iab = stamp(ts,ts,600)                       # run the STAMP algorithm to compute the Matrix Profile
# plot_motif(ts,ts,Pab,Iab,600)

l=50
# Pab, Iab = stamp(ts,query,l)                       #  `run the STAMP algorithm to compute the Matrix Profile
# plot_motif(ts,query,Pab,Iab,l)

motifs=pd.read_csv('../motifs.txt', delim_whitespace=True, header=None)
motifs = motifs.drop(columns=[0]).sort_values([6])
motifs=remove_overlaps(motifs, 50)
print(type(motifs))
allPabs = []
for index, row in motifs.iterrows():
    # print("all={}, index={}".format(len(allPabs), index))
    start = int(row[1])
    stop = int(row[2])
    query = my_data.values.flatten()[start:stop]
    Pab, Iab = stamp(ts, query, l)  # run the STAMP algorithm to compute the Matrix Profile
    plot_motif(ts, query, Pab, Iab, l)
    break
    # allPabs=np.concatenate((allPabs, Pab))

# plot_motif(ts, query, Pab, Iab, l)






