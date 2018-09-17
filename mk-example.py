

import pandas as pd

print(__name__)

import sys
print(sys.path)
sys.path.append('C:\\Users\\Ramin\\PycharmProjects\\GGS-git\\mkalgo')
print(sys.path)

from mkalgo.mk import mk_eab, mk
from ldms import *



def test():
    path = 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'

    print('ldms_example')

    ldmsInstance = LDMSInstance()

    meminfo = ldmsInstance.getMetricSet("meminfo").getDataFrame()
    shm_sampler = ldmsInstance.getMetricSet("shm_sampler").getDataFrame()
    vmstat = ldmsInstance.getMetricSet("vmstat").getDataFrame()
    procstat = ldmsInstance.getMetricSet("procstat").getDataFrame()
    procnetdev = ldmsInstance.getMetricSet("procnetdev").getDataFrame()
    milestoneRun = ldmsInstance.getMetricSet("milestoneRun").getDataFrame()

    thefile = open('test.txt', 'w')
    for item in shm_sampler['MPI_Issend.calls.0'].values:
        thefile.write("%s\n" % item)

    # obj = mk_eab(l=20, metric='euclidean', r=10)
    # print(obj)
    # motif_a, motif_b = obj.search(shm_sampler['MPI_Issend.calls.0'].values)
    # print(motif_a)
    # print(motif_b)
    #
    # obj = mk_eab(l=128, metric='euclidean', r=10)
    # c = pd.read_csv('s.txt')
    # motif_a, motif_b = obj.search(c['header'].values)
    # print(motif_a)
    # print(motif_b)


if __name__ == '__main__':
    print("start")
    test()