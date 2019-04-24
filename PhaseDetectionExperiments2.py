import sys

from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from os import walk
from ldms import *
import statistics
import csv
import ldms_transform as lt

from scipy.stats import chisquare

from PhaseDetection import *
from PhaseDetectionPlot import *

from owlpy.core import *

def authors_example():
    data = pd.read_csv('ECGFiveDays_TRAIN', sep=',', header=None)
    label = data.pop(data.columns[0])

    print(data.head())

    my_data = pd.read_csv('test.txt', sep=',', header=None)
    print(my_data)
    print(my_data.values.flatten())

    # ts = data.iloc[:10].values.flatten() # pick a random sample from class 0
    ts = my_data.values.flatten()

    query = my_data.values.flatten()[1781:2181]

    l = 50
    # Pab, Iab = stamp(ts,query,l)                       #  `run the STAMP algorithm to compute the Matrix Profile
    # plot_motif(ts,query,Pab,Iab,l)

    motifs = pd.read_csv('motifs.txt', delim_whitespace=True, header=None)
    motifs = motifs.drop(columns=[0]).sort_values([6])
    motifs = remove_overlaps(motifs, 50)
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


def findTopKMotifs(values, k, m):

    print("findTopKMotifs")

    dataset = pd.DataFrame({'Pab': values})

    # print(dataset.sort_values(['Pab']).index.values)

    kList = list(range(0, k))
    # result = list(np.argpartition(values, kth=k))
    result = list(dataset.sort_values(['Pab']).index.values)
    i = 1
    pointer = 0
    # print("before")
    # print(result)
    # print(values[result])
    while pointer < len(result) - 1:
        i = pointer + 1
        while i < len(result):
            if abs(result[pointer] - result[i]) < m:
                # print("deleting " + str(i) + " = " + str(result[i]))
                del result[i]
            else:
                i = i + 1
        pointer = pointer + 1
    # print("after")
    # print(result)
    # print(values[result])
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


def plot_motif(Ta, Tb, values, indexes, m, df):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(Ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))
    print(np.argmin(values))
    print(np.argpartition(values, kth=[0, 1, 2]))
    print(values)

    showThis = findTopKMotifs(values, 3, m)

    counter = 1
    colors = ['g', 'y', 'b', 'r', 'c']



    for item in showThis:
        plt.plot(range(item, item + m), Ta[item:item + m], c=colors[counter % len(colors)],
                 label='Motif #' + str(counter))
        counter = counter + 1

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
    plt.show()['#Time']

def plot_selected_motifs(ax, Ta, df, selectedMotifs, m, metric):
    ax.set_title(metric[:-4])
    ax.plot(df['#Time'].values, Ta, linestyle='--', alpha=0.5)

    counter = 1
    colors = ['g', 'y', 'b', 'r', 'c', 'm', 'k']

    for item in selectedMotifs:
        ax.plot(df['#Time'].iloc[range(item, item + m)], Ta[item:item + m], c=colors[counter % len(colors)],
                    label='Motif #' + str(counter))
        counter = counter + 1

def plot_one_selected_motifs(ax, Ta, df, selectedMotifs, m, metric):
    ax.set_title("The best motif")
    # ax.plot(df['#Time'].values, Ta, linestyle='--', alpha=0.5)

    counter = 1
    colors = ['g', 'y', 'b', 'r', 'c', 'm', 'k']

    for item in selectedMotifs:
        ax.plot(df['#Time'].iloc[range(item, item + m)], Ta, c=colors[counter % len(colors)],
                    label='Motif #' + str(counter))
        counter = counter + 1

def plot_matrix_profile(ax, df, values):
    ax.set_title('Matrix Profile')
    ax.plot(df['#Time'].iloc[range(0, len(values))], values, '#ff5722')
    ax.plot(df['#Time'].iloc[np.argmax(values)], np.max(values), marker='x', c='r', ms=10)
    ax.plot(df['#Time'].iloc[np.argmin(values)], np.min(values), marker='^', c='g', ms=10)

def plot_motif_with_time(Ta, Tb, values, indexes, m, df, model, selectedMotifs, current_motifs, sharedMotifs, metric):
    from matplotlib import gridspec
    # plt.figure(figsize=(8, 4))
    # plt.subplot(211)
    fig, axs = plt.subplots(nrows=5, sharex=True)

    selectedMotifsIndex = 1
    oneMotifIndex = 2
    matrixProfileIndex = 0
    modelIndex = 4
    currentMotifsIntervalPlot = 3
    # sharedMotifsIntervalPlot = 4

    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)



    axs[selectedMotifsIndex].xaxis.set_major_formatter(xfmt)
    axs[selectedMotifsIndex].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[selectedMotifsIndex].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    plot_selected_motifs(axs[selectedMotifsIndex], Ta, df, selectedMotifs, m, metric)

    axs[oneMotifIndex].xaxis.set_major_formatter(xfmt)
    axs[oneMotifIndex].xaxis.set_major_locator(md.SecondLocator(interval=25))
    axs[oneMotifIndex].xaxis.set_minor_locator(md.SecondLocator(interval=5))
    plot_one_selected_motifs(axs[oneMotifIndex], Tb, df, [selectedMotifs[0]], len(Tb), metric)


    axs[matrixProfileIndex].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[matrixProfileIndex].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    axs[matrixProfileIndex].xaxis.set_major_formatter(xfmt)
    plot_matrix_profile(axs[matrixProfileIndex],df,values)

    axs[modelIndex].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[modelIndex].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    axs[modelIndex].xaxis.set_major_formatter(xfmt)
    axs[modelIndex].set_title("model motifs")
    plot_model(model,axs[modelIndex],df['#Time'].min(), df['#Time'].max())


    axs[currentMotifsIntervalPlot].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[currentMotifsIntervalPlot].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    axs[currentMotifsIntervalPlot].xaxis.set_major_formatter(xfmt)
    axs[currentMotifsIntervalPlot].set_title("current motifs")

    plot_interval(find_shared_period([current_motifs]), axs[currentMotifsIntervalPlot], df['#Time'].min(), df['#Time'].max())

    # axs[sharedMotifsIntervalPlot].xaxis.set_major_locator(md.SecondLocator(interval=100))
    # axs[sharedMotifsIntervalPlot].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    # axs[sharedMotifsIntervalPlot].xaxis.set_major_formatter(xfmt)
    # axs[sharedMotifsIntervalPlot].set_title("shared motifs")
    #
    # plot_interval(sharedMotifs, axs[sharedMotifsIntervalPlot], df['#Time'].min(), df['#Time'].max())

    fig.set_size_inches(h=11.176, w=15.232)
    fig.autofmt_xdate()
    print('saving figure')

    name = metric[:-4]
    format = '.png'
    path=''
    print(path + name + format)
    fig.savefig(path + name + format, dpi=600)

    fig.clf()
    plt.clf()
    plt.close()
    gc.collect()

    # axs[1].set_xlim((0, len(Ta)))
    # fig.autofmt_xdate()
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()

def writeColumnInFile(df,column='MPI_Issend.calls.0', transform=None, fileName=None, folder=''):
    this_sampler = df
    if transform != None:
        transformed_df = df
        if transform == "rate":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], True,False,False)
        elif transform == "log":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], False, False, True)
        elif transform == "sum":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [],False,True,False)
        else:
            print("unknown transform: " + transform)
        transformed_df.fillna(0, inplace=True)
        this_sampler = transformed_df

    if fileName == None:
        suffix = ''
        if transform != None:
            suffix = "." + transform + ".txt"
        else:
            suffix = ".txt"
        fileName = column + suffix
    thefile = open(folder + fileName, 'w')
    if transform != None:
        for item in this_sampler[ret_names[0]].values:
            thefile.write("%s\n" % item)
    else:
        for item in this_sampler[column].values:
            thefile.write("%s\n" % item)


def load_data(fileName='test.txt',folder=None):
    # if folder is None:
    #     fileName="metrics/"+fileName
    my_data = pd.read_csv(fileName, sep=',', header=None)
    ts = my_data.values.flatten()
    return ts

def load_motifs(fileName='motifs.txt', folder=None):
    if folder is not None:
        fileName = folder + fileName
    motifs = pd.read_csv(fileName, delim_whitespace=True, header=None)
    motifs = motifs.drop(columns=[0]).sort_values([6])
    return motifs

def extractMoreFromSelectedMotifAndPlot(selectedMotif, ts, topK, df, allMotifs, model, metric):
    Iab, Pab, current_motifs, l, query, selectedMotifs, sharedMotifs = extractMoreUsingSelectedMotif(allMotifs, df,
                                                                                                     selectedMotif,
                                                                                                     topK, ts)

    plot_motif_with_time(ts, query, Pab, Iab, l, df, model, selectedMotifs, current_motifs, sharedMotifs, metric)
    gc.collect()


#def calcIBSMDistance(allTimes, phaseSet1, phaseSet2):

    y = calcIBSMDistance(df['#Time'], get_highLevel_interval_tree_from_model(model), find_shared_period([current_motifs]))


    return y


def extractMoreUsingSelectedMotif(allMotifs, df, selectedMotif, topK, ts):

    Iab, Pab, selectedMotifs = findMoreMotifsUsingOne(selectedMotif, topK, ts)

    start = int(selectedMotif[1])
    stop = int(selectedMotif[2])
    l = int(selectedMotif[5])
    query = ts[start:stop]

    current_motifs = get_interval_tree_motifList(df, selectedMotifs, l)
    allMotifs.append(current_motifs)
    sharedMotifs = find_shared_period(allMotifs)
    return Iab, Pab, current_motifs, l, query, selectedMotifs, sharedMotifs


def findMoreMotifsUsingOne(selectedMotif, topK, ts):
    start = int(selectedMotif[1])
    stop = int(selectedMotif[2])
    l = int(selectedMotif[5])
    print("l={}, start={}, stop={}".format(l, start, stop))
    query = ts[start:stop]
    Pab, Iab = stamp(ts, query, l)
    selectedMotifs = findTopKMotifs(Pab, topK, l)
    return Iab, Pab, selectedMotifs


def extract_motifs(metricFileNames, motifFileNames, dfs, model, l=400):
    topK=10
    allMotifs = []
    for metric, motif, df in zip(metricFileNames, motifFileNames, dfs):
        y = extract_motifs_from_one_metric(allMotifs, df, l, metric, model, motif, topK)
    return y


def extract_motifs_from_one_metric(allMotifs, df, l, metric, model, motif, topK):
    print("\nmetric: " + metric)
    print("\nmotif: " + motif)
    ts = load_data(metric)
    motifs = load_motifs(motif)
    motifs = remove_overlaps(motifs, l)
    selectedMotif = motifs.iloc[0]
    y = extractMoreFromSelectedMotifAndPlot(selectedMotif, ts, topK, df, allMotifs, model, metric)
    return y


def test_motifs(metricFileName, motifFileName,df,model, l=50):
    ts = load_data(metricFileName)
    motifs = load_motifs(motifFileName)
    motifs = remove_overlaps(motifs, l)
    for index, row in motifs.iterrows():
        start = int(row[1])
        stop = int(row[2])
        query = ts[start:stop]
        Pab, Iab = stamp(ts, query, l)  # run the STAMP algorithm to compute the Matrix Profile
        plot_motif_with_time(ts, query, Pab, Iab, l, df, model)
        break

def ldms_example():
    print('ldms_example')

    ldmsInstance = LDMSInstance(datasets=['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'waleElemXflowMixFrac3.5m'],
        path='C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/')
    meminfo = ldmsInstance.getMetricSet("meminfo").getDataFrame()
    shm_sampler = ldmsInstance.getMetricSet("shm_sampler").getDataFrame()
    vmstat = ldmsInstance.getMetricSet("vmstat").getDataFrame()
    procstat = ldmsInstance.getMetricSet("procstat").getDataFrame()
    procnetdev = ldmsInstance.getMetricSet("procnetdev").getDataFrame()
    model = ldmsInstance.getMetricSet("waleElemXflowMixFrac3.5m").getDataFrame()

    writeColumnInFile(shm_sampler,transform="rate")

    # test_motifs('test.txt', 'motifs.txt', l=50)
    # test_motifs('MPI_Issend.calls.0.rate.txt', 'motif.rate.txt', l=50)

    # test_motifs('test.txt', 'motifs.400.txt', l=400)
    # test_motifs('MPI_Issend.calls.0.rate.txt', 'motif.400.rate.txt', l=400)

    # test_motifs('MPI_Issend.calls.0.txt', 'motifs.60.txt', l=60)
    shm_sampler['#Time'] = shm_sampler['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    # test_motifs('MPI_Issend.calls.0.rate.txt', 'motifs.400.txt', df=shm_sampler, model=model, l=400)
    # test_motifs('MPI_Issend.calls.0.rate.txt', 'motifs.600.txt', df=shm_sampler, model=model, l=420)
    extract_motifs(['MPI_Issend.calls.0.rate.txt'], ['motifs.600.txt'], dfs=[shm_sampler], model=model, l=420)

def test_all_dfs():
    print('test_all_dfs')

    ldmsInstance = LDMSInstance(datasets=['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev','procnfs', 'waleElemXflowMixFrac3.5m'],
        path='C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/')
    meminfo = ldmsInstance.getMetricSet("meminfo").getDataFrame()
    shm_sampler = ldmsInstance.getMetricSet("shm_sampler").getDataFrame()
    vmstat = ldmsInstance.getMetricSet("vmstat").getDataFrame()
    procstat = ldmsInstance.getMetricSet("procstat").getDataFrame()
    procnetdev = ldmsInstance.getMetricSet("procnetdev").getDataFrame()
    procnfs = ldmsInstance.getMetricSet("procnfs").getDataFrame()
    model = ldmsInstance.get_interval_tree_motifListgetMetricSet("waleElemXflowMixFrac3.5m").getDataFrame()

    # writeColumnInFile(shm_sampler,column='MPI_Issend.calls.0', transform=None)
    # writeColumnInFile(meminfo, column='Dirty', transform=None)
    # writeColumnInFile(procstat, column='user', transform=None)
    # writeColumnInFile(vmstat, column='numa_hit', transform=None)
    # writeColumnInFile(procnetdev, column='tx_bytes#eth0', transform=None)
    # writeColumnInFile(procnfs, column='numcalls', transform=None)
    # writeColumnInFile(procnfs, column='read', transform=None)
    # writeColumnInFile(procnfs, column='write', transform=None)
    #
    # writeColumnInFile(shm_sampler,column='MPI_Issend.calls.0', transform="rate")
    # writeColumnInFile(procstat, column='user', transform="rate")
    # writeColumnInFile(vmstat, column='numa_hit', transform="rate")
    # writeColumnInFile(procnetdev, column='tx_bytes#eth0', transform="rate")
    # writeColumnInFile(procnfs, column='numcalls', transform="rate")
    # writeColumnInFile(procnfs, column='read', transform="rate")
    # writeColumnInFile(procnfs, column='write', transform="rate")
    shm_sampler['#Time'] = shm_sampler['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    meminfo['#Time'] = meminfo['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    vmstat['#Time'] = vmstat['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    procstat['#Time'] = procstat['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    procnetdev['#Time'] = procnetdev['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    procnfs['#Time'] = procnfs['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

    # metricFileNames = ['Dirty.txt','MPI_Issend.calls.0.rate.txt','MPI_Issend.calls.0.txt','numa_hit.rate.txt','numa_hit.txt','numcalls.rate.txt','numcalls.txt','read.rate.txt',
    #                    'read.txt','tx_bytes#eth0.rate.txt','tx_bytes#eth0.txt','user.rate.txt','user.txt','write.rate.txt','write.txt']
    metricFileNames = ['Dirty.txt','numcalls.txt','tx_bytes#eth0.rate.txt','MPI_Issend.calls.0.rate.txt']
    motifFileNames = metricFileNames
    # dfs=[meminfo, shm_sampler, shm_sampler, vmstat, vmstat, procnfs, procnfs, procnfs, procnfs, procnetdev, procnetdev, procstat, procstat, procnfs, procnfs]
    dfs = [meminfo, procnfs, procnetdev, shm_sampler]
    extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=model, l=250)



def testFloss():
    # data = pd.read_csv('Data/EEGRat_10_1000.txt', header=None)
    data = pd.read_csv('Data/DutchFactory_24_2184.txt', header=None)

    # ts = data.values.flatten()
    ts = load_data(fileName='write.txt', folder="metrics/")
    print(ts)
    print(len(ts))
    crossCount = runSegmentation(ts,50)
    print(crossCount)
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(crossCount, linestyle='--', alpha=0.5)
    plt.subplot(212)
    plt.plot(ts)
    plt.show()

def write_all():
    base_path = 'D:/ac/PhD/Research/data/pd/02 - testAll/'
    # 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'
    path_Xeon_milestoneRun_abnormal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000'

    # 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN2/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'
    path_Xeon_milestoneRun_normal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000'

    # 'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'
    path_KNL_WaleElem = 'ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000'

    # 'D:/ac/PhD/Research/data/01/KNL_Overhead_RUN5/overhead/ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000'
    path_KNL_MilestoneRun = 'ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000'

    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']

    # intput_paths = ['C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/']
    input_paths = [path_Xeon_milestoneRun_abnormal, path_Xeon_milestoneRun_normal, path_KNL_WaleElem,
                   path_KNL_MilestoneRun]
    counter = 0
    for ip in input_paths:
        print("path={}\n\t".format(ip))
        directory = "metrics/" + ip + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)


        ldmsInstance = LDMSInstance(datasets=all_samplers,
                                    path=base_path + ip + '/')

        all_samplers_df = {}
        for s in all_samplers:
            all_samplers_df[s] = ldmsInstance.getMetricSet(s).getDataFrame()
            # all_samplers_df[s]['#Time'] = all_samplers_df[s]['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

        base_dir = directory
        for n,df in all_samplers_df.items():
            print("df={}\n\t\t".format(n))

            directory = base_dir + n + "/"

            if not os.path.exists(directory):
                os.makedirs(directory)

            for c in df.columns:
                if c in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id']:
                    continue
                print("C={}\n\t\t\t".format(c))
                for t in ["rate", None]:
                    print("t={}\n\t\t\t".format(t))
                    suffix = ''
                    if t != None:
                        suffix = "." + t + ".txt"
                    else:
                        suffix = ".txt"
                    fileName = c + suffix
                    writeColumnInFile(df, column=c, transform=t,folder=directory, fileName=fileName)


def compare_all():
    print('compare_all')
    # 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'
    path_Xeon_milestoneRun_abnormal = 'D:/ac/PhD/Research/data/pd/02 - testAll/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'

#'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN2/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'
    path_Xeon_milestoneRun_normal = 'D:/ac/PhD/Research/data/pd/02 - testAll/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'

#'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'
    path_KNL_WaleElem = 'D:/ac/PhD/Research/data/pd/02 - testAll/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'

#'D:/ac/PhD/Research/data/01/KNL_Overhead_RUN5/overhead/ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000'
    path_KNL_MilestoneRun = 'D:/ac/PhD/Research/data/pd/02 - testAll/ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000/'



    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev','procnfs']

    # intput_paths = ['C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/']
    input_paths = [path_Xeon_milestoneRun_abnormal, path_Xeon_milestoneRun_normal, path_KNL_WaleElem, path_KNL_MilestoneRun]

    ModelMap = {
        path_Xeon_milestoneRun_abnormal : 'milestonerun',
        path_Xeon_milestoneRun_normal : 'milestonerun',
        path_KNL_WaleElem : 'waleElemXflowMixFrac3.5m',
        path_KNL_MilestoneRun : 'milestonerun'
    }

    # models = ['waleElemXflowMixFrac3.5m']

    motifLengths = [250]

    selectedMetricsFromSamplers = {
        'meminfo' : 'Dirty.txt',
        'procnfs': 'numcalls.txt',
        'procnetdev': 'tx_bytes#eth0.rate.txt',
        'shm_sampler': 'MPI_Issend.calls.0.rate.txt'
    }

    for ip in input_paths:
        model = ModelMap[ip]
        # for model in models:
        for ml in motifLengths:
            print("path={}\nmodel={}\nmotifLength={}".format(ip, model, ml))
            ds = all_samplers + [model]
            ldmsInstance = LDMSInstance(datasets=ds,
                path=ip)

            all_samplers_df = {}
            for s in all_samplers:
                all_samplers_df[s] = ldmsInstance.getMetricSet(s).getDataFrame()
                all_samplers_df[s]['#Time'] = all_samplers_df[s]['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
            dfs = []
            for s,m in selectedMetricsFromSamplers.items():
                dfs.append(all_samplers_df[s])

            metricFileNames = selectedMetricsFromSamplers.values()

            motifFileNames = metricFileNames
            extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=ldmsInstance.getMetricSet(model).getDataFrame(), l=ml)

def compare_all2():
    print('compare_all2')
    # 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'
    path_Xeon_milestoneRun_abnormal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'

#'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN2/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'
    path_Xeon_milestoneRun_normal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'

#'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'
    path_KNL_WaleElem = 'ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'

#'D:/ac/PhD/Research/data/01/KNL_Overhead_RUN5/overhead/ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000'
    path_KNL_MilestoneRun = 'ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000/'



    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev','procnfs']

    # intput_paths = ['C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/']
    input_paths = [path_Xeon_milestoneRun_abnormal, path_Xeon_milestoneRun_normal, path_KNL_WaleElem, path_KNL_MilestoneRun]

    ModelMap = {
        path_Xeon_milestoneRun_abnormal : 'milestonerun',
        path_Xeon_milestoneRun_normal : 'milestonerun',
        path_KNL_WaleElem : 'waleElemXflowMixFrac3.5m',
        path_KNL_MilestoneRun : 'milestonerun'
    }

    # models = ['waleElemXflowMixFrac3.5m']

    motifLengths = [250]

    selectedMetricsFromSamplers = {
        'meminfo' : 'Dirty.txt',
        'procnfs': 'numcalls.txt',
        'procnetdev': 'tx_bytes#eth0.rate.txt',
        'shm_sampler': 'MPI_Issend.calls.0.rate.txt'
    }

    # all_cases = pd.DataFrame()
    all_cases_list = []
    for ip in input_paths:
        model = ModelMap[ip]
        ds = all_samplers + [model]
        ldmsInstance = LDMSInstance(datasets=ds,
                                    path=ip)

        all_samplers_df = {}
        print(ip)
        for s in all_samplers:
            all_samplers_df[s] = ldmsInstance.getMetricSet(s).getDataFrame()
            all_samplers_df[s]['#Time'] = all_samplers_df[s]['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

        for (dirpath, samplers, filenames) in walk("metrics/"+ ip ):

            for sampler in samplers:
                print(len(all_cases_list))
                print(sampler)
                # sampler_df = all_samplers_df[sampler]
                # dfs = [sampler_df]
                for (dirpath, dirnames, metrics) in walk("metrics/" + ip + sampler):
                    print(len(metrics))
                    for metric in metrics:
                        # print(dirpath + "/" + metric)
                        # metricFileNames = [dirpath + "/" + metric]

                        for (dirpath, motifLengths, fs) in walk("motifs/" + ip + sampler):
                            for motifLength in motifLengths:
                                # print(dirpath + "/" + motifLength + "/" + metric)
                                # motifFileNames = [dirpath + "/" + motifLength + "/" + metric]

                                all_cases_list.append([ip, model, sampler, metric, motifLength, -1])
                                # print(len(all_cases_list))

                                # y = extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=ldmsInstance.getMetricSet(model).getDataFrame(), l=motifLength)
                                #
                                # with open('employee_file.csv', mode='w') as employee_file:
                                #     employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"',
                                #                                  quoting=csv.QUOTE_MINIMAL)
                                # output = "".format()
    all_cases = pd.DataFrame(all_cases_list, columns=['workload', 'model', 'sampler', 'metric', 'motifLength', 'IBSM'])
    all_cases.to_csv(path_or_buf ='all_cases.csv' ,index=False)

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # results = pool.starmap(process_column, args)
    # pool.close()
    # pool.join()
    #
    # chunk_size = int(all_cases.shape[0] / 4)
    # for start in range(0, all_cases.shape[0], chunk_size):
    #     df_subset = all_cases.iloc[start:start + chunk_size]
    #     process_data(df_subset)


        #
        #     motifFileNames = metricFileNames
        #
        #     print()
        # # for model in models:
        # for ml in range(50,800,50):
        #     print("path={}\nmodel={}\nmotifLength={}".format(ip, model, ml))
        #
        #
        #
        #
        #     dfs = []
        #     for s,m in selectedMetricsFromSamplers.items():
        #         dfs.append(all_samplers_df[s])
        #
        #     metricFileNames = selectedMetricsFromSamplers.values()
        #
        #     motifFileNames = metricFileNames
        #     extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=ldmsInstance.getMetricSet(model).getDataFrame(), l=ml)



def process_single(row):
    print(row)

    # 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'
    path_Xeon_milestoneRun_abnormal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'

    # 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN2/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'
    path_Xeon_milestoneRun_normal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'

    # 'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'
    path_KNL_WaleElem = 'ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'

    # 'D:/ac/PhD/Research/data/01/KNL_Overhead_RUN5/overhead/ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000'
    path_KNL_MilestoneRun = 'ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000/'

    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']

    # intput_paths = ['C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/MPI_Sampler/Nalu/experiments/07/nfs/ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/']
    input_paths = [path_Xeon_milestoneRun_abnormal, path_Xeon_milestoneRun_normal, path_KNL_WaleElem,
                   path_KNL_MilestoneRun]

    ModelMap = {
        path_Xeon_milestoneRun_abnormal: 'milestoneRun',
        path_Xeon_milestoneRun_normal: 'milestoneRun',
        path_KNL_WaleElem: 'waleElemXflowMixFrac3.5m',
        path_KNL_MilestoneRun: 'milestoneRun'
    }



    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']

    for  p,m in ModelMap.items():
        ds = all_samplers + [m]

        ldms_instance_map[p] = LDMSInstance(datasets=ds,
                                path=p)

    # print(type(row))
    # print(len(row))
    # print(row)
    # = pd.DataFrame(all_cases_list, columns=['workload', 'model', 'sampler', 'metric', 'motifLength', 'IBSM'])
    metric = row['metric']
    motifLength = row['motifLength']

    workload = row['workload']
    sampler = row['sampler']
    model = ModelMap[workload]  # row['model']

    motifFileNames = ["motifs/" + workload + sampler + "/" + str(motifLength) + "/" + metric]
    metricFileNames = ["metrics/" + workload + sampler + "/" + metric]


    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']
    all_samplers_df = {}

    all_samplers_df_initialized = False
    # model = ModelMap[ip]
    ldmsInstance = ldms_instance_map[workload]


    for s in all_samplers:
        all_samplers_df[s] = ldmsInstance.getMetricSet(s).getDataFrame()
        all_samplers_df[s]['#Time'] = all_samplers_df[s]['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

    sampler_df = all_samplers_df[sampler]
    dfs = [sampler_df]

    try:
        y = extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=ldmsInstance.getMetricSet(model).getDataFrame(), l=motifLength)
    except Exception as e:
        y = e.message
    print("y=")
    print(y)
    row['IBSM'] = y


def process_data(df):
    print(multiprocessing.current_process().name)
    total = df.shape[0]
    df.apply(lambda x: process_single(x),axis=1)
    df.to_csv(path_or_buf=multiprocessing.current_process().name + '.csv', index=False)
    # print(df2)
    # print(df3)
    # print(df4)
    # print(df5)
    # print(df6)


def compare_all3(tid):
    print('compare_all3')

    ppcounter = 0
    total = 0



    all_cases = read_csv("all_cases.csv")
    all_cases_list = all_cases.values.tolist()
    print(len(all_cases_list))
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # results = pool.map(process_data, all_cases_list)

    # data_split = np.array_split(all_cases, 4)
    # pool.map(process_data,data_split)

    chunk_size = int(all_cases.shape[0] / multiprocessing.cpu_count() - 1)
    print("ch=" + str(all_cases.shape[0] / 63))
    results = []
    for start in range(0, all_cases.shape[0], chunk_size):
        df_subset = all_cases.iloc[start:start + chunk_size]
        # print(df_subset)
        # pool.apply(process_data,df_subset)
        # pool.map(process_data, df_subset)
        args = []
        args.append((df_subset))
        # print(args)
        result = pool.apply_async(process_data,args)
        results.append(result)
        # process_data(df_subset)
    for result in results:
        result.get()
    pool.close()
    pool.join()

    # = pd.DataFrame(all_cases_list, columns=['workload', 'model', 'sampler', 'metric', 'motifLength', 'IBSM'])

ldms_instance_map = {}

ppcounter = 0
total = 0

def appl_f(row):
    if row[5].isnumeric():
        return row

def concat_all_dfs():
    # df_list = []
    # for i in range(0,63):
    #     print(i)
    #     data = pd.read_csv('D:/ac/PhD/Research/data/pd/02 - testAll/logsandcsvs/csvs/p' + str(i) + '.csv', sep=',', header=None)
    #     df_list.append(data)
    # result = pd.concat(df_list)
    result = pd.read_csv('allData.csv', sep=',',
                       header=None)
    print(result.shape)
    # print(result[result[5].isnumeric()].shape)
    # r = result[result.apply(lambda x: print(type(x[5])),axis=1)]
    r= result[pd.to_numeric(result[5], errors='coerce').notnull()]
    print(r.shape)
    r.to_csv(path_or_buf= 'allData_filterred.csv', index=False)


def transform_workload(row):
    workload = row['workload']
    if workload == 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/':
        row['workload'] = 'w0'
    if workload == 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/':
        row['workload'] = 'w1'
    if workload == 'ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/':
        row['workload'] = 'w2'
    if workload == 'ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000/':
        row['workload'] = 'w3'
    return row

def study_data(workload='w2'):
    # o = pd.read_csv('allData.csv', sep=',',
    #                    header=None)
    result = pd.read_csv('allData_filterred.csv', sep=',')
    print(result.shape)
    # print(result.columns)
    # result.columns = ['workload',  'model'  ,'sampler' , 'metric' , 'motifLength' , 'IBSM']
    # print(result.columns)
    # print(result.head())
    # print(result['IBSM'].describe())
    # print(result[result['IBSM']> 5].describe())
    # result[result['IBSM'] > 5].to_csv(path_or_buf='allData_filterred.csv', index=False)
    # print(result[result[5]<20])
    # result[result[5] < 20].to_csv(path_or_buf= 'allData_lt20.csv', index=False)

    r = result.apply(transform_workload, axis=1)
    # print(r[r['workload'] == 'w3'].describe())
    # hist = r.hist(bins=20,by=['workload','sampler'])


    print(r[(r['workload'] == workload) & (r['sampler'] == 'meminfo')].groupby('metric').describe())
    r[(r['workload'] == workload) & (r['sampler'] == 'meminfo')].groupby('metric').describe().to_csv("meminfo.csv")
    r[(r['workload'] == workload) & (r['sampler'] == 'procnfs')].groupby('metric').describe().to_csv("procnfs.csv")
    r[(r['workload'] == workload) & (r['sampler'] == 'procstat')].groupby('metric').describe().to_csv("procstat.csv")
    r[(r['workload'] == workload) & (r['sampler'] == 'procnetdev')].groupby('metric').describe().to_csv("procnetdev.csv")
    r[(r['workload'] == workload) & (r['sampler'] == 'vmstat')].groupby('metric').describe().to_csv("vmstat.csv")
    r[(r['workload'] == workload) & (r['sampler'] == 'shm_sampler')].groupby('metric').describe().to_csv("shm_sampler.csv")
    hist = r[(r['workload'] == workload) & (r['sampler'] == 'meminfo')].hist(bins=20, by=['sampler'])

    plt.show()

def study_current_params_gen_raphs(metrics, sampler,motifLength = 250):
    path_Xeon_milestoneRun_abnormal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'
    path_Xeon_milestoneRun_normal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'
    path_KNL_WaleElem = 'ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'
    path_KNL_MilestoneRun = 'ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000/'

    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']

    input_paths = [path_Xeon_milestoneRun_abnormal, path_Xeon_milestoneRun_normal, path_KNL_WaleElem,
                   path_KNL_MilestoneRun]

    ModelMap = {
        path_Xeon_milestoneRun_abnormal: 'milestoneRun',
        path_Xeon_milestoneRun_normal: 'milestoneRun',
        path_KNL_WaleElem: 'waleElemXflowMixFrac3.5m',
        path_KNL_MilestoneRun: 'milestoneRun'
    }

    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']

    for p, m in ModelMap.items():
        ds = all_samplers + [m]

        ldms_instance_map[p] = LDMSInstance(datasets=ds,
                                            path=p)

    origin_workload = path_KNL_WaleElem
    ldmsInstance = ldms_instance_map[origin_workload]

    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']
    all_samplers_df = {}

    sampler_motif_length = {}
    for s in all_samplers:
        all_samplers_df[s] = ldmsInstance.getMetricSet(s).getDataFrame()
        all_samplers_df[s]['#Time'] = all_samplers_df[s]['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
        sampler_motif_length[s] = motifLength

    sampler_df = all_samplers_df[sampler]
    dfs = [sampler_df]

    model = ModelMap[origin_workload]

    sampler_motif_length['procnetdev'] = 200
    sampler_motif_length['vmstat'] = 200

    motifLength = sampler_motif_length[sampler]
    for metric in metrics:
        print(metric)
        metricFileNames = ["metrics/" + origin_workload + sampler + "/" + metric]
        motifFileNames = ["motifs/" + origin_workload + sampler + "/" + str(motifLength) + "/" + metric]
        y = extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=ldmsInstance.getMetricSet(model).getDataFrame(),
                       l=motifLength)
        gc.collect()
        print(y)

def study_current_params(df,metrics, sampler,workload='w2'):
    # r = df[(df['workload'] == workload) & (df['sampler'] == sampler) & (df.metric.isin(metrics))]
    # print(r.groupby(['metric','motifLength']).describe())
    # df.groupby(['workload','sampler','metric','motifLength']).describe().to_csv('workload_sampler_metrics.csv')
    # hist = df.hist(bins=20, by=['motifLength'])

    # df.groupby(['workload']).describe().to_csv('based_workload.csv')
    # hist = df.hist(bins=20, by=['motifLength'])
    #
    # df.groupby(['workload','sampler']).describe().to_csv('based_workload_sampler.csv')
    # hist = df.hist(bins=20, by=['motifLength'])
    #
    # df.groupby(['workload','motifLength']).describe().to_csv('based_workload_motiflength.csv')
    # hist = df.hist(bins=20, by=['motifLength'])
    #
    # df.groupby(['workload','sampler','metric']).describe().to_csv('based_workload_sampler_metric.csv')
    # hist = df.hist(bins=20, by=['motifLength'])

    df.groupby(['sampler','motifLength']).describe().to_csv('based_sampler_motifLength.csv')
    hist = df.hist(bins=20, by=['motifLength'])

    plt.show()



def study_specific_metrics():
    result = pd.read_csv('allData_filterred.csv', sep=',')
    r = result.apply(transform_workload, axis=1)
    sampler_metrics = {
        'procnfs' : ['write.rate.txt', 'numcalls.txt'],
        'meminfo' : ['Dirty.txt', 'Slab.rate.txt'],
        'procnetdev' : ['rx_bytes#eth0.rate.txt', 'rx_packets#eth0.rate.txt', 'tx_bytes#eth0.rate.txt', 'tx_packets#eth0.rate.txt'],
        'vmstat' : ['nr_writeback.txt','pgactivate.rate.txt','nr_dirty.txt'],
        'procstat' : ['user.rate.txt', 'per_core_user0.rate.txt', 'sys.rate.txt', 'per_core_sys1.txt', 'per_core_softirqd0.rate.txt'],
        'shm_sampler' : ['MPI_Send.calls.1.txt','MPI_Ssend.calls.1.rate.txt','MPI_Wait.calls.0.rate.txt', 'MPI_Irecv.calls.0.rate.txt', 'MPI_Isend.calls.0.rate.txt', 'MPI_Issend.calls.1.rate.txt']
    }

    study_current_params(r, None, None)
    for s,m in sampler_metrics.items():
        print(s)
        # study_current_params(r, m, s)
        # study_current_params_gen_raphs(m,s)
        gc.collect()

def test_worst_case():
    result = pd.read_csv('allData_filterred.csv', sep=',')
    # r = result.apply(transform_workload, axis=1)
    sampler_metrics = {
        'shm_sampler' : ['MPI_Allreduce.calls.0.txt']
    }
    study_current_params_gen_raphs(['MPI_Allreduce.calls.0.txt'], 'shm_sampler',motifLength=400)
    # study_current_params_gen_raphs(['MPI_Allreduce.calls.0.txt'], 'shm_sampler', motifLength=400)


def chi2IsUniform(dataSet, significance=0.05):
    print(dataSet)
    chisq, pvalue = chisquare(dataSet)
    print("chisq={}, pvalue={}".format(chisq,pvalue))
    return pvalue > significance

def init_for_motif_finding(motifLength, sampler):
    path_Xeon_milestoneRun_abnormal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/'
    path_Xeon_milestoneRun_normal = 'ModelmilestoneRunPlacementVersion6SamplingVersion1RUN2Interval100000/'
    path_KNL_WaleElem = 'ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/'
    path_KNL_MilestoneRun = 'ModelmilestoneRunPlacementVersion1SamplingVersion1NProc272RUN5Interval1000000/'
    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']
    input_paths = [path_Xeon_milestoneRun_abnormal, path_Xeon_milestoneRun_normal, path_KNL_WaleElem,
                   path_KNL_MilestoneRun]
    ModelMap = {
        path_Xeon_milestoneRun_abnormal: 'milestoneRun',
        path_Xeon_milestoneRun_normal: 'milestoneRun',
        path_KNL_WaleElem: 'waleElemXflowMixFrac3.5m',
        path_KNL_MilestoneRun: 'milestoneRun'
    }
    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']
    for p, m in ModelMap.items():
        ds = all_samplers + [m]

        ldms_instance_map[p] = LDMSInstance(datasets=ds,
                                            path=p)
    origin_workload = path_KNL_WaleElem
    ldmsInstance = ldms_instance_map[origin_workload]
    all_samplers_df = {}
    sampler_motif_length = {}
    for s in all_samplers:
        all_samplers_df[s] = ldmsInstance.getMetricSet(s).getDataFrame()
        all_samplers_df[s]['#Time'] = all_samplers_df[s]['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
        sampler_motif_length[s] = motifLength
    sampler_df = all_samplers_df[sampler]
    dfs = [sampler_df]
    model = ModelMap[origin_workload]
    sampler_motif_length['procnetdev'] = 200
    sampler_motif_length['vmstat'] = 200
    motifLength = sampler_motif_length[sampler]
    return dfs, ldmsInstance, model, motifLength, origin_workload


def get_motifs_from_metric(metric, motifLength, origin_workload, sampler):

    topK=10

    print(metric)
    metricFileName = "metrics/" + origin_workload + sampler + "/" + metric
    motifFileName = "motifs/" + origin_workload + sampler + "/" + str(motifLength) + "/" + metric

    ts = load_data(metricFileName)
    motifs = load_motifs(motifFileName)
    motifs = remove_overlaps(motifs, motifLength)
    selectedMotif = motifs.iloc[0]

    Iab, Pab, selectedMotifs = findMoreMotifsUsingOne(selectedMotif, topK, ts)

    print(selectedMotifs)
    return selectedMotifs



def testchi2IsUniform():
    print("testchi2IsUniform")
    result = pd.read_csv('allData_filterred.csv', sep=',')
    # r = result.apply(transform_workload, axis=1)
    sampler_metrics = {
        'shm_sampler' : ['MPI_Allreduce.calls.0.txt']
    }

    dfs, ldmsInstance, model, motifLength, origin_workload = init_for_motif_finding(motifLength=250, sampler='shm_sampler')
    selectedMotifs1 = get_motifs_from_metric('MPI_Issend.calls.1.rate.txt', motifLength, origin_workload, sampler='shm_sampler')

    dfs, ldmsInstance, model, motifLength, origin_workload = init_for_motif_finding(motifLength=400, sampler='shm_sampler')
    selectedMotifs2 = get_motifs_from_metric('MPI_Allreduce.calls.1.rate.txt', motifLength, origin_workload, sampler='shm_sampler')
    # selectedMotifs.remove(51)
    # print(selectedMotifs)
    test = [5,205,405,605,805,1005,1205]
    test2 = [199809, 200665, 199607, 200270, 199649]
    test3 = [300,400,600,700,900,1000]
    test4 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    test5 = [0.01,0.2,0.5,0.8,0.9]

    selectedMotifs = np.ediff1d(sorted(selectedMotifs1))
    selectedMotifs2 = np.ediff1d(sorted(selectedMotifs2))
    test = np.ediff1d(sorted(test))
    test2 = np.ediff1d(sorted(test2))
    test3 = np.ediff1d(sorted(test3))
    test4 = np.ediff1d(sorted(test4))
    test5 = np.ediff1d(sorted(test5))

    # print()

    # test5 = test5 / max(test5)
    print(chi2IsUniform(selectedMotifs))
    print(statistics.stdev(selectedMotifs))

    selectedMotifs1.remove(51)
    selectedMotifs = np.ediff1d(sorted(selectedMotifs1))
    print(chi2IsUniform(selectedMotifs))


    selectedMotifs = selectedMotifs / (max(selectedMotifs))
    print(chi2IsUniform(selectedMotifs))
    print(statistics.stdev(selectedMotifs))




    print(chi2IsUniform(selectedMotifs2))
    print(statistics.stdev(selectedMotifs2))
    selectedMotifs2 = selectedMotifs2 / (max(selectedMotifs2))
    print(chi2IsUniform(selectedMotifs2))
    print(statistics.stdev(selectedMotifs2))

    print(chi2IsUniform(test))
    print(statistics.stdev(test))
    print(chi2IsUniform(test2))
    print(statistics.stdev(test2))
    print(chi2IsUniform(test3))
    print(statistics.stdev(test3))
    print(chi2IsUniform(test4))
    print(statistics.stdev(test4))
    print(chi2IsUniform(test5))
    print(statistics.stdev(test5))







if __name__ == '__main__':
    # pool = ThreadPool(8)
    # tid = int(sys.argv[1])
    xAxis = '#Time'
    value_name = 'value'
    # authors_example()
    ppcounter = 0
    total = 0
    # ldms_example()

    # test_all_dfs()
    # compare_all()
    # write_all()
    # compare_all3(tid)
    # concat_all_dfs()
    # study_data()
    # study_specific_metrics()
    # test_worst_case()
    testchi2IsUniform()

    # testFloss()

    # authors_example()
    # testSimilarityMethod()
