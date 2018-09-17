from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from os import walk
from ldms import *
import csv
import ldms_transform as lt

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
    plt.show()

def plot_selected_motifs(ax, Ta, df, selectedMotifs, m, metric):
    ax.set_title(metric[:-4])
    ax.plot(df['#Time'].values, Ta, linestyle='--', alpha=0.5)

    counter = 1
    colors = ['g', 'y', 'b', 'r', 'c', 'm', 'k']

    for item in selectedMotifs:
        ax.plot(df['#Time'].iloc[range(item, item + m)], Ta[item:item + m], c=colors[counter % len(colors)],
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
    fig, axs = plt.subplots(nrows=4, sharex=True)

    selectedMotifsIndex = 1
    matrixProfileIndex = 0
    modelIndex = 3
    currentMotifsIntervalPlot = 2
    # sharedMotifsIntervalPlot = 4

    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)



    axs[selectedMotifsIndex].xaxis.set_major_formatter(xfmt)
    axs[selectedMotifsIndex].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[selectedMotifsIndex].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    plot_selected_motifs(axs[selectedMotifsIndex], Ta, df, selectedMotifs, m, metric)

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

def extractMoreFromSelectedMotif(selectedMotif, ts, topK, df, allMotifs, model, metric):
    start = int(selectedMotif[1])
    stop = int(selectedMotif[2])
    l = int(selectedMotif[5])
    print("l={}, start={}, stop={}".format(l, start, stop))
    query = ts[start:stop]
    Pab, Iab = stamp(ts, query, l)

    selectedMotifs = findTopKMotifs(Pab, topK, l)
    current_motifs = get_interval_tree_motifList(df, selectedMotifs, l)
    allMotifs.append(current_motifs)
    sharedMotifs = find_shared_period(allMotifs)

    # plot_motif_with_time(ts, query, Pab, Iab, l, df, model, selectedMotifs, current_motifs, sharedMotifs, metric)


#def calcIBSMDistance(allTimes, phaseSet1, phaseSet2):

    y = calcIBSMDistance(df['#Time'], get_highLevel_interval_tree_from_model(model), find_shared_period([current_motifs]))


    return y

def extract_motifs(metricFileNames, motifFileNames, dfs, model, l=400):
    topK=10
    allMotifs = []
    for metric, motif, df in zip(metricFileNames, motifFileNames, dfs):
        print("\nmetric: " + metric)
        ts = load_data(metric)
        motifs = load_motifs(motif)
        # print("before")
        # print(motifs)
        motifs = remove_overlaps(motifs, l)
        # print("after")
        # print(motifs)
        selectedMotif = motifs.iloc[0]
        # for index, selectedMotif in motifs.iterrows():
        #     print("Selected motif: {}".format(index))
        #     print(selectedMotif)
        y = extractMoreFromSelectedMotif(selectedMotif, ts, topK, df, allMotifs, model, metric)
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
    model = ldmsInstance.getMetricSet("waleElemXflowMixFrac3.5m").getDataFrame()

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
        path_Xeon_milestoneRun_abnormal: 'milestonerun',
        path_Xeon_milestoneRun_normal: 'milestonerun',
        path_KNL_WaleElem: 'waleElemXflowMixFrac3.5m',
        path_KNL_MilestoneRun: 'milestonerun'
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
    model = row['model']
    workload = row['workload']
    sampler = row['sampler']

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

    y = extract_motifs(metricFileNames, motifFileNames, dfs=dfs, model=ldmsInstance.getMetricSet(model).getDataFrame(), l=motifLength)

    row['IBSM'] = y


def process_data(df):
    print(multiprocessing.current_process().name)
    df.apply(lambda x: process_single(x),axis=1)
    df.to_csv(path_or_buf=multiprocessing.current_process().name + '.csv', index=False)
    # print(df2)
    # print(df3)
    # print(df4)
    # print(df5)
    # print(df6)


def compare_all3():
    print('compare_all3')





    all_cases = read_csv("all_cases.csv")
    all_cases_list = all_cases.values.tolist()
    print(len(all_cases_list))
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # results = pool.map(process_data, all_cases_list)

    # data_split = np.array_split(all_cases, 4)
    # pool.map(process_data,data_split)

    chunk_size = int(all_cases.shape[0] / multiprocessing.cpu_count() - 1)
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

if __name__ == '__main__':
    # pool = ThreadPool(8)
    xAxis = '#Time'
    value_name = 'value'
    # authors_example()

    # ldms_example()

    # test_all_dfs()
    # compare_all()
    # write_all()
    compare_all3()

    # testFloss()

    # authors_example()
    # testSimilarityMethod()
