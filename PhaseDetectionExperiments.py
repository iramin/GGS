from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

from ldms import *
import ldms_transform as lt

from PhaseDetection import *
from PhaseDetectionPlot import *


class PDExperiment(object):

    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, name, *args, **kwargs):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.name = name

        self.ldmsInstance = LDMSInstance(datasets=['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'milestoneRun'])

        self.KMaxMap['meminfo'] = kwargs.get('kMax_meminfo', 8)
        self.KMaxMap['shm_sampler'] = kwargs.get('kMax_shm_sampler', 234)
        self.KMaxMap['vmstat'] = kwargs.get('kMax_vmstat', 20)
        self.KMaxMap['procstat'] = kwargs.get('kMax_procstat', 20)
        self.KMaxMap['procnetdev'] = kwargs.get('kMax_procnetdev', 20)
        self.KMaxMap['milestoneRun'] = kwargs.get('kMax_milestoneRun', 234)

        self.lambMap['meminfo'] = kwargs.get('lamb_meminfo', 1e-1)
        self.lambMap['shm_sampler'] = kwargs.get('lamb_shm_sampler', 1e-1)
        self.lambMap['vmstat'] = kwargs.get('lamb_vmstat', 1e-1)
        self.lambMap['procstat'] = kwargs.get('klamb_procstat', 1e-1)
        self.lambMap['procnetdev'] = kwargs.get('lamb_procnetdev', 1e-1)
        self.lambMap['milestoneRun'] = kwargs.get('lamb_milestoneRun', 1e-1)

        self.phaseTMin = kwargs.get('phaseTMin', timedelta(microseconds=500000))
        self.phaseTMax = kwargs.get('phaseTMax', timedelta(microseconds=10000000))

        self.verbose = kwargs.get('verbose', False)
        self.numThreads = kwargs.get('numThreads', 8)

        self.selectedMetrics['meminfo'] = kwargs.get('selectedMetrics_meminfo', 8)
        self.selectedMetrics['shm_sampler'] = kwargs.get('selectedMetrics_shm_sampler', 234)
        self.selectedMetrics['vmstat'] = kwargs.get('selectedMetrics_vmstat', 20)
        self.selectedMetrics['procstat'] = kwargs.get('selectedMetrics_procstat', 20)
        self.selectedMetrics['procnetdev'] = kwargs.get('selectedMetrics_procnetdev', 20)
        self.selectedMetrics['milestoneRun'] = kwargs.get('selectedMetrics_milestoneRun', 234)

        self.outputPath = kwargs.get('outputPath',
                               'D:/ac/PhD/Research/data/pd/01/all_metrics/')


        self.data = kwargs.get('data')
        self.doPreProcess = kwargs.get('preprocess', True)
        path = kwargs.get('path')
        if path != None:
            self.loadDataFrom(path)

    def loadDataFrom(self, path, verbose=True):
        self.data = read_csv(path)
        if verbose:
            print('{}: {}'.format(self.name, self.data.shape))
        if self.doPreProcess:
            self.preProcess()

    def tryAllMetricCombinations(self,shm_sampler, procstat, meminfo, milestoneRun,  motif_Tmin=None, motif_Tmax=None, Kmax=20, lamb=1e-1,features=[0], verbose=True):
        print("try_all_metric_combinations")

        for metricSet in self.ldmsInstance.metricSets:
            times = parallel_calc_breakpoints_for("{}{}_{}.pickle".format(
                self.outputPath , metricSet , self.KMaxMap[metricSet]), self.numThreads, self.ldmsInstance.metricSets[metricSet], self.KMaxMap[metricSet],
                ['Dirty', 'KernelStack', 'Writeback', 'SReclaimable', 'PageTables'], {}, motif_Tmin, motif_Tmax, lamb,
                verbose)

        # results = pool.starmap(calc_breakpoints_for, [(meminfo, 8, lamb, verbose), (shm_sampler, 20, lamb, verbose), (procstat, 3, lamb, verbose)])
        #
        # pool.close()
        # pool.join()
        #
        # meminfo_times_list = results[0]
        # shm_sampler_times_list = results[1]
        # procstat_times_list = results[2]

        meminfo_times_list = parallel_calc_breakpoints_for(
            'D:/ac/PhD/Research/data/pd/01/all_metrics/meminfo_26.pickle', 8, meminfo, 26,
            ['Dirty', 'KernelStack', 'Writeback', 'SReclaimable', 'PageTables'], {}, motif_Tmin, motif_Tmax, lamb,
            verbose)
        shm_sampler_times_list = parallel_calc_breakpoints_for(
            'D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler_234.pickle', 8, shm_sampler, 234,
            ['MPI_Allreduce.calls.0', 'MPI_Issend.calls.0', 'MPI_Ssend.calls.0', 'MPI_Irecv.calls.16',
             'MPI_Send.calls.16', 'MPI_Wtime.calls.0'], {}, motif_Tmin, motif_Tmax, lamb, verbose)
        procstat_times_list = parallel_calc_breakpoints_for(
            'D:/ac/PhD/Research/data/pd/01/all_metrics/procstat_26.pickle', 8, procstat, 26,
            ['per_core_iowait0', 'per_core_softirqd0', 'per_core_sys5', 'procs_blocked', 'procs_running'], {},
            motif_Tmin, motif_Tmax, lamb, verbose)

        print("All breakpoints have been found!")
        # pool.close()

        # meminfo_times_list = calc_breakpoints_for(meminfo, 8, lamb, verbose)
        # shm_sampler_times_list = calc_breakpoints_for(shm_sampler, 20, lamb, verbose)
        # procstat_times_list = calc_breakpoints_for(procstat, 3, lamb, verbose)

        # calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, lamb)

        parallel_calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler,
                                       procstat, meminfo, milestoneRun,
                                       procstat_selected_keys=['per_core_iowait0', 'per_core_softirqd0',
                                                               'per_core_sys5', 'procs_blocked', 'procs_running'],
                                       meminfo_selected_keys=['Dirty', 'KernelStack', 'Writeback', 'SReclaimable',
                                                              'PageTables'],
                                       shm_sampler_selected_keys=['MPI_Allreduce.calls.0', 'MPI_Issend.calls.0',
                                                                  'MPI_Ssend.calls.0', 'MPI_Irecv.calls.16',
                                                                  'MPI_Send.calls.16', 'MPI_Wtime.calls.0'])
        #     calc_score_store_plot_metric_list(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=['per_core_iowait0','per_core_softirqd0','per_core_sys5','procs_blocked','procs_running'],
        #                                    meminfo_selected_keys=['Dirty','KernelStack','Writeback','SReclaimable','PageTables'], shm_sampler_selected_keys=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'])

        print("Done")


def try_all_metric_combinations(shm_sampler, procstat, meminfo, milestoneRun,  motif_Tmin=None, motif_Tmax=None, Kmax=20, lamb=1e-1,features=[0], verbose=True):
    print("try_all_metric_combinations")


    # results = pool.starmap(calc_breakpoints_for, [(meminfo, 8, lamb, verbose), (shm_sampler, 20, lamb, verbose), (procstat, 3, lamb, verbose)])
    #
    # pool.close()
    # pool.join()
    #
    # meminfo_times_list = results[0]
    # shm_sampler_times_list = results[1]
    # procstat_times_list = results[2]

    meminfo_times_list = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/meminfo_26.pickle', 8,meminfo, 26, ['Dirty','KernelStack','Writeback','SReclaimable','PageTables'],{}, motif_Tmin, motif_Tmax, lamb, verbose)
    shm_sampler_times_list = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler_234.pickle', 8,shm_sampler, 234,['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'],{}, motif_Tmin, motif_Tmax, lamb, verbose)
    procstat_times_list = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/procstat_26.pickle', 8,procstat, 26, ['per_core_iowait0','per_core_softirqd0','per_core_sys5','procs_blocked','procs_running'], {}, motif_Tmin, motif_Tmax, lamb, verbose)

    print("All breakpoints have been found!")
    # pool.close()

    # meminfo_times_list = calc_breakpoints_for(meminfo, 8, lamb, verbose)
    # shm_sampler_times_list = calc_breakpoints_for(shm_sampler, 20, lamb, verbose)
    # procstat_times_list = calc_breakpoints_for(procstat, 3, lamb, verbose)

    # calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, lamb)

    parallel_calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=['per_core_iowait0','per_core_softirqd0','per_core_sys5','procs_blocked','procs_running'],
                                  meminfo_selected_keys=['Dirty','KernelStack','Writeback','SReclaimable','PageTables'], shm_sampler_selected_keys=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'])
#     calc_score_store_plot_metric_list(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=['per_core_iowait0','per_core_softirqd0','per_core_sys5','procs_blocked','procs_running'],
#                                    meminfo_selected_keys=['Dirty','KernelStack','Writeback','SReclaimable','PageTables'], shm_sampler_selected_keys=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'])

    print("Done")


def calc_shm_only(shm_sampler,milestoneRun, metrics=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'],motif_Tmin=None, motif_Tmax=None,Kmax=750, lamb=1e-1, verbose=False):
    metric_transform_map = {}
    # for m in metrics:
    #     metric_transform_map[m] = "rate"
    shm_sampler_times_list = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler_207_new.pickle', 8,shm_sampler, 207, ['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'],metric_transform_map,motif_Tmin, motif_Tmax, lamb, verbose)
    model_intervals = get_interval_tree_from_model(milestoneRun)
    ldms_time_data = [md.epoch2num(shm_sampler['#Time'])]
    for s in metrics:
        print(s)
        print(shm_sampler_times_list[s])
        ldms_data = [shm_sampler[s]]
        if(s in metric_transform_map):
            transformed_df, [transformed_metric_name] = lt.create_transform_event(shm_sampler, [s], [], True, False, False)
            print(transformed_metric_name)
            ldms_data = [transformed_df[transformed_metric_name]]
        times_list = [shm_sampler_times_list[s]]
        print(times_list)

        filterred_trees, ignored_trees = get_interval_trees_from_times_list(times_list, motif_Tmin, motif_Tmax)
        shared_intervals = find_shared_period(filterred_trees)
        print(len(shared_intervals))
        union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
            shared_intervals, model_intervals,
            sorted(set(ldms_time_data[0].tolist())))
        print([union_score, score2, coverage_shared_interval, coverage_model_interval])

        shared_intervals_remaining = find_shared_period(ignored_trees)
        print(len(shared_intervals_remaining))
        union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
            shared_intervals_remaining, model_intervals,
            sorted(set(ldms_time_data[0].tolist())))
        print([union_score, score2, coverage_shared_interval, coverage_model_interval])



        plot_all(shared_intervals, shared_intervals_remaining, milestoneRun, ldms_data, ldms_time_data, name= "234_rate_" + s,
             savePath='D:/ac/PhD/Research/data/pd/01/all_metrics/')


def calc_shm_multiVariate(shm_sampler,milestoneRun, metrics=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'],motif_Tmin=None, motif_Tmax=None,Kmax=750, lamb=1e-1, verbose=False):
    transform = "rate"

    metrics = ['MPI_Allreduce.calls.0', 'MPI_Issend.calls.0', 'MPI_Ssend.calls.0', 'MPI_Wtime.calls.0']#,
               #'MPI_Irecv.calls.16', 'MPI_Send.calls.16']

    shm_sampler_times_list = calcBreakpointsForMultipleColumns('D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler_234_MV_4_metrics_rate.pickle', shm_sampler, 234,
        metrics, transform, motif_Tmin, motif_Tmax, lamb, verbose)
    model_intervals = get_interval_tree_from_model(milestoneRun)
    ldms_time_data = [md.epoch2num(shm_sampler['#Time'])]

    ldms_data = []
    for m in metrics:
        ldms_data.append(shm_sampler[m])

    if (transform != None):
        transformed_df, [transformed_metric_name] = lt.create_transform_event(shm_sampler, metrics, [], True, False, False)
        print(transformed_metric_name)
        ldms_data = []
        for m in transformed_metric_name:
            ldms_data.append(shm_sampler[m])
    times_list = [shm_sampler_times_list]
    filterred_trees, ignored_trees = get_interval_trees_from_times_list(times_list, motif_Tmin, motif_Tmax)
    shared_intervals = find_shared_period(filterred_trees)

    union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
        shared_intervals, model_intervals,
        sorted(set(ldms_time_data[0].tolist())))
    print([union_score, score2, coverage_shared_interval, coverage_model_interval])

    shared_intervals_remaining = find_shared_period(ignored_trees)
    print(len(shared_intervals_remaining))
    union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
        shared_intervals_remaining, model_intervals,
        sorted(set(ldms_time_data[0].tolist())))
    print([union_score, score2, coverage_shared_interval, coverage_model_interval])

    plot_all(shared_intervals, shared_intervals_remaining, milestoneRun, ldms_data, ldms_time_data, name="234_MV_4metrics_rate",
             savePath='D:/ac/PhD/Research/data/pd/01/all_metrics/')



def try_largeKMax(shm_sampler, procstat, meminfo, milestoneRun,motif_Tmin=None, motif_Tmax=None, Kmax=20, lamb=1e-1,features=[0], verbose=True):
    print("try_largeKMax")

    meminfo_times_list, meminfo_times_list_remaining = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/meminfo.pickle', 8,meminfo, 8,None,{},motif_Tmin, motif_Tmax, lamb, verbose)
    shm_sampler_times_list, shm_sampler_times_list_remaining = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler_234_new2.pickle', 8,shm_sampler, 234, ['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'],{},motif_Tmin, motif_Tmax, lamb, verbose)
    procstat_times_list, procstat_times_list_remaining = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/procstat.pickle', 8,procstat, 20, None,{}, motif_Tmin, motif_Tmax, lamb, verbose)

    print("All breakpoints have been found!")

    parallel_calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=['per_core_iowait0','per_core_softirqd0','per_core_sys5','procs_blocked','procs_running'],
                                      meminfo_selected_keys=['Dirty','KernelStack','Writeback','SReclaimable','PageTables'], shm_sampler_selected_keys=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'])

    # calc_score_store_plot_metric_list(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=['per_core_iowait0','per_core_softirqd0','per_core_sys5','procs_blocked','procs_running'],
    #                                meminfo_selected_keys=['Dirty','KernelStack','Writeback','SReclaimable','PageTables'], shm_sampler_selected_keys=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'])

def calcbp_for(shm_sampler, metrics=['MPI_Allreduce.calls.0','MPI_Issend.calls.0','MPI_Ssend.calls.0','MPI_Irecv.calls.16','MPI_Send.calls.16','MPI_Wtime.calls.0'],Kmax=750, lamb=1e-1):
    for c in  metrics:
        # bps_this_sampler = findbp_plot(shm_sampler.T, Kmax, lamb, [c])
        print(c)
        lls = GGSCrossVal(shm_sampler.T, Kmax=25, lambList=[0.1, 1, 10], features=[c], verbose=False)
        print(lls)
        trainLikelihood = lls[0][1][0]
        testLikelihood = lls[0][1][1]
        plt.plot(range(25 + 1), testLikelihood)
        plt.plot(range(25 + 1), trainLikelihood)
        plt.legend(['Test LL', 'Train LL'], loc='best')
        plt.show()

def plotbp_for(meminfo, metrics=['Dirty','KernelStack','Writeback','SReclaimable'],Kmax=50, lamb=1e-1):
    for m in  metrics:
        print(m)
        bps_this_sampler = findbp_plot(meminfo.T, Kmax, lamb, [m])

def testSimilarityMethod(motif_Tmin=None, motif_Tmax=None,Kmax=233, lamb=1e-1, verbose=False, numThreads=8):
    metrics = ['MPI_Issend.calls.0', 'MPI_Allreduce.calls.0',  'MPI_Ssend.calls.0', 'MPI_Irecv.calls.16',
               'MPI_Send.calls.16', 'MPI_Wtime.calls.0']
    metric_transform_map = {}
    print("testSimilarityMethod")
    ldmsInstance = LDMSInstance()
    shm_sampler = ldmsInstance.getMetricSet("shm_sampler").getDataFrame()
    milestoneRun = ldmsInstance.getMetricSet("milestoneRun").getDataFrame()

    model_intervals = get_interval_tree_from_model(milestoneRun)

    model_intervals, ts_intervals = get_two_level_interval_tree_from_model(milestoneRun)

    Kmax=len(model_intervals) - 1
    shm_sampler_times_list = parallel_calc_breakpoints_for('D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler_207_new.pickle', numThreads,shm_sampler, Kmax, metrics,metric_transform_map,motif_Tmin, motif_Tmax, lamb, verbose)

    # print(tree1)
    # print(tree2)
    # print(len(tree1))
    # print(len(tree2))

    shm_sampler['#Time'] = shm_sampler['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    for c in metrics:
        print(c)
        times_list = [shm_sampler_times_list[c]]
        filterred_trees, ignored_trees = get_interval_trees_from_times_list(times_list, motif_Tmin, motif_Tmax)
        shared_intervals = find_shared_period(filterred_trees)
        if len(shared_intervals) != len(model_intervals):
            print("ignoring becuase of the different length {} vs {} ".format(len(shared_intervals), len(model_intervals)))
            continue
        calcIBSMDistance(shm_sampler['#Time'], model_intervals, shared_intervals)






def ldms_example():
    print('ldms_example')

    ldmsInstance = LDMSInstance()
    meminfo = ldmsInstance.getMetricSet("meminfo").getDataFrame()
    shm_sampler = ldmsInstance.getMetricSet("shm_sampler").getDataFrame()
    vmstat = ldmsInstance.getMetricSet("vmstat").getDataFrame()
    procstat = ldmsInstance.getMetricSet("procstat").getDataFrame()
    procnetdev = ldmsInstance.getMetricSet("procnetdev").getDataFrame()
    milestoneRun = ldmsInstance.getMetricSet("milestoneRun").getDataFrame()

    # try_meminfo_example(meminfo)
    # try_procstat_example(procstat)
    # try_all_samplers(shm_sampler, procstat, meminfo, milestoneRun)
    # try_all_metric_combinations(shm_sampler, procstat, meminfo, milestoneRun)
    # try_largeKMax(shm_sampler, procstat, meminfo, milestoneRun, motif_Tmin=timedelta(microseconds=500000), motif_Tmax=timedelta(microseconds=10000000))

    calc_shm_only(shm_sampler, milestoneRun, motif_Tmin=timedelta(microseconds=1000), motif_Tmax=timedelta(microseconds=100000000))

    # plotbp_for(procstat,['per_core_sys5'])

    # calc_shm_multiVariate(shm_sampler, milestoneRun, motif_Tmin=timedelta(microseconds=100000), motif_Tmax=timedelta(microseconds=10000000))

    # calcbp_for(shm_sampler)
    # try_shm_sampler_example(shm_sampler)



def try_this_sampler(this_sampler, Kmax=8, lamb=1e-1, features=[0]):
    print('try_this_sampler')
    # this_sampler = np.reshape(this_sampler[features], (this_sampler[features].shape[0], len(features)))
    this_sampler = this_sampler.T  # Convert to an n-by-T matrix
    bps_this_sampler = findbp_plot(this_sampler, Kmax, lamb, features)

    for k in range(0,Kmax + 1):
        print("k=" + str(k))
        print(bps_this_sampler[k])
        # print(bps_this_sampler)
        # Plot predicted Mean/Covs
        Plot_predicted_Mean_Covs(bps_this_sampler[k], this_sampler, features)

def try_meminfo_example(meminfo, Kmax=8, lamb=1e-1, column='Dirty'):
    print('try_meminfo_example')
    this_sampler = meminfo[column]  # procstat['procs_running'] #shm_sampler_group_df #meminfo['Dirty'] #shm_sampler #
    try_this_sampler(this_sampler, Kmax, lamb)

def try_procstat_example(procstat, Kmax=3, lamb=1e-1, column='procs_running'):
    print('try_procstat_example')
    this_sampler = procstat[column]  # procstat['procs_running'] #shm_sampler_group_df #meminfo['Dirty'] #shm_sampler #
    try_this_sampler(this_sampler, Kmax, lamb)

def try_shm_sampler_example(shm_sampler, Kmax=20, lamb=1e-1, column='MPI_Issend'):
    print('try_shm_sampler_example')

    rank_based_events = create_rank_based_events(mpi_events=['MPI_Issend'])
    this_sampler = shm_sampler#[rank_based_events[0]]  # procstat['procs_running'] #shm_sampler_group_df #meminfo['Dirty'] #shm_sampler #
    try_this_sampler(this_sampler, Kmax, lamb, rank_based_events)

    shm_sampler_group_df, [gr1] = create_shm_event_mpi_issend_goup_df(shm_sampler)
    shm_sampler_group_df.fillna(0,inplace=True)
    rank_based_events = create_rank_based_events(mpi_events=['MPI_Issend'], prefix='d_', postfix='_dt')
    this_sampler = shm_sampler_group_df#['d_' + column + '_dt']
    try_this_sampler(this_sampler, Kmax, lamb,rank_based_events)



def try_all_samplers(shm_sampler, procstat, meminfo, milestoneRun, Kmax=20, lamb=1e-1,features=[0]):
    print("try_all_samplers")
    times_list = []
    ldms_data = [shm_sampler['MPI_Issend.calls.4'], procstat['procs_running'], meminfo['Dirty']]
    ldms_time_data = [md.epoch2num(shm_sampler['#Time']),md.epoch2num(procstat['#Time']),md.epoch2num(meminfo['#Time'])]

    this_sampler = meminfo.T  # Convert to an n-by-T matrix
    bps_this_sampler = findbp_plot(this_sampler, 8, lamb, ['Dirty'])
    times_list.append(convert_break_to_time(this_sampler.T, bps_this_sampler[8]))

    this_sampler = procstat.T  # Convert to an n-by-T matrix
    bps_this_sampler = findbp_plot(this_sampler, 3, lamb, ['procs_running'])
    times_list.append(convert_break_to_time(this_sampler.T, bps_this_sampler[3]))

    rank_based_events = create_rank_based_events(mpi_events=['MPI_Issend'])
    this_sampler = shm_sampler.T  # Convert to an n-by-T matrix
    bps_this_sampler = findbp_plot(this_sampler, Kmax, lamb, rank_based_events)
    times_list.append(convert_break_to_time(this_sampler.T, bps_this_sampler[Kmax]))

    shared_intervals = find_shared_period(times_list, bps_this_sampler[Kmax])

    model_intervals = get_interval_tree_from_model(milestoneRun)


    plot_all(shared_intervals, milestoneRun, ldms_data, ldms_time_data, name="without_rate")


    shm_sampler_group_df, [gr1] = create_shm_event_mpi_issend_goup_df(shm_sampler)
    shm_sampler_group_df.fillna(0,inplace=True)
    rank_based_events = create_rank_based_events(mpi_events=['MPI_Issend'], prefix='d_', postfix='_dt')
    this_sampler = shm_sampler_group_df.T#['d_' + column + '_dt']
    bps_this_sampler = findbp_plot(this_sampler, Kmax, lamb, rank_based_events)
    times_list.append(convert_break_to_time(this_sampler.T, bps_this_sampler[Kmax]))
    ldms_data.append(shm_sampler_group_df['d_MPI_Issend.calls.4_dt'])
    ldms_time_data.append(md.epoch2num(shm_sampler_group_df['#Time']))

    shared_intervals = find_shared_period(times_list, bps_this_sampler[Kmax])


    union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(shared_intervals, model_intervals,
                                                    sorted(set(ldms_time_data[2].tolist()) | set(ldms_time_data[1].tolist()) | set(ldms_time_data[0].tolist()) | set(ldms_time_data[3].tolist())))
    print("similarity score for the found "  + str(len(shared_intervals)) + " intervals and " + str(len(model_intervals)) + ", union_score= " + str(union_score) + " coverage_shared_interval, coverage_model_interval= " + str( coverage_shared_interval) + " " + str(coverage_model_interval))

    plot_all(shared_intervals, milestoneRun, ldms_data, ldms_time_data, name="with_rate")

def create_shm_event_mpi_issend_goup_df(shm_sampler_df, ranks=range(0,30), mpi_events = ['MPI_Issend'], ratetransform = True, sumtransform= False, logtransform = False):
    return lt.create_transform_event(shm_sampler_df, create_rank_based_events(ranks, mpi_events), mpi_events, ratetransform, sumtransform, logtransform)

def create_rank_based_events(ranks=range(4,5), mpi_events = ['MPI_Issend'], prefix='', postfix=''):
    gr = []
    for r in ranks:
        for i, item in enumerate(mpi_events):
            gr.append(prefix + item + '.calls.' + str(r) + postfix)
    return gr

def findbp_plot(data, Kmax=10, lamb=1e-1, features = [], verbose = False):
    print("findbp_plot")
    # print(data[features])
    # Find up to 10 breakpoints at lambda = 1e-1
    bps, objectives = findBP(data, Kmax, lamb, features, verbose)
    print("found " + str(len(bps)) + " breakpoints")
    print(objectives)
    plotBP(objectives)
    return bps

def authors_example(num_samples_per_segment = 1000, Kmax=10, lamb=1e-1):
    print('authors_example')
    np.random.seed(0)
    # Generate synthetic 1D data
    # First 1000 samples: mean = 1, SD = 1
    # Second 1000 samples: mean = 0, SD = 0.1
    # Third 1000 samples: mean = 0, SD = 1
    d1 = np.random.normal(1, 1, num_samples_per_segment)
    d2 = np.random.normal(0, 0.5, num_samples_per_segment)
    d3 = np.random.normal(-1, 1, num_samples_per_segment)

    data = np.concatenate((d1, d2, d3))

    data = np.reshape(data, (num_samples_per_segment * 3, 1))

    data = data.T  # Convert to an n-by-T matrix
    bps = findbp_plot(data, Kmax, lamb)
    # Plot predicted Mean/Covs
    Plot_predicted_Mean_Covs(bps[2], data)





if __name__ == '__main__':
    pool = ThreadPool(8)
    xAxis = '#Time'
    value_name = 'value'
    # authors_example()
    ldms_example()
    # testSimilarityMethod()
