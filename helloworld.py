from ggs import *
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib.dates as md


def concat_metrics(src_df, metric_list):
    final_df = pd.DataFrame()
    for m in metric_list:
        dest_df = src_df.loc[:,(xAxis, m)]
        dest_df.rename(columns={m: value_name}, inplace=True)
        dest_df.loc[:, 'metric'] = m
        final_df = pd.concat([final_df, dest_df])
    # final_df[xAxis] = md.epoch2num(final_df[xAxis])
    final_df[xAxis] = final_df[xAxis] - final_df[xAxis].min()
    final_df[xAxis] = md.epoch2num(final_df[xAxis])
    final_df.loc[:, 'Dummy'] = 0
    return final_df


def calcRate(df, metric):
    df['d_' + metric + '_dt'] = (df[metric] - df[metric].shift()) / (df[xAxis]-df[xAxis].shift())
    return df

def calcLog(df, metric):
    df['log_' + metric] = df[metric].apply(lambda x: np.log(x))
    return df

def processRatesForEventGroup(df, group):
    ret_group = []
    for i, item in enumerate(group):
        df = calcRate(df, group[i])
        print('d_' + group[i] + '_dt')
        ret_group.append('d_' + group[i] + '_dt')
    return df, ret_group

def processSumsForEventGroup(df, group, group_name):
    ret_group = ['sum_' + group_name]
    ret_df = None
    for i, item in enumerate(group):
        if(ret_df == None):
            ret_df = df[group[i]]
            ret_df['sum_' + group_name] = df[group[i]]
            ret_df.drop(group[i])
        else:
            ret_df = ret_df + df[group[i]]

    return ret_df, ret_group

def processLogsForEventGroup(df, group):
    ret_group = []
    for i, item in enumerate(group):
        df = calcLog(df, group[i])
        ret_group.append('log_' + group[i])
    return df, ret_group

def do_log_transform(df, metric_groups):
    for i, mg in enumerate(metric_groups):
        df, metric_groups[i] = processLogsForEventGroup(df, mg)
    return df, metric_groups


def do_rate_transform(df, metric_groups):
    for i, mg in enumerate(metric_groups):
        df, metric_groups[i] = processRatesForEventGroup(df, mg)
    return df, metric_groups

def do_sum_transform(df, metric_groups, group_names):
    for i, mg in enumerate(metric_groups):
        df, metric_groups[i] = processSumsForEventGroup(df, mg, group_names[i])
    return df, metric_groups

def create_metric_groups(df, metric_groups):
    mem_dfs = []
    for i, mg in enumerate(metric_groups):
        mem_dfs.append(concat_metrics(df, mg))
    return mem_dfs

def create_rank_based_events(ranks=range(4,5), mpi_events = ['MPI_Issend'], prefix='', postfix=''):
    gr = []
    for r in ranks:
        for i, item in enumerate(mpi_events):
            gr.append(prefix + item + '.calls.' + str(r) + postfix)
    return gr

def create_shm_event_mpi_issend_goup_df(shm_sampler_df, ranks=range(0,30), mpi_events = ['MPI_Issend'], ratetransform = True, sumtransform= False, logtransform = False):
    gr1 = create_rank_based_events(ranks, mpi_events)
    if(sumtransform):
        shm_sampler_df, [gr1]= do_sum_transform(shm_sampler_df, [gr1], [mpi_events]);

    if(ratetransform):
        shm_sampler_df, [gr1] = do_rate_transform(shm_sampler_df, [gr1])

    if(logtransform):
        shm_sampler_df, [gr1] = do_log_transform(shm_sampler_df, [gr1])

    # return create_metric_groups(shm_sampler_df, [gr1]), [gr1]
    return shm_sampler_df, [gr1]

def pre_process(all_dfs):
    result_dfs = []
    for df in all_dfs:
        df.loc[:, 'Dummy'] = 0
        # df['#Time'] = df['#Time'].apply(lambda x: datetime.fromtimestamp(x))

        df.index = pd.DatetimeIndex(df['#Time'])
        result_dfs.append(df)


    return result_dfs

def read_data(path = 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/', datasets=['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev']):
    print('reading data')
    all_dfs = []
    for ds in datasets:
        print(ds)
        ds_path = path + ds + '.csv'
        ds_df = pd.read_csv(ds_path)
        print(ds_df.shape)
        all_dfs.append(ds_df)
    return all_dfs

def findbp_plot(data, Kmax=10, lamb=1e-1, features = [], verbose = False):
    print("finding breakpoints")
    # Find up to 10 breakpoints at lambda = 1e-1
    bps, objectives = GGS(data, Kmax, lamb, features, verbose)

    # Plot objective vs. number of breakpoints. Note that the objective essentially
    # stops increasing after K = 2, since there are only 2 "true" breakpoints
    plotVals = range(len(objectives))
    plt.plot(plotVals, objectives, 'or-')
    plt.xlabel('Number of Breakpoints')
    plt.ylabel(r'$\phi(b)$')
    plt.show()
    return bps


def Plot_predicted_Mean_Covs(breaks, data,features = [], verbose = False):
    print("plotting mean covs")
    mcs = GGSMeanCov(data, breaks, 1e-1, features, verbose)
    predicted = []
    varPlus = []
    varMinus = []
    max_var = 0
    min_var = 0
    print(data.T[features].shape)
    max_data = data.T[features].max()[0]
    min_data = data.T[features].min()[0]
    for i in range(len(mcs)):
        for j in range(breaks[i + 1] - breaks[i]):
            max_var = max(math.sqrt(mcs[i][1][0]) + mcs[i][0], max_var)
            min_var = min(mcs[i][0] - math.sqrt(mcs[i][1][0]), min_var)
            predicted.append(mcs[i][0])  # Estimate the mean
            varPlus.append(mcs[i][0] + math.sqrt(mcs[i][1][0]))  # One standard deviation above the mean
            varMinus.append(mcs[i][0] - math.sqrt(mcs[i][1][0]))  # One s.d. below

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(data.T[features].reset_index(drop = True))
    axarr[0].set_ylim([min_data + 0.1 * min_data, max_data + 0.1 * max_data])
    axarr[0].set_ylabel('Actual Data')



    axarr[1].plot(predicted)
    axarr[1].plot(varPlus, 'r--')
    axarr[1].plot(varMinus, 'r--')
    axarr[1].set_ylim([min_var + 0.1 * min_var, max_var + 0.1 * max_var])
    axarr[1].set_ylabel('Predicted mean (+/- 1 S.D.)')

    plt.show()

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

def try_this_sampler(this_sampler, Kmax=8, lamb=1e-1, features=[0]):
    print('try_this_sampler')
    print(this_sampler.shape)
    # this_sampler = np.reshape(this_sampler[features], (this_sampler[features].shape[0], len(features)))
    print(this_sampler.shape)
    this_sampler = this_sampler.T  # Convert to an n-by-T matrix
    print(this_sampler.shape)
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
    print(rank_based_events)
    this_sampler = shm_sampler#[rank_based_events[0]]  # procstat['procs_running'] #shm_sampler_group_df #meminfo['Dirty'] #shm_sampler #
    print(this_sampler.shape)
    try_this_sampler(this_sampler, Kmax, lamb, rank_based_events)

    shm_sampler_group_df, [gr1] = create_shm_event_mpi_issend_goup_df(shm_sampler)

    # grs = {}
    # grs[0] = gr1
    # this_sampler = shm_sampler[column]  # procstat['procs_running'] #shm_sampler_group_df #meminfo['Dirty'] #shm_sampler #
    shm_sampler_group_df.fillna(0,inplace=True)
    rank_based_events = create_rank_based_events(mpi_events=['MPI_Issend'], prefix='d_', postfix='_dt')
    this_sampler = shm_sampler_group_df#['d_' + column + '_dt']
    try_this_sampler(this_sampler, Kmax, lamb,rank_based_events)


def ldms_example():
    print('ldms_example')
    [meminfo, shm_sampler, vmstat, procstat, procnetdev] = pre_process(read_data())

    # try_meminfo_example(meminfo)
    # try_procstat_example(procstat)
    try_shm_sampler_example(shm_sampler)



xAxis = '#Time'
value_name = 'value'
# authors_example()
ldms_example()


