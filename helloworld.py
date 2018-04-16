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
        ret_group.append('d_' + group[i] + '_dt')
    return df, ret_group

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

def create_metric_groups(df, metric_groups):
    mem_dfs = []
    for i, mg in enumerate(metric_groups):
        mem_dfs.append(concat_metrics(df, mg))
    return mem_dfs

def create_shm_event_mpi_issend_goup_df(shm_sampler_df, ranks=[0,1,2,3,4,5,6,7], ratetransform = True, logtransform = False):
    all = ['MPI_Issend']


    gr1 = []


    for r in ranks:
        for i, item in enumerate(all):
            if i == 0:
                gr1.append(item + '.calls.' + str(r))
    if(ratetransform):
        shm_sampler_df, [gr1] = do_rate_transform(shm_sampler_df, [gr1])

    if(logtransform):
        shm_sampler_df, [gr1] = do_log_transform(shm_sampler_df, [gr1])

    return create_metric_groups(shm_sampler_df, [gr1]), [gr1]

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

xAxis='#Time'
value_name = 'value'

[meminfo, shm_sampler, vmstat, procstat, procnetdev] = pre_process(read_data())

shm_sampler_group_df, [gr1] = create_shm_event_mpi_issend_goup_df(shm_sampler)
grs = {}
grs[0] = gr1



print(type(shm_sampler_group_df))
print(shm_sampler_group_df)
this_sampler = procstat['procs_running'] #shm_sampler_group_df #meminfo['Dirty'] #shm_sampler #
print(type(this_sampler))

print(this_sampler.shape)

np.random.seed(0)
# Generate synthetic 1D data
# First 1000 samples: mean = 1, SD = 1
# Second 1000 samples: mean = 0, SD = 0.1
# Third 1000 samples: mean = 0, SD = 1
d1 = np.random.normal(1,1,1000)
d2 = np.random.normal(0,0.5,1000)
d3 = np.random.normal(-1,1,1000)

data = np.concatenate((d1,d2,d3))

data = np.reshape(data, (3000,1))
this_sampler = np.reshape(this_sampler, (4947,1))
print(this_sampler.shape)
data = data.T #Convert to an n-by-T matrix
this_sampler = this_sampler.T #Convert to an n-by-T matrix
print(this_sampler)
print(this_sampler[this_sampler == 0])
# null_columns=this_sampler.columns[this_sampler.le(0).any()]

# print("here")
# print(this_sampler[null_columns].isnull().sum())


def findbp_plot(data, Kmax=10, lamb=1e-1):
    # Find up to 10 breakpoints at lambda = 1e-1
    bps, objectives = GGS(data, Kmax, lamb)

    print(bps)
    print(objectives)

    # Plot objective vs. number of breakpoints. Note that the objective essentially
    # stops increasing after K = 2, since there are only 2 "true" breakpoints
    plotVals = range(len(objectives))
    plt.plot(plotVals, objectives, 'or-')
    plt.xlabel('Number of Breakpoints')
    plt.ylabel(r'$\phi(b)$')
    plt.show()
    return bps


def Plot_predicted_Mean_Covs(breaks, data):

    mcs = GGSMeanCov(data, breaks, 1e-1)
    predicted = []
    varPlus = []
    varMinus = []
    for i in range(len(mcs)):
        for j in range(breaks[i + 1] - breaks[i]):
            predicted.append(mcs[i][0])  # Estimate the mean
            varPlus.append(mcs[i][0] + math.sqrt(mcs[i][1][0]))  # One standard deviation above the mean
            varMinus.append(mcs[i][0] - math.sqrt(mcs[i][1][0]))  # One s.d. below

    f, axarr = plt.subplots(2, sharex=True)
    # print(data.shape)
    # print(type(data))
    # print(data)
    # print(data[0])
    # print(predicted)

    axarr[0].plot(data[0])
    axarr[0].set_ylim([-1, 50])
    axarr[0].set_ylabel('Actual Data')
    axarr[1].plot(predicted)
    axarr[1].plot(varPlus, 'r--')
    axarr[1].plot(varMinus, 'r--')
    axarr[1].set_ylim([-70, 70])
    axarr[1].set_ylabel('Predicted mean (+/- 1 S.D.)')
    plt.show()

bps = findbp_plot(data, Kmax=10, lamb=1e-1)

bps_this_sampler = findbp_plot(this_sampler, Kmax = 3, lamb = 1e-1)



#Plot predicted Mean/Covs
Plot_predicted_Mean_Covs(bps[2], data)



#Plot predicted Mean/Covs
Plot_predicted_Mean_Covs(bps_this_sampler[0], this_sampler)
Plot_predicted_Mean_Covs(bps_this_sampler[1], this_sampler)
Plot_predicted_Mean_Covs(bps_this_sampler[2], this_sampler)
Plot_predicted_Mean_Covs(bps_this_sampler[3], this_sampler)
