
from ggs import *
import numpy as np
import pickle
import itertools
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib.dates as md
import matplotlib.patches as patches
import gc
from intervaltree import Interval, IntervalTree
from datetime import datetime, timedelta

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

def read_data(path = 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/', datasets=['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'milestoneRun']):
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
    print("found " + str(len(bps)) + " breakpoints")

    # Plot objective vs. number of breakpoints. Note that the objective essentially
    # stops increasing after K = 2, since there are only 2 "true" breakpoints
    # plotVals = range(len(objectives))
    # plt.plot(plotVals, objectives, 'or-')
    # plt.xlabel('Number of Breakpoints')
    # plt.ylabel(r'$\phi(b)$')
    # plt.show()
    return bps

def convert_break_to_time(data, breaks):
    time = []
    for b in breaks:
        time.append(data.iloc[min(b,data.shape[0] - 1), data.columns.get_loc('#Time')])
    return time

def filter_motifs(time_list, breaks, motif_Tmin=None, motif_Tmax=None, verbose=True):

    start = 0
    stop = start + 1
    filtered_list = []
    filtered_break_list = []

    if(motif_Tmin is None and motif_Tmax is None):
        return breaks,time_list
    filtered_list.append(time_list[start])
    filtered_break_list.append(breaks[start])
    while(stop < len(time_list)):
        if(motif_Tmin is not None):
            while(stop < len(time_list) and time_list[stop] - time_list[start] < motif_Tmin):
                if(verbose):
                    print("ignoring (" + str(time_list[start]) + "," + str(time_list[stop]) + ") because its length is shorter than the defined Tmin: " + str(motif_Tmin))
                stop = stop + 1

        if (motif_Tmax is not None):
            while (stop < len(time_list) and time_list[stop] - time_list[start] > motif_Tmax):
                if(verbose):
                    print("ignoring (" + str(time_list[start]) + "," + str(time_list[stop]) + ") because its length is longer than the defined Tmax: " + str(motif_Tmax))
                stop = stop + 1

        if(stop < len(time_list)):
            filtered_list.append(time_list[stop])
            filtered_break_list.append(breaks[stop])

        start = stop
        stop = start + 1

    return filtered_break_list, filtered_list


def find_shared_period(times_list, breaks_list, epsilon_p=timedelta(microseconds=1)):
    print("finding shared region")


    # for times in times_list:
    #     for start in range(0, len(times) - 1):
    #         stop = start + 1


    trees = []
    for times in times_list:
        current_tree = IntervalTree()
        for start in range(0, len(times) - 1):
            stop = start + 1
            current_tree.add(Interval(datetime.utcfromtimestamp(times[start]), datetime.utcfromtimestamp(times[stop])))
        trees.append(current_tree)
    result = trees[0]


    for t in range(1,len(trees) - 1):
        search_result = set()
        for interval_obj in trees[t+1]:

            # print("searching for")
            # print(interval_obj)
            # print("in")
            # print(result)

            # initial value for result=trees[0]
            # search for each member of trees[1] in result
            # for  each search result do a max on the beginning of the searched range and the first member of the search result and do max for end of those ranges

            # print("found")
            current_search_result = result.search(Interval(interval_obj.begin - epsilon_p, interval_obj.end + epsilon_p))
            # print(current_search_result)
            if(len(current_search_result) > 0):
                first = list(sorted(current_search_result))[0]
                if(first.begin < interval_obj.begin - epsilon_p):
                    current_search_result.remove(first)
                    current_search_result.add(Interval(interval_obj.begin - epsilon_p, first.end))

                last = list(sorted(current_search_result))[len(current_search_result) - 1]
                if(last.end > interval_obj.end + epsilon_p):
                    current_search_result.remove(last)
                    current_search_result.add(Interval(last.begin, interval_obj.end + epsilon_p))
            search_result.update(current_search_result)
            # print("revised")
            # print(search_result)
        result = IntervalTree(sorted(search_result))
    return sorted(set(result))

def plot_interval(intervals, ax,min_start, max_end, fill=True, patterns = ['-', '+', 'x', 'o', 'O', '.', '*'], colors= ["red", "blue","green","yellow","purple","cyan","white"],separateTSWithPattern=True):
    print("plotting intervals")
    y0 = 0
    height = 1
    tsIndex = 0
    data_counters = {}
    for index, interval in enumerate(intervals):
        begin = md.date2num(interval.begin)
        end = md.date2num(interval.end)

        if(interval.data != None):
            if interval.data in data_counters:
                data_counters[interval.data] = data_counters[interval.data] + 1
            else:
                data_counters[interval.data] = 1
            if(separateTSWithPattern == True):
                pattern = patterns[tsIndex % len(patterns)]
                color = colors[index % len(colors)]
            else:
                color = colors[tsIndex % len(colors)]
                pattern = patterns[index % len(patterns)]
            if interval.data == 'Timestep':
                tsIndex = tsIndex + 1
        else:
            color = colors[index % len(colors)]
            pattern = patterns[index % len(patterns)]


        p = patches.Rectangle(
            (begin, y0),
            end - begin,
            height,
            hatch=pattern,
            facecolor=color,
            fill=fill
        )
        ax.add_patch(p)
    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)
    ax.xaxis.set_major_locator(md.SecondLocator(interval=60))
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlim(min_start, max_end)
    ax.set_ylim(y0, y0+height)


def calc_interval_counts(intervals, datapoints):
    print("calc_interval_counts")
    interval_counter = []
    numOfDataPoints = len(datapoints)
    dpIndex = 0
    last_dp_index = 0
    for interval in intervals:
        current_counter = 0
        while(dpIndex < numOfDataPoints):
            dp = datapoints[dpIndex]
            # if(len(datapoints) == 5645 and len(intervals) == 234):# or len(datapoints) == 5339 or len(datapoints) == 4947):
            #     print(str(md.num2date(dp)) + " cmp " + str(interval))

            if dp >= md.date2num(interval.begin):
                if dp <= md.date2num(interval.end):
                    current_counter = current_counter + 1
                    dpIndex = dpIndex + 1
                else:
                    break
            else:
                if dp <= md.date2num(interval.end):
                    dpIndex = dpIndex + 1
                else:
                    break
        interval_counter.append(current_counter)
    return interval_counter

def calc_intervals_similarity_score(intervals1, intervals2, datapoints):

    print("calc_intervals_similarity_score using " + str(len(datapoints)) + " datapoints")
    interval_counter1 = calc_interval_counts(intervals1, datapoints)
    interval_counter2 = calc_interval_counts(intervals2, datapoints)

    score = 0
    small_interval_counter = interval_counter2
    large_interval_counter = interval_counter1
    if(len(interval_counter1) < len(interval_counter2)):
        small_interval_counter = interval_counter1
        large_interval_counter = interval_counter2

    small_interval_counter_len = len(small_interval_counter)
    large_interval_counter_len = len(large_interval_counter)

    interval_counter_index = 0

    while interval_counter_index < small_interval_counter_len:
        score = score + abs(small_interval_counter[interval_counter_index] - large_interval_counter[interval_counter_index])
        interval_counter_index = interval_counter_index + 1
    print("score = " + str(score) + " after comparing " + str(small_interval_counter_len) + " intervals")
    while interval_counter_index < large_interval_counter_len:
        score = score + large_interval_counter[interval_counter_index]
        interval_counter_index = interval_counter_index + 1
    print("score = " + str(score) + " after comparing " + str(large_interval_counter_len) + " intervals")

    return score, sum(interval_counter1) / len(datapoints), sum(interval_counter2) / len(datapoints)


def get_interval_tree_from_model(model):
    current_tree = IntervalTree()
    for i in range(0,model.shape[0] - 1):
        start = model.iloc[i]['#Time']
        stop = model.iloc[i+1]['#Time']
        if start > stop:
            start = stop -  (start-stop)
            # stop = start + timedelta(microseconds=1)

        current_tree.add(Interval(datetime.utcfromtimestamp(start), datetime.utcfromtimestamp(stop), model.iloc[i+1]['metric']))
    return sorted(set(current_tree))

def plot_model(model, ax,min_start, max_end, fill=True, patterns = ['-', '+', 'x', 'o', 'O', '.', '*'], colors= ["red", "blue","green","yellow","purple","cyan","white"]):
    print("plotting model")
    plot_interval(get_interval_tree_from_model(model), ax, min_start, max_end, fill, patterns, colors)

def plot_all(intervals, milestoneRun,  ldms_data, ldms_time_data, name='all_in_one', savePath='D:/ac/PhD/Research/data/pd/01/', format='.pdf'):
    print("plotting all (" + name + ")")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    number_of_plots = 2 + len(ldms_data)
    fig, axs_ret = plt.subplots(nrows=number_of_plots)
    axs = {}
    if number_of_plots == 1:
        axs[0] = axs_ret
    else:
        axs = axs_ret


    begin = md.date2num(intervals[0].begin)
    end = md.date2num(intervals[len(intervals) - 1].end)

    x_min = begin + begin - md.date2num(intervals[0].end)
    x_max = end + end - md.date2num(intervals[len(intervals) - 1].begin)

    for index, data in enumerate(ldms_data):
        if(ldms_time_data[index].min() < x_min):
            x_min = ldms_time_data[index].min()
        if(ldms_time_data[index].max() > x_max):
            x_max = ldms_time_data[index].max()

    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)
    plot_model(milestoneRun, axs[0], x_min, x_max)
    for index, data in enumerate(ldms_data):
        axs[index+1].plot(ldms_time_data[index],data)
        axs[index+1].xaxis.set_major_locator(md.SecondLocator(interval=60))
        axs[index+1].xaxis.set_major_formatter(xfmt)
        axs[index+1].set_xlim(x_min, x_max)
    plot_interval(intervals, axs[number_of_plots - 1], x_min, x_max)

    fig.set_size_inches(h=18.5, w=20)
    fig.autofmt_xdate()
    print('saving figure')
    fig.savefig(savePath + name + format, dpi=2400)

    fig.clf()
    plt.close()
    gc.collect()

def check_motif_criteria(data, breaks, motif_Tmin=30, motif_Tmax=300, verbose=True):
    print("checking motif criteria")
    time_list = convert_break_to_time(data, breaks)

    if(len(time_list) > 2):
        find_shared_period([(time_list[0],time_list[2]),(time_list[1],time_list[2])], breaks)

    breaks, time_list  = filter_motifs(time_list, breaks, motif_Tmin, motif_Tmax, verbose)

    if (verbose):
        for start in range(0,len(time_list) - 1):
            stop = start + 1
            print("(" + str(time_list[start]) + "," + str(time_list[stop]) + ")[" + str(breaks[start]) + "," + str(breaks[stop]) +  "] = " + str(time_list[stop] - time_list[start]))

    return breaks



def Plot_predicted_Mean_Covs(breaks, data,features = [], motif_Tmin=30, motif_Tmax=300, verbose = True):
    print("plotting mean covs")
    breaks = check_motif_criteria(data.T, breaks,motif_Tmin, motif_Tmax, verbose)
    if len(breaks) <= 1:
        return

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


    union_score, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(shared_intervals, model_intervals,
                                                    sorted(set(ldms_time_data[2].tolist()) | set(ldms_time_data[1].tolist()) | set(ldms_time_data[0].tolist()) | set(ldms_time_data[3].tolist())))
    print("similarity score for the found "  + str(len(shared_intervals)) + " intervals and " + str(len(model_intervals)) + ", union_score= " + str(union_score) + " coverage_shared_interval, coverage_model_interval= " + str( coverage_shared_interval) + " " + str(coverage_model_interval))

    plot_all(shared_intervals, milestoneRun, ldms_data, ldms_time_data, name="with_rate")

def mygrouper(n, iterable):
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.zip_longest(*args))

def process_column(c, df, this_sampler, Kmax, lamb, counter = 0, verbose=True):
    times_list = []
    if c in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id', 'MemTotal', 'MemFree', 'MemAvailable',
             'Cached']:
        return times_list
    if verbose:
        print(str(counter) + '/' + str(len(df.columns)) + ' c:' + c)
    try:
        bps_this_sampler = findbp_plot(this_sampler, Kmax, lamb, [c])
    except np.linalg.linalg.LinAlgError:
        return times_list
    else:
        print(len(bps_this_sampler))
        print(bps_this_sampler)
        if (type(bps_this_sampler[0]) is int):
            if (len(bps_this_sampler) > 2):
                times_list = convert_break_to_time(this_sampler.T, bps_this_sampler)
        else:
            times_list = convert_break_to_time(this_sampler.T, bps_this_sampler[len(bps_this_sampler) - 1])
    return times_list

def parallel_calc_breakpoints_for(numThreads,df, Kmax, lamb=1e-1, verbose=True):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    columnList = mygrouper(numThreads, df.columns)
    counter = 0
    times_list = {}
    this_sampler = df.T  # Convert to an n-by-T matrix
    for columns in columnList:
        if(len(columns) == numThreads):
            args = []
            for c in columns:
                counter = counter + 1
                args.append((c, df, this_sampler, Kmax, lamb, counter, verbose))

            results = pool.starmap(process_column, args)

            for index, a in enumerate(args):
                if(len(results[index]) > 0):
                    times_list[a[0]] = results[index]
        else:
            for c in columns:
                counter = counter + 1
                result = process_column(c, df, this_sampler, Kmax, lamb, counter, verbose)
                if(len(result) > 0):
                    times_list[c] = result
    pool.close()
    pool.join()
    return times_list

def calc_breakpoints_for(df, Kmax, lamb=1e-1, verbose=True):
    times_list = {}
    this_sampler = df.T  # Convert to an n-by-T matrix
    counter = 0
    for c in df.columns:
        counter = counter + 1
        if c in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id', 'MemTotal', 'MemFree', 'MemAvailable','Cached']:
            continue
        if verbose:
            print(str(counter) + '/' + str(len(df.columns)) + ' c:' + c)
        try:
            bps_this_sampler = findbp_plot(this_sampler, Kmax, lamb, [c])
        except np.linalg.linalg.LinAlgError:
            continue
        else:
            print(len(bps_this_sampler))
            print(bps_this_sampler)
            if(type(bps_this_sampler[0]) is int):
                if (len(bps_this_sampler) > 2):
                    times_list[c] = convert_break_to_time(this_sampler.T, bps_this_sampler)
            else:
                times_list[c] = convert_break_to_time(this_sampler.T, bps_this_sampler[len(bps_this_sampler) - 1])


    return  times_list


def try_all_metric_combinations(shm_sampler, procstat, meminfo, milestoneRun, Kmax=20, lamb=1e-1,features=[0], verbose=True):
    print("try_all_metric_combinations")
    model_intervals = get_interval_tree_from_model(milestoneRun)
    ldms_time_data = [md.epoch2num(shm_sampler['#Time']), md.epoch2num(procstat['#Time']),
                      md.epoch2num(meminfo['#Time'])]

    # results = pool.starmap(calc_breakpoints_for, [(meminfo, 8, lamb, verbose), (shm_sampler, 20, lamb, verbose), (procstat, 3, lamb, verbose)])
    #
    # pool.close()
    # pool.join()
    #
    # meminfo_times_list = results[0]
    # shm_sampler_times_list = results[1]
    # procstat_times_list = results[2]

    meminfo_times_list = parallel_calc_breakpoints_for(8,meminfo, 8, lamb, verbose)
    shm_sampler_times_list = parallel_calc_breakpoints_for(8,shm_sampler, 20, lamb, verbose)
    procstat_times_list = parallel_calc_breakpoints_for(8,procstat, 20, lamb, verbose)

    with open('D:/ac/PhD/Research/data/pd/01/all_metrics/meminfo.pickle', 'wb') as handle:
        pickle.dump(meminfo_times_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('D:/ac/PhD/Research/data/pd/01/all_metrics/shm_sampler.pickle', 'wb') as handle:
        pickle.dump(shm_sampler_times_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('D:/ac/PhD/Research/data/pd/01/all_metrics/procstat.pickle', 'wb') as handle:
        pickle.dump(procstat_times_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("All breakpoints have been found!")
    # pool.close()

    # meminfo_times_list = calc_breakpoints_for(meminfo, 8, lamb, verbose)
    # shm_sampler_times_list = calc_breakpoints_for(shm_sampler, 20, lamb, verbose)
    # procstat_times_list = calc_breakpoints_for(procstat, 3, lamb, verbose)

    df_result = pd.DataFrame(columns = ['procstat','meminfo','shm_sampler','KMax','lamb','union_score', 'coverage_shared_interval','coverage_model_interval'])

    index = 0
    for p in procstat_times_list:
        for m in meminfo_times_list:
            for s in shm_sampler_times_list:
                print(index)
                ldms_data = [shm_sampler[s], procstat[p], meminfo[m]]
                times_list = [shm_sampler_times_list[s], procstat_times_list[p], meminfo_times_list[m]]
                shared_intervals = find_shared_period(times_list, None)
                plot_all(shared_intervals, milestoneRun, ldms_data, ldms_time_data, name=p + "_" + m + "_" + s, savePath='D:/ac/PhD/Research/data/pd/01/all_metrics/')
                union_score, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
                    shared_intervals, model_intervals,
                    sorted(set(ldms_time_data[2].tolist()) | set(ldms_time_data[1].tolist()) | set(
                        ldms_time_data[0].tolist())))
                df_result[index] = (p,m,s,'8-20-20',lamb, union_score, coverage_shared_interval, coverage_model_interval)
                df_result.to_csv('D:/ac/PhD/Research/data/pd/01/all_metrics/df_scores.csv')
                index = index + 1
                gc.collect()

    print("Done")



def ldms_example():
    print('ldms_example')
    [meminfo, shm_sampler, vmstat, procstat, procnetdev, milestoneRun] = pre_process(read_data())

    # try_meminfo_example(meminfo)
    # try_procstat_example(procstat)
    # try_all_samplers(shm_sampler, procstat, meminfo, milestoneRun)
    try_all_metric_combinations(shm_sampler, procstat, meminfo, milestoneRun)
    # try_shm_sampler_example(shm_sampler)

if __name__ == '__main__':
    pool = ThreadPool(8)
    xAxis = '#Time'
    value_name = 'value'
    # authors_example()
    ldms_example()


