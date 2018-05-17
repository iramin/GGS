from datetime import datetime, timedelta
import matplotlib.dates as md
import itertools
import pandas as pd
import os.path
import pickle
import gc
from pandas import DatetimeIndex
from scipy.spatial.distance import cdist

from intervaltree import Interval, IntervalTree

from ggs import *

import ldms_transform as lt



def findBP(data, Kmax=10, lamb=1e-1, features = [], verbose = False):
    return GGS(data, Kmax, lamb, features, verbose)

def findGGSMeanCovForBP(data, breakpoints, lamb=1e-1, features = [], verbose = False):
    return GGSMeanCov(data, breakpoints, lamb, features, verbose)

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

def get_two_level_interval_tree_from_model(model):
    current_tree = IntervalTree()
    ts_tree = IntervalTree()
    for i in range(0,model.shape[0] - 1):
        start = model.iloc[i]['#Time']
        stop = model.iloc[i+1]['#Time']
        if start > stop:
            start = stop -  (start-stop)
            # stop = start + timedelta(microseconds=1)
        if model.iloc[i+1]['metric'] == 'Timestep':
            ts_tree.add(Interval(datetime.utcfromtimestamp(start), datetime.utcfromtimestamp(stop), model.iloc[i+1]['metric']))
        else:
            current_tree.add(Interval(datetime.utcfromtimestamp(start), datetime.utcfromtimestamp(stop), model.iloc[i+1]['metric']))
    return sorted(set(current_tree)), sorted(set(ts_tree))

def filter_motifs(time_list, breaks, motif_Tmin=None, motif_Tmax=None, verbose=True):
    verbose = True
    start = 0
    stop = start + 1
    filtered_list = []
    filtered_break_list = []
    ignored_interval_list = set()

    print("filter motif is called with motif_Tmin={} and motif_Tmax={} and verbose = {}".format(motif_Tmin, motif_Tmax, verbose))

    if(motif_Tmin is None and motif_Tmax is None):
        return breaks,time_list

    while(stop < len(time_list)):
        if(motif_Tmin is not None):

            while(stop < len(time_list) and (datetime.utcfromtimestamp(time_list[stop]) - datetime.utcfromtimestamp(time_list[start])) < motif_Tmin):
                if(verbose):
                    print("ignoring (" + str(time_list[start]) + "," + str(time_list[stop]) + ") because its length is shorter than the defined Tmin: " + str(motif_Tmin))
                    ignored_interval_list.add(time_list[start])
                    ignored_interval_list.add(time_list[stop])
                stop = stop + 1

        if (motif_Tmax is not None):
            while (stop < len(time_list) and (datetime.utcfromtimestamp(time_list[stop]) - datetime.utcfromtimestamp(time_list[start])) > motif_Tmax):
                if(verbose):
                    print("ignoring (" + str(time_list[start]) + "," + str(time_list[stop]) + ") because its length is longer than the defined Tmax: " + str(motif_Tmax))
                    ignored_interval_list.add(time_list[start])
                    ignored_interval_list.add(time_list[stop])
                start = stop
                stop = start + 1

        if(stop < len(time_list)):
            if len(filtered_list) == 0:
                filtered_list.append(time_list[start])
                if breaks != None:
                    filtered_break_list.append(breaks[start])
            filtered_list.append(time_list[stop])

            if breaks != None:
                filtered_break_list.append(breaks[stop])

        start = stop
        stop = start + 1

    return filtered_break_list, filtered_list, sorted(ignored_interval_list)

def convert_break_to_time(data, breaks):
    time = []
    for b in breaks:
        time.append(data.iloc[min(b,data.shape[0] - 1), data.columns.get_loc('#Time')])
    return time

def get_interval_tree_from_time_list(time_list, motif_Tmin=None, motif_Tmax=None):
    filterred_tree = IntervalTree()
    ignored_tree = IntervalTree()

    if motif_Tmin != None or motif_Tmax != None:
        verbose = True
        start = 0
        stop = start + 1

        while (stop < len(time_list)):
            if (motif_Tmin is not None):
                while (stop < len(time_list) and (
                        datetime.utcfromtimestamp(time_list[stop]) - datetime.utcfromtimestamp(
                        time_list[start])) < motif_Tmin):
                    if (verbose):
                        print("ignoring (" + str(time_list[start]) + "," + str(
                            time_list[stop]) + ") because its length is shorter than the defined Tmin: " + str(
                            motif_Tmin))
                        ignored_tree.add(Interval(datetime.utcfromtimestamp(time_list[start]), datetime.utcfromtimestamp(time_list[stop])))
                    stop = stop + 1
            if (motif_Tmax is not None):
                while (stop < len(time_list) and (
                        datetime.utcfromtimestamp(time_list[stop]) - datetime.utcfromtimestamp(
                        time_list[start])) > motif_Tmax):
                    if (verbose):
                        print("ignoring (" + str(time_list[start]) + "," + str(
                            time_list[stop]) + ") because its length is longer than the defined Tmax: " + str(
                            motif_Tmax))
                        ignored_tree.add(Interval(datetime.utcfromtimestamp(time_list[start]), datetime.utcfromtimestamp(time_list[stop])))
                    start = stop
                    stop = start + 1
            if (stop < len(time_list)):
                 filterred_tree.add(Interval(datetime.utcfromtimestamp(time_list[start]), datetime.utcfromtimestamp(time_list[stop])))
            start = stop
            stop = start + 1
    else:
        for start in range(0, len(time_list) - 1):
            stop = start + 1
            filterred_tree.add(Interval(datetime.utcfromtimestamp(time_list[start]), datetime.utcfromtimestamp(time_list[stop])))
    return filterred_tree, ignored_tree

def get_interval_trees_from_times_list(times_list, motif_Tmin=None, motif_Tmax=None):
    filterred_trees = []
    ignored_trees = []
    for times in times_list:
        filterred_tree, ignored_tree = get_interval_tree_from_time_list(times, motif_Tmin, motif_Tmax)
        filterred_trees.append(filterred_tree)
        ignored_trees.append(ignored_tree)
    return filterred_trees, ignored_trees

def find_shared_period(interval_trees_list, epsilon_p=timedelta(microseconds=1)):
    print("finding shared region")
    trees = interval_trees_list
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

def calc_interval_counts(intervals, datapoints):
    # print("calc_interval_counts")
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

def calc_intervals_similarity_score2(intervals1, intervals1_counter, intervals2, intervals2_counter):

    small_interval_counter = intervals1
    large_interval_counter = intervals2
    small_interval_counter_counter = intervals1_counter
    large_interval_counter_counter = intervals2_counter
    if(len(intervals2) < len(intervals1)):
        small_interval_counter = intervals2
        large_interval_counter = intervals1
        small_interval_counter_counter = intervals2_counter
        large_interval_counter_counter = intervals1_counter

    small_interval_counter_len = len(small_interval_counter)
    large_interval_counter_len = len(large_interval_counter)

    small_index=  0
    large_index = 0

    score = 0
    while(small_index < small_interval_counter_len and large_index < large_interval_counter_len):

        small_interval = small_interval_counter[small_index]
        large_interval = large_interval_counter[large_index]

        C_small_interval = small_interval_counter_counter[small_index]
        C_large_interval = large_interval_counter_counter[large_index]

        if small_interval.begin <= small_interval.end < large_interval.begin <= large_interval.end:
            small_index += 1
        elif(small_interval.begin  <= large_interval.begin <= small_interval.end <= large_interval.end):
            score += abs(C_small_interval - C_large_interval)
            small_index += 1
        elif (small_interval.begin <= large_interval.begin <= large_interval.end <= small_interval.end):
            score += abs(C_small_interval - C_large_interval)
            large_index += 1
        elif (large_interval.begin <= large_interval.end < small_interval.begin <= small_interval.end):
            large_index += 1
        elif (large_interval.begin <= small_interval.begin <=  large_interval.end <= small_interval.end):
            score += abs(C_small_interval - C_large_interval)
            large_index += 1
        elif (large_interval.begin <= small_interval.begin <= small_interval.end <= large_interval.end):
            score += abs(C_small_interval - C_large_interval)
            small_index += 1
    return score


def calc_intervals_similarity_score(intervals1, intervals2, datapoints):

    # print("calc_intervals_similarity_score using " + str(len(datapoints)) + " datapoints")
    interval_counter1 = calc_interval_counts(intervals1, datapoints)
    interval_counter2 = calc_interval_counts(intervals2, datapoints)

    score2 = calc_intervals_similarity_score2(intervals1, interval_counter1, intervals2, interval_counter2)

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
    # print("score = " + str(score) + " after comparing " + str(small_interval_counter_len) + " intervals")
    while interval_counter_index < large_interval_counter_len:
        score = score + large_interval_counter[interval_counter_index]
        interval_counter_index = interval_counter_index + 1
    # print("score = " + str(score) + " after comparing " + str(large_interval_counter_len) + " intervals")

    return score, score2, sum(interval_counter1) / len(datapoints), sum(interval_counter2) / len(datapoints)

def concat_metrics(src_df, metric_list):
    xAxis = '#Time'
    value_name = 'value'

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


def create_metric_groups(df, metric_groups):
    mem_dfs = []
    for i, mg in enumerate(metric_groups):
        mem_dfs.append(concat_metrics(df, mg))
    return mem_dfs

def mygrouper(n, iterable):
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.zip_longest(*args))

def process_column(c, df, this_sampler, Kmax, lamb, counter = 0, transform=None, motif_Tmin=None, motif_Tmax=None, verbose=True):
    times_list = []
    if c in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id', 'MemTotal', 'MemFree', 'MemAvailable',
             'Cached']:
        return times_list
    if verbose:
        print(str(counter) + '/' + str(len(df.columns)) + ' c:' + c)
    if transform != None:
        if transform == "rate":
            transformed_df, [ret_names] = lt.create_transform_event(df, [c], [], True,False,False)
        elif transform == "log":
            transformed_df, [ret_names] = lt.create_transform_event(df, [c], [], False, False, True)
        elif transform == "sum":
            transformed_df, [ret_names] = lt.create_transform_event(df, [c], [],False,True,False)
        else:
            print("unknown transform: " + transform)
            return times_list
        transformed_df.fillna(0, inplace=True)
        this_sampler = transformed_df.T
        # ldms_data.append(shm_sampler_group_df['d_MPI_Issend.calls.4_dt'])
    try:
        bps_this_sampler,objectives = findBP(this_sampler, Kmax, lamb, [c])
    except np.linalg.linalg.LinAlgError:
        return times_list
    else:
        if verbose:
            print(len(bps_this_sampler))
            print(bps_this_sampler)
        if (type(bps_this_sampler[0]) is int):
            if (len(bps_this_sampler) > 2):
                times_list = convert_break_to_time(this_sampler.T, bps_this_sampler)
        else:
            times_list = convert_break_to_time(this_sampler.T, bps_this_sampler[len(bps_this_sampler) - 1])
    # filtered_break_list, filtered_times_list, remaining_times_list = filter_motifs(times_list, None, motif_Tmin, motif_Tmax, verbose)


    return times_list

def processMultiColumn(columns, df, this_sampler, Kmax, lamb, counter = 0, transform=None, motif_Tmin=None, motif_Tmax=None, verbose=True):
    print("processMultiColumn")
    times_list = []
    if transform != None:
        transformed_df = df
        if transform == "rate":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, columns, [], True,False,False)
        elif transform == "log":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, columns, [], False, False, True)
        elif transform == "sum":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, columns, [],False,True,False)
        else:
            print("unknown transform: " + transform)
            return times_list
        transformed_df.fillna(0, inplace=True)
        this_sampler = transformed_df.T
    try:
        bps_this_sampler, objectives = findBP(this_sampler, Kmax, lamb, columns)
    except np.linalg.linalg.LinAlgError:
        return times_list
    else:
        if verbose:
            print(len(bps_this_sampler))
            print(bps_this_sampler)
        if (type(bps_this_sampler[0]) is int):
            if (len(bps_this_sampler) > 2):
                times_list = convert_break_to_time(this_sampler.T, bps_this_sampler)
        else:
            times_list = convert_break_to_time(this_sampler.T, bps_this_sampler[len(bps_this_sampler) - 1])
    return times_list

def calcBreakpointsForMultipleColumns(name,df, Kmax,listOfMetrics=None, transform=None,  motif_Tmin=None, motif_Tmax=None, lamb=1e-1, verbose=True):
    print("calcBreakpointsForMultipleColumns")
    times_list = []
    if(os.path.exists(name)):
        print("found the data on disk. loading from " + name)
        file = open(name, 'rb')
        object_file = pickle.load(file)
        file.close()
        return object_file
    else:
        print("did not find the data on disk. calculating and stroing into " + name)
    this_sampler = df.T
    result = processMultiColumn(listOfMetrics, df, this_sampler, Kmax, lamb, 0, transform, motif_Tmin,
                            motif_Tmax, verbose)
    if (len(result) > 0):
        times_list = result
    with open(name, 'wb') as handle:
        pickle.dump(times_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return times_list

def parallel_calc_breakpoints_for(name, numThreads,df, Kmax,listOfMetrics=None, metric_transform_map={},  motif_Tmin=None, motif_Tmax=None, lamb=1e-1, verbose=True):

    if(os.path.exists(name)):
        print("found the data on disk. loading from " + name)
        file = open(name, 'rb')
        object_file = pickle.load(file)
        file.close()
        return object_file
    else:
        print("did not find the data on disk. calculating and stroing into " + name)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    if listOfMetrics == None:
        columnList = mygrouper(numThreads, df.columns)
    else:
        columnList = mygrouper(numThreads, listOfMetrics)
    counter = 0
    times_list = {}
    this_sampler = df.T  # Convert to an n-by-T matrix
    for columns in columnList:
        if(len(columns) == numThreads):
            args = []
            for c in columns:
                counter = counter + 1
                args.append((c, df, this_sampler, Kmax, lamb, counter,None, motif_Tmin, motif_Tmax, verbose))

            results = pool.starmap(process_column, args)

            for index, a in enumerate(args):
                if(len(results[index][0]) > 0):
                    times_list[a[0]] = results[index]
        else:
            for c in columns:
                counter = counter + 1
                result = process_column(c, df, this_sampler, Kmax, lamb, counter, None,  motif_Tmin, motif_Tmax, verbose)
                if(len(result) > 0):
                    times_list[c] = result
    pool.close()
    pool.join()

    with open(name, 'wb') as handle:
        pickle.dump(times_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
            bps_this_sampler, objectives = findBP(this_sampler, Kmax, lamb, [c])
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

def process_metric_combination(index,s,p,m, meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, ldms_time_data, model_intervals):
    # print("processing " + str(index) + " / " + str(len(procstat_times_list.keys()) *  len(meminfo_times_list.keys()) * len(shm_sampler_times_list.keys())))
    ldms_data = [shm_sampler[s], procstat[p], meminfo[m]]
    times_list = [shm_sampler_times_list[s], procstat_times_list[p], meminfo_times_list[m]]

    filterred_trees, ignored_trees = get_interval_trees_from_times_list(times_list)

    shared_intervals = find_shared_period(filterred_trees)
    # if(index % 1000 == 0):
    plot_all(shared_intervals, None, milestoneRun, ldms_data, ldms_time_data, name=p + "_" + m + "_" + s,
         savePath='D:/ac/PhD/Research/data/pd/01/all_metrics/')
    union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
        shared_intervals, model_intervals,
        sorted(set(ldms_time_data[2].tolist()) | set(ldms_time_data[1].tolist()) | set(
            ldms_time_data[0].tolist())))
    return [union_score, score2, coverage_shared_interval, coverage_model_interval]

def process_metric_list_combination(index,sList,pList,mList, meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, ldms_time_data, model_intervals):
    # print("processing " + str(index) + " / " + str(len(procstat_times_list.keys()) *  len(meminfo_times_list.keys()) * len(shm_sampler_times_list.keys())))
    ldms_data = []
    times_list = []
    for s,p,m in zip(sList, pList,mList):
        ldms_data.append(shm_sampler[s])
        ldms_data.append(procstat[p])
        ldms_data.append(meminfo[m])
        times_list.append(shm_sampler_times_list[s])
        times_list.append(procstat_times_list[p])
        times_list.append(meminfo_times_list[m])
    times_list = [shm_sampler_times_list[s], procstat_times_list[p], meminfo_times_list[m]]
    filterred_trees, ignored_trees = get_interval_trees_from_times_list(times_list)
    shared_intervals = find_shared_period(filterred_trees)
    # if(index % 1000 == 0):
    plot_all(shared_intervals, milestoneRun, ldms_data, ldms_time_data, name="all_proc_mem_shm_k200",
         savePath='D:/ac/PhD/Research/data/pd/01/all_metrics/')
    union_score, score2, coverage_shared_interval, coverage_model_interval = calc_intervals_similarity_score(
        shared_intervals, model_intervals,
        sorted(set(ldms_time_data[2].tolist()) | set(ldms_time_data[1].tolist()) | set(
            ldms_time_data[0].tolist())))
    return [union_score, score2, coverage_shared_interval, coverage_model_interval]

def calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, lamb=1e-1):

    model_intervals = get_interval_tree_from_model(milestoneRun)
    ldms_time_data = [md.epoch2num(shm_sampler['#Time']), md.epoch2num(procstat['#Time']),
                      md.epoch2num(meminfo['#Time'])]

    df_result = pd.DataFrame(columns = ['procstat','meminfo','shm_sampler','KMax','lamb','union_score', 'coverage_shared_interval','coverage_model_interval'])


    index = 0
    for p in procstat_times_list:
        for m in meminfo_times_list:
            for s in shm_sampler_times_list:

                result = process_metric_combination(index, s, p, m, meminfo_times_list, procstat_times_list,
                                           shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun,
                                           ldms_time_data, model_intervals)
                df_result[index] = (p,m,s,'8-20-20',lamb, result[0], result[1], result[2], result[3])
                df_result.to_csv('D:/ac/PhD/Research/data/pd/01/all_metrics/df_scores.csv')
                index = index + 1
                gc.collect()

def calc_score_store_plot_metric_list(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=None, meminfo_selected_keys=None, shm_sampler_selected_keys=None, numThreads=None, lamb=1e-1):

    model_intervals = get_interval_tree_from_model(milestoneRun)
    ldms_time_data = [md.epoch2num(shm_sampler['#Time']), md.epoch2num(procstat['#Time']),
                      md.epoch2num(meminfo['#Time'])]

    df_result = pd.DataFrame(
        columns=['procstat', 'meminfo', 'shm_sampler', 'KMax', 'lamb', 'union_score', 'score2', 'coverage_shared_interval',
                 'coverage_model_interval'])

    if(procstat_selected_keys == None):
        procstat_selected_keys = procstat_times_list.keys()
    if (meminfo_selected_keys == None):
        meminfo_selected_keys = meminfo_times_list.keys()
    if (shm_sampler_selected_keys == None):
        shm_sampler_selected_keys = shm_sampler_times_list.keys()

    result = process_metric_list_combination(0, shm_sampler_selected_keys, procstat_selected_keys, meminfo_selected_keys, meminfo_times_list, procstat_times_list,
                               shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun,
                               ldms_time_data, model_intervals)
    df_result.loc[0] = (
    "allm", "allp", "alls", '8-20-200', lamb, result[0], result[1],
    result[2], result[3])
    df_result.to_csv('D:/ac/PhD/Research/data/pd/01/all_metrics/df_scores_all_k200.csv')

def parallel_calc_score_store_plot(meminfo_times_list, procstat_times_list, shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun, procstat_selected_keys=None, meminfo_selected_keys=None, shm_sampler_selected_keys=None, numThreads=None, lamb=1e-1):
    if numThreads == None:
        numThreads = multiprocessing.cpu_count()
    print("parallel_calc_score_store_plot using " + str(numThreads) + " threads")
    model_intervals = get_interval_tree_from_model(milestoneRun)
    ldms_time_data = [md.epoch2num(shm_sampler['#Time']), md.epoch2num(procstat['#Time']),
                      md.epoch2num(meminfo['#Time'])]
    df_result = pd.DataFrame(
        columns=['procstat', 'meminfo', 'shm_sampler', 'KMax', 'lamb', 'union_score', 'score2', 'coverage_shared_interval',
                 'coverage_model_interval'])
    if(procstat_selected_keys == None):
        procstat_selected_keys = procstat_times_list.keys()
    if (meminfo_selected_keys == None):
        meminfo_selected_keys = meminfo_times_list.keys()
    if (shm_sampler_selected_keys == None):
        shm_sampler_selected_keys = shm_sampler_times_list.keys()
    all_metrics_combinations = list(itertools.product(procstat_selected_keys, meminfo_selected_keys, shm_sampler_selected_keys))

    metric_combinations_set = mygrouper(numThreads, all_metrics_combinations)
    pool = multiprocessing.Pool(numThreads)
    index = 0
    min_score = 100000
    max_score = 0
    min_score2 = 2000000
    max_score2 = 0
    total_combinations_len = len(procstat_times_list.keys()) * len(meminfo_times_list.keys()) * len(shm_sampler_times_list.keys())
    for metric_combination_subset in metric_combinations_set:
        print("processing " + str(index) + " / " + str(total_combinations_len) + " min_score= " + str(min_score) + " max_score= " + str(max_score) + " min_score2= " + str(min_score2) + " max_score2= " + str(max_score2))
        if(len(metric_combination_subset) == numThreads):
            args = []
            for metric_combination in metric_combination_subset:
                args.append((index, metric_combination[2], metric_combination[0], metric_combination[1], meminfo_times_list, procstat_times_list,
                                           shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun,
                                           ldms_time_data, model_intervals))
                index = index+1

            results = pool.starmap(process_metric_combination, args)


            for i, a in enumerate(args):
                df_result.loc[index+i-8] = (a[2],a[3],a[1],'8-20-20',lamb, results[i][0], results[i][1], results[i][2], results[i][3])
                min_score = min(min_score, results[i][0])
                max_score = max(max_score, results[i][0])
                min_score2 = min(min_score2, results[i][1])
                max_score2 = max(max_score2, results[i][1])
                if((index + i) % 1000 == 0):
                    df_result.to_csv('D:/ac/PhD/Research/data/pd/01/all_metrics/df_scores.csv')
        else:
            for metric_combination in metric_combination_subset:
                result = process_metric_combination(index, metric_combination[2], metric_combination[0], metric_combination[1], meminfo_times_list, procstat_times_list,
                                           shm_sampler_times_list, shm_sampler, procstat, meminfo, milestoneRun,
                                           ldms_time_data, model_intervals)
                df_result.loc[index-8] = (metric_combination[0],metric_combination[1],metric_combination[2],'8-20-20',lamb, result[0], result[1], result[2], result[3])
                # df_result.to_csv('D:/ac/PhD/Research/data/pd/01/all_metrics/df_scores.csv')
                index = index+1
    pool.close()
    pool.join()
    print("Done! flushing results...")
    df_result.to_csv('D:/ac/PhD/Research/data/pd/01/all_metrics/df_scores.csv')


def checkIntervalColumn(row,column):
    # print(column)
    if (column != 'rowIndex'):
        if column[0] <= row['rowIndex'] < column[1]:
            return 1
        else:
            return 0
        # print("row.index={} row[{}]={}".format(row['rowIndex'],column,row[column]))

def checkIntervals(row, columns):
    # print(row)
    # return columns.apply(lambda column: checkIntervalColumn(row,column))
    for c in columns:
        if(c == 'rowIndex'):
            continue
        if c[0] <= row['rowIndex'] < c[1]:
            row[c] = 1
        else:
            row[c] = 0
        # print("row.index={} row[{}]={}".format(row['rowIndex'],c,row[c]))
        # print(row[c])
    return row

def testNewFunc(x):
    print("testNewFunc1")
    print(x)
    print("testNewFunc2")
    print(x.name)
    print("testNewFunc3")
    print(x.index)
    if x.name[0] <= x.index < x.name[1]:
        return 1
    else:
        return 0

def my_fun(x,y):
    print("test my_func 1")
    print(x)
    print("test my_func 2")
    print(y)

def calcIBSMDistance(allTimes, phaseSet1, phaseSet2):
    print("calcIBSMDistance")
    df1 = pd.DataFrame(index=allTimes, columns=phaseSet1)
    df2 = pd.DataFrame(index=allTimes, columns=phaseSet2)


    df1['rowIndex'] = df1.index
    df2['rowIndex'] = df2.index

    df1 = df1.apply(lambda row: checkIntervals(row, pd.Series(df1.columns)), axis=1)
    df2 = df2.apply(lambda row: checkIntervals(row, pd.Series(df2.columns)), axis=1)


    df1.drop(columns='rowIndex', inplace=True)
    df2.drop(columns='rowIndex', inplace=True)

    # print(df1.sum())
    # print(df2.sum())
    # print(df1.sum(axis=1))
    # print(df2.sum(axis=1))

    diff_df = abs(df1.values - df2.values)
    print("Y=")
    print(math.sqrt(diff_df.sum()))



from PhaseDetectionPlot import plot_all