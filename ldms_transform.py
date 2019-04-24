
from numpy import log
import datetime

def calcRate(df, metric, timeMetric='#Time'):
    # print(type(df[timeMetric][0]))
    # # print(df[timeMetric]-df[timeMetric].shift())
    # # print(df[timeMetric].apply(lambda x: datetime.datetime.timestamp((x))))
    # df['test_movaght'] = df[timeMetric].apply(lambda x: datetime.datetime.timestamp((x)))
    # return ((df[metric] - df[metric].shift()) / ( df['test_movaght']-df['test_movaght'].shift()))
    return ((df[metric] - df[metric].shift()) / (df[timeMetric] - df[timeMetric].shift()))

def calcLog(df, metric, prefix='log_', postfix=''):
    df[prefix + metric + postfix] = df[metric].apply(lambda x: log(x))
    return df

def processRatesForEventGroup(df, group, timeMetric='#Time', prefix='d_', postfix='_dt'):
    ret_group = []
    for i, item in enumerate(group):
        newMetricName = prefix + group[i] + postfix
        df[newMetricName] = calcRate(df, group[i],timeMetric)
        ret_group.append(newMetricName)
    return df, ret_group

def processSumsForEventGroup(df, group, group_name, timeMetric='#Time', prefix='sum_', postfix=''):
    newMetricName = prefix + group_name + postfix
    ret_group = [newMetricName]
    ret_df = None
    for i, item in enumerate(group):
        if(ret_df == None):
            ret_df = df[group[i]]
            ret_df[newMetricName] = df[group[i]]
            ret_df.drop(group[i])
        else:
            ret_df = ret_df + df[group[i]]
    return ret_df, ret_group

def processLogsForEventGroup(df, group , timeMetric='#Time',  prefix='log_', postfix=''):
    ret_group = []
    for i, item in enumerate(group):
        newMetricName = prefix + group[i] + postfix
        df[newMetricName] = calcLog(df, group[i], timeMetric)
        ret_group.append(newMetricName)
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



def create_transform_event(df, events = [], transformed_names=[], ratetransform = True, sumtransform= False, logtransform = False):
    if(sumtransform):
        transformed_df, [ret_names]= do_sum_transform(df, [events], [transformed_names])

    if(ratetransform):
        transformed_df, [ret_names] = do_rate_transform(df, [events])

    if(logtransform):
        transformed_df, [ret_names] = do_log_transform(df, [events])
    return transformed_df, [ret_names]