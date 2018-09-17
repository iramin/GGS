import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.patches as patches
import gc

from PhaseDetection import get_interval_tree_from_model
from PhaseDetection import *

def plot_all(intervals, intervals_blocked, milestoneRun,  ldms_data, ldms_time_data, name='all_in_one', savePath='D:/ac/PhD/Research/data/pd/01/', format='.pdf'):
    print("plotting all (" + name + ")")
    number_of_plots = 2 + len(ldms_data)

    begin = md.date2num(intervals[0].begin)
    end = md.date2num(intervals[len(intervals) - 1].end)

    x_min = begin + begin - md.date2num(intervals[0].end)
    x_max = end + end - md.date2num(intervals[len(intervals) - 1].begin)

    model_intervals = get_interval_tree_from_model(milestoneRun)

    begin = md.date2num(model_intervals[0].begin)
    end = md.date2num(model_intervals[len(model_intervals) - 1].end)

    x_min = min(x_min, begin + begin - md.date2num(model_intervals[0].end))
    x_max = max(x_max, end + end - md.date2num(model_intervals[len(model_intervals) - 1].begin))

    if intervals_blocked != None and len(intervals_blocked) > 0:
        begin = md.date2num(intervals_blocked[0].begin)
        end = md.date2num(intervals_blocked[len(intervals_blocked) - 1].end)

        x_min = min(x_min, begin + begin - md.date2num(intervals_blocked[0].end))
        x_max = max(x_max, end + end - md.date2num(intervals_blocked[len(intervals_blocked) - 1].begin))

        number_of_plots = number_of_plots + 1


    fig, axs_ret = plt.subplots(nrows=number_of_plots, sharex=True)
    axs = {}
    if number_of_plots == 1:
        axs[0] = axs_ret
    else:
        axs = axs_ret


    for index, data in enumerate(ldms_data):
        if(ldms_time_data[index%len(ldms_time_data)].min() < x_min):
            x_min = ldms_time_data[index%len(ldms_time_data)].min()
        if(ldms_time_data[index%len(ldms_time_data)].max() > x_max):
            x_max = ldms_time_data[index%len(ldms_time_data)].max()

    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)

    plot_interval(intervals, axs[number_of_plots - 1], x_min, x_max)
    if intervals_blocked != None and len(intervals_blocked) > 0:
        plot_interval(intervals_blocked, axs[number_of_plots - 2], x_min, x_max)
        plot_model(milestoneRun, axs[number_of_plots - 3], x_min, x_max)
    else:
        plot_model(milestoneRun, axs[number_of_plots - 2], x_min, x_max)
    for index, data in enumerate(ldms_data):
        axs[index].plot(ldms_time_data[index%len(ldms_time_data)],data)
        axs[index].xaxis.set_major_locator(md.SecondLocator(interval=60))
        axs[index].xaxis.set_minor_locator(md.SecondLocator(interval=10))
        axs[index].xaxis.set_major_formatter(xfmt)
        axs[index].set_xlim(x_min, x_max)


    fig.set_size_inches(h=18.5, w=20)
    fig.autofmt_xdate()
    # print('saving figure')
    fig.savefig(savePath + name + format, dpi=2400)

    fig.clf()
    plt.close()
    gc.collect()

def plot_interval(intervals, ax,min_start, max_end, fill=True, patterns = ['-', '*', '.', '+',  'o', 'x', 'O' ], colors= ["red", "blue","green","yellow","purple","cyan","white"],separateTSWithPattern=True, verbose=False):
    if verbose:
        print("plot_interval")
    y0 = 0
    height = 1
    tsIndex = -1
    data_counters = {}
    for index, interval in enumerate(intervals):
        begin = md.date2num(interval.begin)
        end = md.date2num(interval.end)

        if(interval.data != None):
            if index != 0:
                if interval.data == 'Timestep':
                    tsIndex = tsIndex + 1
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
            if verbose:
                print("metric={}, tsIndex={}, pattern={}, color={}".format(interval.data, tsIndex, pattern, color))
            if index == 0:
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
    ax.xaxis.set_minor_locator(md.SecondLocator(interval=10))
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlim(min_start, max_end)
    ax.set_ylim(y0, y0+height)

    # for dc in data_counters:
    #     print("{}= {}".format(dc,data_counters[dc]))

def plot_model(model, ax,min_start, max_end, fill=True, patterns = ['-', '*', '.', '+',  'o', 'x', 'O'], colors= ["red", "blue","green","yellow","purple","cyan","white"], verbose=False):
    plot_interval(get_interval_tree_from_model(model), ax, min_start, max_end, fill, patterns, colors, verbose=verbose)

def plotBP(objectives):
    # Plot objective vs. number of breakpoints. Note that the objective essentially
    # stops increasing after K = 2, since there are only 2 "true" breakpoints
    plotVals = range(len(objectives))
    plt.plot(plotVals, objectives, 'or-')
    plt.xlabel('Number of Breakpoints')
    plt.ylabel(r'$\phi(b)$')
    plt.show()

def Plot_predicted_Mean_Covs(breaks, data,features = [], motif_Tmin=30, motif_Tmax=300, verbose = True):
    print("plotting mean covs")
    if len(breaks) <= 1:
        return

    mcs = findGGSMeanCovForBP(data, breaks, 1e-1, features, verbose)

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