import logging
import subprocess
import warnings

import matplotlib

import _winapi
import multiprocessing.spawn

import pandas as pd
from scipy.stats import chisquare

from owlpy.core import *

from matrixprofile import *

from matrixprofile.discords import discords


from ldms import *
from PhaseDetectionPlot import *
# from sklearn import mixture
# from sklearn.cluster import KMeans
from sklearn import cluster, datasets as skds, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from numpy.random import seed
from numpy.random import rand

from itertools import cycle, islice

class OneMetricPhaseSet(object):
    def __init__(self, name, phase_identifier, category, metric, motif_length, *args, **kwargs):
        """"""
        self.name = name

        self.phase_identifier = phase_identifier

        self.sampler_name = phase_identifier.categoriesToSamplerMap[category]
        motif_file_exists = True
        self.metric_file_name = phase_identifier.metricsFolder + self.sampler_name + "/" + metric + phase_identifier.inputFileNameExtension
        self.motif_file_name = phase_identifier.motifsFolder + self.sampler_name + "/" + str(
            motif_length) + "/" + metric + phase_identifier.inputFileNameExtension
        if not motif_file_exists:
            print("Motif file does not exist")

        self.ditanceColIndexInHimeOutput = kwargs.get('ditanceColIndexInHimeOutput',6)
        self.startColIndexInHimeOutput = kwargs.get('startColIndexInHimeOutput', 1)
        self.stopColIndexInHimeOutput = kwargs.get('stopColIndexInHimeOutput', 2)
        self.motifLengthColIndexInHimeOutput = kwargs.get('motifLengthColIndexInHimeOutput', 5)

        self.colmunsToDropInHimeOutput = kwargs.get('colmunsToDropInHimeOutput', [0])
        self.motif_length = motif_length
        self.metric = metric
        self.sampler_df = self.phase_identifier.get_sampler_for_category(category)

        loadData = kwargs.get('loadData', True)



        # self.load_metrics_motifs()
        if metric == 'user.rate':
            self.range_load_metrics_motifs_range(start_range=[0], stop_range=[self.sampler_df.shape[0]], transform='rate')
            self.metric_ts = self.metric_ts_range[0]
            self.motifs = self.motifs_range[0]
        else:
            if loadData:
                self.load_metrics_motifs()
            else:
                self.metric_ts = self.load_metric_data_1column(self.metric_file_name)

    def range_load_metrics_motifs_range(self, start_range, stop_range, transform=None):
        self.metric_ts_range = []
        self.motifs_range = []
        for i in range(len(start_range)):
            logger = logging.getLogger(__name__)
            logger.info("start={}, stop={}".format(start_range[i], stop_range[i]))



            metric_file_name = self.phase_identifier.metricsFolder + self.sampler_name + "/" + self.metric + '_' + str(
                start_range[i]) + '_' + str(stop_range[i]) + self.phase_identifier.inputFileNameExtension
            motif_file_name = self.phase_identifier.motifsFolder + self.sampler_name + "/" + str(
                self.motif_length) + "/" + self.metric + '_' + str(start_range[i]) + '_' + str(
                stop_range[i]) + self.phase_identifier.inputFileNameExtension

            try:
                self.metric_ts_range.append(self.load_metric_data_1column(metric_file_name))
            except Exception as e:
                print(e)
                directory = self.phase_identifier.metricsFolder + self.sampler_name + "/"
                fileName = self.metric + '_' + str(start_range[i]) + '_' + str(
                stop_range[i]) + self.phase_identifier.inputFileNameExtension



                self.writeColumnInFile(self.sampler_df.iloc[start_range[i]:stop_range[i]], column=self.metric[:-5], transform=transform, folder=directory, fileName=fileName)
                self.metric_ts_range.append(self.load_metric_data_1column(metric_file_name))
            try:
                self.motifs_range.append(self.load_motifs_from_hime_output(motif_file_name, sort_on_distance=True))
            except Exception as e:
                print(e)
                subprocess.run(["java","-cp", "D://ac//PhD//Research//src//TestHime//bin" , "Run" , str(self.motif_length) ,self.phase_identifier.metricsFolder , self.sampler_name ,metric_file_name])# -cp bin D:/ac/PhD/Research/src/TestHime/bin/Run "])# + str(self.motif_length) + " " + self.phase_identifier.metricsFolder + " " + self.sampler_name + " " + self.metric])
                self.motifs_range.append(self.load_motifs_from_hime_output(motif_file_name, sort_on_distance=True))



    def load_motifs_from_hime_output(self, file_path, sort_on_distance=True):
        logger = logging.getLogger(__name__)
        logger.info("load motifs from " + file_path)
        motifs = read_csv(file_path, delim_whitespace=True, header=None)
        if sort_on_distance:
            logger.info("sort based on distance")
            motifs = motifs.drop(columns=self.colmunsToDropInHimeOutput).sort_values([self.ditanceColIndexInHimeOutput])
        return motifs

    def load_metric_data_1column(self, file_path):
        logger = logging.getLogger(__name__)
        logger.info("load metrics from " + file_path)
        my_data = read_csv(file_path, sep=',', header=None)
        ts = my_data.values.flatten()
        return ts


    def load_metrics_motifs(self):
        self.metric_ts = self.load_metric_data_1column(self.metric_file_name)
        try:
            self.motifs = self.load_motifs_from_hime_output(self.motif_file_name, sort_on_distance=True)
        except Exception as e:
            print(e)
            cp = subprocess.run(["java","-cp", "D://ac//PhD//Research//src//TestHime//bin" , "Run" , str(self.motif_length) ,self.phase_identifier.metricsFolder , self.sampler_name ,self.metric_file_name])# -cp bin D:/ac/PhD/Research/src/TestHime/bin/Run "])# + str(self.motif_length) + " " + self.phase_identifier.metricsFolder + " " + self.sampler_name + " " + self.metric])
            cp.check_returncode()
            self.motifs = self.load_motifs_from_hime_output(self.motif_file_name, sort_on_distance=True)



    def writeColumnInFile(self, df, column='MPI_Issend.calls.0', transform=None, fileName=None, folder=''):
        print(column)
        this_sampler = df
        if transform != None:
            transformed_df = df
            if transform == "rate":
                transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], True, False,
                                                                        False)
            elif transform == "log":
                transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], False, False,
                                                                        True)
            elif transform == "sum":
                transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], False, True,
                                                                        False)
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

    def load_metrics_motifs_range(self, start, stop, transform=None):
        logger = logging.getLogger(__name__)
        logger.info("start={}, stop={}".format(start, stop))
        try:
            self.metric_ts = self.load_metric_data_1column(self.metric_file_name)
        except Exception as e:
            print(e)
            directory = self.phase_identifier.metricsFolder + self.sampler_name + "/"
            fileName = self.metric + self.phase_identifier.inputFileNameExtension
            self.writeColumnInFile(self.sampler_df[start:stop], column=self.metric, transform=transform, folder=directory, fileName=fileName)
            self.metric_ts = self.load_metric_data_1column(self.metric_file_name)
        try:
            self.motifs = self.load_motifs_from_hime_output(self.motif_file_name, sort_on_distance=True)
        except Exception as e:
            print(e)
            subprocess.run(["java","-cp", "D://ac//PhD//Research//src//TestHime//bin" , "Run" , str(self.motif_length) ,self.phase_identifier.metricsFolder , self.sampler_name ,self.metric_file_name])# -cp bin D:/ac/PhD/Research/src/TestHime/bin/Run "])# + str(self.motif_length) + " " + self.phase_identifier.metricsFolder + " " + self.sampler_name + " " + self.metric])
            self.motifs = self.load_motifs_from_hime_output(self.motif_file_name, sort_on_distance=True)


    def remove_overlaps_from_hime_output(self):
        logger = logging.getLogger(__name__)
        logger.info("start")
        pointer = 0
        while pointer < len(self.motifs.index) - 1:
            i = pointer + 1
            while i < len(self.motifs.index):
                if abs(self.motifs.iloc[pointer].iloc[0] - self.motifs.iloc[i].iloc[0]) < self.motif_length:
                    self.motifs.drop(self.motifs.index[i], inplace=True)
                else:
                    i = i + 1
            pointer = pointer + 1
        return self.motifs

    def select_best_motif(self):
        # self.tuned_hime_motifs = self.remove_overlaps_from_hime_output()
        self.tuned_hime_motifs = self.motifs
        # print(self.tuned_hime_motifs)
        self.selected_motif = self.tuned_hime_motifs.iloc[0]

    def select_best_motif_in_range(self):
        # self.tuned_hime_motifs = self.remove_overlaps_from_hime_output()

        # logger = logging.getLogger(__name__)
        # logger.info("start")
        # pointer = 0
        # while pointer < len(self.motifs.index) - 1:
        #     i = pointer + 1
        #     while i < len(self.motifs.index):
        #         if abs(self.motifs.iloc[pointer].iloc[0] - self.motifs.iloc[i].iloc[0]) < self.motif_length:
        #             self.motifs.drop(self.motifs.index[i], inplace=True)
        #         else:
        #             i = i + 1
        #     pointer = pointer + 1
        # return self.motifs


        # print(self.tuned_hime_motifs)
        self.selected_motif_range = []
        for i in range(len(self.motifs_range)):
            self.selected_motif_range.append(self.motifs_range[i].iloc[0])

    def build_query_from_selected_motif(self):
        start = int(self.selected_motif[self.startColIndexInHimeOutput])
        stop = int(self.selected_motif[self.stopColIndexInHimeOutput])
        self.real_motif_length = int(self.selected_motif[self.motifLengthColIndexInHimeOutput])
        self.query = self.metric_ts[start:stop]

    def build_query_from_selected_motif_in_range(self):
        self.real_motif_length_in_range = []
        self.query_in_range = []

        for i in range(len(self.selected_motif_range)):
            start = int(self.selected_motif_range[i][self.startColIndexInHimeOutput])
            stop = int(self.selected_motif_range[i][self.stopColIndexInHimeOutput])
            self.real_motif_length_in_range.append(int(self.selected_motif_range[i][self.motifLengthColIndexInHimeOutput]))
            self.query_in_range.append(self.metric_ts_range[i][start:stop])

    def build_query_from_selected_motif_range(self):
        start = int(self.selected_motif[self.startColIndexInHimeOutput])
        stop = int(self.selected_motif[self.stopColIndexInHimeOutput])
        self.real_motif_length = int(self.selected_motif[self.motifLengthColIndexInHimeOutput])
        self.query = self.metric_ts[start:stop]

    def select_best_motifs_from_stamp(self):
        logger = logging.getLogger(__name__)
        logger.info("start")

        dataset = pd.DataFrame({'Pab': self.matrix_profile})
        result = list(dataset.sort_values(['Pab']).index.values)
        pointer = 0
        while pointer < len(result) - 1:
            i = pointer + 1
            while i < len(result):
                if abs(result[pointer] - result[i]) < self.motif_length:
                    del result[i]
                else:
                    i = i + 1
            pointer = pointer + 1
        self.selected_motifs = result

    def search_for_similar_patterns_in(self):
        logger = logging.getLogger(__name__)
        # print(self.query)
        start = int(self.selected_motif[self.startColIndexInHimeOutput])
        stop = int(self.selected_motif[self.stopColIndexInHimeOutput])
        logger.info("use stamp to search for patterns similar to start={}, stop={}, real_motif_length={}".format(start, stop,self.real_motif_length))
        self.matrix_profile, self.nearest_neighbour_index = stamp(self.metric_ts, self.query, self.real_motif_length)
        self.select_best_motifs_from_stamp()

    def search_for_similar_patterns_in_range(self, range):
        logger = logging.getLogger(__name__)
        # print(self.query)
        start = int(self.selected_motif[self.startColIndexInHimeOutput])
        stop = int(self.selected_motif[self.stopColIndexInHimeOutput])
        logger.info("use stamp to search for patterns similar to start={}, stop={}, real_motif_length={}".format(start, stop,self.real_motif_length))
        self.matrix_profile, self.nearest_neighbour_index = stamp(range, self.query, self.real_motif_length)
        self.select_best_motifs_from_stamp()

    def range_search_for_similar_patterns_in_range(self, range, query, real_motif_length):
        logger = logging.getLogger(__name__)
        # print(self.query)
        # start = int(self.selected_motif[self.startColIndexInHimeOutput])
        # stop = int(self.selected_motif[self.stopColIndexInHimeOutput])
        # logger.info("use stamp to search for patterns similar to start={}, stop={}, real_motif_length={}".format(start, stop,self.real_motif_length))
        self.matrix_profile, self.nearest_neighbour_index = stamp(range, query, real_motif_length)
        self.select_best_motifs_from_stamp()


class PhaseIdentifier(object):

    """

    Attributes:

    """

    def __init__(self, name, workload_path, model, *args, **kwargs):
        """"""
        self.name = name

        self.all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']
        self.categoriesToSamplerMap = {}
        self.categoriesToSamplerMap['filesystem reads'] = 'procnfs'
        self.categoriesToSamplerMap['filesystem writes'] = 'procnfs'
        self.categoriesToSamplerMap['Memory usage'] = 'meminfo'
        self.categoriesToSamplerMap['Slow network interface usage'] = 'procnetdev'
        self.categoriesToSamplerMap['Fast interface network usage'] = 'procnetdev'
        self.categoriesToSamplerMap['MPI calls'] = 'shm_sampler'
        self.categoriesToSamplerMap['Cpu usage'] = 'procstat'
        self.categoriesToSamplerMap['vmstat'] = 'vmstat'
        self.categoriesToMetricsMap = {}
        self.categoriesToMetricsMap['filesystem reads'] = ['Numcalls','read.rate']
        self.categoriesToMetricsMap['filesystem writes'] = ['Numcalls','write.rate']
        self.categoriesToMetricsMap['Memory usage'] = ['Dirty','MemFree']
        self.categoriesToMetricsMap['Slow network interface usage'] = ['tx_bytes#eth0.rate']
        self.categoriesToMetricsMap['Fast interface network usage'] = ['tx_bytes#eth0.rate']
        self.categoriesToMetricsMap['MPI calls'] = ['MPI_Issend.calls.1.rate', 'MPI_Isend.calls.0.rate', 'MPI_Irecv.calls.0.rate', 'MPI_Wait.calls.0.rate', 'MPI_Ssend.calls.1.rate', 'MPI_Allreduce.calls.0.rate']
        self.categoriesToMetricsMap['Cpu usage'] = ['Per_core_softirqd0.rate', 'Per_core_sys1', 'Sys.rate']
        self.categoriesToMetricsMap['vmstat'] = 'numa_hit'

        self.metricsToCategoriesMap = {}
        self.metricsToCategoriesMap['write.rate'] = 'filesystem writes'
        self.metricsToCategoriesMap['read.rate'] = 'filesystem reads'
        self.metricsToCategoriesMap['Dirty'] = 'Memory usage'
        self.metricsToCategoriesMap['MemFree'] = 'Memory usage'
        self.metricsToCategoriesMap['tx_bytes#eth0.rate'] = 'Fast interface network usage'
        self.metricsToCategoriesMap['MPI_Issend.calls.0'] = 'MPI calls'
        self.metricsToCategoriesMap['MPI_Issend.calls.0.rate'] = 'MPI calls'
        self.metricsToCategoriesMap['MPI_Allreduce.calls.0.rate'] = 'MPI calls'
        self.metricsToCategoriesMap['sys.rate'] = 'Cpu usage'
        self.metricsToCategoriesMap['per_core_softirqd0.rate'] = 'Cpu usage'
        self.metricsToCategoriesMap['nr_dirty'] = 'vmstat'
        self.metricsToCategoriesMap['numa_hit'] = 'vmstat'

        self.samplersToCategoriesMap = {}
        self.samplersToCategoriesMap['meminfo'] = 'Memory usage'
        self.samplersToCategoriesMap['shm_sampler'] = 'MPI calls'
        self.samplersToCategoriesMap['procstat'] = 'Cpu usage'
        self.samplersToCategoriesMap['vmstat'] = 'vmstat'
        self.samplersToCategoriesMap['procnfs'] = 'filesystem reads'
        self.samplersToCategoriesMap['procnetdev'] = 'Slow network interface usage'

        ds = self.all_samplers
        if model != None:
            ds = ds  + [model]

        self.ldmsInstance = LDMSInstance(
            datasets=ds, path=workload_path)

        # for s in self.all_samplers:
        #     self.ldmsInstance.getMetricSet(s).getDataFrame()['#Time'] = self.ldmsInstance.getMetricSet(s).getDataFrame()['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

        self.workload_path = workload_path
        self.model = model

        self.outputPath = kwargs.get('outputPath',
                                     workload_path)
        self.inputFileNameExtension = kwargs.get('inputFileNameExtension',
                                     '.txt')
        self.metricsFolder = kwargs.get('metricsFolder',
                                     "metrics/" + workload_path)
        self.motifsFolder = kwargs.get('motifsFolder',
                                     "motifs/" + workload_path)




    def select_best_motif_using_hime(self, category, metric, motif_length):
        logger = logging.getLogger(__name__)
        logger.info("start")
        omps = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                 motif_length=motif_length)
        omps.select_best_motif()
        return omps

    def select_next_best_motif_using_hime(self, category, metric, motif_length, motif_start_index):
        logger = logging.getLogger(__name__)
        logger.info("start")
        omps = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                 motif_length=motif_length)
        # omps.select_best_motif()
        # self.tuned_hime_motifs = self.remove_overlaps_from_hime_output()
        omps.tuned_hime_motifs = omps.motifs
        print(omps.tuned_hime_motifs)
        omps.selected_motif = omps.tuned_hime_motifs.iloc[motif_start_index]
        return omps

    def identify_phases(self, metric, category, motif_length):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))
        identifiedPhases = []

        one_metric_phase_set = self.select_best_motif_using_hime(category, metric, motif_length)
        one_metric_phase_set.build_query_from_selected_motif()
        one_metric_phase_set.search_for_similar_patterns_in()



        return one_metric_phase_set

    def identify_phases_using_snippets(self, metric, snippet_index, motif_length, fraction):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))
        identifiedPhases = []

        category = self.metricsToCategoriesMap[metric]


        one_metric_phase_set = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                 motif_length=motif_length, loadData=False)

        # one_metric_phase_set.selected_motif = [snippet_index]

        one_metric_phase_set.real_motif_length = motif_length

        one_metric_phase_set.query = one_metric_phase_set.metric_ts[snippet_index:snippet_index+motif_length]


        one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = stamp(one_metric_phase_set.metric_ts, one_metric_phase_set.query, one_metric_phase_set.real_motif_length)
        one_metric_phase_set.select_best_motifs_from_stamp()

        # one_metric_phase_set.selected_phase_set = one_metric_phase_set.selected_motifs

        selected_motifs_interval_tree = get_interval_tree_motifList(one_metric_phase_set.sampler_df,
                                                                    one_metric_phase_set.selected_motifs,
                                                                    one_metric_phase_set.real_motif_length)

        self.plot_motif_with_time(one_metric_phase_set, selected_motifs_interval_tree, given_name='snippet_' + metric + "_" + str(motif_length) + "_" + str(snippet_index) + "_" + str(fraction) )
        # gc.collect()

        # plot_motifs_complement(self, [one_metric_phase_set], 'snippet_' + metric + "_" + str(motif_length))

        return one_metric_phase_set

    def identify_phases_and_plot_long_run(self, metric, motif_length):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))
        identifiedPhases = []
        category = self.metricsToCategoriesMap[metric]

        one_metric_phase_set = self.select_best_motif_using_hime(category, metric, motif_length)
        one_metric_phase_set.build_query_from_selected_motif()
        one_metric_phase_set.search_for_similar_patterns_in()

        selected_motifs_interval_tree = get_interval_tree_motifList(one_metric_phase_set.sampler_df,
                                                                    one_metric_phase_set.selected_motifs,
                                                                    one_metric_phase_set.real_motif_length)

        self.plot_motif_with_time(one_metric_phase_set, selected_motifs_interval_tree, given_name='hime_' + metric + "_" + str(motif_length) )
        return one_metric_phase_set

    def self_join_and_plot_long_run(self, metric, motif_length):
        category = self.metricsToCategoriesMap[metric]
        self.sampler_df = self.get_sampler_for_category(category)
         # = self.sampler_df.shape[0]

        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))
        identifiedPhases = []

        one_metric_phase_set = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                 motif_length=motif_length, loadData=False)

        one_metric_phase_set.real_motif_length = motif_length

        one_metric_phase_set.query = one_metric_phase_set.metric_ts

        one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index =  matrixProfile.stomp(one_metric_phase_set.metric_ts, one_metric_phase_set.real_motif_length)

        # new_matrix_profile =  STAMP(one_metric_phase_set.metric_ts, query_series=None, subsequence_length=one_metric_phase_set.real_motif_length, max_time=10, verbose=True, parallel=True)

        # ['Matrix_Profile_Index', 'Matrix_Profile']
        # one_metric_phase_set.matrix_profile = new_matrix_profile['Matrix_Profile']
        # one_metric_phase_set.nearest_neighbour_index = new_matrix_profile['Matrix_Profile_Index']

        # one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = stamp(
        #     one_metric_phase_set.metric_ts, one_metric_phase_set.metric_ts, one_metric_phase_set.real_motif_length)

        one_metric_phase_set.select_best_motifs_from_stamp()

        print(one_metric_phase_set.matrix_profile)

        # dataset = pd.DataFrame({'Pab': one_metric_phase_set.matrix_profile})
        # one_metric_phase_set.selected_motifs = list(dataset.sort_values(['Pab']).index.values)

        print(one_metric_phase_set.selected_motifs)

        selected_motifs_interval_tree = get_interval_tree_motifList(one_metric_phase_set.sampler_df,
                                                                    one_metric_phase_set.selected_motifs,
                                                                    one_metric_phase_set.real_motif_length)


        self.plot_motif_with_time(one_metric_phase_set, selected_motifs_interval_tree,
                              given_name='mp_' + metric + "_" + str(motif_length))

    def single_mp_create_self_join(self, metric, motif_length, category):
        # print("metric={}, motif_length={}".format(metric, motif_length))
        one_metric_phase_set = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                                        motif_length=motif_length, loadData=False)
        one_metric_phase_set.real_motif_length = motif_length
        one_metric_phase_set.query = one_metric_phase_set.metric_ts
        one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = matrixProfile.stomp(one_metric_phase_set.metric_ts,
                                                                                                                one_metric_phase_set.real_motif_length)
        one_metric_phase_set.select_best_motifs_from_stamp()
        # print("metric={}, motif_length={} Done!".format(metric, motif_length))
        return one_metric_phase_set

    def parallel_mp_create_self_join(self, metrics, motif_length, category):
        logger.info("motif_length={}".format(motif_length))
        args = []
        pool = multiprocessing.Pool(4)
        for metric in metrics:
            args.append(( metric, motif_length, category))

        results = pool.starmap(self.single_mp_create_self_join, args)
        pool.close()
        pool.join()
        return results

    def pd_self_join_and_plot_mp_side_by_side(self, metrics, category, name, motif_length):

        self.sampler_df = self.get_sampler_for_category(category)
         # = self.sampler_df.shape[0]

        logger = logging.getLogger(__name__)
        logger.info("name={}, motif_length={}".format(name, motif_length))
        one_metric_phase_set_map = {}
        selected_motifs_interval_tree_map = {}

        results = self.parallel_mp_create_self_join(metrics, motif_length, category)

        for m,r in zip(metrics, results):
            one_metric_phase_set_map[m] = r

        self.plot_mp_side_by_side(one_metric_phase_set_map,
                              given_name='mp_procstat_side_by_side_' + name + "_" + str(motif_length))

    def multiple_data_set_parallel_mp_create_self_join(self, pis, metric, motif_length, category):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))

        # pool = multiprocessing.Pool(4)

        results = {}
        for key in pis.keys():
            logger.info("key={}".format(key))
            pi = pis.get(key)
            # results[key] = pool.apply(pi.single_mp_create_self_join, args=(metric, motif_length, category))
            results[key] = pi.single_mp_create_self_join(metric, motif_length, category)
        # pool.close()
        # pool.join()
        return results

    def multiple_data_set_pd_self_join_and_plot_mp_side_by_side(self, pis, metric, category, name, motif_length):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))

        one_metric_phase_set_map = self.multiple_data_set_parallel_mp_create_self_join(pis, metric, motif_length, category)

        self.plot_mp_side_by_side(one_metric_phase_set_map,
                              given_name='mp_multiple_runs_side_by_side_' + name + "_" + str(motif_length),
                                  path='D:/ac/PhD/Research/data/pd/compare-10runs/mpi-plots-rate/')


    def multiple_data_set_pd_other_join_and_plot_mp_side_by_side(self, pis, metric, pi_to_join, metric_to_join, category, name, motif_length):
        logger = logging.getLogger(__name__)
        # logger.info("metric={}, motif_length={}".format(metric, motif_length))
        print("metric={}, motif_length={}".format(metric, motif_length))

        OneMetricPhaseSet_metric_to_join = OneMetricPhaseSet(name=metric_to_join, phase_identifier=pi_to_join,
                                                             category=category, metric=metric_to_join,
                                                             motif_length=motif_length, loadData=False)

        results = {}
        for key in pis.keys():
            logger.info("key={}".format(key))
            pi = pis.get(key)
            one_metric_phase_set = OneMetricPhaseSet(name=metric, phase_identifier=pi, category=category, metric=metric,
                                                        motif_length=motif_length, loadData=False)
            one_metric_phase_set.real_motif_length = motif_length
            one_metric_phase_set.query = one_metric_phase_set.metric_ts

            one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = matrixProfile._matrixProfile(OneMetricPhaseSet_metric_to_join.metric_ts, one_metric_phase_set.real_motif_length, order.linearOrder, distanceProfile.massDistanceProfile, one_metric_phase_set.metric_ts)

            # one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = matrixProfile.stomp(
            #     one_metric_phase_set.metric_ts,
            #     one_metric_phase_set.real_motif_length, OneMetricPhaseSet_metric_to_join.metric_ts)
            one_metric_phase_set.select_best_motifs_from_stamp()
            results[key] = one_metric_phase_set

        print("metric={}, motif_length={} Done!".format(metric, motif_length))

        self.plot_mp_side_by_side(results,
                              given_name='mp_multiple_runs_diff_side_by_side_' + name + "_" + str(motif_length),
                                  path='D:/ac/PhD/Research/data/pd/compare-10runs/diff-rev-mpi-rate-plots/')


        return results



    def single_mp_create(self, logger, metric, metric_to_join, motif_length, category):
        print("metric={}, motif_length={}".format(metric, motif_length))
        OneMetricPhaseSet_metric_to_join = OneMetricPhaseSet(name=metric_to_join, phase_identifier=self, category=category, metric=metric_to_join,
                                                        motif_length=motif_length, loadData=False)
        one_metric_phase_set = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                                        motif_length=motif_length, loadData=False)
        one_metric_phase_set.real_motif_length = motif_length
        one_metric_phase_set.query = one_metric_phase_set.metric_ts
        one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = matrixProfile.stomp(one_metric_phase_set.metric_ts,
                                                                                                                one_metric_phase_set.real_motif_length, OneMetricPhaseSet_metric_to_join.metric_ts)
        one_metric_phase_set.select_best_motifs_from_stamp()
        print("metric={}, motif_length={} Done!".format(metric, motif_length))
        return one_metric_phase_set

    def parallel_mp_create(self, metrics, metric_to_join, motif_length, category):
        logger.info("metric_to_join={}, motif_length={}".format(metric_to_join, motif_length))
        args = []
        pool = multiprocessing.Pool(4)
        for metric in metrics:
            args.append((logger, metric, metric_to_join, motif_length, category))

        results = pool.starmap(self.single_mp_create, args)
        pool.close()
        pool.join()
        return results

    def pd_join_different_metrics_and_plot_mp_side_by_side(self, metric_to_join, metrics, category, name, motif_length):
        self.sampler_df = self.get_sampler_for_category(category)
         # = self.sampler_df.shape[0]

        logger = logging.getLogger(__name__)
        logger.info("metric_to_join={}, motif_length={}".format(metric_to_join, motif_length))
        one_metric_phase_set_map = {}
        selected_motifs_interval_tree_map = {}

        results = self.parallel_mp_create(metrics, metric_to_join, motif_length, category)

        for m,r in zip(metrics, results):
            one_metric_phase_set_map[m] = r

        # OneMetricPhaseSet_metric_to_join = OneMetricPhaseSet(name=metric_to_join, phase_identifier=self, category=category, metric=metric_to_join,
        #                                                 motif_length=motif_length, loadData=False)
        # for m in metrics:
        #     logger.info("metric={}, motif_length={}".format(m, motif_length))
        #     one_metric_phase_set_map[m] = OneMetricPhaseSet(name=m, phase_identifier=self, category=category, metric=m,
        #                          motif_length=motif_length, loadData=False)
        #     one_metric_phase_set_map[m].real_motif_length = motif_length
        #     one_metric_phase_set_map[m].query = one_metric_phase_set_map[m].metric_ts
        #     one_metric_phase_set_map[m].matrix_profile, one_metric_phase_set_map[m].nearest_neighbour_index =  matrixProfile.stomp(one_metric_phase_set_map[m].metric_ts, one_metric_phase_set_map[m].real_motif_length, OneMetricPhaseSet_metric_to_join.metric_ts)
        #     one_metric_phase_set_map[m].select_best_motifs_from_stamp()
        #     selected_motifs_interval_tree_map[m] = get_interval_tree_motifList(one_metric_phase_set_map[m].sampler_df,
        #                                                                 one_metric_phase_set_map[m].selected_motifs,
        #                                                                 one_metric_phase_set_map[m].real_motif_length)

        self.plot_mp_side_by_side(one_metric_phase_set_map,
                              given_name='mp_diff_metrics_rev_side_by_side_' + name + "_" + str(motif_length))


    def find_row_number(self, df, time_ge):
        counter = 0
        for i, row in df.iterrows():

            if row['#Time'] >= time_ge:
                return counter
            counter = counter + 1
        return counter



    def identify_phases_in_large_phase(self, metric, category, motif_length, large_phase_set,range_real_motif_length, range_sampler_df):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))
        identifiedPhases = []


        one_metric_phase_set = self.select_best_motif_using_hime(category, metric, motif_length)

        # one_metric_phase_set.build_query_from_selected_motif()

        all_starts = []
        all_stops = []

        for phase_start in large_phase_set:
            current_start = self.find_row_number(one_metric_phase_set.sampler_df,
                                                 range_sampler_df.iloc[phase_start]['#Time'])
            current_stop = current_start + range_real_motif_length
            all_starts.append(current_start)
            all_stops.append(current_stop)

        # one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

        one_metric_phase_set.range_load_metrics_motifs_range(all_starts, all_stops, transform='rate')
        one_metric_phase_set.select_best_motif_in_range()
        one_metric_phase_set.build_query_from_selected_motif_in_range()
        counter = 0
        for i in range(len(one_metric_phase_set.selected_motif_range)):
            print(i)
            current_start = all_starts[i]
            current_stop = all_stops[i]
            searchRange = one_metric_phase_set.metric_ts_range[i]#[current_start:current_stop]
            one_metric_phase_set.range_search_for_similar_patterns_in_range(searchRange, one_metric_phase_set.query_in_range[i], one_metric_phase_set.real_motif_length_in_range[i])
            if counter == 0:
                all_phases = []
            for sm in one_metric_phase_set.selected_motifs:
                all_phases.append(current_start + sm)
            counter = counter + 1
            one_metric_phase_set.real_motif_length = one_metric_phase_set.real_motif_length_in_range[i]
            print(one_metric_phase_set.real_motif_length)
        counter = 0

        # for phase_start in large_phase_set:
        #
        #     current_start = self.find_row_number(one_metric_phase_set.sampler_df,
        #                                          range_sampler_df.iloc[phase_start]['#Time'])
        #     current_stop = current_start + range_real_motif_length
        #
        #
        #
        #     current_start = self.find_row_number(one_metric_phase_set.sampler_df,range_sampler_df.iloc[phase_start]['#Time'])
        #     searchRange = one_metric_phase_set.metric_ts[current_start:current_start + range_real_motif_length]
        #     logger.info("counter={}, phase_start:phase_stop={}:{}".format(counter, current_start,current_start + range_real_motif_length))
        #     one_metric_phase_set.search_for_similar_patterns_in_range(searchRange)
        #     if counter == 0:
        #         all_phases = []
        #     for sm in one_metric_phase_set.selected_motifs:
        #         all_phases.append(current_start+sm)
        #     counter = counter + 1
        one_metric_phase_set.selected_motifs = sorted(all_phases)

        return one_metric_phase_set

    def identify_phases_no_remove(self, metric, category, motif_length):
        logger = logging.getLogger(__name__)
        logger.info("metric={}, motif_length={}".format(metric, motif_length))
        identifiedPhases = []
        motif_start_index = 0
        while True:
            try:
                # one_metric_phase_set = self.select_next_best_motif_using_hime(category, metric, motif_length, motif_start_index)

                omps = OneMetricPhaseSet(name=metric, phase_identifier=self, category=category, metric=metric,
                                         motif_length=motif_length)
                # omps.select_best_motif()
                # self.tuned_hime_motifs = self.remove_overlaps_from_hime_output()
                omps.tuned_hime_motifs = omps.motifs
                if motif_start_index == 0:
                    print(omps.tuned_hime_motifs)

                omps.selected_motif = omps.tuned_hime_motifs.iloc[motif_start_index]
                one_metric_phase_set = omps



                one_metric_phase_set.build_query_from_selected_motif()
                print("select " + str(motif_start_index) + " distance: " + str(one_metric_phase_set.selected_motif[
                    one_metric_phase_set.ditanceColIndexInHimeOutput]))
                # self.real_motif_length = int(self.selected_motif[self.motifLengthColIndexInHimeOutput])

                if motif_start_index == 10:
                    break

                # if one_metric_phase_set.selected_motif[one_metric_phase_set.ditanceColIndexInHimeOutput] > 10:
                #     print("more than 10!")
                #     break

                one_metric_phase_set.search_for_similar_patterns_in()

                test_calculate_Likelihood(pd.DataFrame(one_metric_phase_set.metric_ts),
                                          one_metric_phase_set.selected_motifs,
                                          one_metric_phase_set.real_motif_length)
                one_metric_phase_set.selected_phase_set = test_chi2IsUniform(one_metric_phase_set)
                plot_motifs_complement(self, [one_metric_phase_set], 'enum_' + metric + "_" + str(motif_length) + "_" + str(motif_start_index))
                motif_start_index = motif_start_index + 1
            except Exception as e:
                logger.error(e)
                break
        return one_metric_phase_set

    def calculateLikelihood(self,data, breaks, motif_length, lamb=1e-1):
        ll = 0
        for i in range(len(breaks)):
            tempData = np.float64(data.iloc[breaks[i]:breaks[i]+motif_length, :])
            m, n = tempData.shape
            empCov = np.cov(tempData.T, bias=True)
            ll = ll - (m * np.linalg.slogdet(empCov + float(lamb) * np.identity(n) / m)[1] - float(lamb) * np.trace(
                np.linalg.inv(empCov + float(lamb) * np.identity(n) / m)))
        return ll


    def enumerate_motif_length_range_all(self, metric, category, motif_length_range=range(50,800,50)):
        logger = logging.getLogger(__name__)
        logger.info("metric={}".format(metric))

        for motif_length in motif_length_range:

            try:
                one_metric_phase_set = self.identify_phases_no_remove(metric, category, motif_length)
                logger.info("FINISH for motif_length={}".format(motif_length))

                # plot_motifs_complement(pi, [memory_set], 'memory_set')
                gc.collect()

            except Exception as e:
                logger.error(e)
                continue


    def enumerate_motif_length_range(self, metric, category, motif_length_range=range(50,800,50)):
        logger = logging.getLogger(__name__)
        logger.info("metric={}".format(metric))

        for motif_length in motif_length_range:

            try:
                one_metric_phase_set = self.identify_phases(metric, category, motif_length)
                test_calculate_Likelihood(pd.DataFrame(one_metric_phase_set.metric_ts),
                                                 one_metric_phase_set.selected_motifs,
                                                 one_metric_phase_set.real_motif_length)
                one_metric_phase_set.selected_phase_set = test_chi2IsUniform(one_metric_phase_set)
                # score1 = self.calculateLikelihood(pd.DataFrame(one_metric_phase_set.metric_ts),
                #                                  one_metric_phase_set.selected_motifs,
                #                                  one_metric_phase_set.real_motif_length)
                # score2 = self.calculateLikelihood(pd.DataFrame(one_metric_phase_set.metric_ts),
                #                                   one_metric_phase_set.selected_phase_set,
                #                                  one_metric_phase_set.real_motif_length)
                plot_motifs_complement(self, [one_metric_phase_set], 'enum_' + metric + "_" + str(motif_length))
                logger.info("FINISH for motif_length={}".format(motif_length))

                # plot_motifs_complement(pi, [memory_set], 'memory_set')
                gc.collect()

            except Exception as e:
                logger.error(e)
                continue




    def get_sampler_for_category(self, category):
        return self.ldmsInstance.getMetricSet(self.categoriesToSamplerMap[category]).getDataFrame()

    def plot_selected_motifs(self,ax, Ta, df, selectedMotifs, m, metric):
        # print(df)
        ax.set_title(metric)
        ax.plot(df['#Time'].values, Ta, linestyle='--', alpha=0.5)

        counter = 1
        colors = ['g', 'y', 'b', 'r', 'c', 'm', 'k']

        for item in selectedMotifs:
            if item + m - 1 > len(Ta):
                continue
            ax.plot(df['#Time'].iloc[range(item, item + m - 1)], Ta[item:item + m - 1], c=colors[counter % len(colors)],
                    label='Motif #' + str(counter))
            counter = counter + 1

    def plot_one_selected_motifs(self, ax, Ta, df, selectedMotifs, m, metric):
        ax.set_title("The best motif")
        # ax.plot(df['#Time'].values, Ta, linestyle='--', alpha=0.5)

        counter = 1
        colors = ['g', 'y', 'b', 'r', 'c', 'm', 'k']

        for item in selectedMotifs:
            ax.plot(df['#Time'].iloc[range(item, item + m)], Ta, c=colors[counter % len(colors)],
                    label='Motif #' + str(counter))
            counter = counter + 1

    def indicate_discords(self, ax, df, profile, ex_zone, k=2):
        anoms = discords(profile, ex_zone, k)
        df.iloc[anoms]
        for i in anoms:
            ax.plot(df['#Time'].iloc[i],profile[i], marker = 'X', c = 'r', ms = 20)

    def indicate_discords_metric(self, ax, df, Ta, profile, ex_zone, anoms = None, k=2):
        if anoms == None:
            anoms = discords(profile, ex_zone, k)
        for i in anoms:
            ax.plot(df['#Time'].iloc[i], Ta[i], marker='X', c='r', ms=20)


    def plot_matrix_profile(self, ax, df, values):
        ax.set_title('Matrix Profile')
        ax.plot(df['#Time'].iloc[range(0, len(values))], values, '#ff5722')
        # ax.plot(df['#Time'].iloc[np.argmax(values)], np.max(values), marker='x', c='r', ms=10)
        ax.plot(df['#Time'].iloc[np.argmin(values)], np.min(values), marker='^', c='g', ms=10)



    def plot_mp_side_with_anomaly(self, one_metric_phase_set, path, extra_metric_one_metric_phase_set, given_name = None):
        logger = logging.getLogger(__name__)
        logger.info("start")

        date_fmt = '%M:%S'
        xfmt = md.DateFormatter(date_fmt)

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        figIndex = 0
        matrixProfileIndex = 0
        metricSetIndex = 1

        # one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df['#Time'] - \
        #                                            one_metric_phase_set.sampler_df[
        #                                                            '#Time'].min()
        # extra_metric_one_metric_phase_set.sampler_df['#Time'] = extra_metric_one_metric_phase_set.sampler_df['#Time'] - \
        #                                                         extra_metric_one_metric_phase_set.sampler_df[
        #                                                            '#Time'].min()
        # try:
        #     one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df[
        #         '#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
        #     extra_metric_one_metric_phase_set.sampler_df['#Time'] = extra_metric_one_metric_phase_set.sampler_df[
        #         '#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
        # except TypeError as e:
        #     print(e)

        mp_ax = axs[matrixProfileIndex]
        mp_ax.xaxis.set_major_locator(md.SecondLocator(interval=60))
        mp_ax.xaxis.set_minor_locator(md.SecondLocator(interval=6))
        mp_ax.xaxis.set_major_formatter(xfmt)
        mp_ax.set_xlim(one_metric_phase_set.sampler_df['#Time'].min(),
                                            one_metric_phase_set.sampler_df['#Time'].max())
        self.plot_matrix_profile(mp_ax, one_metric_phase_set.sampler_df,
                                 one_metric_phase_set.matrix_profile)

        self.indicate_discords(mp_ax, one_metric_phase_set.sampler_df, one_metric_phase_set.matrix_profile, one_metric_phase_set.motif_length, k=2)


        metric_ax = axs[metricSetIndex]

        metric_ax.xaxis.set_major_formatter(xfmt)
        metric_ax.xaxis.set_major_locator(md.SecondLocator(interval=30))
        metric_ax.xaxis.set_minor_locator(md.SecondLocator(interval=6))
        metric_ax.set_xlim(one_metric_phase_set.sampler_df['#Time'].min(),
                                            one_metric_phase_set.sampler_df['#Time'].max())
        self.plot_selected_motifs(metric_ax, one_metric_phase_set.metric_ts,
                                  one_metric_phase_set.sampler_df, one_metric_phase_set.selected_motifs,
                                  one_metric_phase_set.real_motif_length, one_metric_phase_set.metric)

        self.indicate_discords_metric(metric_ax, one_metric_phase_set.sampler_df, one_metric_phase_set.metric_ts,
                               one_metric_phase_set.matrix_profile, one_metric_phase_set.motif_length, k=2)

        metric_ax2 = axs[2]

        metric_ax2.xaxis.set_major_formatter(xfmt)
        metric_ax2.xaxis.set_major_locator(md.SecondLocator(interval=30))
        metric_ax2.xaxis.set_minor_locator(md.SecondLocator(interval=6))
        metric_ax2.set_xlim(extra_metric_one_metric_phase_set.sampler_df['#Time'].min(),
                           extra_metric_one_metric_phase_set.sampler_df['#Time'].max())
        self.plot_selected_motifs(metric_ax2, extra_metric_one_metric_phase_set.metric_ts,
                                  extra_metric_one_metric_phase_set.sampler_df, extra_metric_one_metric_phase_set.selected_motifs,
                                  extra_metric_one_metric_phase_set.real_motif_length, extra_metric_one_metric_phase_set.metric)

        anoms = discords(one_metric_phase_set.matrix_profile, one_metric_phase_set.motif_length, k=2)

        extra_metric_anom_index = []
        for a in one_metric_phase_set.sampler_df['#Time'].iloc[anoms]:
            less = extra_metric_one_metric_phase_set.sampler_df[extra_metric_one_metric_phase_set.sampler_df['#Time'] < a]
            extra_metric_anom_index.append(len(less))

        self.indicate_discords_metric(metric_ax2, extra_metric_one_metric_phase_set.sampler_df, extra_metric_one_metric_phase_set.metric_ts,
                                      one_metric_phase_set.matrix_profile, extra_metric_one_metric_phase_set.motif_length, extra_metric_anom_index, k=2)


        fig.set_size_inches(h=11.176, w=15.232)
        fig.autofmt_xdate()

        if given_name == None:
            name = one_metric_phase_set.metric[:-4]
        else:
            name = given_name
        format = '.png'

        print('saving figure: ' + path + name + format)
        fig.savefig(path + name + format, dpi=600)

        fig.clf()
        plt.clf()
        plt.close()


    def plot_mp_side_by_side(self, one_metric_phase_set_map, majorLocator=md.SecondLocator(100), minorLocator=md.SecondLocator(25), given_name=None, path = ''):
        logger = logging.getLogger(__name__)
        logger.info("start")

        date_fmt = '%M:%S'
        xfmt = md.DateFormatter(date_fmt)

        fig, axs = plt.subplots(nrows=len(one_metric_phase_set_map), ncols=2, sharex=True)

        matrixProfileIndex = 0
        selectedMotifsIndex = 1
        for metric in one_metric_phase_set_map.keys():
            print(metric + ' - ' + one_metric_phase_set_map[metric].sampler_name + ' - ' +  one_metric_phase_set_map[metric].metric)

            # one_metric_phase_set_map[metric].sampler_df['#Time'] = md.(one_metric_phase_set_map[metric].sampler_df['#Time'])
            # final_df[xAxis] = final_df[xAxis] - final_df[xAxis].min()
            # print(one_metric_phase_set_map[metric].sampler_df['#Time'])
            one_metric_phase_set_map[metric].sampler_df['#Time'] = one_metric_phase_set_map[metric].sampler_df['#Time'] - one_metric_phase_set_map[metric].sampler_df['#Time'].min()

            # one_metric_phase_set_map[metric].sampler_df['#Time'] = md.epoch2num(one_metric_phase_set_map[metric].sampler_df['#Time'])
            try:
                one_metric_phase_set_map[metric].sampler_df['#Time'] = one_metric_phase_set_map[metric].sampler_df['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
            except TypeError as e:
                print(e)



            max_end = one_metric_phase_set_map[metric].sampler_df['#Time'].max()
            min_start = one_metric_phase_set_map[metric].sampler_df['#Time'].min()
            # break
        logger.info("axs={}".format(len(axs)))
        # print(min_start)
        # print(max_end)
        for metric in one_metric_phase_set_map.keys():
            logger.info(metric)

            # md.SecondLocator(interval=60)

            one_metric_phase_set = one_metric_phase_set_map[metric]
            axs[matrixProfileIndex][0].xaxis.set_major_locator(md.SecondLocator(interval=60))
            axs[matrixProfileIndex][0].xaxis.set_minor_locator(md.SecondLocator(interval=6))
            axs[matrixProfileIndex][0].xaxis.set_major_formatter(xfmt)
            axs[matrixProfileIndex][0].set_xlim(one_metric_phase_set.sampler_df['#Time'].min(),
                                                one_metric_phase_set.sampler_df['#Time'].max())
            self.plot_matrix_profile(axs[matrixProfileIndex][0], one_metric_phase_set.sampler_df, one_metric_phase_set.matrix_profile)

            # logger.info("mp locators")
            # logger.info(len(axs[matrixProfileIndex][0].xaxis.get_major_locator()()))
            # logger.info(len(axs[matrixProfileIndex][0].xaxis.get_minor_locator()()))

            axs[matrixProfileIndex][1].xaxis.set_major_formatter(xfmt)
            axs[matrixProfileIndex][1].xaxis.set_major_locator(md.SecondLocator(interval=30))
            axs[matrixProfileIndex][1].xaxis.set_minor_locator(md.SecondLocator(interval=6))
            axs[matrixProfileIndex][1].set_xlim(one_metric_phase_set.sampler_df['#Time'].min(), one_metric_phase_set.sampler_df['#Time'].max())
            self.plot_selected_motifs(axs[matrixProfileIndex][1], one_metric_phase_set.metric_ts, one_metric_phase_set.sampler_df, one_metric_phase_set.selected_motifs, one_metric_phase_set.real_motif_length, metric + '_' + one_metric_phase_set.metric)
            # logger.info("motif locators")
            # logger.info(len(axs[matrixProfileIndex][1].xaxis.get_major_locator()()))
            # logger.info(len(axs[matrixProfileIndex][1].xaxis.get_minor_locator()()))

            matrixProfileIndex = matrixProfileIndex + 1
            # selectedMotifsIndex = selectedMotifsIndex + 2

            max_end = max(one_metric_phase_set.sampler_df['#Time'].max(), max_end)
            min_start = min(one_metric_phase_set.sampler_df['#Time'].min(), min_start)


        ax = axs[matrixProfileIndex - 1][1]
        ax.xaxis.set_major_locator(md.SecondLocator(interval=30))
        ax.xaxis.set_minor_locator(md.SecondLocator(interval=6))
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_xlim(min_start, max_end)


        fig.set_size_inches(h=11.176, w=15.232)
        fig.autofmt_xdate()
        # logger.info('saving figure')

        if given_name == None:
            name = one_metric_phase_set.metric[:-4]
        else:
            name = given_name
        format = '.png'

        print('saving figure: ' + path + name + format)
        fig.savefig(path + name + format, dpi=600)

        fig.clf()
        plt.clf()
        plt.close()


        # ax.set_ylim(y0, y0+height)



    def plot_motif_with_time(self, one_metric_phase_set,selected_motifs_interval_tree, given_name=None):
        logger = logging.getLogger(__name__)
        logger.info("start")
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
        axs[selectedMotifsIndex].xaxis.set_major_locator(md.MinuteLocator(interval=100))
        axs[selectedMotifsIndex].xaxis.set_minor_locator(md.MinuteLocator(interval=25))
        logger.info("plot_selected_motifs")
        self.plot_selected_motifs(axs[selectedMotifsIndex], one_metric_phase_set.metric_ts, one_metric_phase_set.sampler_df, one_metric_phase_set.selected_motifs, one_metric_phase_set.real_motif_length, one_metric_phase_set.metric)

        axs[oneMotifIndex].xaxis.set_major_formatter(xfmt)
        axs[oneMotifIndex].xaxis.set_major_locator(md.MinuteLocator(interval=100))
        axs[oneMotifIndex].xaxis.set_minor_locator(md.MinuteLocator(interval=25))
        logger.info("plot_one_selected_motifs")
        self.plot_one_selected_motifs(axs[oneMotifIndex], one_metric_phase_set.metric_ts[one_metric_phase_set.selected_motifs[0]:one_metric_phase_set.selected_motifs[0]+one_metric_phase_set.real_motif_length], one_metric_phase_set.sampler_df, [one_metric_phase_set.selected_motifs[0]], one_metric_phase_set.real_motif_length, one_metric_phase_set.metric)

        axs[matrixProfileIndex].xaxis.set_major_locator(md.MinuteLocator(interval=100))
        axs[matrixProfileIndex].xaxis.set_minor_locator(md.MinuteLocator(interval=25))
        axs[matrixProfileIndex].xaxis.set_major_formatter(xfmt)
        logger.info("plot_matrix_profile")
        self.plot_matrix_profile(axs[matrixProfileIndex], one_metric_phase_set.sampler_df, one_metric_phase_set.matrix_profile)

        axs[modelIndex].xaxis.set_major_locator(md.MinuteLocator(interval=100))
        axs[modelIndex].xaxis.set_minor_locator(md.MinuteLocator(interval=25))
        axs[modelIndex].xaxis.set_major_formatter(xfmt)
        axs[modelIndex].set_title("model motifs")
        if self.model != None:
            logger.info("plot_model")
            plot_model(self.ldmsInstance.getMetricSet(self.model).getDataFrame(), axs[modelIndex], one_metric_phase_set.sampler_df['#Time'].min(), one_metric_phase_set.sampler_df['#Time'].max())

        axs[currentMotifsIntervalPlot].xaxis.set_major_locator(md.MinuteLocator(interval=100))
        axs[currentMotifsIntervalPlot].xaxis.set_minor_locator(md.MinuteLocator(interval=25))
        axs[currentMotifsIntervalPlot].xaxis.set_major_formatter(xfmt)
        axs[currentMotifsIntervalPlot].set_title("current motifs")
        logger.info("plot_interval")
        plot_interval(find_shared_period([selected_motifs_interval_tree]), axs[currentMotifsIntervalPlot], one_metric_phase_set.sampler_df['#Time'].min(),
                      one_metric_phase_set.sampler_df['#Time'].max())

        # axs[sharedMotifsIntervalPlot].xaxis.set_major_locator(md.SecondLocator(interval=100))
        # axs[sharedMotifsIntervalPlot].xaxis.set_minor_locator(md.SecondLocator(interval=25))
        # axs[sharedMotifsIntervalPlot].xaxis.set_major_formatter(xfmt)
        # axs[sharedMotifsIntervalPlot].set_title("shared motifs")
        #
        # plot_interval(sharedMotifs, axs[sharedMotifsIntervalPlot], df['#Time'].min(), df['#Time'].max())

        fig.set_size_inches(h=11.176, w=15.232)
        fig.autofmt_xdate()
        logger.info('saving figure')

        if given_name == None:
            name = one_metric_phase_set.metric[:-4]
        else:
            name = given_name
        format = '.png'
        path = ''
        logger.info(path + name + format)
        fig.savefig(path + name + format, dpi=600)

        fig.clf()
        plt.clf()
        plt.close()
        # gc.collect()

        # axs[1].set_xlim((0, len(Ta)))
        # fig.autofmt_xdate()
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.show()

    def identify_and_plot_phases(self, metric, category, motif_length):
        one_metric_phase_set = self.identify_phases(metric, category, motif_length)
        selected_motifs_interval_tree = get_interval_tree_motifList(one_metric_phase_set.sampler_df,
                                                                    one_metric_phase_set.selected_motifs,
                                                                    one_metric_phase_set.real_motif_length)
        print(selected_motifs_interval_tree)
        self.plot_motif_with_time(one_metric_phase_set, selected_motifs_interval_tree)
        gc.collect()

        # def calcIBSMDistance(allTimes, phaseSet1, phaseSet2):

        y = calcIBSMDistance(one_metric_phase_set.sampler_df['#Time'], get_highLevel_interval_tree_from_model(self.ldmsInstance.getMetricSet(self.model).getDataFrame()),
                             find_shared_period([selected_motifs_interval_tree]))

        # get_motifs_from_metric('MPI_Issend.calls.1.rate.txt', motifLength, origin_workload, sampler='shm_sampler')
        return one_metric_phase_set.selected_motifs

def plot_all_motifs(pi, metric_phase_sets):
    logger = logging.getLogger(__name__)
    logger.info("start")
    fig, axs = plt.subplots(nrows=len(metric_phase_sets) + 1, sharex=True)
    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)
    index = 0
    for metric_phase_set in metric_phase_sets:
        logger.info(str(index) + ": " + metric_phase_set.metric)
        selected_motifs_interval_tree = get_interval_tree_motifList(metric_phase_set.sampler_df,
                                                                                     metric_phase_set.selected_motifs,
                                                                                     metric_phase_set.real_motif_length)
        axs[index].xaxis.set_major_locator(md.SecondLocator(interval=100))
        axs[index].xaxis.set_minor_locator(md.SecondLocator(interval=25))
        axs[index].xaxis.set_major_formatter(xfmt)
        axs[index].set_title(metric_phase_set.metric)
        # print(selected_motifs_interval_tree)
        plot_interval(find_shared_period([selected_motifs_interval_tree]), axs[index], metric_phase_set.sampler_df['#Time'].min(),
                      metric_phase_set.sampler_df['#Time'].max())
        index = index + 1
    modelIndex = index
    axs[modelIndex].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[modelIndex].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    axs[modelIndex].xaxis.set_major_formatter(xfmt)
    axs[modelIndex].set_title("model motifs")
    plot_model(pi.ldmsInstance.getMetricSet(pi.model).getDataFrame(), axs[modelIndex],
               metric_phase_sets[0].sampler_df['#Time'].min(), metric_phase_sets[0].sampler_df['#Time'].max(), verbose=True)


    fig.set_size_inches(h=11.176, w=15.232)
    fig.autofmt_xdate()

    name = 'all_phases'
    format = '.png'
    path = ''

    logger.info('saving figure' + path + name + format)
    fig.savefig(path + name + format, dpi=600)

    fig.clf()
    plt.clf()
    plt.close()
    gc.collect()

def get_complement_motifs(metric_phase_set):

    complement_motifs_start = []

    complement_motifs_stop = []
    if len(metric_phase_set.selected_phase_set) == 0:
        return complement_motifs_start, complement_motifs_stop

    sorted_list = sorted(metric_phase_set.selected_phase_set)
    if sorted_list[0] == 0:
        sorted_list = sorted_list[1:]
        complement_motifs_start.append(metric_phase_set.real_motif_length + 1)
        # complement_motifs_start = sorted_list[1] - 1
    else:
        complement_motifs_start.append(0)
    for s in sorted_list:
        # print(s)
        complement_motifs_stop.append(s-1)
        complement_motifs_start.append(s + metric_phase_set.real_motif_length + 1)

    complement_motifs_stop.append(len(metric_phase_set.sampler_df['#Time']) - 1)

    return complement_motifs_start, complement_motifs_stop




def plot_motifs_complement(pi, metric_phase_sets, name=None):
    logger = logging.getLogger(__name__)
    logger.info("start")
    fig, axs = plt.subplots(nrows=len(metric_phase_sets) * 3 + 1, sharex=True)
    date_fmt = '%H:%M:%S'
    xfmt = md.DateFormatter(date_fmt)
    index = 0

    for metric_phase_set in metric_phase_sets:

        logger.info(str(index) + ": " + metric_phase_set.metric)

        selected_motifs_interval_tree = get_interval_tree_motifList(metric_phase_set.sampler_df,
                                                                    metric_phase_set.selected_motifs,
                                                                    metric_phase_set.real_motif_length)
        axs[index].xaxis.set_major_locator(md.SecondLocator(interval=100))
        axs[index].xaxis.set_minor_locator(md.SecondLocator(interval=25))
        axs[index].xaxis.set_major_formatter(xfmt)
        axs[index].set_title(metric_phase_set.metric + "_found_motifs")
        # print(selected_motifs_interval_tree)
        plot_interval(find_shared_period([selected_motifs_interval_tree]), axs[index],
                      metric_phase_set.sampler_df['#Time'].min(),
                      metric_phase_set.sampler_df['#Time'].max())
        index = index + 1

        # if (len)
        selected_motifs_interval_tree = get_interval_tree_motifList(metric_phase_set.sampler_df,
                                                                    metric_phase_set.selected_phase_set,
                                                                    metric_phase_set.real_motif_length)
        axs[index].xaxis.set_major_locator(md.SecondLocator(interval=100))
        axs[index].xaxis.set_minor_locator(md.SecondLocator(interval=25))
        axs[index].xaxis.set_major_formatter(xfmt)
        axs[index].set_title(metric_phase_set.metric + "_evenly_distributed_only")
        # print(selected_motifs_interval_tree)
        plot_interval(find_shared_period([selected_motifs_interval_tree]), axs[index],
                      metric_phase_set.sampler_df['#Time'].min(),
                      metric_phase_set.sampler_df['#Time'].max())
        index = index + 1

        complement_motifs_start, complement_motifs_stop = get_complement_motifs(metric_phase_set)
        complement_motifs_interval_tree = get_interval_tree_complement_motifList(metric_phase_set.sampler_df, complement_motifs_start, complement_motifs_stop)

        axs[index].xaxis.set_major_locator(md.SecondLocator(interval=100))
        axs[index].xaxis.set_minor_locator(md.SecondLocator(interval=25))
        axs[index].xaxis.set_major_formatter(xfmt)
        axs[index].set_title(metric_phase_set.metric + "_regions_not_in_evenly_distributed_phases")
        plot_interval(complement_motifs_interval_tree, axs[index],
                      metric_phase_set.sampler_df['#Time'].min(),
                      metric_phase_set.sampler_df['#Time'].max())
        index = index + 1

    modelIndex = index
    axs[modelIndex].xaxis.set_major_locator(md.SecondLocator(interval=100))
    axs[modelIndex].xaxis.set_minor_locator(md.SecondLocator(interval=25))
    axs[modelIndex].xaxis.set_major_formatter(xfmt)
    axs[modelIndex].set_title("model motifs")
    plot_model(pi.ldmsInstance.getMetricSet(pi.model).getDataFrame(), axs[modelIndex],
               metric_phase_sets[0].sampler_df['#Time'].min(), metric_phase_sets[0].sampler_df['#Time'].max())


    fig.set_size_inches(h=11.176, w=15.232)
    fig.autofmt_xdate()

    if name == None:
        name = 'plot_complement'
    format = '.png'
    path = ''

    plt.tight_layout()
    #  fig.subplots_adjust(hspace=4)

    logger.info('saving figure' + path + name + format)
    fig.savefig(path + name + format, dpi=600)

    fig.clf()
    plt.clf()
    plt.close()
    gc.collect()


def test_identify_mpi_calls(pi):
    return pi.identify_and_plot_phases(metric='MPI_Issend.calls.1.rate', category='MPI calls',motif_length=250)

def calculate_Likelihood(data, breaks, motif_length, lamb=1e-1):
    ll = 0
    for i in range(len(breaks)):
        tempData = np.float64(data.iloc[breaks[i]:breaks[i]+motif_length, :])
        m, n = tempData.shape
        empCov = np.cov(tempData.T, bias=True)
        ll = ll - (m * np.linalg.slogdet(empCov + float(lamb) * np.identity(n) / m)[1] - float(lamb) * np.trace(
            np.linalg.inv(empCov + float(lamb) * np.identity(n) / m)))
    return ll

def test_calculate_Likelihood(data, phase_set, motif_length, lamb=1e-5):
    logger = logging.getLogger(__name__)
    current_phase_set = phase_set
    score = calculate_Likelihood(data, current_phase_set, motif_length, lamb)
    logger.info('regular: ' + str(score))
    current_phase_set = sorted(current_phase_set)
    logger.info(current_phase_set)
    modified_phase_set = current_phase_set
    num_of_removed_intervals = 1
    left = True

    while(len(modified_phase_set) > 0):
        logger.info('len=' + str(len(modified_phase_set)) + ', left? ' + str(not left))
        score = calculate_Likelihood(data, modified_phase_set, motif_length, lamb)
        logger.info('score: ' + str(score))

        if left:
            modified_phase_set = current_phase_set[num_of_removed_intervals:]
            left = False
        else:
            modified_phase_set = current_phase_set[:(-1) * num_of_removed_intervals]
            left = True
            num_of_removed_intervals = num_of_removed_intervals + 1

def chi2IsUniform(dataSet, significance=0.05):
    logger = logging.getLogger(__name__)
    chisq, pvalue = chisquare(dataSet)
    logger.info("chisq={}, pvalue={}".format(chisq,pvalue))
    return pvalue > significance

def test_chi2IsUniform(phase_set):
    logger = logging.getLogger(__name__)
    current_phase_set = phase_set.selected_motifs
    chisq = False
    chisq = chi2IsUniform(current_phase_set)

    logger.info('regular: ' + str(chisq))
    current_phase_set = sorted(current_phase_set)
    logger.info(current_phase_set)
    modified_phase_set = current_phase_set
    num_of_removed_intervals = 1
    left = True

    while(len(modified_phase_set) > 0 and not chisq):
        logger.info('len=' + str(len(modified_phase_set)) + ', left? ' + str(not left))
        chisq = chi2IsUniform(np.ediff1d(sorted(modified_phase_set)))
        logger.info('sorted diff: ' + str(chisq))

        if chisq:
            break

        if left:
            modified_phase_set = current_phase_set[num_of_removed_intervals:]
            left = False
        else:
            modified_phase_set = current_phase_set[:(-1) * num_of_removed_intervals]
            left = True
            num_of_removed_intervals = num_of_removed_intervals + 1
    logger.info(modified_phase_set)
    logger.info('********************************************************************************************************************')

    return modified_phase_set


def test_identify_using_multiple_metrics(pi):
    logger = logging.getLogger(__name__)
    logger.info("start")

    fs_read_set = pi.identify_phases(metric='read.rate', category='filesystem reads',motif_length=250)
    test_chi2IsUniform(fs_read_set)
    fs_write_set = pi.identify_phases(metric='write.rate', category='filesystem reads', motif_length=250)
    test_chi2IsUniform(fs_write_set)
    memory_set = pi.identify_phases(metric='Dirty', category='Memory usage', motif_length=250)
    test_chi2IsUniform(memory_set)
    network_set = pi.identify_phases(metric='tx_bytes#eth0.rate', category='Slow network interface usage', motif_length=200)
    test_chi2IsUniform(network_set)
    mpi_phase_set = pi.identify_phases(metric='MPI_Issend.calls.1.rate', category='MPI calls', motif_length=250)
    test_chi2IsUniform(mpi_phase_set)
    cpu_usage_set = pi.identify_phases(metric='Sys.rate', category='Cpu usage',
                                        motif_length=250)
    test_chi2IsUniform(cpu_usage_set)

    plot_all_motifs(pi, [fs_read_set,fs_write_set,memory_set, network_set, mpi_phase_set, cpu_usage_set])

def test_identify_hirearchical(pi):
    logger = logging.getLogger(__name__)
    logger.info("start")

    # fs_read_set = pi.identify_phases(metric='read.rate', category='filesystem reads',motif_length=250)
    # test_chi2IsUniform(fs_read_set)
    # fs_write_set = pi.identify_phases(metric='write.rate', category='filesystem reads', motif_length=250)
    # test_chi2IsUniform(fs_write_set)
    memory_set = pi.identify_phases(metric='Dirty', category='Memory usage', motif_length=250)
    print(memory_set.selected_motifs)
    test_chi2IsUniform(memory_set)
    # network_set = pi.identify_phases(metric='tx_bytes#eth0.rate', category='Slow network interface usage', motif_length=200)
    # test_chi2IsUniform(network_set)
    # mpi_phase_set = pi.identify_phases(metric='MPI_Issend.calls.1.rate', category='MPI calls', motif_length=250)
    # test_chi2IsUniform(mpi_phase_set)

    cpu_usage_set = pi.identify_phases_in_large_phase(metric='user.rate', category='Cpu usage',
                                        motif_length=40, large_phase_set=memory_set.selected_motifs, range_real_motif_length=memory_set.real_motif_length, range_sampler_df=memory_set.sampler_df)


    # cpu_usage_set = pi.identify_phases(metric='user.rate', category='Cpu usage',
    #                                     motif_length=50)
    print(cpu_usage_set.selected_motifs)
    test_chi2IsUniform(cpu_usage_set)

    plot_all_motifs(pi, [memory_set, cpu_usage_set])

def test_collapse(pi):
    logger = logging.getLogger(__name__)
    logger.info("start")


    fs_read_set = pi.identify_phases(metric='read.rate', category='filesystem reads',motif_length=250)
    fs_read_set.selected_phase_set = test_chi2IsUniform(fs_read_set)
    memory_set = pi.identify_phases(metric='Dirty', category='Memory usage', motif_length=250)
    memory_set.selected_phase_set = test_chi2IsUniform(memory_set)
    network_set = pi.identify_phases(metric='tx_bytes#eth0.rate', category='Slow network interface usage', motif_length=200)
    network_set.selected_phase_set= test_chi2IsUniform(network_set)
    #'Per_core_softirqd0.rate', 'Per_core_sys1'
    cpu_usage_set = pi.identify_phases(metric='Per_core_softirqd0.rate', category='Cpu usage',
                                        motif_length=250)
    cpu_usage_set.selected_phase_set = test_chi2IsUniform(cpu_usage_set)

    fs_write_set = pi.identify_phases(metric='write.rate', category='filesystem reads', motif_length=250)
    fs_write_set.selected_phase_set = test_chi2IsUniform(fs_write_set)
    mpi_phase_set = pi.identify_phases(metric='MPI_Issend.calls.1.rate', category='MPI calls', motif_length=250)
    mpi_phase_set.selected_phase_set = test_chi2IsUniform(mpi_phase_set) #mpi_phase_set.selected_motifs

    plot_motifs_complement(pi, [fs_write_set,mpi_phase_set], 'fs_write_mpi')
    gc.collect()
    plot_motifs_complement(pi, [memory_set,  mpi_phase_set], 'memory_mpi')
    gc.collect()
    plot_motifs_complement(pi, [ network_set, mpi_phase_set], 'network_mpi')
    gc.collect()
    plot_motifs_complement(pi, [cpu_usage_set, mpi_phase_set], 'cpu_usage_mpi')

def test_motif_length_range(pi):
    logger = logging.getLogger(__name__)
    logger.info("start")
    # pi.enumerate_motif_length_range(metric='MPI_Issend.calls.1.rate', category='MPI calls')
    memory_set = pi.enumerate_motif_length_range_all(metric='Dirty', category='Memory usage')

    # fs_read_set = pi.enumerate_motif_length_range_all(metric='commit.rate', category='filesystem writes')
    # network_set = pi.enumerate_motif_length_range(metric='tx_bytes#eth0.rate', category='Slow network interface usage')
    # cpu_usage_set = pi.enumerate_motif_length_range(metric='Per_core_softirqd0.rate', category='Cpu usage')




def test_new_motifs(pi):
    logger = logging.getLogger(__name__)
    logger.info("start")

    mpi_set = pi.identify_phases(metric='MPI_Issend.calls.1', category='MPI calls', motif_length=40)

    print(mpi_set.selected_motifs)




def testIdentifyPhases():
    logger = logging.getLogger(__name__)
    logger.info("start")

    all_samplers = ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'procnfs']

    categories = ['memory']

    pi = PhaseIdentifier(name='PhaseIdentifier', workload_path='ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/', model='waleElemXflowMixFrac3.5m')

    # identifiedPhases = test_identify_mpi_calls(pi)
    # logger.info("identified Phases: ")
    # logger.info(identifiedPhases)

    # test_identify_using_multiple_metrics(pi)
    # test_collapse(pi)
    # test_motif_length_range(pi)
    test_identify_hirearchical(pi)
    # test_new_motifs(pi)

def init_config():
    logging.basicConfig(format='%(asctime)s - %(funcName)s() - %(levelname)s: %(message)s',level=logging.INFO)

def getInfoFromSnippetRow(r):
    metric = r['metric']
    length = r['length']
    count = r['count']
    fractions = []
    snippets = []

    for i in range(count):
        fname = 'f' + str(i+1)
        sname = 's' + str(i+1)
        fractions.append(r[fname])
        snippets.append(r[sname])
    fractionsnippets = pd.DataFrame({'fractions':fractions, 'snippets':snippets})
    fractionsnippets.sort_values(['fractions'], ascending=False, inplace=True)
    return metric, length, count, fractionsnippets

def apply_snippet_row_fraction(fractionsnippet, **kwargs):
    metric = kwargs['metric']
    metric = metric[:- 4]
    length = kwargs['length']
    count = kwargs['count']
    pi = kwargs['pi']
    fraction = fractionsnippet['fractions']
    snippet = fractionsnippet['snippets']

    print(str(fraction) + ' ' + str(snippet))

    res = pi.identify_phases_using_snippets(metric, int(snippet), length, fraction)
    print(res.selected_motifs)



def apply_snippet_row(r, **kwargs):
    metric, length, count, fractionsnippets = getInfoFromSnippetRow(r)
    print(metric + ' ' +  str(length) + ' ' + str(count))
    pi = kwargs['pi']
    fractionsnippets.apply(apply_snippet_row_fraction, axis=1 , metric=metric, length=length, count=count, pi=pi)
    # print(fractionsnippets)
    # print(snippets)

def use_snippet_results():
    pi = PhaseIdentifier(name='PhaseIdentifier',
                         workload_path='ModelwaleElemXflowMixFrac3.5mVersion1RUN1Interval1000000/',
                         model='waleElemXflowMixFrac3.5m')

    path = 'D:/ac/PhD/Research/data/pd/snippet-finder-results/snippet.csv'
    snippets = read_csv(path, sep=',')
    snippets.apply(apply_snippet_row, axis=1, pi=pi)

def snippet_cluster(df):
    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = skds.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    print(noisy_circles)
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}
    # f1 = df[' f1']
    f1 = df.as_matrix(columns=[' f1'])
    datasets = [
        (f1, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        print(dataset)
        X = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')

        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('Birch', birch),
            ('GaussianMixture', gmm)
        )

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                            "connectivity matrix is [0-9]{1,2}" +
                            " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                            " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
    print("before")
    plt.show()
    print("after")

def apply_all_clustering(df, length = None, count = None):
    # df = pd.read_csv(self.repo_positions_file)
    print("apply_all_clustering")
    # print(df['metric'].unique())
    clusters_count = 5
    if length != None:
        df = df[df[' length'] == length]
    if count != None:
        df = df[df[' count'] == count]
    df = df[df[' f1'] > 0]
    df = df[df[' f2'] > 0]
    df = df[df[' count'] > 2]
    df = df[df[' length'] < 700]
    X = df[[' f1', ' f2']]
    params = {'quantile': .3,
                'eps': .1,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': clusters_count}
    plot_num = 1
    # estimate bandwidth for mean shift
    print("before estimate_bandwidth")
    # bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
    print("before kneighbors_graph")
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    # ============
    # Create cluster objects
    # ============
    # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    kmeans = cluster.KMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('KMeans', kmeans),
        # ('AffinityPropagation', affinity_propagation),
        # ('MeanShift', ms),
        # ('SpectralClustering', spectral),
        # ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        # ('DBSCAN', dbscan),
        # ('Birch', birch),
        ('GaussianMixture', gmm)
    )
    print("before for")
    for name, algorithm in clustering_algorithms:
        print(name)
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(2, math.ceil(len(clustering_algorithms)/2), plot_num)
        plt.title(name, size=14)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00',
                                             '#FF9F33', '#96FF33', '#33FF74',
                                             '#33FF96', '#33FFD1', '#33DDFF',
                                             '#C133FF', '#EC33FF', '#FF33C1',
                                             '#FF3380', '#FF336E'
                                             ]),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[' f1'].tolist(), X[' f2'].tolist(), s=14, color=colors[y_pred])

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

    plt.show()

def plot_hist_fraction(df, findex={1,2,3,4,5,6,7,8,9,10,16}, filterZero=True, filterOne=False, filterCount=True):
    for fi in findex:

        # if fi != 1:
        #     continue
        cname = ' f' + str(fi)
        fdf = df
        if filterZero:
            fdf = fdf[fdf[cname] > 0]
        if filterOne:
            fdf = fdf[fdf[cname] < 1]
        if filterCount:
            fdf = fdf[fdf[' count'] > 1]
        if len(fdf[cname]) == 0:
            continue
        fdf[cname].plot(kind='hist', bins=1000)

def metrics_cluster_by_fraction_snippet(idf, findex={1,2}):

    idf = idf[idf[' count'] > 1]
    counts = [2, 3, 4, 5]#, 6, 7, 8, 9, 10, 16]
    lengths = [10, 20, 50, 80]#, 100, 150, 200, 250, 300, 400, 500, 700]
    # for l in lengths:
    #     print(l)
    metrics = { 100: {},
                90: {},
               80: {},
               70: {},
               60: {},
               50: {},
               40: {},
               30: {},
               20: {},
               10: {},
               00: {},
               }
    df = idf #[idf[' length'] >= 200]
    total = 0
    for f,s in metrics.items():
        cdf = df[((df[' f1'] >= f / 100) & (df[' f1'] < f / 100 + 0.1)) | ((df[' f2'] >= f / 100) & (df[' f2'] < f / 100 + 0.1))]
        print(f)
        g = cdf.groupby(['metric']).count()[[' f1']]
        print(g)
        # g.to_csv('milestonRun1_all_lengths_groupby_' + str(f) + '.csv')
        current_len = len(cdf['metric'].unique())
        total += current_len
        print(str(f) + ":" + str(current_len) + "/" + str(total))
        # for fi in findex:
        #     # if fi != 1:
        #     #     continue
        #     cname = ' f' + str(fi)
        #     print(cname)
        #     cdf = df[df[cname] >= f / 100][df[cname] < f / 100 + 0.1]
        #     g = cdf.groupby([' length']).count()[[cname]]
        #     print(g)
        #     g.to_csv('groupby_' + str(f) + '_' + cname + '.csv')
        #     g = cdf.groupby([' count']).count()[[cname]]
        #     print(g)
        #     # print()
        #     # for m in cdf.iterrows():
        #     #     try:
        #     #         # print(m)
        #     #         test = s[m['metric']]
        #     #         s[m['metric']] += m[' count']#s[m] + 1
        #     #     except KeyError:
        #     #         s[m['metric']] = []#1
        #
        #     # s |= set(df[df[cname] >= f/100][df[cname] < f/100 + 0.1]['metric'].unique())
        # print(s)

def plot_hist_snipeets():
    path= 'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/PhaseDetection/snippetfinderCode(1)/result_all_metrics_all.csv'
    df = read_csv(path, sep=',')

    # df[df[' count'] > 1][' f1'].plot(kind='hist', bins=100)
    # df[df[' count'] > 1][' f2'].plot(kind='hist', bins=100)
    # df[df[' count'] > 2][' f3'].plot(kind='hist', bins=100)
    # df[df[' count'] > 3][df[' f4'] > 0][' f4'].plot(kind='hist', bins=100)
    # df[df[' count'] > 4][' f5'].plot(kind='hist', bins=100)
    # df[df[' count'] > 5][' f6'].plot(kind='hist', bins=100)
    # df[df[' count'] > 6][' f7'].plot(kind='hist', bins=100)
    # df[df[' length'] == 500][' f1'].plot(kind='hist', bins=100)
    # scikit-learn kmeans, em(gausian mixture)
    # df.plot(kind='scatter',x=' length', y=' f1')
    # df.plot(kind='scatter', x=' length', y=' f2')
    # df.plot(kind='scatter', x=' length', y=' f3')
    # plot_hist_fraction(df)
    # clf = mixture.GaussianMixture(n_components=4, covariance_type='full')
    # clf.fit(df)
    # X=df[' f1']
    # X[' f2'] = df[' f2']
    # kmeans = KMeans(4, random_state=0)
    # labels = kmeans.fit(X).predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

    # df.plot(kind='scatter', x=' f2', y=' f3')
    # df[' f2'].plot(kind='hist', bins=20)
    # df[' f3'].plot(kind='hist', bins=20)
    # plt.show()
    # snippet_cluster(df)
    lengths = [10, 20, 50, 80, 100, 150, 200, 250, 300, 400, 500, 700]
    counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16]
    # for l in [200,300]:
    #     print(l)
    # for c in counts:
    #     print(c)
    # apply_all_clustering(df)
    metrics_cluster_by_fraction_snippet(df)
    print(df.columns)

def snippet_analysis_milestoneRun1():
    path= 'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/PhaseDetection/snippetfinderCode(1)/result_all_metrics_milestoneRun1.csv'
    df = read_csv(path, sep=',')
    metrics_cluster_by_fraction_snippet(df)

def load_LDMS_Metrics_Write_Separate_Files(base_path = 'D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/',
                                           samplers=['shm_sampler',
                                                     'vmstat','meminfo', 'procnetdev', 'procnfs', 'procstat']):

    csv_path = base_path
    file_extension = '.csv'
    for sampler in samplers:
        print(sampler)
        file_path = csv_path + sampler + file_extension
        df = read_csv(file_path, sep=',')
        for c in df.columns:
            if c in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id']:
                continue
            writeColumnInFile(df, c, transform=None, fileName=None, folder=base_path + 'metrics/' + sampler + '/')
            writeColumnInFile(df, c, transform='rate', fileName=None, folder=base_path + 'metrics/' + sampler + '/')


def writeColumnInFile(df, column, transform=None, fileName=None, folder=''):
    print(column)
    this_sampler = df
    if transform != None:
        transformed_df = df
        if transform == "rate":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], True, False,
                                                                    False)
        elif transform == "log":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], False, False,
                                                                    True)
        elif transform == "sum":
            transformed_df, [ret_names] = lt.create_transform_event(transformed_df, [column], [], False, True,
                                                                    False)
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

def use_snippet_results_long_run():
    pi = PhaseIdentifier(name='PhaseIdentifier', metricsFolder='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         workload_path='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         model=None)

    path = 'C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/PhaseDetection/snippetfinderCode(1)/milestoneRun_LongRun_selectedMetrics.csv'
    snippets = read_csv(path, sep=',')
    print(snippets.shape)
    snippets = snippets[snippets['count'] == 16]
    print(snippets.shape)
    snippets.apply(apply_snippet_row, axis=1, pi=pi)


def long_run_plot():
    pi = PhaseIdentifier(name='PhaseIdentifier', metricsFolder='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         motifsFolder='C:/Users/Ramin/PycharmProjects/GGS-git/motifs/csvs/',
                         workload_path='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         model=None)
    metrics = ["Dirty", "MPI_Issend.calls.0.rate", "sys.rate", "MPI_Allreduce.calls.0.rate", "MemFree", "numa_hit"] #
    motif_lengths = [7200, 3600, 1800, 600, 300, 120, 60, 30]
        # [30, 60, 120, 300, 600, 1800, 3600, 7200]

    for motif_length in motif_lengths:
        for m in metrics:
            pi.self_join_and_plot_long_run(m, motif_length)


    # for metric in metrics:
    #     for motif_length in motif_lengths:
    #         if motif_length == 60 and metric == "Dirty":
    #             continue
    #         try:
    #             res = pi.identify_phases_and_plot_long_run(metric, motif_length)
    #             print(res.selected_motifs)
    #         except subprocess.CalledProcessError as e:
    #             print(e)

def process_8_metrics(pi, metrics, mpi_metric, r_start, motif_length):
    pi.pd_self_join_and_plot_mp_side_by_side(metrics, 'MPI calls', mpi_metric + "_" + str(r_start), motif_length)




def pd_and_plot_mp_side_by_side():
    pi = PhaseIdentifier(name='PhaseIdentifier', metricsFolder='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         motifsFolder='C:/Users/Ramin/PycharmProjects/GGS-git/motifs/csvs/',
                         workload_path='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         model=None)
    mpi_metrics = ["MPI_Issend.calls.", "MPI_Send.calls.", "MPI_Irecv.calls.", "MPI_Allreduce.calls.", "MPI_Isend.calls.", "MPI_Wait.calls.", "MPI_Wtime.calls.", "MPI_Ssend.calls.", "MPI_Reduce.calls.", "MPI_Scatter.calls."]
    mpi_metric = "MPI_Reduce.calls." #"MPI_Issend.calls."
    ranks = range(0,32, 8)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    args = []
    metrics = []
    motif_length = 7200
    for motif_length in [7200, 3600, 1800]:
        print("motif_length=" + str(motif_length))
        for mpi_metric in mpi_metrics:
            print("mpi_metric=" + mpi_metric)
            for r_start in ranks:
                print(r_start)
                metrics = []

                for r in range(r_start, r_start+8):
                    current_metric = mpi_metric + str(r)
                    metrics.append(current_metric)
                # args.append((pi, metrics, mpi_metric, r_start, motif_length))
                pi.pd_self_join_and_plot_mp_side_by_side(metrics, 'MPI calls', mpi_metric + "_" + str(r_start),
                                                         motif_length)

    # pool.starmap(process_8_metrics, args)
    # pool.close()
    # pool.join()
            # pi.pd_self_join_and_plot_mp_side_by_side(metrics, 'MPI calls', mpi_metric + "_" + str(r_start), motif_length)
    # metrics = ["MPI_Issend.calls.0.rate", "MPI_Issend.calls.1.rate", "MPI_Issend.calls.2.rate", "MPI_Issend.calls.3.rate", "MPI_Issend.calls.4.rate", "MPI_Issend.calls.5.rate"]


def pd_and_plot_different_metrics_mp_side_by_side():
    pi = PhaseIdentifier(name='PhaseIdentifier', metricsFolder='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         motifsFolder='C:/Users/Ramin/PycharmProjects/GGS-git/motifs/csvs/',
                         workload_path='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         model=None)
    mpi_metrics = ["MPI_Issend.calls.", "MPI_Send.calls.", "MPI_Irecv.calls.", "MPI_Allreduce.calls."]#, "MPI_Isend.calls.", "MPI_Wait.calls.", "MPI_Wtime.calls.", "MPI_Ssend.calls.", "MPI_Reduce.calls.", "MPI_Scatter.calls."]
    mpi_metric = "MPI_Reduce.calls." #"MPI_Issend.calls."
    ranks = range(0,32, 8)
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    args = []
    metrics = []
    motif_length = 7200
    category = 'MPI calls'
    for motif_length in [7200, 3600, 1800]:
        print("motif_length=" + str(motif_length))
        for transform in [".rate", ""]:
            for mpi_metric in mpi_metrics:
                print("mpi_metric=" + mpi_metric)
                metric_to_join = None
                for r_start in ranks:
                    # if r_start in (0,8,16) and mpi_metric == "MPI_Issend.calls." and transform == ".rate" and motif_length == 7200:
                    #     continue
                    print(r_start)
                    metrics = []

                    for r in range(r_start, r_start+8):
                        current_metric = mpi_metric + str(r) + transform
                        if metric_to_join == None:
                            metric_to_join = current_metric
                        metrics.append(current_metric)
                    # args.append((pi, metrics, mpi_metric, r_start, motif_length))
                    name = mpi_metric + transform +  "_" + str(r_start)
                    pi.pd_join_different_metrics_and_plot_mp_side_by_side(metric_to_join, metrics, category,
                                                                           name, motif_length)

def pd_and_plot_mp_procstat_side_by_side():
    pi = PhaseIdentifier(name='PhaseIdentifier', metricsFolder='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         motifsFolder='C:/Users/Ramin/PycharmProjects/GGS-git/motifs/csvs/',
                         workload_path='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         model=None)
    procstat_metrics = ["per_core_user", "per_core_sys","per_core_idle","per_core_iowait"]
    processes = range(0,32, 8)
    for motif_length in [7200, 3600, 1800]:
        print("motif_length=" + str(motif_length))
        for transform in [".rate", ""]:
            for procstat_metric in procstat_metrics:
                print("procstat_metric=" + procstat_metric)
                metric_to_join = None
                for p_start in processes:
                    print(p_start)
                    metrics = []

                    for p in range(p_start, p_start+8):
                        current_metric = procstat_metric + str(p) + transform
                        if metric_to_join == None:
                            metric_to_join = current_metric
                        metrics.append(current_metric)
                    # args.append((pi, metrics, mpi_metric, r_start, motif_length))
                    pi.pd_self_join_and_plot_mp_side_by_side(metrics, 'Cpu usage', procstat_metric +  transform + "_" + str(p_start),
                                                             motif_length)

def pd_and_plot_different_metrics_mp_procstat_side_by_side():
    pi = PhaseIdentifier(name='PhaseIdentifier', metricsFolder='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         motifsFolder='C:/Users/Ramin/PycharmProjects/GGS-git/motifs/csvs/',
                         workload_path='D:/ac/PhD/Research/data/pd/snippet-finder-results/milestonRun-long-run/LongNaluRun/csvs/',
                         model=None)
    procstat_metrics = ["per_core_user", "per_core_sys","per_core_idle","per_core_iowait"]
    processes = range(0,32, 8)
    for motif_length in [7200, 3600, 1800]:
        print("motif_length=" + str(motif_length))
        for transform in [".rate", ""]:
            for procstat_metric in procstat_metrics:
                print("procstat_metric=" + procstat_metric)
                for p_start in processes:
                    print(p_start)
                    metrics = []

                    for p in range(p_start, p_start+8):
                        current_metric = procstat_metric + str(p) + transform
                        metrics.append(current_metric)
                    # args.append((pi, metrics, mpi_metric, r_start, motif_length))
                    pi.pd_self_join_and_plot_mp_side_by_side(metrics, 'Cpu usage', procstat_metric +  transform + "_" + str(p_start),
                                                             motif_length)

def run_single_pd_plot_multiple_run(run_key, pis, metric, category, name, motif_length):
    print('run_key=' + run_key + ' and metric=' + metric)
    pis[run_key].multiple_data_set_pd_self_join_and_plot_mp_side_by_side(pis, metric, category, name,
                                                                        motif_length)


def pd_plot_multiple_run_non_mpi():
    logger = logging.getLogger(__name__)
    logger.info("start")
    name = 'PhaseIdentifier'
    metric_base_path = 'D:/ac/PhD/Research/data/05/data/'
    config_specific = 'XeonModelmilestoneRunRUN{}/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN{}Interval100000/'
    metric_run_paths = {

        'RUN1': config_specific.format(1, 1),
        'RUN2': config_specific.format(2, 2),
        'RUN3': config_specific.format(3, 3),
        'RUN4': config_specific.format(4, 4),
        'RUN5': config_specific.format(5, 5),
        'RUN6': config_specific.format(6, 6),
        'RUN7': config_specific.format(7, 7),
        'RUN8': config_specific.format(8, 8),
        'RUN9': config_specific.format(9, 9),
        'RUN10': config_specific.format(10, 10)
    }
    pis = {}

    for r in metric_run_paths.keys():
        pis[r] = PhaseIdentifier(name='PhaseIdentifier_' + r,
                                 metricsFolder=metric_base_path + metric_run_paths[r] + 'metrics/',
                                 workload_path=metric_base_path + metric_run_paths[r],
                                 model=None)

        # load_LDMS_Metrics_Write_Separate_Files(
        #         base_path=metric_base_path+metric_run_paths[r])

    pool = multiprocessing.Pool(4)

    plots_paths = 'D:/ac/PhD/Research/data/pd/compare-10runs/plots/'
    for motif_length in [60, 120, 200, 300, 600, 1200]:
        logger.info("motif_length=" + str(motif_length))
        for transform in [""]:
            for sampler in ['meminfo','vmstat', 'procnetdev', 'procnfs' , 'procstat']:
                args = []
                sampler_df = pis['RUN1'].ldmsInstance.getMetricSet(sampler).getDataFrame()
                category = pis['RUN1'].samplersToCategoriesMap[sampler]
                counter = 0
                print(sampler_df.columns)
                for metric in sampler_df.columns:
                    if metric in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id', 'Dummy']:
                        continue

                    args.append(('RUN' + str(counter % 10 + 1), pis, metric, category, metric, motif_length))
                    counter = counter + 1
                    if counter % 10 == 0:
                        pool.starmap(run_single_pd_plot_multiple_run, args)
                        args = []
                pool.starmap(run_single_pd_plot_multiple_run, args)
    pool.close()
    pool.join()

def pd_plot_multiple_run():
    logger = logging.getLogger(__name__)
    logger.info("start")
    name = 'PhaseIdentifier'
    metric_base_path = 'D:/ac/PhD/Research/data/05/data/'
    config_specific = 'XeonModelmilestoneRunRUN{}/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN{}Interval100000/'
    metric_run_paths = {

        'RUN1' : config_specific.format(1,1),
        'RUN2': config_specific.format(2, 2),
        'RUN3': config_specific.format(3, 3),
        'RUN4': config_specific.format(4, 4),
        'RUN5': config_specific.format(5, 5),
        'RUN6': config_specific.format(6, 6),
        'RUN7': config_specific.format(7, 7),
        'RUN8': config_specific.format(8, 8),
        'RUN9': config_specific.format(9, 9),
        'RUN10': config_specific.format(10, 10)
    }
    pis = {}
    category = 'MPI calls'
    for r in metric_run_paths.keys():
        pis[r] = PhaseIdentifier(name='PhaseIdentifier_'+ r, metricsFolder=metric_base_path+metric_run_paths[r] + 'metrics/',
                         workload_path=metric_base_path+metric_run_paths[r],
                         model=None)

        # load_LDMS_Metrics_Write_Separate_Files(
        #         base_path=metric_base_path+metric_run_paths[r])

    pool = multiprocessing.Pool(4)

    plots_paths = 'D:/ac/PhD/Research/data/pd/compare-10runs/plots/'

    mpi_metrics = ["MPI_Issend.calls.",  "MPI_Send.calls.", "MPI_Irecv.calls.", "MPI_Allreduce.calls."]
    ranks = range(0, 32, 8)
    for motif_length in [60, 120, 200, 300, 600, 1200]:
        logger.info("motif_length=" + str(motif_length))
        for transform in [".rate"]:
            for mpi_metric in mpi_metrics:
                logger.info("mpi_metric=" + mpi_metric)
                for r_start in ranks:
                    # if r_start in (0,8,16) and mpi_metric == "MPI_Issend.calls." and transform == ".rate" and motif_length == 7200:
                    #     continue
                    logger.info("r_start=" + str(r_start))
                    r_end = r_start+8
                    if(r_start == 24):
                        r_end = 30
                    args = []
                    for r in range(r_start, r_end):
                        logger.info("r=" + str(r))
                        metric = mpi_metric + str(r) + transform
                        name = metric

                        run_key = 'RUN' + str(r % 4 + 1)
                        args.append((run_key, pis, metric, category, name, motif_length))
                    pool.starmap(run_single_pd_plot_multiple_run, args)

                        # pis['RUN1'].multiple_data_set_pd_self_join_and_plot_mp_side_by_side(pis, metric, category, name,
                        #                                                                     motif_length)
    pool.close()
    pool.join()

def run_single_pd_plot_multiple_run_different_metrics(run_key, pis, metric, category, name, motif_length, pi_to_join, metric_to_join):
    print('run_key=' + run_key + ' and metric=' + metric)

    pis[run_key].multiple_data_set_pd_other_join_and_plot_mp_side_by_side(pis, metric, pi_to_join, metric_to_join, category, name,
                                                                        motif_length)

def pd_plot_multiple_run_different_metrics_mp_side_by_side():
    logger = logging.getLogger(__name__)
    logger.info("start")
    name = 'PhaseIdentifier'
    metric_base_path = 'D:/ac/PhD/Research/data/05/data/'
    config_specific = 'XeonModelmilestoneRunRUN{}/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN{}Interval100000/'
    metric_run_paths = {

        'RUN1' : config_specific.format(1,1),
        'RUN2': config_specific.format(2, 2),
        'RUN3': config_specific.format(3, 3),
        'RUN4': config_specific.format(4, 4),
        'RUN5': config_specific.format(5, 5),
        'RUN6': config_specific.format(6, 6),
        'RUN7': config_specific.format(7, 7),
        'RUN8': config_specific.format(8, 8),
        'RUN9': config_specific.format(9, 9),
        'RUN10': config_specific.format(10, 10)
    }
    pis = {}



    category = 'MPI calls'
    for r in metric_run_paths.keys():
        pis[r] = PhaseIdentifier(name='PhaseIdentifier_'+ r, metricsFolder=metric_base_path+metric_run_paths[r] + 'metrics/',
                         workload_path=metric_base_path+metric_run_paths[r],
                         model=None)

        if r== 'RUN10':
            pi_to_join = pis[r]


        # load_LDMS_Metrics_Write_Separate_Files(
        #         base_path=metric_base_path+metric_run_paths[r])

    pool = multiprocessing.Pool(4)
    ranks = range(0, 32, 8)
    mpi_metrics = ["MPI_Issend.calls.", "MPI_Send.calls.", "MPI_Irecv.calls.", "MPI_Allreduce.calls."]
    ranks = range(0, 32, 8)
    for motif_length in [60, 120, 200, 300, 600, 1200]:
        logger.info("motif_length=" + str(motif_length))
        for transform in [".rate"]:
            for mpi_metric in mpi_metrics:
                logger.info("mpi_metric=" + mpi_metric)
                for r_start in ranks:
                    # if r_start in (0,8,16) and mpi_metric == "MPI_Issend.calls." and transform == ".rate" and motif_length == 7200:
                    #     continue
                    logger.info("r_start=" + str(r_start))
                    r_end = r_start+8
                    if(r_start == 24):
                        r_end = 30
                    args = []
                    for r in range(r_start, r_end):
                        logger.info("r=" + str(r))
                        metric = mpi_metric + str(r) + transform
                        name = metric

                        run_key = 'RUN' + str(r % 4 + 1)
                        args.append((run_key, pis, metric, category, name, motif_length, pi_to_join, metric))

                    pool.starmap(run_single_pd_plot_multiple_run_different_metrics, args)

                        # pis['RUN1'].multiple_data_set_pd_self_join_and_plot_mp_side_by_side(pis, metric, category, name,
                        #                                                                     motif_length)
    pool.close()
    pool.join()


def pd_plot_multiple_run_different_metrics_non_mpi():
    logger = logging.getLogger(__name__)
    logger.info("start")
    name = 'PhaseIdentifier'
    metric_base_path = 'D:/ac/PhD/Research/data/05/data/'
    config_specific = 'XeonModelmilestoneRunRUN{}/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN{}Interval100000/'
    metric_run_paths = {

        'RUN1': config_specific.format(1, 1),
        'RUN2': config_specific.format(2, 2),
        'RUN3': config_specific.format(3, 3),
        'RUN4': config_specific.format(4, 4),
        'RUN5': config_specific.format(5, 5),
        'RUN6': config_specific.format(6, 6),
        'RUN7': config_specific.format(7, 7),
        'RUN8': config_specific.format(8, 8),
        'RUN9': config_specific.format(9, 9),
        'RUN10': config_specific.format(10, 10)
    }
    pis = {}

    for r in metric_run_paths.keys():
        pis[r] = PhaseIdentifier(name='PhaseIdentifier_' + r,
                                 metricsFolder=metric_base_path + metric_run_paths[r] + 'metrics/',
                                 workload_path=metric_base_path + metric_run_paths[r],
                                 model=None)
        if r== 'RUN10':
            pi_to_join = pis[r]

        # load_LDMS_Metrics_Write_Separate_Files(
        #         base_path=metric_base_path+metric_run_paths[r])

    pool = multiprocessing.Pool(4)

    plots_paths = 'D:/ac/PhD/Research/data/pd/compare-10runs/plots/'
    for motif_length in [60, 120, 200, 300, 600, 1200]:
        logger.info("motif_length=" + str(motif_length))
        for transform in [""]:
            for sampler in ['meminfo','vmstat', 'procnetdev', 'procnfs' , 'procstat']:
                args = []
                sampler_df = pis['RUN1'].ldmsInstance.getMetricSet(sampler).getDataFrame()
                category = pis['RUN1'].samplersToCategoriesMap[sampler]
                counter = 0
                print(sampler_df.columns)
                for metric in sampler_df.columns:
                    if metric in ['#Time', 'Time_usec', 'ProducerName', 'component_id', 'job_id', 'Dummy']:
                        continue
                    run_key = 'RUN' + str(counter % 10 + 1)
                    args.append((run_key, pis, metric, category, metric, motif_length, pi_to_join, metric))

                    # args.append(('RUN' + str(counter % 10 + 1), pis, metric, category, metric, motif_length))
                    counter = counter + 1
                    if counter % 10 == 0:
                        pool.starmap(run_single_pd_plot_multiple_run_different_metrics, args)
                        args = []
                pool.starmap(run_single_pd_plot_multiple_run_different_metrics, args)
    pool.close()
    pool.join()


def test_target_discord():
    df = pd.read_csv('C:/Users/Ramin/OneDrive - Knights - University of Central Florida/Dropbox/ac/PhD/Research/OVIS/LDMS/PhaseDetection/Target Anomaly Example/nab/realKnownCause/realKnownCause/nyc_taxi.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    df = df.resample('1H').sum()
    a = df.values.squeeze()
    m = 24
    profile = matrixProfile.stomp(a, m)
    df['profile'] = np.append(profile[0], np.zeros(m - 1) + np.nan)
    df['profile_index'] = np.append(profile[1], np.zeros(m - 1) + np.nan)

    # Plot the signal data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    df['value'].plot(ax=ax1, title='Raw Data')

    # Plot the Matrix Profile
    df['profile'].plot(ax=ax2, c='r', title='Matrix Profile')

    # exclude up to a day on the left and right side
    ex_zone = 24
    # we look for the 5 events specified in the data explaination
    anoms = discords(df['profile'], ex_zone, k=5)
    print(df.iloc[anoms])

    plt.show()

    # print(df.head())

def ldms_test_target_discord():
    file_path = 'D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/metrics/shm_sampler/MPI_Send.calls.0.txt'
    df = read_csv(file_path, sep=',', header=None)
    # df['#Time'] = pd.to_datetime(df['#Time'])
    # df = df.set_index('#Time').sort_index()
    a = df.values.squeeze()
    df['value'] = a
    m = 200
    profile = matrixProfile.stomp(a, m)

    df['profile'] = np.append(profile[0], np.zeros(m - 1) + np.nan)
    df['profile_index'] = np.append(profile[1], np.zeros(m - 1) + np.nan)

    # Plot the signal data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    df['value'].plot(ax=ax1, title='Raw Data')

    # Plot the Matrix Profile
    df['profile'].plot(ax=ax2, c='r', title='Matrix Profile')

    # exclude up to a day on the left and right side
    ex_zone = 200
    # we look for the 5 events specified in the data explaination
    anoms = discords(df['profile'], ex_zone, k=4)
    print(df.iloc[anoms])

    plt.show()

# use annotation vector
    # seed random number generator
    seed(1)
    # generate random numbers between 0-1
    av = rand(len(profile[0]) - 1000)
    prep = np.ones(1000)
    av = np.append(prep, np.ones(len(av)))
    print(len(av))
    print(len(profile[0]))
    print(len(profile[1]))
    profile2 = utils.apply_av(profile, av)

    print(type(profile2))

    test2 = np.append(profile2, np.zeros(m - 1) + np.nan)
    print(len(profile2))
    print(len(profile[1]))
    print(len(test2))

    df['profile2'] = np.append(profile2, np.zeros(m - 1) + np.nan)
    df['profile_index2'] = np.append(profile[1], np.zeros(m - 1) + np.nan)

    # Plot the signal data
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    df['value'].plot(ax=ax1, title='Raw Data')

    # Plot the Matrix Profile
    df['profile2'].plot(ax=ax2, c='r', title='Matrix Profile')

    # exclude up to a day on the left and right side
    ex_zone = 200
    # we look for the 5 events specified in the data explaination
    anoms = discords(df['profile2'], ex_zone, k=4)
    print(df.iloc[anoms])

    plt.show()

def create_av_filter_begin_end(profile, length_begin, length_end):
    av = np.ones(len(profile[0]) - length_begin - length_end)
    app = np.zeros(length_end)
    prep = np.zeros(length_begin)
    av = np.append(prep, av)
    av = np.append(av, app)
    return av

def create_av_filter_end(profile, length):
    av = np.ones(len(profile[0]) - length)
    app = np.zeros(length)
    av = np.append(av, app)
    return av

def create_av_filter_beginning(profile, length):
    av = np.ones(len(profile[0]) - length)
    prep = np.zeros(length)
    av = np.append(prep, av)
    return av

def apply_av_begin_end(profile, length):
    av = create_av_filter_begin_end(profile, length, length)
    return utils.apply_av(profile, av)

def find_discords_abnormal_normal_join():
    metric_base_path = 'D:/ac/PhD/Research/data/05/data/'
    config_specific = 'XeonModelmilestoneRunRUN{}/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN{}Interval100000/'
    metric_run_paths = {

        'RUN1': config_specific.format(1, 1),
        'RUN2': config_specific.format(2, 2),
        'RUN3': config_specific.format(3, 3),
        'RUN4': config_specific.format(4, 4),
        'RUN5': config_specific.format(5, 5),
        'RUN6': config_specific.format(6, 6),
        'RUN7': config_specific.format(7, 7),
        'RUN8': config_specific.format(8, 8),
        'RUN9': config_specific.format(9, 9),
        'RUN10': config_specific.format(10, 10)
    }
    r = 'RUN1'
    pi = PhaseIdentifier(name='PhaseIdentifier_',
                             metricsFolder=metric_base_path + metric_run_paths[r] + 'metrics/',
                             workload_path=metric_base_path + metric_run_paths[r],
                             model=None)
    pis = {}
    pis[r] = pi
    metric = "MPI_Allreduce.calls.0"
    motif_length = 600
    category = 'MPI calls'

    r = 'RUN10'
    pi_to_join = PhaseIdentifier(name='PhaseIdentifier_',
                             metricsFolder=metric_base_path + metric_run_paths[r] + 'metrics/',
                             workload_path=metric_base_path + metric_run_paths[r],
                             model=None)
    metric_to_join = metric
    name = 'abnormal_normal_join_' + metric
    # one_metric_phase_set = pi.multiple_data_set_pd_other_join_and_plot_mp_side_by_side(pis, metric, pi_to_join, metric_to_join,
    #                                                              category, name, motif_length)

    OneMetricPhaseSet_metric_to_join = OneMetricPhaseSet(name=metric_to_join, phase_identifier=pi_to_join,
                                                         category=category, metric=metric_to_join,
                                                         motif_length=motif_length, loadData=False)

    one_metric_phase_set = OneMetricPhaseSet(name=metric, phase_identifier=pi, category=category, metric=metric,
                                             motif_length=motif_length, loadData=False)
    one_metric_phase_set.real_motif_length = motif_length
    one_metric_phase_set.query = one_metric_phase_set.metric_ts

    one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = matrixProfile._matrixProfile(
        OneMetricPhaseSet_metric_to_join.metric_ts, one_metric_phase_set.real_motif_length, order.linearOrder,
        distanceProfile.massDistanceProfile, one_metric_phase_set.metric_ts)

    # one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index = matrixProfile.stomp(
    #     one_metric_phase_set.metric_ts,
    #     one_metric_phase_set.real_motif_length, OneMetricPhaseSet_metric_to_join.metric_ts)
    one_metric_phase_set.select_best_motifs_from_stamp()



    extra_metric_one_metric_phase_set = pi.single_mp_create_self_join('procs_running', 200, 'Cpu usage')
    pi.plot_mp_side_with_anomaly(one_metric_phase_set, '', extra_metric_one_metric_phase_set,
                                 given_name=name)


def find_discords_abnormal_self_join():
    metric_base_path = 'D:/ac/PhD/Research/data/05/data/'
    config_specific = 'XeonModelmilestoneRunRUN{}/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN{}Interval100000/'
    metric_run_paths = {

        'RUN1': config_specific.format(1, 1),
        'RUN2': config_specific.format(2, 2),
        'RUN3': config_specific.format(3, 3),
        'RUN4': config_specific.format(4, 4),
        'RUN5': config_specific.format(5, 5),
        'RUN6': config_specific.format(6, 6),
        'RUN7': config_specific.format(7, 7),
        'RUN8': config_specific.format(8, 8),
        'RUN9': config_specific.format(9, 9),
        'RUN10': config_specific.format(10, 10)
    }
    r = 'RUN1'
    pi = PhaseIdentifier(name='PhaseIdentifier_',
                             metricsFolder=metric_base_path + metric_run_paths[r] + 'metrics/',
                             workload_path=metric_base_path + metric_run_paths[r],
                             model=None)
    category = 'MPI calls'
    mpi_metrics = ["MPI_Send.calls.","MPI_Irecv.calls.", "MPI_Allreduce.calls."]#, "MPI_Issend.calls.",  "MPI_Irecv.calls.", "MPI_Allreduce.calls."]


    param_map = {
        'MPI calls' : ["MPI_Send.calls.0","MPI_Irecv.calls.0", "MPI_Allreduce.calls.0"],
        'Cpu usage' : ['Sys', 'per_core_sys0'],
        'Memory usage': ['Cached']
    }
    count_map = {
        'MPI calls': 0,
        'Cpu usage': 0,
        'Memory usage': 0
    }
    for motif_length in [200]:
        logger.info("motif_length=" + str(motif_length))
        for transform in [""]:
            for category in param_map.keys():
                logger.info("category=" + category)
                for metric in param_map[category]:
                    one_metric_phase_set = pi.single_mp_create_self_join(metric, motif_length, category)
                    extra_metric_one_metric_phase_set = pi.single_mp_create_self_join('procs_running', motif_length, 'Cpu usage')
                    if count_map[category] == 0:
                        one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df['#Time'] - \
                                                                   one_metric_phase_set.sampler_df[
                                                                       '#Time'].min()
                        if count_map['Cpu usage'] == 0:
                            extra_metric_one_metric_phase_set.sampler_df['#Time'] = \
                                extra_metric_one_metric_phase_set.sampler_df['#Time'] - \
                                extra_metric_one_metric_phase_set.sampler_df[
                                    '#Time'].min()
                        try:

                            one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df[
                                '#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
                            if count_map['Cpu usage'] == 0:
                                extra_metric_one_metric_phase_set.sampler_df['#Time'] = \
                                    extra_metric_one_metric_phase_set.sampler_df[
                                        '#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
                        except TypeError as e:
                            print(e)
                        count_map[category] = count_map[category] + 1
                        count_map['Cpu usage'] = count_map['Cpu usage'] + 1

                    one_metric_phase_set.matrix_profile = apply_av_begin_end([one_metric_phase_set.matrix_profile, one_metric_phase_set.nearest_neighbour_index], motif_length)

                    pi.plot_mp_side_with_anomaly(one_metric_phase_set, '', extra_metric_one_metric_phase_set,
                                                 given_name=metric)



    # ranks = [0]#range(0, 32, 8)
    # count = 0
    # for motif_length in [200]:
    #     logger.info("motif_length=" + str(motif_length))
    #     for transform in [""]:
    #         for mpi_metric in mpi_metrics:
    #             logger.info("mpi_metric=" + mpi_metric)
    #             for r_start in ranks:
    #                 logger.info("r_start=" + str(r_start))
    #                 r_end = r_start+8
    #                 if(r_start == 0):
    #                     r_end = 1
    #                 args = []
    #                 for r in range(r_start, r_end):
    #                     metric = mpi_metric + str(r) + transform
    #                     name = metric
    #
    #                     one_metric_phase_set = pi.single_mp_create_self_join(metric, motif_length, category)
    #                     # extra_metric_one_metric_phase_set = pi.single_mp_create_self_join('procs_running', motif_length, 'Cpu usage')
    #                     extra_metric_one_metric_phase_set = pi.single_mp_create_self_join(metric + ".rate", motif_length,
    #                                                                                       category)
    #                     if count == 0:
    #                         one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df['#Time'] - \
    #                                                                    one_metric_phase_set.sampler_df[
    #                                                                        '#Time'].min()
    #                         extra_metric_one_metric_phase_set.sampler_df['#Time'] = \
    #                         extra_metric_one_metric_phase_set.sampler_df['#Time'] - \
    #                         extra_metric_one_metric_phase_set.sampler_df[
    #                             '#Time'].min()
    #                         try:
    #                             one_metric_phase_set.sampler_df['#Time'] = one_metric_phase_set.sampler_df[
    #                                 '#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    #                             extra_metric_one_metric_phase_set.sampler_df['#Time'] = \
    #                             extra_metric_one_metric_phase_set.sampler_df[
    #                                 '#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))
    #                         except TypeError as e:
    #                             print(e)
    #                         count = count + 1
    #
    #                     pi.plot_mp_side_with_anomaly(one_metric_phase_set, '', extra_metric_one_metric_phase_set , given_name=metric)


if __name__ == '__main__':
    multiprocessing.spawn.set_executable(_winapi.GetModuleFileName(0))
    # matplotlib.use('Agg')
    init_config()
    logger = logging.getLogger(__name__)
    logger.info("start")


    # testIdentifyPhases()
    # use_snippet_results()
    # plot_hist_snipeets()
    # snippet_analysis_milestoneRun1()
    # load_LDMS_Metrics_Write_Separate_Files()
    # use_snippet_results_long_run()
    # long_run_plot()
    # pd_and_plot_mp_side_by_side()

    # print(multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(processes=2)
    # names = ['Brown', 'Wilson', 'Bartlett', 'Rivera', 'Molloy', 'Opie']
    # results = pool.starmap(merge_names, product(names, repeat=2))
    # print(results)
    # pool.close()
    # pool.join()

    # pd_and_plot_different_metrics_mp_side_by_side()
    # pd_and_plot_mp_procstat_side_by_side()
    # pd_plot_multiple_run()
    # pd_plot_multiple_run_non_mpi()


    # pd_plot_multiple_run_different_metrics_mp_side_by_side()

    # test_target_discord()

    # ldms_test_target_discord()

    # find_discords_abnormal_self_join()
    find_discords_abnormal_normal_join()

    # pd_plot_multiple_run_different_metrics_non_mpi()

    # pd_plot_multiple_run()
