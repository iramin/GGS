from pandas import DatetimeIndex, read_csv
from datetime import datetime


class LDMSMetricSet(object):
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
        self.data = kwargs.get('data')
        self.doPreProcess = kwargs.get('preprocess', True)
        path = kwargs.get('path')
        if path != None:
            self.loadDataFrom(path)

    def loadDataFrom(self, path, verbose=False):
        self.data = read_csv(path)
        if verbose:
            print('{}: {}'.format(self.name, self.data.shape))
        if self.doPreProcess:
            self.preProcess()

    def preProcess(self):
        self.data.loc[:, 'Dummy'] = 0
        self.data.index = DatetimeIndex(self.data['#Time'])
        # self.data['#Time'] = self.data['#Time'].apply(lambda x: datetime.utcfromtimestamp((x)))

    def getDataFrame(self):
        return self.data

class LDMSInstance(object):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, *args, **kwargs):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.datasetNames = kwargs.get('datasets', ['meminfo', 'shm_sampler', 'vmstat', 'procstat', 'procnetdev', 'milestoneRun'])
        self.filetype = kwargs.get('filetype', '.csv')
        self.doPreProcess = kwargs.get('preprocess', True)
        self.metricSets = {}
        self.path = kwargs.get('path','D:/ac/PhD/Research/data/05/data/XeonModelmilestoneRunRUN1/overheadX/ModelmilestoneRunPlacementVersion6SamplingVersion1RUN1Interval100000/')
        self.loadAllData()

    def loadAllData(self, verbose=False):
        if verbose:
            print('loading all data')
        for dsName in self.datasetNames:
            self.metricSets[dsName] = LDMSMetricSet(dsName, path=self.path + dsName + self.filetype, preprocess=self.doPreProcess)

    def getMetricSet(self,name):
        return self.metricSets[name]

    def getAllMetricSets(self):
        return self.metricSets.values()

    def addMetricSet(self,name):
        self.datasetNames.append(name)
        self.metricSets[name] = LDMSMetricSet(name, path=self.path + name + self.filetype,
                                                  preprocess=self.doPreProcess)
