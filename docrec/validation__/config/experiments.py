import os
from configobj import ConfigObj
from time import time
from dataset import DatasetConfig
from algorithms import AlgorithmsConfig


class ExperimentsConfig:
    
    def __init__(self, filename, dset_filename, alg_filename):
                
        assert(os.path.exists(filename))
        
        self.dataset_config = DatasetConfig(dset_filename)
        self.algorithms_config = AlgorithmsConfig(alg_filename)
        
        self.config = ConfigObj(filename)
        self.nruns = int(self.config['nruns'])
        self.ndisp = int(self.config['ndisp'])
        self.email = self.config['email']
        self.pwd = self.config['pwd']
        self.cache = self.config['cache']
        self.results = self.config['results']
        self.thresholds = map(float, self.config['thresholds'].split())
        self._build_cache()
        self._build_results()
        
        self.seed = self._get_seed()
        
    
    def _make_dir(self, path):
        ''' Make a directory if not existing. '''
        
        if not os.path.exists(path):
            os.makedirs(path)
    
    
    def _build_cache(self):
        ''' Build the experiment cache directory structure. '''
                
        self._make_dir(self.path_cache())


    def _build_results(self):
        ''' Build the experiment results directory structure. '''
                
        self._make_dir(self.path_results())


    def _get_seed(self):
        ''' Return the saved seed (if existing) or generate a new one. '''
    
        filename = self.path_cache('seed.txt')
        if os.path.exists(filename):
            return int(open(filename, 'r').read())
        seed = int(time())
        open(filename, 'w').write(str(seed))
        return seed
    
    
    def path_cache(self, *names):
        ''' Build a tree directory structure in cache. '''
        
        return os.path.join(self.cache, *names)


    def path_results(self, *names):
        ''' Build a tree directory structure in results. '''
        
        return os.path.join(self.results, *names)
