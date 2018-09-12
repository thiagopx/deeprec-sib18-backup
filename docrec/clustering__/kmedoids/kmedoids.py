from algorithms.pam import pam
from algorithms.clarans import clarans
from itertools import combinations
import numpy as np


class KMedoids:
    ''' K-Medoids clustering.
    
    References:
    [1] http://www.mathworks.com/help/stats/kmedoids.html
    [2] https://en.wikipedia.org/wiki/K-medoids
    '''

    def __init__(
        self, algorithm='clarans', init='random', seed=None, verbose=False,
        **kwargs
    ):
        
        assert algorithm in ['pam', 'clarans']

        # General parameters
        self.algorithm = algorithm
        self.init = init
        self.seed = seed
        self.verbose = verbose

        # Algorithm functions
        self.functions = {'pam': pam, 'clarans': clarans}

        # Custom parameters
        self.custom_params = \
            {'pam': {'max_it': 100},
             'clarans': {'num_local': 2, 'max_neighbor': 200}
            }

        # Adjusting parameters
        for key, value in kwargs.items():
            if key in self.custom_params[algorithm]:
                self.custom_params[algorithm][key] = value
            else:
                raise Exception('%s: parameter %s invalid for algorihtm %s.'
                                % (self.__init__.__name__, key, algorithm))

        self.clusters = None
        self.medoids = None
        self.cost = None
        self.X = None

        self.uniformity = None
        self.uniformity_avg = None
        self.uniformity_std = None


    def run(self, X, k):
        ''' Run clustering. '''
        
        function = self.functions[self.algorithm]
        custom_params = self.custom_params[self.algorithm]

        # Run kmedoids function
        self.clusters, self.cost = function(
            X, k, init=self.init, seed=self.seed, verbose=self.verbose,
            **custom_params
        )
        
        # Uniformity computation
        f_unf = lambda cluster: float('inf') if len(cluster) < 2  else \
            np.mean([X[i, j] for i, j in combinations(cluster, 2)])
            
#        f_unf = lambda cluster: float('inf') if len(cluster) < 1  else \
#            np.mean([X[i, j] for i, j in combinations(cluster, 2)])

        values = map(f_unf, self.clusters)
        data = zip(self.clusters, values)
        filtered_data = filter(lambda item: item[1] != float('inf'), data)
        self.clusters, self.uniformity = zip(*filtered_data)
        self.medoids = [cluster[0] for cluster in self.clusters]
        self.uniformity_avg = np.array(self.uniformity).mean()
        self.uniformity_std = np.array(self.uniformity).std()
        return self
    
    