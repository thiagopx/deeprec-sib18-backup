import os
from configobj import ConfigObj
from convert import str2num, str2bool
from operator import itemgetter


def process_params(params):
    ''' Convert string params to float, int or bool. '''
            
    def _convert_param(param):
        ''' Convert a single parameter. '''
        
        try:
            return str2bool(param)
        except ValueError:
            return str2num(param)

            
    params = {
        key : _convert_param(value) for key, value in params.items()
    }
    return params
    

class Algorithm:
    
    def __init__(self, id, classname, color, legend, params):
        
        self.id = id
        self.classname = classname
        self.color = color
        self.legend = legend
        self.params = process_params(params)


class AlgorithmsConfig:
    
    def __init__(self, filename):
        
        assert(os.path.exists(filename))
        
        self.config = ConfigObj(filename)
        temp = [(id, int(data['position']))
                 for id, data in self.config.items()
                 if str2bool(data['active'])]
        ids, _ = zip(*sorted(temp, key=itemgetter(1)))
        self.algorithms = {}
        self.algorithms_list = []
        for id in ids:
            entry = self.config[id]
            algorithm = Algorithm(
                id, entry['classname'], entry['color'],
                entry['legend'], entry['params']
            )
            self.algorithms[id] = algorithm
            self.algorithms_list.append(self.algorithms[id])
            
            