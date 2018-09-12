import os
from configobj import ConfigObj


class Document:
    
    def __init__(self, id, path):
        
        self.id = id
        self.path = path


class Dataset:
    
    def __init__(self, id, path, language):
        
        self.id = id
        self.path = path
        self.language = language
        self.docs = []
        for id in os.listdir(path):
            doc = Document(id, os.path.join(path, id))
            self.docs.append(doc)
        

class DatasetConfig:
    
    def __init__(self, filename):
        ''' Dataset object constructor. '''
        
        assert(os.path.exists(filename))
        
        self.config = ConfigObj(filename)
        self.datasets = {}
        for id in self.config:
            entry = self.config[id]
            dataset = Dataset(entry['id'], entry['path'], entry['language'])
            self.datasets[entry['id']] = dataset