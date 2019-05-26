# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03 13:18:58 2014

@author: hdemarch
"""
import pandas as pd
prefix = 'exp_market'
dic_input_params.update({'prefix': prefix})
# if linux
# dic_input_params.update({'root': '/mnt/research-safe/fellows/hdemarch/'})
dic_input_params.update({'root': 'H:\dev_python\projects\HJB Market orders\data\ '.strip()})
print "storing properties in <%s>..." % ('%(root)s%(prefix)s_%(ExperimentNumber)d.h5' % dic_input_params) 
storage = pd.HDFStore('%(root)s%(prefix)s_%(ExperimentNumber)d.h5' % dic_input_params) 
storage['ref'] = pd.DataFrame([dic_input_params])
storage.close()

class Storage(object):
    def __init__(self ):
        self.dic_input_params = None
        pass

    def store_results(self, dic_input_params, dic_df):
        """
        store in my hdf5 file some matrices
        
        :param dic_input_params: my reference
        :param dic_df: a dictionary of matrices to stor {'dEU_d': array1, 'dEU_dQ': array2}
        """
        storage = pd.HDFStore('%(root)s%(prefix)s_%(ExperimentNumber)d.h5' % dic_input_params) 
        for k, df in dic_df.items():
            print "storing <%s> in <%s>..." % ( k,'%(root)s%(prefix)s_%(ExperimentNumber)d.h5' % dic_input_params) 
            storage[k] = df
        storage.close()
        
        self.dic_input_params = dic_input_params
        
    def list(self, dic_input_params=None):
        if dic_input_params is None:
            dic_input_params = self.dic_input_params
        storage = pd.HDFStore('%(root)s%(prefix)s_%(ExperimentNumber)d.h5' % dic_input_params)
        print storage
        k = storage.keys()      
        storage.close()
        return [u[1:] for u in k]
        
my_storage = Storage()