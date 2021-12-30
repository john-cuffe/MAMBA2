# -*- coding: utf-8 -*-
'''
This is a file for the generic helper functions we will need
'''
from joblib import dump, load
import copy
import json

def logTime():
    t = time.time() - t0
    logging.info('Minutes lapsed: {0}'.format(round(t/60, 2)))


def dump_model(model_dict, saved_model_target, imputation_method):
   '''
   Dump the model to two files: a .txt file of the parameters, and a .modlib file of the actual model
   :param model_dict: The output of the choose_model file
   :return:
   '''
   ##Dump the model
   dump(model_dict['model'], saved_model_target)
   out_dict = copy.deepcopy(model_dict)
   out_dict.pop('model')
   ##now the imputer if we didn't use a nominal method
   if imputation_method=='Imputer':
       dump(model_dict['imputer'], '{}.imp'.format(saved_model_target))
       out_dict.pop('imputer')
       ###dump the remaining to the file
       with open('{}.txt'.format(saved_model_target), 'w') as file:
           file.write(json.dumps(out_dict))
   else:
       with open('{}.txt'.format(saved_model_target), 'w') as file:
           file.write(json.dumps(out_dict))

def load_model(saved_model, imputation_method):
    '''This function loads a model
    '''
    ##load the base
    f = open('{}.txt'.format(saved_model))
    out_dict=json.load(f)
    out_dict['model'] = load(saved_model)
    if imputation_method=='Imputer':
        out_dict['imputer'] = load('{}.imp'.format(saved_model))
    return out_dict

if __name__=='__main__':
    print('boo')

