# -*- coding: utf-8 -*-
'''
This is a file for the generic helper functions we will need
'''
from joblib import dump, load
import copy
import json


def dump_model(model_dict, config_dict):
   '''
   Dump the model to two files: a .txt file of the parameters, and a .modlib file of the actual model
   :param model_dict: The output of the choose_model file
   :return:
   '''
   ##Dump the model
   dump(model_dict['model'], '{}/{}'.format(config_dict['projectPath'],config_dict['saved_model_target']))
   out_dict = copy.deepcopy(model_dict)
   out_dict.pop('model')
   ##now the imputer if we didn't use a nominal method
   if config_dict['imputation_method']=='Imputer':
       dump(model_dict['imputer'], '{}.imp'.format(config_dict['saved_model_target']))
       out_dict.pop('imputer')
       ###dump the remaining to the file
       with open('{}/{}.txt'.format(config_dict['projectPath'],config_dict['saved_model_target'].replace('.joblib','')), 'w') as file:
           file.write(json.dumps(out_dict))
   else:
       with open('{}/{}.txt'.format(config_dict['projectPath'],config_dict['saved_model_target'].replace('.joblib','')), 'w') as file:
           file.write(json.dumps(out_dict))

def load_model(config_dict, model_name):
    '''This function loads a model given the CONFIG and the particular model name
    '''
    ##load the base
    f = open('{}/{}.txt'.format(config_dict['projectPath'],model_name.replace('.joblib','')))
    out_dict=json.load(f)
    out_dict['model'] = load('{}/{}'.format(config_dict['projectPath'],model_name))
    if config_dict['imputation_method']=='Imputer':
        out_dict['imputer'] = load('{}/{}.imp'.format(config_dict['projectPath'],model_name))
    return out_dict

if __name__=='__main__':
    print('boo')

