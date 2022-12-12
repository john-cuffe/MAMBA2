# -*- coding: utf-8 -*-
import sys
import os
import re
sys.path.append(os.getcwd())
import time
import pandas as pd
###turn off chain warnings
pd.options.mode.chained_assignment = None  # default='warn'
import ast
#from sklearn.preprocessing import Imputer
import logging
#from runner_block import *
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
####Load the CONFIG
##get properties
from programs.config import *
CONFIG = {}
############
###CHUNK: SET DATA AND PROGRAM PATH, TIME, AND IMPORT FEBRL
############
####
projectPath = sys.argv[1]
##Check if it ends in a /, if it does, leave it, otherwise add a /
if projectPath[-1:] == '/':
    projectPath = projectPath[:-1]

####read in the mamba properties file
read_properties('{}/mamba.properties'.format(sys.argv[1]), CONFIG)

import datetime as dt
CONFIG['date']=dt.datetime.now().date().strftime("%Y_%m_%d")


prediction = ast.literal_eval(str(CONFIG['prediction']))
#trainingdata = str(CONFIG['trainingdata'])
#clericalreview = ast.literal_eval(str(CONFIG['clericalreview']))
if ast.literal_eval(CONFIG['use_logit'])==True:
    scoringcriteria='accuracy'
else:
    scoringcriteria = str(CONFIG['scoringcriteria'])


debug = ast.literal_eval(str(CONFIG['debugmode']))
logging.info('Scoring Criteria: {0}'.format(scoringcriteria))
logging.info('Debug Mode? {0}'.format(debug))

import programs.febrl_methods as feb

###setting the start time
starttime = time.time()

# load the febrl source code
################
##Starting the program
################
start = time.time()
###This list of methods we will use
methods = [feb.jaro, feb.winkler, feb.bagdist, feb.seqmatch, feb.qgram2, feb.qgram3, feb.posqgram3, feb.editdist,
           feb.lcs2, feb.lcs3, feb.charhistogram, feb.swdist, feb.sortwinkler]
###The list of names for those methods
namelist = [i.__name__ for i in methods]
##date and time
date = dt.datetime.now().date()
numWorkers=int(CONFIG['numWorkers'])

###import the blocks and variable types
file = open('{}/mamba_variable_types.csv'.format(projectPath), mode='r', newline='\n')
var_types=[]
lines = file.readlines()
###Set up the keys
keys = lines[0].replace('\n','').split(',')
###little function to figure out if number is all digits, negative, or a .
def number_finder(s):
    return all(c in '0123456789.-' for c in s)

for line in lines[1:]:
    if len(line.replace('\n','')) > 0:
        ###get the first bit of the line
        line_split = re.split(',\s*(?![^{}]*\})', line.replace('\n',''))
        out_dict={}
        for key in range(len(keys)):
            if line_split[key]=='':
                out_dict[keys[key]] = None
            else:
                out_dict[keys[key]]=line_split[key]
        for key in [CONFIG['data1_name'], CONFIG['data2_name'], 'variable_name']:
            out_dict[key] = out_dict[key].lower()
        ###finally, do the json check
        if out_dict['custom_variable_kwargs'] is not None:
            out_dict['custom_variable_kwargs']=json.loads(out_dict['custom_variable_kwargs'])
            ###now, for each key, if it's only got +/- and number codes, convert to a float
            for mykey in out_dict['custom_variable_kwargs'].keys():
                if number_finder(out_dict['custom_variable_kwargs'][mykey])==True:
                    out_dict['custom_variable_kwargs'][mykey] = float(out_dict['custom_variable_kwargs'][mykey])
        var_types.append(out_dict)



blocks = pd.read_csv('{}/{}'.format(projectPath,CONFIG['block_file_name'])).fillna(-1).to_dict('records')
###quickly strip out any spaces
for block in blocks:
    for key in block.keys():
        if type(block[key])==str:
            block[key] = block[key].replace(' ','').lower()
###create the address_component_tag_mapping
address_components = pd.read_csv('Documentation/address_component_mapping.csv').to_dict('records')
###makte the address component mapping.###lower on blocks is the issue here
address_component_mapping={}
for component in [add['address_component'] for add in address_components]:
    component_lower=component.lower()
    if component_lower in [block['block_name'] for block in blocks] or component_lower in [var['variable_name'] for var in var_types]:
        address_component_mapping[component] = component
    else:
        address_component_mapping[component] = 'address1'


if __name__=='__main__':
    print('why did you do this?')