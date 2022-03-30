# -*- coding: utf-8 -*-
import sys
import os
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
var_types = pd.read_csv('{}/mamba_variable_types.csv'.format(projectPath),keep_default_na=False).replace({'':None}).to_dict('records')
for k in var_types:
    for key in [CONFIG['data1_name'],CONFIG['data2_name'],'variable_name']:
        k[key] = k[key].lower()


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