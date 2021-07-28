# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
###turn off chain warnings
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import time
import itertools as it
import re
import sklearn.utils
import datetime as dt
from sqlalchemy import create_engine
from scipy import stats as stats
import sqlite3
import importlib as imp
import timeit
import multiprocessing
import unicodedata
import random
import csv
import ast
import glob as glob
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import chi2, SelectKBest, RFECV
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve
from sklearn import metrics as metrics
import logging
from scipy.stats import randint as sp_randint
import string
#from runner_block import *
from multiprocessing import Pool
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
import re

prediction = ast.literal_eval(str(os.environ['prediction']))
#trainingdata = str(os.environ['trainingdata'])
#clericalreview = ast.literal_eval(str(os.environ['clericalreview']))
if ast.literal_eval(os.environ['use_logit'])==True:
    scoringcriteria='accuracy'
else:
    scoringcriteria = str(os.environ['scoringcriteria'])


debug = ast.literal_eval(str(os.environ['debugmode']))
logging.info('Scoring Criteria: {0}'.format(scoringcriteria))
logging.info('Debug Mode? {0}'.format(debug))

from programs.connect_sqlite import *
from programs.create_db_helpers import *
from programs.general_helpers import *
import programs.febrl_methods as feb

from inspect import getmembers, isfunction

############
###CHUNK: SET DATA AND PROGRAM PATH, TIME, AND IMPORT FEBRL
############
inputPath = os.environ['inputPath']
outputPath=os.environ['outputPath']
##Check if it ends in a /, if it does, leave it, otherwise add a /
if inputPath[-1:] == '/':
    inputPath = inputPath
else:
    inputPath = inputPath + '/'

###Repeating for output path
if outputPath[-1:] == '/':
    outputPath = outputPath
else:
    outputPath = outputPath + '/'

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
numWorkers=int(os.environ['numWorkers'])
# Open log file
# logging.basicConfig(filename=outputPath + 'br_match_log{}.log'.format(date), level=logging.DEBUG)
# ############
# ##The parameters for the run, called from the .bash file
# ##NOTE--order matters here, so don't change the order of the arguments in the .bash file
# logging.info('############################')
# logging.info('############################')
# logging.info('###### Starting Main Match Program ######')
# logging.info('############################')
# logging.info('############################')
#
# logging.info('\n\n###### PARAMS ######\n\n')
# #logging.info('Number of workers: {0}'.format(os.environ['numWorkers']))
# #logging.info('Number of CPUs: {0}'.format(os.environ['numcpus']))
# #logging.info('Memory Usage: {0}'.format(os.environ['memusage']))
#
#
# ###get the list of blocks we are looking for
# blocks=get_block_variable_list(os.environ['block_file_name'])
# logging.info('Blocks Used:')
# for i in blocks:
#     logging.info('{}'.format(i['block_name']))
###import the blocks and variable types
var_types = pd.read_csv('mamba_variable_types.csv').to_dict('record')
blocks = pd.read_csv('block_names.csv').to_dict('record')
###create the address_component_tag_mapping
address_components = pd.read_csv('address_component_mapping.csv').to_dict('record')
###makte the address component mapping.
address_component_mapping={}
for component in [add['address_component'] for add in address_components]:
    if component in [block['block_name'] for block in blocks] or component in [var['variable_name'] for var in var_types]:
        address_component_mapping[component] = component
    else:
        address_component_mapping[component] = 'address1'


if __name__=='__main__':
    print('why did you do this?')