# -*- coding: utf-8 -*-

'''
################################################################################
# Author: John Cuffe and Nathan Goldschlag
# Date: 06/2019
# Description: Mamba 2.0.  Simpler.  Mamba-ier
################################################################################
'''
import os
import sys
from subprocess import call
import logging
import csv
import pandas as pd
import numpy as np
import datetime as dt
import time
import ast
import multiprocessing
import sqlite3
from sqlalchemy import create_engine
import nltk
import sklearn.feature_extraction.text as txt
import itertools as it
from programs.global_vars import *
from programs.create_db_helpers import *
from programs.general_helpers import *
from programs.connect_sqlite import *
###Set data and program paths
outputPath=CONFIG['outputPath']

##debugmode will run on a 1% sample of each side of the data
debugmode = ast.literal_eval(str(CONFIG['debugmode']))

###get the list of blocks we are looking for
blocks=get_block_variable_list(CONFIG['block_file_name'])

if __name__ == '__main__':
    logging.info('Date Time: {0}'.format(date))
    createDatabase(CONFIG['db_name'])

