# -*- coding: utf-8 -*-

'''
################################################################################
# Author: John Cuffe and Nathan Goldschlag
# Date: 06/2019
# Description: Mamba 2.0.  Simpler.  Mamba-ier
################################################################################
'''
from programs.create_db_helpers import *
from programs.global_vars import *

###Set data and program paths
outputPath=CONFIG['outputPath']

##debugmode will run on a 1% sample of each side of the data
debugmode = ast.literal_eval(str(CONFIG['debugmode']))

###get the list of blocks we are looking for
blocks=get_block_variable_list(CONFIG['block_file_name'])

if __name__ == '__main__':
    logging.info('Date Time: {0}'.format(date))
    createDatabase(CONFIG['db_name'])

