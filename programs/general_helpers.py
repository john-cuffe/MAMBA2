# -*- coding: utf-8 -*-
'''
This is a file for the generic helper functions we will need
'''
import pandas as pd


def logTime():
    t = time.time() - t0
    logging.info('Minutes lapsed: {0}'.format(round(t/60, 2)))

def get_block_variable_list(filename):
    '''
    This function grabs the list of blocking variables we are going to use
    :param filename: the filename we are targeting
    :return:
    '''
    return pd.read_csv(filename).to_dict('record')

if __name__=='__main__':
    print('boo')

