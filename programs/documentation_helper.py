from programs.febrl_methods import editdist, qgram3
import pandas as pd
import sqlite3
import os
my_db = sqlite3.connect('{}/Documentation/variable_filter_test.db'.format(os.getcwd()))

def get_table_noconn(qry, db):
    #logger.info('Query: ' + search_path + ', ' + qry)
    cur = db.cursor()
    cur.execute(qry)
    columns = [i[0] for i in cur.description]
    results = [dict(zip(columns, row)) for row in cur]
    return results

def run_block_fun(block, qry=''):
    '''
    The function that pulls the block data from the db on a simple join, then runs the function (just testing the length here)
    :param block: block identifier
    :return:
    '''
    if qry=='':
        qry = ('''select a.first_name as left_firstname, a.dob as left_dob, b.first_name as right_firstname, b.dob as right_dob from 
               (select * from data1 where {blockvar} = '{block}') a left join (select * from data2 
               where {blockvar} = '{block}') b on a.{blockvar}=b.{blockvar}''').format(**{'blockvar':list(block.keys())[0], 'block':block[list(block.keys())[0]]})
    my_data = pd.DataFrame(get_table_noconn(qry, my_db))
    if len(my_data) > 0:
        ##get the length
        qgram_call = my_data.apply(lambda x: qgram3(x['left_firstname'], x['right_firstname']),1)
        dob_call = my_data.apply(lambda x: editdist(x['left_dob'], x['right_dob']),1)
    return len(my_data)