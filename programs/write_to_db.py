'''
This is a function to write data to databases to clean up the code
'''

from psycopg2.extras import execute_values
import programs.global_vars as global_vars
from programs.connect_db import *
from programs.logger_setup import *
import pandas as pd
from sqlalchemy import create_engine
logger=logger_setup(CONFIG['log_file_name'])

def write_to_db(the_data, target_table):
    '''
    This is the function to write to the database
    :param the_data: either a dictionary or list of dictionaries
    :param target_table: the table to target
    :param exists: What to do if the table is alredy present (for sqlite only)
    :return:
    '''
    ###connect to the database
    db = get_db_connection(CONFIG)
    cur = db.cursor()
    try:
        ####now set up the list of columns
        if isinstance(the_data, list)==True:
            ##remove any index column, assemble into a list
            columns = [k for k in the_data[0].keys() if k!='index']
            columns_list = tuple(column for column in columns)
            values = [tuple(i[column] for column in columns) for i in the_data]
        else:
            columns_list = str(tuple([str(j) for j in the_data.keys() if j != 'index'])).replace("'", '')
            values = [tuple([the_data[key] for key in the_data if key in columns_list])]
        ###now write to the individual types of database
        val_len = len(values[0])
        if CONFIG['sql_flavor'] == 'sqlite':
            ###get the length of the values
            insert_statement = '''insert into {} values ({})'''.format(target_table, ", ".join('?'*val_len))
            cur.executemany(insert_statement,values)
            db.commit()
            cur.close()
            db.close()
        elif CONFIG['sql_flavor'] == 'postgres':
            insert_statement = '''insert into {} {} VALUES %s'''.format(target_table,columns_list)
            execute_values(cur, insert_statement, values)
            db.commit()
            db.close()
    except Exception as error:
        logger.info("Error in writing to Database.  Error {}".format(error))
        db.close()
        return the_data

if __name__=='__main__':
    print('Why did you do this?')
    os._exit(0)