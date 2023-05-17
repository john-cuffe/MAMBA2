# -*- coding: utf-8 -*-

import sqlite3
import os
import psycopg2


def get_db_connection(db_info, timeout=1):
    if db_info['sql_flavor']=='sqlite':
        db = sqlite3.connect('{}/{}.db'.format(db_info['projectPath'],db_info['db_name']), timeout=timeout)
    elif db_info['sql_flavor']=='postgres':
        db = psycopg2.connect(dbname=db_info['db_name'],
                              host=db_info['db_host'],
                              port=db_info['db_port'],
                              user=db_info['db_user'],
                              password=db_info['db_password'],
                              options='-c search_path={}'.format(db_info['db_schema']),
                              connect_timeout=10)
    return db



def get_table(qry, db_info):
    #logger.info('Query: ' + search_path + ', ' + qry)
    db = get_db_connection(db_info)
    cur = db.cursor()
    cur.execute(qry)
    columns = [i[0] for i in cur.description]
    results = [dict(zip(columns, row)) for row in cur]
    cur.close()
    db.close()
    return results

###get table without the need to get a connection

def get_table_noconn(qry, db):
    #logger.info('Query: ' + search_path + ', ' + qry)
    cur = db.cursor()

    cur.execute(qry)

    columns = [i[0] for i in cur.description]
    results = [dict(zip(columns, row)) for row in cur]

    return results


if __name__=='__main__':
    print('boo')
