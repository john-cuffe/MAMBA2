# -*- coding: utf-8 -*-

import sqlite3
import time

def get_connection_sqlite(db_name, timeout=1):

    db = sqlite3.connect(db_name, timeout=timeout)
    return db



def get_table(qry, db_name):
    #logger.info('Query: ' + search_path + ', ' + qry)
    db = get_connection_sqlite(db_name)
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