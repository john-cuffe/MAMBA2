# -*- coding: utf-8 -*-

'''
This is a list of helper functions to create our database
'''
import programs.global_vars as global_vars
from programs.connect_db import *
from programs.logger_setup import *
from programs.write_to_db import write_to_db
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
logger=logger_setup(CONFIG['log_file_name'])
if CONFIG['sql_flavor']=='postgres':
    from psycopg2.extras import execute_values
from glob import glob
###check if we have local usaddress or a proper version
if ast.literal_eval(CONFIG['parse_address'])==True:
    if 'programs/usaddress' in glob("programs/*/"):
        import usaddress as usadd
    else:
        import usaddress as usadd
    from programs.address_standardizer.standardizer import *
###a query to create the batch_summary and batch_statistics tables
batch_info_qry='''
create table batch_summary(
batch_id integer primary key autoincrement,
batch_started timestamp,
batch_completed timestamp,
batch_status text,
batch_config text,
failure_message text);

create index batch_summary_batch_idx on batch_summary(batch_id);

create table batch_statistics(
batch_id bigint,
block_level text,
block_id text,
block_time float,
block_size bigint,
block_matches bigint,
block_matches_avg_score float,
block_non_matches bigint,
block_non_matches_avg_score float,
match_pairs_removed_filter bigint);

create index batch_stat_idx on batch_statistics(batch_id);

create table block_model_mapping(
batch_id bigint,
block_id text,
model_name text
);

create index block_mapping_idx on block_model_mapping(block_id);
'''

seq_query='''CREATE SEQUENCE batch_summary_query
INCREMENT 1
START 1
MINVALUE 1
MAXVALUE 9223372036854775807
CACHE 1;'''

###Alter if we aren't dealing with an sqlite database
if CONFIG['sql_flavor']!='sqlite':
    batch_info_qry = seq_query + '\n' + batch_info_qry.replace('\n',' ')

def intersect(x0, x1, y0, y1):
    '''
    This function generates  a true or false flag for the intersection
    inputs:
    x0--first year on one side
    x1--last year on one side
    yy0-first year on other side
    y1--last year on other side
    '''
    r1=set(range(x0, x1+1))
    r2=set(range(y0, y1+1))
    intersect=r1.intersection(r2)
    if len(intersect) > 0:
        return True
    else:
        return False

def generate_batch_id(db):
    '''
    This function will query the central mojo db to generate our new batch number.
    :param db: the database connection
    :return: batch number to use
    '''
    cur = db.cursor()
    if CONFIG['sql_flavor']=='sqlite':
        out=get_table_noconn('''select count(*) cnt from sqlite_master where type='table' and name='batch_summary' ''', db)[0]
        if out['cnt'] > 0:
            cur.execute('''select max(batch_id)+1 batch_id from batch_summary''')
        else:
            return 1
    else:
        cur.execute('''select coalesce(max(batch_id)+1,1) batch_id from batch_summary''')
    #
    columns = [i[0] for i in cur.description]
    last_batch = [dict(zip(columns, row)) for row in cur]    #
    if last_batch[0]['batch_id']:
        return int(last_batch[0]['batch_id'])
    else:
        return 1

def stem_fuzzy(x):
    if x=='NULL':
        return None
    else:
        return stemmer.stem(x)

def create_table(dataname,column_types):
    '''
    Create a table given the data snippet
    :param dataname: name of the table
    :param column_types: the types of columns
    :return:
    '''
    try:
        db = get_db_connection(CONFIG)
        cur = db.cursor()
        col_dict=column_types.to_dict()
        sql_trans={}
        for c in col_dict:
            if col_dict[c] in ['O']:
                sql_trans[c]='text'
            elif col_dict[c]=='int64':
                sql_trans[c]='int'
            elif col_dict[c]=='float64':
                sql_trans[c]='float'
        ###Make the full statement
        #print(sql_trans)
        col_stmt = ', '.join(['{} {}'.format(k, sql_trans[k]) for k in sql_trans])
        if CONFIG['sql_flavor']=='postgres':
            col_stmt = col_stmt.replace('int','bigint')
        full_statement = 'create table {} ({})'.format(dataname, col_stmt)
        cur.execute(full_statement)
        db.commit()
        return 'success'
    except Exception as error:
        logger.info('Failed to Create DB Table {}, error {}'.format(dataname, error))
        return 'fail'

def get_stem_data(data_source):
    '''
    This function loads and then stems the data
    :param dataname:
    :return:
    '''
    ##get the CSV name
    ###if training is in the dataname, strip it out
    dataname = data_source.split('_training')[0]
    # the the fuzzy matches
    ###dictionary to read all blocks as strings
    str_dict = {item[dataname]: str for item in global_vars.blocks}
    str_dict['id'] = str
    ###If we have a date variable, find it and ensure it's read as a string
    date_columns = [i[dataname] for i in global_vars.var_types if i['match_type'] == 'date']
    if len(date_columns) > 0:
        for col in date_columns:
            str_dict[col]=str
    ###Get list of the fuzzy variables
    fuzzy_vars = [i[dataname] for i in global_vars.var_types if i['match_type'].lower() == 'fuzzy']
    ###add address1 if we are using the remaining parsed addresses
    if ast.literal_eval(CONFIG['use_remaining_parsed_address']) == True:
        ###append the address 1 to the fuzzy vars
        fuzzy_vars.append('address1')
    ##chunk data so we don't blow up memory
    if sys.platform == 'win32':
        csvname = '{}\\{}.csv'.format(CONFIG['projectPath'], dataname)
    else:
        csvname = '{}/{}.csv'.format(CONFIG['projectPath'], dataname)
    ###flagging if the table exists
    table_exists=False
    row_number=0
    for data in pd.read_csv(csvname, chunksize=int(CONFIG['create_db_chunksize']), engine='c', dtype=str_dict):
        #print('foo {}'.format(table_exists))
        ####If we have addresses to standardize for the data source
        if dataname in CONFIG['address_to_standardize'].keys():
            cols = CONFIG['address_to_standardize'][dataname].split(',')
            for col in cols:
                data[col] = data[col].apply(lambda x: standardize(x, 'n'))
        data['matched'] = 0
        for block in global_vars.blocks:
            #print(block['block_name'])
            ###for 'full' blocks, we will divide everything into equally sized chunks.  This is much easier to do on the front end while we are writing
            ###than to deal with later when we dn't want to pull the entire DB and have SQL-dependencies
            ###And yes I mean it's easier for me. Sue me.
            if block['block_name'].lower()=='full':
                row_seq = [np.floor(i/int(block['chunk_size'])) for i in list(range(row_number, row_number+len(data)))]
                data['full'] = row_seq
                row_number += len(data)
        ###convert to dictionary
        #print('got the full')
        data = data.to_dict('records')
        ########
        ##Are we using the parsed address feature? if so, add those variables into the row
        ########
        if ast.literal_eval(CONFIG['parse_address']) == True:
            ###For each row, parse the address
            for row in data:
                parsed_address = usadd.tag(row[CONFIG['address_column_{}'.format(dataname)]],
                                           tag_mapping=global_vars.address_component_mapping)
                parsed_address = {k.lower(): v for k, v in parsed_address[0].items()}
                ###convert all the keys to lower
                for parsed_block in [v for v in global_vars.blocks if v['parsed_block'] == 1]:
                    if len(re.findall('zipcode[0-9]', parsed_block['block_name'])) > 0:
                        row[parsed_block['block_name']] = parsed_address['zipcode'][
                                                          0:int(parsed_block['block_name'][-1])]
                    else:
                        row[parsed_block['block_name']] = parsed_address[parsed_block['block_name']]
                for variable in [var for var in global_vars.var_types if var['parsed_variable'] == '1']:
                    if variable['variable_name'] in parsed_address.keys():
                        row[variable['variable_name']] = parsed_address[variable['variable_name']]
                    else:
                        row[variable['variable_name']] = None
                if ast.literal_eval(CONFIG['use_remaining_parsed_address']) == True:
                    if 'address1' in parsed_address[0].keys():
                        row['address1'] = parsed_address[0]['address1']
                    else:
                        row['address1'] = None
        for p in fuzzy_vars:
            for r in range(len(data)):
                ###fill in NULL for the fuzzy vars
                if pd.isna(data[r][p]) or data[r][p] is None:
                    data[r][p] = 'NULL'
                data[r][p] = str(data[r][p]).upper()
                if 'zip' in p.lower():
                    ###find the max length
                    out = data[r][p].astype(str).tolist()
                    ###get the max length
                    maxlen = max([len(k) for k in out])
                    ###log
                    logger.info(
                        '''Variable {} for data {} is likely to be a zipcode variable.  Converted to a zero-filled string.  If you didn't want this to happen, change the variable name to not include 'zip' '''.format(
                            p, dataname))
                    data[r][p] = data[r][p].astype(str).str.zfill(maxlen)
        # 2) Standardized via stemming any fuzzy matching variables
        if ast.literal_eval(CONFIG['stem_phrase']) == True:
            for var in fuzzy_vars:
                ###first convert to all upper case
                ###get the list of the entries
                to_stem = [j[var] for j in data]
                ###to get through our stemmer, we need to convert nulls to
                out = []
                for k in to_stem:
                    out.append(stem_fuzzy(k))
                # out=pool.map(stem_fuzzy, to_stem)
                ###now we have it, replace the original value
                for i in range(len(data)):
                    data[i][var] = out[i]
        # 3) check if table exists
        ###convert all column headers to lower case
        data = [{k.lower(): v for k, v in x.items()} for x in data]
        ####First a quick check on if the table exists.  If not, create it.
        if table_exists==False:
            #print('creating table')
            #print(pd.DataFrame(data).dtypes)
            outcome = create_table(data_source,pd.DataFrame(data).dtypes)
            if outcome =='success':
                table_exists=True
            else: break
        # 4) Push to DB
        if ast.literal_eval(CONFIG['prediction']) == True:
            write_out= write_to_db(data, data_source)
            if write_out:
                logger.info('Failed to write data to database, breaking')
                break
        if ast.literal_eval(CONFIG['prediction']) == False and ast.literal_eval(CONFIG['clerical_review_candidates']) == True:
            data = pd.DataFrame(data).sample(frac=.05).to_dict('record')
            write_out= write_to_db(data, data_source)
            if write_out:
                logger.info('Failed to write data to database, breaking')
                break

def createDatabase(databaseName):
    # create individual data tables
    # ###for each input dataset, need to
    for data_source in [CONFIG['data1_name'],CONFIG['data2_name'], '{}_training'.format(CONFIG['data1_name']), '{}_training'.format(CONFIG['data2_name'])]:
        try:
            if os.path.isfile('{}/{}.csv'.format(CONFIG['projectPath'],data_source))==True:
                #print(data_source)
                get_stem_data(data_source)
                ####now index the tables
                db=get_db_connection(CONFIG)
                cur=db.cursor()
                for i in global_vars.blocks:
                    cur.execute('''create index {source}_{variable}_idx on {source} ({variable});'''.format(variable=i[data_source.split('_training')[0]], source=data_source))
                ###additional index on id
                cur.execute('''create index {}_id_idx on {} (id)'''.format(data_source, data_source))
                ###clerical_review_candidates and the two matched pairs table
                db.commit()
            else:
                logger.info('File {} does not exist.  Passing'.format(data_source))
                ###Handbreak--if clerical review candidates/matched pairs exist, change their names to the current date/time
        except Exception as error:
            logger.info("Error creating table {}, error {}".format(data_source, error))
    '''
    Removing this: From now on, will just attach the batch_id to the inputs
    '''
    # for table in ['clerical_review_candidates','matched_pairs']:
    #     if CONFIG['sql_flavor']=='sqlite':
    #         ret=get_table_noconn('''SELECT name FROM sqlite_master WHERE type='table' AND name='{}' '''.format(table), db)
    #     elif CONFIG['sql_flavor']=='postgres':
    #         ret=get_table_noconn('''SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='{}' '''.format(table), db)
    #     if len(ret) > 0:
    #         cur.execute('''alter table {} rename to {}{}'''.format(table,table,dt.datetime.now().strftime('%Y_%M_%d_%H_%m')))
    #         db.commit()
    cur.execute('''create table clerical_review_candidates ({}_id text, {}_id text, predicted_probability float, threshold_value float, batch_id float);'''.format(CONFIG['data1_name'], CONFIG['data2_name']))
    cur.execute('''create table matched_pairs ({data1}_id bigint, {data2}_id bigint, predicted_probability float, {data1}_rank float, {data2}_rank float, batch_id float, match_pair_info json);'''.format(data1=CONFIG['data1_name'], data2=CONFIG['data2_name']))
    ###Finally, if the project path has a 'block_model_mapping.csv' file, dump that into the database with a batch identifier.
    if os.path.exists('{}/block_model_mapping.csv'.format(CONFIG['projectPath'])):
        ###load it
        mapping_dat = pd.read_csv('{}/block_model_mapping.csv'.format(CONFIG['projectPath']), engine='c', dtype={'block_id':str}).to_dict('records')
        if len(mapping_dat) > 0:
            for k in mapping_dat:
                k['batch_id'] = CONFIG['batch_id']
                if k['model_name'][-7:]!='.joblib':
                    k['model_name']=k['model_name']+'.joblib'
            ###dump into database
            write_out=write_to_db(mapping_dat,'block_model_mapping')
        else:
            logger.info('block_model_mapping.csv file present but no data found.  Skipping.')
    db.commit()

def update_batch_summary(batch_summary):
    '''
    This function will update the batch summary as we go for each value in batch_summary
    :param batch_summary: the dictionary we are using
    :return:
    '''
    db=get_db_connection(CONFIG)
    cur=db.cursor()
    ###two cases...this is a new batch (so batch id doesn't exist) or the batch already exists
    batch_exists=get_table_noconn('''select * from batch_summary where batch_id={}'''.format(batch_summary['batch_id']), db)
    if CONFIG['sql_flavor']=='sqlite':
        if len(batch_exists)==0:
            # create individual data tables
            diskEngine = create_engine('sqlite:///{}/{}.db'.format(CONFIG['projectPath'],CONFIG['db_name']))
            ###IF the batch doesn't exist, make the row (it's just the batch status and batch_id)
            pd.DataFrame(batch_summary, index=[0]).to_sql('batch_summary', diskEngine, if_exists='append', index=False)
            db.commit()
        else:
            ##First check for any json and do a json.dumps
            for key in batch_summary:
                if batch_summary[key].__class__==dt.datetime:
                    batch_summary[key] = batch_summary[key].strftime('%Y-%m-%d %H:%M:%S')
                if batch_summary[key].__class__==dict:
                    batch_summary[key]=json.dumps(batch_summary[key])
                if batch_summary[key]!=batch_exists[0][key]:
                    update_statement='update batch_summary set {}=? where batch_id={}'.format(key, batch_summary['batch_id'])
                    try:
                        cur.execute(update_statement, (batch_summary[key],))
                        db.commit()
                    except Exception as err:
                        print(err)
                        print(batch_summary[key])
    elif CONFIG['sql_flavor']=='postgres':
        if len(batch_exists) == 0:
            ###IF the batch doesn't exist, make the row (it's just the batch status and batch_id)
            columns = batch_summary.keys()
            columns_list = str(tuple([str(i) for i in columns])).replace("'", "")
            values = [tuple(batch_summary[key] for key in batch_summary)]
            insert_statement = 'insert into {table} {collist} values %s'.format(table='batch_summary',
                                                                                collist=columns_list)
            execute_values(cur, insert_statement, values, page_size=len(values))
            db.commit()
        else:
            ##First check for any json and do a json.dumps
            for key in batch_summary:
                if batch_summary[key].__class__ == dict:
                    batch_summary[key] = json.dumps(batch_summary[key])
            columns = [i for i in batch_summary.keys() if i != 'batch_id']
            columns_statement = ', '.join(['{}=%s'.format(i) for i in columns])
            values = [tuple(batch_summary[key] for key in batch_summary if key != 'batch_id')]
            update_statement = 'update batch_summary set {} where batch_id={}'.format(
                columns_statement, batch_summary['batch_id'])
            cur.execute(update_statement, values[0])
            db.commit()
    cur.close()
    db.close()

if __name__=='__main__':
    print('why did you do this?')
