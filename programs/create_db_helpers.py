# -*- coding: utf-8 -*-

'''
This is a list of helper functions to create our database
'''
import programs.global_vars as global_vars
from programs.logger_setup import *
from programs.connect_db import *
import usaddress as usadd
from programs.address_standardizer.standardizer import *
from psycopg2.extras import execute_values
logger=logger_setup(CONFIG['log_file_name'])

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
batch_id text,
block_level text,
block_id text,
block_time float,
block_size bigint,
block_matches bigint,
block_matches_avg_score float,
block_non_matches bigint,
block_non_matches_avg_score bigint);

create index batch_stat_idx on batch_statistics(batch_id)
'''

seq_query='''CREATE SEQUENCE batch_summary_query
INCREMENT 1
START 1
MINVALUE 1
MAXVALUE 9223372036854775807
CACHE 1;'''

###Alter if we aren't dealing with an sqlite database
if CONFIG['sql_flavor']!='sqlite':
    batch_info_qry = seq_query + '\n' + batch_info_qry.replace('\n',' ').replace('batch_id integer primary key autoincrement','''bigint primary key DEFAULT nextval('hermes_central.regional_goals_id_seq'::regclass)''')

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
    cur.execute('''select max(batch_id)+1 batch_id from batch_summary''')
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


def get_stem_data(dataname):
    '''
    This function loads and then stems the data
    :param dataname:
    :return:
    '''
    ##get the CSV name
    # 1) Load
    output=[]
    # the the fuzzy matches
    ###dictionary to read all blocks as strings
    str_dict={item[dataname]:str for item in global_vars.blocks}
    ###Get list of the fuzzy variables
    fuzzy_vars = [i[dataname] for i in global_vars.var_types if i['match_type'].lower() == 'fuzzy']
    ###add address1 if we are using the remaining parsed addresses
    if ast.literal_eval(CONFIG['use_remaining_parsed_address']) == True:
        ###append the address 1 to the fuzzy vars
        fuzzy_vars.append('address1')
    ##chunk data so we don't blow up memory
    if sys.platform == 'win32':
        csvname='{}\\{}.csv'.format(CONFIG['inputPath'], dataname)
    else:
        csvname='{}/{}.csv'.format(CONFIG['inputPath'], dataname)
    for data in pd.read_csv(csvname, chunksize=int(CONFIG['create_db_chunksize']), engine='c',dtype=str_dict):
        ###If we have a date variable, find it and convert to a date
        date_columns=[i[dataname] for i in global_vars.var_types if i['match_type']=='date']
        if len(date_columns) > 0:
            for date_col in date_columns:
                data[date_col]=pd.to_datetime(data[date_col], format=CONFIG['date_format']).dt.strftime('%Y-%m-%d')
        ####If we have addresses to standardize, do so
        ###ID from config
        ####If we have addresses to standardize for the data source
        if dataname in CONFIG['address_to_standardize'].keys():
            cols = CONFIG['address_to_standardize'][dataname].split(',')
            for col in cols:
                data[col] = data[col].apply(lambda x: standardize(x, 'n'))
        data['matched'] = 0
        if 'full' in [b['block_name'] for b in global_vars.blocks]:
            data['full']=1
        ###convert to dictionary
        data = data.to_dict('record')
        ########
        ##Are we using the parsed address feature? if so, add those variables into the row
        ########
        if ast.literal_eval(CONFIG['parse_address'])==True:
            ###For each row, parse the address
            for row in data:
                parsed_address = usadd.tag(row[CONFIG['address_column_{}'.format(dataname)]], tag_mapping=global_vars.address_component_mapping)
                for parsed_block in [v for v in global_vars.blocks if v['parsed_block']==1]:
                    if len(re.findall('ZipCode[0-9]', parsed_block['block_name'])) > 0:
                        row[parsed_block['block_name']] = parsed_address[0]['ZipCode'][0:int(parsed_block['block_name'][-1])]
                    else:
                        row[parsed_block['block_name']] = parsed_address[0][parsed_block['block_name']]
                for variable in [var for var in global_vars.var_types if var['parsed_variable']==1]:
                    if variable['variable_name'] in parsed_address[0].keys():
                        row[variable['variable_name']] = parsed_address[0][variable['variable_name']]
                    else:
                        row[variable['variable_name']] = None
                if ast.literal_eval(CONFIG['use_remaining_parsed_address'])==True:
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
                    out=data[r][p].astype(str).tolist()
                    ###get the max length
                    maxlen=max([len(k) for k in out])
                    ###log
                    logger.info('''Variable {} for data {} is likely to be a zipcode variable.  Converted to a zero-filled string.  If you didn't want this to happen, change the variable name to not include 'zip' '''.format(p, dataname))
                    data[r][p]=data[r][p].astype(str).str.zfill(maxlen)
        # 2) Standardized via stemming any fuzzy matching variables
        if ast.literal_eval(CONFIG['stem_phrase'])==True:
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
        output.extend(data)
    return output


def createDatabase(databaseName):
    # create individual data tables
    # ###for each input dataset, need to
    for data_source in [CONFIG['data1_name'],CONFIG['data2_name']]:
        #print(data_source)
        out=get_stem_data(data_source)
        # 3) Push to DB
        if ast.literal_eval(CONFIG['prediction'])==True:
            if CONFIG['sql_flavor']=='sqlite':
                diskEngine = create_engine('sqlite:///' + databaseName)
                pd.DataFrame(out).to_sql(data_source, diskEngine, if_exists='replace',index=False)
            elif CONFIG['sql_flavor']=='postgres':
                db=get_db_connection(CONFIG)
                out = out.to_dict('record')
                columns = out[0].keys()
                values = [tuple(i[column] for column in columns) for i in out]
                # logger.info('Reports Written')
                columns_list = str(tuple([str(i) for i in columns])).replace("'", '')
                cur = db.cursor()
                insert_statement = 'insert into {table} {collist} values %s'.format(table=data_source,
                                                                                    collist=columns_list)
                execute_values(cur, insert_statement, values)
                db.commit()
        if ast.literal_eval(CONFIG['prediction'])==False and ast.literal_eval(CONFIG['clerical_review_candidates'])==True:
            if CONFIG['sql_flavor']=='sqlite':
                pd.DataFrame(out).sample(frac=.05).to_sql(data_source, diskEngine,if_exists='replace', index=False)
            elif CONFIG['sql_flavor'] == 'postgres':
                out = out.sample(frac=.05).to_dict('record')
                columns = out[0].keys()
                values = [tuple(i[column] for column in columns) for i in out]
                # logger.info('Reports Written')
                columns_list = str(tuple([str(i) for i in columns])).replace("'", '')
                cur = db.cursor()
                insert_statement = 'insert into {table} {collist} values %s'.format(table=data_source,
                                                                                    collist=columns_list)
                execute_values(cur, insert_statement, values)
                db.commit()
        ####now index the tables
        db=get_db_connection(CONFIG)
        cur=db.cursor()
        for i in global_vars.blocks:
            cur.execute('''create index {source}_{variable}_idx on {source} ({variable});'''.format(variable=i[data_source], source=data_source))
        ###additional index on id
        cur.execute('''create index {}_id_idx on {} (id)'''.format(data_source, data_source))
        ###clerical_review_candidates and the two matched pairs table
        db.commit()
        ###Handbreak--if clerical review candidates/matched pairs exist, change their names to the current date/time
    for table in ['clerical_review_candidates','matched_pairs']:
        if CONFIG['sql_flavor']=='sqlite':
            ret=get_table_noconn('''SELECT name FROM sqlite_master WHERE type='table' AND name='{}' '''.format(table), db)
        elif CONFIG['sql_flavor']=='postgres':
            ret=get_table_noconn('''SELECT name FROM pg_catalog.pg_tables WHERE type='table' AND name='{}' '''.format(table), db)

        if len(ret) > 0:
            cur.execute('''alter table {} rename to {}{}'''.format(table,table,dt.datetime.now().strftime('%Y_%M_%d_%H_%m')))
            db.commit()
    cur.execute('''create table clerical_review_candidates ({}_id text, {}_id text, predicted_probability float);'''.format(CONFIG['data1_name'], CONFIG['data2_name']))
    cur.execute('''create table matched_pairs ({data1}_id text, {data2}_id text, predicted_probability float, {data1}_rank float, {data2}_rank float);'''.format(data1=CONFIG['data1_name'], data2=CONFIG['data2_name']))
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
            diskEngine = create_engine('sqlite:///' + CONFIG['db_name'])
            ###IF the batch doesn't exist, make the row (it's just the batch status and batch_id)
            pd.DataFrame(batch_summary, index=[0]).to_sql('batch_summary', diskEngine, if_exists='append', index=False)
            db.commit()
        else:
            ##First check for any json and do a json.dumps
            for key in batch_summary:
                if batch_summary[key].__class__==dict:
                    batch_summary[key]=json.dumps(batch_summary[key])
                if batch_summary[key]!=batch_exists[0][key]:
                    update_statement='update batch_summary set {}=? where batch_id={}'.format(key, batch_summary['batch_id'])
                    cur.execute(update_statement, batch_summary[key])
                    db.commit()
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
            update_statement = 'update batch_summaries set {} where batch_id={}'.format(
                columns_statement, batch_summary['batch_id'])
            cur.execute(update_statement, values[0])
            db.commit()
    cur.close()
    db.close()

if __name__=='__main__':
    print('why did you do this?')