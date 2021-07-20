# -*- coding: utf-8 -*-

'''
This is a list of helper functions to create our database
'''
from programs.global_vars import *
from programs.logger_setup import *
import usaddress as usadd
logger=logger_setup(os.environ['log_file_name'])
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
    str_dict={item[dataname]:str for item in blocks}
    ###Get list of the fuzzy variables
    fuzzy_vars = [i[dataname] for i in var_types if i['match_type'].lower() == 'fuzzy']
    ###add address1 if we are using the remaining parsed addresses
    if ast.literal_eval(os.environ['use_remaining_parsed_address']) == True:
        ###append the address 1 to the fuzzy vars
        fuzzy_vars.append('address1')
    ##chunk data so we don't blow up memory
    if sys.platform == 'win32':
        csvname='{}\\{}.csv'.format(os.environ['inputPath'], dataname)
    else:
        csvname='{}/{}.csv'.format(os.environ['inputPath'], dataname)
    for data in pd.read_csv(csvname, chunksize=int(os.environ['create_db_chunksize']), engine='c',dtype=str_dict):
        ###If we have a date variable, find it and convert to a date
        date_columns=[i[dataname] for i in var_types if i['match_type']=='date']
        if len(date_columns) > 0:
            for date_col in date_columns:
                data[date_col]=pd.to_datetime(data[date_col], format=os.environ['date_format']).dt.strftime('%Y-%m-%d')
        data['matched'] = 0
        if 'full' in [b['block_name'] for b in blocks]:
            data['full']=1
        ###convert to dictionary
        data = data.to_dict('record')
        ########
        ##Are we using the parsed address feature? if so, add those variables into the row
        ########
        if ast.literal_eval(os.environ['parse_address'])==True:
            ###For each row, parse the address
            for row in data:
                parsed_address = usadd.tag(row[os.environ['address_column_{}'.format(dataname)]], tag_mapping=address_component_mapping)
                for parsed_block in [v for v in blocks if v['parsed_block']==1]:
                    if len(re.findall('ZipCode[0-9]', parsed_block['block_name'])) > 0:
                        row[parsed_block['block_name']] = parsed_address[0]['ZipCode'][0:int(parsed_block['block_name'][-1])]
                    else:
                        row[parsed_block['block_name']] = parsed_address[0][parsed_block['block_name']]
                for variable in [var for var in var_types if var['parsed_variable']==1]:
                    row[variable['variable_name']] = parsed_address[0][variable['variable_name']]
                if ast.literal_eval(os.environ['use_remaining_parsed_address'])==True:
                    row['address1'] = parsed_address[0]['address1']
        for p in fuzzy_vars:
            for r in range(len(data)):
                ###fill in NULL for the fuzzy vars
                if pd.isna(data[r][p]):
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
        # pool=Pool(2)
        if ast.literal_eval(os.environ['stem_phrase'])==True:
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

    # Initializes database
    diskEngine = create_engine('sqlite:///'+databaseName)
    ###for each input dataset, need to
    for data_source in [os.environ['data1_name'],os.environ['data2_name']]:
        out=get_stem_data(data_source)
        # 3) Push to DB
        if ast.literal_eval(os.environ['prediction'])==True:
            pd.DataFrame(out).to_sql(data_source, diskEngine, if_exists='replace',index=False)

        if ast.literal_eval(os.environ['prediction'])==False and ast.literal_eval(os.environ['clerical_review_candidates'])==True:
            pd.DataFrame(out).sample(frac=.05).to_sql(data_source, diskEngine,if_exists='replace', index=False)
        ####now index the tables
        db=get_connection_sqlite(databaseName)
        cur=db.cursor()
        blocks=pd.read_csv('block_names.csv').to_dict('record')
        for i in blocks:
            cur.execute('''create index {source}_{variable}_idx on {source} ({variable});'''.format(variable=i[data_source], source=data_source))
        ###additional index on id
        cur.execute('''create index {}_id_idx on {} (id)'''.format(data_source, data_source))
        ###clerical_review_candidates and the two matched pairs table
        ###Handbreak--if clerical review candidates/matched pairs exist, change their names to the current date/time
    for table in ['clerical_review_candidates','matched_pairs']:
        ret=get_table_noconn('''SELECT name FROM sqlite_master WHERE type='table' AND name='{}' '''.format(table), db)
        if len(ret) > 0:
            cur.execute('''alter table {} rename to {}{}'''.format(table,table,dt.datetime.now().strftime('%Y_%M_%d_%H_%m')))
            db.commit()
    cur.execute('''create table clerical_review_candidates ({}_id text, {}_id text, predicted_probability float);'''.format(os.environ['data1_name'], os.environ['data2_name']))
    cur.execute('''create table matched_pairs ({data1}_id text, {data2}_id text, predicted_probability float, {data1}_rank float, {data2}_rank float);'''.format(data1=os.environ['data1_name'], data2=os.environ['data2_name']))
    db.commit()


if __name__=='__main__':
    print('why did you do this?')