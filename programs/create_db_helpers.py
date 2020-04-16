# -*- coding: utf-8 -*-

'''
This is a list of helper functions to create our database
'''
from programs.global_vars import *
from programs.logger_setup import *
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
    var_types=pd.read_csv('mamba_variable_types.csv').to_dict('record')
    blocks = pd.read_csv('block_names.csv').to_dict('record')
    ###dictionary to read all blocks as strings
    str_dict={item[dataname]:str for item in blocks}
    blocks=[i['block_name'] for i in blocks]
    ###Get list of the fuzzy variables
    fuzzy_vars = [i[dataname] for i in var_types if i['match_type'].lower() == 'fuzzy']
    ##chunk data so we don't blow up memory
    if sys.platform == 'win32':
        csvname='{}\\{}.csv'.format(os.environ['inputPath'], dataname)
    else:
        csvname='{}/{}.csv'.format(os.environ['inputPath'], dataname)
    for data in pd.read_csv(csvname, chunksize=int(os.environ['create_db_chunksize']), engine='c',dtype=str_dict):
        ###for each chunk, loop through
        data[fuzzy_vars] = data[fuzzy_vars].replace(np.nan, 'NULL')
        data['matched'] = 0
        if 'full' in blocks:
            data['full']=1
        for p in fuzzy_vars:
            data[p] = data[p].str.upper()
        data = data.to_dict('record')

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
        if prediction==True:
            pd.DataFrame(out).to_sql(data_source, diskEngine, if_exists='replace',index=False)

        if not prediction and clericalreview:
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
        cur.execute('''drop table if exists clerical_review_candidates;''')
        cur.execute('''drop table if exists matched_pairs''')
        cur.execute('''create table clerical_review_candidates ({}_id text, {}_id text, predicted_probability float);'''.format(os.environ['data1_name'], os.environ['data2_name']))
        cur.execute('''create table matched_pairs ({data1}_id text, {data2}_id text, predicted_probability float, {data1}_rank float, {data2}_rank float);'''.format(data1=os.environ['data1_name'], data2=os.environ['data2_name']))
        db.commit()


if __name__=='__main__':
    print('why did you do this?')