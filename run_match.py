from programs.global_vars import *
from programs.create_db_helpers import *
from programs.match_helpers import *
from programs.logger_setup import *
logger=logger_setup(os.environ['log_file_name'])
import datetime as dt
os.environ['date']=dt.datetime.now().date().strftime("%Y_%m_%d")
###actuaion function
if __name__=='__main__':
    ###create the database
    createDatabase(os.environ['db_name'])
    ####Create the Random Forest Model
    training_data=pd.read_csv('{}.csv'.format(os.environ['training_data_name']), engine='c', dtype={'{}_id'.format(os.environ['data1_name']):str,'{}_id'.format(os.environ['data2_name']):str})
    ###generate the rf_mod
    mod=choose_model(training_data)
    ###get the list of blocks
    blocks=pd.read_csv('block_names.csv').sort_values('order').to_dict('record')
    ###for block in blocks
    ##run
    for block in blocks:
        logger.info('### Starting Block {}'.format(block['block_name']))
        run_block(block, mod)
        logger.info('### Completed Block {}'.format(block['block_name']))
    ###Once we are done, spit out the final information on the number of matches, and the .csv files
    db=get_connection_sqlite(os.environ['db_name'])
    pd.DataFrame(get_table_noconn('''select * from matched_pairs''', db)).to_csv('output/all_matches_{}.csv'.format(os.environ['date']), index=False)
    pd.DataFrame(get_table_noconn('''select * from clerical_review_candidates''', db)).to_csv('output/clerical_review_candidates_{}.csv'.format(os.environ['date']), index=False)
    logger.info('Match Complete')
    if ast.literal_eval(os.environ['prediction'])==True:
        summstats=get_table_noconn('''select count(distinct {}_id) data1_matched, count(distinct {}_id) data2_matched, count(*) total_pairs from matched_pairs'''.format(os.environ['data1_name'], os.environ['data2_name']), db)[0]
        logger.info('Matched {} records for {}'.format(summstats['data1_matched'], os.environ['data1_name']))
        logger.info('Matched {} records for {}'.format(summstats['data2_matched'], os.environ['data2_name']))
        logger.info('{} Total matched pairs'.format(summstats['total_pairs']))
    db.close()
    os._exit(0)
