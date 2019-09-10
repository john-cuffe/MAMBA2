from programs.global_vars import *
from programs.create_db_helpers import *
from programs.match_helpers import *
from programs.logger_setup import *
logger=logger_setup(os.environ['log_file_name'])
###actuaion function
if __name__=='__main__':
    ###create the database
    createDatabase(os.environ['db_name'])
    ####Create the Random Forest Model
    training_data=pd.read_csv('{}.csv'.format(os.environ['training_data_name']), engine='c', dtype={'{}_id'.format(os.environ['data1_name']):str,'{}_id'.format(os.environ['data2_name']):str})
    ###generate the rf_mod
    mod=generate_rf_mod(training_data)
    ###get the list of blocks
    blocks=pd.read_csv('block_names.csv').sort_values('order').to_dict('record')
    ###for block in blocks
    ##run
    for block in blocks:
        logger.info('### Starting Block {}'.format(block['block_name']))
        run_block(block)
        logger.info('### Completed Block {}'.format(block['block_name']))
    ###Once we are done, spit out the final information on the number of matches, and the .csv files
    pd.DataFrame(get_table_noconn('''select * from matched_pairs''', db)).to_csv('output/all_matches_{}.csv'.format(os.environ['date']), index=False)
    pd.DataFrame(get_table_noconn('''select * from clerical_review_candidates''', db)).to_csv('output/clerical_review_candidates_{}.csv'.format(os.environ['date']), index=False)

