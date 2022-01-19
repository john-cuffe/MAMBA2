from programs.global_vars import *
from programs.create_db_helpers import *
from programs.match_helpers import *
from programs.logger_setup import *
from programs.general_helpers import *
logger=logger_setup('{}/{}'.format(CONFIG['projectPath'],CONFIG['log_file_name']))
import datetime as dt
import traceback
###actuaion function
if __name__=='__main__':
    ###start up by printing out the full config file
    for key in CONFIG:
        logger.info('#############')
        logger.info('''{}: {}'''.format(key, CONFIG[key]))
    batch_summary={}
    ###set the start time
    batch_summary['batch_started'] = dt.datetime.now()
    batch_summary['batch_status'] = 'in_progress'
    batch_summary['batch_config'] = {}
    for key in CONFIG:
        if 'password' not in key:
            batch_summary['batch_config'][key]=CONFIG[key]
    batch_summary['batch_config']=json.dumps(batch_summary['batch_config']).encode('utf-8')    ###check if the database exists already
    if os.path.isfile(CONFIG['db_name'])==False:
        db=get_db_connection(CONFIG)
        cur = db.cursor()
        ##sqlite doesn't like multi-line statements
        if CONFIG['sql_flavor']=='sqlite':
            for stm in batch_info_qry.split(';'):
                cur.execute(stm)
            db.commit()
        else:
            cur.execute(batch_info_qry)
            db.commit()
    db = get_db_connection(CONFIG)
    batch_summary['batch_id'] = CONFIG['batch_id'] = generate_batch_id(db)
    logger.info('#############')
    logger.info('''Batch ID: {}'''.format(CONFIG['batch_id']))
    ###update the batch summary
    update_batch_summary(batch_summary)
###create the database
    try:
        createDatabase(CONFIG['db_name'])
    except Exception as error:
        logger.info('Error in creating database. Error: {}'.format(''.join(traceback.format_tb(error.__traceback__))))
        batch_summary['batch_status'] = 'failed'
        batch_summary['failure_message'] = 'Failed to create database tables.  See logs for information'
        update_batch_summary(batch_summary)
        os._exit(0)
    ####Create the Random Forest Model
    try:
        training_data=pd.read_csv('{}/training_data_key.csv'.format(CONFIG['projectPath']), engine='c', dtype={'{}_id'.format(CONFIG['data1_name']):str,'{}_id'.format(CONFIG['data2_name']):str})
        if '.joblib' in CONFIG['saved_model']:
            mod = load_model(CONFIG)
        ###generate the rf_mod
        else:
            mod=choose_model(training_data)
            ###Dump it
            dump_model(mod, CONFIG['saved_model_target'], CONFIG['imputation_method'])
    except Exception as error:
        logger.info('Error in selecting . Error: {}'.format(''.join(traceback.format_tb(error.__traceback__))))
        batch_summary['batch_status'] = 'failed'
        batch_summary['failure_message'] = 'Failed to select a model.  See logs for information'
        update_batch_summary(batch_summary)
        os._exit(0)
    ###get the list of blocks
    try:
        ##run
        for block in blocks:
            logger.info('### Starting Block {}'.format(block['block_name']))
            run_block(block, mod)
            logger.info('### Completed Block {}'.format(block['block_name']))
    except Exception as error:
        logger.info('Error in running block {} . Error: {}'.format(block['block_name'],''.join(traceback.format_tb(error.__traceback__))))
        batch_summary['batch_status'] = 'failed'
        batch_summary['failure_message'] = 'Failed to run block {}.  See logs for information'.format(block['block_name'])
        update_batch_summary(batch_summary)
        os._exit(0)
    ###Once we are done, spit out the final information on the number of matches, and the .csv files
    db=get_db_connection(CONFIG)
    pd.DataFrame(get_table_noconn('''select * from matched_pairs''', db)).to_csv('output/all_matches_{}.csv'.format(CONFIG['date']), index=False)
    pd.DataFrame(get_table_noconn('''select * from clerical_review_candidates''', db)).to_csv('output/clerical_review_candidates_{}.csv'.format(CONFIG['date']), index=False)
    logger.info('Match Complete')
    if ast.literal_eval(CONFIG['prediction'])==True:
        summstats=get_table_noconn('''select count(distinct {}_id) data1_matched, count(distinct {}_id) data2_matched, count(*) total_pairs from matched_pairs'''.format(CONFIG['data1_name'], CONFIG['data2_name']), db)[0]
        logger.info('Matched {} records for {}'.format(summstats['data1_matched'], CONFIG['data1_name']))
        logger.info('Matched {} records for {}'.format(summstats['data2_matched'], CONFIG['data2_name']))
        logger.info('{} Total matched pairs'.format(summstats['total_pairs']))
        ####get the block summary data
        sumstats = get_table_noconn('''select block_level,
         count(block_matches) total_blocks,
         sum(block_size) total_pairs, 
         sum(block_time) total_time,
        sum(block_matches) total_matches, 
        sum(block_non_matches) total_non_matches,
        (sum(block_matches_avg_score * block_matches)/sum(block_matches)) average_match_score,
         (sum(block_non_matches_avg_score * block_non_matches)/sum(block_non_matches)) average_non_match_score
           from batch_statistics where batch_id = {} group by block_level'''.format(CONFIG['batch_id']),db)
        logger.info(sumstats)
    batch_summary['batch_status']='complete'
    batch_summary['batch_completed']=dt.datetime.now()
    update_batch_summary(batch_summary)
    db.close()
    os._exit(0)

