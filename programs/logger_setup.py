import logging, os, sys, time
from programs.global_vars import *
#from programs.psql_logging_handler import postgresqlLoggingHandler

def logger_setup(log_file):
    root = logging.getLogger()
    handlers = root.handlers

    for handler in handlers:
        root.removeHandler(handler)
    if len(handlers)==1:
        root.removeHandler(handlers[0])

    LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Logging handler for outputing to file
    #if write_log_to_db==False:
    logging.basicConfig(filename=log_file, format=LOGGING_FORMAT, filemode='a')
    # # Handler for outputing to db
    # if write_log_to_db:
    #     # The handler decrpyts now
    #     # if USE_AES:
    #     #     aesc = AESCipher(os.environ['MOJO_KEY'])
    #     #     password = aesc.decrypt(CONFIG['password'])
    #     # else:
    #     #     password = CONFIG['password']
    #     ####ALWAYS LOG TO HERMES_CENTRAL
    #     psqlLoggingHandler = postgresqlLoggingHandler({'host':CONFIG['host'], 'user': CONFIG['user'], 'password': CONFIG['password'], 'database':CONFIG['dbname'],'search_path':'hermes_central','batch_id':batch_id})
    #     root.addHandler(psqlLoggingHandler)
    #     #logging._addHandlerRef(psqlLoggingHandler)

    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(LOGGING_FORMAT)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    return root
'''
def close_logging_thread():
    for log_handler in logging.getLogger().handlers:
        if type(log_handler) == postgresqlLoggingHandler:
            # Check to see if we have any log left in the stack
            # if there is, we set the new queue limit to the remaining queue size and wait for the thread to picks it up
            while len(log_handler.logging_queue) != 0:
                log_handler.stack_limit = len(log_handler.logging_queue)
                time.sleep(3)
            # Signal the thread to break the while loop and join the main thread
            log_handler.stack_limit = -1
            log_handler.logging_thread.join()
'''

def logger_setup_old(log_file):
    ##log setup
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger