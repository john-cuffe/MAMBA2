import datetime as dt
import logging
import socket
import threading
import time
import traceback

import psycopg2
from psycopg2.extras import execute_values


class postgresqlLoggingHandler(logging.Handler):

    def format_time(self, record):
        record.dbtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))

    def connect(self):
        try:

            # if USE_AES:
            #     aesc = AESCipher(os.environ['MOJO_KEY'])
            #     password = aesc.decrypt(self.__password)
            # else:
            #     password = self.__password

            self.__connect = psycopg2.connect(
                database=self.__database,
                host = self.__host,
                user = self.__user,
                password = CONFIG['password'],
                sslmode="disable",
                options='-c search_path={}'.format(self.__search_path))
            cursor = self.__connect.cursor()
            self.__connect.commit()
            self.__connect.cursor().close()
            return True
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            return False

    def __init__(self, params):

        if not params:
            raise Exception ("No database where to log")

        self.__database = params['database']
        self.__host = params['host']
        self.__user = params['user']
        self.__password = params['password']
        self.__search_path = params['search_path']
        self.__batch_id = params['batch_id']
        self.__connect = None

        self.logging_queue = []
        self.stack_limit = LOGGING_QUEUE_SIZE

        if not self.connect():
            raise Exception ("Database connection error, no logging")

        logging.Handler.__init__(self)

        # self.__connect.cursor().execute(psqlHandler.initial_sql)
        # self.__connect.commit()
        # self.__connect.cursor().close()

        self.logging_thread = threading.Thread(target=self.log_queue)
        if "daemon" in params.keys():
            self.logging_thread.setDaemon(params["daemon"])
        self.logging_thread.start()

    def merge_two_dicts (self, x, y):
        z = x.copy()
        z.update(y)
        return z

    def log_queue(self):
        """
        This thread loop and check queue size and flush it out
        :return: 
        """
        while(self.stack_limit != -1):
            if len(self.logging_queue) >= self.stack_limit and len(self.logging_queue) != 0:
                try:
                    try:
                        cur = self.__connect.cursor()
                    except:
                        self.connect()
                        cur = self.__connect.cursor()

                    hostname_dict = {'hostname': socket.gethostname(),'batch_id':self.__batch_id}

                    insert_queue = []
                    for record in self.logging_queue[0:LOGGING_QUEUE_SIZE]:
                        self.format(record)
                        #self.format_time(record)
                        if record.exc_info:
                            record.exc_text = logging._defaultFormatter.formatException(record.exc_info)
                        else:
                            record.exc_text = ""
                        record.__dict__['timestamp']=dt.datetime.strptime(record.__dict__['asctime'], '%Y-%m-%d %H:%M:%S,%f')
                        insert_queue.append(self.merge_two_dicts(record.__dict__, hostname_dict))
                    ###a dictionary of original and new values for the records
                    columns = ['timestamp','batch_id','hostname','levelno','message','exc_text','module','funcName','lineno','filename']
                    values = [tuple(i[column] for column in columns) for i in insert_queue]
                    ####convert values to a timestamp
                    columns_list = str(tuple(['dated','batch_id','hostname', 'level', 'message', 'stack_trace', 'source_module', 'source_function_name', 'source_lineno', 'source_filename'])).replace("'","")
                    insert_statement = 'insert into {table} {collist} values %s'.format(table='hermes_central.hermes_logs',
                                                                                        collist=columns_list)
                    execute_values(cur, insert_statement, values, page_size=len(insert_queue))
                    self.__connect.commit()
                    self.__connect.cursor().close()

                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                finally:
                    # chop chop the front
                    self.logging_queue = self.logging_queue[LOGGING_QUEUE_SIZE:]
            time.sleep(3)

    def emit(self, record):
        # add record to the log queue
        self.logging_queue.append(record)


if __name__ == "__main__":

    myh = psqlHandler(CONFIG)

    l = logging.getLogger("TEST")
    l.setLevel(logging.DEBUG)
    l.addHandler(myh)


    for i in xrange(1):
        l.info("test%i"%i)