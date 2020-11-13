#!/usr/bin/bash
import os
os.chdir('C:\\users\\cuffe002\\desktop\\projects\\mamba2\\'), ###fill in your mamba directory here
os.environ['data1_name']='data1_test'  ###First dataset name
os.environ['data2_name']='data2_test'  ###Second dataset name
os.environ['db_name']='mamba_db' ###database name
os.environ['outputPath']='C:\\users\\cuffe002\\desktop\\projects\\mamba2\\output' ##the path for our output files
os.environ['debugmode']='False' ###run in debug mode?
os.environ['block_file_name']='block_names.csv' ###The file (including .csv ending) where the names of the blocking variables live
os.environ['create_db_chunksize']='100000'  ###the size of the chunks we want to read in
os.environ['inputPath']='C:\\users\\cuffe002\\desktop\\projects\\mamba2\\data' ##path to your data
os.environ['training_data_name']='training_data'  ###name of the training data.csv file.  don't include .csv ending
os.environ['rf_jobs']='1'  ###Number of jobs in the random forest
os.environ['clerical_review_candidates']='True'  ##do we want to generate clerical review candidates
os.environ['clerical_review_threshold']='.3'  ###What's the lowest predicted probability you want returned for a clerical review candidate?
os.environ['match_threshold']='.5'  ##what is the match threshold you want to look at
os.environ['chatty_logger']='True'  ##if on, logger logs after every block
os.environ['log_file_name']='mamba_test_log' ##the anme of your log file
os.environ['numWorkers']='3' ##number of workers in the Runner object
os.environ['prediction']='True' ##are we generating match predictions?
os.environ['scoringcriteria']='accuracy'
os.environ['stem_phrase']='False'
os.environ['ignore_duplicate_ids']='False'
os.environ['use_logit']='True'