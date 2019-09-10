#!/usr/bin/bash
cd / ###fill in your mamba directory here
export data1_name=data1_test  ###First dataset name
export data2_name=data2_test  ###Second dataset name
export db_name=mamba_db ###database name
export outputPath=C:\Users\cuffe002\Desktop\Projects\MAMBA2\output ##the path for our output files
export debugmode=False ###run in debug mode?
export block_file_name=block_names.csv ###The file (including .csv ending) where the names of the blocking variables live
export create_db_chunksize=100000  ###the size of the chunks we want to read in
export inputPath=C:\Users\cuffe002\Desktop\Projects\MAMBA2\data ##path to your data
export null_match_method=zero  ###placeholder. right now fills in fuzzy matched with 0
export training_data_name=training_data  ###name of the training data.csv file.  don't include .csv ending
export rf_jobs=1  ###Number of jobs in the random forest
export clerical_review_candidates=True  ##do we want to generate clerical review candidates
export clerical_review_threshold=.3  ###What's the lowest predicted probability you want returned for a clerical review candidate?
export match_threshold=.5  ##what is the match threshold you want to look at
export chatty_logger=True  ##if on, logger logs after every blog
export numWorkers=1 ##number of workers in the Runner object

/apps/anaconda/bin/python run_match.py
