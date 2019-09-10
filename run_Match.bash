#!/usr/bin/bash
cd / ###fill in your mamba directory here
export data1_name=data1_test  ###First dataset name
export data2_name=data2_test  ###Second dataset name
export db_name=mamba_db ###database name
export outputPath=C:\Users\cuffe002\Desktop\Projects\MAMBA2\output ##the path for our output files
export debugmode=False ###run in debug mode?
export block_file_name=block_names.csv ###The file (including .csv ending) where the names of the blocking variables live
export create_db_chunksize=100000  ###the size of the chunks we want to read in
export inputPath=C:\Users\cuffe002\Desktop\Projects\MAMBA2\data
export nullmatch=True
export training_data_name=training_data
export rf_jobs=1
export clerical_review_candidates=True
export clerical_review_threshold=.3
export match_threshold=.5
export chatty_logger=True
export numWorkers=1

/apps/anaconda/bin/python run_match.py
