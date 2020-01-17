# MAMBA2
Second Generation of the MAMBA Software

# Notes

## Inputs: 

* Data to me matched--Two data files to be matched, in comma delimited format (?), with names set as parameters in run_Match.bash. 
* Block dictionary--Dictionary of blocking variables in block_names.csv, with dataset names that match whats stored in ./data/ and the parameters in run_Match.bash.
* How-to-match dictionary--Dictionary of how each variable should be treated (fuzzy match, exact match, etc.) in mamba_variable_types.csv. 
* Training data--Truth deck of id-id-match category (0/1) for the data to be matched in training_data.csv. We are agnostic about how this training data is generated. However, there are some best practices and clerical review interfaces to help. 

## Outputs:

* ?


