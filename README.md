MAMBA2 User Guide

# Table of Contents

[Introduction](#Introduction)

[Getting started](#Getting-started)

[What Does MAMBA do?](#WhatDoesMAMBAdo)

[MAMBA Files](#MAMBAFiles)

[Setting Up](#settingup)

[Parsing Addresses](#parsingaddresses)

[Custom Matching Models](#custom_models)

[Recursive Feature Elimination](#recursive_feature_elimination)

[Manual Filter Selection](#manual_filter_selection)
# Introduction

MAMBA2 represents a substantial improvement in the flexibility and scalability of the original MAMBA, enabling the use on datasets beyond the Census' Business Register, using Natural Language Processing techniques to stem and clean textual data, and allow for comparisons of different match metrics in the same model. It also selects the best performing model from a list of Machine Learning algorithms.

## Getting started
 <a id="Getting-started"/>
 
System requirements:

Python 3.7 >

Getting MAMBA:

Download from our GitHub repo, [https://github.com/john-cuffe/MAMBA2](https://github.com/john-cuffe/MAMBA2)

### What Does MAMBA do?
 <a id="#WhatDoesMAMBAdo"/>

MAMBA is going to match two datafiles. You can call them whatever you want, and save the names of the .csv files (without the .csv ending) in the run_Match.bash file as data1_name and data2_name respectively. This .bash file will then feed these names through all of the subsequent programs.

##MAMBA Files
 <a id="#MAMBAFiles"/>

### /data/\*data1_name\*.csv

The name of the first dataset you want to match. Must contain a unique column 'id' to server as your unique identifier for that record. Whenever you see \*data1_name\* in this guide, substitute this for the name of the dataset in your code/.bash file.

### /data/\*data2_name\*.csv

The name of the second dataset you want to match. Must contain a unique column 'id' to server as your unique identifier for that record. Whenever you see \*data2_name\* in this guide, substitute this for the name of the dataset in your code/.bash file.

### Block_names.csv

This file is going to tell MAMBA which variables in your dataset serve as 'blocks'. A block is a set of records (e.g. all records in a zip code, a state etc) where _all_ records within that block will be compared to each other.

#### Variables:

    - order: which order (1 being the lowest level) you want the blocks run in. If a record on a dataset is matched in a block, it is not examined in any subsequent blocks.
    - block_name: A naming convention for the block.
    - \*data1_name\*: This column header should be your data1_name in your run_match.bash file. This is the name of the variable that corresponds to the block in the first dataset you wish to match
    - \*data2_name\*: This column header should be your data1_name in your run_match.bash file. This is the name of the variable that corresponds to the block in the second dataset you wish to match.
    - Parsed_block: See below. 1 if the block comes from a parsed variable, 0 otherwise.

#### NOTE ON PARSE_ADDRESS AND BLOCKS

    - If you are using the parse_address function and want to include outputs from the parsed addresses as block (e.g. you want to use the city and zip code) you must use the exact names from the _address_component_matching.csv_ file or the [usaddress docs](https://usaddress.readthedocs.io/en/latest/) for the block name, \*data1_name\* and \*data2_name\* column.
    - You can reduce the ZipCode variable to any number of digits by including the digit at the end of ZipCode. For example, to block by 3-digit zip code, include a block that has 'ZipCode3' as its name.

#### Figure 1. Demonstration of block_names.csv file.

![plot](./Documentation/figure_1_block_names.png)

### mamba_variable_types.csv

This file tells MAMBA what kind of analysis to do on different kinds of variables you are using for matching. All of these models will be used to generate the matching models.

#### Variables:

  - Variable_name: generic name for the comparison.
  - \*data1_name\*: This column header should be your data1_name in your run_match.bash file. This is the name of the variable that corresponds to the variable in the first dataset you wish to match
  - \*data2_name\*: This column header should be your data1_name in your run_match.bash file. This is the name of the variable that corresponds to the variable in the second dataset you wish to match.
  - match_type: What kind of match analysis do you want to perform?
    - fuzzy: generate 12 fuzzy string comparators for the pair
    - num_distance: difference between the two values.
    - exact: If the two values match, scored as a 1, otherwise scored as a 0.
    - geo_distance: Distance between two points (in kilometers) based on Haversine formula. Using this type requires that each dataset has a latitude and longitude variable, and this column is completely filled for all observations.
    - Date: Date variables following the format 'Date_Format' entered in the .bash file. This type of date variable will be compared using the Levenstein (edit) distance between the two strings.
    - If either value is missing, returns 0. 
    - We just use the edit distance here because it provides an informative view of the number of edits and substitutions required to make the strings match. This is preferred over a strict subtraction of dates. For example, 1/1/2021 and 1/1/1011 is most likely a clerical error and requires only one substitution would match, but a simple distance calculation would give these two dates as a millennia apart.
  - Parsed_variable
    - Is the variable an output from parsing an address? If so, 1, otherwise 0
  - custom_variable_name
    - With this column, you can enter the name of a custom scoring metric from _programs.custom_scoring_methods_.  See below for details. \
    - If not using a custom scoring metric for the variable in question, leave blank.

#### Figure 2. Demonstration of mamba_variable_types.csv

![plot](./Documentation/figure_2_variable_names.png)

#### NOTE ON PARSED VARIABLES:

- Ensure that variable_name, \*data1_name\* and \*data2_name\* are all filled out with the exact address_component from _address_component_matching.csv_ or the [usaddress docs](https://usaddress.readthedocs.io/en/latest/)

#### NOTE ON CUSTOM SCORING METHODS:

- If you have a custom scoring method you wish to use (e.g. counting characters, a different fuzzy comparator) include it as a function in _programs.custom_scoring_methods.py_.
- The function must accept as an argument two strings, _and also must handle missing values_ (e.g. by giving a particular value to missings).
- {data1_name} and {data2_name} should appear as they do in their respective datasets.

### training_data.csv
    - This is the data that will tell MAMBA what you believe is a match and which is not. Currently, MAMBA requires a truth deck of matches in order to build off of. This file only contains three columns.
      - \*data1_name\*_id: the id for the record in the first dataset
      - \*data2_name\*_id: the id for the record in the second dataset
      - match: 1 if the pair is a match, 0 otherwise.

### MAMBA.properties
 <a id="mamba_properties"/>
This is the file you will edit to run MAMBA.  After setting your configurations, just run run_match.py.

#### Variables:

    - Data1_name
      - The name of the first dataset. Exclude the '.csv' ending.
    - data2_name
      - The name of the second dataset. Exclude the '.csv' ending.
    - db_flavor
      - The 'flavor' of sql database.  Current options: sqlite, postgres.
        - NOTE: if using postgres, ensure you have the correct db_host, db_port, db_user, and db_password in the mamba.properties file
    - db_name
      - The name you want to give your database. Exclude the '.db' ending
    - outputPath:
      - The output directory you want to use
    - Debugmode
      - True/False that skips the majority of matches if used. Set to False
    - block_file_name:
      - the name of the file that you are using to define the blocks.
    - create_db_chunksize:
      - A variable to set how big of a 'chunk' you push to your db at once.
    - inputPath:
      - Path to the datasets you want to match.
    - Training_data_name:
      - The name of the training data you are using. Exclude the '.csv' ending
    - Rf_jobs:
      - Number of jobs you want to calculate the random forest on. Note, for python you want to have one job per CPU for reasons I don't entirely understand.
    - clerical_review_candidates:
      - If True, then python generates clerical review candidates for you to do further training data creation.
    - clerical_review_threshold:
      - What predicted probability do you want to limit clerical review candidates to? Generally, you want this to be closer to .5 and 0, as low probability matches won't help the model determine harder cases.
    - match_threshold:
      - what is the threshold you want to consider a 'match'.
    - chatty_logger:
      - If True, logger returns an entry after every block matched.
    - Log_file_name:
      - Name of the .log file you want to use. Exclude '.log' from the name
    - numWorkers:
      - number of workers you want to run the matching on. As with rf_jobs, assume one job per CPU you are utilizing.
    - Prediction:
      - Do you want MAMBA to predict matches. If set to False, then you can generate clerical review candidates only (if clerical_review_candidates=True)
    - Scoringcriteria:
      - A scoring criteria selected from scikit-learn's list. See [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - Ignore_duplicate_ids:
      - If True, assumes that you are attempting to de-duplicate the same file, and thus does not compare records with matching ID variables. If False, then compares all records as normal.
    - Use_logit:
      - If True, run a logit and set scoringcriteria to accuracy. Logger will return a warning message about collinearity if it's bad. If False, doesn't run a logit.
    - date_format:
      - The format of the date columns in your .csv files. For example, '09-01-2020' would be '%m-%d-%Y'. See format codes here [https://docs.python.org/3/library/datetime.html](https://docs.python.org/3/library/datetime.html)
    - Parse_address:
      - See below for details
      - If parse_address is True, then the following three variables must be used.
    - Address_column_\*data1_name\*:
      - The name of the address column to parse in the \*data1_name\* csv file
    - Address_column_\*data2_name\*:
      - The name of the address column to parse in the \*data2_name\* csv file
    - Use_remaining_parsed_address:
      - Do you want to use any remaining address information beyond what is specified as a variable or a block.
    - Use_custom_model:
      - Do you have a custom matching model you want to run? See Custom Models for details
    - Imputation_method:
      - Either 'Nominal' or 'Imputer'.  See Imputation Methods below for details
    - saved_model:
      - If you have a model that you have saved and wish to use, enter its name here (include the .joblib ending).
    - saved_model_target:
      - If you want to _save_ your model, enter a file name here, ending with '.joblib'
    - feature_elimination_mode:
      - If this is set to True, the models generated will be fitted with recursive feature elimination.  
    - use_variable_filter:
      - If this is set to True, MAMBA will use a filter to pre-sort any match candidates.
    - variable_filter_info:
      - json dictionary required details if use_variable_filter is set to True.  See below for details.


## Setting up:
 <a id="#settingup"/>
 
 - Edit mamba.properties so each variable is configured correctly
 - Configure variable_names.csv and block_names.csv are configured correctly
 - In a terminal, CD into the /programs directory
 - enter _python setup.py build_ext --inplace_
   - This will cythonize the fuzzy methods. You can skip this step if you want, but MAMBA will be much slower. 
 - CD into the main MAMBA directory
 - Enter run_match.py
 _ Watch MAMBA go!

## Parsing Addresses
 <a id="#parsingaddresses"/>

Much pre-processing for record linkage programs focuses on how best to parse addresses into useful strings. Although MAMBA makes minimal assumptions about data structure, it does offer the ability to use unparsed addresses to create variables to feed into the record linkage model, as well as blocks to structure the record linkage. This feature uses python's _usaddress_ module to identify component blocks of an address. Imagine you have a column that contains an unparsed address (e.g 123 Main Street, Apartment 1, Anytown, AS, USA), the _usaddress_ library is able to identify the street number (123), the street name (Main), the street name type (Street), the occupancy type (Apartment) and the occupancy type number (1), as well as the city, state, and zip code. This presents a massive amount of data for MAMBA to use to match.
As described above, to use this feature, ensure _parse_address_ is set to True in your .properties file, and enter in the corresponding address columns for both of your datasets.
- In your mamba_variable_types file, enter in any variables you want and the type of match, using the _exact_ naming convention used in the _address_component_mapping.csv_ file (also available on the usaddress website) for both the variable_name and the columns with the corresponding name for your datasets (Columns B and C). Indicate any parsed variables with a 1 in the parsed_variable column.
- In your _block_names.csv_ file, indicate any parsed blocks and their order you want to use using the _exact_ naming convention used in the _address_component_mapping.csv_ file (also available on the usaddress website) for both the block_name and the columns with the corresponding name for your datasets (Columns C and D). Indicate any parsed variables with a 1 in the parsed_block column.

### Using the use_remaining_parsed_address feature

  - While parsing addresses, users may not want to compare multiple strings, but rather only identify some components of a parsed address to use as blocks or strings. This function allows any remainder to be used as a fuzzy variable.
  - For example, imagine parsing the address '123 Main Street SW Apartment 1, Anytown, AS, 12345'. If we wanted to match using city, state, and zip as blocks and include the address number as its own variable, MAMBA would remove those features from the string but leave &quot;Main Street SW Apartment 1&quot; as a string, which itself contains valuable information. Leaving the _use_remaining_parsed_address_ feature as 'True' tells MAMBA to create a new fuzzy variable to use in the model based on all of the address components _not otherwise used as a block or a separate variable_.
 
## Custom Matching Models
 <a id="custom_models"/>
 
  - Users may wish to user their own custom matching model in addition to those available in MAMBA.  
  - If so, user must enter a fully self-contained function in _custom_matching_function.py_ to run, as well as set the custom_matching function in _mamba.properties_ to True.
  - Currently, this model will be compared to the standard suite of matching models available to MAMBA, but future iterations will allow users to replace the MAMBA models completely.

## Imputation Methods
 <a id="imputation_methods"/>
  - Imputation of missing data is a major element of record linkage problems.  MAMBA takes a more hands-off approach to this problem, but offers users two options to fill in misisng values, set in the imputation_method variable of _mamba.properties_.
    - Imputer:
      - With this option, any missing values for 'fuzzy' or 'numeric distance' variables are replaced iterative imputer.  See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) for futher details.
        - While this option is easy to implement, it may result in non-sensical outcomes or potentially be subject to existing missing biases in the data.
    - Nominal:
      - This option follows a more traditional approach of converting the continuous fuzzy and numeric variables to nominal variables, and then assigning cases with missing data a particular value.
        - In the case of 'fuzzy' variables, the nominal variable created is divided evenly between every .1 value (so 0, .1, .2...1.0).  In a case where _either_ value is missing, the score is -1.
        - In the case of numeric variables, any case with missing data is filled with ten times the maximum possible value for the comparison across the entire dataset.  I should probabaly research if this was a good idea.
    
  - Additional Imputation Notes:
    - Distance variables are imputed with a value, in kilometers, of 4 times the diameter of the earth.  Future iterations will impute based on any possible geographic information, e.g. taking mean difference between all cases in a certain zip code.
    - Exact matching variables are coded as 0 (non-match), 1 (match), and -1 (either case is missing).
   
 
 ## Recursive Feature Elimination
  <a id="recursive_feature_elimination"/>
 One of the biggest drawbacks to MAMBA's use of multiple string comparators is, unsurprisingly, that MAMBA is then forced to generate scores for each of the 13 string comparators for any fuzzy variables the user wishes.  While the overall fit of the models improves with more string comparators (Cuffe and Goldschlag, 2016), no amount of clever computing can overcome the need to do this many calculations.  
 To give the user an opportunity to avoid this issue, this mode selects the lowest number of features to maximize the score for the model, while still using a randomized, paramterized grid search.  This mode can save substantial time:  For example, for a single 'fuzzy' variable, it can take MAMBA approximately 4.17 seconds to create 1000 scores.  However, the individual string comparators themselves are created in a mean time of .317 seconds, with a median of approximately .1 seconds.  In large runs, the extra time required to generate models that filter out features that don't contribute to the overall performance of the model will save substantial time. 

## Manual Filter Selection
<a id="manual_filter_selection"/>

- In general, the MAMBA philosophy for matching is to let the model do all of the work, which should result in less human-generated bias in any matches.  However, MAMBA does offer a feature where the user can introduce a filter for matches, where no candidate pair with a score below a certain threshold can possibly be a match.  If this feature is used, MAMBA takes the following steps:
  1) Generate a score, for each match candidate pair, on the variable chosen
  2) Retain only those match candidate pairs that have a score exceeding the chosen threshold
  3) Generate full scores/model predictions for remaining pairs
- The intent behind this metric is to allow users some level of control. For example, SMEs may determining that a entity names that score less than .75 on a Jaro comparator will _never_ result in a match.  This feature then allows MAMBA to skip those matches.  The additional benefit is that by only calculating one single score, and then deleting a subset, this makes MAMBA run substantially faster.
- To implement, turn the use_variable_filter parameter to True in your _mamba.properties_ file.
- Then edit the variable_filter_info json
- Keys:
    - variable_name: name of the variable as it appears in mamba_variable_types.csv
    - fuzzy_name: IF the variable is a fuzzy variable, which particular character do you want to use 
      - Options:'jaro', 'winkler', 'bagdist', 'seqmatch', 'qgram2', 'qgram3', 'posqgram3', 'editdist', 'lcs2', 'lcs3', 'charhistogram', 'swdist', 'sortwinkler'
    - test: the test you want to apply.
      - Options: =, !=, >, >=, <, <=
    - filter_value: the value you want to compare.demo for fuzzy {'variable_name': name, 'fuzzy_name': 'jaro', 'test':'>', 'filter_value':.75}
- Notes:
  - For exact matches, user still must select either 1 or 0 for the filter value.  You can select something other than '=' as the test but isn't this complicated enough already?
  - For date variables, user must give the gap between two days in days.
  - The logs and stats for the run will show how many observations were cut by the filter.