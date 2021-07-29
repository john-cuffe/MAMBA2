MAMBA2 User Guide

# Table of Contents

[Introduction](#Introduction)

[Getting started](#Getting-started)

[What Does MAMBA do?](#WhatDoesMAMBAdo)

[MAMBA Files](#MAMBAFiles)


# Introduction

MAMBA2 represents a substantial improvement in the flexibility and scalability of the original MAMBA, enabling the use on datasets beyond the Census&#39; Business Register, using Natural Language Processing techniques to stem and clean textual data, and allow for comparisons of different match metrics in the same model. It also selects the best performing model from a list of Machine Learning algorithms.

## Getting started
 <a name="Getting-started"/>
 
System requirements:

Python 3.7

Getting MAMBA:

Download from our GitHub repo, [https://github.com/john-cuffe/MAMBA2](https://github.com/john-cuffe/MAMBA2)

### What Does MAMBA do?
 <a name="#WhatDoesMAMBAdo"/>

MAMBA is going to match two datafiles. You can call them whatever you want, and save the names of the .csv files (without the .csv ending) in the run_Match.bash file as data1_name and data2_name respectively. This .bash file will then feed these names through all of the subsequent programs.

##MAMBA Files
 <a name="#MAMBAFiles"/>


### /data/\*data1_name\*.csv

The name of the first dataset you want to match. Must contain a unique column &#39;id&#39; to server as your unique identifier for that record. Whenever you see \*data1_name\* in this guide, substitute this for the name of the dataset in your code/.bash file.

### /data/\*data2_name\*.csv

The name of the second dataset you want to match. Must contain a unique column &#39;id&#39; to server as your unique identifier for that record. Whenever you see \*data2_name\* in this guide, substitute this for the name of the dataset in your code/.bash file.

### Block_names.csv

This file is going to tell MAMBA which variables in your dataset serve as &#39;blocks&#39;. A block is a set of records (e.g. all records in a zip code, a state etc) where _all_ records within that block will be compared to each other.

#### Variables:

    - rder: which order (1 being the lowest level) you want the blocks run in. If a record on a dataset is matched in a block, it is not examined in any subsequent blocks.
    - block_name: A naming convention for the block.
    - \*data1_name\*: This column header should be your data1_name in your run_match.bash file. This is the name of the variable that corresponds to the block in the first dataset you wish to match
    - \*data2_name\*: This column header should be your data1_name in your run_match.bash file. This is the name of the variable that corresponds to the block in the second dataset you wish to match.
    - Parsed_block: See below. 1 if the block comes from a parsed variable, 0 otherwise.

#### NOTE ON PARSE_ADDRESS AND BLOCKS

    - If you are using the parse_address function and want to include outputs from the parsed addresses as block (e.g. you want to use the city and zip code) you must use the exact names from the _address_component_matching.csv_ file or the [usaddress docs](https://usaddress.readthedocs.io/en/latest/) for the block name, \*data1_name\* and \*data2_name\* column.
    - You can reduce the ZipCode variable to any number of digits by including the digit at the end of ZipCode. For example, to block by 3-digit zip code, include a block that has &#39;ZipCode3&#39; as its name.

#### Figure 1. Demonstration of block_names.csv file.

![](RackMultipart20210728-4-wfnh29_html_16987f19080a2da0.png)

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
    - Date: Date variables following the format &#39;Date_Format&#39; entered in the .bash file. This type of date variable will be compared using the Levenstein (edit) distance between the two strings. If either value is missing, returns 0. We just use the edit distance here because it provides an informative view of the number of edits and substitutions required to make the strings match. This is preferred over a strict subtraction of dates. For example, 1/1/2021 and 1/1/1011 is most likely a clerical error and requires only one substitution would match, but a simple distance calculation would give these two dates as a millennia apart.
  - Parsed_variable
    - Is the variable an output from parsing an address? If so, 1, otherwise 0

#### Figure 2. Demonstration of mamba_variable_types.csv

![plot](./Documentation/figure_2_variable_names.png)

_NOTE ON PARSED VARIABLES:_

- Ensure that variable_name, \*data1_name\* and \*data2_name\* are all filled out with the exact address_component from _address_component_matching.csv_ or the [usaddress docs](https://usaddress.readthedocs.io/en/latest/)

  - training_data.csv
    - This is the data that will tell MAMBA what you believe is a match and which is not. Currently, MAMBA requires a truth deck of matches in order to build off of. This file only contains three columns.
      - \*data1_name\*_id: the id for the record in the first dataset
      - \*data2_name\*_id: the id for the record in the second dataset
      - match: 1 if the pair is a match, 0 otherwise.

### Run_match.bash

This is the actual .bash file you need to edit to run MAMBA. Edit the following environment variables, which are picked up by Python

#### Variables:

    - Data1_name
      - The name of the first dataset. Exclude the &#39;.csv&#39; ending.
    - data2_name
      - The name of the second dataset. Exclude the &#39;.csv&#39; ending.
    - db_name
      - The name you want to give your database. Exclude the &#39;.db. ending
    - utputPath:
      - The output directory you want to use
    - Debugmode
      - True/False that skips the majority of matches if used. Set to False
    - block_file_name:
      - the name of the file that you are using to define the blocks.
    - create_db_chunksize:
      - A variable to set how big of a &#39;chunk&#39; you push to your db at once.
    - inputPath:
      - Path to the datasets you want to match.
    - Training_data_name:
      - The name of the training data you are using. Exclude the &#39;.csv&#39; ending
    - Rf_jobs:
      - Number of jobs you want to calculate the random forest on. Note, for python you want to have one job per CPU for reasons I don&#39;t entirely understand.
    - clerical_review_candidates:
      - If True, then python generates clerical review candidates for you to do further training data creation.
    - clerical_review_threshold:
      - What predicted probability do you want to limit clerical review candidates to? Generally, you want this to be closer to .5 and 0, as low probability matches won&#39;t help the model determine harder cases.
    - match_threshold:
      - what is the threshold you want to consider a &#39;match&#39;.
    - chatty_logger:
      - If True, logger returns an entry after every block matched.
    - Log_file_name:
      - Name of the .log file you want to use. Exclude &#39;.log&#39; from the name
    - numWorkers:
      - number of workers you want to run the matching on. As with rf_jobs, assume one job per CPU you are utilizing.
    - Prediction:
      - Do you want MAMBA to predict matches. If set to False, then you can generate clerical review candidates only (if clerical_review_candidates=True)
    - Scoringcriteria:
      - A scoring criteria selected from scikit-learn&#39;s list. See [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - Ignore_duplicate_ids:
      - If True, assumes that you are attempting to de-duplicate the same file, and thus does not compare records with matching ID variables. If False, then compares all records as normal.
    - Use_logit:
      - If True, run a logit and set scoringcriteria to accuracy. Logger will return a warning message about collinearity if it&#39;s bad. If False, doesn&#39;t run a logit.
    - date_format:
      - The format of the date columns in your .csv files. For example, &#39;09-01-2020&#39; would be &#39;%m-%d-%Y&#39;. See format codes here [https://docs.python.org/3/library/datetime.html](https://docs.python.org/3/library/datetime.html)
    - Parse_address:
      - See below for details
      - If parse_address is True, then the following three variables must be used.
    - Address_column_\*data1_name\*:
      - The name of the address column to parse in the \*data1_name\* csv file
    - Address_column_\*data2_name\*:
      - The name of the address column to parse in the \*data2_name\* csv file
    - Use_remaining_parsed_address:
      - Do you want to use any remaining address information beyond what is specified as a variable or a block.

## Parsing Addresses

Much pre-processing for record linkage programs focuses on how best to parse addresses into useful strings. Although MAMBA makes minimal assumptions about data structure, it does offer the ability to use unparsed addresses to create variables to feed into the record linkage model, as well as blocks to structure the record linkage. This feature uses python&#39;s _usaddress_ module to identify component blocks of an address. Imagine you have a column that contains an unparsed address (e.g 123 Main Street, Apartment 1, Anytown, AS, USA), the _usaddress_ library is able to identify the street number (123), the street name (Main), the street name type (Street), the occupancy type (Apartment) and the occupancy type number (1), as well as the city, state, and zip code. This presents a massive amount of data for MAMBA to use to match.

### Setting up:

As described above, to use this feature, ensure _parse_address_ is set to True in your .bash file, and enter in the corresponding address columns for both of your datasets.

      - In your mamba_variable_types file, enter in any variables you want and the type of match, using the _exact_ naming convention used in the _address_component_mapping.csv_ file (also available on the usaddress website) for both the variable_name and the columns with the corresponding name for your datasets (Columns B and C). Indicate any parsed variables with a 1 in the parsed_variable column.
      - In your _block_names.csv_ file, indicate any parsed blocks and their order you want to use using the _exact_ naming convention used in the _address_component_mapping.csv_ file (also available on the usaddress website) for both the block_name and the columns with the corresponding name for your datasets (Columns C and D). Indicate any parsed variables with a 1 in the parsed_block column.

_Using the use_remaining_parsed_address feature_.

  - While parsing addresses, users may not want to compare multiple strings, but rather only identify some components of a parsed address to use as blocks or strings. This function allows any remainder to be used as a fuzzy variable.
  - For example, imagine parsing the address &#39;123 Main Street SW Apartment 1, Anytown, AS, 12345&#39;. If we wanted to match using city, state, and zip as blocks and include the address number as its own variable, MAMBA would remove those features from the string but leave &quot;Main Street SW Apartment 1&quot; as a string, which itself contains valuable information. Leaving the _use_remaining_parsed_address_ feature as &#39;True&#39; tells MAMBA to create a new fuzzy variable to use in the model based on all of the address components _not otherwise used as a block or a separate variable_.