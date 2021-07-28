from standardizer import standardize
import comparator
import networking
from amgScore import fidComparator
import pandas as pd

"""
read a .txt file and turn it into a dataframe
"""
def convertInputFileToFrame(file_path):
    input_file = open(file_path, 'r')
    lines = input_file.readlines()
    id = 0
    df = pd.DataFrame(columns = ['ID', 'Address'])
    for line in lines:
        line = line.rstrip('\n')
        df.loc[id] = [id, line]
        id += 1
    return df

"""
!!assumes there are csv headings!!
turns a csv into a dataframe, standardizing columns
    file_path: the filepath to the csv
    id_col: the column name of the id/index values
    address_col: the column name of the raw address data
    delimiter: the delimiter for the csv
    nrows: the number of rows to take
"""
def csv_to_frame(file_path, id_col, address_col, delimiter = ',', nrows = None):
    result = pd.read_csv(file_path, usecols = [id_col, address_col], \
        sep = delimiter, nrows = nrows)
    result = result.rename(columns = {id_col : 'ID', address_col : 'Address'})
    result = result.set_index('ID')
    return result

#TODO: find a better way of error-catching
"""
at least one of labels or fidlist must be True
labels, fidlist, and original are boolean flags
-- labels gives every parsed label category that has at least one value
-- fidlist gives the needed values for fidCompare in a list
-- original maintains the original address

if you want to block on a parsed label, labels should be True!
    input_df: a dataframe with an 'Address' column and indexed by unique ids
    labels: flag to put all outputs from standardize() into the frame
    fidlist: flag to put the condensed values from fid_prepare in its own column
    original: flag to retain the original address as a column
"""     
def standardize_df(input_df, labels = True, fidlist = True, original = False):

    # nested function that handles individual addresses
    def clean(addr, labels, fidlist):
        try:        
            stan = standardize(addr)
            if fidlist:
                flist = {"FidList" : comparator.fid_prepare(stan)}
                if labels:
                    stan.update(flist)
                    return stan
                else:
                    return flist
            else:
                return stan
        except:
            # maybe add an ERROR column? with specific error codes?
            return {'ERROR' : addr}
    
    result_df = pd.DataFrame(list(input_df['Address'].apply( \
        clean, labels = labels, fidlist = fidlist)), index = input_df.index)
    if original:
        result_df['Address'] = input_df['Address']
    
    return result_df

# TODO: do some real error handling, ya loser
"""
a helper function for df.apply() in deduplicate() and match()
    col_address, constant: two address records stripped by fid_prepare
    -- list of six values each
returns fidCompare's score, or fails with -1
"""
def column_matches(col_address, constant):
    try:
        fid_list = comparator.fid_pair(col_address, constant)
        return fidComparator(*fid_list)
    except:
        return -1

"""
matches a dataframe of records with itself and generates a frame of match scores
    data: the input dataframe
    score_threshold: two records match if their score >= the threshold
    block: a single label (i.e. ZIP) to block on
"""
def deduplicate(data, score_threshold = 800, block = None):

    result = pd.DataFrame(columns = ['Address1', 'Address2', 'FidList'])
    # breaks down df into blocked df slices
    if block:
        blocked = data.groupby(block)
        blocked_chunks = [blocked.get_group(blk)['FidList'] for blk in blocked.groups.keys()]
    else:
        blocked_chunks = [data['FidList']]

    # iterate through blocks
    for b in blocked_chunks:    
        for idx, x in zip(b.index, b):
            matches = b.apply(column_matches, constant = x)
            matches = pd.DataFrame(matches)
            matches['Address1'] = matches.index
            matches['Address2'] = idx
            cleaned = matches.loc[matches['FidList'] >= score_threshold]
            result = result.append(cleaned, ignore_index = True)
            b = b.drop(idx)

    # apply this here or in each iteration? memory vs. speed
    # better to clean in iterations if lots of matches

    return result

"""
compares two dataframes of records and generates a frame of match scores
    dataA, dataB: two input dataframes
    score_threshold: two records match if their score >= the threshold
    block: a single label (i.e. ZIP) to block on
"""
def match(dataA, dataB, score_threshold = 800, block = None):

    result = pd.DataFrame(columns = ['Address1', 'Address2', 'FidList'])
    # make blocked pairs of grouped df slices
    if block:
        blockedA = dataA.groupby(block)
        blockedB = dataB.groupby(block)
        blocked_pairs = [(blockedA.get_group(blk)['FidList'], \
            blockedB.get_group(blk)['FidList']) for blk in \
                blockedA.groups.keys() if blk in blockedB.groups.keys()]
    else:
        blocked_pairs = [(dataA['FidList'], dataB['FidList'])]

    # iterate through paired blocks
    for a, b in blocked_pairs:    
        for idx, x in zip(a.index, a):
            matches = b.apply(column_matches, constant = x)
            matches = pd.DataFrame(matches)
            matches['Address1'] = matches.index
            matches['Address2'] = idx
            cleaned = matches.loc[matches['FidList'] >= score_threshold]            
            result = result.append(cleaned, ignore_index = True)

    # apply this here or in each iteration? memory vs. speed
    # better to clean in iterations if lots of matches

    return result


### still doesn't actually support using multiple blocks yet
"""
preliminary consolidation function;
TODO: option checking for bad inputs; impossible combinations
TODO: actually implementing multiple blocks
TODO: make this less ugly
"""
def records_to_matches(file1, file2 = None, blocks = None, output = 'matches', score_threshold = 800):
    frames = [file1, file2] if file2 is not None else [file1]
    stand = []
    for frame in frames:
        if blocks:
            stand.append(standardize_df(file1, labels = True))
        else:
            stand.append(standardize_df(file1, labels = False))

    if file2 is not None:
        matches = match(stand[0], stand[1], block = blocks, score_threshold = 800)
    else:
        matches = deduplicate(stand[0], block = blocks, score_threshold = 800)

    if output == 'matches':
        return matches
    elif output == 'graph':
        return networking.match_network(matches)
    elif output == 'clusters':
        return networking.disentangle(networking.match_network(matches))
