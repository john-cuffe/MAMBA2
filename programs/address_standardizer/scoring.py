
import jellyfish



#### TODO: How should we weigh categorical differences (type, separator, etc.) from percentage scores here?
"""
Scores two strings based off of edit distance, a percentage score
- uses Damerau-Levanshtein
input:
output:
"""
def DL_percentage(a, b):
    max_len = max(len(a), len(b))
    diff = jellyfish.damerau_levanshtein_distance(a, b)
    # weight extra parts & subdivisions by some amount
    return (max_len - diff)/max_len    


def HN_compare(HN_A, HN_B, HNPRE_A, HNPRE_B, HNSUF_A, HNSUF_B):
    edit_score = DL_percentage(HN_A, HN_B)
    pre_match = HNPRE_A == HNPRE_B
    suf_match = HNSUF_A == HNSUF_B
    sep_match = HNSEP_A == HNSEP_B
    # weight extra parts & subdivisions by some amount
    return False


# HN types should probably just be yes/no
# HN1 matching but HN2 not is closer(?) than the other wa   y around?

"""
same concept as above;
"""
def WSID1_compare(WSDESC1_A, WSDESC1_B, WSID1_A, WSID1_B):
    edit_score = DL_percentage(WSID1_A, WSID1_B)
    desc_match = WSDESC1_A == WSDESC1_B
    # How do I weight this?
    # if WSDESC1_A 
    return edit_score

