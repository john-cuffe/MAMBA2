import usaddress
import string
from programs.address_standardizer.constants import (
    DIRECTIONAL_ABBREVIATIONS,
    STATE_ABBREVIATIONS,
    STREET_NAME_ABBREVIATIONS,
    STREET_NAME_POST_ABBREVIATIONS,
    OCCUPANCY_TYPE_ABBREVIATIONS,
    STREET_TYPE_CODES,
    DIRECTION_CODES,
    EXTENSION_CODES
)
import programs.address_standardizer.number_processing as number_processing
import re
## It seems that we aren't using STREET_NAME_ABBR as its own category
STREET_NAME_POST_ABBREVIATIONS.update(STREET_NAME_ABBREVIATIONS) 

# replace if existent, return original (or other replacement) otherwise
# return dictionary[potential_key] if potential_key in dictionary else pkey
# note: user-defined replacement can't be None
def abbreviate(potential_key, dictionary):
    return dictionary[potential_key] if potential_key in dictionary else potential_key

# built for usaddress.tag, not usaddress.parse
# very preliminary, can be improved; let's talk about if we should parse replacements outside of specific labels
# applies abbreviate to each word parsed by usaddress

"""
input: tagged_address - output from usaddress.tag, in format Dict[label, words]
       master_dict - a dict of functions, in format Dict[label, function], for processing terms
output: a parsed address with words substituted when possible, in format Dict[label, substitution]
Dict[label, words], Dict[label, words -> substitution] -> Dict[label, substitution] 
"""
def clean(tagged_address, master_dict):
    cleaned = {}
    for (label, words) in tagged_address.items():
        if label in master_dict:
            result = master_dict[label](words)
            # TODO: handle HN parsing into multiple categories
            if label == "HN":
                if result:
                    cleaned["HN1"] = result[0]
                    cleaned["HNSEP"] = result[1]
                    cleaned["HN2"] = result[2]
                cleaned["HN"] = words
            else:
                cleaned[label] = result
            # apply to full phrase (e.g. "country road", "one hundred and one")
            # result = master_dict[label](words)
            # if result != words:
            #     cleaned[label] = master_dict[label](words)
            #     result
            # # apply to each word individually
            # else:
            #     cleaned[label] = " ".join([master_dict[label](word) for word in words.split(" ")])
        else:
            cleaned[label] = words
    return cleaned
    # [(master_dict[label](word), label) if label in master_dict else (word, label) for (word, label) in parsed_address]
# TODO: implement processing between single-word and full-phrase

# examples: "one hundred eighty first", "vermont", "one hundred eighty fouth washington street"


# substitution codes
code_dict = {
    'SNSD' : DIRECTION_CODES,
    'SNPD' : DIRECTION_CODES,
    'SNST' : STREET_TYPE_CODES,
    'SNPT' : STREET_TYPE_CODES,
    'SNE' : EXTENSION_CODES
}

"""
standardizes and replaces the following patterns:
    - state name to state abbreviations ("CALIFORNIA" -> "CA")
  if output = "number":
    - number words to numbers ("TWENTY THREE" -> "23")
      - handles hyphens, "and" ("ONE-HUNDRED-THREE", "ONE HUNDRED AND THREE")
      - handles ordinal words ("FORTY-FIFTH" -> "45")
    - ordinal number endings ("23RD" -> "23")
  if output = "word":
    - numbers to number words ("23" -> "TWENTY THREE")
      - handles ordinals ("23RD" -> "TWENTY THREE")
input: words, a string separated by spaces representing a street's name, uppercased
       ordinal, a boolean (default False) indicating if outputs are numerical ordinals
       output, either "number" (default) or "word" (indicating output format)
output: the same string, with relevant substitutions made
"""
# add options: raw numericals, numericals w/ ordinal endings, words, etc.
# port to preprocessing before standardizer (?); isolate
# -- make into a new(?) package / file, callable outside of usaddress
def street_process(words, ordinal = False, output = "number"):
    processed = []
    # terms = words.split()
    terms = re.split('-|\s',words)
    number_words = ""
    for word in terms:
        # check for number words if desired
        # needs to be updated to handle hyphens(?)
        if output == "number" and (word in number_processing.number_system or word == "and"):
            number_words = number_words + " " + word            
        else:
            # replace any possible abbreviations first; join unreplaced chunks together
            # can eventually make a more general "common abbreviations dict" if we want ?
            if number_words:
                processed.append(str(number_processing.word_to_number(number_words, ordinal)))
                number_words = ""

            # PROCESS NON-NUMERIC WORDS HERE
            # if abbreviations give you numerical results this will have to be changed
            word = abbreviate(word, STATE_ABBREVIATIONS)

            # turn "101st" to "101"
            ## BEWARE OF WORDS THAT CONTAIN NUMERIC CHARACTERS - ONLY THE NUMBER WILL REMAIN
            digits = "".join([d for d in word if d.isdigit()])
            if digits:
                # turn "101" to "one hundred one"
                if output == "word":
                    word = number_processing.number_to_word(digits, ordinal)
                elif not ordinal:
                    word = digits

            processed.append(word)
    if number_words:
        processed.append(str(number_processing.word_to_number(number_words)))
    return " ".join(processed).upper()


def HN_process(HN):
    # check if there is a separator
    separator = "".join([x for x in HN if not x.isnumeric()])
    if separator:
        parts = HN.split(separator)
        # fail if there isn't 1 separator or splitting doesn't work
        if len(parts) != 2:
            return False
        else:
            return (parts[0], separator, parts[1])
    else:
        return False


# dict of functions - how each label value should be standardized
processing_dict = {
    'SNST' : (lambda x : abbreviate(x, STREET_NAME_POST_ABBREVIATIONS)),
    'SNPT' : (lambda x : abbreviate(x, STREET_NAME_POST_ABBREVIATIONS)),
    'SNPD' : (lambda x : abbreviate(x, DIRECTIONAL_ABBREVIATIONS)),
    'SNSD' : (lambda x : abbreviate(x, DIRECTIONAL_ABBREVIATIONS)),
    'StateName' : (lambda x : abbreviate(x, STATE_ABBREVIATIONS)),
    'WSD' : (lambda x : abbreviate(x, OCCUPANCY_TYPE_ABBREVIATIONS)),
    'OSN' : street_process,
    'HN' : HN_process
}


# TODO: expand
label_mappings = {
    'AddressNumberPrefix' : 'HNPRE',
    'AddressNumber' : 'HN', # gets parsed into HN1, HN2 and HNSEP
    'AddressNumberSuffix' : 'HNSUF',
    'StreetNamePreModifier' : 'OSN', # concat w/ StreetName
    'StreetNamePreDirectional' : 'SNPD',
    'StreetNamePreType' : 'SNPT',
    'StreetName': 'OSN',
    'StreetNamePostType' : 'SNST',
    'StreetNamePostModifier' : 'SNE', # Not sure if this is the correct correspondence?
    'StreetNamePostDirectional' : 'SNSD',
    'SubaddressType' : 'WSDESC1',
    'SubaddressIdentifier' : 'WSID1',
    #rmw mod - undo if breaks
    'OccupancyIdentifier':'WS',
    'BuildingName' : 'SI',
    'ZipCode' : 'ZIP',
    'USPSBoxType' : 'BXD',
    'USPSBoxID' : 'BXI'
}

## HN = HN1 + HNSEP + HN2
## WS = WSDESC1 + WSID1

# Other labels:
#     'USPSBoxGroupType',
#     'USPSBoxGroupID',
#     'IntersectionSeparator',
#     'Recipient',
#     'NotAddress',
#     'OccupancyType',
#     'OccupancyIdentifier',
#     'CornerOf',
#     'LandmarkName',
#     'PlaceName',
#     'StateName',
# }


"""
input: address, any given address
       code, which can be "a" (append), "r" (replace), or "n" (none)
output: a list formatted like that of usaddress.parse, but with certain key words abbreviated and standardized
String -> Dict[label: String, word: String] ### REMOVED ### List[(word: String, label: String)]
"""
def standardize(address, code = "a"):
    if code not in ["a", "r", "n"]:
        raise InputError("code must be a (append), r (replace), or n (none)")
    # make case insensitive, apply usaddress parsing
    tagged = usaddress.tag(address.upper(), label_mappings)
    tagged = tagged[0]
    # remove punctuation from results (not removed beforehand, as punctuation can affect parsing)
    stripped = {label: words if label == 'HN' else \
        words.translate(str.maketrans('', '', string.punctuation)).strip() \
            for (label, words) in tagged.items()}

    # apply replacements
    substituted = clean(stripped, processing_dict)
    # add codes for directions, extensions, etc. if desired
    if code != "n":
        pairs = list(substituted.items())
        for (label, word) in pairs:
            # confirm label is substitutable and substitution is known
            if label in code_dict and word in code_dict.get(label):
                # add to dictionary
                substituted[label+"C"] = code_dict[label].get(word)
                # remove original value if requested
                if code == "r":
                    substituted.pop(label)
    # add concatenated WSN
    if "WSDESC1" in substituted or "WSID1" in substituted:
        if "WSDESC1" not in substituted:
            substituted["WS"] = substituted["WSID1"]
        elif "WSID1" not in substituted:
            substituted["WS"] = substituted["WSDESC1"]
        else:
            substituted["WS"] =" ".join([substituted["WSDESC1"], substituted["WSID1"]])
    return ' '.join([substituted[key] for key in substituted.keys()])

# if __name__== '__main__':
#     """
#     condition allows for 'interactive' testing and development when not being used as a library
    
#     None of this will be run when it is "imported" which is helpful/cleaner
#     """
#     #as a rule, try to avoid typing the same term over and over again, a standard input file helps with testing
#     testDataPath = r'testData.txt' # stored in current dir, for reasons...
#     with open(testDataPath, 'r') as temp:
#         data = [x[:-1] for x in temp.readlines()] #no header, 1 line per input, remove newline character
    
#     for item in data:
#         print(standardize(item))
        
#     print('\n\nDone!') #old habits die hard, helpful to know if something is hanging...
