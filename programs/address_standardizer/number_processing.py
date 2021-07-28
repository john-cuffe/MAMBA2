"""
OVERVIEW:
(If this expands more, this may become a full readme)
The two main functions of note here are:
- word_to_number, which returns a numeric string from spelled-out numbers
  -- e.g. "two hundred fifty six" -> "256"
- number_to_word, which returns a spelled-out string from int/double/numeric string
  -- e.g. 1250 -> "one thousand two hundred fifty"

Both functions have an optional boolean "ordinal" argument; if set to true,
the results will be ordinal numerals (i.e. "twenty first" or "21st")
-- the "word_ordinal_suffix" and "num_ordinal_suffix" functions will also apply
   this conversion directly
    
FUTURE/POTENTIAL IMPROVEMENTS:
- fraction parsing (i.e. "five thirds" <-> "5/3")
- ordinal suffix remover ("third" -> "three")
"""

import string


############# word_to_num globals ################
number_system = {
    'zero': 0,
    'oh': 0,
    'o': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'hundred': 100,
    'thousand': 10**3,
    'million': 10**6,
    'billion': 10**9,
    'trillion': 10**12,
    'quadrillion': 10**15,
    'quintillion': 10**18,
    'sextillion': 10**21,
    'septillion': 10**24,
    "octillion": 10**27,
    "nonillion": 10**30,
    "decillion": 10**33, 
    "undecillion": 10**36, 
    "duodecillion": 10**39, 
    "tredecillion": 10**42,
    "quattuordecillion": 10**45, 
    "quindecillion": 10**48, 
    "sexdecillion": 10**51, 
    "septendecillion": 10**54, 
    "octodecillion": 10**57, 
    "novemdecillion": 10**60, 
    "vigintillion": 10**63,
    'point': '.',
    'zeroth': 0,
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'fifth': 5,
    'sixth': 6,
    'seventh': 7,
    'eighth': 8,
    'ninth': 9,
    'tenth': 10,
    'eleventh': 11,
    'twelfth': 12,
    'thirteenth': 13,
    'fourteenth': 14,
    'fifteenth': 15,
    'sixteenth': 16,
    'seventeenth': 17,
    'eighteenth': 18,
    'nineteenth': 19,
    'twentieth': 20,
    'thirtieth': 30,
    'fortieth': 40,
    'fiftieth': 50,
    'sixtieth': 60,
    'seventieth': 70,
    'eightieth': 80,
    'ninetieth': 90,
    'hundredth': 100,
    'thousandth': 10**3,
    'millionth': 10**6,
    'billionth': 10**9,
    'trillionth': 10**12,
    'quadrillionth': 10**15,
    'quintillionth': 10**18,
    'sextillionth': 10**21,
    'septillionth': 10**24,
    "octillionth": 10**27,
    "nonillionth": 10**30,
    "decillionth": 10**33, 
    "undecillionth": 10**36, 
    "duodecillionth": 10**39, 
    "tredecillionth": 10**42,
    "quattuordecillionth": 10**45, 
    "quindecillionth": 10**48, 
    "sexdecillionth": 10**51, 
    "septendecillionth": 10**54, 
    "octodecillionth": 10**57, 
    "novemdecillionth": 10**60, 
    "vigintillionth": 10**63
    }

############# num_to_word globals ################
ones = ["", "one ","two ","three ","four ", "five ",
    "six ","seven ","eight ","nine "]
tens = ["ten ","eleven ","twelve ","thirteen ", "fourteen ",
    "fifteen ","sixteen ","seventeen ","eighteen ","nineteen "]
twenties = ["","","twenty ","thirty ","forty ",
    "fifty ","sixty ","seventy ","eighty ","ninety "]
thousands = ["","thousand ","million ", "billion ", "trillion ",
    "quadrillion ", "quintillion ", "sextillion ", "septillion ","octillion ",
    "nonillion ", "decillion ", "undecillion ", "duodecillion ", "tredecillion ",
    "quattuordecillion ", "quindecillion", "sexdecillion ", "septendecillion ", 
    "octodecillion ", "novemdecillion ", "vigintillion "]
decimals = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']







"""
standardizes and replaces the following patterns:
    - state name to state abbreviations ("CALIFORNIA" -> "CA")
    - number words to numbers ("TWENTY THREE" -> "23")
      - handles hyphens, "and" ("ONE-HUNDRED-THREE", "ONE HUNDRED AND THREE")
      - handles ordinal words ("FORTY-FIFTH" -> "45")
    - ordinal number endings ("23RD" -> "23")
input: words, a string separated by spaces representing a phrase (traditionally a street name, uppercased)
output: the same string, with relevant substitutions made
String -> String
"""
# TODO: add options: raw numericals, numericals w/ ordinal endings, words, etc.
# port to preprocessing before standardizer (?); isolate
# -- make into a new(?) package / file, callable outside of usaddress
def number_process(words, ordinal = False):
    processed = []
    terms = words.lower().split()
    number_words = ""
    # break into [processed, [unprocessed_phrases], processed, ...] blocks
    for word in terms:
        # check for number words
        if word in number_system or word == "and":
            number_words = number_words + " " + word
        else:
            if number_words:
                processed.append(str(word_to_number(number_words, ordinal)))
                number_words = ""
            # PROCESS NON-NUMERIC WORDS HERE
            # something like "result = process(word)" and then "processed.append(result)" below

            if not ordinal:
            # turn "101st" to "101"
            ## BEWARE OF WORDS THAT CONTAIN NUMERIC CHARACTERS - ONLY THE NUMBER WILL REMAIN
                digits = "".join([d for d in word if d.isdigit()])
                if digits:
                    word = digits
            processed.append(word)
    # handle number_words for the last time
    if number_words:
        processed.append(str(word_to_number(number_words)))
    return " ".join(processed).upper()


############# ordinal converters ################
"""
input: number sentence, a string series of words representing a number (i.e. "one hundred one")
output: number sentence with the last word modified to be the ordinal form ("one hundred first")
throws: ValueError, if input is not a string
"""
def word_ordinal_suffix(number_sentence):
    if type(number_sentence) is not str:
        raise ValueError("input must be string")

    # this is hardly elegant, but it does work pretty well
    elif number_sentence[-3:] == "one":
        return number_sentence[:-3] + "first"
    elif number_sentence[-3:] == "two":
        return number_sentence[:-3] + "second"
    elif number_sentence[-5:] == "three":
        return number_sentence[:-5] + "third"
    elif number_sentence[-1:] == "t":
        return number_sentence + "h"
    elif number_sentence[-2:] == "ve":
        return number_sentence[:-2] + "fth"
    elif number_sentence[-2:] == "ne":
        return number_sentence[:-1] + "th"
    elif number_sentence[-1:] == "y":
        return number_sentence[:-1] + "ieth"
    else:
        return number_sentence + "th"



"""
input: number, a numerical string representing an integer (i.e. "101")
output: number with an appropriate two-letter suffix appended (i.e. "101st")
throws: ValueError, if number has non-numerical characters (as per isnumeric())
"""
def num_ordinal_suffix(number):
    if not number.isnumeric():
        raise ValueError("input must be only numeric characters 0-9")
    teens = {"11" : "11th", "12" : "12th", "13" : "13th"}
    if number in {"11", "12", "13"}:
        match = "th"
    else:
        suffixes = {"1" : "st", "2" : "nd", "3" : "rd"}
        ending = number[-1]
        match = suffixes.get(ending) if ending in suffixes else "th"
    return number + match


############# number_to_word ################
# code built off & credit to:    
# https://www.daniweb.com/programming/software-development/code/216839/number-to-word-converter-python
"""
input: n, a number (int, float, etc. acceptable)
       ordinal, a boolean (if True, the output will in ordinal terms, i.e. "twenty second")
output: a string / text representation of n
"""
def number_to_word(n, ordinal = False):

    # break the number into groups of 3 digits using slicing
    # each group representing hundred, thousand, million, billion, ...
    if int(n) == 0:
        return "zero"
    n3 = []
    r1 = ""
    dec = ""
    # create numeric string
    ns = str(n)

    # check for decimals
    if "." in ns:
        # isolate the decimals & remove from the string
        decimal_index = ns.index(".")
        decimal_digits = ns[decimal_index+1:]
        ns = ns[:decimal_index]
        # process into words - rather blessedly simple
        decimal_words = ["point"]
        decimal_words.extend([decimals[int(digit)] for digit in decimal_digits])
        dec = " ".join(decimal_words)

    for k in range(3, 33, 3):
        r = ns[-k:]
        q = len(ns) - k
        # break if end of ns has been reached
        if q < -2:
            break
        else:
            if  q >= 0:
                n3.append(int(r[:3]))
            elif q >= -1:
                n3.append(int(r[:2]))
            elif q >= -2:
                n3.append(int(r[:1]))
        r1 = r
    
        # break each group of 3 digits into
    # ones, tens/twenties, hundreds
    # and form a string
    nw = ""
    for i, x in enumerate(n3):
        if x == 0:
            continue  # skip
        else:
            t = thousands[i]
            digit_1 = x % 10
            digit_2 = (x % 100)//10
            digit_3 = (x % 1000)//100
            if digit_2 == 0:
                nw = ones[digit_1] + t + nw
            elif digit_2 == 1:
                nw = tens[digit_1] + t + nw
            elif digit_2 > 1:
                nw = twenties[digit_2] + ones[digit_1] + t + nw
            if digit_3 > 0:
                nw = ones[digit_3] + "hundred " + nw
    result = (nw + dec).strip()
    if ordinal:
        result = word_ordinal_suffix(result)
    return result


############# word_to_number and helpers ################
"""
input: number_words, a string of words that encode an integer
       ordinal, a boolean (if True, the output will have an ordinal suffix i.e. "nth")
output: a string of the corresponding integer (with or without ordinal suffix), interpreted:
  -- words outside of the number_system corpus will be removed
  -- supports certain non-standard ways to represent numbers
    -- "year-like" formats:
      -- "seventeen seventy-six" -> 1776
      -- "nineteen hundred fifty three" -> 1953
      -- "twenty oh eight" -> 2008
    -- "telephone number-like" formats:
      -- "seven seven three two oh two" -> 773202
      -- "two eight hundred eighty eight" -> 280088
    !! these outputs are *not* guaranteed to be entirely consistent !!
"""
def word_to_number(number_words, ordinal = False):
    number_words, decimal_words = clean_numbers(number_words)
    results = 0
    chunk = []
    for element in number_words:
        value = number_system[element]
        # break into three-digit chunks, scaled (xxx thousand, million, etc.)
        if value > 999:
            results += (sub_thousands(chunk) * value) if chunk else value
            chunk = []
        else:
            chunk.append(value)
    # handle the hundreds/tens/ones
    if chunk:
        results += sub_thousands(chunk)
    results = str(results)

    # parse decimals
    if decimal_words:
        decimal_values = [str(number_system[number]) for number in decimal_words]
        decimal_values = "".join(decimal_values)
        results = results + "." + decimal_values

    # add ordinal suffix as needed
    if ordinal:
        results = num_ordinal_suffix(results)
    return results


## try to include nonstandard ways of saying numbers ("two-twenty" = 220, "sixteen hundred" = 1600)
"""
generates numbers in the hundreds, tens, and ones digits
input: numbers, a list of ints from 0-999
output: a single int, combined from the values of numbers
"""
def sub_thousands(numbers):
    # 
    for index in range(1, len(numbers)):
        # scale "hundreds" by the previous figure
        if numbers[index] == 100:
            numbers[index-1] *= numbers[index]
            numbers[index] = -1
        # turn "20" and "3" into "23"
        elif numbers[index-1] > 19 and numbers[index] < 10 and numbers[index] > 0:
            numbers[index-1] += numbers[index]
            numbers[index] = -1

    # -1 is a placeholder
    numbers = list(filter(lambda x: x != -1, numbers))
    str_numbers = list(map(lambda x: str(x), numbers))

    # whether or not numbers are added or interpreted as a sequence depends on
    # if they "fit together" - "two hundred thirty two" is intuitively added to 232, 
    # but "eighteen sixty-five" is concatenated to "1865"
    ordered = True
    idx = 1
    while ordered and idx < len(str_numbers):
        # this is an imperfect heuristic but it works for the properly-formatted case
        # and works for most variant cases
        if len(str_numbers[idx - 1]) > len(str_numbers[idx]):
            ordered = ordered and int(str_numbers[idx - 1][-len(str_numbers[idx])]) == 0
        else:
            ordered = False
        idx += 1

    if ordered:
        return sum(numbers)
    else:
        return int("".join(str_numbers))
        
"""
preprocesses input text for word_to_number
input: number_sentence, a string of words that encode an integer (from word_to_number)
output: number_sentence, stripped of whitespace, punctuation characters, and other words
"""
def clean_numbers(number_sentence):
    if type(number_sentence) is not str:
        raise ValueError("input must be string")

    number_sentence = number_sentence.replace('-', ' ')
    number_sentence = number_sentence.lower()  # converting input to lowercase
    number_sentence = number_sentence.translate(str.maketrans('', '', string.punctuation)) 

    split_words = number_sentence.strip().split()  # strip extra spaces and split sentence into words

    clean_decimals = []

    if "point" in split_words:
        decimal_index = split_words.index("point")
        clean_decimals = split_words[decimal_index+1:]
        ## number_system also works but decimals is more precise
        clean_decimals = [word for word in clean_decimals if word in decimals]
        split_words = split_words[:decimal_index]

    # removing and, & etc.
    # TODO: refine; keep non-numeric words; potentially parse fractions
    clean_numbers = [word for word in split_words if word in number_system]
    # TODO: expand to decimals
    # clean_decimal_numbers = []

    if not clean_numbers:
        raise ValueError("input must contain number-related words")

    return (clean_numbers, clean_decimals)


# ## lazy testing:
# def print_if_error(expression, result):
#     if expression != result:
#         print(expression)
#         print(result)
# # word_to_number, standard
# print_if_error(word_to_number("two hundred ninety three million"), "293000000")
# print_if_error(word_to_number("seventy fifth"), "75")
# print_if_error(word_to_number("one"), "1")
# print_if_error(word_to_number("seventeen"), "17")
# print_if_error(word_to_number("two hundred twenty two"), "222")
# print_if_error(word_to_number("one billion two thousand"), "1000002000")
# print_if_error(word_to_number("one eighty five"), "185")
# print_if_error(word_to_number("seventy hundred eighty five"), "7085")
# print_if_error(word_to_number("seventeen twelve"), "1712")
# print_if_error(word_to_number("ten twenty three one"), "10231")
# print_if_error(word_to_number("seventeen thousand six twenty three"), "17623")
# print_if_error(word_to_number("seventeen thousand six hundred twenty three"), "17623") 
# print_if_error(word_to_number("one million six twenty three thousand two fifty"), "1623250")
# print_if_error(word_to_number("thousand and one"), "1001")
# print_if_error(word_to_number("zero"), "0")
# print_if_error(word_to_number("zero million"), "0")
# print_if_error(word_to_number("twenty zero three"), "2003")
# print_if_error(word_to_number("seventeen three nine"), "1739")
# print_if_error(word_to_number("eight hundred five eight eight"), "800588")
# print_if_error(word_to_number("eight hundred seventeen eight"), "800178")
# print_if_error(word_to_number("eight hundred seventy eight"), "878")

# # word_to_number, ordinal
# print_if_error(word_to_number("ten ten twenty three", ordinal = True), "101023rd")
# print_if_error(word_to_number("ten ten twenty third", ordinal = True), "101023rd")
# print_if_error(word_to_number("one hundred eighty seventh", ordinal = True), "187th")
# print_if_error(word_to_number("one thousand one", ordinal = True), "1001st")
# print_if_error(word_to_number("thirteenth", ordinal = True), "13th")
# print_if_error(word_to_number("twenty-two", ordinal = True), "22nd")

# #number_to_word, ordinal
# print_if_error(number_to_word("101023", ordinal = True), "one hundred one thousand twenty third")
# print_if_error(number_to_word("241", ordinal = True), "two hundred forty first")
# print_if_error(number_to_word("15", ordinal = True), "fifteenth")
# print_if_error(number_to_word("112", ordinal = True), "one hundred twelfth")
# print_if_error(number_to_word("10009", ordinal = True), "ten thousand ninth")
# print_if_error(number_to_word("12080", ordinal = True), "twelve thousand eightieth")
# print_if_error(number_to_word("17623", ordinal = True), "seventeen thousand six hundred twenty third") 

# #number_to_word
# print_if_error(number_to_word("17623"), "seventeen thousand six hundred twenty three") 

# print("done")