'''
Credit: https://github.com/CodeDrome/soundex-python and fuzzy
'''
import os

def soundex(name):

    """
    The Soundex algorithm assigns a 1-letter + 3-digit code to strings,
    the intention being that strings pronounced the same but spelled
    differently have identical encodings; words pronounced similarly
    should have similar encodings.
    """

    soundexcoding = [' ', ' ', ' ', ' ']
    soundexcodingindex = 1

    #           ABCDEFGHIJKLMNOPQRSTUVWXYZ
    mappings = "01230120022455012623010202"

    soundexcoding[0] = name[0].upper()

    for i in range(1, len(name)):

         c = ord(name[i].upper()) - 65

         if c >= 0 and c <= 25:

             if mappings[c] != '0':

                 if mappings[c] != soundexcoding[soundexcodingindex-1]:

                     soundexcoding[soundexcodingindex] = mappings[c]
                     soundexcodingindex += 1

                 if soundexcodingindex > 3:

                     break

    if soundexcodingindex <= 3:
        while(soundexcodingindex <= 3):
            soundexcoding[soundexcodingindex] = '0'
            soundexcodingindex += 1

    return ''.join(soundexcoding)


import re

_vowels = 'AEIOU'


def replace_at(text, position, fromlist, tolist):
    for f, t in zip(fromlist, tolist):
        if text[position:].startswith(f):
            return ''.join([text[:position],
                            t,
                            text[position + len(f):]])
    return text


def replace_end(text, fromlist, tolist):
    for f, t in zip(fromlist, tolist):
        if text.endswith(f):
            return text[:-len(f)] + t
    return text


def nysiis(name):
    name = re.sub(r'\W', '', name).upper()
    name = replace_at(name, 0, ['MAC', 'KN', 'K', 'PH', 'PF', 'SCH'],
                      ['MCC', 'N', 'C', 'FF', 'FF', 'SSS'])
    name = replace_end(name, ['EE', 'IE', 'DT', 'RT', 'RD', 'NT', 'ND'],
                       ['Y', 'Y', 'D', 'D', 'D', 'D', 'D'])
    key, key1 = name[0], ''
    i = 1
    while i < len(name):
        # print(i, name, key1, key)
        n_1, n = name[i - 1], name[i]
        n1_ = name[i + 1] if i + 1 < len(name) else ''
        name = replace_at(name, i, ['EV'] + list(_vowels), ['AF'] + ['A'] * 5)
        name = replace_at(name, i, 'QZM', 'GSN')
        name = replace_at(name, i, ['KN', 'K'], ['N', 'C'])
        name = replace_at(name, i, ['SCH', 'PH'], ['SSS', 'FF'])
        if n == 'H' and (n_1 not in _vowels or n1_ not in _vowels):
            name = ''.join([name[:i], n_1, name[i + 1:]])
        if n == 'W' and n_1 in _vowels:
            name = ''.join([name[:i], 'A', name[i + 1:]])
        if key and key[-1] != name[i]:
            key += name[i]
        i += 1
    key = replace_end(key, ['S', 'AY', 'A'], ['', 'Y', ''])
    return key1 + key

if __name__=='__main__':
    print('Why did you do this?')
    os._exit(0)