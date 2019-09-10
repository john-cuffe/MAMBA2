# =============================================================================
# AUSTRALIAN NATIONAL UNIVERSITY OPEN SOURCE LICENSE (ANUOS LICENSE)
# VERSION 1.3
#
# The contents of this file are subject to the ANUOS License Version 1.3
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at:
#
#   https://sourceforge.net/projects/febrl/
#
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See
# the License for the specific language governing rights and limitations
# under the License.
#
# The Original Software is: "encode.py"
#
# The Initial Developer of the Original Software is:
#   Dr Peter Christen (Research School of Computer Science, The Australian
#                      National University)
#
# Copyright (C) 2002 - 2011 the Australian National University and
# others. All Rights Reserved.
#
# Contributors:
#
# Alternatively, the contents of this file may be used under the terms
# of the GNU General Public License Version 2 or later (the "GPL"), in
# which case the provisions of the GPL are applicable instead of those
# above. The GPL is available at the following URL: http://www.gnu.org/
# If you wish to allow use of your version of this file only under the
# terms of the GPL, and not to allow others to use your version of this
# file under the terms of the ANUOS License, indicate your decision by
# deleting the provisions above and replace them with the notice and
# other provisions required by the GPL. If you do not delete the
# provisions above, a recipient may use your version of this file under
# the terms of any one of the ANUOS License or the GPL.
# =============================================================================
#
# Freely extensible biomedical record linkage (Febrl) - Version 0.4.2
#
# See: http://datamining.anu.edu.au/linkage.html
#
# =============================================================================

"""Module encode.py - Various phonetic name encoding methods.

Encoding methods provided:

  soundex         Soundex
  mod_soundex     Modified Soundex
  phonex          Phonex
  nysiis          NYSIIS
  dmetaphone      Double-Metaphone
  phonix          Phonix
  fuzzy_soundex   Fuzzy Soundex based on q-gram substitutions and letter
                  encodings
  get_substring   Simple function which extracts and returns a sub-string
  freq_vector     Count characters and put into a vector

See doc strings of individual routines for detailed documentation.

There is also a routine called 'phonix_transform' which only performs the
Phonix string transformation without the final numerical encoding. This can
be useful for approximate string comparison functions.

Note that all encoding routines assume the input string only contains letters
and whitespaces, but not digits or other ASCII characters.

If called from the command line, a test routine is run which prints example
encodings for various strings.
"""

# =============================================================================
# Imports go here

import logging
import string
import time

# =============================================================================

def do_encode(encode_method, in_str):
  """A 'chooser' functions which performs the selected string encoding method.

  For each encoding method, two calling versions are provided. One limiting the
  length of the code to 4 characters (and possibly pads shorter codes with a
  fill character, for example '0' for soundex), the other returning an
  unlimited length code.

  Possible values for 'encode_method' are:

    soundex           Unlimited length Soundex encoding
    soundex4          Soundex limited/padded to length 4
    mod_soundex       Modified unlimited length Soundex encoding
    mod_soundex4      Modified Soundex limited/padded to length 4
    phonex            Unlimited length Phonex encoding
    phonex4           Phonex limited/padded to length 4
    phonix            Unlimited length Phonix encoding
    phonix4           Phonix limited/padded to length 4
    phonix_transform  Only perform Phonix string transformation without
                      numerical encoding
    nysiis            Unlimited length NYSIIS encoding
    nysiis4           NYSIIS limited/padded to length 4
    dmetaphone        Unlimited length Double-Metaphone encoding
    dmetaphone4       Double-Metaphone limited/padded to length 4
    fuzzy_soundex     Fuzzy Soundex
    fuzzy_soundex4    Fuzzy Soundex limited/padded to length 4

  This functions returns the phonetic code as well as the time needed to
  generate it (as floating-point value in seconds).
  """

  if (encode_method[-1] == '4'):
    maxlen = 4
  else:
    maxlen = -1

  if (encode_method.startswith('soundex')):
    start_time = time.time()
    phonetic_code = soundex(in_str, maxlen)
    time_used = time.time() - start_time

  elif (encode_method.startswith('mod_soundex')):
    start_time = time.time()
    phonetic_code = mod_soundex(in_str, maxlen)
    time_used = time.time() - start_time

  elif (encode_method.startswith('phonex')):
    start_time = time.time()
    phonetic_code = phonex(in_str, maxlen)
    time_used = time.time() - start_time

  elif (encode_method.startswith('phonix_transform')):
    start_time = time.time()
    phonetic_code = phonix_transform(in_str)
    time_used = time.time() - start_time

  elif (encode_method.startswith('phonix')):
    start_time = time.time()
    phonetic_code = phonix(in_str, maxlen)
    time_used = time.time() - start_time

  elif (encode_method.startswith('nysiis')):
    start_time = time.time()
    phonetic_code = nysiis(in_str, maxlen)
    time_used = time.time() - start_time

  elif (encode_method.startswith('dmetaphone')):
    start_time = time.time()
    phonetic_code = dmetaphone(in_str, maxlen)
    time_used = time.time() - start_time

  elif (encode_method.startswith('fuzzy_soundex')):
    start_time = time.time()
    phonetic_code = fuzzy_soundex(in_str, maxlen)
    time_used = time.time() - start_time

  else:
    logging.exception('Illegal string encoding method: %s' % (encode_method))
    raise Exception

  return phonetic_code, time_used

# =============================================================================

def soundex(s, maxlen=4):
  """Compute the soundex code for a string.

  USAGE:
    code = soundex(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.
             If 'maxlen' is negative the soundex code will not be padded with
             '0' to 'maxlen' characters.

  DESCRIPTION:
    For more information on Soundex see:
    - http://www.bluepoof.com/Soundex/info.html
    - http://www.nist.gov/dads/HTML/soundex.html
  """

  if (not s):
    if (maxlen > 0):
      return maxlen*'0'  # Or 'z000' for compatibility with other
                         # implementations
    else:
      return '0'

  # Translation table and characters that will not be used for soundex  - - - -
  #
  transtable = string.maketrans('abcdefghijklmnopqrstuvwxyz', \
                                '01230120022455012623010202')
  # deletechars='aeiouhwy '
  deletechars = ' '

  s2 = string.translate(s[1:],transtable,deletechars)

  s3 = s[0]  # Keep first character of original string

  # Only add numbers if they are not the same as the previous number
  #
  for i in s2:
    if (i != s3[-1]):
      s3 = s3+i

  # Remove all '0'
  s4 = s3.replace('0', '')

  # Fill up with '0' to maxlen length
  #
  s4 = s4+maxlen*'0'

  if (maxlen > 0):
    resstr = s4[:maxlen]  # Return first maxlen characters
  else:
    resstr = s4

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Soundex encoding for string: "%s": %s' % (s, resstr))

  return resstr

# =============================================================================

def mod_soundex(s, maxlen=4):
  """Compute the modified soundex code for a string.

  USAGE:
    code = mod_soundex(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.
             If 'maxlen' is negative the soundex code will not be padded with
             '0' to 'maxlen' characters.

  DESCRIPTION:
    For more information on the modified Soundex see:
    - http://www.bluepoof.com/Soundex/info2.html
  """

  # Translation table and characters that will not be used for soundex  - - - -
  #
  transtable = string.maketrans('abcdefghijklmnopqrstuvwxyz', \
                                '01360240043788015936020505')
  deletechars='aeiouhwy '

  if (not s):
    if (maxlen > 0):
      return maxlen*'0'  # Or 'z000' for compatibility with other
                         # implementations
    else:
      return '0'

  s2 = string.translate(s[1:],transtable, deletechars)

  s3 = s[0]  # Keep first character of original string

  # Only add numbers if they are not the same as the previous number
  for i in s2:
    if (i != s3[-1]):
      s3 = s3+i

  # Fill up with '0' to maxlen length
  #
  s4 = s3+maxlen*'0'

  if (maxlen > 0):
    resstr = s4[:maxlen]  # Return first maxlen characters
  else:
    resstr = s4

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Mod Soundex encoding for string: "%s": %s' % (s, resstr))

  return resstr

# =============================================================================

def phonex(s, maxlen=4):
  """Compute the phonex code for a string.

  USAGE:
    code = phonex(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.
             If 'maxlen' is negative the soundex code will not be padded with
             '0' to 'maxlen' characters.

  DESCRIPTION:
    Based on the algorithm as described in:
    "An Assessment of Name Matching Algorithms, A.J. Lait and B. Randell,
     Technical Report number 550, Department of Computing Science,
     University of Newcastle upon Tyne, 1996"

    Available at:
      http://www.cs.ncl.ac.uk/~brian.randell/home.informal/
             Genealogy/NameMatching.pdf

    Bug-fixes regarding 'h','ss','hss' etc. strings thanks to Marion Sturtevant
  """

  if (not s):
    if (maxlen > 0):
      return maxlen*'0'  # Or 'z000' for compatibility with other
                         # implementations
    else:
      return '0'

  # Preprocess input string - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  while (s and s[-1] == 's'):  # Remove all 's' at the end
    s = s[:-1]

  if (not s):
    if (maxlen > 0):
      return maxlen*'0'  # Or 'z000' for compatibility with other
                         # implementations
    else:
      return '0'

  if (s[:2] == 'kn'):    # Remove 'k' from beginning if followed by 'n'
    s = s[1:]
  elif (s[:2] == 'ph'):  # Replace 'ph' at beginning with 'f'
    s = 'f'+s[2:]
  elif (s[:2] == 'wr'):  # Remove 'w' from beginning if followed by 'r'
    s = s[1:]

  if (s[0] == 'h'):      # Remove 'h' from beginning
    s = s[1:]

  if (not s):
    if (maxlen > 0):
      return maxlen*'0'  # Or 'z000' for compatibility with other
                         # implementations
    else:
      return '0'

  # Make phonetic equivalence of first character
  #
  if (s[0] in 'eiouy'):
    s = 'a'+s[1:]
  elif (s[0] == 'p'):
    s = 'b'+s[1:]
  elif (s[0] == 'v'):
    s = 'f'+s[1:]
  if (s[0] in 'kq'):
    s = 'c'+s[1:]
  elif (s[0] == 'j'):
    s = 'g'+s[1:]
  elif (s[0] == 'z'):
    s = 's'+s[1:]

  # Modified soundex coding - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  s_len = len(s)
  code = ''  # Phonex code
  i = 0

  while (i < s_len):  # Loop over all characters in s
    s_i = s[i]
    code_i = '0'  # Default code

    if (s_i in 'bfpv'):
      code_i = '1'

    elif (s_i in 'cskgjqxz'):
      code_i = '2'

    elif (s_i in 'dt') and (i < s_len-1) and (s[i+1] != 'c'):
      code_i = '3'

    elif (s_i == 'l') and ((i == s_len-1) or \
                           ((i < s_len-1) and (s[i+1] in 'aeiouy'))):
      code_i = '4'

    elif (s_i in 'mn'):
      code_i = '5'
      if (i < s_len-1) and (s[i+1] in 'dg'):
        s = s[:i+1]+s_i+s[i+2:]  # Replace following D or G with current M or N

    elif (s_i == 'r') and ((i == s_len-1) or \
                           ((i < s_len-1) and (s[i+1] in 'aeiouy'))):
      code_i = '6'

    if (i == 0):  # Handle beginning of string
      last = code_i
      code += s_i  # Start code with a letter

    else:
      if (code_i != last) and (code_i != '0'):

        # If the code differs from previous code and it's not a vowel code
        #
        code += code_i

      last = code[-1]
    i += 1

  # Fill up with '0' to maxlen length
  #
  code += maxlen*'0'

  if (maxlen > 0):
    resstr = code[:maxlen]  # Return first maxlen characters
  else:
    resstr = code

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Phonex encoding for string: "%s": %s' % (s, resstr))

  return resstr

# =============================================================================

def phonix(s, maxlen=4):
  """Compute the phonix code for a string.

  USAGE:
    code = phonix(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.
             If 'maxlen' is negative the soundex code will not be padded with
             '0' to 'maxlen' characters.

  DESCRIPTION:
    Based on the Phonix implementation from Ulrich Pfeifer's WAIS, see:

      http://search.cpan.org/src/ULPFR/WAIT-1.800/

    For more information on Phonix see:
    "PHONIX: The algorithm", Program: automated library and information
    systems, 24(4),363-366, 1990, by T. Gadd
  """

  if (not s):
    if (maxlen > 0):
      return 'a'+(maxlen-1)*'0'
    else:
      return ''

  # First apply Phonix transformation
  #
  phonixstr = phonix_transform(s)

  if (phonixstr == ''):
    if (maxlen > 0):
      return 'a'+(maxlen-1)*'0'
    else:
      return ''

  # Translation table and characters that will not be used for Phonix
  #
  transtable = string.maketrans('abcdefghijklmnopqrstuvwxyz', \
                                '01230720022455012683070808')

  # deletechars='aeiouhwy '
  deletechars = ' '

  s2 = string.translate(phonixstr[1:],transtable,deletechars)

  # If first character is a vowel or 'y' replace it with 'V' otherwise keep it
  # (assume all other characters are lowercase)
  #
  if (phonixstr[0] in 'aeiouy'):
    s3 = 'V'  # Different from lowercase 'v'
  else:
    s3 = phonixstr[0]

  # Only add numbers if they are not the same as the previous number
  #
  for i in s2:
    if (i != s3[-1]):
      s3 = s3+i

  # Remove all '0'
  s4 = s3.replace('0', '')

  # Fill up with '0' to maxlen length
  #
  s4 = s4+maxlen*'0'

  if (maxlen > 0):
    resstr = s4[:maxlen]  # Return first maxlen characters
  else:
    resstr = s4

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Phonix encoding for string: "%s": %s' % (s, resstr))

  return resstr

# =============================================================================

def phonix_transform(s):
  """Do Phonix transformation for a string.

  USAGE:
    phonixstr = phonix_transform(s, maxlen)

  ARGUMENTS:
    s  A string containing a name.

  DESCRIPTION:
    This function only does the Phonix transformation of the given input string
    without the final numerical encoding.

    Based on the Phonix implementation from Ulrich Pfeifer's WAIS, see:

      http://search.cpan.org/src/ULPFR/WAIT-1.800/

    For more information on Phonix see:
    "PHONIX: The algorithm", Program: automated library and information
    systems, 24(4),363-366, 1990, by T. Gadd
  """

  if (s == ''):
    return s

  # Function which replaces a pattern in a string - - - - - - - - - - - - - - -
  # - where can be one of: 'ALL','START','END','MIDDLE'
  # - Pre-condition (default None) can be 'V' for vowel or 'C' for consonant
  # - Post-condition (default None) can be 'V' for vowel or 'C' for consonant
  #
  def phonix_replace(s, where, orgpat, newpat, precond, postcond):

    vowels = 'aeiouy'

    tmpstr = s

    start_search = 0  # Position from where to start the search
    pat_len =   len(orgpat)

    while (orgpat in tmpstr[start_search:]):  # As long as pattern is in string

      pat_start = tmpstr.find(orgpat,start_search)
      str_len =   len(tmpstr)

      # Check conditions of previous and following character
      #
      OKpre = False   # Previous character condition
      OKpost = False  # Following character condition

      if (precond == None):
        OKpre = True
      elif (pat_start > 0):
        if (((precond == 'V') and (tmpstr[pat_start-1] in vowels)) or \
            ((precond == 'C') and (tmpstr[pat_start-1] not in vowels))):
          OKpre = True

      if (postcond == None):
        OKpost = True
      else:
        pat_end = pat_start+pat_len
        if (pat_end < str_len):
          if (((postcond == 'V') and (tmpstr[pat_end] in vowels)) or \
              ((postcond == 'C') and (tmpstr[pat_end] not in vowels))):
            OKpost = True

      # Replace pattern if conditions and position OK
      #
      if ((OKpre == True) and (OKpost == True)) and \
         (((where == 'START') and (pat_start == 0)) or \
          ((where == 'MIDDLE') and (pat_start > 0) and \
                                   (pat_start+pat_len < str_len)) or \
          ((where == 'END') and (pat_start+pat_len == str_len)) or \
          (where == 'ALL')):
        tmpstr = tmpstr[:pat_start]+newpat+tmpstr[pat_start+pat_len:]

        start_search = pat_start
      else:
        #start_search += 1
        start_search = pat_start+1

    return tmpstr

  # Replacement table according to Gadd's definition - - - - - - - - - - - - -
  #
  replace_table = [('ALL',    'dg',    'g'),
                   ('ALL',    'co',    'ko'),
                   ('ALL',    'ca',    'ka'),
                   ('ALL',    'cu',    'ku'),
                   ('ALL',    'cy',    'si'),
                   ('ALL',    'ci',    'si'),
                   ('ALL',    'ce',    'se'),
                   ('START',  'cl',    'kl',    None, 'V'),
                   ('ALL',    'ck',    'k'),
                   ('END',    'gc',    'k'),
                   ('END',    'jc',    'k'),
                   ('START',  'chr',   'kr',    None, 'V'),
                   ('START',  'cr',    'kr',    None, 'V'),
                   ('START',  'wr',    'r'),
                   ('ALL',    'nc',    'nk'),
                   ('ALL',    'ct',    'kt'),
                   ('ALL',    'ph',    'f'),
                   ('ALL',    'aa',    'ar'),
                   ('ALL',    'sch',   'sh'),
                   ('ALL',    'btl',   'tl'),
                   ('ALL',    'ght',   't'),
                   ('ALL',    'augh',  'arf'),
                   ('MIDDLE', 'lj',    'ld',    'V',  'V'),
                   ('ALL',    'lough', 'low'),
                   ('START',  'q',     'kw'),
                   ('START',  'kn',    'n'),
                   ('END',    'gn',    'n'),
                   ('ALL',    'ghn',   'n'),
                   ('END',    'gne',   'n'),
                   ('ALL',    'ghne',  'ne'),
                   ('END',    'gnes',  'ns'),
                   ('START',  'gn',    'n'),
                   ('MIDDLE', 'gn',    'n',     None, 'C'),
                   ('END',    'gn',    'n'),                # None, 'C'
                   ('START',  'ps',    's'),
                   ('START',  'pt',    't'),
                   ('START',  'cz',    'c'),
                   ('MIDDLE', 'wz',    'z',     'V',  None),
                   ('MIDDLE', 'cz',    'ch'),
                   ('ALL',    'lz',    'lsh'),
                   ('ALL',    'rz',    'rsh'),
                   ('MIDDLE', 'z',     's',     None, 'V'),
                   ('ALL',    'zz',    'ts'),
                   ('MIDDLE', 'z',     'ts',    'C',  None),
                   ('ALL',    'hroug', 'rew'),
                   ('ALL',    'ough',  'of'),
                   ('MIDDLE', 'q',     'kw',    'V',  'V'),
                   ('MIDDLE', 'j',     'y',     'V',  'V'),
                   ('START',  'yj',    'y',     None, 'V'),
                   ('START',  'gh',    'g'),
  #                ('END',    'e',     'gh',    'V', None), # Wrong in Pfeifer
                   ('END',    'gh',    'e',     'V', None), # From Zobel code
                   ('START',  'cy',    's'),
                   ('ALL',    'nx',    'nks'),
                   ('START',  'pf',    'f'),
                   ('END',    'dt',    't'),
                   ('END',    'tl',    'til'),
                   ('END',    'dl',    'dil'),
                   ('ALL',    'yth',   'ith'),
                   ('START',  'tj',    'ch',    None, 'V'),
                   ('START',  'tsj',   'ch',    None, 'V'),
                   ('START',  'ts',    't',     None, 'V'),
                   ('ALL',    'tch',   'ch'),  # Wrong funct call in Pfeifer
                   ('MIDDLE', 'wsk',   'vskie', 'V',  None),
                   ('END',    'wsk',   'vskie', 'V',  None),
                   ('START',  'mn',    'n',     None, 'V'),
                   ('START',  'pn',    'n',     None, 'V'),
                   ('MIDDLE', 'stl',   'sl',    'V',  None),
                   ('END',    'stl',   'sl',    'V',  None),
                   ('END',    'tnt',   'ent'),
                   ('END',    'eaux',  'oh'),
                   ('ALL',    'exci',  'ecs'),
                   ('ALL',    'x',     'ecs'),
                   ('END',    'ned',   'nd'),
                   ('ALL',    'jr',    'dr'),
                   ('END',    'ee',    'ea'),
                   ('ALL',    'zs',    's'),
                   ('MIDDLE', 'r',     'ah',    'V',  'C'),
                   ('END',    'r',     'ah',    'V',  None),  # 'V', 'C'
                   ('MIDDLE', 'hr',    'ah',    'V',  'C'),
                   ('END',    'hr',    'ah',    'V',  None),  # 'V', 'C'
                   ('END',    'hr',    'ah',    'V',  None),
                   ('END',    're',    'ar'),
                   ('END',    'r',     'ah',    'V',  None),
                   ('ALL',    'lle',   'le'),
                   ('END',    'le',    'ile',   'C',  None),
                   ('END',    'les',   'iles',  'C',  None),
                   ('END',    'e',     ''),
                   ('END',    'es',    's'),
                   ('END',    'ss',    'as',    'V',  None),
                   ('END',    'mb',    'm',     'V',  None),
                   ('ALL',    'mpts',  'mps'),
                   ('ALL',    'mps',   'ms'),
                   ('ALL',    'mpt',   'mt')]

  workstr = s

  for rtpl in replace_table:  # Check all transformations in the table

    if (len(rtpl) == 3):
      rtpl += (None,None)

    workstr = phonix_replace(workstr,rtpl[0],rtpl[1],rtpl[2],rtpl[3],rtpl[4])

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Phonix transformation: "%s" into "%s"' % (s, workstr))

  return workstr

# =============================================================================

def nysiis(s, maxlen=4):
  """Compute the NYSIIS code for a string.

  USAGE:
    code = nysiis(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.
             If 'maxlen' is negative the soundex code will not be padded with
             '0' to 'maxlen' characters.

  DESCRIPTION:
    For more information on NYSIIS see:
    - http://www.dropby.com/indexLF.html?content=/NYSIIS.html
    - http://www.nist.gov/dads/HTML/nysiis.html
  """

  if (not s):
    return ''

  # Remove trailing S or Z
  #
  while s and s[-1] in 'sz':
    s = s[:-1]

  # Translate first characters of string
  #
  if (s[:3] == 'mac'):  # Initial 'MAC' -> 'MC'
    s = 'mc'+s[3:]
  elif (s[:2] == 'pf'):  # Initial 'PF' -> 'F'
    s = s[1:]

  # Translate some suffix characters:
  #
  suff_dict = {'ix':'ic', 'ex':'ec', 'ye':'y', 'ee':'y', 'ie':'y', \
               'dt':'d', 'rt':'d', 'rd':'d', 'nt':'n', 'nd':'n'}
  suff = s[-2:]
  s = s[:-2]+suff_dict.get(suff, suff)  # Replace suffix if in dictionary

  # Replace EV with EF
  #
  if (s[2:].find('ev') > -1):
    s = s[:-2]+s[2:].replace('ev','ef')

  if (not s):
    return ''

  first = s[0]  # Save first letter for final code

  # Replace all vowels with A and delete whitespaces
  #
  voweltable = string.maketrans('eiou', 'aaaa')
  s2 = string.translate(s,voweltable, ' ')

  if (not s2):  # String only contained whitespaces
    return ''

  # Remove all W that follow an A
  #
  s2 = s2.replace('aw','a')

  # Various replacement patterns
  #
  s2 = s2.replace('ght','gt')
  s2 = s2.replace('dg','g')
  s2 = s2.replace('ph','f')
  s2 = s2[0]+s2[1:].replace('ah','a')
  s3 = s2[0]+s2[1:].replace('ha','a')
  s3 = s3.replace('kn','n')
  s3 = s3.replace('k','c')
  s4 = s3[0]+s3[1:].replace('m','n')
  s5 = s4[0]+s4[1:].replace('q','g')
  s5 = s5.replace('sh','s')
  s5 = s5.replace('sch','s')
  s5 = s5.replace('yw','y')
  s5 = s5.replace('wr','r')

  # If not first or last, replace Y with A
  #
  s6 = s5[0]+s5[1:-1].replace('y','a')+s5[-1]

  # If not first character, replace Z with S
  #
  s7 = s6[0]+s6[1:].replace('z','s')

  # Replace trailing AY with Y
  #
  if (s7[-2:] == 'ay'):
    s7 = s7[:-2]+'y'

  # Remove trailing vowels (now only A)
  #
  while s7 and s7[-1] == 'a':
    s7 = s7[:-1]

  if (len(s7) == 0):
    resstr = ''
  else:
    resstr = s7[0]

    # Only add letters if they differ from the previous letter
    #
    for i in s7[1:]:
      if (i != resstr[-1]):
        resstr=resstr+i

  # Now compile final result string
  #
  if (first in 'aeiou'):
    resstr = first+resstr[1:]

  if (maxlen > 0):
    resstr = resstr[:maxlen]  # Return first maxlen characters

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('NYSIIS encoding for string: "%s": %s' % (s, resstr))

  return resstr

# =============================================================================

def dmetaphone(s, maxlen=4):
  """Compute the Double Metaphone code for a string.

  USAGE:
    code = dmetaphone(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.

  DESCRIPTION:
    Based on:
    - Lawrence Philips C++ code as published in C/C++ Users Journal (June 2000)
      and available at:
      http://www.cuj.com/articles/2000/0006/0006d/0006d.htm
    - Perl/C implementation
      http://www.cpan.org/modules/by-authors/id/MAURICE/
    See also:
    - http://aspell.sourceforge.net/metaphone/
    - http://www.nist.gov/dads/HTML/doubleMetaphone.html
  """

  if (not s):
    return ''

  primary = ''
  secondary = ''
  alternate = ''
  primary_len = 0
  secondary_len = 0

  # Sub routines  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  def isvowel(c):
    if (c in 'aeiouy'):
      return 1
    else:
      return 0

  def slavogermanic(str):
    if (str.find('w')>-1) or (str.find('k')>-1) or (str.find('cz')>-1) or \
       (str.find('witz')>-1):
      return 1
    else:
      return 0


  length = len(s)
  if len < 1:
    return ''
  last = length-1

  current = 0  # Current position in string
  workstr = s+'      '

  if (workstr[0:2] in ['gn','kn','pn','wr','ps']):
    current = current+1  # Skip first character

  if (workstr[0] == 'x'):  # Initial 'x' is pronounced like 's'
    primary = primary+'s'
    primary_len = primary_len+1
    secondary = secondary+'s'
    secondary_len = secondary_len+1
    current = current+1

  if (maxlen < 1):  # Calculate maximum length to check
    check_maxlen = length
  else:
    check_maxlen = maxlen

  while (primary_len < check_maxlen) or (secondary_len < check_maxlen):
    if (current >= length):
      break

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Main loop, analyse current character
    #
    c = workstr[current]

    if (c in 'aeiouy'):
      if (current == 0):  # All initial vowels map to 'a'
        primary = primary+'a'
        primary_len = primary_len+1
        secondary = secondary+'a'
        secondary_len = secondary_len+1
      current=current+1

    elif (c == 'b'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      primary = primary+'p'
      primary_len = primary_len+1
      secondary = secondary+'p'
      secondary_len = secondary_len+1
      if (workstr[current+1] == 'b'):
        current=current+2
      else:
        current=current+1

    # elif (s == 'c'):  # C
    #    primary = primary+'s'
    #    primary_len = primary_len+1
    #    secondary = secondary+'s'
    #    secondary_len = secondary_len+1
    #    current = current+1

    elif (c == 'c'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (current > 1) and (not isvowel(workstr[current-2])) and \
         workstr[current-1:current+2] == 'ach' and \
         (workstr[current+2] != 'i' and \
         (workstr[current+2] != 'e' or \
          workstr[current-2:current+4] in ['bacher','macher'])):
        primary = primary+'k'  # Various germanic special cases
        primary_len = primary_len+1
        secondary = secondary+'k'
        secondary_len = secondary_len+1
        current = current+2
      elif (current == 0) and (workstr[0:6] == 'caesar'):
        primary = primary+'s'
        primary_len = primary_len+1
        secondary = secondary+'s'
        secondary_len = secondary_len+1
        current = current+2
      elif (workstr[current:current+4] == 'chia'): # Italian 'chianti'
        primary = primary+'k'
        primary_len = primary_len+1
        secondary = secondary+'k'
        secondary_len = secondary_len+1
        current = current+2
      elif (workstr[current:current+2] == 'ch'):
        if (current > 0) and (workstr[current:current+4] == 'chae'):
          primary = primary+'k'  # Find 'michael'
          primary_len = primary_len+1
          secondary = secondary+'x'
          secondary_len = secondary_len+1
          current = current+2
        elif (current == 0) and \
           (workstr[current+1:current+6] in ['harac','haris'] or \
            workstr[current+1:current+4] in \
              ['hor','hym','hia','hem']) and \
           workstr[0:6] != 'chore':
          primary = primary+'k'  # Greek roots, eg. 'chemistry'
          primary_len = primary_len+1
          secondary = secondary+'k'
          secondary_len = secondary_len+1
          current = current+2
        elif (workstr[0:4] in ['van ','von '] or \
              workstr[0:3] == 'sch') or \
            workstr[current-2:current+4] in \
              ['orches','archit','orchid'] or \
            workstr[current+2] in ['t','s'] or \
            ((workstr[current-1] in ['a','o','u','e'] or \
              current==0) and \
            workstr[current+2] in \
              ['l','r','n','m','b','h','f','v','w',' ']):
          primary = primary+'k'
          primary_len = primary_len+1
          secondary = secondary+'k'
          secondary_len = secondary_len+1
          current = current+2
        else:
          if (current > 0):
            if (workstr[0:2] == 'mc'):
              primary = primary+'k'
              primary_len = primary_len+1
              secondary = secondary+'k'
              secondary_len = secondary_len+1
              current = current+2
            else:
              primary = primary+'x'
              primary_len = primary_len+1
              secondary = secondary+'k'
              secondary_len = secondary_len+1
              current = current+2
          else:
            primary = primary+'x'
            primary_len = primary_len+1
            secondary = secondary+'x'
            secondary_len = secondary_len+1
            current=current+2
      elif (workstr[current:current+2] == 'cz') and \
         (workstr[current-2:current+2] != 'wicz'):
        primary = primary+'s'
        primary_len = primary_len+1
        secondary = secondary+'x'
        secondary_len = secondary_len+1
        current=current+2
      elif (workstr[current+1:current+4] == 'cia'):
        primary = primary+'x'
        primary_len = primary_len+1
        secondary = secondary+'x'
        secondary_len = secondary_len+1
        current=current+3
      elif (workstr[current:current+2] == 'cc') and \
           not (current==1 and workstr[0] == 'm'):
        if (workstr[current+2] in ['i','e','h']) and \
           (workstr[current+2:current+4] != 'hu'):
          if (current == 1 and workstr[0] == 'a') or \
             (workstr[current-1:current+4] in ['uccee','ucces']):
            primary = primary+'ks'
            primary_len = primary_len+2
            secondary = secondary+'ks'
            secondary_len = secondary_len+2
            current=current+3
          else:
            primary = primary+'x'
            primary_len = primary_len+1
            secondary = secondary+'x'
            secondary_len = secondary_len+1
            current=current+3
        else:  # Pierce's rule
          primary = primary+'k'
          primary_len = primary_len+1
          secondary = secondary+'k'
          secondary_len = secondary_len+1
          current=current+2
      elif (workstr[current:current+2] in ['ck','cg','cq']):
        primary = primary+'k'
        primary_len = primary_len+1
        secondary = secondary+'k'
        secondary_len = secondary_len+1
        current=current+2
      elif (workstr[current:current+2] in ['ci','ce','cy']):
        if (workstr[current:current+3] in ['cio','cie','cia']):
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'x'
          secondary_len = secondary_len+1
          current=current+2
        else:
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'s'
          secondary_len = secondary_len+1
          current=current+2
      else:
        primary = primary+'k'
        primary_len = primary_len+1
        secondary = secondary+'k'
        secondary_len = secondary_len+1
        if (workstr[current+1:current+3] in [' c',' q',' g']):
          current=current+3
        else:
          if (workstr[current+1] in ['c','k','q']) and \
             (workstr[current+1:current+3] not in ['ce','ci']):
            current=current+2
          else:
            current=current+1

    elif (c == 'd'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current:current+2] == 'dg'):
        if (workstr[current+2] in ['i','e','y']):  # Eg. 'edge'
          primary = primary+'j'
          primary_len = primary_len+1
          secondary = secondary+'j'
          secondary_len = secondary_len+1
          current=current+3
        else:  # Eg. 'edgar'
          primary = primary+'tk'
          primary_len = primary_len+2
          secondary = secondary+'tk'
          secondary_len = secondary_len+2
          current=current+2
      elif (workstr[current:current+2] in ['dt','dd']):
        primary = primary+'t'
        primary_len = primary_len+1
        secondary = secondary+'t'
        secondary_len = secondary_len+1
        current=current+2
      else:
        primary = primary+'t'
        primary_len = primary_len+1
        secondary = secondary+'t'
        secondary_len = secondary_len+1
        current=current+1

    elif (c == 'f'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'f'):
        current=current+2
      else:
        current=current+1
      primary = primary+'f'
      primary_len = primary_len+1
      secondary = secondary+'f'
      secondary_len = secondary_len+1

    elif (c == 'g'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'h'):
        if (current > 0 and not isvowel(workstr[current-1])):
          primary = primary+'k'
          primary_len = primary_len+1
          secondary = secondary+'k'
          secondary_len = secondary_len+1
          current=current+2
        elif (current==0):
          if (workstr[current+2] == 'i'): # Eg. ghislane, ghiradelli
            primary = primary+'j'
            primary_len = primary_len+1
            secondary = secondary+'j'
            secondary_len = secondary_len+1
            current=current+2
          else:
            primary = primary+'k'
            primary_len = primary_len+1
            secondary = secondary+'k'
            secondary_len = secondary_len+1
            current=current+2
        elif (current>1 and workstr[current-2] in ['b','h','d']) or \
             (current>2 and workstr[current-3] in ['b','h','d']) or \
             (current>3 and workstr[current-4] in ['b','h']):
          current=current+2
        else:
          if (current > 2) and (workstr[current-1] == 'u') and \
             (workstr[current-3] in ['c','g','l','r','t']):
            primary = primary+'f'
            primary_len = primary_len+1
            secondary = secondary+'f'
            secondary_len = secondary_len+1
            current=current+2
          else:
            if (current > 0) and (workstr[current-1] != 'i'):
              primary = primary+'k'
              primary_len = primary_len+1
              secondary = secondary+'k'
              secondary_len = secondary_len+1
              current=current+2
            else:
              current=current+2
      elif (workstr[current+1] == 'n'):
        if (current==1) and (isvowel(workstr[0])) and \
           (not slavogermanic(workstr)):
          primary = primary+'kn'
          primary_len = primary_len+2
          secondary = secondary+'n'
          secondary_len = secondary_len+1
          current=current+2
        else:
          if (workstr[current+2:current+4] != 'ey') and \
             (workstr[current+1] != 'y') and \
             (not slavogermanic(workstr)):
            primary = primary+'n'
            primary_len = primary_len+1
            secondary = secondary+'kn'
            secondary_len = secondary_len+2
            current=current+2
          else:
            primary = primary+'kn'
            primary_len = primary_len+2
            secondary = secondary+'kn'
            secondary_len = secondary_len+2
            current=current+2
      elif (workstr[current+1:current+3] == 'li') and \
           (not slavogermanic(workstr)):
        primary = primary+'kl'
        primary_len = primary_len+2
        secondary = secondary+'l'
        secondary_len = secondary_len+1
        current=current+2
      elif (current==0) and ((workstr[current+1] == 'y') or \
           (workstr[current+1:current+3] in \
           ['es','ep','eb','el','ey','ib','il','in','ie','ei','er'])):
        primary = primary+'k'
        primary_len = primary_len+1
        secondary = secondary+'j'
        secondary_len = secondary_len+1
        current=current+2
      elif (workstr[current+1:current+3] == 'er' or \
           workstr[current+1] == 'y') and \
           workstr[0:6] not in ['danger','ranger','manger'] and \
           workstr[current-1] not in ['e','i'] and \
           workstr[current-1:current+2] not in ['rgy','ogy']:
        primary = primary+'k'
        primary_len = primary_len+1
        secondary = secondary+'j'
        secondary_len = secondary_len+1
        current=current+2
      elif (workstr[current+1] in ['e','i','y']) or \
           (workstr[current-1:current+3] in ['aggi','oggi']):
        if (workstr[0:4] in ['van ','von ']) or \
           (workstr[0:3] == 'sch') or \
           (workstr[current+1:current+3] == 'et'):
          primary = primary+'k'
          primary_len = primary_len+1
          secondary = secondary+'k'
          secondary_len = secondary_len+1
          current=current+2
        else:
          if (workstr[current+1:current+5] == 'ier '):
            primary = primary+'j'
            primary_len = primary_len+1
            secondary = secondary+'j'
            secondary_len = secondary_len+1
            current=current+2
          else:
            primary = primary+'j'
            primary_len = primary_len+1
            secondary = secondary+'k'
            secondary_len = secondary_len+1
            current=current+2
      else:
        if (workstr[current+1] == 'g'):
          current=current+2
        else:
          current=current+1
        primary = primary+'k'
        primary_len = primary_len+1
        secondary = secondary+'k'
        secondary_len = secondary_len+1

    elif (c =='h'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (current == 0 or isvowel(workstr[current-1])) and \
         isvowel(workstr[current+1]):
        primary = primary+'h'
        primary_len = primary_len+1
        secondary = secondary+'h'
        secondary_len = secondary_len+1
        current=current+2
      else:
        current=current+1

    elif (c =='j'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current:current+4] == 'jose') or \
         (workstr[0:4] == 'san '):
        if (current == 0 and workstr[4] == ' ') or \
           (workstr[0:4] == 'san '):
          primary = primary+'h'
          primary_len = primary_len+1
          secondary = secondary+'h'
          secondary_len = secondary_len+1
          current=current+1
        else:
          primary = primary+'j'
          primary_len = primary_len+1
          secondary = secondary+'h'
          secondary_len = secondary_len+1
          current=current+1
      elif (current==0) and (workstr[0:4] != 'jose'):
        primary = primary+'j'
        primary_len = primary_len+1
        secondary = secondary+'a'
        secondary_len = secondary_len+1
        if (workstr[current+1] == 'j'):
          current=current+2
        else:
          current=current+1
      else:
        if (isvowel(workstr[current-1])) and \
           (not slavogermanic(workstr)) and \
           (workstr[current+1] in ['a','o']):
          primary = primary+'j'
          primary_len = primary_len+1
          secondary = secondary+'h'
          secondary_len = secondary_len+1
        else:
          if (current == last):
            primary = primary+'j'
            primary_len = primary_len+1
            #secondary = secondary+''
            #secondary_len = secondary_len+0
          else:
            if (workstr[current+1] not in \
               ['l','t','k','s','n','m','b','z']) and \
               (workstr[current-1] not in ['s','k','l']):
              primary = primary+'j'
              primary_len = primary_len+1
              secondary = secondary+'j'
              secondary_len = secondary_len+1
        if (workstr[current+1] == 'j'):
          current=current+2
        else:
          current=current+1

    elif (c =='k'):  #  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'k'):
        current=current+2
      else:
        current=current+1
      primary = primary+'k'
      primary_len = primary_len+1
      secondary = secondary+'k'
      secondary_len = secondary_len+1

    elif (c == 'l'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'l'):
        if (current == (length-3)) and \
           (workstr[current-1:current+3] in ['illo','illa','alle']) or \
           ((workstr[last-1:last+1] in ['as','os']  or
           workstr[last] in ['a','o']) and \
           workstr[current-1:current+3] == 'alle'):
          primary = primary+'l'
          primary_len = primary_len+1
          #secondary = secondary+''
          #secondary_len = secondary_len+0
          current=current+2
        else:
          primary = primary+'l'
          primary_len = primary_len+1
          secondary = secondary+'l'
          secondary_len = secondary_len+1
          current=current+2
      else:
        primary = primary+'l'
        primary_len = primary_len+1
        secondary = secondary+'l'
        secondary_len = secondary_len+1
        current=current+1

    elif (c == 'm'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current-1:current+2] == 'umb' and \
         ((current+1) == last or \
          workstr[current+2:current+4] == 'er')) or \
         workstr[current+1] == 'm':
        current=current+2
      else:
        current=current+1
      primary = primary+'m'
      primary_len = primary_len+1
      secondary = secondary+'m'
      secondary_len = secondary_len+1

    elif (c == 'n'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'n'):
        current=current+2
      else:
        current=current+1
      primary = primary+'n'
      primary_len = primary_len+1
      secondary = secondary+'n'
      secondary_len = secondary_len+1

    elif (c == 'p'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'h'):
        primary = primary+'f'
        primary_len = primary_len+1
        secondary = secondary+'f'
        secondary_len = secondary_len+1
        current=current+2
      elif (workstr[current+1] in ['p','b']):
        primary = primary+'p'
        primary_len = primary_len+1
        secondary = secondary+'p'
        secondary_len = secondary_len+1
        current=current+2
      else:
        primary = primary+'p'
        primary_len = primary_len+1
        secondary = secondary+'p'
        secondary_len = secondary_len+1
        current=current+1

    elif (c == 'q'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'q'):
        current=current+2
      else:
        current=current+1
      primary = primary+'k'
      primary_len = primary_len+1
      secondary = secondary+'k'
      secondary_len = secondary_len+1

    elif (c == 'r'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (current==last) and (not slavogermanic(workstr)) and \
         (workstr[current-2:current] == 'ie') and \
         (workstr[current-4:current-2] not in ['me','ma']):
        # primary = primary+''
        # primary_len = primary_len+0
        secondary = secondary+'r'
        secondary_len = secondary_len+1
      else:
        primary = primary+'r'
        primary_len = primary_len+1
        secondary = secondary+'r'
        secondary_len = secondary_len+1
      if (workstr[current+1] == 'r'):
        current=current+2
      else:
        current=current+1

    elif (c == 's'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current-1:current+2] in ['isl','ysl']):
        current=current+1
      elif (current==0) and (workstr[0:5] == 'sugar'):
        primary = primary+'x'
        primary_len = primary_len+1
        secondary = secondary+'s'
        secondary_len = secondary_len+1
        current=current+1
      elif (workstr[current:current+2] == 'sh'):
        if (workstr[current+1:current+5] in \
           ['heim','hoek','holm','holz']):
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'s'
          secondary_len = secondary_len+1
          current=current+2
        else:
          primary = primary+'x'
          primary_len = primary_len+1
          secondary = secondary+'x'
          secondary_len = secondary_len+1
          current=current+2
      elif (workstr[current:current+3] in ['sio','sia']) or \
           (workstr[current:current+4] == 'sian'):
        if (not slavogermanic(workstr)):
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'x'
          secondary_len = secondary_len+1
          current=current+3
        else:
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'s'
          secondary_len = secondary_len+1
          current=current+3
      elif ((current==0) and (workstr[1] in ['m','n','l','w'])) or \
           (workstr[current+1] == 'z'):
        primary = primary+'s'
        primary_len = primary_len+1
        secondary = secondary+'x'
        secondary_len = secondary_len+1
        if (workstr[current+1] == 'z'):
          current=current+2
        else:
          current=current+1
      elif (workstr[current:current+2] == 'sc'):
        if (workstr[current+2] == 'h'):
          if (workstr[current+3:current+5] in \
             ['oo','er','en','uy','ed','em']):
            if (workstr[current+3:current+5] in ['er','en']):
              primary = primary+'x'
              primary_len = primary_len+1
              secondary = secondary+'sk'
              secondary_len = secondary_len+2
              current=current+3
            else:
              primary = primary+'sk'
              primary_len = primary_len+2
              secondary = secondary+'sk'
              secondary_len = secondary_len+2
              current=current+3
          else:
            if (current==0) and (not isvowel(workstr[3])) and \
               (workstr[3] != 'w'):
              primary = primary+'x'
              primary_len = primary_len+1
              secondary = secondary+'s'
              secondary_len = secondary_len+1
              current=current+3
            else:
              primary = primary+'x'
              primary_len = primary_len+1
              secondary = secondary+'x'
              secondary_len = secondary_len+1
              current=current+3
        elif (workstr[current+2] in ['i','e','y']):
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'s'
          secondary_len = secondary_len+1
          current=current+3
        else:
          primary = primary+'sk'
          primary_len = primary_len+2
          secondary = secondary+'sk'
          secondary_len = secondary_len+2
          current=current+3
      elif (current==last) and \
           (workstr[current-2:current] in ['ai','oi']):
        # primary = primary+''
        # primary_len = primary_len+0
        secondary = secondary+'s'
        secondary_len = secondary_len+1
        if (workstr[current+1] in ['s','z']):
          current=current+2
        else:
          current=current+1
      else:
        primary = primary+'s'
        primary_len = primary_len+1
        secondary = secondary+'s'
        secondary_len = secondary_len+1
        if (workstr[current+1] in ['s','z']):
          current=current+2
        else:
          current=current+1

    elif (c == 't'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current:current+4] == 'tion'):
        primary = primary+'x'
        primary_len = primary_len+1
        secondary = secondary+'x'
        secondary_len = secondary_len+1
        current=current+3
      elif (workstr[current:current+3] in ['tia','tch']):
        primary = primary+'x'
        primary_len = primary_len+1
        secondary = secondary+'x'
        secondary_len = secondary_len+1
        current=current+3
      elif (workstr[current:current+2] == 'th') or \
           (workstr[current:current+3] == 'tth'):
        if (workstr[current+2:current+4] in ['om','am']) or \
           (workstr[0:4] in ['von ','van ']) or (workstr[0:3] == 'sch'):
          primary = primary+'t'
          primary_len = primary_len+1
          secondary = secondary+'t'
          secondary_len = secondary_len+1
          current=current+2
        else:
          primary = primary+'0'
          primary_len = primary_len+1
          secondary = secondary+'t'
          secondary_len = secondary_len+1
          current=current+2
      elif (workstr[current+1] in ['t','d']):
        primary = primary+'t'
        primary_len = primary_len+1
        secondary = secondary+'t'
        secondary_len = secondary_len+1
        current=current+2
      else:
        primary = primary+'t'
        primary_len = primary_len+1
        secondary = secondary+'t'
        secondary_len = secondary_len+1
        current=current+1

    elif (c == 'v'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'v'):
        current=current+2
      else:
        current=current+1
      primary = primary+'f'
      primary_len = primary_len+1
      secondary = secondary+'f'
      secondary_len = secondary_len+1

    elif (c == 'w'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current:current+2] == 'wr'):
        primary = primary+'r'
        primary_len = primary_len+1
        secondary = secondary+'r'
        secondary_len = secondary_len+1
        current=current+2
      else:
        if (current==0) and (isvowel(workstr[1]) or \
           workstr[0:2] == 'wh'):
          if (isvowel(workstr[current+1])):
            primary = primary+'a'
            primary_len = primary_len+1
            secondary = secondary+'f'
            secondary_len = secondary_len+1
            #current=current+1
          else:
            primary = primary+'a'
            primary_len = primary_len+1
            secondary = secondary+'a'
            secondary_len = secondary_len+1
            #current=current+1
        if (current==last and isvowel(workstr[current-1])) or \
           workstr[current-1:current+4] in \
           ['ewski','ewsky','owski','owsky'] or \
           workstr[0:3] == 'sch':
          # primary = primary+''
          # primary_len = primary_len+0
          secondary = secondary+'f'
          secondary_len = secondary_len+1
          current=current+1
        elif (workstr[current:current+4] in ['witz','wicz']):
          primary = primary+'ts'
          primary_len = primary_len+2
          secondary = secondary+'fx'
          secondary_len = secondary_len+2
          current=current+4
        else:
          current=current+1

    elif (c == 'x'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if not (current==last and \
         (workstr[current-3:current] in ['iau','eau'] or \
          workstr[current-2:current] in ['au','ou'])):
        primary = primary+'ks'
        primary_len = primary_len+2
        secondary = secondary+'ks'
        secondary_len = secondary_len+2
      if (workstr[current+1] in ['c','x']):
        current=current+2
      else:
        current=current+1

    elif (c == 'z'):  # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      if (workstr[current+1] == 'h'):
        primary = primary+'j'
        primary_len = primary_len+1
        secondary = secondary+'j'
        secondary_len = secondary_len+1
        current=current+2
      else:
        if (workstr[current+1:current+3] in ['zo','zi','za']) or \
           (slavogermanic(workstr) and \
           (current > 0 and workstr[current-1] != 't')):
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'ts'
          secondary_len = secondary_len+2
          if (workstr[current+1] == 'z'):
            current=current+2
          else:
            current=current+1
        else:
          primary = primary+'s'
          primary_len = primary_len+1
          secondary = secondary+'s'
          secondary_len = secondary_len+1
          if (workstr[current+1] == 'z'):
            current=current+2
          else:
            current=current+1

    else:   # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      current=current+1

    # End main loop

  if (primary == secondary):
    # If both codes are the same set the second's length to 0 so it's not used
    secondary_len = 0

# else:
#   print 'Two D-metaphone codes for "%s": "%s" / "%s"' % (s,primary,secondary)

  # if (secondary_len > 0):
  #   return [primary[:maxlen], secondary[:maxlen]]
  # else:
  #   return [primary[:maxlen]]

  if (maxlen > 0):
    resstr = primary[:maxlen]  # Only return primary encoding
  else:
    resstr = primary

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Double Metaphone encoding for string: "%s": prim: %s, ' % \
                (s, primary) + '(sec: %s)' % (secondary))

  return resstr

# =============================================================================

def fuzzy_soundex(s, maxlen=4):
  """Compute the fuzzy soundex code for a string.

  USAGE:
    code = fuzzy_soundex(s, maxlen)

  ARGUMENTS:
    s        A string containing a name.
    maxlen   Maximal length of the returned code. If a code is longer than
             'maxlen' it is truncated. Default value is 4.
             If 'maxlen' is negative the soundex code will not be padded with
             '0' to 'maxlen' characters.

  DESCRIPTION:
    Based on ideas described in:

      "Improving Precision and Recall for Soundex Retrieval"
      by David Holmes and M. Catherine McCabe, 2002.

    This method does q-gram based substitution of sub-strings before encoding
    the input string.
  """

  if (not s):
    if (maxlen > 0):
      return maxlen*'0'  # Or 'z000' for compatibility with other
                         # implementations
    else:
      return '0'

  # Translation table and characters that will not be used for soundex  - - - -
  #
  transtable = string.maketrans('abcdefghijklmnopqrstuvwxyz', \
                                '01930170077455017693010709')
  # Soundex:                    '01230120022455012623010202')
  # Differences:                   *   *  **     * *    * *

  qgram_prefix_sub_dict = {'cs':'ss', 'cz':'ss', 'ts':'ss', 'tz':'ss',
                           'gn':'nn', 'hr':'rr', 'wr':'rr', 'hw':'ww',
                           'kn':'nn', 'ng':'nn'}

  qgram_sub_dict = {'chl':'kl',  'chr':'kr',  'mac':'mk', 'nst':'nss',
                    'sch':'sss', 'tio':'sio', 'tia':'sio', 'tch':'chh',
                    'ca':'ka', 'cc':'kk', 'ck':'kk', 'ce':'se', 'cl':'kl',
                    'cr':'kr', 'ci':'si', 'co':'ko', 'cu':'ku', 'cy':'sy',
                    'dg':'gg', 'gh':'hh', 'mc':'mk', 'pf':'ff', 'ph':'ff'}

  qgram_suffix_sub_dict = {'ch':'kk', 'nt':'tt', 'rt':'rr', 'rdt':'rr'}

  for subs in qgram_sub_dict:
    assert subs not in qgram_prefix_sub_dict
    assert subs not in qgram_suffix_sub_dict
    qgram_prefix_sub_dict[subs] = qgram_sub_dict[subs]
    qgram_suffix_sub_dict[subs] = qgram_sub_dict[subs]

  tmp_str = s  # Work on a copy of input string
  qgram_list = []

  # First process prefixes only
  #
  prefix3 = tmp_str[:3]  # First three characters
  prefix2 = tmp_str[:2]  # First two characters

  if (prefix3 in qgram_prefix_sub_dict):
    qgram_list.append(qgram_prefix_sub_dict[prefix3])
    tmp_str = tmp_str[3:]  # Removed processed prefix

  elif (prefix2 in qgram_prefix_sub_dict):
    qgram_list.append(qgram_prefix_sub_dict[prefix2])
    tmp_str = tmp_str[2:]  # Removed processed prefix

  # Next process suffixes only
  #
  suffix3 = tmp_str[-3:]  # Last three characters
  suffix2 = tmp_str[-2:]  # Last two characters

  if (suffix3 in qgram_suffix_sub_dict):
    suffix_qgram = qgram_suffix_sub_dict[suffix3]
    tmp_str = tmp_str[:-3]  # Removed processed suffix
  elif (suffix2 in qgram_suffix_sub_dict):
    suffix_qgram = qgram_suffix_sub_dict[suffix2]
    tmp_str = tmp_str[2:]  # Removed processed suffix
  else:
    suffix_qgram = []

  # Now process rest of string
  #
  while (tmp_str != ''):
    found_qgram = False

    if (len(tmp_str) >= 3):
      tmp_trigram = tmp_str[:3]
      if (tmp_trigram in qgram_sub_dict):
        qgram_list.append(qgram_sub_dict[tmp_trigram])
        tmp_str = tmp_str[3:]
        found_qgram = True

    if (found_qgram == False) and (len(tmp_str) >= 2):
      tmp_bigram = s[:2]
      if (tmp_bigram in qgram_sub_dict):
        qgram_list.append(qgram_sub_dict[tmp_bigram])
        tmp_str = tmp_str[2:]
        found_qgram = True

    if (found_qgram == False):
      qgram_list.append(tmp_str[0])
      tmp_str = tmp_str[1:]

  qgram_list += suffix_qgram

  s2 = ''.join(qgram_list)

  s3 = string.translate(s2[1:],transtable, ' ')  # Delete spaces

  s4 = s2[0]

  # Only add numbers if they are not the same as the previous number
  for c in s3:
    if (c != s4[-1]):
      s4 += c

  # Remove all '0'
  s5 = s4.replace('0', '')

  # Fill up with '0' to maxlen length
  #
  s5 = s5 + maxlen*'0'

  if (maxlen > 0):
    resstr = s5[:maxlen]  # Return first maxlen characters
  else:
    resstr = s5

  # A log message - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  #
  logging.debug('Fuxxy Soundex encoding for string: "%s": %s' % (s, resstr))

  return resstr

# =============================================================================

def get_substring(s, start_index, end_index):
  """Simple function to extract and return a substring from the given input
     string.
  """

  assert start_index <= end_index

  return s[start_index:end_index]

# =============================================================================

def freq_vector(s, encode=None):
  """Count occurrence of characters in the given string and put them into a
     frequency vector.

  USAGE:
    code = freq_vector(s, encode)

  ARGUMENTS:
    s        A string containing a name.
    encode   An encodingthat can be set to None (default), 'phonix', 'soundex',
             or 'mod_soundex'. For the last three cases different encodings
             will be applied before the frequency vector is being built. Note
             that the resulting vectors will be of different lengths depending
             upon the encoding method used.

  DESCRIPTION:
    Note that only letters will be encoded, all other characters in the input
    strin will not be considered.

    The function returns a list (vector) with frequency counts. For example
    with encoding 'soundex' the string 'peter' will first be encoded as 1 (p),
    0 (e), 3 (t), 0 (e), and 6 (r), then the frequency vector (which will be
    returned) will be: [2,1,0,1,0,0,1,0].

    Another example, without encoding function set: 'christine' will return a
    vector: [0,0,1,0,1,0,0,1,2,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0].
  """

  if (s == ''):
    return ''

  s = s.lower()

  if (encode == 'phonix'):
    trans_dict = {'a':0,'b':1,'c':2,'d':3,'e':0,'f':7,'g':2,'h':0,'i':0,'j':2,
                  'k':2,'l':4,'m':5,'n':5,'o':0,'p':1,'q':2,'r':6,'s':8,'t':3,
                  'u':0,'v':7,'w':0,'x':8,'y':0,'z':8}
    f_vec = [0,0,0,0,0,0,0,0,0]

  elif (encode == 'soundex'):
    trans_dict = {'a':0,'b':1,'c':2,'d':3,'e':0,'f':1,'g':2,'h':0,'i':0,'j':2,
                  'k':2,'l':4,'m':5,'n':5,'o':0,'p':1,'q':2,'r':6,'s':2,'t':3,
                  'u':0,'v':1,'w':0,'x':2,'y':0,'z':2}
    f_vec = [0,0,0,0,0,0,0]

  elif (encode == 'mod_soundex'):
    trans_dict = {'a':0,'b':1,'c':3,'d':6,'e':0,'f':2,'g':4,'h':0,'i':0,'j':4,
                  'k':3,'l':7,'m':8,'n':8,'o':0,'p':1,'q':5,'r':9,'s':3,'t':6,
                  'u':0,'v':2,'w':0,'x':5,'y':0,'z':5}
    f_vec = [0,0,0,0,0,0,0,0,0,0]

  elif (encode == None):
    trans_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,
                  'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,
                  's':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25}
    f_vec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

  else:
    logging.exception('Illegal encoding method: %s' % (encode))
    raise Exception

  skip_count = 0  # Charcters skipped because they are not letters

  for char in s:
    if char in trans_dict:
      f_vec[trans_dict[char]] = f_vec[trans_dict[char]]+1
    else:
      skip_count += 1

  assert (sum(f_vec)+skip_count) == len(s)

  ## Now convert into a character - 0=a, 1=b, 3=c etc.
  ##
  #resstr = ''
  #
  #for c in f_vec:
  #  resstr += chr(97+c)
  #
  #assert len(f_vec) == len(resstr), (f_vec, resstr)
  #
  #return resstr

  return f_vec

# =============================================================================
# Do some tests if called from command line
#

if (__name__ == '__main__'):
  #print 'Febrl module "encode.py"'
  #print '------------------------'
  #print

  #print 'Original names:'
  #print '            Name     Phonex   Soundex  ModSoundex      NYSIIS  ',
  #print '  D-Metaphone   FuzzySoundex   Phonix'
  #print '---------------------------------------------------------------'+ \
       # '--------------------------------------'

  namelist = ['peter','christen','ole','nielsen','markus','hegland',\
              'stephen','steve','roberts','tim','churches','xiong',\
              'ng','miller','millar','foccachio','van de hooch', \
              'xiao ching','asawakun','prapasri','von der felde','vest',
              'west','oioi','ohio','oihcca', 'nielsen', 'kim', 'lim', \
              'computer','record','linkage','probabilistic','gail', 'gayle',
              'christine','christina','kristina','steffi']

  for n in namelist:
    soundex_my =      soundex(n)
    soundex_mod_my =  mod_soundex(n)
    phonex_my =       phonex(n)
    nysiis_my =       nysiis(n)
    dmeta_my =        dmetaphone(n)
    fuzzysoundex_my = fuzzy_soundex(n)
    phonix_my =       phonix(n)

    #print '%16s %10s %9s %11s %11s %15s %14s %8s' % (n, phonex_my, \
          #soundex_my, soundex_mod_my, nysiis_my, dmeta_my, fuzzysoundex_my, \
          #phonix_my)

  #print
  #print 'Reversed names:'
  #print '            Name     Phonex   Soundex  ModSoundex      NYSIIS  ',
  #print '  D-Metaphone   FuzzySoundex   Phonix'
  #print '---------------------------------------------------------------'+ \
        #'--------------------------------------'

  for n in namelist:
    rn = list(n)
    rn.reverse()
    rn = ''.join(rn)
    soundex_my =      soundex(rn)
    soundex_mod_my =  mod_soundex(rn)
    phonex_my =       phonex(rn)
    nysiis_my =       nysiis(rn)
    dmeta_my =        dmetaphone(rn)
    fuzzysoundex_my = fuzzy_soundex(rn)
    phonix_my =       phonix(rn)

    #print '%16s %10s %9s %11s %11s %15s %14s %8s' % (n, phonex_my, \
          #soundex_my, soundex_mod_my, nysiis_my, dmeta_my, fuzzysoundex_my, \
          #phonix_my)

# =============================================================================
if __name__=='__main__':
    print('why did you do this?')