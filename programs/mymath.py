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
# The Original Software is: "mymath.py"
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
"""Module mymath.py - Various mathematical routines.

   See doc strings of individual functions for detailed documentation.
"""

# =============================================================================
# Imports go here

import logging
import random

import math


# =============================================================================

def distL1(vec1, vec2):
    """L1 distance measure, also called Manhattan distance.

       The distance between two points measured along axes at right angles.

       See also:
         http://www.nist.gov/dads/HTML/lmdistance.html
         http://en.wikipedia.org/wiki/Distance
    """

    #  assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    L1_dist = 0.0

    for i in range(vec_len):
        L1_dist += abs(float(vec1[i]) - float(vec2[i]))

    return L1_dist


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def distL2(vec1, vec2):
    """L2 distance measure, also known as the Euclidean distance.

       See also:
         http://www.nist.gov/dads/HTML/lmdistance.html
         http://en.wikipedia.org/wiki/Distance
    """

    #  assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    L2_dist = 0.0

    for i in range(vec_len):
        x = float(vec1[i]) - float(vec2[i])
        L2_dist += x * x

    return math.sqrt(L2_dist)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def distLInf(vec1, vec2):
    """L-Infinity distance measure.

       See also:
         http://www.nist.gov/dads/HTML/lmdistance.html
         http://en.wikipedia.org/wiki/Distance
    """

    #  assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    Linf_dist = -1.0

    for i in range(vec_len):
        x = abs(float(vec1[i]) - float(vec2[i]))
        Linf_dist = max(x, Linf_dist)

    return Linf_dist


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def distCanberra(vec1, vec2):
    """Canberra distance measure.

       See also:
       http://people.revoledu.com/kardi/tutorial/Similarity/CanberraDistance.html
    """

    #  assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    cbr_dist = 0.0

    for i in range(vec_len):
        x = abs(float(vec1[i]) - float(vec2[i]))
        y = abs(float(vec1[i])) + abs(float(vec2[i]))
        if (y > 0.0):
            cbr_dist += x / y

    return cbr_dist


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def distCosine(vec1, vec2):
    """Cosine distance measure.

       Note: This function assumes that all vector elements are non-negative.

       See also:
         http://en.wikipedia.org/wiki/Vector_space_model
    """

    assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    vec1sum = 0.0
    vec2sum = 0.0
    vec12sum = 0.0

    for i in range(vec_len):
        vec1sum += vec1[i] * vec1[i]
        vec2sum += vec2[i] * vec2[i]
        vec12sum += vec1[i] * vec2[i]

    if (vec1sum * vec2sum == 0.0):
        cos_dist = 1.0  # At least one vector is all zeros

    else:
        vec1sum = math.sqrt(vec1sum)
        vec2sum = math.sqrt(vec2sum)

        cos_sim = vec12sum / (vec1sum * vec2sum)

        # Due to rounding errors the similarity can be slightly larger than 1.0
        #
        cos_sim = min(cos_sim, 1.0)

        assert (cos_sim >= 0.0) and (cos_sim <= 1.0), (cos_sim, vec1, vec2)

        cos_dist = 1.0 - cos_sim

    return cos_dist


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## TODO, PC Jan 2008 ***********

def distMahalanobis(vec1, vec2):
    """Mahalanobis distance measure.

       See also:
         http://en.wikipedia.org/wiki/Mahalanobis_distance
    """

    assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    mal_dist = 0.0

    return mal_dist


# =============================================================================

def mean(x):
    """Compute the mean (average)  of a list of numbers.
    """

    if (len(x) == 1):  # Only one element in list
        return float(x[0])

    elif (len(x) == 0):  # Empty list
        logging.info('Empty list given: %s' % (str(x)))
        return None

    else:  # Calculate average
        sum = 0.0
        for i in x:
            sum += i

        res = sum / float(len(x))

        return res


# =============================================================================

def stddev(x):
    """Compute the standard deviation of a list of numbers.
    """

    if (len(x) == 1):  # Only one element in list
        return 0.0

    elif (len(x) == 0):  # Empty list
        logging.info('Empty list given: %s' % (str(x)))
        return None

    else:
        sum = 0.0
        for i in x:
            sum += i

        avrg = sum / float(len(x))

        sum = 0.0
        for i in x:
            sum = sum + (i - avrg) * (i - avrg)

        res = math.sqrt(sum / float(len(x)))

        return res


# =============================================================================

def log2(x):
    """Compute binary logarithm (log2) for a floating-point number.

    USAGE:
      y = log2(x)

    ARGUMENT:
      x  An positive integer or floating-point number

    DESCRIPTION:
      This routine computes and returns the binary logarithm of a positive
      number.
    """

    return math.log(x) / 0.69314718055994529  # = math.log(2.0)


# =============================================================================

# A function to create permutations of a list (from ASPN Python cookbook, see:
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66463)

def getPermutations(a):
    if len(a) == 1:
        yield a
    else:
        for i in range(len(a)):
            this = a[i]
            rest = a[:i] + a[i + 1:]
            for p in getPermutations(rest):
                yield [this] + p


def permute(alist):
    reslist = []
    for l in getPermutations(alist):
        reslist.append(' '.join(l))

    return reslist


# =============================================================================

def perm_tag_sequence(in_tag_seq):
    """Create all permuations of a tag sequence.

    USAGE:
      seq_list = perm_tag_sequence(in_tag_seq)

    ARGUMENT:
      in_tag_seq  Input sequence (list) with tags

    DESCRIPTION:
      This routine computes all permutations of the given input sequence. More
      than one permutation is created if at least one element in the input
      sequence contains more than one tag.

      Returns a list containing tag sequences (lists).
    """

    if (not isinstance(in_tag_seq, list)):
        logging.exception('Input tag sequence is not a list: %s' % \
                          (str(in_tag_seq)))
        raise Exception

    list_len = len(in_tag_seq)
    out_tag_seq = [[]]  # List of output tag sequences, start with one empty list

    for elem in in_tag_seq:
        if ('/' in elem):  # Element contains more than one tag, covert into a list
            elem = elem.split('/')

        tmp_tag_seq = []

        if (isinstance(elem, str)):  # Append a simple string
            for t in out_tag_seq:
                tmp_tag_seq.append(t + [elem])  # Append string to all tag sequences

        else:  # Process a list (that contains more than one tags)
            for tag in elem:  # Add each tag in the list to the temporary tag list
                for t in out_tag_seq:
                    tmp_tag_seq.append(t + [tag])  # Append string to all tag sequences

        out_tag_seq = tmp_tag_seq

    # A log message for high volume log output (level 3) - - - - - - - - - - - -
    #
    logging.debug('Input tag sequence: %s' % (str(in_tag_seq)))
    logging.debug('Output permutations:')
    for p in out_tag_seq:
        logging.debug('    %s' % (str(p)))

    return out_tag_seq


# =============================================================================

def quantiles(in_data, quant_list):
    """Compute the quantiles for the given input data.

    USAGE:
      quant_val_list = quantiles(in_data, quant_list)

    ARGUMENT:
      in_data     A vector of numerical data, e.g. frequency counts
      quant_list  A list with quantile values, e.g. [0.5,0.25,0.50,0.75,0.95]

    DESCRIPTION:
      This routine computes and returns the values for the given quantiles and
      the give ndata.
    """

    len_in_data = len(in_data)

    sort_data = in_data[:]  # Make a copy of the list
    sort_data.sort()

    val_data = []

    for quant in quant_list:
        if (quant < 0.0) or (quant > 1.0):
            logging.exception('Quantile value not between 0 and 1: %f' % (quant))
            raise Exception

        quant_ind = float(quant * (len_in_data - 1))  # Adjust for index start 0!

        quant_ind_floor = math.floor(quant_ind)
        quant_ind_int = int(quant_ind_floor)

        if (quant_ind == quant_ind_floor):  # Check for fractionals
            val_data.append(sort_data[quant_ind_int])
        else:
            quant_ind_frac = quant_ind - quant_ind_floor  # Fractional part

            tmp_val1 = sort_data[quant_ind_int]
            tmp_val2 = sort_data[quant_ind_int + 1]

            tmp_val = tmp_val1 + (tmp_val2 - tmp_val1) * quant_ind_frac

            val_data.append(tmp_val)

    del sort_data  # Delete local copy of list

    return val_data


# =============================================================================
# Special random distributions

def random_linear(n):  # Return triangle distribution
    """Based on Paul Thomas' R code, 23 July 2007.

       Returns a random number 0 >= r < n, with a linear distribution, i.e.
       with p(n) < p(m) if n < m.
    """

    return math.sqrt(random.random() * (n) ** 2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def random_expo(n):  # Return expnential distribution
    """Returns a random number 0 >= r < n, with an exponential distribution.
    """

    r = n * random.expovariate(10.0)

    while (r >= n):  # make sure r is not too large
        r = n * random.expovariate(10.0)

    return r


# =============================================================================
#
# Following code taken from Rational.py module
#
# changed: "import math as _math" as math already imported, then changed all
#          references of _math to math.

def _gcd(a, b):
    if a > b:
        b, a = a, b
    if a == 0:
        return b
    while 1:
        c = b % a
        if c == 0:
            return a
        b = a
        a = c


def _trim(n, d, max_d):
    if max_d == 1:
        return n / d, 1
    last_n, last_d = 0, 1
    current_n, current_d = 1, 0
    while 1:
        div, mod = divmod(n, d)
        n, d = d, mod
        before_last_n, before_last_d = last_n, last_d
        next_n = last_n + current_n * div
        next_d = last_d + current_d * div
        last_n, last_d = current_n, current_d
        current_n, current_d = next_n, next_d
        if mod == 0 or current_d >= max_d:
            break
    if current_d == max_d:
        return current_n, current_d
    i = (max_d - before_last_d) / last_d
    alternative_n = before_last_n + i * last_n
    alternative_d = before_last_d + i * last_d
    alternative = _Rational(alternative_n, alternative_d)
    last = _Rational(last_n, last_d)
    num = _Rational(n, d)
    if abs(alternative - num) < abs(last - num):
        return alternative_n, alternative_d
    else:
        return last_n, last_d


def _approximate(n, d, err):
    r = _Rational(n, d)
    last_n, last_d = 0, 1
    current_n, current_d = 1, 0
    while 1:
        div, mod = divmod(n, d)
        n, d = d, mod
        next_n = last_n + current_n * div
        next_d = last_d + current_d * div
        last_n, last_d = current_n, current_d
        current_n, current_d = next_n, next_d
        app = _Rational(current_n, current_d)
        if mod == 0 or abs(app - r) < err:
            break
    return app


def _float_to_ratio(x):
    """\
    x -> (top, bot), a pair of co-prime longs s.t. x = top/bot.

    The conversion is done exactly, without rounding.
    bot > 0 guaranteed.
#	Some form of binary fp is assumed.
#	Pass NaNs or infinities at your own risk.
#
#	>>> rational(10.0)
#	rational(int(10), int(1))
#	>>> rational(0.0)
#	rational(int(0))
#	>>> rational(-.25)
#	rational(-int(1), 4L)
#	"""

    if x == 0:
        return int(0), int(1)
    signbit = 0
    if x < 0:
        x = -x
        signbit = 1
    f, e = math.frexp(x)
    assert 0.5 <= f < 1.0
    # x = f * 2**e exactly

    # Suck up CHUNK bits at a time; 28 is enough so that we suck
    # up all bits in 2 iterations for all known binary double-
    # precision formats, and small enough to fit in an int.
    CHUNK = 28
    top = int(0)
    # invariant: x = (top + f) * 2**e exactly
    while f:
        f = math.ldexp(f, CHUNK)
        digit = int(f)
        assert digit >> CHUNK == 0
        top = (top << CHUNK) | digit
        f = f - digit
        assert 0.0 <= f < 1.0
        e = e - CHUNK
    assert top

    # now x = top * 2**e exactly; fold in 2**e
    r = _Rational(top, 1)
    if e > 0:
        r = r << e
    else:
        r = r >> -e
    if signbit:
        return -r
    else:
        return r

##coerce function
def coerce(x, y):
    t = type(x + y)
    return (t(x), t(y))

##cmp function (deprectaed in python 3)
def cmp(a, b):
    t= (a > b) - (a < b)
    return t


class _Rational:

    def __init__(self, n, d):
        if d == 0:
            return n / d
        n, d = map(long, (n, d))
        if d < 0:
            n *= -1
            d *= -1
        f = _gcd(abs(n), d)
        self.n = n / f
        self.d = d / f

    def __repr__(self):
        if self.d == 1:
            return 'rational(%r)' % self.n
        return 'rational(%(n)r, %(d)r)' % self.__dict__

    def __str__(self):
        if self.d == 1:
            return str(self.n)
        return '%(n)s/%(d)s' % self.__dict__

    def __coerce__(self, other):
        for int in (type(1), type(int(1))):
            if isinstance(other, int):
                return self, rational(other)
        if type(other) == type(1.0):
            return float(self), other
        return NotImplemented

    def __rcoerce__(self, other):
        return coerce(self, other)

    def __add__(self, other):
        return _Rational(self.n * other.d + other.n * self.d,
                         self.d * other.d)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return _Rational(self.n * other.n, self.d * other.d)

    def __rmul__(self, other):
        return self * other

    def inv(self):
        return _Rational(self.d, self.n)

    def __div__(self, other):
        return self * other.inv()

    def __rdiv__(self, other):
        return self.inv() * other

    def __neg__(self):
        return _Rational(-self.n, self.d)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __long__(self):
        if self.d != 1:
            raise ValueError('cannot convert non-integer')
        return self.n

    def __int__(self):
        return int(long(self))

    def __float__(self):
        # Avoid NaNs like the plague
        if self.d > int(1) << 1023:
            self = self.trim(int(1) << 1023)
        return float(self.n) / float(self.d)

    def __pow__(self, exp, z=None):
        if z is not None:
            raise TypeError('pow with 3 args unsupported')
        if isinstance(exp, _Rational):
            if exp.d == 1:
                exp = exp.n
        if isinstance(exp, type(1)) or isinstance(exp, type(int(1))):
            if exp < 0:
                return _Rational(self.d ** -exp, self.n ** -exp)
            return _Rational(self.n ** exp, self.d ** exp)
        return float(self) ** exp

    def __cmp__(self, other):
        return cmp(self.n * other.d, self.d * other.n)

    def __hash__(self):
        return hash(self.n) ^ hash(self.d)

    def __abs__(self):
        return _Rational(abs(self.n), self.d)

    def __complex__(self):
        return complex(float(self))

    def __nonzero__(self):
        return self.n != 0

    def __pos__(self):
        return self

    def __oct__(self):
        return '%s/%s' % (oct(self.n), oct(self.d))

    def __hex__(self):
        return '%s/%s' % (hex(self.n), hex(self.d))

    def __lshift__(self, other):
        if other.d != 1:
            raise TypeError('cannot shift by non-integer')
        return _Rational(self.n << other.n, self.d)

    def __rshift__(self, other):
        if other.d != 1:
            raise TypeError('cannot shift by non-integer')
        return _Rational(self.n, self.d << other.n)

    def trim(self, max_d):
        n, d = self.n, self.d
        if n < 0:
            n *= -1
        n, d = _trim(n, d, max_d)
        if self.n < 0:
            n *= -1
        r = _Rational(n, d)
        upwards = self < r
        if upwards:
            alternate_n = n - 1
        else:
            alternate_n = n + 1
        if self == _Rational(alternate_n + n, d * 2):
            new_n = min(alternate_n, n)
            return _Rational(new_n, d)
        return r

    def approximate(self, err):
        n, d = self.n, self.d
        if n < 0:
            n *= -1
        app = _approximate(n, d, err)
        if self.n < 0:
            app *= -1
        return app


def _parse_number(num):
    if '/' in num:
        n, d = num.split('/', 1)
        return _parse_number(n) / _parse_number(d)
    if 'e' in num:
        mant, exp = num.split('e', 1)
        mant = _parse_number(mant)
        exp = long(exp)
        return mant * (rational(10) ** rational(exp))
    if '.' in num:
        i, f = num.split('.', 1)
        i = long(i)
        f = rational(long(f), int(10) ** len(f))
        return i + f
    return rational(long(num))


def rational(n, d=int(1)):
    if type(n) in (type(''), type(u'')):
        n = _parse_number(n)
    if type(d) in (type(''), type(u'')):
        d = _parse_number(d)
    if isinstance(n, type(1.0)):
        n = _float_to_ratio(n)
    if isinstance(d, type(1.0)):
        d = _float_to_ratio(d)
    for arg in (n, d):
        if isinstance(arg, type(1j)):
            raise TypeError('cannot convert arguments')
    if isinstance(n, _Rational):
        return rational(n.n, n.d * d)
    if isinstance(d, _Rational):
        return rational(n * d.d, d.n)
    return _Rational(n, d)


import builtins

builtins.rational = rational


# =============================================================================
#
# Arthmetic coder, taken from:
#
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/306626
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  A very slow arithmetic coder for Python.
#
#  "Rationals explode quickly in term of space and ... time."
#              -- comment in Rational.py (probably Tim Peters)
#
# Really.  Don't use this for real work.  Read Mark Nelson's
# Dr. Dobb's article on the topic at
#    http://dogma.net/markn/articles/arith/part1.htm
# It's readable, informative and even includes clean sample code.
#
# Contributed to the public domain
# Andrew Dalke < dalke @ dalke scientific . com >

def arith_coder_train(text):
    """text -> 0-order probability statistics as a dictionary

    Text must not contain the NUL (0x00) character because that's
    used to indicate the end of data.
    """
    assert "\x00" not in text
    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    counts["\x00"] = 1
    tot_letters = sum(counts.values())

    tot = 0
    probs = {}
    prev = rational(0)
    for c, count in counts.items():
        next = rational(tot + count, tot_letters)
        probs[c] = (prev, next)
        prev = next
        tot = tot + count
    assert tot == tot_letters

    return probs


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def arith_coder_encode(text, probs):
    """text and the 0-order probability statistics -> longval, nbits

    The encoded number is rational(longval, 2**nbits)
    """

    minval = rational(0)
    maxval = rational(1)
    for c in text + "\x00":
        prob_range = probs[c]
        delta = maxval - minval
        maxval = minval + prob_range[1] * delta
        minval = minval + prob_range[0] * delta

    # I tried without the /2 just to check.  Doesn't work.
    # Keep scaling up until the error range is >= 1.  That
    # gives me the minimum number of bits needed to resolve
    # down to the end-of-data character.
    delta = (maxval - minval) / 2
    nbits = int(0)
    while delta < 1:
        nbits = nbits + 1
        delta = delta << 1
    if nbits == 0:
        # return 0, 0
        return 0  # Only number of bits needed
    else:
        avg = (maxval + minval) << (nbits - 1)  # using -1 instead of /2
    # Could return a rational instead ...

    # return avg.n//avg.d, nbits  # the division truncation is deliberate
    return nbits  # Only number of bits needed

# =============================================================================
if __name__=='__main__':
    print('why did you do this?')