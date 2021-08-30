'''
This program is where you will store your custom scoring methods.  See readme for further details
'''
import os

def first_char(s1, s2):
    '''
    A quick demo function
    :param s1: first string
    :param s2: second string
    :return:
    '''
    if s1 is not None and s2 is not None:
        if s1[0] == s2[0]:
            return 1
        else:
            return 0
    else:
        return 0


if __name__=='__main__':
    os.exit(0)