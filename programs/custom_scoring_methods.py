'''
This program is where you will store your custom scoring methods.  See readme for further details
'''
import os

def demo_func(x):
    '''
    A quick demo function
    :param x: tuple
    :return:
    '''
    if (x[0] is None and x[1] is None) or (x[0]==x[1]):
        return 1
    else:
        return 0

if __name__=='__main__':
    os.exit(0)