'''
This program is where you will store your custom scoring methods.  See readme for further details
'''
def first_char(x, return_value):
    '''
    See if the first characters match as a demo.
    :param x:
    :return:
    '''
    if x[0] is not None and x[1] is not None and x[0][0]==x[1][0]:
        return return_value
    else:
        return 0


if __name__=='__main__':
    os.exit(0)