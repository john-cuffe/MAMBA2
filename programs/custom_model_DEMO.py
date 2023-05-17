'''
This is an exmaple of a custom model.  The custom model for MAMBA MUST have the following a 'predict_proba' method.

If you want to have MAMBA create your own model on the fly, then you MUST also include all of the associated
needs for the class (so fit, a cost function etc).

If you are using a more simplistic deterministic model, merely a predict_proba function that returns
an array of length X in either of these formats:
In this format, the entry is just the score from a deterministic model
arr = [[1],
       [2]]

OR

If you a returning a model that gives predicted probabilities of a match or other categories, give the 'match' probability first
arr = [[.25,.75],
       [.75,.25]]

This is a really simple model that just returns the sum of X,so the predict_proba function
is just the addition of the two columns. You would then change the match threshold to whatever value you wanted.

For a more complicated logisitic regression class, see https://q-viper.github.io/2020/08/10/writing-a-logistic-regression-class-from-scratch/
'''

class uhe_deterministic_model:
    '''
    Simple deterministic class
    '''
    def __init__(self):
        self.X = None
        self.y = None

    ###predict_proba function
    def predict_proba(self, X):
        '''
        Just return the sum of X
        :param X:
        :return:
        '''
        ###calculate the DPI sum
        return(np.sum(X))
