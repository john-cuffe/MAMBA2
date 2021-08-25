'''
This is a demo on how to run custom in MAMBA.  Note, that you
must have the inputs and setup self-contained in this program in order to run.

For each function, put a dictionary entry (demo below) into the models list.


my_model_name: the name you want to have it appear on the logger.
Also must be the first entry in the return for the model
model_function:
model_type: either supervised or unsupervised (unsupervised not currently used)

###model dictionary demo
def bar(data):
    return 'baf'


def baz(data):
    return bof

[
{
'my_model_name':'foo',
'model_function':bar,
'model_type':'supervised'
},
{
'my_model_name':'baz'
'model_function':baz,
'model_type':'unsupervised'
}
]


'''
import os


def my_custom_model(y, X, X_hdrs):
    '''
    This is a demo of how to implement a custom regression model.
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X
    :return: Dictionay with type, score, model, and means if applicable.
    '''
    ####import
    from sklearn.linear_model import LinearRegression
    mod = LinearRegression().fit(X, y)
    score = mod.score(X, y)
    return {'type': 'Demo Linear Regression Model', 'score': score, 'model': mod}


models = [
    {
        'my_model_name': 'Demo Linear Regression Model',
        'model_function': my_custom_model,
        'model_type': 'supervised'
    }

]


if __name__=='__main__':
    print('foo bar')
    os._exit(0)