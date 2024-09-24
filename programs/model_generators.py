from __future__ import division
import copy
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.stats import randint as sp_randint
from programs.logger_setup import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
logger=logger_setup(CONFIG['log_file_name'])
from sklearn.ensemble import RandomForestClassifier
##Run the Classifier
def runRFClassifier(y, X, nT, nE, mD):
    """
    Description: Executes random forest classifier
    Parameters:
        y - array of truth 1/0 match status
        X - array of features with same index as y
        nT - number of trees to use to estimate
        nE - number of features per tree
        mD - max depth of each tree
    Returns: model, the model object for the random forest classifier
    """
    tfunc = time.time()
    logger.info('#' * 10 + str(round((time.time() - starttime) / 60, 2)) + ' Minutes: Run Random Forest' + '#' * 10)
    # initialize estimator
    logger.info('Random Forest Params:\n Number of Trees={0}\n Max Features={2}\n Max Depth={1}'.format(nT, mD, nE))
    model = RandomForestClassifier(n_estimators=nT, max_features=nE, max_depth=mD, random_state=12345)
    # fit model
    model.fit(X, y)
    # print('Random Forest Time (Min): {0}'.format((time.time()-tfunc)/60))
    return model

####VIF score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X, headers):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif['name'] = headers
    return(vif)


def generate_logit(y, X, X_hdrs):
    '''
    Generates a linear logistic regression model
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X
    :return: Dictionay with type, score, model, and means if applicable.  Note we have a warning for high multicollinearity, AND ONLY USED
    WHEN USING ACCURACY AS THE SCORE (ONLY REAL COMPARABLE OPTION OUT OF THE BOX)
    '''
    logger.info('######CREATE LOGISTIC REGRESSION MODEL ######')
    from sklearn.linear_model import LogisticRegression
    ##Run the Grid Search
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 5
    else:
        niter = 20
    ##Mean-Center
    X=pd.DataFrame(X)
    ##Save the means
    X.columns=X_hdrs
    X_means=X.mean().to_dict()
    X=X-X.mean()
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        myparams = {
            'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        mod=LogisticRegression(random_state=0, solver='liblinear')
        cv_rfc = RandomizedSearchCV(estimator=mod, param_distributions=myparams, cv=10, scoring=scoringcriteria,n_iter=niter)
        cv_rfc.fit(X, y)
        preds=cv_rfc.predict(X)
        score=cv_rfc.score(X,y)
        try:
            vif=calc_vif(X, X_hdrs)
            if max(vif['VIF']) > 5:
                logger.info('WARNING, SOME VARIABLES HAVE HIGH COLINEARITY (EVEN WITH MEAN-CENTERING).  RECONSIDER USING THE LOGIT. VARIABLES WITH ISSUES ARE:')
                for i in vif.to_dict('records'):
                    if i['VIF'] > 5:
                        logger.info('{}: VIF = {}'.format(i['name'], round(i['VIF'],2)))
            return {'type': 'Logistic Regression', 'score': score, 'model': mod, 'means': X_means,
                    'variable_headers': X_hdrs}

        except Exception as error:
            logger.warning("Warning.  Unable to calculate VIF, error {}.  Restart MAMBA with logit disabled. Proceeding by ignoring Logit".format(error))
            return 'fail'
    else:
        myparams = {
            'estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        mod = LogisticRegression(random_state=0, solver='liblinear').fit(X, y)
        selector = RFECV(mod, step=1, cv=5)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        preds = cv_rfc.predict(X)
        score = cv_rfc.score(X, y)
        try:
            vif = calc_vif(X, X_hdrs)
            if max(vif['VIF']) > 5:
                logger.info(
                    'WARNING, SOME VARIABLES HAVE HIGH COLINEARITY (EVEN WITH MEAN-CENTERING).  RECONSIDER USING THE LOGIT. VARIABLES WITH ISSUES ARE:')
                for i in vif.to_dict('records'):
                    if i['VIF'] > 5:
                        logger.info('{}: VIF = {}'.format(i['name'], round(i['VIF'], 2)))
        except Exception as error:
            logger.warning("Warning.  Unable to calculate VIF, error {}.  Restart MAMBA with logit disabled. Proceeding by ignoring Logit".format(error))
            return 'fail'
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            my_ranking = cv_rfc.best_estimator_.ranking_[hdr]
            if my_ranking > 7:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        mod.fit(new_X, y)
        score = np.round(mod.score(new_X, y), 5)
        ###limit X to just what we need
        logger.info('Random Forest Complete.  Score={}'.format(score))
        return {'type': 'Logistic Regression', 'score': score, 'model': mod, 'variable_headers': new_X_hdrs, 'means': X_means}

def generate_rf_mod(y, X, X_hdrs):
    '''
    Generate the random forest model we are going to use for the matching
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X    '''
    logger.info('######CREATE RANDOM FOREST MODEL ######')
    ###Generate the Grid Search to find the ideal values
    features_per_tree = ['sqrt', 'log2']
    rf = RandomForestClassifier(n_jobs=int(CONFIG['rf_jobs']), max_depth=10, max_features='sqrt', n_estimators=10)
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'max_features': features_per_tree,
        'max_depth': sp_randint(2, 25)}
    ##Run the Grid Search
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 1
    else:
        niter = 50
    if ast.literal_eval(CONFIG['feature_elimination_mode']) == False:
        cv_rfc = RandomizedSearchCV(estimator=rf, param_distributions=myparams, cv=5, scoring=scoringcriteria,n_iter=niter)
        cv_rfc.fit(X, y)
        ##Save the parameters
        trees = cv_rfc.best_params_['n_estimators']
        features_per_tree = cv_rfc.best_params_['max_features']
        score = cv_rfc.best_score_
        max_depth = cv_rfc.best_params_['max_depth']
        ###No obvious need for MAMBALITE here
        ##In future could just remove the worst performaning variable using a relimp measure then run the full model
        #if mambalite == False:
        logger.info('Random Forest Completed.  Score {}'.format(score))
        rf_mod = runRFClassifier(y, X, trees, features_per_tree, max_depth)
        return {'type':'Random Forest', 'score':score, 'model':rf_mod, 'variable_headers':X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__n_estimators': sp_randint(1, 250),
            'estimator__max_features': features_per_tree,
            'estimator__max_depth': sp_randint(5, 25)}
        selector = RFECV(rf, step=1, cv=25, scoring=scoringcriteria)
        selector.fit(X,y)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        # generate the model
        rf_mod = runRFClassifier(y, X, cv_rfc.best_estimator_.get_params()['estimator__n_estimators'], cv_rfc.best_estimator_.get_params()['estimator__max_features'], cv_rfc.best_estimator_.get_params()['estimator__max_depth'])
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            if cv_rfc.best_estimator_.support_[hdr] == False:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        rf_mod.fit(new_X, y)
        score = np.round(rf_mod.score(new_X, y), 5)
        ###limit X to just what we need
        logger.info('Random Forest Complete.  Score={}'.format(score))
        return {'type': 'Random Forest', 'score': score, 'model': rf_mod, 'variable_headers': new_X_hdrs}

def generate_ada_boost(y, X, X_hdrs):
    '''
    This function generates the adaboosted model
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X
    '''
    logger.info('######CREATE ADABoost MODEL ######')
    from sklearn.ensemble import AdaBoostClassifier
    ###Generate the Grid Search to find the ideal values
    ##setup the SVM
    ada = AdaBoostClassifier(
                         algorithm="SAMME",
                         n_estimators=200)
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 5
    else:
        niter = int(np.round(len(X)/float(2),0))
    features_per_tree = ['sqrt', 'log2', 10, 15]
    myparams = {
        'n_estimators': sp_randint(1, 25),
        'algorithm':['SAMME','SAMME.R']}
    if ast.literal_eval(CONFIG['feature_elimination_mode'])==False:
        ##First, get the scores
        cv_rfc = RandomizedSearchCV(estimator=ada, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        ##Save the parameters
        trees = cv_rfc.best_params_['n_estimators']
        algo = cv_rfc.best_params_['algorithm']
        score = cv_rfc.best_score_
        ###No obvious need for MAMBALITE here
        ##In future could just remove the worst performaning variable using a relimp measure then run the full model
        # if mambalite == False:
        ada_mod = AdaBoostClassifier(algorithm=algo, n_estimators=trees)
        ada_mod.fit(X,y)
        logger.info('AdaBoost Complete.  Score={}'.format(score))
        return {'type': 'AdaBoost', 'score': score, 'model': ada_mod, 'variable_headers':X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__n_estimators': sp_randint(1, 25),
            'estimator__algorithm': ['SAMME', 'SAMME.R']}
        selector = RFECV(ada, step=1, cv=25, scoring=scoringcriteria)
        selector.fit(X,y)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        # generate the model
        ada_mod = AdaBoostClassifier(algorithm=cv_rfc.best_estimator_.get_params()['estimator__algorithm'], n_estimators=cv_rfc.best_estimator_.get_params()['estimator__n_estimators'])
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            if cv_rfc.best_estimator_.support_[hdr] == False:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        ada_mod.fit(new_X,y)
        score = np.round(ada_mod.score(new_X,y), 5)
        ###limit X to just what we need
        logger.info('AdaBoost Complete.  Score={}'.format(score))
        return {'type': 'AdaBoost', 'score': score, 'model': ada_mod, 'variable_headers':new_X_hdrs}

def generate_svn_mod(y, X, X_Hdrs):
    '''
    Generate the SVM model we are going to use for the matching
    :param y: the truth data column showing the real values.
    :param X: the indepdent variables for the model
    :param X_hdrs: the headers of X    '''
    logger.info('\n\n######CREATE SVN MODEL ######\n\n')

    ###Generate the Grid Search to find the ideal values
    from sklearn import svm
    ##setup the SVM
    svc = svm.SVC(class_weight='balanced', gamma='scale')
    if ast.literal_eval(CONFIG['debugmode'])==True:
        niter = 5
    else:
        niter = int(np.round(len(X)/float(2),0))
    myparams={
        'kernel':['linear','poly','rbf']
    }
    if ast.literal_eval(CONFIG['feature_elimination_mode'])==False:
        ##First, get the scores
        svn_rfc = RandomizedSearchCV(estimator=svc, param_distributions=myparams, cv=5, scoring=scoringcriteria)
        svn_rfc.fit(X, y)
        ##Save the parameters
        kernel = svn_rfc.best_params_['kernel']
        score = svn_rfc.best_score_
        logger.info('SVM Complete.  Max Score={}'.format(score))
        svc=svm.SVC(gamma=gamma, degree=degree, kernel=kernel)
        ###Note for when you return--you need to change the predict function to do cross_val_predict
        return {'type':'SVM', 'score':score, 'model':svc, 'variable_headers':X_hdrs}
    else:
        ##First, get the scores
        myparams = {
            'estimator__kernel':['linear','poly','rbf']}
        selector = RFECV(svc, step=1, cv=5)
        cv_rfc = RandomizedSearchCV(estimator=selector, param_distributions=myparams, cv=10, scoring=scoringcriteria,
                                    n_iter=niter)
        cv_rfc.fit(X, y)
        # generate the model
        svn_mod = svm.SVC(kernel=cv_rfc.best_estimator_.get_params()['estimator__kernel'])
        # Pick the X header values we need to use
        new_X_hdrs = []
        to_delete = []
        for hdr in range(len(X_hdrs)):
            my_ranking = cv_rfc.best_estimator_.ranking_[hdr]
            if my_ranking > cv_rfc.best_estimator_.n_features_:
                to_delete.append(hdr)
            else:
                new_X_hdrs.append(X_hdrs[hdr])
        ####Now delete the columns from X
        new_X = np.delete(X, to_delete, axis=1)
        svn_mod.fit(new_X,y)
        score = np.round(svn_mod.score(new_X,y), 5)
        ###limit X to just what we need
        logger.info('Support Vector Machine Complete.  Score={}'.format(score))
        return {'type': 'SVN', 'score': score, 'model': svn_mod, 'variable_headers':new_X_hdrs}
