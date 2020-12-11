# add a unsupervise model for check the outlier
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
from evalml.data_checks import OutliersDataCheck

class OutlineCheck():
    '''
    check outline rows by sk model
    number columns check data outline max/min/LUQ
    categories columns check category count
    '''
    #TODO add check category/add IQR check

    def __init__(self,clf=None,**kwargs):
        self.clf = clf
        if self.clf is None:
            self.clf = OneClassSVM(**kwargs)
    
    def get_detail(self,X:pd.DataFrame):
        pre_result = self.clf.fit_predict(X)
        inliers = X[pre_result==1]
        return self.clf,inliers
    
    @staticmethod
    def get_predict_detail(clf,X):
        '''
        Params:
        clf:the model by get_detail return
        X:the input data for check
        '''
        pre = clf.predict(X)
        if X.shape[0]>1:
            return X[pre==1]
        else:
            return pre[0]