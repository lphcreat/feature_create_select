import pandas as pd
from evalml.automl import AutoMLSearch
from evalml.objectives import FraudCost
from .utils.trans_model import SModelTrans
from evalml.pipelines import (MulticlassClassificationPipeline as MP,
                                BinaryClassificationPipeline  as BP,
                                RegressionPipeline as RP)
from evalml.pipelines.components import *

class ModelSelect():

    '''
    auto select model by evalml
    '''

    def __init__(self,problem_type:str,self_pipelines = None, objective=None,**kwds):
        '''
        Parameters
        --------
        problem_type: binary,multiclass,regression
        self_pipelines: define yourself pipline,please use define_pipline generating it
        cost_func: default by evalml.objectives.FraudCost or you can set to auto,if you want overwrite it please see
        https://evalml.alteryx.com/en/stable/user_guide/objectives.html
        '''
        self.problem_type = problem_type
        if isinstance(objective,dict):
            objective = FraudCost(retry_percentage=objective.get('retry_percentage',0),
                            interchange_fee=objective.get('interchange_fee',0.04),
                            fraud_payout_percentage=objective.get('loss_percentage',0.9),
                            amount_col=objective['amount_col'])
        elif objective is None:
            objective = 'auto'
        self.auto_ml=AutoMLSearch(problem_type=problem_type,
                        allowed_pipelines=self_pipelines,objective=objective,
                        additional_objectives=['auc', 'f1', 'precision'],**kwds)
    
    def search(self,X:pd.DataFrame,y:pd.Series):
        '''
        Parameters
        --------
        '''
        self.auto_ml.search(X,y,data_checks=None)
        return self.auto_ml.rankings

    @staticmethod
    def define_pipline(problem_type,estimators:list,hyperparameters:dict,preprocessing_components:list=None):
        '''
        define yourself piplines
        Parameters
        --------
        problem_type: binary,multiclass,regression
        estimators: a list contain estimators from eval or SModelTrans generate
        hyperparameters:estimators parameters
        preprocessing_components: a list for processing data,if None will use default. from eval or SModelTrans generate  
        '''
        piplines = []
        pipline_dict = {'binary':BP,'multiclass':MP,'regression':RP}
        pipline_type = pipline_dict[problem_type]
        if preprocessing_components is None:
            preprocessing_components = [DropNullColumns,Imputer,DateTimeFeaturizer,OneHotEncoder,StandardScaler]
        for estimator in estimators:
            class CustomPipeline(pipline_type,estimator):
                custom_name = f"{estimator.name} w/ {' + '.join([component.name for component in preprocessing_components])}"
                component_graph = preprocessing_components + [estimator]
                custom_hyperparameters = hyperparameters
            piplines.append(CustomPipeline)
        return piplines