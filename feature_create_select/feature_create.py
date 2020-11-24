# from sklearn.feature_extraction import *
# if need process text and image you can load feature_extraction and extand the class
import featuretools as ft
import pandas as pd
from featuretools.selection import remove_highly_null_features,remove_single_value_features,remove_highly_correlated_features
from featuretools import variable_types as vtypes
import warnings

class AutoCreate():
    """
    The example in https://github.com/lphcreat/feature_create_select/blob/master/feature_create_test.ipynb.
    Class for create feature use featuretools.
    the parent class from https://github.com/alteryx/featuretools.
    
    Notes
    --------
    1.more function can find in https://featuretools.alteryx.com/en/stable/index.html
    2.you can use featuretools.list_primitives() show functions.
    3.you need define the agg_primitives,trans_primitives,groupby_trans_primitives for generate features,
      the function from list_primitives and by yourself defined
    4.you can defined you function by featuretools.primitives.make_agg_primitive/make_trans_primitive
      https://featuretools.alteryx.com/en/stable/getting_started/primitives.html?highlight=defining-custom-primitives
    """

    def __init__(self,id_name=None):
        if id_name is None:
            id_name = 'auto_create'
        self.auto_create = ft.EntitySet(id = id_name)
        # 限定输入类别string/number/category(object)/datetime
    
    def create_entity(self,entity_id:str,dataframe:pd.DataFrame,**kwds):
        '''
        one by one add pandas dataframe to EntitySet
        '''
        # convert id type to (int32);if ids type are not same,will can't add relation.
        int_types = ['int16', 'int32', 'int64']
        convert_col = dataframe.head.select_dtypes(include=int_types).columns
        dataframe[convert_col] = dataframe[convert_col].astype('int32')
        self.auto_create = self.auto_create.entity_from_dataframe(entity_id=entity_id,
                              dataframe=dataframe,**kwds)

    def add_relation(self,relationships:list):
        '''
        for auto_create add entitys relation.
        Parameters
        --------
        relationships : the entitys relation and relation from parent to child,the format like
                        ['entity1.key1','entity2.key1','entity2.key2','entity3.key2']
        '''
        relationships = [item.split('.') for item in relationships]
        trans_relationships = [ft.Relationship(self.auto_create[parent[0]][parent[1]],
                                self.auto_create[child[0]][child[1]])
                                for parent,child in zip(relationships[::2],relationships[1::2])]
        self.auto_create = self.auto_create.add_relationships(trans_relationships)

    def make_features(self,entityset = None,entities=None, relationships=None,features=None,**kwds):
        '''
        transform data to features,more parameters please read featuretools.dfs;
        if sub entitys can use normalize_entity.
        '''
        if entityset is None:
            entityset = self.auto_create
        if features is None:
            if entities is None:
                feature_matrix, features_def = ft.dfs(entityset=entityset,**kwds)
            else:
                feature_matrix, features_def = ft.dfs(entities=entities,relationships=relationships,**kwds)
            return feature_matrix, features_def
        else:
            if entities is None:
                feature_matrix = ft.calculate_feature_matrix(features,entityset=entityset,**kwds)
            else:
                feature_matrix = ft.calculate_feature_matrix(features,entities=entities,relationships=relationships,**kwds)
            return feature_matrix

    def focus_value(self,entity_id:str,focus_col:str,interesting_values:list = None):
        '''
        Notes
        -------- 
        if you want to add interesting values by multiple columns you can create a new column and input it to this function.
        example ：
        transactions_df['product_id_device'] = transactions_df['product_id'].astype(str) + ' and ' + transactions_df['device']
        self.auto_create["transactions"]["product_id_device"].interesting_values = transactions_df['product_id_device'].unique().tolist()
        '''
        if interesting_values is None:
            interesting_values = self.auto_create[entity_id][focus_col].unique().tolist()
        self.auto_create[entity_id][focus_col].interesting_values = interesting_values

    @staticmethod
    def get_final_data(or_df:pd.DataFrame,features_def,**kwds):
        '''
        check the data types,only support numeric/categorical/boolean,return numeric data.
        1.drop unsupport cols
        2.encode categorical/boolean cols
        '''
        # drop un numeric/categorical cols
        unnum = ['bool','category']
        numeric_and_boolean_dtypes = vtypes.PandasTypes._pandas_numerics + unnum
        clean_df = or_df.select_dtypes(include=numeric_and_boolean_dtypes)
        unuse_col =set(or_df.columns)-set(clean_df.columns)
        features_def = [item for item in features_def if item.get_name() not in unuse_col]
        warnings.warn(f'{unuse_col} columns will be dropped because the dtype')
        # categorical/boolean will be encode to number by one-hot;
        clean_df, features_def = ft.encode_features(clean_df, features_def,**kwds)
        return clean_df,features_def

    @staticmethod
    def clean_features(or_df:pd.DataFrame,features_def,threshold=dict(),count_nan=False,**kwds)->pd.DataFrame:
        '''
        clean features,if you want plot features please use AutoSelect;contain remove_highly_null_features;
        remove_single_value_features;remove_highly_correlated_features.
        '''
        #drop cols by remove function
        or_df,features_def = remove_highly_null_features(or_df,features=features_def,pct_null_threshold=threshold.get('remove_null',0.95))
        or_df,features_def = remove_single_value_features(or_df,features=features_def,count_nan_as_value=count_nan)
        or_df,features_def = remove_highly_correlated_features(or_df,features=features_def,pct_corr_threshold=threshold.get('remove_corr',0.95),**kwds)
        return or_df,features_def

    @staticmethod
    def deploy_features_create(features_enc,model_path):
        '''
        you can save self.features_def to feature_definitions.json for deploying,
        '''
        ft.save_features(features_enc, model_path)
    
    @staticmethod
    def load_features_create(model_path):
        '''
        you can load features_enc;
        Example
        ----------
        1.features = load_features_create('feature_definitions.json')
        2.feature_matrix = make_features(features)
        '''
        return ft.load_features(model_path)

    @staticmethod
    def remove_features(unkeep:list,or_data:pd.DataFrame,features_enc=None):
        '''
        if features_enc is None only return data;with features_enc you will get features_enc for deploying
        '''
        features_def = [item for item in features_enc if item.get_name() not in unkeep]
        keep = [item for item in or_data.columns if item not in unkeep]
        return or_data[keep],features_def