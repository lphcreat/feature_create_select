from featuretools.selection import remove_highly_null_features,remove_single_value_features,remove_highly_correlated_features
from semi_auto_ml.utils.checks import OutlineCheck,IQRCheck,TransCat
from sklearn.pipeline import Pipeline
import pandas as pd
from semi_auto_ml.utils.extract_funcs import remove_model
class DataCheck():
    '''
    check data:outline data/id columns/null columns/unique columns/target columns
    only return the checked columns if you want remove them you can use AutoCreate.remove_features
    '''

    @staticmethod
    def check_highly_null(or_df,threshold=0.90):
        use_cols = remove_highly_null_features(or_df,pct_null_threshold=threshold).columns.tolist()
        return list(set(or_df.columns.tolist()) -set(use_cols))
    
    @staticmethod
    def check_single_value(or_df,count_nan=False):
        use_cols =remove_single_value_features(or_df,count_nan_as_value=count_nan).columns.tolist()
        return list(set(or_df.columns.tolist()) -set(use_cols))

    @staticmethod
    def check_highly_corre(or_df,threshold=0.95):
        '''
        only check num features
        '''
        use_cols = remove_highly_correlated_features(or_df,pct_corr_threshold=threshold).columns.tolist()
        return list(set(or_df.columns.tolist()) -set(use_cols))
    
    @staticmethod
    def check_unless_features(or_df,threshold=0.95):
        #find cols name like id
        contain_id_cols = [item for item in or_df.columns if 'id' in item.lower()]
        #find every row is a unique elements cols
        unique_len = (or_df.nunique()/or_df.shape[0]).to_dict()
        unique_equal_len = [k for k,v in unique_len.items() if v==1]
        return list(set(contain_id_cols)|set(unique_equal_len))
    
    @staticmethod
    def check_target_features(or_df,label_name,threshold=0.95):
        '''
        only calculate number cols
        '''
        if isinstance(label_name,str):
            y = or_df[label_name]
            X = or_df.drop(columns=label_name)
        else:
            y = label_name
            X = or_df
        X = or_df.select_dtypes(include = 'number')
        highly_corr_cols = [label for label, col in X.iteritems() if abs(y.corr(col)) >= threshold]
        return highly_corr_cols
    
    @staticmethod
    def clean_outliners(train_data,OCS=True,IQR=True,TC=True):
        '''
        if use datetime features, first process it and next use this function
        can drop or label outliners and joining category to num(you can set it to categorical data)
        '''
        #TODO need test
        other_columns = train_data.select_dtypes(exclude=['number','object','category']).columns()
        drop_clf = remove_model(train_data,remove_features=other_columns)
        pipeline_step = [('drop_dim', drop_clf)]
        result_data = train_data.select_dtypes(include = 'number')
        result_cat = train_data.select_dtypes(include = ['object','category'])
        if IQR:
            iqr = OutlineCheck(IQRCheck())
            iqr_clf,result_data = iqr.get_detail(result_data)
            pipeline_step.append(('IQR', iqr_clf))
        if OCS:
            ocs = OutlineCheck()
            ocs_clf,result_data = ocs.get_detail(result_data)
            pipeline_step.append(('OCS', ocs_clf))
        if TC:
            tc = TransCat()
            result_cat = tc.fit_transform(result_cat)
            pipeline_step.append(('TransCat', tc))
        joint_clf = Pipeline(steps=pipeline_step)
        return joint_clf,pd.concat([result_cat,result_data],axis=1)