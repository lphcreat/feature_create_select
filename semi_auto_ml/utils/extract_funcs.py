import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.preprocessing import FunctionTransformer

def format_importance(features,feature_importance):
    '''
    features : features name series
    feature_importance : feature importance series
    '''
    feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance})
    feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
    feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
    return feature_importances

def remove_model(X,remove_features=None):
    '''
    create the function for remove features can add to sk pipeline,you can add yourself model like this
    '''
    def drop_columns(X,remove_features=None):
        if remove_features is None:
            return X
        else:
            kf = [item for item in X.columns if item not in remove_features]
            return X[kf]
    return FunctionTransformer(drop_columns,kw_args={'remove_features':remove_features})

def get_IQR(df:pd.DataFrame, k):
    '''
    df : the original data
    k : the multiple of iqr for boundary
    '''
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = pd.Series(q1 - (k * iqr), name='lower_bound')
    upper_bound = pd.Series(q3 + (k * iqr), name='upper_bound')
    return pd.concat([lower_bound, upper_bound], axis=1)

def string_index(trans:pd.Series,threshold)->dict:
    '''
    features : category series
    threshold : the joined categories percentage
    '''
    threshold_num = trans.notnull().sum()*threshold
    temp_df = trans.value_counts(dropna=False,ascending=True).to_frame().reset_index()
    temp_df['label']=temp_df.index
    temp_df.columns = ['cat_name','col_name','label']
    temp_df.loc[temp_df['col_name']<=threshold_num,'label']=temp_df.label.max()+1
    return dict(zip(temp_df.cat_name,temp_df.label))

def reset_threshold(predict_proba,threshold):
    '''
    reset binary label for true by proba threshold.
    '''
    y_pred_true = predict_proba[1]
    y_pred = (y_pred_true>threshold)
    return y_pred