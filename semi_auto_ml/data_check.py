from featuretools.selection import remove_highly_null_features,remove_single_value_features,remove_highly_correlated_features


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
    def check_outline():
        #TODO finish outline_check.py
        pass