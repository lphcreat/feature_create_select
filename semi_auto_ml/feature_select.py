
from sklearn.feature_selection import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import gc
from .utils.extract_funcs import format_importance
from .feature_create import AutoCreate
from itertools import chain

class AutoSelect():
    """
    The example in https://github.com/lphcreat/feature_create_select/blob/master/auto_select_test.ipynb.
    Class for performing feature selection for machine learning.
    
    Notes
    --------
        - Calculating the feature importances requires labels (a supervised learning task) 
    """

    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
        self.removed_features = []
    
    def sk_feature_importances(self,cumulative_importance,selector_way = SelectPercentile,score_func=chi2):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance by sklearn feature_selection -> https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection.

        Parameters
        --------
        selector_way : the model from feature_selection example:SelectPercentile/SelectKBest and so on
        score_func : estimator,Function taking two arrays X and y, 
                        and returning a pair of arrays (scores, pvalues), example:chi2,f_regression
        """
        select_result = selector_way(score_func, percentile=10).fit(self.data, self.labels)
        feature_importances = format_importance(self.data.columns,select_result.scores_)
        self.removed_features.append(feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]['feature'].tolist())
        return feature_importances
    
    def lgb_feature_importances(self,problem_type,cumulative_importance,eval_metric=None, 
                                 n_iterations=10, early_stopping = True):
        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")
        for _ in range(n_iterations):
            if problem_type == 'binary':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
            elif problem_type == 'multiclass':
                eval_metric = 'auc_mu'
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1,objective='multiclass',num_class = len(pd.unique(self.labels)))
            elif problem_type == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)
                    # If training using early stopping need a validation set
            if early_stopping:
                train_features, valid_features, train_labels, valid_labels = train_test_split(self.data, self.labels, test_size = 0.15, stratify=self.labels)
                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric = eval_metric,
                          eval_set = [(valid_features, valid_labels)],
                          early_stopping_rounds = 100, verbose = -1)
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()
            else:
                model.fit(self.data, self.labels)
            feature_importance_values += model.feature_importances_ / n_iterations
        feature_importances = format_importance(self.data.columns,feature_importance_values)
        self.removed_features.append(feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]['feature'].tolist())
        return feature_importances

    @staticmethod
    def plot_feature_importances(feature_importances:pd.DataFrame,plot_n = 15, threshold = None):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.
        Parameters
        --------
        plot_n : int, default = 15
            Number of most important features to plot. Defaults to 15 or the maximum number of features whichever is smaller
        
        threshold : float, between 0 and 1 default = None
            Threshold for printing information about cumulative importances
        """
        # Need to adjust number of features if greater than the features in the data
        if plot_n > feature_importances.shape[0]:
            plot_n = feature_importances.shape[0] - 1

        plt.rcParams = plt.rcParamsDefault
        
        # Make a horizontal bar chart of feature importances
        plt.figure(figsize = (10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(feature_importances.index[:plot_n]))), 
                feature_importances['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(feature_importances.index[:plot_n]))))
        ax.set_yticklabels(feature_importances['feature'][:plot_n], size = 12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size = 16); plt.title('Feature Importances', size = 18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize = (6, 4))
        plt.plot(list(range(1, len(feature_importances) + 1)), feature_importances['cumulative_importance'], 'r-')
        plt.xlabel('Number of Features', size = 14); plt.ylabel('Cumulative Importance', size = 14); 
        plt.title('Cumulative Feature Importance', size = 16);

        if threshold:
            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(feature_importances['cumulative_importance'] > threshold))
            plt.vlines(x = importance_index + 1, ymin = 0, ymax = 1, linestyles='--', colors = 'blue')
            plt.show();
            print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    def remove(self,features_enc=None,keep_cols = None):
        """
        Remove the features from the data according to the specified methods.
        Parameters
        --------
            keep_cols : list/str, default = None the cols will not remove
            features_enc : features defining
        Return
        --------
            data : dataframe
                Dataframe with identified features removed
        """
        features_to_drop = set(chain(*self.removed_features))
        if keep_cols is not None:
            #if keep_cols type is str will keep contain keep_cols cols
            if isinstance(keep_cols,str):
                keep_cols = [col for col in features_to_drop if keep_cols in col]
            features_to_drop = list(features_to_drop - set(keep_cols))
        # Remove the features and return
        return AutoCreate.remove_features(features_to_drop,self.data,features_enc=features_enc)