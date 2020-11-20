# 安装或放入包根目录 FeatureSelector 调用已有的方法，正确的方法在github中
# 引入 sklearn.feature_selection 中的方法 传入对应的model 即可调用对应的方法

from sklearn.feature_selection import *
from feature_selector import FeatureSelector
import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt

class AutoSelect(FeatureSelector):
    """
    Class for performing feature selection for machine learning or data preprocessing.
    the parent class from https://github.com/WillKoehrsen/feature-selector.
    if you want use it you can run 'pip install --no-deps feature_selector'

    Notes
    --------
        - All 6 operations can be run with the `identify_all` method (5 operations from FeatureSelector).
        - Calculating the feature importances requires labels (a supervised learning task) 
        - For the feature importances, to avoid confusion dropped one-hot enconding in FeatureSelector
          so you must input the numberic dataframe.
    """

    def __init__(self,data,labels=None):
        super().__init__(data,labels=labels)
        self.sk_feature_score = None
        self.identify_all_status = False
    
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
        feature_importances = pd.DataFrame({'feature': self.data.columns, 
                    'importance': select_result.scores_})
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
        feature_importances=feature_importances.sort_values('cumulative_importance')
        record_low_importance = feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]
        to_drop = list(record_low_importance['feature'])
        self.sk_feature_score = feature_importances
        self.ops['sk_low_importance'] = to_drop

    def identify_all(self, selection_params):
        """
        Use all six of the methods to identify features to remove.
        
        Parameters
        --------
            
        selection_params : dict
           Parameters to use in the five feature selection methhods.
           Params must contain the keys ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance','sk_low_importance']
        
        """
        
        # Check for all required parameters
        for param in ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance','sk_low_importance']:
            if param not in selection_params.keys():
                raise ValueError('%s is a required parameter for this method.' % param)
        
        # Implement each of the five methods
        self.identify_missing(selection_params['missing_threshold'])
        self.identify_single_unique()
        self.identify_collinear(selection_params['correlation_threshold'])
        self.identify_zero_importance(task = selection_params['task'], eval_metric = selection_params['eval_metric'])
        self.identify_low_importance(selection_params['cumulative_importance'])
        self.sk_feature_importances(selection_params['sk_low_importance'],**selection_params['sk_select_model'])
        
        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)
        print(f'{self.n_identified} features for removal from {self.data.shape[1]} total features')

        # update identify_all_status set to True
        self.identify_all_status = True

    def skplot_feature_importances(self, plot_n = 15, threshold = None):
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
        if plot_n > self.sk_feature_score.shape[0]:
            plot_n = self.sk_feature_score.shape[0] - 1

        self.reset_plot()
        
        # Make a horizontal bar chart of feature importances
        plt.figure(figsize = (10, 6))
        ax = plt.subplot()

        # Need to reverse the index to plot most important on top
        # There might be a more efficient method to accomplish this
        ax.barh(list(reversed(list(self.sk_feature_score.index[:plot_n]))), 
                self.sk_feature_score['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

        # Set the yticks and labels
        ax.set_yticks(list(reversed(list(self.sk_feature_score.index[:plot_n]))))
        ax.set_yticklabels(self.sk_feature_score['feature'][:plot_n], size = 12)

        # Plot labeling
        plt.xlabel('Normalized Importance', size = 16); plt.title('SK Feature Importances', size = 18)
        plt.show()

        # Cumulative importance plot
        plt.figure(figsize = (6, 4))
        plt.plot(list(range(1, len(self.sk_feature_score) + 1)), self.sk_feature_score['cumulative_importance'], 'r-')
        plt.xlabel('Number of Features', size = 14); plt.ylabel('Cumulative Importance', size = 14); 
        plt.title('SK Cumulative Feature Importance', size = 16);

        if threshold:
            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(self.sk_feature_score['cumulative_importance'] > threshold))
            plt.vlines(x = importance_index + 1, ymin = 0, ymax = 1, linestyles='--', colors = 'blue')
            plt.show();
            print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    def plot_all(self):
        """
        plot identify result, if you run this func you must first run identify_all.
        """
        
        # Check identify_all_status
        if not self.identify_all_status:
            raise ValueError('if you run this func you must first run identify_all')
        
        # Implement each of the plot methods
        self.plot_missing()
        self.plot_unique()
        self.plot_collinear()
        self.plot_feature_importances()
        self.skplot_feature_importances()
