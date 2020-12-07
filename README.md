# About
https://github.com/lphcreat/feature_create_select you can easily transform data
to features and select features by this package.
This is a tool for feature create by featuretools and plot features selecting by feature_selector(use 'pip install --no-deps feature_selector'). if you do not need
plot you can use feature_create drop null/unique/corr features.


# Methods
## Import Class
from feature_create_select.feature_create import AutoCreate
from feature_create_select.auto_select_feature import AutoSelect

## Usage
you can see examples in auto_select_test.ipynb/feature_create_test.ipynb

# Future
now we have feature creating and selecting,but in ml,first step is data cleaning.
so will add data cleaning in future,and model selecting and parameters tunning and so on.
the ml path like:data cleaning(包括缺失值处理/异常值剔除等)-> feature creating(还包括数据标准化) -> feature selecting -> model selecting(find best model and parameters) -> model traing -> deploy
