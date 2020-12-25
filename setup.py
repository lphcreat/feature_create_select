from setuptools import setup, find_packages

#当centos安装不了shap时，可以将 \site-packages\evalml\model_understanding\prediction_explanations\_algorithms.py 
# 的 import shap语句注释掉，改模块时计算sharp值的模块
setup(name="semi_auto_ml",
        packages=find_packages(),
        version="0.1.3",
        description="this is for machine learning containing all component,it can improve machine learning effectiveness",
        url="https://github.com/lphcreat/semi_auto_ml.git",
        author="lph",
        author_email="wynfdsm@163.com",
        license="GNU",
        install_notes ="recommend install way:pip install semi_auto_ml --no-deps;pip install -r requires.txt --no-deps",
        zip_safe=True)
