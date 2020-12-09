from setuptools import setup

setup(name="feature_create_select",
        version="0.1.2",
        description="this is for machine learning containing all component,it can improve machine learning effectiveness",
        url="https://github.com/lphcreat/semi_auto_ml.git",
        author="lph",
        author_email="wynfdsm@163.com",
        license="GNU",
        packages=["semi_auto_ml"],
        install_requires=open('core-requirements.txt').readlines(),
        zip_safe=True)
