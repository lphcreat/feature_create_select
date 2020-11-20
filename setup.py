from setuptools import setup

setup(name="feature_create_select",
        version="N/A",
        description="this is for creating and removing features for a dataset intended for machine learning",
        # url="https://github.com/lphcreat/feature_create_select.git",
        author="lph",
        author_email="wynfdsm@163.com",
        license="N/A",
        packages=["feature_create_select"],
        install_requires=[
            "matplotlib>=2.1.2",
            "seaborn>=0.8.1",
            "scikit-learn>=0.22.1",
            'featuretools==0.21.0',
            'numpy>=1.18.1',
            'pandas>=1.0.1',
            ],
        zip_safe=True)
