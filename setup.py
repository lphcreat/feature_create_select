from setuptools import setup
from pathlib import Path
import os
_data_dir = Path(__file__).parent
_data_dir=_data_dir.joinpath('semi_auto_ml.egg-info')
print(_data_dir)
setup(name="semi_auto_ml",
        version="0.1.3",
        description="this is for machine learning containing all component,it can improve machine learning effectiveness",
        url="https://github.com/lphcreat/semi_auto_ml.git",
        author="lph",
        author_email="wynfdsm@163.com",
        license="GNU",
        packages=["semi_auto_ml"],
        install_requires=open(_data_dir / 'requires.txt').readlines(),
        install_notes ="recommend install way:pip install semi_auto_ml --no-deps;pip install -r requires.txt --no-deps",
        zip_safe=True)
