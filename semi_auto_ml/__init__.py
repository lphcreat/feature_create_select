import os
from pathlib import Path
# clear log
_data_dir = Path().parent
file_ = _data_dir / 'evalml_debug.log'
if os.path.exists(file_):
    os.remove(file_)
from .data_check import DataCheck
from .feature_create import AutoCreate
from .feature_select import AutoSelect
from .model_select import ModelSelect
from .model_deploy import ModelDeploy
from .monitor_predict import MonitorPre
from .utils import *