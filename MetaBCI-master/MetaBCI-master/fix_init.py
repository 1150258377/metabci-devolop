# 创建一个脚本来修改__init__.py文件
import os

init_content = """# -*- coding: utf-8 -*-
from .base import BaseDataset
from .alex_mi import AlexMI
from .bnci import BNCI2014001, BNCI2014004
from .tsinghua import Wang2016
from .munich2009 import Munich2009
from .physionet import PhysionetMI
from .cbcic import CBCIC2019001
from .nakanishi2015 import Nakanishi2015
from .cho2017 import Cho2017
from .schirrmeister2017 import Schirrmeister2017
from .tunerl import TuneRL
from .xu2018_minavep import Xu2018_minavep
from .dreamer import DREAMER

__all__ = ['AlexMI',
           'BNCI2014001', 'BNCI2014004',
           'Wang2016',
           'Munich2009',
           'PhysionetMI',
           'CBCIC2019001',
           'Nakanishi2015',
           'Cho2017',
           'Schirrmeister2017',
           'TuneRL',
           'Xu2018_minavep',
           'DREAMER'
           ]
"""

init_path = os.path.join('metabci', 'brainda', 'datasets', '__init__.py')

with open(init_path, 'w') as f:
    f.write(init_content)

print(f"已成功修改 {init_path}") 