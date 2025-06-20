# -*- coding: utf-8 -*-
from .base import BaseDataset
from .alex_mi import AlexMI
from .bnci import BNCI2014001, BNCI2014004
from .tsinghua import Wang2016
from .physionet import PhysionetMI
from .cbcic import CBCIC2019001
from .nakanishi2015 import Nakanishi2015
from .cho2017 import Cho2017
from .schirrmeister2017 import Schirrmeister2017
from .dreamer import DREAMER

__all__ = ['AlexMI',
           'BNCI2014001', 'BNCI2014004',
           'Wang2016',
           'PhysionetMI',
           'CBCIC2019001',
           'Nakanishi2015',
           'Cho2017',
           'Schirrmeister2017',
           'DREAMER'
           ]
