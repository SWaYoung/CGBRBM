# -*- coding: utf-8 -*-
from .gbrbm import GBRBM
from .bbrbm import BBRBM
from .cgbrbm import CGBRBM

# default RBM
RBM = CGBRBM

__all__ = [RBM, BBRBM, GBRBM, CGBRBM]

