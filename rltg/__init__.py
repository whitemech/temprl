# -*- coding: utf-8 -*-

"""Top-level package for RLTG."""


import logging
logging.getLogger('rltg').addHandler(logging.NullHandler())
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

logger = logging.getLogger('matplotlib')
# set WARNING for Matplotlib
logger.setLevel(logging.WARNING)


