# -*- coding: utf-8 -*-

"""Top-level package for TempRL."""


import logging
logging.getLogger('temprl').addHandler(logging.NullHandler())
logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s]: %(message)s',
    level=logging.DEBUG
)
