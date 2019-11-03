# -*- coding: utf-8 -*-

"""Top-level package for TempRL."""

from pythomata.__version__ import __title__, __description__, __url__, __version__
from pythomata.__version__ import __author__, __author_email__, __license__, __copyright__

import logging
logging.getLogger('temprl').addHandler(logging.NullHandler())
logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s]: %(message)s',
    level=logging.DEBUG
)
