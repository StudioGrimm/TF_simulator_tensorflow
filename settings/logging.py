"""
Contains the logging settings.

Created Aug 24 17:07:05 2023

@author: lmienhardt, lkuehl
"""

# logging and decorators
import logging as log

# logging settings
log.basicConfig(
    format='%(asctime)s %(levelname)-8s %(processName)s %(threadName)s %(funcName)-20s %(message)s',
        # log.INFO for normal run
    # level=log.INFO,
        # log.DEBUG for diagnostics
    level=log.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')