# trademark.py module for mentioning trademarks in diagnostic log

version = '9th April 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging as lg

log = lg.getLogger(__name__)

nexus_tm_level = None


def log_nexus_tm(level = lg.INFO):
   global nexus_tm_level
   if isinstance(level, str):
      level = lg.__dict__[level.upper()]
   if nexus_tm_level is None or level > nexus_tm_level:
      preamble = '(ignore severity) ' if level > 20 else ''
      log.log(level, preamble + 'Nexus is a registered trademark of the Halliburton Company')
      nexus_tm_level = level
