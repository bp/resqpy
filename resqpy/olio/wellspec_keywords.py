"""Module defining dictionary of nexus WELLSPEC column keywords"""

version = '19th April 2022'

# Nexus is a registered trademark of the Halliburton Company

import logging
from typing import Any, Dict, Tuple, Type

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import resqpy.olio.keyword_files as kf

# functions defined here:
# def increment_complaints(keyword):
# def known_keyword(keyword):
# def add_unknown_keyword(keyword):
# def default_value(keyword):
# def complaints(keyword):
# def required_out_list():
# def length_unit_conversion_applicable(keyword):

# nexus wellspec columns as required by pagoda
wk_unknown = -3
wk_banned = -2
wk_preferably_not = -1
wk_okay = 0
wk_preferred = 1
wk_required = 2

wellspec_dict: Dict[str, Tuple[int, int, int, Any, bool]] = {}  # mapping wellspec column key to:
#     (warn count, required in, required out, default, length units boolean, )

# NB: changing entries in this list will usually require other code change elsewhere
# second element of tuple should be >= first element
# yapf: disable
wellspec_dict['IW']       = (0, wk_required,  wk_required,  None,   False)
wellspec_dict['JW']       = (0, wk_required,  wk_required,  None,   False)
wellspec_dict['L']        = (0, wk_required,  wk_required,  None,   False)
wellspec_dict['GRID']     = (0, wk_preferred, wk_required,  None,   False)  # or use main grid name as default
wellspec_dict['RADW']     = (0, wk_preferred, wk_required,  0.25,   True)   # use pagoda spec value for i.p. perf
wellspec_dict['KHMULT']   = (0, wk_okay,      wk_okay,      1.0,    False)  # or use 'NA" as default?
wellspec_dict['STAT']     = (0, wk_okay,      wk_okay,      'ON',   False)
wellspec_dict['ANGLA']    = (0, wk_preferred, wk_required,  0.0,    False)
wellspec_dict['ANGLV']    = (0, wk_preferred, wk_required,  0.0,    False)  # default for other perfs (vertical)
wellspec_dict['LENGTH']   = (0, wk_okay,      wk_okay,      None,   True)   # derive default from cell size
wellspec_dict['KH']       = (0, wk_okay,      wk_okay,      None,   True)   # althernative to LENGTH (one required)
wellspec_dict['SKIN']     = (0, wk_okay,      wk_okay,      0.0,    False)
wellspec_dict['PPERF']    = (0, wk_okay,      wk_okay,      1.0,    False)
wellspec_dict['ANGLE']    = (0, wk_okay,      wk_okay,      360.0,  False)
wellspec_dict['IRELPM']   = (0, wk_okay,      wk_okay,      None,   False)  # default fracture IRELPM for i.p. perf
wellspec_dict['RADB']     = (0, wk_okay,      wk_okay,      None,   True)   # calculate from cell size & k_align, k_v
wellspec_dict['WI']       = (0, wk_okay,      wk_okay,      None,   False)  # caluclate from radb, radw & skin
wellspec_dict['K']        = (0, wk_preferably_not, wk_okay, None,   False)  # derive from conductivity for i.p. perf?
wellspec_dict['LAYER']    = (0, wk_preferably_not, wk_okay, None,   False)  # use LGR i.p. layer for i.p. perf
wellspec_dict['DEPTH']    = (0, wk_okay,      wk_okay,      '#',    True)   # # causes nexus to use cell depth
wellspec_dict['X']        = (0, wk_okay,      wk_okay,      None,   True)   # use cell X for i.p. perf
wellspec_dict['Y']        = (0, wk_okay,      wk_okay,      None,   True)   # use cell Y for i.p. perf
wellspec_dict['CELL']     = (0, wk_banned,    wk_banned,    None,   False)  # CELL is for unstructured grids
wellspec_dict['DTOP']     = (0, wk_banned,    wk_banned,    None,   True)   # not compatible with ANGLA, ANGLV
wellspec_dict['DBOT']     = (0, wk_banned,    wk_banned,    None,   True)   # not compatible with ANGLA, ANGLV
wellspec_dict['RADBP']    = (0, wk_preferably_not, wk_okay, None,   True)   # calculate as for RADB
wellspec_dict['RADWP']    = (0, wk_preferably_not, wk_okay, None,   True)   # use pagoda wellbore radius
wellspec_dict['PORTYPE']  = (0, wk_banned,    wk_banned,    None,   False)  # dual porosity: todo: need to check values
wellspec_dict['FM']       = (0, wk_preferably_not, wk_okay, 0.0,    False)  # dual porosity: connection to fracture?
wellspec_dict['MD']       = (0, wk_preferably_not, wk_okay, 'NA',   False)
wellspec_dict['PARENT']   = (0, wk_preferably_not, wk_okay, 'NA',   False)
wellspec_dict['MDCON']    = (0, wk_preferably_not, wk_okay, 'NA',   False)
wellspec_dict['SECT']     = (0, wk_preferably_not, wk_okay, 1,      False)
wellspec_dict['FLOWSECT'] = (0, wk_preferably_not, wk_okay, 1,      False)
wellspec_dict['ZONE']     = (0, wk_preferably_not, wk_okay, 1,      False)
wellspec_dict['GROUP']    = (0, wk_preferably_not, wk_okay, 1,      False)
wellspec_dict['TEMP']     = (0, wk_preferably_not, wk_okay, 'NA',   False)
wellspec_dict['IPTN']     = (0, wk_preferably_not, wk_okay, 1,      False)  # pattern
wellspec_dict['D']        = (0, wk_preferably_not, wk_okay, 'NA',   False)  # non D'Arcy flow
wellspec_dict['ND']       = (0, wk_preferably_not, wk_okay, 'NA',   False)  # non D'Arcy flow
wellspec_dict['DZ']       = (0, wk_preferably_not, wk_okay, None,   True)   # non D'Arcy flow; use LENGTH value? or DZ

wellspec_dtype: Dict[str, Type] = { }  # mapping wellspec column key to expected data type

wellspec_dtype['IW']       = int
wellspec_dtype['JW']       = int
wellspec_dtype['L']        = int
wellspec_dtype['GRID']     = str
wellspec_dtype['RADW']     = float
wellspec_dtype['KHMULT']   = float
wellspec_dtype['STAT']     = str
wellspec_dtype['ANGLA']    = float
wellspec_dtype['ANGLV']    = float
wellspec_dtype['LENGTH']   = float
wellspec_dtype['KH']       = float
wellspec_dtype['SKIN']     = float
wellspec_dtype['PPERF']    = float
wellspec_dtype['ANGLE']    = float
wellspec_dtype['IRELPM']   = int
wellspec_dtype['RADB']     = float
wellspec_dtype['WI']       = float
wellspec_dtype['K']        = float
wellspec_dtype['LAYER']    = int
wellspec_dtype['DEPTH']    = float   # '#' causes nexus to use cell depth
wellspec_dtype['X']        = float   # use cell X for i.p. perf
wellspec_dtype['Y']        = float   # use cell Y for i.p. perf
wellspec_dtype['CELL']     = int     # CELL is for unstructured grids
wellspec_dtype['DTOP']     = float   # not compatible with ANGLA, ANGLV
wellspec_dtype['DBOT']     = float   # not compatible with ANGLA, ANGLV
wellspec_dtype['RADBP']    = float   # calculate as for RADB
wellspec_dtype['RADWP']    = float
wellspec_dtype['PORTYPE']  = str     # dual porosity: todo: need to check type
wellspec_dtype['FM']       = float
wellspec_dtype['MD']       = float
wellspec_dtype['PARENT']   = str
wellspec_dtype['MDCON']    = float
wellspec_dtype['SECT']     = str     # todo: need to check type
wellspec_dtype['FLOWSECT'] = str     # todo: need to check type
wellspec_dtype['ZONE']     = int
wellspec_dtype['GROUP']    = str
wellspec_dtype['TEMP']     = float
wellspec_dtype['IPTN']     = int
wellspec_dtype['D']        = float
wellspec_dtype['ND']       = str
wellspec_dtype['DZ']       = float
# yapf: enable


def increment_complaints(keyword):
    """Increments the count of complaints (warnings) associated with the keyword."""

    global wellspec_dict
    assert (keyword.upper() in wellspec_dict.keys())
    old_entry = wellspec_dict[keyword.upper()]
    wellspec_dict[keyword.upper()] = (old_entry[0] + 1, old_entry[1], old_entry[2], old_entry[3], old_entry[4])


def known_keyword(keyword):
    """Returns True if the keyword exists in the wellspec dictionary."""

    return keyword.upper() in wellspec_dict.keys()


def add_unknown_keyword(keyword):
    """Adds the keyword to the dictionary with attributes flagged as unknown."""

    global wellspec_dict
    assert (not known_keyword(keyword))
    wellspec_dict[keyword.upper()] = (1, wk_unknown, wk_banned, None, False)  # assumes warning or error already given


def default_value(keyword):
    """Returns the default value for the keyword."""

    assert (known_keyword(keyword))
    return wellspec_dict[keyword][3]


def complaints(keyword):
    """Returns the number of complaints (warnings) logged for the keyword."""

    assert (known_keyword(keyword))
    return wellspec_dict[keyword][0]


def check_value(keyword, value):
    """Returns True if the value is acceptable for the keyword."""

    try:
        key = keyword.upper()
        if not known_keyword(key):
            return False
        if key in ['IW', 'JW', 'L', 'LAYER', 'IRELPM', 'CELL', 'SECT', 'FLOWSECT', 'ZONE', 'IPTN']:
            return int(value) > 0
        elif key == 'GRID':
            return len(str(value)) > 0
        elif key == 'STAT':
            return (str(value)).upper() in ['ON', 'OFF']
        elif key == 'ANGLA':
            return -360.0 <= float(value) and float(value) <= 360.0
        elif key == 'ANGLV':
            return 0.0 <= float(value) and float(value) <= 180.0
        elif key in ['RADW', 'RADB', 'RADWP', 'RADBP']:
            return float(value) > 0.0
        elif key in ['WI', 'LENGTH', 'KH', 'KHMULT', 'K', 'DZ']:
            return float(value) >= 0.0
        elif key == 'PPERF':
            return 0.0 <= float(value) and float(value) <= 1.0
        elif key == 'ANGLE':
            return 0.0 <= float(value) and float(value) <= 360.0
        elif key in ['SKIN', 'DEPTH', 'X', 'Y', 'TEMP']:
            float(value)
            return True
        else:
            return True
    except Exception:
        return False


def required_out_list():
    """Returns a list of keywords that are required."""

    list = []
    for key in wellspec_dict.keys():
        if wellspec_dict[key][2] == wk_required:
            list.append(key)
    return list


def length_unit_conversion_applicable(keyword):
    """Returns True if the keyword has a quantity class of length."""

    assert (known_keyword(keyword))
    return wellspec_dict[keyword][4]


def load_wellspecs(wellspec_file, well = None, column_list = []):
    """Reads the Nexus wellspec file returning a dictionary of well name to pandas dataframe.

    args:
       wellspec_file (string): file path of ascii input file containing wellspec keywords
       well (optional, string): if present, only the data for the named well are loaded;
          if None, data for all wells are loaded
       column_list (list of strings, optional): if present, each dataframe returned contains
          these columns, in this order; if None, the resulting dictionary contains only
          well names as keys (each mapping to None rather than a dataframe); if an empty list,
          each dataframe contains the columns listed in the corresponding wellspec header, in
          the order found in the file

    returns:
       dictionary (string: pandas dataframe) mapping each well name found in the wellspec file
          to a dataframe containing the wellspec data
    """

    assert wellspec_file, 'no wellspec file specified'

    if column_list is not None:
        for column in column_list:
            assert column.upper() in wellspec_dict, 'unrecognized wellspec column name ' + str(column)
    selecting = bool(column_list)

    well_dict = {}  # maps from well name to pandas data frame with column_list as columns

    with open(wellspec_file, 'r') as fp:
        while True:
            found = kf.find_keyword(fp, 'WELLSPEC')
            if not found:
                break
            line = fp.readline()
            words = line.split()
            assert len(words) >= 2, 'missing well name after WELLSPEC keyword'
            well_name = words[1]
            if well and well_name.upper() != well.upper():
                continue
            if column_list is None:
                well_dict[well_name] = None
                continue
            kf.skip_blank_lines_and_comments(fp)
            line = kf.strip_trailing_comment(fp.readline()).upper()
            columns_present = line.split()
            if selecting:
                column_map = np.full((len(column_list),), -1, dtype = int)
                for i in range(len(column_list)):
                    column = column_list[i].upper()
                    if column in columns_present:
                        column_map[i] = columns_present.index(column)
                df_col = column_list
            else:
                df_col = columns_present
            data = {col: [] for col in df_col}
            all_null = True
            while True:
                kf.skip_comments(fp)
                if kf.blank_line(fp):
                    break  # unclear from Nexus doc what marks end of table
                if kf.specific_keyword_next(fp, 'WELLSPEC') or kf.specific_keyword_next(fp, 'WELLMOD'):
                    break
                line = kf.strip_trailing_comment(fp.readline())
                words = line.split()
                assert len(words) >= len(columns_present),  \
                    f'insufficient data in line of wellspec table {well} [{line}]'
                if selecting:
                    for col_index, col in enumerate(column_list):
                        if column_map[col_index] < 0:
                            if column_list[col_index].upper() == 'GRID':
                                data[col].extend(['ROOT'])
                            else:
                                data[col].extend([np.NaN])
                        else:
                            v = words[column_map[col_index]]
                            if v == 'NA':
                                data[col].extend([np.NaN])
                            elif v == '#':
                                data[col].extend([v])
                            else:
                                data[col].extend([wellspec_dtype[col.upper()](v)])
                        if data[col][-1] != np.NaN:
                            all_null = False
                else:
                    for col, v in zip(columns_present, words[:len(columns_present)]):
                        if v == 'NA':
                            data[col].extend([np.NaN])
                        elif v == '#':
                            data[col].extend([v])
                        else:
                            data[col].extend([wellspec_dtype[col](v)])
                        if data[col][-1] != np.NaN:
                            all_null = False
            if all_null:
                log.warning(f'skipping null wellspec data for well {well_name}')
                continue
            data = {k: v for k, v in data.items() if v}
            df = pd.DataFrame(data, columns = df_col)
            if well:
                well_dict[well] = df
                break  # NB. if more than one table for a well, this function returns first, Nexus uses last
            well_dict[well_name] = df

    # log.debug(f'load-wellspecs returning:\n{well_dict}')

    return well_dict
