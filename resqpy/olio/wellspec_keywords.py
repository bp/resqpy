"""Module for loading WELLSPEC files. 

The module includes a dictionary of nexus WELLSPEC column keywords, functionality to
read WELLSPEC files and transform the well data into Pandas DataFrames.
"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import datetime
from typing import Any, Dict, Tuple, Type, Optional, List, Union, TextIO

import resqpy.olio.keyword_files as kf

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


def increment_complaints(keyword):  # pragma: no cover
    """Increments the count of complaints (warnings) associated with the keyword."""

    global wellspec_dict
    assert keyword.upper() in wellspec_dict.keys()
    old_entry = wellspec_dict[keyword.upper()]
    wellspec_dict[keyword.upper()] = (
        old_entry[0] + 1,
        old_entry[1],
        old_entry[2],
        old_entry[3],
        old_entry[4],
    )


def known_keyword(keyword):  # pragma: no cover
    """Returns True if the keyword exists in the wellspec dictionary."""

    return keyword.upper() in wellspec_dict.keys()


def add_unknown_keyword(keyword):  # pragma: no cover
    """Adds the keyword to the dictionary with attributes flagged as unknown."""

    global wellspec_dict
    assert not known_keyword(keyword)
    wellspec_dict[keyword.upper()] = (
        1,
        wk_unknown,
        wk_banned,
        None,
        False,
    )  # assumes warning or error already given


def default_value(keyword):  # pragma: no cover
    """Returns the default value for the keyword."""

    assert known_keyword(keyword)
    return wellspec_dict[keyword][3]


def complaints(keyword):  # pragma: no cover
    """Returns the number of complaints (warnings) logged for the keyword."""

    assert known_keyword(keyword)
    return wellspec_dict[keyword][0]


def check_value(keyword, value):
    """Returns True if the value is acceptable for the keyword."""
    try:
        key = keyword.upper()
        if not known_keyword(key):
            return False
        if key in [
                "IW",
                "JW",
                "L",
                "LAYER",
                "IRELPM",
                "CELL",
                "SECT",
                "FLOWSECT",
                "ZONE",
                "IPTN",
        ]:
            return int(value) > 0
        elif key == "GRID":
            return len(str(value)) > 0
        elif key == "STAT":
            return (str(value)).upper() in ["ON", "OFF"]
        elif key == "ANGLA":
            return -360.0 <= float(value) and float(value) <= 360.0
        elif key == "ANGLV":
            return 0.0 <= float(value) and float(value) <= 180.0
        elif key in ["RADW", "RADB", "RADWP", "RADBP"]:
            return float(value) > 0.0
        elif key in ["WI", "LENGTH", "KH", "KHMULT", "K", "DZ"]:
            return float(value) >= 0.0
        elif key == "PPERF":
            return 0.0 <= float(value) and float(value) <= 1.0
        elif key == "ANGLE":
            return 0.0 <= float(value) and float(value) <= 360.0
        elif key in ["SKIN", "DEPTH", "X", "Y", "TEMP"]:
            float(value)
            return True
        else:  # pragma: no cover
            return True
    except Exception:
        return False


def required_out_list():  # pragma: no cover
    """Returns a list of keywords that are required."""
    list = []
    for key in wellspec_dict.keys():
        if wellspec_dict[key][2] == wk_required:
            list.append(key)
    return list


def length_unit_conversion_applicable(keyword):  # pragma: no cover
    """Returns True if the keyword has a quantity class of length."""

    assert known_keyword(keyword)
    return wellspec_dict[keyword][4]


def load_wellspecs(wellspec_file: str,
                   well: Optional[str] = None,
                   column_list: Union[List[str], None] = [],
                   keep_duplicate_cells: bool = False,
                   keep_null_columns: bool = True,
                   last_data_only: bool = True,
                   usa_date_format: bool = False,
                   return_dates_list: bool = False):
    """Reads the Nexus wellspec file returning a dictionary of well name to pandas dataframe.

    arguments:
        wellspec_file (str): file path of ascii input file containing wellspec keywords.
        well (str, optional): if present, only the data for the named well are loaded. If None, data
            for all wells are loaded.
        column_list (List[str]/None): if present, each dataframe returned contains these
            columns, in this order. If None, the resulting dictionary contains only well names as keys
            (each mapping to None rather than a dataframe). If an empty list (default), each dataframe
            contains the columns listed in the corresponding wellspec header, in the order found in
            the file.
        keep_duplicate_cells (bool): if True (default), duplicate cells are kept, otherwise only the
            last entry is kept.
        keep_null_columns (bool): if True (default), columns that contain all NA values are kept,
            otherwise they are removed.
        last_data_only (bool): If True, only the last entry of well data in the file are used in the
            dataframe, otherwise all of the well data are used at different times.
        usa_date_format (bool): If True, wellspec file is expected to contain date formats in MM/DD/YYYY.
            if False, DD/MM/YYYY.
        return_dates_list (bool, default False): if True, a sorted list of unique dates present in the
            wellspec file is also returned, with dates in iso format

    returns:
        well_dict (Dict[str, Union[pd.DataFrame, None]]): mapping each well name found in the
            wellspec file to a dataframe containing the wellspec data
        or (well_dict, dates_list): where dates list is a sorted list of all dates present in the
            wellspec file (including those not relevant to a specific well), in iso format

    note:
        if return_dates_list is True, the returned list always contains all dates from the wellspec file
        that applied to any entry, regardless of the well and last_data_only arguments; the dates list
        will not include a null entry, even if there are wellspec data before the first timestamp
    """
    assert wellspec_file, "No wellspec file specified."

    if column_list is not None:
        for column in column_list:
            assert (column.upper() in wellspec_dict), f"Unrecognized wellspec column name {column}."
    selecting = bool(column_list)

    well_dict = {}
    well_pointers = get_well_pointers(wellspec_file, usa_date_format = usa_date_format)
    dates_list = None
    if return_dates_list:
        dates_list = _prepare_dates_list(well_pointers)

    if column_list is None:
        well_dict = dict.fromkeys(well_pointers, None)
        if return_dates_list:
            return (well_dict, dates_list)
        else:
            return well_dict

    with open(wellspec_file, "r") as file:
        if well:
            well_data = get_all_well_data(
                file,
                well,
                well_pointers[well],
                column_list,
                selecting,
                keep_duplicate_cells,
                keep_null_columns,
                last_data_only,
            )
            if well_data is not None:
                well_dict[well] = well_data
        else:
            for well_name, pointer in well_pointers.items():
                well_data = get_all_well_data(
                    file,
                    well_name,
                    pointer,
                    column_list,
                    selecting,
                    keep_duplicate_cells,
                    keep_null_columns,
                    last_data_only,
                )
                if well_data is not None:
                    well_dict[well_name] = well_data

    if return_dates_list:
        return (well_dict, dates_list)
    else:
        return well_dict


def _prepare_dates_list(well_pointers):
    dates = []
    for entries in well_pointers.values():
        for _, date in entries:
            if date and date not in dates:
                dates.append(date)
    dates.sort()
    return dates


def get_well_pointers(
    wellspec_file: str,
    usa_date_format: bool = False,
    no_date_replacement: Optional[datetime.date] = None,
) -> Dict[str, List[Tuple[int, Union[None, str]]]]:
    """Gets the file locations of each well in the wellspec file for optimised processing of the data.

    arguments:
        wellspec_file (str): file path of ascii input file containing wellspec keywords.
        usa_date_format (bool): if True, the date taken from the wellspec file is in the format
            MM/DD/YYYY, otherwise it is in the format DD/MM/YYYY.
        no_date_replacement (datetime.date, optional): if there is no date given for a well, this
            date is used.

    returns:
        well_pointers (Dict[str, List[Tuple[int, None/str]]]): mapping each well name found in
            the wellspec file to a list of their file locations and dates as tuples. If there is no
            date before the well data in the file, the date is None. If there is a FileNotFoundError
            then None is returned.
    """
    well_pointers: Dict[str, List[Tuple[int, Union[None, str]]]] = {}
    try:
        with open(wellspec_file, "r") as file:
            while True:
                found = kf.find_keyword(file, "WELLSPEC")
                if not found:
                    break
                line = file.readline()
                words = line.split()
                assert len(words) >= 2, "Missing well name after WELLSPEC keyword."
                well_name = words[1]
                if well_name in well_pointers:
                    well_pointers[well_name].append((file.tell(), None))
                else:
                    well_pointers[well_name] = [(file.tell(), None)]
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {wellspec_file} can't be found.")

    time_pointers = {}
    with open(wellspec_file, "r") as file:
        while True:
            found = kf.find_keyword(file, "TIME")
            if not found:
                break
            line = file.readline()
            words = line.split()
            assert len(words) >= 2, "Missing date after TIME keyword."
            date = words[1]
            if '(' in date:
                # sometimes user specifies (HH:MM:SS) along with date - can separate time from date with this check
                date = date.split('(')[0]
            try:
                if usa_date_format:
                    date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
                else:
                    date_obj = datetime.datetime.strptime(date, "%d/%m/%Y").date()
                if no_date_replacement is not None and date_obj < no_date_replacement:
                    raise ValueError(
                        f"The Zero Date {no_date_replacement} must be before the first wellspec TIME {date_obj}.")
                date = date_obj.isoformat()
            except ValueError:
                raise ValueError(f"The date found '{date}' does not match the correct format (usa_date_format "
                                 f"is {usa_date_format}).")
            time_pointers[file.tell()] = date

    current_date = None  # Before first TIME
    if no_date_replacement is not None:
        current_date = no_date_replacement.isoformat()
    for well, well_pointer_list in well_pointers.items():
        for i, well_pointer in enumerate(well_pointer_list):
            for time_pointer, date in time_pointers.items():
                if well_pointer[0] > time_pointer:
                    current_date = date
                else:
                    break
            well_pointers[well][i] = (well_pointer[0], current_date)

    return well_pointers


def get_well_data(
    file: TextIO,
    well_name: str,
    pointer: int,
    column_list: List[str] = [],
    selecting: bool = False,
    keep_duplicate_cells: bool = True,
    keep_null_columns: bool = True,
    date: Optional[str] = None,
) -> Union[pd.DataFrame, None]:
    """Creates a dataframe of the well data for a given well name and at a specific time in the wellspec file.

    The pointer argument is used to go to the file location where the well dataset is located.

    arguments:
        file (TextIO): the opened wellspec file object.
        well_name (str): name of the well.
        pointer (int): the file object's start position of the well data represented as number of
            bytes from the beginning of the file.
        column_list (List[str]): if present, each dataframe returned contains these
            columns, in this order. If None, the resulting dictionary contains only well names as
            keys (each mapping to None rather than a dataframe). If an empty list (default), each
            dataframe contains the columns listed in the corresponding wellspec header, in the order
            found in the file.
        selecting (bool): True if the column_list contains at least one column name, False otherwise
            (default).
        keep_duplicate_cells (bool): if True (default), duplicate cells are kept, otherwise only the
            last entry is kept.
        keep_null_columns (bool): if True (default), columns that contain all NA values are kept,
            otherwise they are removed.
        date (str, optional): the well date which is provided by the get_well_pointers function
            along with the well pointers. 

    Returns:
        Pandas dataframe of the well data or None if all the data are NA.
    """
    file.seek(pointer)
    kf.skip_blank_lines_and_comments(file)
    line = kf.strip_trailing_comment(file.readline()).upper()
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
    data: Dict[str, List] = {col: [] for col in df_col}
    all_null = True
    while True:
        kf.skip_comments(file)
        if (kf.specific_keyword_next(file, "WELLSPEC") or kf.specific_keyword_next(file, "WELLMOD") or
                kf.specific_keyword_next(file, "TIME")):
            break
        line = kf.strip_trailing_comment(file.readline())
        words = line.split()
        if len(words) == 0:
            break  # end of file
        assert len(words) >= len(columns_present), f"Insufficient data in line of wellspec table {well_name} [{line}]."
        if selecting:
            for col_index, col in enumerate(column_list):
                if column_map[col_index] < 0:
                    if column_list[col_index].upper() == "GRID":
                        data[col].append("ROOT")
                    else:
                        data[col].append(np.NaN)
                else:
                    value = words[column_map[col_index]]
                    if value == "NA":
                        data[col].append(np.NaN)
                    elif value == "#":
                        data[col].append(value)
                    elif value:
                        data[col].append(wellspec_dtype[col.upper()](value))
                if not pd.isnull(data[col][-1]):
                    all_null = False
        else:
            for col, value in zip(columns_present, words[:len(columns_present)]):
                if value == "NA":
                    data[col].append(np.NaN)
                elif value == "#":
                    data[col].append(value)
                elif value:
                    data[col].append(wellspec_dtype[col](value))
                if not pd.isnull(data[col][-1]):
                    all_null = False

    if all_null:
        log.warning(f"Null wellspec data for well {well_name}{f' at date {date}' if date is not None else ''}.")
        return None

    df = pd.DataFrame(data)

    if not keep_null_columns:
        df.drop(columns = df.columns[df.isna().all()], inplace = True)

    if not keep_duplicate_cells and any(df.duplicated(subset = ["IW", "JW", "L"])):
        log.warning(f"There are duplicate cells for well {well_name}.")
        df.drop_duplicates(subset = ["IW", "JW", "L"], keep = "last", inplace = True)

    def stat_tranformation(row):
        if row["STAT"] == "ON":
            return 1
        else:
            return 0

    if "STAT" in df.columns:
        df["STAT"] = df.apply(lambda row: stat_tranformation(row), axis = 1)

    return df


def get_all_well_data(
    file: TextIO,
    well_name: str,
    pointers: List[Tuple[int, Union[None, str]]],
    column_list: List[str] = [],
    selecting: bool = False,
    keep_duplicate_cells: bool = False,
    keep_null_columns: bool = True,
    last_data_only: bool = True,
) -> Union[pd.DataFrame, None]:
    """Creates a dataframe of all the well data for a given well name in the wellspec file.

    This differs from the get_well_data function in that here multiple datasets for a well are
    combined into a single dataframe if they exist.

    arguments:
        file (TextIO): the opened wellspec file object.
        well_name (str): name of the well.
        pointers (List[Tuple[int, None/str]]): a list of the file object's start position of the
            well data represented as number of bytes from the beginning of the file and the well's
            date. If no date existed before the well in the file, the date will be None.
        column_list (List[str]): if present, each dataframe returned contains these
            columns, in this order. If None, the resulting dictionary contains only well names as
            keys (each mapping to None rather than a dataframe). If an empty list (default), each
            dataframe contains the columns listed in the corresponding wellspec header, in the order
            found in the file.
        selecting (bool): True if the column_list contains at least one column name, False otherwise
            (default).
        keep_duplicate_cells (bool): if True (default), duplicate cells are kept, otherwise only the
            last entry is kept.
        keep_null_columns (bool): if True (default), columns that contain all NA values are kept,
            otherwise they are removed.
        last_data_only (bool): If True, only the last entry of well data in the file are used in the
            dataframe, otherwise all of the well data are used at different times.

    returns:
        Pandas dataframe of all well data for a specific well name or None if all the data are NA.
    """
    if last_data_only:
        if column_list is None:
            return None
        df = get_well_data(
            file,
            well_name,
            pointers[-1][0],
            column_list,
            selecting,
            keep_duplicate_cells,
            keep_null_columns,
            date = pointers[-1][1],
        )
        return df
    else:
        df_list = []
        for pointer, date in pointers:
            df = get_well_data(
                file,
                well_name,
                pointer,
                column_list,
                selecting,
                keep_duplicate_cells,
                date = date,
            )
            if df is None:
                continue

            df["DATE"] = date
            df_list.append(df)

        if len(df_list) == 0:
            return None

        df_combined = pd.concat(df_list, ignore_index = True)

        if not keep_null_columns:
            df_combined.drop(columns = df_combined.columns[df_combined.isna().all()], inplace = True)

        return df_combined
