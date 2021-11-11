version = '10th November 2021'
import logging

log = logging.getLogger(__name__)
log.debug('well_functions.py version ' + version)

import os

import numpy as np
import pandas as pd

import resqpy.olio.wellspec_keywords as wsk
import resqpy.property as rqp

from .blocked_well import BlockedWell


def add_blocked_wells_from_wellspec(model, grid, wellspec_file):
    """Add a blocked well for each well in a Nexus WELLSPEC file.

    arguments:
       model (model.Model object): model to which blocked wells are added
       grid (grid.Grid object): grid against which wellspec data will be interpreted
       wellspec_file (string): path of ascii file holding Nexus WELLSPEC keyword and data

    returns:
       int: count of number of blocked wells created

    notes:
       this function appends to the hdf5 file and creates xml for the blocked wells (but does not store epc);
       'simulation' trajectory and measured depth datum objects will also be created
    """

    well_list_dict = wsk.load_wellspecs(wellspec_file, column_list = None)

    count = 0
    for well in well_list_dict:
        log.info('processing well: ' + str(well))
        bw = BlockedWell(model,
                         grid = grid,
                         wellspec_file = wellspec_file,
                         well_name = well,
                         check_grid_name = True,
                         use_face_centres = True)
        if not bw.node_count:  # failed to load from wellspec, eg. because of no perforations in grid
            log.warning('no wellspec data loaded for well: ' + str(well))
            continue
        bw.write_hdf5(model.h5_file_name(), mode = 'a', create_for_trajectory_if_needed = True)
        bw.create_xml(model.h5_uuid(), title = well)
        count += 1

    log.info(f'{count} blocked wells created based on wellspec file: {wellspec_file}')

def add_logs_from_cellio(blockedwell, cellio):
    """Creates a WellIntervalPropertyCollection for a given BlockedWell, using a given cell I/O file.

    Arguments:
       blockedwell: a resqml blockedwell object
       cellio: an ascii file exported from RMS containing blocked well geometry and logs. Must contain columns i_index, j_index and k_index, plus additional columns for logs to be imported.
    """
    # Get the initial variables from the blocked well
    assert isinstance(blockedwell, BlockedWell), 'Not a blocked wellbore object'
    collection = rqp.WellIntervalPropertyCollection(frame = blockedwell)
    well_name = blockedwell.trajectory.title.split(" ")[0]
    grid = blockedwell.model.grid()

    # Read the cell I/O file to get the available columns (cols) and the data (data), and write into a dataframe
    with open(cellio, 'r') as fp:
        wellfound = False
        cols, data = [], []
        for line in fp.readlines():
            if line == "\n":
                wellfound = False  # Blankline signifies end of well data
            words = line.split()
            if wellfound:
                if len(words) > 2 and not words[0].isdigit():
                    cols.append(line)
                else:
                    if len(words) > 9:
                        assert len(cols) == len(words), 'Number of columns found should match header of file'
                        data.append(words)
            if len(words) == 3:
                if words[0].upper() == well_name.upper():
                    wellfound = True
        assert len(data) > 0 and len(cols) > 3, f"No data for well {well_name} found in file"
        df = pd.DataFrame(data = data, columns = [x.split()[0] for x in cols])
        df = df.apply(pd.to_numeric)
        # Get the cell_indices from the grid for the given i/j/k
        df['cell_indices'] = grid.natural_cell_indices(
            np.array((df['k_index'] - 1, df['j_index'] - 1, df['i_index'] - 1), dtype = int).T)
        df = df.drop(['i_index', 'j_index', 'k_index', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'], axis = 1)
    assert (df['cell_indices'] == blockedwell.cell_indices
           ).all(), 'Cell indices do not match between blocked well and log inputs'

    # Work out if the data columns are continuous, categorical or discrete
    type_dict = {}
    lookup_dict = {}
    for col in cols:
        words = col.split()
        if words[0] not in ['i_index', 'j_index', 'k_index', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out']:
            if words[1] == 'unit1':
                type_dict[words[0]] = 'continuous'
            elif words[1] == 'DISC' and not words[0] == 'ZONES':
                type_dict[words[0]] = 'categorical'
                lookup_dict[words[0]] = lookup_from_cellio(col, blockedwell.model)
            elif words[1] == 'param' or words[0] == 'ZONES':
                type_dict[words[0]] = 'discrete'
            else:
                raise TypeError(f'unrecognised data type for {col}')

    # Loop over the columns, adding them to the blockedwell property collection
    for log in df.columns:
        if log not in ['cell_indices']:
            data_type = type_dict[log]
            if log == 'ZONES':
                data_type, dtype, null, discrete = 'discrete', int, -1, True
            elif data_type == 'continuous':
                dtype, null, discrete = float, np.nan, False
            else:
                dtype, null, discrete = int, -1, True
            if data_type == 'categorical':
                lookup_uuid = lookup_dict[log]  # For categorical data, find or generate a StringLookupTable
            else:
                lookup_uuid = None
            array_list = np.zeros((np.shape(blockedwell.grid_indices)), dtype = dtype)
            vals = list(df[log])
            for i, index in enumerate(blockedwell.cell_grid_link):
                if index == -1:
                    assert blockedwell.grid_indices[i] == -1
                    array_list[i] = null
                else:
                    if blockedwell.cell_indices[index] == list(df['cell_indices'])[index]:
                        array_list[i] = vals[index]
            collection.add_cached_array_to_imported_list(
                cached_array = array_list,
                source_info = '',
                keyword = f"{os.path.basename(cellio).split('.')[0]}.{blockedwell.trajectory.title}.{log}",
                discrete = discrete,
                uom = None,
                property_kind = None,
                facet = None,
                null_value = null,
                facet_type = None,
                realization = None)
            collection.write_hdf5_for_imported_list()
            collection.create_xml_for_imported_list_and_add_parts_to_model(string_lookup_uuid = lookup_uuid)

def lookup_from_cellio(line, model):
    """Create a StringLookup Object from a cell I/O row containing a categorical column name and details.

    Arguments:
       line: a string from a cell I/O file, containing the column (log) name, type and categorical information
       model: the model to add the StringTableLookup to
    Returns:
       uuid: the uuid of a StringTableLookup, either for a newly created table, or for an existing table if an identical one exists
    """
    lookup_dict = {}
    value, string = None, None
    # Generate a dictionary of values and strings
    for i, word in enumerate(line.split()):
        if i == 0:
            title = word
        elif not i < 2:
            if value is not None and string is not None:
                lookup_dict[value] = string
                value, string = None, None
            if value is None:
                value = int(word)
            else:
                if i == len(line.split()) - 1:
                    lookup_dict[value] = word
                else:
                    string = word

    # Check if a StringLookupTable already exists in the model, with the same name and values
    for existing in model.parts_list_of_type('obj_StringTableLookup'):
        table = rqp.StringLookup(parent_model = model, root_node = model.root_for_part(existing))
        if table.title == title:
            if table.str_dict == lookup_dict:
                return table.uuid  # If the exact table exists, reuse it by returning the uuid

    # If no matching StringLookupTable exists, make a new one and return the uuid
    lookup = rqp.StringLookup(parent_model = model, int_to_str_dict = lookup_dict, title = title)
    lookup.create_xml(add_as_part = True)
    return lookup.uuid