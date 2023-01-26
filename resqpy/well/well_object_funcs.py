"""well_object_funcs.py: resqpy well module for functions that impact well objects"""

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import warnings
import os
import lasio
import numpy as np
import pandas as pd

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.property as rqp
import resqpy.weights_and_measures as bwam
import resqpy.well as rqw
import resqpy.olio.wellspec_keywords as wsk


def add_wells_from_ascii_file(model,
                              crs_uuid,
                              trajectory_file,
                              comment_character = '#',
                              space_separated_instead_of_csv = False,
                              well_col = 'WELL',
                              md_col = 'MD',
                              x_col = 'X',
                              y_col = 'Y',
                              z_col = 'Z',
                              length_uom = 'm',
                              md_domain = None,
                              drilled = False):
    """Creates new md datum, trajectory, interpretation and feature objects for each well in an ascii file.

    arguments:
       crs_uuid (uuid.UUID): the unique identifier of the coordinate reference system applicable to the x,y,z data;
          if None, a default crs will be created, making use of the length_uom and z_inc_down arguments
       trajectory_file (string): the path of the ascii file holding the well trajectory data to be loaded
       comment_character (string, default '#'): character deemed to introduce a comment in the trajectory file
       space_separated_instead_of_csv (boolean, default False): if True, the columns in the trajectory file are space
          separated; if False, comma separated
       well_col (string, default 'WELL'): the heading for the column containing well names
       md_col (string, default 'MD'): the heading for the column containing measured depths
       x_col (string, default 'X'): the heading for the column containing X (usually easting) data
       y_col (string, default 'Y'): the heading for the column containing Y (usually northing) data
       z_col (string, default 'Z'): the heading for the column containing Z (depth or elevation) data
       length_uom (string, default 'm'): the units of measure for the measured depths; should be 'm' or 'ft'
       md_domain (string, optional): the source of the original deviation data; may be 'logger' or 'driller'
       drilled (boolean, default False): True should be used for wells that have been drilled; False otherwise (planned,
          proposed, or a location being studied)
       z_inc_down (boolean, default True): indicates whether z values increase with depth; only used in the creation
          of a default coordinate reference system; ignored if crs_uuid is not None

    returns:
       tuple of lists of objects: (feature_list, interpretation_list, trajectory_list, md_datum_list)

    notes:
       ascii file must be table with first line being column headers, with columns for WELL, MD, X, Y & Z;
       actual column names can be set with optional arguments;
       all the objects are added to the model, with array data being written to the hdf5 file for the trajectories;
       the md_domain and drilled values are stored in the RESQML metadata but are only for human information and do not
       generally affect computations
    """

    assert md_col and x_col and y_col and z_col
    md_col = str(md_col)
    x_col = str(x_col)
    y_col = str(y_col)
    z_col = str(z_col)
    if crs_uuid is None:
        crs_uuid = model.crs_uuid
    assert crs_uuid is not None, 'coordinate reference system not found when trying to add wells'

    try:
        df = pd.read_csv(trajectory_file,
                         comment = comment_character,
                         delim_whitespace = space_separated_instead_of_csv)
        if df is None:
            raise Exception
    except Exception:
        log.error('failed to read ascii deviation survey file: ' + str(trajectory_file))
        raise
    if well_col and well_col not in df.columns:
        log.warning('well column ' + str(well_col) + ' not found in ascii trajectory file: ' + str(trajectory_file))
        well_col = None
    if well_col is None:
        for col in df.columns:
            if str(col).upper().startswith('WELL'):
                well_col = str(col)
                break
    else:
        well_col = str(well_col)
    assert well_col
    unique_wells = set(df[well_col])
    if len(unique_wells) == 0:
        log.warning('no well data found in ascii trajectory file: ' + str(trajectory_file))
        # note: empty lists will be returned, below

    feature_list = []
    interpretation_list = []
    trajectory_list = []
    md_datum_list = []

    for well_name in unique_wells:

        log.debug('importing well: ' + str(well_name))
        # create single well data frame (assumes measured depths increasing)
        well_df = df[df[well_col] == well_name]
        # create a measured depth datum for the well and add as part
        first_row = well_df.iloc[0]
        if first_row[md_col] == 0.0:
            md_datum = rqw.MdDatum(model,
                                   crs_uuid = crs_uuid,
                                   location = (first_row[x_col], first_row[y_col], first_row[z_col]))
        else:
            md_datum = rqw.MdDatum(model, crs_uuid = crs_uuid,
                                   location = (first_row[x_col], first_row[y_col], 0.0))  # sea level datum
        md_datum.create_xml(title = str(well_name))
        md_datum_list.append(md_datum)

        # create a well feature and add as part
        feature = rqo.WellboreFeature(model, feature_name = well_name)
        feature.create_xml()
        feature_list.append(feature)

        # create interpretation and add as part
        interpretation = rqo.WellboreInterpretation(model, is_drilled = drilled, wellbore_feature = feature)
        interpretation.create_xml(title_suffix = None)
        interpretation_list.append(interpretation)

        # create trajectory, write arrays to hdf5 and add as part
        trajectory = rqw.Trajectory(model,
                                    md_datum = md_datum,
                                    data_frame = well_df,
                                    length_uom = length_uom,
                                    md_domain = md_domain,
                                    represented_interp = interpretation,
                                    well_name = well_name)
        trajectory.write_hdf5()
        trajectory.create_xml(title = well_name)
        trajectory_list.append(trajectory)

    return (feature_list, interpretation_list, trajectory_list, md_datum_list)


def well_name(well_object, model = None):
    """Returns the 'best' citation title from the object or related well objects.

    arguments:
       well_object (object, uuid or root): Object for which a well name is required. Can be a
          Trajectory, WellboreInterpretation, WellboreFeature, BlockedWell, WellboreMarkerFrame,
          WellboreFrame, DeviationSurvey or MdDatum object
       model (model.Model, optional): required if passing a uuid or root; not recommended otherwise

    returns:
       string being the 'best' citation title to serve as a well name, form the object or some related objects

    note:
       xml and relationships must be established for this function to work
    """

    def better_root(model, root_a, root_b):
        a = rqet.citation_title_for_node(root_a)
        b = rqet.citation_title_for_node(root_b)
        if a is None or len(a) == 0:
            return root_b
        if b is None or len(b) == 0:
            return root_a
        parts_like_a = model.parts(title = a)
        parts_like_b = model.parts(title = b)
        if len(parts_like_a) > 1 and len(parts_like_b) == 1:
            return root_b
        elif len(parts_like_b) > 1 and len(parts_like_a) == 1:
            return root_a
        a_digits = 0
        for c in a:
            if c.isdigit():
                a_digits += 1
        b_digits = 0
        for c in b:
            if c.isdigit():
                b_digits += 1
        if a_digits < b_digits:
            return root_b
        return root_a

    def best_root(model, roots_list):
        if len(roots_list) == 0:
            return None
        if len(roots_list) == 1:
            return roots_list[0]
        if len(roots_list) == 2:
            return better_root(model, roots_list[0], roots_list[1])
        return better_root(model, roots_list[0], best_root(model, roots_list[1:]))

    def best_root_for_object(well_object, model = None):

        if well_object is None:
            return None
        if model is None:
            model = well_object.model
        root_list = []
        obj_root = None
        obj_uuid = None
        obj_type = None
        traj_root = None

        if isinstance(well_object, str):
            obj_uuid = bu.uuid_from_string(well_object)
            assert obj_uuid is not None, 'well_name string argument could not be interpreted as uuid'
            well_object = obj_uuid
        if isinstance(well_object, bu.uuid.UUID):
            obj_uuid = well_object
            obj_root = model.root_for_uuid(obj_uuid)
            assert obj_root is not None, 'uuid not found in model when looking for well name'
            obj_type = rqet.node_type(obj_root)
        elif rqet.is_node(well_object):
            obj_root = well_object
            obj_type = rqet.node_type(obj_root)
            obj_uuid = rqet.uuid_for_part_root(obj_root)
        elif isinstance(well_object, rqw.Trajectory):
            obj_type = 'WellboreTrajectoryRepresentation'
            traj_root = well_object.root
        elif isinstance(well_object, rqo.WellboreFeature):
            obj_type = 'WellboreFeature'
        elif isinstance(well_object, rqo.WellboreInterpretation):
            obj_type = 'WellboreInterpretation'
        elif isinstance(well_object, rqw.BlockedWell):
            obj_type = 'BlockedWellboreRepresentation'
            if well_object.trajectory is not None:
                traj_root = well_object.trajectory.root
        elif isinstance(well_object, rqw.WellboreMarkerFrame):  # note: trajectory might be None
            obj_type = 'WellboreMarkerFrameRepresentation'
            if well_object.trajectory is not None:
                traj_root = well_object.trajectory.root
        elif isinstance(well_object, rqw.WellboreFrame):  # note: trajectory might be None
            obj_type = 'WellboreFrameRepresentation'
            if well_object.trajectory is not None:
                traj_root = well_object.trajectory.root
        elif isinstance(well_object, rqw.DeviationSurvey):
            obj_type = 'DeviationSurveyRepresentation'
        elif isinstance(well_object, rqw.MdDatum):
            obj_type = 'MdDatum'

        assert obj_type is not None, 'argument type not recognized for well_name'
        if obj_type.startswith('obj_'):
            obj_type = obj_type[4:]
        if obj_uuid is None:
            obj_uuid = well_object.uuid
            obj_root = model.root_for_uuid(obj_uuid)

        if obj_type == 'WellboreFeature':
            interp_parts = model.parts(obj_type = 'WellboreInterpretation')
            interp_parts = model.parts_list_filtered_by_related_uuid(interp_parts, obj_uuid)
            all_parts = interp_parts
            all_traj_parts = model.parts(obj_type = 'WellboreTrajectoryRepresentation')
            if interp_parts is not None:
                for part in interp_parts:
                    traj_parts = model.parts_list_filtered_by_related_uuid(all_traj_parts, model.uuid_for_part(part))
                    all_parts += traj_parts
            if all_parts is not None:
                root_list = [model.root_for_part(part) for part in all_parts]
        elif obj_type == 'WellboreInterpretation':
            feat_roots = model.roots(obj_type = 'WellboreFeature', related_uuid = obj_uuid)  # should return one root
            traj_roots = model.roots(obj_type = 'WellboreTrajectoryRepresentation', related_uuid = obj_uuid)
            root_list = feat_roots + traj_roots
        elif obj_type == 'WellboreTrajectoryRepresentation':
            interp_parts = model.parts(obj_type = 'WellboreInterpretation')
            interp_parts = model.parts_list_filtered_by_related_uuid(interp_parts, obj_uuid)
            all_parts = interp_parts
            all_feat_parts = model.parts(obj_type = 'WellboreFeature')
            if interp_parts is not None:
                for part in interp_parts:
                    feat_parts = model.parts_list_filtered_by_related_uuid(all_feat_parts, model.uuid_for_part(part))
                    all_parts += feat_parts
            if all_parts is not None:
                root_list = [model.root_for_part(part) for part in all_parts]
        elif obj_type in [
                'BlockedWellboreRepresentation', 'WellboreMarkerFrameRepresentation', 'WellboreFrameRepresentation'
        ]:
            if traj_root is None:
                traj_root = model.root(obj_type = 'WellboreTrajectoryRepresentation', related_uuid = obj_uuid)
            root_list = [best_root_for_object(traj_root, model = model)]
        elif obj_type == 'DeviationSurveyRepresentation':
            root_list = [best_root_for_object(model.root(obj_type = 'MdDatum', related_uuid = obj_uuid), model = model)]
        elif obj_type == 'MdDatum':
            pass

        root_list.append(obj_root)

        return best_root(model, root_list)

    title = rqet.citation_title_for_node(best_root_for_object(well_object, model = model))
    return 'WELL' if not title else title


def add_las_to_trajectory(las: lasio.LASFile, trajectory, realization = None, check_well_name = False):
    """Creates a WellLogCollection and WellboreFrame from a LAS file.

    arguments:
       las: an lasio.LASFile object
       trajectory: an instance of :class:`resqpy.well.Trajectory` .
       realization (integer): if present, the single realisation (within an ensemble)
          that this collection is for
       check_well_name (bool): if True, raise warning if LAS well name does not match
          existing wellborefeature citation title

    returns:
       collection, well_frame: instances of :class:`resqpy.property.WellLogCollection`
          and :class:`resqpy.well.WellboreFrame`

    note:
       in this current implementation, the first curve in the las object must be
       Measured Depths, not e.g. TVDSS
    """

    # Lookup relevant related resqml parts
    model = trajectory.model
    well_interp = trajectory.wellbore_interpretation
    well_title = well_interp.title

    if check_well_name and well_title != las.well.WELL.value:
        warnings.warn(f'LAS well title {las.well.WELL.value} does not match resqml tite {well_title}')

    # Create a new wellbore frame, using depth data from first curve in las file
    depth_values = np.array(las.index).copy()
    assert isinstance(depth_values, np.ndarray)
    las_depth_uom = bwam.rq_length_unit(las.curves[0].unit)

    # Ensure depth units are correct
    bwam.convert_lengths(depth_values, from_units = las_depth_uom, to_units = trajectory.md_uom)
    assert len(depth_values) > 0

    well_frame = rqw.WellboreFrame(
        parent_model = model,
        trajectory = trajectory,
        mds = depth_values,
        represented_interp = well_interp,
    )
    well_frame.write_hdf5()
    well_frame.create_xml()

    # Create a WellLogCollection in which to put logs
    collection = rqp.WellLogCollection(frame = well_frame, realization = realization)

    # Read in data from each curve in turn (skipping first curve which has depths)
    for curve in las.curves[1:]:

        collection.add_log(
            title = curve.mnemonic,
            data = curve.data,
            unit = curve.unit,
            realization = realization,
            write = False,
        )
        collection.write_hdf5_for_imported_list()
        collection.create_xml_for_imported_list_and_add_parts_to_model()

    return collection, well_frame


def add_blocked_wells_from_wellspec(model, grid, wellspec_file, usa_date_format = False):
    """Add a blocked well for each well in a Nexus WELLSPEC file.

    arguments:
       model (model.Model object): model to which blocked wells are added
       grid (grid.Grid object): grid against which wellspec data will be interpreted
       wellspec_file (string): path of ascii file holding Nexus WELLSPEC keyword and data
       usa_date_format (bool): mm/dd/yyyy (True) vs. dd/mm/yyyy (False)

    returns:
       int: count of number of blocked wells created

    notes:
       this function appends to the hdf5 file and creates xml for the blocked wells (but does not store epc);
       'simulation' trajectory and measured depth datum objects will also be created
    """

    well_list_dict = wsk.load_wellspecs(wellspec_file, column_list = None, usa_date_format = usa_date_format)

    count = 0
    for well in well_list_dict:
        log.info('processing well: ' + str(well))
        bw = rqw.BlockedWell(model,
                             grid = grid,
                             wellspec_file = wellspec_file,
                             well_name = well,
                             check_grid_name = True,
                             use_face_centres = True,
                             usa_date_format = usa_date_format)
        if not bw.node_count:  # failed to load from wellspec, eg. because of no perforations in grid
            log.warning('no wellspec data loaded for well: ' + str(well))
            continue
        bw.write_hdf5(model.h5_file_name(), mode = 'a', create_for_trajectory_if_needed = True)
        bw.create_xml(model.h5_uuid(), title = well)
        count += 1

    log.info(f'{count} blocked wells created based on wellspec file: {wellspec_file}')

    return count


def add_logs_from_cellio(blockedwell, cellio):
    """Creates a WellIntervalPropertyCollection for a given BlockedWell, using a given cell I/O file.

    arguments:
       blockedwell: a resqml blockedwell object
       cellio: an ascii file exported from RMS containing blocked well geometry and logs;
           must contain columns i_index, j_index and k_index, plus additional columns for logs to be imported
    """
    # Get the initial variables from the blocked well
    assert isinstance(blockedwell, rqw.BlockedWell), 'Not a blocked wellbore object'
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

    arguments:
       line: a string from a cell I/O file, containing the column (log) name, type and categorical information
       model: the model to add the StringTableLookup to

    returns:
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
    for existing_uuid in model.uuids(obj_type = 'StringTableLookup'):
        table = rqp.StringLookup(parent_model = model, uuid = existing_uuid)
        if table.title == title:
            if table.str_dict == lookup_dict:
                return table.uuid  # If the exact table exists, reuse it by returning the uuid

    # If no matching StringLookupTable exists, make a new one and return the uuid
    lookup = rqp.StringLookup(parent_model = model, int_to_str_dict = lookup_dict, title = title)
    lookup.create_xml(add_as_part = True)
    return lookup.uuid
