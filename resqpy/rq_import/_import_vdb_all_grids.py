"""_import_vdb_all_grids.py: Module to import a vdb into resqml format."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import resqpy.olio.vdb as vdb
import resqpy.rq_import as rqi


def import_vdb_all_grids(
    resqml_file_root,  # output path and file name without .epc or .h5 extension
    extent_ijk = None,  # 3 element numpy vector applicable to ROOT
    vdb_file = None,
    vdb_case = None,  # if None, first case in vdb is used (usually a vdb only holds one case)
    corp_xy_units = 'm',
    corp_z_units = 'm',
    corp_z_inc_down = True,
    ijk_handedness = 'right',
    geometry_defined_everywhere = True,
    treat_as_nan = None,
    resqml_xy_units = 'm',
    resqml_z_units = 'm',
    resqml_z_inc_down = True,
    shift_to_local = False,
    local_origin_place = 'centre',  # 'centre' or 'minimum'
    max_z_void = 0.1,  # vertical gaps greater than this will introduce k gaps intp resqml grid
    split_pillars = True,
    split_tolerance = 0.01,  # applies to each of x, y, z differences
    vdb_static_properties = True,
    # if True, static vdb properties are imported (only relevant if vdb_file is not None)
    vdb_recurrent_properties = False,
    decoarsen = True,
    timestep_selection = 'all',
    # 'first', 'last', 'first and last', 'all', or list of ints being reporting timestep numbers
    create_property_set = False):
    """Creates a RESQML dataset containing grids and grid properties, including LGRs, for a single realisation.

    Arguments:
        resqml_file_root (str): output path and file name without .epc or .h5 extension
        extent_ijk (triple float, optional): ijk extents (fortran ordering)
        vdb_file (str, optional): vdb input file, either this or corp_file should be not None. Required if importing from a vdb
        vdb_case (str, optional): required if the vdb contains more than one case. If None, first case in vdb is used
        corp_xy_units (str, default 'm'): xy length units
        corp_z_units (str, default 'm'): z length units
        corp_z_inc_down (bool, default True): if True z values increase with depth
        ijk_handedness (str, default 'right'): 'right' or 'left'
        geometry_defined_everywhere (bool, default True): if False then inactive cells are marked as not having geometry
        treat_as_nan (float, default None): if a value is provided corner points with this value will be assigned nan
        resqml_xy_units (str, default 'm'): output xy units for resqml file
        resqml_z_units (str, default 'm'): output z units for resqml file
        resqml_z_inc_down (bool, default True): if True z values increase with depth for output resqml file
        shift_to_local (bool, default False): if True then a local origin will be used in the CRS
        local_origin_place (str, default 'centre'): 'centre' or 'minimum'. If 'centre' the local origin is placed at the centre of the grid; ignored if shift_to_local is False
        max_z_void (float, default 0.1): maximum z gap between vertically neighbouring corner points. Vertical gaps greater than this will introduce k gaps into resqml grid. Units are corp z units
        split_pillars (bool, default True): if False an unfaulted grid will be generated
        split_tolerance (float, default 0.01): maximum distance between neighbouring corner points before a pillar is considered 'split'. Applies to each of x, y, z differences
        vdb_static_properties (bool, default True): if True, static vdb properties are imported (only relevant if vdb_file is not None)
        vdb_recurrent_properties (bool, default False): # if True, recurrent vdb properties are imported (only relevant if vdb_file is not None)
        decoarsen (bool, default True): where ICOARSE is present, redistribute data to uncoarse cells
        timestep_selection (str, default 'all): 'first', 'last', 'first and last', 'all', or list of ints being reporting timestep numbers. Ignored if vdb_recurrent_properties is False
        create_property_set (bool, default False): if True a resqml PropertySet is created
    """

    vdbase = vdb.VDB(vdb_file)
    case_list = vdbase.cases()
    assert len(case_list) > 0, 'no cases found in vdb'
    if vdb_case is None:
        vdb_case = case_list[0]
    else:
        assert vdb_case in case_list, 'case ' + vdb_case + ' not found in vdb: ' + vdb_file
        vdbase.set_use_case(vdb_case)
    grid_list = vdbase.list_of_grids()
    index = 0
    for grid_name in grid_list:
        if grid_name.upper().startswith('SMALLGRIDS'):
            log.warning('vdb import skipping small grids')
            continue
        log.debug('importing vdb data for grid ' + str(grid_name))
        rqi.import_nexus(
            resqml_file_root,
            extent_ijk = extent_ijk if grid_name == 'ROOT' else None,  # 3 element numpy vector applicable to ROOT
            vdb_file = vdb_file,
            vdb_case = vdb_case,  # if None, first case in vdb is used (usually a vdb only holds one case)
            corp_xy_units = corp_xy_units,
            corp_z_units = corp_z_units,
            corp_z_inc_down = corp_z_inc_down,
            ijk_handedness = ijk_handedness,
            geometry_defined_everywhere = geometry_defined_everywhere,
            treat_as_nan = treat_as_nan,
            resqml_xy_units = resqml_xy_units,
            resqml_z_units = resqml_z_units,
            resqml_z_inc_down = resqml_z_inc_down,
            shift_to_local = shift_to_local,
            local_origin_place = local_origin_place,  # 'centre' or 'minimum'
            max_z_void = max_z_void,  # vertical gaps greater than this will introduce k gaps intp resqml grid
            split_pillars = split_pillars,  # NB: some LGRs may be unsplit even if ROOT is split
            split_tolerance = split_tolerance,  # applies to each of x, y, z differences
            vdb_static_properties = vdb_static_properties,  # if True, static vdb properties are imported
            vdb_recurrent_properties = vdb_recurrent_properties,
            decoarsen = decoarsen,
            timestep_selection = timestep_selection,
            create_property_set = create_property_set,
            grid_title = grid_name,
            mode = 'w' if index == 0 else 'a')
        index += 1
