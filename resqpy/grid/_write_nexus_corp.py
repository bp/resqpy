"""A function for writing out a Nexus CORP file using supplied grid geometry"""

# Nexus is a trademark of Halliburton

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.grid_functions as gf
import resqpy.olio.xml_et as rqet
import resqpy.olio.trademark as tm
import resqpy.olio.write_data as wd
import resqpy.crs as rqc
import resqpy.weights_and_measures as wam


def write_nexus_corp(grid,
                     file_name,
                     local_coords = False,
                     global_xy_units = None,
                     global_z_units = None,
                     global_z_increasing_downward = True,
                     nexus_unit_system = None,
                     write_nx_ny_nz = False,
                     write_units_keyword = False,
                     write_rh_keyword_if_needed = False,
                     write_corp_keyword = False,
                     use_binary = False,
                     binary_only = False,
                     nan_substitute_value = None):
    """Write grid geometry to file in Nexus CORP ordering."""

    log.info('caching Nexus corner points')
    tm.log_nexus_tm('info')
    grid.corner_points(cache_cp_array = True)
    log.debug('duplicating Nexus corner points')
    cp = grid.array_corner_points.copy()
    log.debug('resequencing duplicated Nexus corner points')
    gf.resequence_nexus_corp(cp, eight_mode = False, undo = True)
    corp_extent = np.zeros(3, dtype = 'int')
    corp_extent[0] = grid.cell_count()  # total number of cells in grid
    corp_extent[1] = 8  # 8 corners of cell: k -/+; j -/+; i -/+
    corp_extent[2] = 3  # x, y, z
    ijk_right_handed = grid.extract_grid_is_right_handed()
    if ijk_right_handed is None:
        log.warning('ijk handedness not known')
    elif not ijk_right_handed:
        log.warning('ijk axes are left handed; inverted (fake) xyz handedness required')
    assert grid.crs is not None

    if local_coords:
        source_xy_units = grid.crs.xy_units
        source_z_units = grid.crs.z_units
    else:
        if not global_xy_units:
            global_xy_units = grid.crs.xy_units
        if not global_z_units:
            global_z_units = grid.crs.z_units
        if not global_z_increasing_downward:
            log.warning('global z is not increasing with depth as expected by Nexus')
            tm.log_nexus_tm('warning')
        log.info('converting corner points from local to global reference system')
        cp = grid.local_to_global_crs(cp,
                                      crs_uuid = grid.crs.uuid,
                                      global_xy_units = global_xy_units,
                                      global_z_units = global_z_units,
                                      global_z_increasing_downward = global_z_increasing_downward)
        source_xy_units = global_xy_units
        source_z_units = global_z_units

    if not nexus_unit_system:
        guessing_units = grid.crs.z_units if local_coords else global_z_units
        if not guessing_units:
            guessing_units = 'm'
        if guessing_units.startswith('ft'):
            nexus_unit_system = 'ENGLISH'
        elif guessing_units in ['cm', 'mm', 'nm']:
            nexus_unit_system = 'LAB'
        else:
            nexus_unit_system = 'METRIC'
    assert nexus_unit_system in ['METRIC', 'METBAR', 'METKG/CM2', 'ENGLISH', 'LAB']

    if nexus_unit_system.startswith('MET'):
        target_uom = 'm'
    elif nexus_unit_system == 'LAB':
        target_uom = 'cm'
    else:
        target_uom = 'ft'
    wam.convert_lengths(cp[..., :2], source_xy_units, target_uom)
    wam.convert_lengths(cp[..., 2], source_z_units, target_uom)

    log.info(f'writing simulator corner point file: {file_name}; Nexus unit system: {nexus_unit_system}')
    with open(file_name, 'w') as header:
        header.write('! Nexus corner point data written by resqml_grid module\n')
        header.write('! Nexus is a registered trademark of the Halliburton Company\n\n')
        if write_units_keyword:
            header.write(f'{nexus_unit_system}\n\n')
        if write_nx_ny_nz:
            header.write('NX      NY      NZ\n')
            header.write('{0:<7d} {1:<7d} {2:<7d}\n\n'.format(grid.extent_kji[2], grid.extent_kji[1],
                                                              grid.extent_kji[0]))
        if write_rh_keyword_if_needed:
            if ijk_right_handed is None:
                log.warning('unable to determine whether RIGHTHANDED keyword is needed')
            else:
                if local_coords:
                    z_inc_down = grid.z_inc_down()
                    if not z_inc_down:
                        log.warning('local z is not increasing with depth as expected by Nexus')
                        tm.log_nexus_tm('warning')
                else:
                    z_inc_down = global_z_increasing_downward
                xyz_handedness = rqet.xyz_handedness(grid.crs.axis_order, z_inc_down)
                if xyz_handedness == 'unknown':
                    log.warning(
                        'xyz handedness is not known; unable to determine whether RIGHTHANDED keyword is needed')
                else:
                    if ijk_right_handed == (xyz_handedness == 'right'):  # if either both True or both False
                        header.write('RIGHTHANDED\n\n')

    if write_corp_keyword:
        keyword = 'CORP VALUE'
    else:
        keyword = None

    wd.write_array_to_ascii_file(file_name,
                                 corp_extent,
                                 cp.reshape(tuple(corp_extent)),
                                 target_simulator = 'nexus',
                                 keyword = keyword,
                                 columns = 3,
                                 blank_line_after_i_block = False,
                                 blank_line_after_j_block = True,
                                 append = True,
                                 use_binary = use_binary,
                                 binary_only = binary_only,
                                 nan_substitute_value = nan_substitute_value)
