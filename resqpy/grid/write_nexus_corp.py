"""A function for writing out a Nexus CORP file using supplied grid geometry"""

import logging

import numpy as np

log = logging.getLogger(__name__)

import resqpy.olio.grid_functions as gf
import resqpy.olio.xml_et as rqet
import resqpy.olio.trademark as tm
import resqpy.olio.write_data as wd


def write_nexus_corp(grid,
                     file_name,
                     local_coords = False,
                     global_xy_units = None,
                     global_z_units = None,
                     global_z_increasing_downward = True,
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
    crs_root = grid.extract_crs_root()
    if not local_coords:
        if not global_z_increasing_downward:
            log.warning('global z is not increasing with depth as expected by Nexus')
            tm.log_nexus_tm('warning')
        if crs_root is not None:  # todo: otherwise raise exception?
            log.info('converting corner points from local to global reference system')
            grid.local_to_global_crs(cp,
                                     crs_root,
                                     global_xy_units = global_xy_units,
                                     global_z_units = global_z_units,
                                     global_z_increasing_downward = global_z_increasing_downward)
    log.info('writing simulator corner point file ' + file_name)
    with open(file_name, 'w') as header:
        header.write('! Nexus corner point data written by resqml_grid module\n')
        header.write('! Nexus is a registered trademark of the Halliburton Company\n\n')
        if write_units_keyword:
            if local_coords:
                if crs_root is not None:
                    crs_xy_units_text = rqet.find_tag(crs_root, 'ProjectedUom').text
                    crs_z_units_text = rqet.find_tag(crs_root, 'VerticalUom').text
                    if crs_xy_units_text == 'm' and crs_z_units_text == 'm':
                        header.write('METRIC\n\n')
                    elif crs_xy_units_text == 'ft' and crs_z_units_text == 'ft':
                        header.write('ENGLISH\n\n')
                    else:
                        header.write('! local coordinates mixed (or not recognized)\n\n')
                else:
                    header.write('! local coordinates unknown\n\n')
            elif global_xy_units is not None and global_z_units is not None and global_xy_units == global_z_units:
                if global_xy_units in ['m', 'metre', 'metres']:
                    header.write('METRIC\n\n')
                elif global_xy_units in ['ft', 'feet', 'foot']:
                    header.write('ENGLISH\n\n')
                else:
                    header.write('! globsl coordinates not recognized\n\n')
            else:
                header.write('! global units unknown or mixed\n\n')
        if write_nx_ny_nz:
            header.write('NX      NY      NZ\n')
            header.write('{0:<7d} {1:<7d} {2:<7d}\n\n'.format(grid.extent_kji[2], grid.extent_kji[1],
                                                              grid.extent_kji[0]))
        if write_rh_keyword_if_needed:
            if ijk_right_handed is None or crs_root is None:
                log.warning('unable to determine whether RIGHTHANDED keyword is needed')
            else:
                xy_axes = rqet.find_tag(crs_root, 'ProjectedAxisOrder').text
                if local_coords:
                    z_inc_down = grid.z_inc_down()
                    if not z_inc_down:
                        log.warning('local z is not increasing with depth as expected by Nexus')
                        tm.log_nexus_tm('warning')
                else:
                    z_inc_down = global_z_increasing_downward
                xyz_handedness = rqet.xyz_handedness(xy_axes, z_inc_down)
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
