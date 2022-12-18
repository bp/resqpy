"""Array writing functions."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

# only nexus format currently fully supported

import math as maths
import numpy as np

import resqpy.olio.ab_toolbox as abt


def write_pure_binary_data(binary_file_name, numpy_array):
    """Writes a numpy array to a file in 'pure binary' format."""

    with open(binary_file_name, 'wb') as binary_file_out:
        numpy_array.tofile(binary_file_out)
    log.info('Binary data file %s created', binary_file_name)


def write_array_to_ascii_file(file_name,
                              extent_kji,
                              a,
                              headers = True,
                              keyword = None,
                              columns = 20,
                              data_type = 'real',
                              decimals = 3,
                              target_simulator = 'nexus',
                              blank_line_after_i_block = True,
                              blank_line_after_j_block = False,
                              space_separated = False,
                              append = False,
                              use_binary = False,
                              binary_only = False,
                              nan_substitute_value = None):
    """Writes a 3D array of data to an ascii file."""

    assert columns > 0 and decimals >= 0  # todo: test behaviour when decimals = 0
    assert target_simulator == 'nexus'  # no other simulator formats supported at the moment

    if not (use_binary and binary_only):

        format_str = ''
        if data_type == 'real' or data_type == 'float':
            format_str = '{0:.' + str(decimals) + 'f}'

        try:

            if append:
                file_mode = 'a'
            else:
                file_mode = 'w'

            with open(file_name, file_mode) as new_file:

                if headers:  # write 3 comment lines at start of file
                    new_file.write('! Data written by write_array_to_ascii_file() python function\n')
                    new_file.write('! Extent of array is: [' + str(extent_kji[2]) + ', ' + str(extent_kji[1]) + ', ' +
                                   str(extent_kji[0]) + ']\n')
                    new_file.write('! Maximum ' + str(columns) + ' data items per line\n')

                if keyword is not None:
                    new_file.write(keyword + '\n')

                for k in range(extent_kji[0]):
                    for j in range(extent_kji[1]):
                        for i in range(extent_kji[2]):
                            if (i % columns == 0):
                                new_file.write('\n')
                            elif space_separated:
                                new_file.write(' ')
                            else:
                                new_file.write('\t')
                            if data_type == 'real' or data_type == 'float':  # todo: speed up by using common write, different format
                                v = a[k, j, i]
                                if nan_substitute_value is not None and maths.isnan(v):
                                    v = nan_substitute_value
                                new_file.write(format_str.format(v))
                            elif data_type == 'bool' or data_type == 'boolean':
                                if a[k, j, i]:
                                    new_file.write('1')
                                else:
                                    new_file.write('0')
                            else:
                                new_file.write(str(a[k, j, i]))  # todo: other formatting
                        if blank_line_after_i_block:
                            new_file.write('\n')
                    if blank_line_after_j_block:
                        new_file.write('\n')

                if not (blank_line_after_i_block or blank_line_after_j_block):
                    new_file.write('\n')

            log.info('Ascii data file %s created', file_name)

        except Exception:
            log.error('Failed to write data to ascii file %s', file_name)
            # could abort at this point, or raise

    if use_binary:
        extension, _ = abt.binary_file_extension_and_np_type_for_data_type(data_type)
        binary_file_name = file_name + extension
        if append and (keyword is not None):
            key = keyword.split()[0].lower()
            if key != 'corp':
                binary_file_name = file_name + '_' + key + extension
        if nan_substitute_value is None:
            ap = a
        else:
            ap = np.where(np.isnan(a), nan_substitute_value, a)
        try:
            with open(binary_file_name, 'wb') as binary_file_out:
                ap.tofile(binary_file_out)
            log.info('Binary data file %s created', binary_file_name)
        except Exception:
            log.warning('Failed to write data to binary file %s', binary_file_name)
            # todo: could delete the binary file in case a corrupt file is left for use next time
