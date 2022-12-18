"""Functions to load data from various ASCII simulator file formats."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.olio.ab_toolbox as abt
import resqpy.olio.box_utilities as box
import resqpy.olio.grid_functions as gf
import resqpy.olio.keyword_files as kf
import resqpy.olio.write_data as wd


def file_exists(file_name, must_be_more_recent_than_file = None):
    """Returns True if the file exists (and is more recent than other file, if given)."""

    if file_name is None or len(file_name) == 0:
        return False
    existe = os.path.exists(file_name)
    if not existe:
        return False
    if not must_be_more_recent_than_file or file_name == must_be_more_recent_than_file or \
            not os.path.exists(must_be_more_recent_than_file):
        return True
    return os.path.getmtime(file_name) > os.path.getmtime(must_be_more_recent_than_file)


######################################################################################################
# load_corp_array_from_file():
# function to load a nexus corp array of data from a named file
# if the grid extent is not known, the file must be free of comments
# NB: extent_kji here is extent of grid, rather than that of the corp array
# returns a pagoda style 7D array, resequenced


def load_corp_array_from_file(file_name,
                              extent_kji = None,
                              corp_bin = False,
                              swap_bytes = True,
                              max_lines_for_keyword = 100,
                              comment_char = None,
                              data_free_of_comments = False,
                              use_binary = False,
                              eight_mode = False,
                              use_numbers_only = None):
    """Loads a nexus corner point (CORP) array from a file, returns a 7D numpy array in pagoda ordering.

    arguments:
       file_name: The name of an ascii file holding the CORP data (no other keywords with numeric data
                should be in the file); write access to the directory is likely to be needed if
                use_binary is True
       extent_kji: The extent of the grid as a list or a 3 element numpy array, in the order [NK, NJ, NI].
                If extent_kji is None, the extent is figured out from the data.  It must be
                given for 1D or 2D models
       corp_bin (boolean, default False): if True, input file is in bespoke corp binary format, otherwise ascii
       swap_bytes (boolean, default True): if True, byte ordering of corp bin data is reversed; only relevant
                if corp_bin is True
       max_lines_for_keyword: the maximum number of lines to search for CORP keyword; set to zero if file is
                known to be data only
       comment_char: A single character string which is interpreted as introducing a comment
       data_free_of_comments: If True, once the numeric data is encountered, it is assumed that there are no
                further comments (allowing a faster load)
       use_binary: If True, a more recent file containing a pure binary copy of the data is looked for first,
                in the same directory; if found, the data is loaded directly from that file; if not found, the
                binary file is created after the ascii has been loaded (ready for next time)
       eight_mode: If True, the data is assumed to be in CORP EIGHT ordering; otherwise the normal ordering
                (The code does not look for keywords.); this is not automatically determined from any keyword
                in the file
       use_numbers_only: no longer in use, ignored

    returns:
       A numpy array containing the CORP data in 7D pagoda protocol ordering. The extent of the grid, and hence
       shape of the array is determined from the corner point data unless extent_kji has been specified.
    """

    #   assert(extent_kji is not None or data_free_of_comments)   # no longer a requirement

    if extent_kji is None:
        extent = None
    else:
        extent = [extent_kji[0] * extent_kji[1] * extent_kji[2], 8, 3]

    if corp_bin:
        bin_file_size = os.path.getsize(file_name)
        bin_cell_count, remainder = divmod(bin_file_size,
                                           26 * 4)  # corp bin format has records of 24 + 2 (head, tail) 32 bit words
        if remainder:
            log.error('corp binary file ' + str(file_name) + ' is not a whole number of records')
            return None
        dt = np.dtype('float32')
        if swap_bytes:
            dt = dt.newbyteorder()
        cp_array = np.fromfile(file_name, dtype = dt, count = 26 * bin_cell_count).reshape((-1, 26))[:, 1:25]
        if extent_kji is not None:
            if bin_cell_count != (extent_kji[0] * extent_kji[1] * extent_kji[2]):
                log.error('corp binary file ' + str(file_name) + ' contains data for ' + str(bin_cell_count) +
                          ' cells; extent requires ' + str(extent_kji[0] * extent_kji[1] * extent_kji[2]))

    else:
        cp_array = load_array_from_file(file_name,
                                        extent = extent,
                                        data_type = 'real',
                                        keyword = 'CORP',
                                        max_lines_for_keyword = max_lines_for_keyword,
                                        comment_char = comment_char,
                                        data_free_of_comments = data_free_of_comments,
                                        use_binary = use_binary)

    cell_count, remainder = divmod(cp_array.size, 24)
    if remainder:
        log.error('file ' + file_name + ' contains ' + str(cp_array.size) + ' data, which is not a multiple of 24')
        return None

    cp_array = cp_array.reshape(1, 1, cell_count, 2, 2, 2, 3)  # pagoda 7D, temporarily with all cells on I axis
    log.debug('resequencing corner point data')
    gf.resequence_nexus_corp(cp_array, eight_mode = eight_mode)  # switches IP points where JP = 1 (unless eight_mode)

    if extent_kji is None:
        log.info('determining grid extent from corner points')
        extent_kji = gf.determine_corp_extent(cp_array)
        if extent_kji is None:
            log.error('failed to determine extent of grid from corner points')
            return cp_array

    return cp_array.reshape(extent_kji[0], extent_kji[1], extent_kji[2], 2, 2, 2, 3)


######################################################################################################
# load_array_from_file():
# function to load an array of data from a named file
# if the extent is not known (None), the file must be free of comments
# a new numpy array is returned
# binary data file creation & reuse is supported


def load_array_from_file(file_name,
                         extent = None,
                         data_type = 'real',
                         keyword = None,
                         max_lines_for_keyword = None,
                         comment_char = None,
                         data_free_of_comments = False,
                         use_binary = False,
                         use_numbers_only = None):
    """Load an array from an ascii (or pure binary) file.

    Arguments are similar to those for load_corp_array_from_file().
    """

    if not use_binary:
        return load_array_from_ascii_file(file_name,
                                          extent = extent,
                                          data_type = data_type,
                                          keyword = keyword,
                                          max_lines_for_keyword = max_lines_for_keyword,
                                          comment_char = comment_char,
                                          data_free_of_comments = data_free_of_comments)

    (extension, ab_type) = abt.binary_file_extension_and_np_type_for_data_type(data_type)

    if extent is None:
        cell_count = -1  # np.fromfile interprets this as 'read everything'
        log.debug('Loading unknown number of array data elements from file ' + file_name)
    else:
        cell_count = np.product(extent)
        log.debug('Loading %1d array data elements from file %s', cell_count, file_name)

    ascii_file_name = file_name
    binary_file_name = file_name
    if len(binary_file_name) < 4 or binary_file_name[-3:] != extension:
        binary_file_name += extension
    else:
        ascii_file_name = ascii_file_name[:-3]  # strip off '.db' or similar

    try:  # tentatively try to read data from an existing binary file, if present
        if file_exists(binary_file_name, must_be_more_recent_than_file = ascii_file_name):
            with open(binary_file_name, 'rb') as binary_file_in:
                result = np.fromfile(binary_file_in, dtype = ab_type, count = cell_count)
                if extent is not None:
                    result = result.reshape(extent)
                # check that end of file has been reached, ie. not too much data in file
                try:  # expected to return null
                    c = binary_file_in.read(1)
                    if len(c):
                        log.warning('binary file contains more data than expected: ' + binary_file_name)
                except Exception:
                    pass
                log.info('Data loaded from binary file %s', binary_file_name)
                return result
    except Exception:
        pass

    # read from ascii file
    result = load_array_from_ascii_file(file_name,
                                        extent = extent,
                                        data_type = data_type,
                                        keyword = keyword,
                                        max_lines_for_keyword = max_lines_for_keyword,
                                        comment_char = comment_char,
                                        data_free_of_comments = data_free_of_comments)

    # create a binary file (to be used next time)
    try:
        wd.write_pure_binary_data(binary_file_name, result)
    except Exception:
        log.warn('Failed to write data to binary file %s', binary_file_name)
        # todo: could delete the binary file in case a corrupt file is left for use next time

    return result


# end of load_array_from_file() def
######################################################################################################

######################################################################################################
# load_array_from_ascii_file():
# function to load an array of data from a named ascii file
# if the extent is not known (None), the file must be free of comments
# a new numpy array is returned


def load_array_from_ascii_file(file_name,
                               extent = None,
                               data_type = 'real',
                               keyword = None,
                               max_lines_for_keyword = None,
                               comment_char = None,
                               data_free_of_comments = False,
                               skip_c_space = True,
                               use_numbers_only = None):
    """Returns a numpy array with data loaded from an ascii file.

    arguments:
       file_name: string holding name of the existing ascii data file, eg. 'Cell_depth.dat'
       extent: a python list of integers specifying the extent (shape) of the array, eg. [148, 270, 103];
          if None, all data is read and returned as a 'flat' 1D array (data must be free from comments)
       data_type: the type of individual data elements, one of 'real','float','int', 'integer', 'bool' or 'boolean'
       keyword: if present, an attempt is made to find the keyword before reading data, if keyword is None or is
          not found, data is read from the start of the file
       max_lines_for_keyword: can be used to limit the search for keyword (for speed efficiency)
       comment_char: a single character string being the character used to introduce a comment in the file
       data_free_of_comments: if set to True, a faster load is used once any header line comments have been
          skipped
       skip_c_space: if True then a line starting 'C' followed by white space is skipped as a comment
       use_numbers_only: this argument is no longer in use and is ignored

    returns:
       a numpy array of shape specified in extent argument with dtype matching data_type

    example call:
       depth_array = load_data.load_array_from_ascii_file('Cell_depth.dat', [148, 270, 103])

    notes:
       In all use cases, this function is designed to load a single array of data from an ascii file
       that DOES NOT CONTAIN OTHER ARRAYS as well, ie. data for a single simulation keyword in the file.

       If skip_c_space is True, lines starting 'C ' are
       also treated as comments.  If data_free_of_comments is True, there must be
       at least one blank line before the data begins, and no further comments are permitted.
       (This format is designed to handle data files generated by a commonly used geomodelling package.)

       Repeat counts must not be present in the ascii data.

       The extent, if present, can contain any number of dimensions, typically 3 for reservoir modelling work.
       The total number of numbers in the file must match the number of elements in the given extent
       (ie. the product of the list of numbers in the extent argument).
       The order of indices in extent should be 'slowest changing' first, eg.: k,j,i

       The data_type defaults to 'real'
       'real' and 'float' are synonymous; 'int' and 'integer' are synonymous;
       'bool' and 'boolean' are synonymous; default is 'real'
       The numpy data type will be the default 64 bit float or 64 bit int
    """
    # todo: Code enhancement could cater for 32 bit options (and 8 bit for bool) if needed to reduce memory usage

    if extent is None:
        cell_count = -1  # np.fromfile interprets this as 'read everything'
        log.debug('Loading unknown number of array data elements from ascii file ' + file_name)
    else:
        cell_count = np.product(extent)
        log.debug('Loading %1d array data elements from ascii file %s', cell_count, file_name)

    if data_type in ['real', 'float', float]:
        d_type = 'float'
    elif data_type in ['int', 'integer', int]:
        d_type = 'int'
    elif data_type in ['bool', 'boolean', bool]:
        d_type = 'int'  # read booleans as 0 or 1 and convert after
    else:
        assert False, 'Unknown data_type passed to load_array_from_ascii_file' + str(data_type)

    read_file_name = file_name

    with open(read_file_name, 'r') as data_file:

        if not comment_char and not data_free_of_comments:
            comment_char = kf.guess_comment_char(data_file)
            if not comment_char:
                comment_char = '!'

        if keyword:
            keyword_found = kf.find_keyword(data_file, keyword, max_lines = max_lines_for_keyword)
            if keyword_found:
                data_file.readline()  # skip keyword line

        kf.skip_blank_lines_and_comments(data_file, comment_char = comment_char, skip_c_space = skip_c_space)

        result = None

        if data_free_of_comments:  # use numpy fromfile function after passing header comments

            result = np.fromfile(data_file, dtype = d_type, count = cell_count, sep = ' ')

            if result is None:
                data_file.seek(0)

        if result is None:

            # builds one very big string, stripping trailing comments

            #         s = ''
            #         while True:
            #            r = data_file.readline()
            #            if len(r) == 0: break
            #            s += r.partition(comment_char)[0] + ' '
            #         result = np.fromstring(s, dtype = d_type, count = cell_count, sep = ' ')
            #         # note: extra data will go unnoticed if cell count known!
            #         del s

            b = bytearray(data_file.read().encode())
            bc = comment_char.encode()
            nl = b'\n'
            sp = b' '
            i = 0
            while True:
                i = b.find(bc, i)
                if i < 0:
                    break
                eol = b.find(nl, i)
                if eol < 0:
                    eol = len(b)
                b[i:eol] = sp * (eol - i)

            result = np.fromstring(b.decode(), dtype = d_type, count = cell_count, sep = ' ')
            # note: extra data will go unnoticed if cell count known!
            del b

        if result is None:

            assert (extent is not None)  # TODO: remove this restriction by dynamically extending a flat array

            view_1D = np.zeros(extent, dtype = d_type).flatten()
            elements = view_1D.size
            start_of_line = True

            for index in range(elements):

                while True:
                    ch = data_file.read(1)
                    assert (ch != '')  # premature end of file
                    if ch == comment_char:
                        data_file.readline()
                        start_of_line = True
                        continue
                    if skip_c_space and start_of_line and ch in ['C', 'c']:
                        next_ch = data_file.read(1)
                        if next_ch in ' \t\n':
                            if next_ch != '\n':
                                data_file.readline()
                            continue
                        ch += next_ch
                        break
                    if ch not in ' \t\n':
                        break

                start_of_line = False
                word = ch
                while True:
                    ch = data_file.read(1)
                    if ch == '':  # end of file (this assumes a whitespace character after last datum)
                        log.error('Not enough data in file %s: %1d of %1d numbers read', file_name, index, elements)
                        assert False, 'not enough data in file'
                    if ch in ' \t\n':
                        break
                    word += ch
                if ch == '\n':
                    start_of_line = True
                if d_type == 'float':
                    view_1D[index] = float(word)
                else:
                    view_1D[index] = int(word)

            result = view_1D

        kf.skip_blank_lines_and_comments(data_file, comment_char = comment_char, skip_c_space = skip_c_space)
        if not kf.end_of_file(data_file):
            assert False, 'too much data in ascii file: ' + file_name

    if result is not None and extent is not None:
        result = result.reshape(extent)

    if data_type in ['bool', 'boolean', bool]:
        return result != 0  # convert to boolean
    else:
        return result


# end of load_array_from_ascii_file() def
######################################################################################################
