"""vdb.py: Module providing functions for reading from VDB datasets."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import glob
import os
import zipfile as zf
import numpy as np
from struct import unpack

import resqpy.olio.grid_functions as gf
import resqpy.olio.xml_et as rqet

null_uint32 = 4294967295  # -1 if interpreted as int32

key_dict = {  # vdb key character mapping to: (numpy_dtype, size_in_bytes, unpack_format_ch)
    'R': ('float32', 4, 'f'),
    'D': ('float64', 8, 'd'),
    #   'D': ('int64',   8, 'i'),
    'I': ('int32', 4, 'i'),
    'C': (None, 1, 'c'),  # could map to numpy 'byte' but seems to be used for strings
    'P': ('uint32', 4, 'I'),
    'K': (None, 8, 'c'),  # don't store in numpy format; 8 character strings
    'X': (None, 0, 'c')
}  # used for invalid code character (non-ascii)

init_not_packed = ['DAD', 'KID', 'UID', 'UNPACK']


def coerce(a, dtype):
    """Returns a version of numpy array a with elements coerced to dtype.

    :meta private:
    """
    if dtype is None or a.dtype == dtype:
        return a
    b = np.empty(a.shape, dtype = dtype)
    b[:] = a
    return b


def ensemble_vdb_list(run_dir, sort_list = True):
    """Returns a sorted list of vdb paths found in the directory tree under run_dir."""

    ensemble_list = []

    def recursive_vdb_list(dir):
        nonlocal ensemble_list
        for entry in os.scandir(dir):
            if not entry.is_dir():
                continue
            if entry.name.endswith('.vdb') or entry.name.endswith('.vdb.zip'):
                ensemble_list.append(entry.path)
                continue
            elif entry.name.endswith('.rst'):
                continue  # optimisation
            recursive_vdb_list(entry.path)

    def cmp_to_key(mycmp):
        """Convert a cmp= function into a key= function."""

        class K:

            def __init__(self, obj, *args):
                self.obj = obj

            def __lt__(self, other):
                return mycmp(self.obj, other.obj) < 0

            def __gt__(self, other):
                return mycmp(self.obj, other.obj) > 0

            def __eq__(self, other):
                return mycmp(self.obj, other.obj) == 0

            def __le__(self, other):
                return mycmp(self.obj, other.obj) <= 0

            def __ge__(self, other):
                return mycmp(self.obj, other.obj) >= 0

            def __ne__(self, other):
                return mycmp(self.obj, other.obj) != 0

        return K

    def comparison(a, b):
        """Comparison function for a pair of case names."""
        pa = 0
        pb = 0
        while True:
            sa = pa
            sb = pb
            if pa >= len(a) and pb >= len(b):
                return 0
            while pa < len(a) and not a[pa].isdigit():
                pa += 1
            while pb < len(b) and not b[pb].isdigit():
                pb += 1
            if a[sa:pa].lower() < b[sb:pb].lower():
                return -1
            if a[sa:pa].lower() > b[sb:pb].lower():
                return 1
            if pa >= len(a) and pb >= len(b):
                return 0
            if pa >= len(a):
                return -1
            if pb >= len(b):
                return 1
            sa = pa
            sb = pb
            while pa < len(a) and a[pa].isdigit():
                pa += 1
            while pb < len(b) and b[pb].isdigit():
                pb += 1
            ia = int(a[sa:pa])
            ib = int(b[sb:pb])
            if ia < ib:
                return -1
            if ia > ib:
                return 1
        a_low = a.lower()
        b_low = b.lower()
        if a_low < b_low:
            return -1
        if a_low > b_low:
            return 1
        return 0

    recursive_vdb_list(run_dir)
    if sort_list:
        sorted_list = sorted(ensemble_list, key = cmp_to_key(comparison))
        ensemble_list = sorted_list
    return ensemble_list


class Header():
    """Internal class for handling a Header record in a vdb file."""

    def __init__(self, fp, place):
        """Creates a new Header record object."""

        fp.seek(place)
        block = fp.read(22)
        #      log.debug(f'Header block at place {place} returned {len(block)} bytes')
        self.previous, self.next, c, self.bytes_per_item, self.number_of_items, self.first_fragment, self.max_items = \
            unpack('=IIcBIII', block)
        try:
            self.item_type = c.decode()
        except Exception:
            self.item_type = 'X'  # non-ascii character!
        self.data_place = place + 22


#      log.debug('   header previous: ' + str(self.previous))
#      log.debug('   header next: ' + str(self.next))
#      log.debug('   header item type: ' + str(self.item_type))
#      log.debug('   header bytes per item: ' + str(self.bytes_per_item))
#      log.debug('   header number of items: ' + str(self.number_of_items))
#      log.debug('   header first fragment: ' + str(self.first_fragment))
#      log.debug('   header max items: ' + str(self.max_items))
#      log.debug('   header data place: ' + str(self.data_place))


class FragmentHeader():
    """Internal class for handling a Fragment Header record in a vdb file."""

    def __init__(self, fp, place):
        """Creates a new Fragment Header record object."""

        #      log.debug(f'FragmentHeader init at place {place}')
        fp.seek(place)
        block = fp.read(8)
        self.next, self.number_of_items = unpack('=II', block)
        self.data_place = place + 8


#      log.debug(f'   fragment header number of items {self.number_of_items}; data place {self.data_place}; next {self.next}')


class RawData():
    """Internal class for handling Raw Data records in a vdb file."""

    def __init__(self, fp, place, item_type, count, max_count):
        """Creates a new Raw Data record object."""

        if max_count is not None and count > max_count:
            count = max_count
        self.a = None
        self.c = None
        dtype, byte_size, form_ch = key_dict[item_type]
        #      log.debug('raw data call: place {}; item type {}; count {}; dtype {}, byte size {}, form ch {}'.format(
        #                place, item_type, count, dtype, byte_size, form_ch))
        fp.seek(place)
        if dtype is None:
            if form_ch == 'c':
                chars = fp.read(count * byte_size)
                if item_type == 'C':
                    self.c = chars.decode()
                else:
                    self.c = []
                    for i in range(count):
                        self.c.append(chars[i * byte_size:(i + 1) * byte_size].decode().strip().upper())
            else:  # shouldn't come into play
                block = fp.read(count * byte_size)
                form = '=' + str(count) + form_ch
                self.c = unpack(form, block)
        elif dtype == 'float64':  # try tentative 32bit word swap
            b = fp.read(count * 8)
            #         c = b''
            #         for d in range(count):
            #            c += b[4*d+4:4*d+8] + b[4*d:4*d+4]
            nda = np.ndarray((count, 2), dtype = 'int32', buffer = b)
            self.a = np.empty((count, 2), dtype = int)
            self.a[:, :] = nda[:, :]
        else:
            b = fp.read(count * byte_size)
            self.a = np.frombuffer(buffer = b, dtype = dtype, count = count)


class Data():
    """Internal class for handling Data records in a vdb file."""

    def __init__(self, fp, header):
        """Creates a new Data object."""

        if header is None:
            self.a = None
            self.c = None
        else:
            #         log.debug(f'Data init at place {header.data_place}; type {header.item_type}; number of items {header.number_of_items}')
            raw = RawData(fp, header.data_place, header.item_type, header.number_of_items, header.max_items)
            if raw is not None and raw.a is not None and raw.a.size == 1 and raw.a.dtype == 'int32' and header.next != null_uint32:
                #            log.debug('   skipping integer value of ' + str(raw.a[0]))
                next_head = Header(fp, header.next)
                raw = RawData(fp, next_head.data_place, next_head.item_type, next_head.number_of_items,
                              next_head.max_items)
                header = next_head  # making this up as I go along
            self.a = raw.a
            self.c = raw.c
            if header.first_fragment != null_uint32:
                #            log.debug(f'   chaining from {header.first_fragment}')
                chain = FragmentChain(fp, header.first_fragment, header)
                if self.a is not None:
                    self.a = np.append(self.a, chain.a)
                if self.c is not None:
                    self.c += chain.c

    #   if self.c is not None:
    #      log.debug(f'   data c {self.c}')
    #   elif self.a is not None:
    #      log.debug(f'   data a {self.a}')


class Fragment():
    """Internal class for handling an individual Fragment record in a vdb file."""

    def __init__(self, fp, place, header):
        """Creates a new Fragment record object."""

        #      log.debug(f'Fragment init at place {place}')
        self.head = FragmentHeader(fp, place)
        if self.head.number_of_items == 0:
            #         log.debug('   zero items in fragment')
            self.c = None
            self.a = None
        else:
            raw = RawData(fp, self.head.data_place, header.item_type, self.head.number_of_items, header.max_items)
            self.a = raw.a
            self.c = raw.c


#      if self.c is not None:
#         log.debug(f'   fragment c [{self.c}]')


class FragmentChain():
    """Internal class for handling a chain of Fragment records in a vdb file."""

    def __init__(self, fp, place, header):  # returns either 1D numpy array (numeric data) or list of 8
        """Creates a new Fragment Chain object."""

        self.a = None
        self.c = None
        #      log.debug(f'FragmentChain init at place {place}')
        while place != null_uint32:
            fragment = Fragment(fp, place, header)
            if fragment.head.number_of_items > 0:
                if fragment.a is not None:
                    if self.a is None:
                        self.a = fragment.a
                    else:
                        self.a = np.concatenate(self.a, fragment.a)
                if fragment.c:
                    if self.c is None:
                        self.c = []
                    if isinstance(fragment.c, str):
                        self.c.append(fragment.c)
                    elif isinstance(fragment.c, list):
                        for item in fragment.c:
                            if item.isascii():
                                self.c.append(item)
            place = fragment.head.next
        assert self.c is None or self.a is None, 'mixture of character and numeric data in fragment chain'

    # todo: check number of elements matches header info?


class KP():
    """Internal class for a (Key, Pointer) record in a vdb file."""

    def __init__(self, fp, place = 4):
        """Creates a new (Key, Pointer) record object."""

        #      log.debug(f'KP init at place {place}')
        self.k_head = Header(fp, place)
        assert self.k_head.item_type == 'K', 'did not find expected Key header'
        assert self.k_head.bytes_per_item == 8, 'bytes per item not 8 in Key header'
        assert self.k_head.number_of_items > 0, 'zero items in Key header'
        place = self.k_head.next
        assert place != null_uint32, 'no next header in Key header'
        self.p_head = Header(fp, place)
        assert self.p_head.item_type == 'P', 'did not find expected Pointer header'
        assert self.p_head.bytes_per_item == 4, 'bytes per item not 4 in Pointer header'
        assert self.p_head.number_of_items == self.k_head.number_of_items, \
            'number of items in Pointer header does not match number in Key header'
        self.keywords = Data(fp, self.k_head).c
        assert self.keywords is not None and isinstance(self.keywords,
                                                        list), 'list of keywords not extracted from Key record'
        self.pointers = Data(fp, self.p_head).a
        assert self.pointers is not None and self.pointers.dtype == 'uint32'
        self.fp = fp

    def header_place_for_key(self, key, search = False):
        """Returns file position pointer for a given key."""

        key = key.strip().upper()
        if key in self.keywords:
            return self.pointers[self.keywords.index(key)]
        if not search:
            return None
        for head_key in self.keywords:
            sub_head_place = self.pointers[self.keywords.index(head_key)]
            try:
                sub_kp = KP(self.fp, sub_head_place)
                sub_place = sub_kp.header_place_for_key(key, search = True)
                if sub_place is not None:
                    return sub_place
            except AssertionError:
                pass  # probably not a K type subsidiary
            except Exception:
                raise
        return None

    def head_for_key(self, key, search = False):
        """Returns a Header record object for the given key."""

        head_place = self.header_place_for_key(key, search = search)
        if head_place is None:
            return None
        head = Header(self.fp, head_place)
        if head.item_type == 'X':
            return None  # invalid (non-ascii) item type character
        return head

    def data_for_key(self, key, search = False):
        """Returns a Data object for the given key."""

        head = self.head_for_key(key, search = search)
        if head is None:
            return None
        return Data(self.fp, head)

    def key_list(self, filter = False):
        """Returns a list of keys."""

        # return Data(self.fp, self.k_head).c
        if not filter:
            return self.keywords
        filtered_list = []
        for key in self.keywords:
            # log.debug(f'key_list raw entry: {key}')
            if not bad_keyword(key):
                filtered_list.append(key)
        return filtered_list

    def sub_key_list(self, keyword, filter = False):
        """Returns a list of keys subordinate to keyword."""

        assert keyword in self.key_list(), 'keyword not present: ' + keyword
        sub_head_place = self.pointers[self.keywords.index(keyword)]
        sub_kp = KP(self.fp, sub_head_place)
        return sub_kp.key_list(filter = filter)


class VDB():
    """Class for handling a vdb, particularly to support import of grid and properties."""

    def __init__(self, path):
        """Initialises a VDB object and associates it with the given vdb directory path."""

        self.path = None  # internal zip path of vdb in the case of a zip file
        self.zip_file = None
        self.zipped = zf.is_zipfile(path)
        if self.zipped:
            self.zip_file = path
            with zf.ZipFile(path) as zfp:
                zip_list = zfp.namelist()
                warn = False
                for name in zip_list:
                    if name.endswith('.vdb/'):
                        if self.path is not None:
                            warn = True
                        self.path = name[:-1]
                    elif self.path is None and name == 'main.xml':
                        self.path = ''
            if warn:
                log.warning('more than one vdb in zip file: using last')
            if self.path is None:
                raise Exception('no vdb found in zip file (at top level)')
        else:
            self.path = path
            if not os.path.isdir(path):
                log.warning('vdb path is neither a directory nor a zip file: ' + str(path))
        self.case_list = None
        self.use_case = None
        self.grid_extents_kji = {}  # maps case name, grid_name to grid extent
        self.grid_packings = {}  # maps case name, grid_name to grid unpack array
        self.grid_lists = {}  # maps case name to grid name list
        self.cases()  # sets use_case to first case in list for vdb

    def cases(self):
        """Returns a list of simulation case strings as found in the main xml file."""

        if self.case_list is not None:
            return self.case_list
        try:
            xml_file = os.path.join(self.path, 'main.xml')
            if self.zipped:
                with zf.ZipFile(self.zip_file) as zfp:
                    with zfp.open(xml_file) as fp:
                        tree = rqet.parse(fp)
                        root = tree.getroot()
            else:
                assert os.path.exists(xml_file), 'could not find vdb main xml file: ' + xml_file
                with open(xml_file, 'r') as fp:
                    tree = rqet.parse(fp)
                    root = tree.getroot()
            caselist = rqet.list_of_tag(rqet.find_tag(root, 'CASELIST'), 'CASE')
            self.case_list = []
            for case in caselist:
                self.case_list.append(str(case.attrib['Name']).strip())
            if self.use_case is None and len(self.case_list):
                self.use_case = self.case_list[0]
        except Exception:
            log.exception('failed to extract case list')
        return self.case_list

    def set_use_case(self, case):
        """Sets the simulation case to use in other functions."""

        try:
            assert case in self.cases(), 'failed to find case ' + case + ' in list of vdb cases'
            self.use_case = case
        except Exception:
            log.exception('failed to set use case')

    def print_header_tree(self, relative_path):
        """Low level: prints out the raw header tree found in the vdb file (for debugging)."""

        visited = []

        def print_tree(fp, place = 4, level = 0, dots = False, file_size = None):
            """Recursive function for printing vdb tree."""
            try:
                if file_size is None:
                    fp.seek(0, 2)
                    file_size = int(fp.tell())
                assert 0 <= int(place) < file_size, f'place {place} not within bounds of file of size {file_size}'
                assert place not in visited, 'circular pointers in vdb file header structure'
                visited.append(place)
                head = Header(fp, place = place)
                if head is None:
                    raise ValueError()
                if dots:
                    indent = ((level - 1) * 3) * ' ' + ' ..'
                else:
                    indent = (level * 3) * ' '
                print(indent,
                      head.item_type,
                      ':',
                      head.number_of_items,
                      'items at',
                      head.bytes_per_item,
                      'bytes per item',
                      end = '')
                if head.item_type == 'C':
                    s = RawData(fp, head.data_place, 'C', head.number_of_items, head.max_items)
                    if s is None or s.c is None:
                        print(': ?', end = '')
                    else:
                        print(': "' + s.c + '"', end = '')
                if head.first_fragment != null_uint32:
                    print(' with fragment(s)')
                else:
                    print('')
                if head.item_type == 'P':
                    pointers = RawData(fp, head.data_place, 'P', head.number_of_items, head.max_items)
                    assert (pointers is not None and pointers.a is not None)
                    for p in pointers.a:
                        if p in visited:
                            log.warning('breaking at circular pointer to place ' + str(p))
                            break
                        if int(p) == 0:
                            log.warning('pointer to place zero encountered')
                        elif int(p) > file_size:
                            log.warning('skipping pointer to place beyond end of file ' + str(p))
                        else:
                            print_tree(fp, place = p, level = level + 1, dots = True, file_size = file_size)
                if head.next != null_uint32:
                    if not 0 <= int(head.next) < file_size:
                        log.warning(f'skipping next {head.next} outwith bounds of file of size {file_size}')
                    else:
                        print_tree(fp, place = head.next, level = level, file_size = file_size)
            except Exception:
                print('?')
                log.exception('failed in print tree for headers')
                raise

        try:
            path = os.path.join(self.path, self.use_case, relative_path)
            assert os.path.exists(path), 'failed to find vdb file ' + path
            print('Header tree for: ' + path)
            with open(path, 'rb') as fp:
                print_tree(fp)
        except Exception:
            log.exception('failed to print header tree for relative file ' + relative_path)

    def print_key_tree(self, relative_path):
        """Low level: prints out the keyword tree found in the vdb file."""

        visited = []

        def print_tree(fp, place = 4, level = 0):
            try:
                if int(place) == 0:
                    log.warning('returning due to place zero in vdb file key structure')
                    return
                if place in visited:
                    log.warning(f'returning at circular pointer at place {place} in vdb file key structure')
                    return
                visited.append(place)
                kp = KP(fp, place = place)
            except AssertionError:
                return
            except Exception as e:
                log.error(str(e) + ' at place ' + str(place))
                raise
            key_list = kp.key_list()
            for key in key_list:
                print((level * 3) * ' ', key)
                print_tree(fp, kp.header_place_for_key(key, search = False), level + 1)

        try:
            path = os.path.join(self.path, self.use_case, relative_path)
            if self.zipped:
                with zf.ZipFile(self.zip_file) as zfp:
                    with zfp.open(path) as fp:
                        print_tree(fp)
            else:
                assert os.path.exists(path), 'failed to find vdb file ' + str(path)
                with open(path, 'rb') as fp:
                    print_tree(fp)
        except Exception:
            log.exception('failed to print keyword tree for relative file ' + relative_path)

    def data_for_keyword(self, relative_path, keyword, search = True):
        """Reads data associated with a keyword from a vdb binary file; returns a numpy array (or string)."""

        try:
            path = os.path.join(self.path, self.use_case, relative_path)
            if self.zipped:
                with zf.ZipFile(self.zip_file) as zfp:
                    with zfp.open(path) as fp:
                        assert fp.read(4) == b'NT32', 'first 4 characters not NT32 in file ' + path
                        kp = KP(fp)
                        return kp.data_for_key(keyword, search = search)
            else:
                assert os.path.exists(path), 'failed to find vdb file ' + str(path)
                with open(path, 'rb') as fp:
                    assert fp.read(4) == b'NT32', 'first 4 characters not NT32 in file ' + path
                    kp = KP(fp)
                    return kp.data_for_key(keyword, search = search)
        except Exception:
            log.exception('failed to read keyword data from vdb binary file: ' + path)
        return None

    def data_for_keyword_chain(self, relative_path, keyword_chain):
        """Follows a list of keywords down through hierarchy and returns the data as a numpy array (or string)."""

        def dfkc_fp(fp, path, keyword_chain):
            assert fp.read(4) == b'NT32', 'first 4 characters not NT32 in file ' + path
            chain_list = list(keyword_chain)
            assert len(chain_list) > 0, 'empty chained keyword list'
            head_place = 4
            while len(chain_list) > 1:
                kp = KP(fp, place = head_place)
                head_place = kp.header_place_for_key(chain_list[0], search = False)
                assert head_place is not None, 'failed to find chained keyword: ' + chain_list[0]
                chain_list.pop(0)
            kp = KP(fp, place = head_place)
            return kp.data_for_key(chain_list[0], search = False)

        path = os.path.join(self.path, self.use_case, relative_path)
        try:
            if isinstance(keyword_chain, str):
                return self.data_for_keyword(relative_path, keyword_chain)
            if self.zipped:
                with zf.ZipFile(self.zip_file) as zfp:
                    with zfp.open(path) as fp:
                        return dfkc_fp(fp, path, keyword_chain)
            else:
                assert os.path.exists(path), 'failed to find vdb file ' + str(path)
                with open(path, 'rb') as fp:
                    return dfkc_fp(fp, path, keyword_chain)
        except Exception:
            log.exception('failed to read chained keyword data from vdb binary file: ' + path)
        return None

    def set_extent_kji(self, extent_kji, use_case = None, grid_name = 'ROOT'):
        """Sets extent for one use case (defaults to current use case) as alternative to processing corp data."""

        assert len(extent_kji) == 3, 'triple integer required for extent_kji'
        if use_case is None:
            use_case = self.use_case
        assert use_case is not None, 'no use case for extent setting'
        self.grid_extents_kji[use_case, grid_name.upper()] = extent_kji

    def fetch_corp_patch(self, relative_path):
        """Loads one patch of grid corp data from one file in the vdb; returns (number of cells, 1D array)."""

        a = self.data_for_keyword(relative_path, 'CORP').a
        assert a is not None, 'failed to extract corp data from vdb relative path ' + relative_path
        element_count = a.size
        assert element_count > 0, 'corp data is empty (no numbers)'
        cells, remainder = divmod(element_count, 24)
        assert remainder == 0, 'number of numbers in corp data is not a multiple of 24'
        if a.dtype == 'float32':
            d = np.empty((element_count,), dtype = 'float')  # double precision
            d[:] = a
            del a
            a = d
        return cells, a

    def list_of_grids(self):
        """Returns a list of grid names for which corp data exists in the vdb for the current use case."""

        if self.use_case is None:
            self.cases()
        if self.use_case in self.grid_lists:
            return self.grid_lists[self.use_case]
        glob_path = os.path.join(self.path, self.use_case, 'INIT', '*_corp.bin')
        if self.zipped:
            corp_list = self.zip_glob(glob_path)
        else:
            corp_list = glob.glob(glob_path)
        grid_name_list = []
        for corp_path in corp_list:
            _, corp_file_name = os.path.split(corp_path)
            grid_name = corp_file_name[:-9].upper()
            if grid_name == 'ROOT':
                grid_name_list.insert(0, 'ROOT')
            else:
                grid_name_list.append(grid_name)
        self.grid_lists[self.use_case] = grid_name_list
        return grid_name_list

    def root_corp(self):
        """Loads root grid corp data from vdb; returns pagoda style resequenced 7D numpy array of doubles."""

        return self.grid_corp('ROOT')

    def grid_corp(self, grid_name):
        """Loads corp data for named grid from vdb; returns pagoda style resequenced 7D numpy array of doubles."""
        try:
            if self.use_case is None:
                self.cases()
            grid_name = grid_name.upper()
            assert self.use_case is not None, 'no case found in vdb'
            relative_path = os.path.join('INIT', str(grid_name) + '_corp.bin')
            cells, a = self.fetch_corp_patch(relative_path)
            # check for extra corp patch files (used for big grids)
            glob_path = os.path.join(self.path, self.use_case, 'INIT', str(grid_name) + '_corp')
            glob_path += '_*.bin'
            if self.zipped:
                corp_patch_list = self.zip_glob(glob_path)
            else:
                corp_patch_list = glob.glob(glob_path)
            if len(corp_patch_list) > 0:
                sort_list = []
                for patch_name in corp_patch_list:
                    layer_number = int(patch_name[patch_name.rfind('_') + 1:-4])
                    sort_list.append((layer_number, patch_name))
                sort_list.sort()
                for _, patch_name in sort_list:
                    patch_relative_path = patch_name[patch_name.rfind('INIT'):]
                    patch_cells, patch = self.fetch_corp_patch(patch_relative_path)
                    a = np.append(a, patch)
                    cells += patch_cells
            ap = a.reshape((1, 1, cells, 2, 2, 2, 3))
            gf.resequence_nexus_corp(ap)  # move from Nexus corp ordering to Pagoda ordering
            if (self.use_case, grid_name) in self.grid_extents_kji:
                extent_kji = self.grid_extents_kji[self.use_case, grid_name]
                assert cells == extent_kji[0] * extent_kji[1] * extent_kji[2], 'corp extent mismatch in vdb'
            else:
                extent_kji = gf.determine_corp_extent(ap)
                if extent_kji is None:
                    log.warning('failed to determine extent of root grid from corp data')
                    extent_kji = (1, 1, cells)
                else:
                    self.grid_extents_kji[self.use_case, grid_name] = extent_kji
            shape_7d = (extent_kji[0], extent_kji[1], extent_kji[2], 2, 2, 2, 3)
            return ap.reshape(shape_7d)
        except Exception:
            log.exception('failed to extract root grid corp data from vdb')
        return None

    def load_init_mapdata_array(self, file, keyword, dtype = None, unpack = False, grid_name = 'ROOT'):
        """Loads an INIT MAPDATA array from vdb; returns 3D numpy array coerced to dtype (if not None)."""

        try:
            if self.use_case is None:
                self.cases()
            assert self.use_case is not None, 'no case found in vdb'
            relative_path = os.path.join('INIT', 'MAPDATA', file)
            a = self.data_for_keyword(relative_path, keyword, search = True).a
            assert a is not None, 'failed to extract data for keyword ' + keyword + ' from vdb relative path ' + relative_path
            if unpack:
                un = self.grid_unpack(grid_name)
                if un is not None:
                    null_start = np.zeros((a.size + 1,), dtype = a.dtype)  # better to use Nan for null value?
                    null_start[1:] = a
                    a = null_start[un]
            a = self.grid_shaped(grid_name, coerce(a, dtype))
            return a
        except Exception:
            log.exception('failed to extract data from vdb for grid ' + str(grid_name))
        return None

    def load_recurrent_mapdata_array(self, file, keyword, dtype = None, unpack = False, grid_name = 'ROOT'):
        """Loads a RECUR MAPDATA array from vdb; returns 3D numpy array coerced to dtype (if not None)."""

        try:
            if bad_keyword(keyword):
                log.warning('ignoring attempt to load recurrent vdb mapdata for corrupt keyword')
                return None
            if self.use_case is None:
                self.cases()
            assert self.use_case is not None, 'no case found in vdb'
            relative_path = os.path.join('RECUR', 'MAPDATA', file)
            data = self.data_for_keyword(relative_path, keyword, search = True)
            if not data:
                return None
            a = data.a
            if a is None:
                return None
            if unpack:
                un = self.grid_unpack(grid_name)
                if un is not None:
                    null_start = np.zeros((a.size + 1,), dtype = a.dtype)  # better to use Nan for null value?
                    null_start[1:] = a
                    a = null_start[un]
            a = self.grid_shaped(grid_name, coerce(a, dtype))
            return a
        except Exception:
            log.exception('failed to extract grid recurrent data from vdb for keyword: ' + keyword)
        return None

    def root_dad(self):
        """Loads and returns the IROOTDAD array from vdb; returns 3D numpy int32 array."""

        return self.grid_dad('ROOT')

    def grid_dad(self, grid_name):
        """Loads and returns the DAD array from vdb for the named grid; returns 3D numpy int32 array."""

        # todo: check file naming and values for LGRs
        grid_name = grid_name.upper
        # DAD data appear to be unique cell ids, in usual sequence, starting at 1, same as UID
        return self.load_init_mapdata_array(file = 'I' + grid_name + 'DAD.bin',
                                            keyword = 'DAD',
                                            dtype = 'int32',
                                            unpack = False,
                                            grid_name = grid_name)

    def root_kid(self):
        """Loads and returns the IROOTKID array from vdb; returns 3D numpy int32 array (can be inactive cell mask)."""

        return self.grid_kid('ROOT')

    def grid_kid(self, grid_name):
        """Loads and returns the IROOTKID array from vdb; returns 3D numpy int32 array (can be inactive cell mask)."""

        grid_name = grid_name.upper()
        # see comments in grid_kid_inactive_mask() for KID values
        return self.load_init_mapdata_array(file = 'I' + grid_name + 'KID.bin',
                                            keyword = 'KID',
                                            dtype = 'int32',
                                            unpack = False,
                                            grid_name = grid_name)

    def root_kid_inactive_mask(self):
        """Loads the IROOTKID array and returns boolean mask of cells inactive in ROOT grid."""

        return self.grid_kid_inactive_mask('ROOT')

    def grid_kid_inactive_mask(self, grid_name):
        """Loads the KID array for the named grid and returns boolean mask of cells inactive in grid."""

        # KID values:
        # 0 - active in this grid
        # >0 - (in ROOT grid) cell inactive in ROOT as assigned to an LGR (KID value is LGR number)
        # -1 – inactive for normal reasons (zero pore volume or DEADCELL set)
        # -2 - (in LGR grids) cell omitted from LGR (possibly omitted as inactive in parent grid?)
        # -3 - inactive due to coarsening (coarsening repurposes one cell in parent grid for each coarsened cell)
        i = self.grid_kid(grid_name)
        if i is None:
            return None
        b = np.empty(i.shape, dtype = 'bool')
        b[:] = (i != 0)
        return b

    def root_uid(self):
        """Loads and returns the IROOTUID array from vdb; returns 3D numpy int32 array."""

        return self.grid_uid('ROOT')

    def grid_uid(self, grid_name):
        """Loads and returns the UID array from vdb for the named grid; returns 3D numpy int32 array."""

        grid_name = grid_name.upper()
        # UID data appear to be unique cell ids, in usual sequence, starting at 1, same as DAD
        return self.load_init_mapdata_array(file = 'I' + grid_name + 'UID.bin',
                                            keyword = 'UID',
                                            dtype = 'int32',
                                            unpack = False,
                                            grid_name = grid_name)

    def root_unpack(self):
        """Loads and returns the IROOTUNPACK array from vdb; returns 3D numpy int32 array."""

        return self.grid_unpack('ROOT')

    def grid_unpack(self, grid_name):
        """Loads and returns the IROOTUNPACK array from vdb; returns 3D numpy int32 array."""

        grid_name = grid_name.upper()
        if self.use_case is not None and (self.use_case, grid_name) in self.grid_packings.keys():
            return self.grid_packings[self.use_case, grid_name]
        # UNPACK data appear to be index into packed array for each cell, zero for inactive/absent
        un = self.load_init_mapdata_array(file = 'I' + grid_name + 'UNPACK.bin',
                                          keyword = 'UNPACK',
                                          dtype = 'int32',
                                          unpack = False,
                                          grid_name = grid_name)
        if self.use_case is not None:
            self.grid_packings[self.use_case,
                               grid_name] = un  # cache unpack array for later packing & unpacking operations
        return un

    def list_of_static_properties(self):
        """Returns list of static property keywords present in the vdb for ROOT."""

        return self.grid_list_of_static_properties('ROOT')

    def grid_list_of_static_properties(self, grid_name):
        """Returns list of static property keywords present in the vdb for named grid."""

        grid_name = grid_name.upper()
        glob_path = os.path.join(self.path, self.use_case, 'INIT', 'MAPDATA', 'I' + grid_name + '*.bin')
        if self.zipped:
            file_list = self.zip_glob(glob_path)
        else:
            file_list = glob.glob(glob_path)
        keyword_list = []
        for p in file_list:
            _, f = os.path.split(p)
            keyword = f[1 + len(grid_name):-4]
            keyword_list.append(keyword)
        return keyword_list

    def root_static_property(self, keyword, dtype = None, unpack = None):
        """Loads and returns a ROOT static property array."""

        return self.grid_static_property('ROOT'.keyword, dtype = dtype, unpack = unpack)

    def grid_static_property(self, grid_name, keyword, dtype = None, unpack = None):
        """Loads and returns a static property array for named grid."""

        grid_name = grid_name.upper()
        keyword = keyword.strip().upper()
        if unpack is None:
            unpack = (keyword not in init_not_packed)
        return self.load_init_mapdata_array(file = 'I' + grid_name + keyword + '.bin',
                                            keyword = keyword,
                                            dtype = dtype,
                                            unpack = unpack,
                                            grid_name = grid_name)

    def list_of_timesteps(self):
        """Returns a list of integer timesteps for which a ROOT recurrent mapdata file exists."""

        return self.grid_list_of_timesteps('ROOT')

    def grid_list_of_timesteps(self, grid_name):
        """Returns a list of integer timesteps for which a recurrent mapdata file for the named grid exists."""

        # todo: check file naming for LGR recurrent properties
        grid_name = grid_name.upper()
        glob_path = os.path.join(self.path, self.use_case, 'RECUR', 'MAPDATA', 'R' + grid_name + '_*.bin')
        if self.zipped:
            file_list = self.zip_glob(glob_path)
        else:
            file_list = glob.glob(glob_path)
        timestep_list = []
        for p in file_list:
            timestep_list.append(int(p[p.rfind('_') + 1:-4]))
        return sorted(timestep_list)

    def list_of_recurrent_properties(self, timestep):
        """Returns list of recurrent property keywords present in the vdb for given timestep."""

        return self.grid_list_of_recurrent_properties('ROOT', timestep)

    def grid_list_of_recurrent_properties(self, grid_name, timestep):
        """Returns list of recurrent property keywords present in the vdb for named grid for given timestep."""

        def keyword_list_from_fp(fp, path):
            """Returns a list of keywords (strings) found in a recurrent file."""
            assert fp.read(4) == b'NT32', 'first 4 characters not NT32 in file ' + path
            kp = KP(fp)
            keyword_list = kp.sub_key_list('MAPDATA', filter = True)
            try:
                keyword_list.remove('LASTMOD')
            except Exception:
                pass
            bad_key_indices = []
            for i in range(len(keyword_list)):
                key = keyword_list[i]
                if bad_keyword(key):
                    bad_key_indices.append(i)
            if len(bad_key_indices):
                log.warning(str(len(bad_key_indices)) + ' non-ascii keywords ignored in recurrent file ' + path)
                bad_key_indices.reverse()
                for i in bad_key_indices:
                    keyword_list.pop(i)
            return keyword_list

        # todo: check file naming for LGR recurrent properties
        grid_name = grid_name.upper()
        try:
            path = os.path.join(self.path, self.use_case, 'RECUR', 'MAPDATA',
                                'R' + grid_name + '_' + str(timestep) + '.bin')
            if self.zipped:
                with zf.ZipFile(self.zip_file) as zfp:
                    with zfp.open(path) as fp:
                        return keyword_list_from_fp(fp, path)
            else:
                assert os.path.exists(path), 'failed to find vdb file ' + str(path)
                with open(path, 'rb') as fp:
                    return keyword_list_from_fp(fp, path)
        except Exception:
            log.exception('failed to read keyword data from vdb binary file: ' + path)
        return None

    def root_recurrent_property_for_timestep(self, keyword, timestep, dtype = None, unpack = True):
        """Loads and returns a ROOT recurrent property array for one timestep."""

        return self.grid_recurrent_property_for_timestep('ROOT', keyword, timestep, dtype = dtype, unpack = unpack)

    def grid_recurrent_property_for_timestep(self, grid_name, keyword, timestep, dtype = None, unpack = True):
        """Loads and returns a recurrent property array for named grid for one timestep."""

        # todo: check file naming convention for LGRs
        grid_name = grid_name.upper()
        keyword = keyword.strip().upper()
        return self.load_recurrent_mapdata_array(file = 'R' + grid_name + '_' + str(timestep) + '.bin',
                                                 keyword = keyword,
                                                 dtype = dtype,
                                                 unpack = unpack,
                                                 grid_name = grid_name)

    def header_place_for_keyword(self, relative_path, keyword, search = True):
        """Low level function to return file position for header relating to given keyword."""

        try:
            path = os.path.join(self.path, relative_path)
            assert os.path.exists(path), 'failed to find vdb file ' + str(path)
            with open(path, 'rb') as fp:
                assert fp.read(4) == b'NT32', 'first 4 characters not NT32 in file ' + path
                kp = KP(fp)
                return kp.header_place_for_key(keyword, search = search)
        except Exception:
            log.exception('failed to read keyword data from vdb binary file: ' + path)
        return None

    def root_shaped(self, a):
        """Returns array reshaped to root grid extent for current use case, if known; otherwise unchanged."""

        return self.grid_shaped('ROOT', a)

    def grid_shaped(self, grid_name, a):
        """Returns array reshaped to named grid extent for current use case, if known; otherwise unchanged."""

        grid_name = grid_name.upper()
        if self.use_case is not None and (self.use_case, grid_name) in self.grid_extents_kji.keys():
            cell_count = np.prod(self.grid_extents_kji[self.use_case, grid_name])
            if a.size == cell_count:
                return a.reshape(tuple(self.grid_extents_kji[self.use_case, grid_name]))
        return a

    def zip_glob(self, path_with_asterisk):
        """Performs glob.glob like function for zipped file, path must contain a single asterisk."""

        assert self.zipped

        star = path_with_asterisk.find('*')
        assert star >= 0, 'no asterisk in zip glob path'
        no_tail = (star == len(path_with_asterisk) - 1)
        match_list = []
        with zf.ZipFile(self.zip_file) as zfp:
            zip_list = zfp.namelist()
            for name in zip_list:
                if ((star == 0 or name.startswith(path_with_asterisk[:star])) and
                    (no_tail or name.endswith(path_with_asterisk[star + 1:]))):
                    match_list.append(name)
        return match_list


def bad_keyword(key):
    """Return False if key is a valid keyword, otherwise True."""
    if not key:
        return True
    if key.isalnum():
        return False
    for c in key:
        if not c.isalnum() and c not in '-_':
            return True
    return False
