"""Class for a collection of grid properties"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.property as rqp
import resqpy.olio.ab_toolbox as abt
import resqpy.olio.box_utilities as bxu
import resqpy.olio.load_data as ld
import resqpy.olio.write_data as wd
import resqpy.olio.xml_et as rqet
import resqpy.property.property_common as rqp_c


class GridPropertyCollection(rqp.PropertyCollection):
    """Class for RESQML Property collection for an IJK Grid, inheriting from PropertyCollection."""

    def __init__(self, grid = None, property_set_root = None, realization = None):
        """Creates a new property collection related to an IJK grid.

        arguments:
           grid (grid.Grid object, optional): must be present unless creating a completely blank, empty collection
           property_set_root (optional): if present, the collection is populated with the properties defined in the xml tree
              of the property set; grid must not be None when using this argument
           realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
              if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
              realisation number), or is for properties that do not have multiple realizations

        returns:
           the new GridPropertyCollection object

        note:
           usually a grid should be passed, however a completely blank collection may be created prior to using
           collection inheritance methods to populate from another collection, in which case the grid can be lazily left
           as None here

        :meta common:
        """

        if grid is not None:
            log.debug('initialising grid property collection for grid: ' + str(rqet.citation_title_for_node(grid.root)))
            log.debug('grid uuid: ' + str(grid.uuid))
        super().__init__(support = grid, property_set_root = property_set_root, realization = realization)
        self._copy_support_to_grid_attributes()

        # NB: RESQML documentation is not clear which order is correct; should be kept consistent with same data in fault.py
        # face_index_map maps from (axis, p01) to face index value in range 0..5
        #     self.face_index_map = np.array([[0, 1], [4, 2], [5, 3]], dtype = int)
        self.face_index_map = np.array([[0, 1], [2, 4], [5, 3]], dtype = int)  # order: top, base, J-, I+, J+, I-
        # and the inverse, maps from 0..5 to (axis, p01)
        #     self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 1], [2, 1], [1, 0], [2, 0]], dtype = int)
        self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [1, 1], [2, 0]], dtype = int)

    def _copy_support_to_grid_attributes(self):
        # following three pseudonyms are for backward compatibility
        self.grid = self.support
        self.grid_root = self.support_root
        self.grid_uuid = self.support_uuid

    def set_grid(self, grid, grid_root = None, modify_parts = True):
        """Sets the supporting representation object for the collection to be the given grid.

        note:
           this method does not need to be called if the grid was passed during object initialisation.
        """

        self.set_support(support = grid, modify_parts = modify_parts)
        self._copy_support_to_grid_attributes()

    def h5_slice_for_box(self, part, box):
        """Returns a subset of the array for part, without loading the whole array (unless already cached).

        arguments:
           part (string): the part name for which the array slice is required
           box (numpy int array of shape (2, 3)): the min, max indices for K, J, I axes for which an array
              extract is required

        returns:
           numpy array that is a hyper-slice of the hdf5 array

        note:
           this method always fetches from the hdf5 file and does not attempt local caching; the whole array
           is not loaded; all axes continue to exist in the returned array, even where the sliced extent of
           an axis is 1; the upper indices indicated in the box are included in the data (unlike the python
           protocol)
        """

        slice_tuple = (slice(box[0, 0], box[1, 0] + 1), slice(box[0, 1],
                                                              box[1, 1] + 1), slice(box[0, 2], box[1, 2] + 1))
        return self.h5_slice(part, slice_tuple)

    def extend_imported_list_copying_properties_from_other_grid_collection(self,
                                                                           other,
                                                                           box = None,
                                                                           refinement = None,
                                                                           coarsening = None,
                                                                           realization = None,
                                                                           copy_all_realizations = False,
                                                                           uncache_other_arrays = True):
        """Extends this collection's imported list with properties from other collection.

        Optionally extract for a box.

        arguments:
           other: another GridPropertyCollection object which might relate to a different grid object
           box: (numpy int array of shape (2, 3), optional): if present, a logical ijk cuboid subset of the source arrays
              is extracted, box axes are (min:max, kji0); if None, the full arrays are copied
           refinement (resqpy.olio.fine_coarse.FineCoarse object, optional): if present, other is taken to be a collection
              for a coarser grid and the property values are sampled for a finer grid based on the refinement mapping
           coarsening (resqpy.olio.fine_coarse.FineCoarse object, optional): if present, other is taken to be a collection
              for a finer grid and the property values are upscaled for a coarser grid based on the coarsening mapping
           realization (int, optional): if present, only properties for this realization are copied; if None, only
              properties without a realization number are copied unless copy_all_realizations is True
           copy_all_realizations (boolean, default False): if True (and realization is None), all properties are copied;
              if False, only properties with a realization of None are copied; ignored if realization is not None
           uncache_other_arrays (boolean, default True): if True, after each array is copied, the original is uncached from
              the source collection (to free up memory)

        notes:
           this function can be used to copy properties between one grid object and another compatible one, for example
           after a grid manipulation has generated a new version of the geometry; it can also be used to select an ijk box
           subset of the property data and/or refine or coarsen the property data;
           if a box is supplied, the first index indicates min (0) or max (1) and the second index indicates k (0), j (1) or i (2);
           the values in box are zero based and cells matching the maximum indices are included (unlike with python ranges);
           when coarsening or refining, this function ignores the 'within...box' attributes of the FineCoarse object so
           the box argument must be used to effect a local grid coarsening or refinement
        """

        # todo: optional use of active cell mask in coarsening

        _extend_imported_initial_assertions(other, box, refinement, coarsening)

        if coarsening is not None:  # static upscaling of key property kinds, simple sampling of others
            _extend_imported_with_coarsening(self, other, box, coarsening, realization, copy_all_realizations,
                                             uncache_other_arrays)

        else:
            _extend_imported_no_coarsening(self, other, box, refinement, realization, copy_all_realizations,
                                           uncache_other_arrays)

    def import_nexus_property_to_cache(self,
                                       file_name,
                                       keyword,
                                       extent_kji = None,
                                       discrete = False,
                                       uom = None,
                                       time_index = None,
                                       null_value = None,
                                       property_kind = None,
                                       local_property_kind_uuid = None,
                                       facet_type = None,
                                       facet = None,
                                       realization = None,
                                       use_binary = True):
        """Reads a property array from an ascii (or pure binary) file, caches and adds to imported list.

        Does not add to collection dict.

        arguments:
           file_name (string): the name of the file to read the array data from; should contain data for one array only, without
              the keyword
           keyword (string): the keyword to associate with the imported data, which will become the citation title
           extent_kji (optional, default None): if present, [nk, nj, ni] being the extent (shape) of the array to be imported;
              if None, the shape of the grid associated with this collection is used
           discrete (boolean, optional, default False): if True, integer data is imported, if False, float data
           uom (string, optional, default None): the resqml units of measure applicable to the data
           time_index (integer, optional, default None): if present, this array is for time varying data and this value is the
              index into a time series associated with this collection
           null_value (int or float, optional, default None): if present, this is used in the metadata to indicate that
              this value is to be interpreted as a null value wherever it appears in the data (this does not change the
              data during import)
           property_kind (string): resqml property kind, or None
           local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None for standard property kind
           facet_type (string): resqml facet type, or None
           facet (string): resqml facet, or None
           realization (int): realization number, or None
           use_binary (boolean, optional, default True): if True, and an up-to-date pure binary version of the file exists,
              then the data is loaded from that instead of from the ascii file; if True but the binary version does not
              exist (or is older than the ascii file), the pure binary version is written as a side effect of the import;
              if False, the ascii file is used even if a pure binary equivalent exists, and no binary file is written

        note:
           this function only performs the first importation step of actually reading the array into memory; other steps
           must follow to include the array as a part in the resqml model and in this collection of properties (see doc
           string for add_cached_array_to_imported_list() for the sequence of steps)
        """

        log.debug(f'importing {keyword} array from file {file_name}')
        # note: code in resqml_import adds imported arrays to model and to dict
        if self.imported_list is None:
            self.imported_list = []
        if extent_kji is None:
            extent_kji = self.grid.extent_kji
        if discrete:
            data_type = 'integer'
        else:
            data_type = 'real'
        try:
            import_array = ld.load_array_from_file(file_name,
                                                   extent_kji,
                                                   data_type = data_type,
                                                   comment_char = '!',
                                                   data_free_of_comments = False,
                                                   use_binary = use_binary)
        except Exception:
            log.exception('failed to import {} array from file {}'.format(keyword, file_name))
            return None

        self.add_cached_array_to_imported_list(import_array,
                                               file_name,
                                               keyword,
                                               discrete = discrete,
                                               uom = uom,
                                               time_index = time_index,
                                               null_value = null_value,
                                               property_kind = property_kind,
                                               local_property_kind_uuid = local_property_kind_uuid,
                                               facet_type = facet_type,
                                               facet = facet,
                                               realization = realization,
                                               points = False)
        return import_array

    def import_vdb_static_property_to_cache(self,
                                            vdbase,
                                            keyword,
                                            grid_name = 'ROOT',
                                            uom = None,
                                            realization = None,
                                            property_kind = None,
                                            facet_type = None,
                                            facet = None):
        """Reads a vdb static property array, caches and adds to imported list (but not collection dict).

        arguments:
           vdbase: an object of class vdb.VDB, already initialised with the path of the vdb
           keyword (string): the Nexus keyword (or equivalent) of the static property to be loaded
           grid_name (string): the grid name as used in the vdb
           uom (string): The resqml unit of measure that applies to the data
           realization (optional, int): The realization number that this property belongs to; use None
              if not applicable
           property_kind (string, optional): the RESQML property kind of the property
           facet_type (string, optional): a RESQML facet type for the property
           facet (string, optional): the RESQML facet value for the given facet type for the property

        returns:
           cached array containing the property data; the cached array is in an unpacked state
           (ie. can be directly indexed with [k0, j0, i0])

        note:
           when importing from a vdb (or other sources), use methods such as this to build up a list
           of imported arrays; then write hdf5 for the imported arrays; finally create xml for imported
           properties
        """

        log.info('importing vdb static property {} array'.format(keyword))
        keyword = keyword.upper()
        try:
            discrete = True
            dtype = None
            if keyword[0].upper() == 'I' or keyword in ['KID', 'UID', 'UNPACK', 'DAD']:
                # coerce to integer values (vdb stores integer data as reals!)
                dtype = 'int32'
            elif keyword in ['DEADCELL', 'LIVECELL']:
                dtype = 'bool'  # could use the default dtype of 64 bit integer
            else:
                dtype = 'float'  # convert to 64 bit; could omit but RESQML states 64 bit
                discrete = False
            import_array = vdbase.grid_static_property(grid_name, keyword, dtype = dtype)
            assert import_array is not None
        except Exception:
            log.exception(f'failed to import static property {keyword} from vdb')
            return None

        self.add_cached_array_to_imported_list(import_array,
                                               vdbase.path,
                                               keyword,
                                               discrete = discrete,
                                               uom = uom,
                                               time_index = None,
                                               null_value = None,
                                               realization = realization,
                                               property_kind = property_kind,
                                               facet_type = facet_type,
                                               facet = facet)
        return import_array

    def import_vdb_recurrent_property_to_cache(self,
                                               vdbase,
                                               timestep,
                                               keyword,
                                               grid_name = 'ROOT',
                                               time_index = None,
                                               uom = None,
                                               realization = None,
                                               property_kind = None,
                                               facet_type = None,
                                               facet = None):
        """Reads a vdb recurrent property array for one timestep, caches and adds to imported list.

        Does not add to collection dict.

        arguments:
           vdbase: an object of class vdb.VDB, already initialised with the path of the vdb
           timestep (int): the Nexus timestep number at which the property array was generated; NB. this is
              not necessarily the same as a resqml time index
           keyword (string): the Nexus keyword (or equivalent) of the recurrent property to be loaded
           grid_name (string): the grid name as used in the vdb
           time_index (int, optional): if present, used as the time index, otherwise timestep is used
           uom (string): The resqml unit of measure that applies to the data
           realization (optional, int): the realization number that this property belongs to; use None
              if not applicable
           property_kind (string, optional): the RESQML property kind of the property
           facet_type (string, optional): a RESQML facet type for the property
           facet (string, optional): the RESQML facet value for the given facet type for the property

        returns:
           cached array containing the property data; the cached array is in an unpacked state
           (ie. can be directly indexed with [k0, j0, i0])

        notes:
           when importing from a vdb (or other sources), use methods such as this to build up a list
           of imported arrays; then write hdf5 for the imported arrays; finally create xml for imported
           properties
        """

        log.info('importing vdb recurrent property {0} array for timestep {1}'.format(keyword, str(timestep)))
        if time_index is None:
            time_index = timestep
        keyword = keyword.upper()
        try:
            import_array = vdbase.grid_recurrent_property_for_timestep(grid_name, keyword, timestep, dtype = 'float')
            assert import_array is not None
        except Exception:
            # could raise an exception (as for static properties)
            log.error(f'failed to import recurrent property {keyword} from vdb for timestep {timestep}')
            return None

        self.add_cached_array_to_imported_list(import_array,
                                               vdbase.path,
                                               keyword,
                                               discrete = False,
                                               uom = uom,
                                               time_index = time_index,
                                               null_value = None,
                                               realization = realization,
                                               property_kind = property_kind,
                                               facet_type = facet_type,
                                               facet = facet)
        return import_array

    def import_ab_property_to_cache(self,
                                    file_name,
                                    keyword,
                                    extent_kji = None,
                                    discrete = None,
                                    uom = None,
                                    time_index = None,
                                    null_value = None,
                                    property_kind = None,
                                    local_property_kind_uuid = None,
                                    facet_type = None,
                                    facet = None,
                                    realization = None):
        """Reads a property array from a pure binary file, caches and adds to imported list (but not collection dict).

        arguments:
           file_name (string): the name of the file to read the array data from; should contain data for one array only in
              'pure binary' format (as used by ab_* suite of utilities)
           keyword (string): the keyword to associate with the imported data, which will become the citation title
           extent_kji (optional, default None): if present, [nk, nj, ni] being the extent (shape) of the array to be imported;
              if None, the shape of the grid associated with this collection is used
           discrete (boolean, optional, default False): if True, integer data is imported, if False, float data
           uom (string, optional, default None): the resqml units of measure applicable to the data
           time_index (integer, optional, default None): if present, this array is for time varying data and this value is the
              index into a time series associated with this collection
           null_value (int or float, optional, default None): if present, this is used in the metadata to indicate that
              this value is to be interpreted as a null value wherever it appears in the data (this does not change the
              data during import)
           property_kind (string): resqml property kind, or None
           local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
           facet_type (string): resqml facet type, or None
           facet (string): resqml facet, or None
           realization (int): realization number, or None

        note:
           this function only performs the first importation step of actually reading the array into memory; other steps
           must follow to include the array as a part in the resqml model and in this collection of properties (see doc
           string for add_cached_array_to_imported_list() for the sequence of steps)
        """

        if self.imported_list is None:
            self.imported_list = []
        if extent_kji is None:
            extent_kji = self.grid.extent_kji
        assert file_name[-3:] in ['.db', '.fb', '.ib', '.lb',
                                  '.bb'], 'file extension not in pure binary array expected set for: ' + file_name
        if discrete is None:
            discrete = (file_name[-3:] in ['.ib', '.lb', '.bb'])
        else:
            assert discrete == (file_name[-3:]
                                in ['.ib', '.lb',
                                    '.bb']), 'discrete argument is not consistent with file extension for: ' + file_name
        try:
            import_array = abt.load_array_from_ab_file(
                file_name, extent_kji, return_64_bit = False)  # todo: RESQML indicates 64 bit for everything
        except Exception:
            log.exception('failed to import property from pure binary file: ' + file_name)
            return None

        self.add_cached_array_to_imported_list(import_array,
                                               file_name,
                                               keyword,
                                               discrete = discrete,
                                               uom = uom,
                                               time_index = time_index,
                                               null_value = null_value,
                                               property_kind = property_kind,
                                               local_property_kind_uuid = local_property_kind_uuid,
                                               facet_type = facet_type,
                                               facet = facet,
                                               realization = realization)
        return import_array

    def decoarsen_imported_list(self, decoarsen_array = None, reactivate = True):
        """Decoarsen imported Nexus properties if needed.

        arguments:
           decoarsen_array (int array, optional): if present, the naturalised cell index of the coarsened host cell, for each fine cell;
              if None, the ICOARS keyword is searched for in the imported list and if not found KID data is used to derive the mapping
           reactivate (boolean, default True): if True, the parent grid will have decoarsened cells' inactive flag set to that of the
              host cell

        returns:
           a copy of the array used for decoarsening, if established, or None if no decoarsening array was identified

        notes:
           a return value of None indicates that no decoarsening occurred;
           coarsened values are redistributed quite naively, with coarse volumes being split equally between fine cells, similarly for
           length and area based properties; default used for most properties is simply to replicate the coarse value;
           the ICOARS array itself is left unchanged, which means the method should only be called once for an imported list;
           if no array is passed and no ICOARS array found, the KID values are inspected and the decoarsen array reverse engineered;
           the method must be called before the imported arrays are written to hdf5;
           reactivation only modifies the grid object attribute and does not write to hdf5, so the method should be called prior to
           writing the grid in this situation
        """
        # imported_list is list pf:
        # (0: uuid, 1: file_name, 2: keyword, 3: cached_name, 4: discrete, 5: uom, 6: time_index, 7: null_value, 8: min_value, 9: max_value,
        # 10: property_kind, 11: facet_type, 12: facet, 13: realization, 14: indexable_element, 15: count, 16: local_property_kind_uuid,
        # 17: const_value)

        skip_keywords = ['UID', 'ICOARS', 'KID', 'DAD']  # TODO: complete this list
        decoarsen_length_kinds = ['length', 'cell length', 'thickness', 'permeability thickness', 'permeability length']
        decoarsen_area_kinds = ['transmissibility']
        decoarsen_volume_kinds = ['volume', 'rock volume', 'pore volume', 'fluid volume']

        assert self.grid is not None

        kid_attr_name = None
        k_share = j_share = i_share = None

        if decoarsen_array is None:
            for import_item in self.imported_list:
                if (import_item[14] is None or import_item[14] == 'cells') and import_item[4] and hasattr(
                        self, import_item[3]):
                    if import_item[2] == 'ICOARS':
                        decoarsen_array = self.__dict__[import_item[3]] - 1  # ICOARS values are one based
                        break
                    if import_item[2] == 'KID':
                        kid_attr_name = import_item[3]

        if decoarsen_array is None and kid_attr_name is not None:
            kid = self.__dict__[kid_attr_name]
            kid_mask = (kid == -3)  # -3 indicates cell inactive due to coarsening
            assert kid_mask.shape == tuple(self.grid.extent_kji)
            if np.any(kid_mask):
                log.debug(f'{np.count_nonzero(kid_mask)} cells marked as requiring decoarsening in KID data')
                decoarsen_array = np.full(self.grid.extent_kji, -1, dtype = int)
                k_share = np.zeros(self.grid.extent_kji, dtype = int)
                j_share = np.zeros(self.grid.extent_kji, dtype = int)
                i_share = np.zeros(self.grid.extent_kji, dtype = int)
                natural = 0
                for k0 in range(self.grid.nk):
                    for j0 in range(self.grid.nj):
                        for i0 in range(self.grid.ni):
                            #                     if decoarsen_array[k0, j0, i0] < 0:
                            if kid[k0, j0, i0] == 0:
                                #                        assert not kid_mask[k0, j0, i0]
                                ke = k0 + 1
                                while ke < self.grid.nk and kid_mask[ke, j0, i0]:
                                    ke += 1
                                je = j0 + 1
                                while je < self.grid.nj and kid_mask[k0, je, i0]:
                                    je += 1
                                ie = i0 + 1
                                while ie < self.grid.ni and kid_mask[k0, j0, ie]:
                                    ie += 1
                                # todo: check for conflict and resolve
                                decoarsen_array[k0:ke, j0:je, i0:ie] = natural
                                k_share[k0:ke, j0:je, i0:ie] = ke - k0
                                j_share[k0:ke, j0:je, i0:ie] = je - j0
                                i_share[k0:ke, j0:je, i0:ie] = ie - i0
                            elif not kid_mask[k0, j0, i0]:  # inactive for reasons other than coarsening
                                decoarsen_array[k0, j0, i0] = natural
                                k_share[k0, j0, i0] = 1
                                j_share[k0, j0, i0] = 1
                                i_share[k0, j0, i0] = 1
                            natural += 1
                assert np.all(decoarsen_array >= 0)

        if decoarsen_array is None:
            return None

        cell_count = decoarsen_array.size
        host_count = len(np.unique(decoarsen_array))
        log.debug(f'{host_count} of {cell_count} are hosts; difference is {cell_count - host_count}')
        assert cell_count == self.grid.cell_count()
        if np.all(decoarsen_array.flatten() == np.arange(cell_count, dtype = int)):
            return None  # identity array

        if k_share is None:
            sharing_needed = False
            for import_item in self.imported_list:
                kind = import_item[10]
                if kind in decoarsen_volume_kinds or kind in decoarsen_area_kinds or kind in decoarsen_length_kinds:
                    sharing_needed = True
                    break
            if sharing_needed:
                k_share = np.zeros(self.grid.extent_kji, dtype = int)
                j_share = np.zeros(self.grid.extent_kji, dtype = int)
                i_share = np.zeros(self.grid.extent_kji, dtype = int)
                natural = 0
                for k0 in range(self.grid.nk):
                    for j0 in range(self.grid.nj):
                        for i0 in range(self.grid.ni):
                            if k_share[k0, j0, i0] == 0:
                                ke = k0 + 1
                                while ke < self.grid.nk and decoarsen_array[ke, j0, i0] == natural:
                                    ke += 1
                                je = j0 + 1
                                while je < self.grid.nj and decoarsen_array[k0, je, i0] == natural:
                                    je += 1
                                ie = i0 + 1
                                while ie < self.grid.ni and decoarsen_array[k0, j0, ie] == natural:
                                    ie += 1
                                k_share[k0:ke, j0:je, i0:ie] = ke - k0
                                j_share[k0:ke, j0:je, i0:ie] = je - j0
                                i_share[k0:ke, j0:je, i0:ie] = ie - i0
                            natural += 1

        if k_share is not None:
            assert np.all(k_share > 0) and np.all(j_share > 0) and np.all(i_share > 0)
            volume_share = (k_share * j_share * i_share).astype(float)
            k_share = k_share.astype(float)
            j_share = j_share.astype(float)
            i_share = i_share.astype(float)

        property_count = 0
        for import_item in self.imported_list:
            if import_item[3] is None or not hasattr(self, import_item[3]):
                continue  # todo: handle decoarsening of const arrays?
            if import_item[14] is not None and import_item[14] != 'cells':
                continue
            coarsened = self.__dict__[import_item[3]].flatten()
            assert coarsened.size == cell_count
            keyword = import_item[2]
            if keyword.upper() in skip_keywords:
                continue
            kind = import_item[10]
            if kind in decoarsen_volume_kinds:
                redistributed = coarsened[decoarsen_array] / volume_share
            elif kind in decoarsen_area_kinds:
                # only transmissibilty currently in this set of supported property kinds
                log.warning(
                    f'decoarsening of transmissibility {keyword} skipped due to simple methods not yielding correct values'
                )
            elif kind in decoarsen_length_kinds:
                facet_dir = import_item[12] if import_item[11] == 'direction' else None
                if kind in ['thickness', 'permeability thickness'] or (facet_dir == 'K'):
                    redistributed = coarsened[decoarsen_array] / k_share
                elif facet_dir == 'J':
                    redistributed = coarsened[decoarsen_array] / j_share
                elif facet_dir == 'I':
                    redistributed = coarsened[decoarsen_array] / i_share
                else:
                    log.warning(f'decoarsening of length property {keyword} skipped as direction not established')
            else:
                redistributed = coarsened[decoarsen_array]
            self.__dict__[import_item[3]] = redistributed.reshape(self.grid.extent_kji)
            property_count += 1

        if property_count:
            log.debug(f'{property_count} properties decoarsened')

        if reactivate and hasattr(self.grid, 'inactive'):
            log.debug('reactivating cells inactive due to coarsening')
            pre_count = np.count_nonzero(self.grid.inactive)
            self.grid.inactive = self.grid.inactive.flatten()[decoarsen_array].reshape(self.grid.extent_kji)
            post_count = np.count_nonzero(self.grid.inactive)
            log.debug(f'{pre_count - post_count} cells reactivated')

        return decoarsen_array

    def write_nexus_property(
            self,
            part,
            file_name,
            keyword = None,
            headers = True,
            append = False,
            columns = 20,
            decimals = 3,  # note: decimals only applicable to real numbers
            blank_line_after_i_block = True,
            blank_line_after_j_block = False,
            space_separated = False,  # default is tab separated
            use_binary = False,
            binary_only = False,
            nan_substitute_value = None):
        """Writes the property array to a file in a format suitable for including as nexus input.

        arguments:
           part (string): the part name for which the array is to be exported
           file_name (string): the path of the file to be created (any existing file will be overwritten)
           keyword (string, optional, default None): if not None, the Nexus keyword to be included in the
              ascii export file (otherwise data only is written, without a keyword)
           headers (boolean, optional, default True): if True, some header comments are included in the
              ascii export file, using a Nexus comment character
           append (boolean, optional, default False): if True, any existing file is appended to rather than
              overwritten
           columns (integer, optional, default 20): the maximum number of data items to be written per line
           decimals (integer, optional, default 3): the number of decimal places included in the values
              written to the ascii export file (ignored for integer data)
           blank_line_after_i_block (boolean, optional, default True): if True, a blank line is inserted
              after each I-block of data (ie. when the J index changes)
           blank_line_after_j_block (boolean, optional, default False): if True, a blank line is inserted
              after each J-block of data (ie. when the K index changes)
           space_separated (boolean, optional, default False): if True, a space is inserted between values;
              if False, a tab is used
           use_binary (boolean, optional, default False): if True, a pure binary copy of the array is
              written
           binary_only (boolean, optional, default False): if True, and if use_binary is True, then no
              ascii file is generated; if False (or if use_binary is False) then an ascii file is written
           nan_substitute_value (float, optional, default None): if a value is supplied, any not-a-number
              values are replaced with this value in the exported file (the cached property array remains
              unchanged); if None, then 'nan' or 'Nan' will appear in the ascii export file
        """

        array_ref = self.cached_part_array_ref(part)
        assert (array_ref is not None)
        extent_kji = array_ref.shape
        assert (len(extent_kji) == 3)
        wd.write_array_to_ascii_file(file_name,
                                     extent_kji,
                                     array_ref,
                                     headers = headers,
                                     keyword = keyword,
                                     columns = columns,
                                     data_type = rqet.simplified_data_type(array_ref.dtype),
                                     decimals = decimals,
                                     target_simulator = 'nexus',
                                     blank_line_after_i_block = blank_line_after_i_block,
                                     blank_line_after_j_block = blank_line_after_j_block,
                                     space_separated = space_separated,
                                     append = append,
                                     use_binary = use_binary,
                                     binary_only = binary_only,
                                     nan_substitute_value = nan_substitute_value)

    def write_nexus_property_generating_filename(
            self,
            part,
            directory,
            use_title_for_keyword = False,
            headers = True,
            columns = 20,
            decimals = 3,  # note: decimals only applicable to real numbers
            blank_line_after_i_block = True,
            blank_line_after_j_block = False,
            space_separated = False,  # default is tab separated
            use_binary = False,
            binary_only = False,
            nan_substitute_value = None):
        """Writes the property array to a file using a filename generated from the citation title etc.

        arguments:
           part (string): the part name for which the array is to be exported
           directory (string): the path of the diractory into which the file will be written
           use_title_for_keyword (boolean, optional, default False): if True, the citation title for the property part
              is used as a keyword in the ascii export file
           for other arguments, see the docstring for the write_nexus_property() function

        note:
           the generated filename consists of:
              the citation title (with spaces replaced with underscores);
              the facet type and facet, if present;
              _t_ and the time_index, if the part has a time index
              _r_ and the realisation number, if the part has a realisation number
        """

        title = self.citation_title_for_part(part).replace(' ', '_')
        if use_title_for_keyword:
            keyword = title
        else:
            keyword = None
        fname = title
        facet_type = self.facet_type_for_part(part)
        if facet_type is not None:
            fname += '_' + facet_type.replace(' ', '_') + '_' + self.facet_for_part(part).replace(' ', '_')
        time_index = self.time_index_for_part(part)
        if time_index is not None:
            fname += '_t_' + str(time_index)
        realisation = self.realization_for_part(part)
        if realisation is not None:
            fname += '_r_' + str(realisation)
        # could add .dat extension
        self.write_nexus_property(part,
                                  os.path.join(directory, fname),
                                  keyword = keyword,
                                  headers = headers,
                                  append = False,
                                  columns = columns,
                                  decimals = decimals,
                                  blank_line_after_i_block = blank_line_after_i_block,
                                  blank_line_after_j_block = blank_line_after_j_block,
                                  space_separated = space_separated,
                                  use_binary = use_binary,
                                  binary_only = binary_only,
                                  nan_substitute_value = nan_substitute_value)

    def write_nexus_collection(self,
                               directory,
                               use_title_for_keyword = False,
                               headers = True,
                               columns = 20,
                               decimals = 3,
                               blank_line_after_i_block = True,
                               blank_line_after_j_block = False,
                               space_separated = False,
                               use_binary = False,
                               binary_only = False,
                               nan_substitute_value = None):
        """Writes a set of files, one for each part in the collection.

        arguments:
           directory (string): the path of the diractory into which the files will be written
           for other arguments, see the docstrings for the write_nexus_property_generating_filename() and
              write_nexus_property() functions

        note:
           the generated filenames are based on the citation titles etc., as for
           write_nexus_property_generating_filename()
        """

        for part in self.dict.keys():
            self.write_nexus_property_generating_filename(part,
                                                          directory,
                                                          use_title_for_keyword = use_title_for_keyword,
                                                          headers = headers,
                                                          columns = columns,
                                                          decimals = decimals,
                                                          blank_line_after_i_block = blank_line_after_i_block,
                                                          blank_line_after_j_block = blank_line_after_j_block,
                                                          space_separated = space_separated,
                                                          use_binary = use_binary,
                                                          binary_only = binary_only,
                                                          nan_substitute_value = nan_substitute_value)


def _array_box(collection, part, box = None, uncache_other_arrays = True):
    full_array = collection.cached_part_array_ref(part)
    if box is None:
        a = full_array.copy()
    else:
        a = full_array[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1, box[0, 2]:box[1, 2] + 1].copy()
    full_array = None
    if uncache_other_arrays:
        collection.uncache_part_array(part)
    return a


def _coarsening_sample(coarsening, a):
    # for now just take value from first cell in box
    # todo: find most common element in box
    a_coarsened = np.empty(tuple(coarsening.coarse_extent_kji), dtype = a.dtype)
    assert a.shape == tuple(coarsening.fine_extent_kji)
    # todo: try to figure out some numpy slice operations to avoid use of for loops
    for k in range(coarsening.coarse_extent_kji[0]):
        for j in range(coarsening.coarse_extent_kji[1]):
            for i in range(coarsening.coarse_extent_kji[2]):
                # local box within lgc space of fine cells, for 1 coarse cell
                cell_box = coarsening.fine_box_for_coarse((k, j, i))
                a_coarsened[k, j, i] = a[tuple(cell_box[0])]
    return a_coarsened


def _coarsening_sum(coarsening, a, axis = None):
    a_coarsened = np.empty(tuple(coarsening.coarse_extent_kji))
    assert a.shape == tuple(coarsening.fine_extent_kji)
    # todo: try to figure out some numpy slice operations to avoid use of for loops
    for k in range(coarsening.coarse_extent_kji[0]):
        for j in range(coarsening.coarse_extent_kji[1]):
            for i in range(coarsening.coarse_extent_kji[2]):
                cell_box = coarsening.fine_box_for_coarse(
                    (k, j, i))  # local box within lgc space of fine cells, for 1 coarse cell
                # yapf: disable
                a_coarsened[k, j, i] = np.nansum(a[cell_box[0, 0]:cell_box[1, 0] + 1,
                                                 cell_box[0, 1]:cell_box[1, 1] + 1,
                                                 cell_box[0, 2]:cell_box[1, 2] + 1])
                # yapf: enable
                if axis is not None:
                    axis_1 = (axis + 1) % 3
                    axis_2 = (axis + 2) % 3
                    # yapf: disable
                    divisor = ((cell_box[1, axis_1] + 1 - cell_box[0, axis_1]) *
                               (cell_box[1, axis_2] + 1 - cell_box[0, axis_2]))
                    # yapf: enable
                    a_coarsened[k, j, i] = a_coarsened[k, j, i] / float(divisor)
    return a_coarsened


def _coarsening_weighted_mean(coarsening, a, fine_weight, coarse_weight = None, zero_weight_result = np.NaN):
    a_coarsened = np.empty(tuple(coarsening.coarse_extent_kji))
    assert a.shape == tuple(coarsening.fine_extent_kji)
    assert fine_weight.shape == a.shape
    if coarse_weight is not None:
        assert coarse_weight.shape == a_coarsened.shape
    for k in range(coarsening.coarse_extent_kji[0]):
        for j in range(coarsening.coarse_extent_kji[1]):
            for i in range(coarsening.coarse_extent_kji[2]):
                _coarsening_weighted_mean_singlecell(a_coarsened, a, coarsening, k, j, i, fine_weight, coarse_weight,
                                                     zero_weight_result)
    if coarse_weight is not None:
        mask = np.logical_or(np.isnan(coarse_weight), coarse_weight == 0.0)
        a_coarsened = np.where(mask, zero_weight_result, a_coarsened / coarse_weight)
    return a_coarsened


def _coarsening_weighted_mean_singlecell(a_coarsened, a, coarsening, k, j, i, fine_weight, coarse_weight,
                                         zero_weight_result):
    cell_box = coarsening.fine_box_for_coarse((k, j, i))  # local box within lgc space of fine cells, for 1 coarse cell
    a_coarsened[k, j, i] = np.nansum(
        a[cell_box[0, 0]:cell_box[1, 0] + 1, cell_box[0, 1]:cell_box[1, 1] + 1, cell_box[0, 2]:cell_box[1, 2] + 1] *
        fine_weight[cell_box[0, 0]:cell_box[1, 0] + 1, cell_box[0, 1]:cell_box[1, 1] + 1,
                    cell_box[0, 2]:cell_box[1, 2] + 1])
    if coarse_weight is None:
        weight = np.nansum(fine_weight[cell_box[0, 0]:cell_box[1, 0] + 1, cell_box[0, 1]:cell_box[1, 1] + 1,
                                       cell_box[0, 2]:cell_box[1, 2] + 1])
        if np.isnan(weight) or weight == 0.0:
            a_coarsened[k, j, i] = zero_weight_result
        else:
            a_coarsened[k, j, i] /= weight


def _add_to_imported(collection, a, title, info, null_value = None, const_value = None):
    collection.add_cached_array_to_imported_list(
        a,
        title,
        info[10],  # citation_title
        discrete = not info[4],
        indexable_element = 'cells',
        uom = info[15],
        time_index = info[12],
        null_value = null_value,
        property_kind = info[7],
        local_property_kind_uuid = info[17],
        facet_type = info[8],
        facet = info[9],
        realization = info[0],
        const_value = const_value,
        points = info[21],
        time_series_uuid = info[11])


def _extend_imported_initial_assertions(other, box, refinement, coarsening):
    import resqpy.grid as grr  # at global level was causing issues due to circular references, ie. grid importing this module
    assert other.support is not None and isinstance(other.support,
                                                    grr.Grid), 'other property collection has no grid support'
    assert refinement is None or coarsening is None, 'refinement and coarsening both specified simultaneously'

    if box is not None:
        assert bxu.valid_box(box, other.grid.extent_kji)
        if refinement is not None:
            assert tuple(bxu.extent_of_box(box)) == tuple(refinement.coarse_extent_kji)
        elif coarsening is not None:
            assert tuple(bxu.extent_of_box(box)) == tuple(coarsening.fine_extent_kji)
    # todo: any contraints on realization numbers ?


def _extend_imported_with_coarsening(collection, other, box, coarsening, realization, copy_all_realizations,
                                     uncache_other_arrays):
    assert collection.support is not None and tuple(collection.support.extent_kji) == tuple(
        coarsening.coarse_extent_kji)

    source_rv, source_ntg, source_poro, source_sat, source_perm = _extend_imported_get_fine_collections(
        other, realization)

    fine_rv_array, coarse_rv_array = _extend_imported_coarsen_rock_volume(source_rv, other, box, collection,
                                                                          realization, uncache_other_arrays, coarsening,
                                                                          copy_all_realizations)

    fine_ntg_array, coarse_ntg_array = _extend_imported_coarsen_ntg(source_ntg, other, box, collection, realization,
                                                                    uncache_other_arrays, coarsening,
                                                                    copy_all_realizations, fine_rv_array,
                                                                    coarse_rv_array)

    fine_nrv_array, coarse_nrv_array = _extend_imported_nrv_arrays(fine_ntg_array, coarse_ntg_array, fine_rv_array,
                                                                   coarse_rv_array)

    fine_poro_array, coarse_poro_array = _extend_imported_coarsen_poro(source_poro, other, box, collection, realization,
                                                                       uncache_other_arrays, coarsening,
                                                                       copy_all_realizations, fine_nrv_array,
                                                                       coarse_nrv_array)

    _extend_imported_coarsen_sat(source_sat, other, box, collection, realization, uncache_other_arrays, coarsening,
                                 copy_all_realizations, fine_nrv_array, coarse_nrv_array, fine_poro_array,
                                 coarse_poro_array)

    _extend_imported_coarsen_perm(source_perm, other, box, collection, realization, uncache_other_arrays, coarsening,
                                  copy_all_realizations, fine_nrv_array, coarse_nrv_array)

    _extend_imported_coarsen_lengths(other, box, collection, realization, uncache_other_arrays, coarsening,
                                     copy_all_realizations)

    _extend_imported_coarsen_other(other, box, collection, realization, uncache_other_arrays, coarsening,
                                   copy_all_realizations, fine_rv_array, coarse_rv_array)


def _extend_imported_get_fine_collections(other, realization):
    # look for properties by kind, process in order: rock volume, net to gross ratio, porosity, permeability, saturation
    source_rv = rqp_c.selective_version_of_collection(other, realization = realization, property_kind = 'rock volume')
    source_ntg = rqp_c.selective_version_of_collection(other,
                                                       realization = realization,
                                                       property_kind = 'net to gross ratio')
    source_poro = rqp_c.selective_version_of_collection(other, realization = realization, property_kind = 'porosity')
    source_sat = rqp_c.selective_version_of_collection(other, realization = realization, property_kind = 'saturation')
    source_perm = rqp_c.selective_version_of_collection(other,
                                                        realization = realization,
                                                        property_kind = 'permeability rock')
    # todo: add kh and some other property kinds

    return source_rv, source_ntg, source_poro, source_sat, source_perm


def _extend_imported_coarsen_rock_volume(source_rv, other, box, collection, realization, uncache_other_arrays,
                                         coarsening, copy_all_realizations):
    # bulk rock volume
    fine_rv_array = coarse_rv_array = None
    if source_rv.number_of_parts() == 0:
        log.debug('computing bulk rock volume from fine and coarse grid geometries')
        source_rv_array = other.support.volume()
        if box is None:
            fine_rv_array = source_rv_array
        else:
            fine_rv_array = source_rv_array[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1, box[0, 2]:box[1, 2] + 1]
        coarse_rv_array = collection.support.volume()
    else:
        for (part, info) in source_rv.dict.items():
            if not copy_all_realizations and info[0] != realization:
                continue
            fine_rv_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
            coarse_rv_array = _coarsening_sum(coarsening, fine_rv_array)
            _add_to_imported(collection, coarse_rv_array, 'coarsened from grid ' + str(other.support.uuid), info)

    return fine_rv_array, coarse_rv_array


def _extend_imported_coarsen_ntg(source_ntg, other, box, collection, realization, uncache_other_arrays, coarsening,
                                 copy_all_realizations, fine_rv_array, coarse_rv_array):

    # net to gross ratio
    # note that coarsened ntg values may exceed one when reference bulk rock volumes are from grid geometries
    fine_ntg_array = coarse_ntg_array = None
    for (part, info) in source_ntg.dict.items():
        if not copy_all_realizations and info[0] != realization:
            continue
        fine_ntg_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
        coarse_ntg_array = _coarsening_weighted_mean(coarsening,
                                                     fine_ntg_array,
                                                     fine_rv_array,
                                                     coarse_weight = coarse_rv_array,
                                                     zero_weight_result = 0.0)
        _add_to_imported(collection, coarse_ntg_array, 'coarsened from grid ' + str(other.support.uuid), info)

    return fine_ntg_array, coarse_ntg_array


def _extend_imported_nrv_arrays(fine_ntg_array, coarse_ntg_array, fine_rv_array, coarse_rv_array):
    """Note: these arrays are generated only in memory for coarsening calculations for other properties. These are not added to the property collection"""
    if fine_ntg_array is None:
        fine_nrv_array = fine_rv_array
        coarse_nrv_array = coarse_rv_array
    else:
        fine_nrv_array = fine_rv_array * fine_ntg_array
        coarse_nrv_array = coarse_rv_array * coarse_ntg_array
    return fine_nrv_array, coarse_nrv_array


def _extend_imported_coarsen_poro(source_poro, other, box, collection, realization, uncache_other_arrays, coarsening,
                                  copy_all_realizations, fine_nrv_array, coarse_nrv_array):
    fine_poro_array = coarse_poro_array = None
    for (part, info) in source_poro.dict.items():
        if not copy_all_realizations and info[0] != realization:
            continue
        fine_poro_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
        coarse_poro_array = _coarsening_weighted_mean(coarsening,
                                                      fine_poro_array,
                                                      fine_nrv_array,
                                                      coarse_weight = coarse_nrv_array,
                                                      zero_weight_result = 0.0)
        _add_to_imported(collection, coarse_poro_array, 'coarsened from grid ' + str(other.support.uuid), info)

    return fine_poro_array, coarse_poro_array


def _extend_imported_coarsen_sat(source_sat, other, box, collection, realization, uncache_other_arrays, coarsening,
                                 copy_all_realizations, fine_nrv_array, coarse_nrv_array, fine_poro_array,
                                 coarse_poro_array):
    # saturations
    fine_sat_array = coarse_sat_array = None
    fine_sat_weight = fine_nrv_array
    coarse_sat_weight = coarse_nrv_array
    if fine_poro_array is not None:
        fine_sat_weight *= fine_poro_array
        coarse_sat_weight *= coarse_poro_array
    for (part, info) in source_sat.dict.items():
        if not copy_all_realizations and info[0] != realization:
            continue
        fine_sat_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
        coarse_sat_array = _coarsening_weighted_mean(coarsening,
                                                     fine_sat_array,
                                                     fine_sat_weight,
                                                     coarse_weight = coarse_sat_weight,
                                                     zero_weight_result = 0.0)
        _add_to_imported(collection, coarse_sat_array, 'coarsened from grid ' + str(other.support.uuid), info)


def _extend_imported_coarsen_perm(source_perm, other, box, collection, realization, uncache_other_arrays, coarsening,
                                  copy_all_realizations, fine_nrv_array, coarse_nrv_array):
    # permeabilities
    # todo: use more harmonic, arithmetic mean instead of just bulk rock volume weighted; consider ntg
    for (part, info) in source_perm.dict.items():
        if not copy_all_realizations and info[0] != realization:
            continue
        fine_perm_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
        coarse_perm_array = _coarsening_weighted_mean(coarsening,
                                                      fine_perm_array,
                                                      fine_nrv_array,
                                                      coarse_weight = coarse_nrv_array,
                                                      zero_weight_result = 0.0)
        _add_to_imported(collection, coarse_perm_array, 'coarsened from grid ' + str(other.support.uuid), info)


def _extend_imported_coarsen_lengths(other, box, collection, realization, uncache_other_arrays, coarsening,
                                     copy_all_realizations):
    # cell lengths
    source_cell_lengths = rqp_c.selective_version_of_collection(other,
                                                                realization = realization,
                                                                property_kind = 'cell length')
    for (part, info) in source_cell_lengths.dict.items():
        if not copy_all_realizations and info[0] != realization:
            continue
        fine_cl_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
        assert info[5] == 1  # count
        facet_type = info[8]
        facet = info[9]
        if not facet_type:
            (_, facet_type, facet) = rqp.property_kind_and_facet_from_keyword(info[10])
        assert facet_type == 'direction'
        axis = 'KJI'.index(facet[0].upper())
        coarse_cl_array = _coarsening_sum(coarsening, fine_cl_array, axis = axis)
        _add_to_imported(collection, coarse_cl_array, 'coarsened from grid ' + str(other.support.uuid), info)


def _extend_imported_coarsen_other(other, box, collection, realization, uncache_other_arrays, coarsening,
                                   copy_all_realizations, fine_rv_array, coarse_rv_array):
    # TODO: all other supported property kinds requiring special treatment
    # default behaviour is bulk volume weighted mean for continuous data, first cell in box for discrete
    handled_kinds = ('rock volume', 'net to gross ratio', 'porosity', 'saturation', 'permeability rock',
                     'rock permeability', 'cell length')
    for (part, info) in other.dict.items():
        if not copy_all_realizations and info[0] != realization:
            continue
        if info[7] in handled_kinds:
            continue
        fine_ordinary_array = _array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
        if info[4]:
            coarse_ordinary_array = _coarsening_weighted_mean(coarsening,
                                                              fine_ordinary_array,
                                                              fine_rv_array,
                                                              coarse_weight = coarse_rv_array)
        else:
            coarse_ordinary_array = _coarsening_sample(coarsening, fine_ordinary_array)
        _add_to_imported(collection, coarse_ordinary_array, 'coarsened from grid ' + str(other.support.uuid), info)


def _extend_imported_no_coarsening(collection, other, box, refinement, realization, copy_all_realizations,
                                   uncache_other_arrays):
    if realization is None:
        source_collection = other
    else:
        source_collection = rqp_c.selective_version_of_collection(other, realization = realization)

    for (part, info) in source_collection.dict.items():
        _extend_imported_no_coarsening_single(source_collection, part, info, collection, other, box, refinement,
                                              realization, copy_all_realizations, uncache_other_arrays)


def _extend_imported_no_coarsening_single(source_collection, part, info, collection, other, box, refinement,
                                          realization, copy_all_realizations, uncache_other_arrays):
    if not copy_all_realizations and info[0] != realization:
        return

    const_value = info[20]
    if const_value is None:
        a = _array_box(source_collection, part, box = box, uncache_other_arrays = uncache_other_arrays)
    else:
        a = None

    if refinement is not None and a is not None:  # simple resampling
        a = _extend_imported_no_coarsening_single_resampling(a, info, collection, refinement)

    collection.add_cached_array_to_imported_list(
        a,
        'copied from grid ' + str(other.support.uuid),
        info[10],  # citation_title
        discrete = not info[4],
        indexable_element = 'cells',
        uom = info[15],
        time_index = info[12],
        null_value = None,  # todo: extract from other's xml
        property_kind = info[7],
        local_property_kind_uuid = info[17],
        facet_type = info[8],
        facet = info[9],
        realization = info[0],
        const_value = const_value,
        points = info[21],
        time_series_uuid = info[11])


def _extend_imported_no_coarsening_single_resampling(a, info, collection, refinement):
    if info[6] != 'cells':
        # todo: appropriate refinement of data for other indexable elements
        return
    # todo: dividing up of values when needed, eg. volumes, areas, lengths
    assert tuple(a.shape) == tuple(refinement.coarse_extent_kji)
    assert collection.support is not None and tuple(collection.support.extent_kji) == tuple(refinement.fine_extent_kji)
    k_ratio_vector = refinement.coarse_for_fine_axial_vector(0)
    a_refined_k = np.empty(
        (refinement.fine_extent_kji[0], refinement.coarse_extent_kji[1], refinement.coarse_extent_kji[2]),
        dtype = a.dtype)
    a_refined_k[:, :, :] = a[k_ratio_vector, :, :]
    j_ratio_vector = refinement.coarse_for_fine_axial_vector(1)
    a_refined_kj = np.empty(
        (refinement.fine_extent_kji[0], refinement.fine_extent_kji[1], refinement.coarse_extent_kji[2]),
        dtype = a.dtype)
    a_refined_kj[:, :, :] = a_refined_k[:, j_ratio_vector, :]
    i_ratio_vector = refinement.coarse_for_fine_axial_vector(2)
    a = np.empty(tuple(refinement.fine_extent_kji), dtype = a.dtype)
    a[:, :, :] = a_refined_kj[:, :, i_ratio_vector]
    # for cell length properties, scale down the values in accordance with refinement
    if info[4] and info[7] == 'cell length' and info[8] == 'direction' and info[5] == 1:
        a = _extend_imported_no_coarsening_single_resampling_length(a, info, refinement)
    return a


def _extend_imported_no_coarsening_single_resampling_length(a, info, refinement):
    dir_ch = info[9].upper()
    log.debug(f'refining cell lengths for axis {dir_ch}')
    if dir_ch == 'K':
        a *= refinement.proportions_for_axis(0).reshape((-1, 1, 1))
    elif dir_ch == 'J':
        a *= refinement.proportions_for_axis(1).reshape((1, -1, 1))
    elif dir_ch == 'I':
        a *= refinement.proportions_for_axis(2).reshape((1, 1, -1))
    return a
