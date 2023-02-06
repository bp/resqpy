"""model.py: Main resqml interface module handling epc packing & unpacking and xml structures."""

import logging

log = logging.getLogger(__name__)

import h5py
import os
import warnings
from typing import Iterable, Optional, Union

import resqpy.model._catalogue as m_c
import resqpy.model._forestry as m_f
import resqpy.model._grids as m_g
import resqpy.model._hdf5 as m_h
import resqpy.model._xml as m_x
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


class Model():
    """Class for RESQML (v2) based models.

    Examples:
          To open an existing dataset::

             Model(epc_file = 'filename.epc')

          To create a new, empty model ready to populate::

             Model(epc_file = 'new_file.epc', new_epc = True, create_basics = True, create_hdf5_ext = True)

          Alternatively, use the module level convenience function::

             new_model(epc_file = 'new_file.epc')

          To copy an existing dataset then open the new copy::

             Model(epc_file = 'new_file.epc', copy_from = 'existing.epc')
    """

    def __init__(self,
                 epc_file: Optional[str] = None,
                 full_load: bool = True,
                 epc_subdir: Optional[Union[str, Iterable]] = None,
                 new_epc: bool = False,
                 create_basics: Optional[bool] = None,
                 create_hdf5_ext: Optional[bool] = None,
                 copy_from: Optional[str] = None,
                 quiet = False):
        """Create an empty model; load it from epc_file if given.

        Note:

           if epc_file is given and the other arguments indicate that it will be a new dataset (new_epc is True
           or copy_from is given) then any existing .epc and .h5 file(s) with this name will be deleted
           immediately

        Arguments:
           epc_file (str, optional): if present, and new_epc is False and copy_from is None, the name
              of an existing epc file which is opened, unzipped and parsed to determine the list of parts
              and relationships comprising the model; if present, and new_epc is True or copy_from is
              specified, the name of a new file to be created - any existing file (and .h5 paired hdf5
              file) will be immediately deleted
              if None, an empty model is created (ie. with no parts) unless copy_from is present
           full_load (boolean): only relevant if epc_file is not None and new_epc is False (or copy_from
              is specified); if True (recommended), the xml for each part is parsed and stored in a tree
              structure in memory; if False, only the list of parts is loaded
           epc_subdir (string or list of strings, optional): if present, parts are only included in the load
              if they are in the top level directory of the epc internal structure, or in the specified
              subdirectory (or one of the subdirectories in the case of a list); only relevant if epc_file
              is not None and new_epc is False (or copy_from is specified)
           new_epc (boolean, default False): if True, a new model is created, empty unless copy_from is given
           create_basics (boolean, optional): if True and epc_file is None or new_epc is True,
              then the minimum essential parts are added to the empty Model; this is equivalent to
              calling the create_root(), create_rels_part() and create_doc_props() methods; if None, defaults
              to the same value as new_epc
           create_hdf5_ref (boolean, optional): if True and new_epc is True and create_basics is True
              and epc_file is not None, then an hdf5 external part is created, equivalent to calling the
              create_hdf5_ext() method; an empty hdf5 file is also created; if None, defaults to same
              value as new_epc
           copy_from: (str, optional): if present, and epc_file is also present, then the epc file
              named in copy_from, together with its paired h5 file, are copied to epc_file (overwriting
              any previous instances) before epc_file is opened; this argument is primarily to facilitate
              repeated testing of code that modifies the resqml dataset, eg. by appending new parts
           quiet (boolean, default False): if True, reading and saving info logging messages are suppressed

        Returns:
           The newly created Model object

        :meta common:
        """

        info_fn = log.debug if quiet else log.info

        if epc_file and not epc_file.endswith('.epc'):
            epc_file += '.epc'
        if copy_from and not copy_from.endswith('.epc'):
            copy_from += '.epc'
        if copy_from == epc_file:
            copy_from = None
        if create_basics is None:
            create_basics = new_epc and not copy_from
        if create_hdf5_ext is None:
            create_hdf5_ext = new_epc and not copy_from
        self.initialize()
        if epc_file and (copy_from or not new_epc):
            self.load_epc(epc_file,
                          full_load = full_load,
                          epc_subdir = epc_subdir,
                          copy_from = copy_from,
                          quiet = quiet)
        else:
            if epc_file and new_epc:
                try:
                    h5_file = epc_file[:-4] + '.h5'
                    os.remove(h5_file)
                    info_fn('old hdf5 file deleted: ' + str(h5_file))
                except Exception:
                    pass
                try:
                    os.remove(epc_file)
                    info_fn('old epc file deleted: ' + str(epc_file))
                except Exception:
                    pass
            if epc_file:
                self.set_epc_file_and_directory(epc_file)
            if create_basics:
                self.create_root()
                self.create_rels_part()
                self.create_doc_props()
                if epc_file and create_hdf5_ext:
                    assert epc_file.endswith('.epc')
                    h5_file = epc_file[:-4] + '.h5'
                    self.create_hdf5_ext(add_as_part = True, file_name = h5_file)
                    with h5py.File(h5_file, 'w') as _:
                        pass

    def initialize(self):
        """Set model contents to empty.

        note:
           not usually called directly (semi-private)
        """

        self.epc_file = None
        self.epc_directory = None
        # hdf5 stuff
        self.h5_dict = {}  # dictionary keyed on hdf5 uuid.bytes; mapping to hdf5 file name (full path)
        self.h5_currently_open_path = None
        self.h5_currently_open_root = None  # h5 file handle for open hdf5 file
        self.h5_currently_open_mode = None
        self.default_h5_override = 'full'  # one of 'none', 'dir', or 'full'
        self.main_h5_uuid = None  # uuid of main hdf5 file
        # xml stuff
        self.main_tree = None
        self.main_root = None
        self.crs_uuid = None  # primary coordinate reference system for model
        self.grid_root = None  # extracted from tree as speed optimization (useful for single grid models), for 'main' grid
        self.time_series = None  # extracted as speed optimization (single time series only for now)
        self.parts_forest = {}  # dictionary keyed on part_name; mapping to (content_type, uuid, xml_tree)
        self.uuid_part_dict = {}  # dictionary keyed on uuid.int; mapping to part_name
        self.uuid_rels_dict = {
        }  # dictionary keyed on uuid.int; mapping to (uuid.int that are depended on, uuid.int that depend on, uuid.int soft relationships)
        self.rels_present = False
        self.rels_forest = {}  # dictionary keyed on part_name; mapping to (uuid, xml_tree)
        self.other_forest = {}  # dictionary keyed on part_name; mapping to (content_type, xml_tree); used for docProps
        # grid(s): single grid models only for now
        self.grid_list = []  # list of grid.Grid objects
        self.main_grid = None  # grid.Grid object for the 'main' grid
        self.reservoir_dict = [
        ]  # todo: mapping from reservoir name (citation title) to list of grids for that reservoir
        self.consolidation = None  # Consolidation object for mapping equivalent uuids
        self.modified = False

    def parts(self,
              parts_list = None,
              obj_type = None,
              uuid = None,
              title = None,
              title_mode = 'is',
              title_case_sensitive = False,
              metadata = {},
              extra = {},
              related_uuid = None,
              related_mode = None,
              epc_subdir = None,
              sort_by = None):
        """Returns a list of parts matching all of the arguments passed.

        Arguments:
           parts_list (list of strings, optional): if present, an 'input' list of parts to be filtered;
              if None then all the parts in the model are considered
           obj_type (string, optional): if present, only parts of this resqml type will be included
           uuid (uuid.UUID, optional): if present, the uuid of a part to select
           title (string, optional): if present, a citation title or substring to filter on, based on
              the title_mode argument
           title_mode (string, default 'is'): one of 'is', 'starts', 'ends', 'contains',
              'is not', 'does not start', 'does not end', or 'does not contain'; how to compare each
              part's citation title with the title argument; ignored if title is None
           title_case_sensitive (boolean, default False): if True, title comparisons are made on a
              case sensitive basis; otherwise comparisons are insensitive to case
           metadata (dictionary of key:value pairs, optional): if present, only parts which have within
              their top level metadata all the items in this argument, are included in the filtered list
           extra (dictionary of key:value pairs, optional): if present, only parts which have within
              their extra metadata all the items in this argument, are included in the filtered list
           related_uuid (uuid.UUID, optional): if present, only parts which are related to this uuid
              are included in the filtered list
           related_mode (Optional[int]): if provided, filters by the type of relationship. 0 is parts
              referenced by related_uuid, 1 is parts that reference related_uuid, 2 is other soft related parts.
           epc_subdir (string, optional): if present, only parts which reside within the specified
              subdirectory path of the epc are included in the filtered list
           sort_by (string, optional): one of 'newest', 'oldest', 'title', 'uuid', 'type'

        Returns:
           a list of strings being the names of parts which match all filter arguments

        Examples:
           a full list of parts in the model::

              model.parts()

           a list of IjkGrid parts::

              model.parts(obj_type = 'IjkGridRepresentation')

           a list containing the part name for a uuid::

              model.parts(uuid = 'a869e7cc-5d30-4b31-8502-c74b1d87c777')

           a list of IjkGrid parts with titles beginning LGR, sorted by title::

              model.parts(obj_type = 'IjkGridRepresentation', title = 'LGR', title_mode = 'starts', sort_by = 'title')

        :meta common:
        """

        return m_c._parts(self,
                          parts_list = parts_list,
                          obj_type = obj_type,
                          uuid = uuid,
                          title = title,
                          title_mode = title_mode,
                          title_case_sensitive = title_case_sensitive,
                          metadata = metadata,
                          extra = extra,
                          related_uuid = related_uuid,
                          related_mode = related_mode,
                          epc_subdir = epc_subdir,
                          sort_by = sort_by)

    def uuid_is_present(self, uuid):
        """Returns True if the uuid is present in the model's catalogue, False otherwise."""

        return m_f._uuid_is_present(self, uuid)

    def part(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             metadata = {},
             extra = {},
             related_uuid = None,
             related_mode = None,
             epc_subdir = None,
             multiple_handling = 'exception'):
        """Returns the name of a part matching all of the arguments passed.

        arguments:
           (as for parts() except no sort_by argument)
           multiple_handling (string, default 'exception'): one of 'exception', 'none', 'first', 'oldest', 'newest'

        returns:
           string being the part name of the single part matching all of the criteria, or None

        notes:
           this method can be used where a single part is being identified; if no parts match the criteria, None is
           returned; if more than one part matches the criteria, the multiple_handling argument determines what happens:
           'exception' causes a ValueError exception to be raised; 'none' causes None to be returned; 'first' causes the
           first part (as stored in the epc file or added) to be returned; 'oldest' causes the part with the oldest
           creation timestamp in the citation block to be returned; 'newest' causes the newest part to be returned

        :meta common:
        """

        return m_c._part(self,
                         parts_list = parts_list,
                         obj_type = obj_type,
                         uuid = uuid,
                         title = title,
                         title_mode = title_mode,
                         title_case_sensitive = title_case_sensitive,
                         metadata = metadata,
                         extra = extra,
                         related_uuid = related_uuid,
                         related_mode = related_mode,
                         epc_subdir = epc_subdir,
                         multiple_handling = multiple_handling)

    def uuids(self,
              parts_list = None,
              obj_type = None,
              uuid = None,
              title = None,
              title_mode = 'is',
              title_case_sensitive = False,
              metadata = {},
              extra = {},
              related_uuid = None,
              related_mode = None,
              epc_subdir = None,
              sort_by = None):
        """Returns a list of uuids of parts matching all of the arguments passed.

        arguments:
           (as for parts() method)

        returns:
           list of uuids

        :meta common:
        """

        return m_c._uuids(self,
                          parts_list = parts_list,
                          obj_type = obj_type,
                          uuid = uuid,
                          title = title,
                          title_mode = title_mode,
                          title_case_sensitive = title_case_sensitive,
                          metadata = metadata,
                          extra = extra,
                          related_uuid = related_uuid,
                          related_mode = related_mode,
                          epc_subdir = epc_subdir,
                          sort_by = sort_by)

    def uuid(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             metadata = {},
             extra = {},
             related_uuid = None,
             related_mode = None,
             epc_subdir = None,
             multiple_handling = 'exception'):
        """Returns the uuid of a part matching all of the arguments passed.

        arguments:
           (as for part())

        returns:
           uuid of the single part matching all of the criteria, or None

        :meta common:
        """

        return m_c._uuid(self,
                         parts_list = parts_list,
                         obj_type = obj_type,
                         uuid = uuid,
                         title = title,
                         title_mode = title_mode,
                         title_case_sensitive = title_case_sensitive,
                         metadata = metadata,
                         extra = extra,
                         related_uuid = related_uuid,
                         related_mode = related_mode,
                         epc_subdir = epc_subdir,
                         multiple_handling = multiple_handling)

    def roots(self,
              parts_list = None,
              obj_type = None,
              uuid = None,
              title = None,
              title_mode = 'is',
              title_case_sensitive = False,
              metadata = {},
              extra = {},
              related_uuid = None,
              related_mode = None,
              epc_subdir = None,
              sort_by = None):
        """Returns a list of xml root nodes of parts matching all of the arguments passed.

        arguments:
           (as for parts() method)

        returns:
           list of lxml.etree.Element objects

        :meta common:
        """

        return m_c._roots(self,
                          parts_list = parts_list,
                          obj_type = obj_type,
                          uuid = uuid,
                          title = title,
                          title_mode = title_mode,
                          title_case_sensitive = title_case_sensitive,
                          metadata = metadata,
                          extra = extra,
                          related_uuid = related_uuid,
                          related_mode = related_mode,
                          epc_subdir = epc_subdir,
                          sort_by = sort_by)

    def root(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             metadata = {},
             extra = {},
             related_uuid = None,
             related_mode = None,
             epc_subdir = None,
             multiple_handling = 'exception'):
        """Returns the xml root node of a part matching all of the arguments passed.

        arguments:
           (as for part())

        returns:
           lxml.etree.Element object being the root node of the xml for the single part matching all of the criteria, or None

        :meta common:
        """

        return m_c._root(self,
                         parts_list = parts_list,
                         obj_type = obj_type,
                         uuid = uuid,
                         title = title,
                         title_mode = title_mode,
                         title_case_sensitive = title_case_sensitive,
                         metadata = metadata,
                         extra = extra,
                         related_uuid = related_uuid,
                         related_mode = related_mode,
                         epc_subdir = epc_subdir,
                         multiple_handling = multiple_handling)

    def titles(self,
               parts_list = None,
               obj_type = None,
               uuid = None,
               title = None,
               title_mode = 'is',
               title_case_sensitive = False,
               metadata = {},
               extra = {},
               related_uuid = None,
               related_mode = None,
               epc_subdir = None,
               sort_by = None):
        """Returns a list of citation titles of parts matching all of the arguments passed.

        arguments:
           (as for parts() method)

        returns:
           list of strings being the citation titles of matching parts

        :meta common:
        """

        return m_c._titles(self,
                           parts_list = parts_list,
                           obj_type = obj_type,
                           uuid = uuid,
                           title = title,
                           title_mode = title_mode,
                           title_case_sensitive = title_case_sensitive,
                           metadata = metadata,
                           extra = extra,
                           related_uuid = related_uuid,
                           related_mode = related_mode,
                           epc_subdir = epc_subdir,
                           sort_by = sort_by)

    def title(self,
              parts_list = None,
              obj_type = None,
              uuid = None,
              title = None,
              title_mode = 'is',
              title_case_sensitive = False,
              metadata = {},
              extra = {},
              related_uuid = None,
              related_mode = None,
              epc_subdir = None,
              multiple_handling = 'exception'):
        """Returns the citation title of a part matching all of the arguments passed.

        arguments:
           (as for part())

        returns:
           string being the citation title of the single part matching all of the criteria, or None

        :meta common:
        """

        return m_c._title(self,
                          parts_list = parts_list,
                          obj_type = obj_type,
                          uuid = uuid,
                          title = title,
                          title_mode = title_mode,
                          title_case_sensitive = title_case_sensitive,
                          metadata = metadata,
                          extra = extra,
                          related_uuid = related_uuid,
                          related_mode = related_mode,
                          epc_subdir = epc_subdir,
                          multiple_handling = multiple_handling)

    def set_modified(self):
        """Marks the model as having been modified and assigns a new uuid.

        note:
           this modification tracking functionality is not part of the resqml standard and is only loosely
           applied by the library code; not usually called directly
        """

        self.modified = True

    def uuids_as_int_related_to_uuid(self, uuid):
        """Returns set of ints being uuids of objects related to uuid by any category of relationship.

        note:
            this method returns a set of ints; use olio.uuid.uuid_from_int() to get a UUID object
        """

        return m_c._uuids_as_int_related_to_uuid(self, uuid)

    def uuids_as_int_referenced_by_uuid(self, uuid):
        """Returns set of ints being uuids of objects which uuid has a reference to.

        note:
            this method returns a set of ints; use olio.uuid.uuid_from_int() to get a UUID object
        """

        return m_c._uuids_as_int_referenced_by_uuid(self, uuid)

    def uuids_as_int_referencing_uuid(self, uuid):
        """Returns set of ints being uuids of objects which have a reference to uuid.

        note:
            this method returns a set of ints; use olio.uuid.uuid_from_int() to get a UUID object
        """

        return m_c._uuids_as_int_referencing_uuid(self, uuid)

    def uuids_as_int_softly_related_to_uuid(self, uuid):
        """Returns set of ints being uuids of objects related to uuid by only a soft relationship.

        note:
            resqpy uses the term 'soft relationship' for those relationships held in the _rels xml area
            but not as reference nodes in the main xml of either part involved in the relationship;
            the Model.create_reciprocal_relationship() and create_reciprocal_relationship_uuid()
            methods can be used by application code to create such soft relationships;
            this method returns a set of ints; use olio.uuid.uuid_from_int() to get a UUID object
        """

        return m_c._uuids_as_int_softly_related_to_uuid(self, uuid)

    @property
    def crs_root(self):
        """XML node corresponding to self.crs_uuid, the 'main' crs for the model."""

        return self.root_for_uuid(self.crs_uuid)

    def create_tree_if_none(self):
        """Checks that model has an xml tree; if not, an empty tree is created; not usually called directly."""

        m_x._create_tree_if_none(self)

    def load_part(self, epc, part_name, is_rels = None):
        """Load and parse xml tree for given part name, storing info in parts forest (or rels forest).

        arguments:
           epc: an open ZipFile handle for the epc file
           part_name (string): the name of the 'file' within the epc bundle containing the part (or relationship)
           is_rels: (boolean, optional): if True, the part to be loaded is a relationship part; if False,
              it is a main part; if None, its value is derived from the part name

        returns:
           boolean: True if part loaded successfully, False if part failed to load

        note:
           parts forest must already have been initialized before calling this method;
           if False is returned, calling code should probably delete part from forest;
           not usually called directly
        """

        return m_f._load_part(self, epc, part_name, is_rels = is_rels)

    def set_epc_file_and_directory(self, epc_file):
        """Sets the full path and directory of the epc_file.

        arguments:
           epc_file (string): the path of the epc file

        note:
           not usually needed to be called directly, except perhaps when creating a new dataset
        """

        self.epc_file = epc_file  # full path, if provided as such
        (self.epc_directory, _) = os.path.split(epc_file)
        if self.epc_directory is None or len(self.epc_directory) == 0:
            self.epc_directory = '.'

    def fell_part(self, part_name):
        """Removes the named part from the in-memory parts forest.

        arguments:
           part_name (string): the name of the part to be removed

        note:
           no check is made for references or relationships to the part being deleted;
           not usually called directly
        """

        m_f._fell_part(self, part_name)

    def remove_part_from_main_tree(self, part):
        """Removes the named part from the main (Content_Types) tree.

        note:
           not usually called directly
        """

        m_f._remove_part_from_main_tree(self, part)

    def tidy_up_forests(self, tidy_main_tree = True, tidy_others = False, remove_extended_core = True):
        """Removes any parts that do not have any related data in dictionaries.

        note:
           not usually called directly
        """

        m_f._tidy_up_forests(self,
                             tidy_main_tree = tidy_main_tree,
                             tidy_others = tidy_others,
                             remove_extended_core = remove_extended_core)

    def load_epc(self, epc_file, full_load = True, epc_subdir = None, copy_from = None, quiet = False):
        """Load xml parts of model from epc file (HDF5 arrays are not loaded).

        Arguments:
           epc_file (string): the path of the epc file
           full_load (boolean, default True): if True (recommended), the xml for each part is parsed
              and stored in a tree structure in memory; if False, only the list of parts is loaded
           epc_subdir (string or list of strings, optional): if present, only parts in the top
              level directory within the epc structure, or in the specified subdirectory(ies) are
              included in the load
           copy_from (string, optional): if present, the .epc and .h5 are copied from this source
              to epc_file (and paired .h5) prior to opening epc_file; any previous files named
              as epc_file will be overwritten
           quiet (boolean, default False): if True, info logging message is emitted as debug

        Returns:
           None

        Note:
           when copy_from is specified, the entire contents of the source dataset are copied,
           regardless of the epc_subdir setting which only affects the subsequent load into memory
        """

        m_f._load_epc(self,
                      epc_file,
                      full_load = full_load,
                      epc_subdir = epc_subdir,
                      copy_from = copy_from,
                      quiet = quiet)

    def store_epc(self,
                  epc_file = None,
                  main_xml_name = '[Content_Types].xml',
                  only_if_modified = False,
                  quiet = False):
        """Write xml parts of model to epc file (HDF5 arrays are not written here).

        Arguments:
           epc_file (string): the name of the output epc file to be written (any existing file will be
              overwritten)
           main_xml_name (string, do not pass): this argument should not be passed as the resqml standard
              requires the default value; (the argument exists in code because the resqml standard value
              is based on a slight misunderstanding of the opc standard, so could perhaps change in
              future versions of resqml)
           only_if_modified (boolean, default False): if True, the epc file is only written if the model
              is flagged as having been modified (at least one part added or removed)
           quiet (boolean, default False): if True, info logging is emitted at debug level

        Returns:
           None

        Note:
           the main tree, parts forest and rels forest must all be up to date before calling this method

        :meta common:
        """

        m_f._store_epc(self,
                       epc_file = epc_file,
                       main_xml_name = main_xml_name,
                       only_if_modified = only_if_modified,
                       quiet = quiet)

    def parts_list_of_type(self, type_of_interest = None, uuid = None):
        """Returns a list of part names for parts of type of interest, optionally matching a uuid.

        arguments:
           type_of_interest (string): the resqml object class of interest, in string form, eg. 'obj_IjkGridRepresentation'
           uuid (uuid.UUID object, optional): if present, only a part with this uuid is included in the list

        returns:
           a list of strings being the part names which match the arguments

        note:
           usually either a type of interest or a uuid is passed; if neither are passed, all parts are returned;
           this method is maintained for backward compatibility and for efficiency reasons;
           it is equivalent to: self.parts(obj_type = type_of_interest, uuid = uuid)
        """

        return m_c._parts_list_of_type(self, type_of_interest = type_of_interest, uuid = uuid)

    def list_of_parts(self, only_objects = True):
        """Return a complete list of parts."""

        return m_c._list_of_parts(self, only_objects = only_objects)

    def number_of_parts(self):
        """Retuns the number of parts in the model, including external parts such as the link to an hdf5 file."""

        return m_c._number_of_parts(self)

    def part_for_uuid(self, uuid):
        """Returns the part name which has the given uuid.

        arguments:
           uuid (uuid.UUID object or string): the uuid of the part of interest

        returns:
           a string being the part name which matches the uuid, or None if not found

        :meta common:
        """

        return m_c._part_for_uuid(self, uuid)

    def root_for_uuid(self, uuid):
        """Returns the xml root for the part which has the given uuid.

        arguments:
           uuid (uuid.UUID object or string): the uuid of the part of interest

        returns:
           the xml root node for the part with the given uuid, or None if not found

        :meta common:
        """

        return m_c._root_for_uuid(self, uuid)

    def parts_count_by_type(self, type_of_interest = None):
        """Returns a sorted list of (type, count) for parts.

        arguments:
           type_of_interest (string, optional): if not None, the returned list only contains one pair, with
              a count for that type, ie. resqml object class

        returns:
           list of pairs, each being (string, int) representing part type, ie. resqml object class, without leading
           obj underscore, and count
        """

        return m_c._parts_count_by_type(self, type_of_interest = type_of_interest)

    def parts_list_filtered_by_related_uuid(self, parts_list, uuid, uuid_is_source = None, related_mode = None):
        """From a list of parts, returns a list of those parts which have a relationship with the given uuid.

        arguments:
           parts_list (list of strings): input list of parts from which a selection is made
           uuid (uuid.UUID): the uuid of a part for which related parts are required
           uuid_is_source (boolean, default None): if None, relationships in either direction qualify;
              if True, only those where uuid is sourceObject qualify; if False, only those where
              uuid is destinationObject qualify
           related_mode (Optional[int]): if provided, filters by the type of relationship. 0 is parts
              referenced by this uuid, 1 is parts that reference this uuid, 2 is other soft related parts.

        returns:
           list of strings being the subset of parts_list which are related to the object with the
           given uuid

        note:
           the part to which the given uuid applies might or might not be in the input parts list;
           this method scans the relationship info for every present part, looking for uuid in rels
        """

        return m_c._parts_list_filtered_by_related_uuid(self,
                                                        parts_list,
                                                        uuid,
                                                        uuid_is_source = uuid_is_source,
                                                        related_mode = related_mode)

    def supporting_representation_for_part(self, part):
        """Returns the uuid of the supporting representation for the part, if found, otherwise None."""

        return m_c._supporting_representation_for_part(self, part)

    def parts_list_filtered_by_supporting_uuid(self, parts_list, uuid):
        """From a list of parts, returns a list of those parts which have the given uuid as supporting representation.

        arguments:
           parts_list (list of strings): input list of parts from which a selection is made
           uuid (uuid.UUID): the uuid of a supporting representation part for which related parts are required

        returns:
           list of strings being the subset of parts_list which have as their supporting representation
           the object with the given uuid

        note:
           the part to which the given uuid applies might or might not be in the input parts list
        """

        return m_c._parts_list_filtered_by_supporting_uuid(self, parts_list, uuid)

    def parts_list_related_to_uuid_of_type(self, uuid, type_of_interest = None):
        """Returns a list of parts of type of interest that relate to part with given uuid.

        arguments:
           uuid (uuid.UUID): the uuid of a part for which related parts are required
           type_of_interest (string): the type of parts (resqml object class) of the related
              parts of interest

        returns:
           list of strings being the part names of the type of interest, related to the uuid
        """

        return m_c._parts_list_related_to_uuid_of_type(self, uuid, type_of_interest = type_of_interest)

    def external_parts_list(self):
        """Returns a list of part names for external part references.

        Returns:
           list of strings being the part names for external part references

        Note:

           in practice, external part references are only used for hdf5 files;
           furthermore, all current datasets have adopted the practice of using
           a single hdf5 file for a given epc file
        """

        return m_c._external_parts_list(self)

    def uuid_for_part(self, part_name, is_rels = None):
        """Returns the uuid for the named part.

        arguments:
           part_name (string): the part name for which the uuid is required
           is_rels (boolean, optional): if True, the part is a relationship part;
              if False, it is a main part; if None, its value is determined from the part name

        returns:
           uuid.UUID for the specified part

        note:
           this method will fail with an exception if the part is not in this model; a quicker alternative
           to this method is simply to extract the uuid from the part name using olio.xml_et.uuid_in_part_name()

        :meta common:
        """

        return m_c._uuid_for_part(self, part_name, is_rels = is_rels)

    def uuid_for_root(self, root_node):
        """Returns the uuid for an object given an xml root node.

        arguments:
           root_node: the xml root node for the object for which the uuid is required

        returns:
           uuid.UUID for the specified object
        """

        return rqet.uuid_for_part_root(root_node)

    def type_of_part(self, part_name, strip_obj = False):
        """Returns content type for the named part (does not apply to rels parts).

        arguments:
           part_name (string): the part for which the type is required
           strip_obj (boolean, default False): if True, the leading obj and underscore
              is removed from the returned string

        returns:
           string being the type (resqml object class) for the named part

        :meta common:
        """

        return m_c._type_of_part(self, part_name, strip_obj = strip_obj)

    def type_of_uuid(self, uuid, strip_obj = False):
        """Returns content type for the uuid.

        arguments:
           uuid (uuid.UUID or str): the uuid for which the type is required
           strip_obj (boolean, default False): if True, the leading obj and underscore
              is removed from the returned string

        returns:
           string being the type (resqml object class) for the named part

        :meta common:
        """

        return m_c._type_of_uuid(self, uuid, strip_obj = strip_obj)

    def tree_for_part(self, part_name, is_rels = None):
        """Returns parsed xml tree for the named part.

        arguments:
           part_name (string): the part name for which the xml tree is required
           is_rels (boolean, optional): if True, the part is a relationship part;
              if False, it is a main part; if None, its value is determined from the part name

        returns:
           parsed xml tree (defined in lxml or ElementTree package) for the named part
        """

        return m_c._tree_for_part(self, part_name, is_rels = is_rels)

    def root_for_part(self, part_name, is_rels = None):
        """Returns root of parsed xml tree for the named part.

        arguments:
           part_name (string): the part name for which the root of the xml tree is required
           is_rels (boolean, optional): if True, the part is a relationship part;
              if False, it is a main part; if None, its value is determined from the part name

        returns:
           root node of the parsed xml tree (defined in lxml or ElementTree package) for the named part

        :meta common:
        """

        return m_c._root_for_part(self, part_name, is_rels = is_rels)

    def change_hdf5_uuid_in_hdf5_references(self, node, old_uuid, new_uuid):
        """Scan node for hdf5 references and set the uuid of the hdf5 file itself to new_uuid.

        arguments:
           node: the root node of an xml tree within which hdf5 internal paths are to have hdf5 uuids changed
           old_uuid (uuid.UUID or str): the ext uuid currently to be found in the hdf5 references; if None, all will be replaced
           new_uuid (uuid.UUID or str): the new ext (hdf5) uuid to replace the old one

        returns:
           None

        note:
           use this method when the uuid of the hdf5 ext part is changing; if the uuid of the high level part itself is changing
           use change_uuid_in_hdf5_references() instead
        """

        m_h._change_hdf5_uuid_in_hdf5_references(self, node, old_uuid, new_uuid)

    def change_uuid_in_hdf5_references(self, node, old_uuid, new_uuid):
        """Scan node for hdf5 references using the old_uuid and replace with the new_uuid.

        arguments:
           node: the root node of an xml tree within which hdf5 internal paths are to have uuids changed
           old_uuid (uuid.UUID or str): the uuid currently to be found in the hdf5 references
           new_uuid (uuid.UUID or str): the new uuid to replace the old one

        returns:
           None

        notes:
           use this method when the uuid of the high level part itself is changing; if the uuid of the hdf5 ext part
           itself is changing, use change_hdf5_uuid_in_hdf5_references() instead; this method does not modify the
           internal path names in the hdf5 file itself, if that has already been written
        """

        m_h._change_uuid_in_hdf5_references(self, node, old_uuid, new_uuid)

    def change_uuid_in_supporting_representation_reference(self, node, old_uuid, new_uuid, new_title = None):
        """Look for supporting representation reference using the old_uuid and replace with the new_uuid.

        arguments:
           node: the root node of an xml tree within which the supporting representation uuid is to be changed
           old_uuid (uuid.UUID or str): the uuid currently to be found in the supporting representation reference
           new_uuid (uuid.UUID or str): the new uuid to replace the old one
           new_title (string, optional): if present, the title stored in the xml reference block is changed to this

        returns:
           boolean: True if the change was carried out; False otherwise

        notes:
           this method is typically used to temporarily set a supporting representation to a locally mocked
           representation object when the actual supporting representation is not present in the dataset
        """

        return m_x._change_uuid_in_supporting_representation_reference(self,
                                                                       node,
                                                                       old_uuid,
                                                                       new_uuid,
                                                                       new_title = new_title)

    def change_filename_in_hdf5_rels(self, new_hdf5_filename = None):
        """Scan relationships forest for hdf5 external parts and patch in a new filename.

        arguments:
           new_hdf5_filename: the new filename to patch into the xml; if None, the epc filename is used with
              no directory path and with the extension changed to .h5

        returns:
           None

        notes:
           no check is made that the new filename is for an existing file;
           all hdf5 file references will be modified
        """

        m_h._change_filename_in_hdf5_rels(self, new_hdf5_filename = new_hdf5_filename)

    def copy_part(self, existing_uuid, new_uuid, change_hdf5_refs = False):
        """Makes a new part as a copy of an existing part with only a new uuid set; the new part can then be modified.

        arguments:
           existing_uuid (uuid.UUID): the uuid of the existing part
           new_uuid (uuid.UUID): the uuid to inject into the new part after copying of the xml tree
           change_hdf5_refs (boolean): if True, the new tree is scanned for hdf5 refs using the existing_uuid and
              they are replaced with the new_uuid

        returns:
           string being the new part name

        notes:
           Resqml objects have a unique identifier and should be considered immutable; therefore to modify an object,
           it should first be duplicated; this function does some of the xml work needed for such duplication: the
           xml tree is copied; the uuid attribute in the root node is changed; the new part is added to the parts
           forest (with its name matched to the new uuid);
           NB: relationships are not currently copied or modified;
           also note that hdf5 data and high level objects maintained by other modules are not duplicated here;
           use this method to duplicate a part within a model prior to modifying the duplicated part in some way;
           to import a part from another model, use copy_part_from_other_model() instead; for copying a grid
           it is best to use the higher level derived_model.copy_grid() function
        """

        return m_f._copy_part(self, existing_uuid, new_uuid, change_hdf5_refs = change_hdf5_refs)

    def root_for_ijk_grid(self, uuid = None, title = None):
        """Return root for IJK Grid part.

        arguments:
           uuid (uuid.UUID, optional): if present, the uuid of the ijk grid part for which the root is required;
              if None, a single ijk grid part is expected and the root for that part is returned
           title (string, optional): if present, the citation title for the grid; defaults to 'ROOT' if more
              than one ijk grid present and no uuid supplied

        returns:
           root node in xml tree for the ijk grid part in this model

        notes:
           if uuid and title are both supplied, they must match in the corresponding grid part;
           if a title but no uuid is given, the first ijk grid encountered that has a matching title will be returned;
           if neither title nor uuid are given, the first ijk grid with title 'ROOT' will be returned, unless there is
           only one grid part in which case the root npde for that part is returned regardless;
           failure to find a matching grid part results in an assertion exception
        """

        return m_g._root_for_ijk_grid(self, uuid = uuid, title = title)

    def citation_title_for_part(self, part):  # duplicate functionality to title_for_part()
        """Returns the citation title for the specified part.

        :meta common:
        """

        return m_c._citation_title_for_part(self, part)

    def root_for_time_series(self, uuid = None):
        """Return root for time series part.

        argument:
           uuid (uuid.UUID, optional): if present, the uuid of the time series part for which the root is required;
              if None, a single time series part is expected and the root for that part is returned

        returns:
           root node in xml tree for the time series part in this model

        note:
           if no uuid is given and the model contains more than one time series, the one with the earliest creation
           date is returned
        """

        return m_c._root_for_time_series(self, uuid = uuid)

    def resolve_grid_root(self, grid_root = None, uuid = None):
        """If grid root argument is None, returns the root for the IJK Grid part instead.

        arguments:
           grid_root (optional): if not None, this method simply returns this argument
           uuid (uuid.UUID, optional): if present, the uuid of the ijk grid part for which the root is required;
              if None, an ijk grid part is sought and the root for that part is returned

        returns:
           root node in xml tree for the ijk grid part in this model ('ROOT' grid if more than one present)

        notes:
           if grid_root and uuid are both None and there are multiple grids in the model, the oldest
           grid with a citation title of 'ROOT' will be returned; an exception will be raised if no
           grid part is present in the model
        """

        return m_g._resolve_grid_root(self, grid_root = grid_root, uuid = uuid)

    def grid(self, title = None, uuid = None, find_properties = True):
        """Returns a shared Grid (or RegularGrid) object for this model, by default the 'main' grid.

        arguments:
           title (string, optional): if present, the citation title of the IjkGridRepresentation
           uuid (uuid.UUID, optional): if present, the uuid of the IjkGridRepresentation
           find_properties (boolean, default True): if True, the property_collection attribute
              of the returned grid object will be populated

        returns:
           grid.Grid object for the specified grid or the main ijk grid part in this model

        note:
           if neither title nor uuid are given, the model should contain just one grid, or a grid named 'ROOT';
           unlike most classes of object, a central list of resqpy Grid objects can be maintained within
           a Model by using this method which will return a shared object from this list, instantiating a new
           object and adding it to the list when necessary; an assertion exception will be raised if a
           suitable grid part is not present in the model

        :meta common:
        """

        return m_g._grid(self, title = title, uuid = uuid, find_properties = find_properties)

    def add_grid(self, grid_object, check_for_duplicates = False):
        """Add grid object to list of shareable grids for this model.

        arguments:
           grid_object (grid.Grid object): the ijk grid object to be added to the list
              of grids in the model
           check_for_duplicates (boolean, default False): if True, a check is made for any
              grid objects already in the grid list with the same root as the new grid object

        returns:
           None
        """

        return m_g._add_grid(self, grid_object = grid_object, check_for_duplicates = check_for_duplicates)

    def grid_list_uuid_list(self):
        """Returns list of uuid's for the grid objects in the cached grid list."""

        return m_g._grid_list_uuid_list(self)

    def grid_for_uuid_from_grid_list(self, uuid):
        """Returns the cached grid object matching the given uuid, if found in the grid list, otherwise None."""

        return m_g._grid_for_uuid_from_grid_list(self, uuid)

    def resolve_time_series_root(self, time_series_root = None):
        """If time_series_root is None, finds the root for a time series in the model.

        arguments:
           time_series_root (optional): if not None, this method simply returns this argument

        returns:
           root node in xml tree for the time series part in this model, or None if there is
           no time series part

        note:
           an assertion exception will be raised if time_series_root is None and there
           is more than one time series part in the model
        """

        return m_c._resolve_time_series_root(self, time_series_root = time_series_root)

    def h5_set_default_override(self, override):
        """Sets the default hdf5 filename override mode for the model.

        arguments:
           override (str): 'none', 'dir' or 'full' being the override mode to use by default

        note:
           this mode will be used by default when determining a filename for accessing hdf5 data;
           see h5_file_name() notes for more information
        """

        m_h._h5_set_default_override(self, override)

    def h5_uuid_and_path_for_node(self, node, tag = 'Values'):
        """Returns a (hdf5_uuid, hdf5_internal_path) pair for an xml array node.

        arguments:
           node: xml node for which the array reference is required
           tag (string, default 'Values'): the tag of the child of node for which the
              array reference is required

        returns:
           (uuid.UUID, string) pair being the uuid of the hdf5 external part reference and
           the hdf5 internal path for the array of interest

        note:
           this method provides the key data needed to actually access array data within
           the resqml dataset
        """

        return m_h._h5_uuid_and_path_for_node(self, node, tag = tag)

    def h5_uuid_list(self, node):
        """Returns a list of all uuids for hdf5 external part(s) referred to in recursive tree."""

        return m_h._h5_uuid_list(self, node)

    def h5_uuid(self):
        """Returns the uuid of the 'main' hdf5 file."""

        return m_h._h5_uuid(self)

    def h5_file_name(self, uuid = None, override = 'default', file_must_exist = True):
        """Returns full path for hdf5 file with given uuid.

        arguments:
           uuid (uuid.UUID, optional): the uuid of the hdf5 external part reference for which the
              file name is required; if None, the 'main' hdf5 uuid is used
           override (str or bool, default 'default'): if str, one of 'default', 'none', 'dir' or 'full';
              if bool (deprecated), False means 'dir' and True means 'full';
              if 'default', the default h5 override mode for the model is used;
              if 'dir', any directory in the rels hdf5 file name is replaced with the epc's directory;
              if 'full', the hdf5 full path is generated by using the epc path but replacing the .epc
              extension with .h5
           file_must_exist (boolean, default True): if True, the existence of the hdf5 file is checked
           upon first call and a FileNotFound exception is raised if the file is not found

        returns:
           string being the full path of the hdf5 file

        notes:
           in practice, a resqml model usually consists of a pair of files in the same directory,
           with names like: a.epc and a.h5;
           to allow copying, moving and renaming of files, the practical approach is simply
           to assume a one-to-one correspondence between epc and hdf5 files, and assume they
           are in the same directory, which will be the default behaviour of resqpy;
           to change the default behaviour for the model, call the h5_set_default_override() method;
           an hdf5 file name is cached once determined for a given ext uuid; to clear the cache,
           call the h5_clear_filename_cache() method
        """

        return m_h._h5_file_name(self, uuid = uuid, override = override, file_must_exist = file_must_exist)

    def h5_access(self, uuid = None, mode = 'r', override = 'default', file_path = None):
        """Returns an open h5 file handle for the hdf5 file with the given uuid.

        arguments:
           uuid (uuid.UUID): the uuid of the hdf5 external part reference for which the
              open file handle is required; required if override is False and file_path is None
           mode (string): the hdf5 file mode ('r', 'w' or 'a') with which to open the file
           override (str or bool, default 'default'): if str, one of 'default', 'none', 'dir' or 'full';
              if bool (deprecated), False means 'dir' and True means 'full';
              if 'default', the default h5 override mode for the model is used;
              if 'dir', any directory in the rels hdf5 file name is replaced with the epc's directory;
              if 'full', the hdf5 full path is generated by using the epc path but replacing the .epc
              extension with .h5
           file_path (string, optional): if present, is used as the hdf5 file path, otherwise
              the path will be determined based on the uuid and override arguments

        returns:
           a file handle to the opened hdf5 file

        note:
           an exception will be raised if the hdf5 file cannot be opened; note that sometimes another
           piece of code accessing the file might cause a 'resource unavailable' exception
        """

        return m_h._h5_access(self, uuid = uuid, mode = mode, override = override, file_path = file_path)

    def h5_release(self):
        """Releases (closes) the currently open hdf5 file.

        returns:
           None

        :meta common:
        """

        m_h._h5_release(self)

    def h5_array_shape_and_type(self, h5_key_pair):
        """Returns the shape and dtype of the array, as stored in the hdf5 file.

        arguments:
           h5_key_pair (uuid.UUID, string): the uuid of the hdf5 external part reference and the hdf5 internal path for the array

        returns:
           (tuple of ints, type): simply the shape and dtype attributes of the referenced hdf5 array; (None, None) is returned
           if the hdf5 file is not found, or the array is not found within it
        """

        return m_h._h5_array_shape_and_type(self, h5_key_pair)

    def h5_array_element(self,
                         h5_key_pair,
                         index = None,
                         cache_array = False,
                         object = None,
                         array_attribute = None,
                         dtype = 'float',
                         required_shape = None):
        """Returns one element from an hdf5 array and/or caches the array.

        arguments:
           h5_key_pair (uuid.UUID, string): the uuid of the hdf5 external part reference and the hdf5 internal path for the array
           index (pair or triple int, optional): if None, the only purpose of the call is to ensure that the array is cached in
              memory; if not None, the (k0, pillar_index) or (k0, j0, i0) index of the cell for which the value is required
           cache_array (boolean, default False): if True, a copy of the whole array is cached in memory as an attribute of the
              object; if already cached, the array is not uncached, regardless of this argument
           object (optional, defaults to self): the object in which a cached version of the array is an attribute, or will be
              created as an attribute if cache_array is True
           array_attribute (string): the attribute name to use for the cached version of the array,
              required to cache or access cached array
           dtype (string or data type): the data type of the elements of the array (need not match hdf5 array in precision)
           required_shape (tuple of ints, optional): if not None, the hdf5 array will be reshaped to this shape; if index
              is not None, it is taken to be applicable to the required shape

        returns:
           if index is None, then None;
           if index is not None, then the value of the array for the cell identified by index

        note:
           this function can be used to access an individual element from an hdf5 array, or to cache a whole array in memory;
           when accessing an individual element, the index style must match the array indexing; in particular for IJK grid points,
           a (k0, pillar_index) is needed when the grid has split pillars, whereas a (k0, j0, i0) is needed when the grid does
           not have any split pillars
        """

        return m_h._h5_array_element(self,
                                     h5_key_pair,
                                     index = index,
                                     cache_array = cache_array,
                                     object = object,
                                     array_attribute = array_attribute,
                                     dtype = dtype,
                                     required_shape = required_shape)

    def h5_array_slice(self, h5_key_pair, slice_tuple):
        """Loads a slice of an hdf5 array.

        arguments:
           h5_key_pair (uuid, string): the uuid of the hdf5 ext part and the hdf5 internal path to the
              required hdf5 array
           slice_tuple (tuple of slice objects): each element should be constructed using the python built-in
              function slice()

        returns:
           numpy array that is a hyper-slice of the hdf5 array, with the same ndim as the source hdf5 array

        notes:
           this method always fetches from the hdf5 file and does not attempt local caching; the whole array
           is not loaded; all axes continue to exist in the returned array, even where the sliced extent of
           an axis is 1
        """

        return m_h._h5_array_slice(self, h5_key_pair, slice_tuple)

    def h5_overwrite_array_slice(self, h5_key_pair, slice_tuple, array_slice):
        """Overwrites (updates) a slice of an hdf5 array.

        arguments:
           h5_key_pair (uuid, string): the uuid of the hdf5 ext part and the hdf5 internal path to the
              required hdf5 array
           slice_tuple (tuple of slice objects): each element should be constructed using the python built-in
              function slice()
           array_slice (numpy array of shape to match slice_tuple): the data to write

        notes:
           this method naively updates a slice in an hdf5 array without using mpi to look after parallel updates;
           metadata (such as uuid or property min, max values) is not modified in any way by the method
        """

        return m_h._h5_overwrite_array_slice(self, h5_key_pair, slice_tuple, array_slice)

    def h5_clear_filename_cache(self):
        """Clears the cached filenames associated with all ext uuids."""

        m_h._h5_clear_filename_cache(self)

    def create_root(self):
        """Initialises an empty main xml tree for model.

        note:
           not usually called directly
        """

        m_x._create_root(self)

    def add_part(self, content_type, uuid, root, add_relationship_part = True, epc_subdir = None):
        """Adds a (recently created) node as a new part in the model's parts forest.

        arguments:
           content_type (string): the resqml object class of the new part
           uuid (uuid.UUID): the uuid for the new part
           root: the root node of the xml tree for the new part
           add_relationship_part: (boolean, default True): if True, a relationship part is also
              created to go with the new part (empty of actual relationships)
           epc_subdir (string, optional): if present, the subdirectory path within the epc that the
              part is to be located within; if None, the xml will reside at the top level of the epc

        returns:
           None

        notes:
           NB: xml tree for part is not written to epc file by this function (store_epc() handles that);
           do not use this function for the main rels extension part: use create_rels_part() instead
        """

        m_f._add_part(self,
                      content_type,
                      uuid,
                      root,
                      add_relationship_part = add_relationship_part,
                      epc_subdir = epc_subdir)

    def patch_root_for_part(self, part, root):
        """Updates the xml tree for the part without changing the uuid."""

        m_f._patch_root_for_part(self, part, root)

    def remove_part(self, part_name, remove_relationship_part = True):
        """Removes a part from the parts forest; optionally remove corresponding rels part and other relationships."""

        m_f._remove_part(self, part_name, remove_relationship_part)

    def new_obj_node(self, flavour, name_space = 'resqml2', is_top_lvl_obj = True):
        """Creates a new main object element and sets attributes (does not add children).

        arguments:
           flavour (string): the resqml object class (type of part) for which a new xml tree
              is required
           name_space (string, default 'resqml2'): the xml namespace identifier to use for the node
           is_top_lvl_obj (boolean, default True): if True, the xsi:type is set in the xml node,
              as required for top level objects (parts); if False, the type atttribute is not set

        returns:
           newly created root node for xml tree for flavour of object, without any children
        """

        return m_x._new_obj_node(flavour, name_space = name_space, is_top_lvl_obj = is_top_lvl_obj)

    def referenced_node(self, ref_node, consolidate = False):
        """For a given xml reference node, returns the node for the object referred to, if present.

        note:
            if consolidating and an equivalent referenced object exists, the uuid in the ref_node
            is modified by this method; it does not update entries in the uuid_rels_dict
        """

        # note: the RESQML standard allows referenced objects to be missing from the package (model)

        return m_x._referenced_node(self, ref_node, consolidate = consolidate)

    def create_ref_node(self, flavour, title, uuid, content_type = None, root = None):
        """Create a reference node, optionally add to root.

        arguments:
           flavour (string): the resqml object class (type of part) for which a
              reference node is required
           title (string): used as the Title subelement text in the reference node
           uuid: (uuid.UUID): the uuid of the part being referenced
           content_type (string, optional): if None, the referenced content type is
              determined from the flavour argument (recommended)
           root (optional): if not None, an xml node to which the reference node is
              appended as a child

        returns:
           newly created reference xml node
        """

        return m_x._create_ref_node(self, flavour, title, uuid, content_type = content_type, root = root)

    def uom_node(self, root, uom):
        """Add a generic unit of measure sub element to root.

        arguments:
           root: xml node to which unit of measure subelement (child) will be added
           uom (string): the resqml unit of measure

        returns:
           newly created unit of measure node (having already been added to root)

        note:
           does not currently check that uom is a valid Energistics unit of measure;
           use weights_and_measures module functionality to check if needed
        """

        return m_x._uom_node(root, uom)

    def create_rels_part(self):
        """Adds a relationships reference node as a new part in the model's parts forest.

        returns:
           newly created main relationships reference xml node

        note:
           there can only be one relationships reference part in the model
        """

        return m_x._create_rels_part(self)

    def create_citation(self, root = None, title = '', originator = None):
        """Creates a citation xml node and optionally appends as a child of root.

        arguments:
           root (optional): if not None, the newly created citation node is appended as
              a child to this node
           title (string): the citation title: a human readable string; this is the main point
              of having a citation node, so the argument should be used wisely
           originator (string, optional): the name of the human being who created the object
              which this citation is for; default is to use the login name

        returns:
           newly created citation xml node
        """

        return m_x._create_citation(root = root, title = title, originator = originator)

    def title_for_root(self, root = None):
        """Returns the Title text from the Citation within the given root node.

        arguments:
           root: the xml node for the object for which the citation title is required

        returns:
           string being the Title text from the citation node which is a child of root,
           or None if not found

        :meta common:
        """

        return m_c._title_for_root(self, root = root)

    def title_for_part(self, part_name):  # duplicate functionality to citation_title_for_part()
        """Returns the Title text from the Citation for the given main part name (not for rels).

        arguments:
           part_name (string): the name of the part for which the citation title is required

        returns:
           string being the Title text from the citation node which is a child of the root xml node
           for the part, or None if not found

        :meta common:
        """

        return m_c._title_for_part(self, part_name)

    def create_unknown(self, root = None):
        """Creates an Unknown node and optionally adds as child of root.

        arguments:
           root (optional): if present, the newly created Unknown node is appended as a child
              of this xml node

        returns:
           the newly created Unknown xml node

        :meta common:
        """

        return m_x._create_unknown(root = root)

    def create_doc_props(self, add_as_part = True, root = None, originator = None):
        """Creates a document properties stub node and optionally adds as child of root and/or to parts forest.

        arguments:
           add_as_part (boolean, default True): if True, the newly created node is also added as a part
           root (optional, usually None): if not None, the newly created node is appended to this root
              as a child
           originator (string, optional): used as the creator in the doc props node; if None, the
              login name is used

        returns:
           the newly created doc props xml node

        note:
           the doc props part of a resqml dataset is intended to hold documentation and
           other stuff that is not covered by the standard; there should be exactly one doc props part
        """

        return m_x._create_doc_props(self, add_as_part = add_as_part, root = root, originator = originator)

    def create_crs_reference(self, root = None, crs_uuid = None):
        """Creates a node refering to an existing crs node and optionally adds as child of root.

        arguments:
           root: the xml node to which the new reference node is to appended as a child (ie. the xml node
              for the object that is referring to the crs)
           crs_uuid: the uuid of the crs

        returns:
           newly created crs reference xml node
        """

        return m_x._create_crs_reference(self, root = root, crs_uuid = crs_uuid)

    def create_md_datum_reference(self, md_datum_root, root = None):
        """Creates a node refering to an existing measured depth datum and optionally adds as child of root.

        arguments:
           md_datum_root: the root xml node for the measured depth datum being referenced
           root: the xml node to which the new reference node is to appended as a child (ie. the xml node
              for the object that is referring to the md datum)

        returns:
           newly created measured depth datum reference xml node
        """

        return m_x._create_md_datum_reference(self, md_datum_root, root = root)

    def create_hdf5_ext(self,
                        add_as_part = True,
                        root = None,
                        title = 'Hdf Proxy',
                        originator = None,
                        file_name = None,
                        uuid = None):
        """Creates an hdf5 external node and optionally adds as child of root and/or to parts forest.

        arguments:
           add_as_part (boolean, default True): if True the newly created ext node is added to the model
              as a part
           root (optional, usually None): if not None, the newly created ext node is appended as a child
              of this node
           title (string): used as the Title text in the citation node, usually left at the default 'Hdf Proxy'
           originator (string, optional): the name of the human being who created the ext object;
              default is to use the login name
           file_name (string, optional): the filename to be stored as the Target in the relationship node;
              if None, will default to the epc filename with the extenstion replaced with .h5
           uuid (uuid.UUID, optional): the ext uuid tp associate with the external part; if None, a new
              uuid will be generated

        returns:
           newly created hdf5 external part xml node

        note:
           this method is typically called when creating a new dataset (Model); if the intention is to
           share an existing hdf5 file, then pass the file_name and (ext) uuid arguments; if the intention
           is to create a new hdf5 file amongst many used by the Model, then pass the file_name
        """

        return m_x._create_hdf5_ext(self,
                                    add_as_part = add_as_part,
                                    root = root,
                                    title = title,
                                    originator = originator,
                                    file_name = file_name,
                                    uuid = uuid)

    def create_hdf5_dataset_ref(self, hdf5_uuid, object_uuid, group_tail, root, title = 'Hdf Proxy'):
        """Creates a pair of nodes referencing an hdf5 dataset (array) and adds to root.

        arguments:
           hdf5_uuid (uuid.UUID): the uuid of the hdf5 external part being referenced
           object_uuid (uuid.UUID): the uuid of the high level object (part) which owns the hdf5 array
              being referenced
           group_tail (string): the tail of the hdf5 internal path, which is appended to the part
              name section of the internal path
           root: the xml node to which the newly created hdf5 reference is appended as a child
           title (string): used as the Title text in the citation node, usually left at the default 'Hdf Proxy'

        returns:
           the newly created xml node holding the hdf5 internal path
        """

        return m_x._create_hdf5_dataset_ref(self, hdf5_uuid, object_uuid, group_tail, root, title = title)

    def create_supporting_representation(self,
                                         grid_root = None,
                                         support_uuid = None,
                                         root = None,
                                         title = None,
                                         content_type = 'obj_IjkGridRepresentation'):
        """Craate a supporting representation reference node refering to an IjkGrid and optionally add to root.

        arguments:
           grid_root: the xml node of the grid (or other) object which is the supporting representation
              being referred to; could also be used for other classes of supporting object; this or
              support_uuid must be provided
           support_uuid: the uuid of the grid (or other supporting representation) being referred to;
              this or grid_root must be provided
           root: if not None, the newly created supporting representation node is appended as a child
              to this node
           title: the Title to use in the supporting representation node
           content_type: the resqml object class of the supporting representation being referenced;
              defaults to 'obj_IjkGridRepresentation'

        returns:
           newly created xml node for supporting representation reference

        notes:
           a property array needs a supporting representation which is the structure that the property
           values belong to; for example, a grid property array has the grid object as the supporting
           representation; one of grid_root or support_uuid should be passed when calling this method
        """

        return m_x._create_supporting_representation(self,
                                                     support_root = grid_root,
                                                     support_uuid = support_uuid,
                                                     root = root,
                                                     title = title,
                                                     content_type = content_type)

    def create_source(self, source, root = None):
        """Create an extra meta data node holding information on the source of the data, optionally add to root.

        arguments:
           source (string): text describing the source of an object
           root: if not None, the newly created extra metadata node is appended as a child of this node

        returns:
           the newly created extra metadata xml node
        """

        return m_x._create_source(source, root = root)

    def create_patch(self,
                     p_uuid,
                     ext_uuid = None,
                     root = None,
                     patch_index = 0,
                     hdf5_type = 'DoubleHdf5Array',
                     xsd_type = 'double',
                     null_value = None,
                     const_value = None,
                     const_count = None,
                     points = False):
        """Create a node for a patch of values, including ref to hdf5 data set, optionally add to root.

        arguments:
           p_uuid (uuid.UUID): the uuid of the object for which this patch is a component
           ext_uuid (uuid.UUID): the uuid of the hdf5 external part holding the array; required unless
              const_value and const_count are not None
           root: if not None, the newly created patch of values xml node is appended as a child
              to this node
           patch_index (int, default 0): the patch index number; patches must be numbered
              sequentially starting at 0
           hdf5_type (string, default 'DoubleHdf5Array'): the type of the hdf5 array; usually one of
              'DoubleHdf5Array', 'IntegerHdf5Array', or 'BooleanHdf5Array'; replaced with equivalent
              constant array type if const_value is not None
           xsd_type (string, default 'double'): the xsd simple type of each element of the array
           null_value: Used in a null value sub-node to specify what value in an array of discrete data represents null;
              if None, a value of -1 is used for signed integers, 2^32 - 1 for uints (even 64 bit uints!)
           const_value (float, int, or bool, optional): if not None, the patch is created as a constant array;
              const_count must also be present if const_value is not None
           const_count (int, optional): the number of elements in (size of) the constant array; required if
              const_value is not None, ignored otherwise
           points (bool, default False): if True, the created node will be for a patch of points,
              otherwise a patch of values

        returns:
           newly created xml node for the patch of values

        note:
           this function does not write the data to the hdf5; that should be done separately before
           calling this method;
           RESQML usually stores array data in the hdf5 file however constant arrays are flagged as
           such in the xml and no data is stored in the hdf5
        """

        return m_x._create_patch(self,
                                 p_uuid,
                                 ext_uuid = ext_uuid,
                                 root = root,
                                 patch_index = patch_index,
                                 hdf5_type = hdf5_type,
                                 xsd_type = xsd_type,
                                 null_value = null_value,
                                 const_value = const_value,
                                 const_count = const_count,
                                 points = points)

    def create_time_series_ref(self, time_series_uuid, root = None):
        """Create a reference node to a time series, optionally add to root.

        arguments:
           time_series_uuid (uuid.UUID): the uuid of the time series part being referenced
           root (optional): if present, the newly created time series reference xml node is added
              as a child to this node

        returns:
           the newly created time series reference xml node
        """

        return m_x._create_ref_node(self, 'TimeSeries', 'time series', time_series_uuid, root = root)

    def create_solitary_point3d(self, flavour, root, xyz):
        """Creates a subelement to root for a solitary point in 3D space.

        arguments:
           flavour (string): the object class (type) of the point node to be created
           root: the xml node to which the newly created solitary point node is appended
              as a child
           xyz (triple float): the x, y, z coordinates of the solitary point

        returns:
           the newly created xml node for the solitary point
        """

        return m_x._create_solitary_point3d(flavour, root, xyz)

    def create_reciprocal_relationship(self, node_a, rel_type_a, node_b, rel_type_b, avoid_duplicates = True):
        """Adds a node to each of a pair of trees in the rels forest, to represent a two-way relationship.

        arguments:
           node_a: one of the two xml nodes to be related
           rel_type_a (string): the Type (role) associated with node_a in the relationship;
              usually 'sourceObject' or 'destinationObject'
           node_b: the other xml node to be related
           rel_type_b (string): the Type (role) associated with node_b in the relationship
              usually 'sourceObject' or 'destinationObject' (opposite of rel_type_a)
           avoid_duplicates (boolean, default True): if True, xml for a relationship is not added
              where it already exists; if False, a duplicate will be created in this situation

        returns:
           None

        note:
           this method has the same effect as create_reciprocal_relationship_uuids() but takes xml root nodes
           rather than uuids as arguments
        """

        return m_x._create_reciprocal_relationship(self,
                                                   node_a,
                                                   rel_type_a,
                                                   node_b,
                                                   rel_type_b,
                                                   avoid_duplicates = avoid_duplicates)

    def create_reciprocal_relationship_uuids(self, uuid_a, rel_type_a, uuid_b, rel_type_b, avoid_duplicates = True):
        """Adds a node to each of a pair of trees in the rels forest, to represent a two-way relationship.

        arguments:
           uuid_a: uuid of one of the two parts to be related
           rel_type_a (string): the Type (role) associated with uuid_a in the relationship;
              usually 'sourceObject' or 'destinationObject'
           uuid_b: uuid of the other part to be related
           rel_type_b (string): the Type (role) associated with uuid_b in the relationship
              usually 'sourceObject' or 'destinationObject' (opposite of rel_type_a)
           avoid_duplicates (boolean, default True): if True, xml for a relationship is not added
              where it already exists; if False, a duplicate will be created in this situation

        returns:
           None

        note:
           this method has the same effect as create_reciprocal_relationship() but takes uuids rather than
           xml root nodes as arguments
        """

        node_a = self.root_for_uuid(uuid_a)
        node_b = self.root_for_uuid(uuid_b)
        assert node_a is not None and node_b is not None, 'part not present when creating relationship'
        return m_x._create_reciprocal_relationship(self,
                                                   node_a,
                                                   rel_type_a,
                                                   node_b,
                                                   rel_type_b,
                                                   avoid_duplicates = avoid_duplicates)

    def duplicate_node(self, existing_node, add_as_part = True):
        """Creates a deep copy of the xml node (typically from another model) and optionally adds as part.

        arguments:
           existing_node: the existing xml node, usually in another model, to be duplicated
           add_as_part (boolean, default True): if True, the newly created xml node is added as a part
              in this model

        returns:
           the newly created duplicate xml node

        notes:
           hdf5 data is not copied by this function and any reference to hdf5 arrays will be naively
           duplicated;
           the uuid of the part is not changed by this function, so if the source and target models are
           the same, add as part should be set False and calling code will need to assign a new uuid
           prior to adding as part;
           if add_as_part is True, the part is only added if the uuid does not already exist in this
           model
        """

        return m_f._duplicate_node(self, existing_node, add_as_part = add_as_part)

    def force_consolidation_uuid_equivalence(self, immigrant_uuid, resident_uuid):
        """Force immigrant object to be teated as equivalent to resident during consolidation."""

        m_f._force_consolidation_uuid_equivalence(self, immigrant_uuid, resident_uuid)

    def force_consolidation_equivalence_for_class_ignoring_extra_metadata(self, other_model, resqpy_class):
        """Force immigrant objects of type to be teated as equivalent where only extra metadata differs during consolidation.

        notes:
            this method should be called prior to calling copy_part_from_other_model() or
            copy_uuid_from_other_model() to override the more stringent equivalence checks which
            include extra metadata; the resqpy class must have an is_equivalent() method which
            supports the check_extra_metadata boolean argument;
            typically used for organisational classes such as features and interpretations;
            an exception is raised if more than one matching part already exists in this model, for a
            particular immigrant title (for the object type)
        """

        obj_type = resqpy_class.resqml_type
        resident_uuids = self.uuids(obj_type = obj_type)
        if len(resident_uuids) == 0:
            log.debug(f'no resident parts for class {resqpy_class}')
            return
        immigrant_uuids = other_model.uuids(obj_type = obj_type)
        if len(immigrant_uuids) == 0:
            log.debug(f'no immigrant parts for class {resqpy_class} object type {obj_type}')
            return
        resident_objects_dict = {}
        for resident_uuid in resident_uuids:
            resident_objects_dict[resident_uuid] = resqpy_class(self, uuid = resident_uuid)
        for immigrant_uuid in immigrant_uuids:
            immigrant_obj = resqpy_class(other_model, uuid = immigrant_uuid)
            log.debug(f'looking for equivalent for {obj_type} {immigrant_obj.title}')
            for resident_uuid in resident_uuids:
                if resident_objects_dict[resident_uuid].is_equivalent(immigrant_obj, check_extra_metadata = False):
                    log.debug(f'forcing equivalence between immigrant {immigrant_uuid} and resident {resident_uuid} ' +
                              f'of class {resqpy_class} object type {obj_type}')
                    m_f._force_consolidation_uuid_equivalence(self, immigrant_uuid, resident_uuid)
                    break

    def remove_extra_metadata(self, uuid):
        """Removes extra metadata from in memory xml for uuid.

        note:
            this method will not modify any resqpy objects already instantiated
        """

        rqet.cut_extra_metadata(self.root_for_uuid(uuid))

    def copy_part_from_other_model(self,
                                   other_model,
                                   part,
                                   realization = None,
                                   consolidate = True,
                                   force = False,
                                   cut_refs_to_uuids = None,
                                   cut_node_types = None,
                                   self_h5_file_name = None,
                                   h5_uuid = None,
                                   other_h5_file_name = None,
                                   uuid_int = None):
        """Fully copies part in from another model, with referenced parts, hdf5 data and relationships.

        arguments:
           other model (Model): the source model from which to copy a part
           part (string): the part name in the other model to copy into this model
           realization (int, optional): if present and the part is a property, the realization
              will be set to this value, instead of the value in use in the other model if any
           consolidate (boolean, default True): if True and an equivalent part already exists in
              this model, do not duplicate but instead note uuids as equivalent
           force (boolean, default False): if True, the part itself is copied without much checking
              and all references are required to be handled by an entry in the consolidation object
           cut_refs_to_uuids (list of UUIDs, optional): if present, then xml reference nodes
              referencing any of the listed uuids are cut out in the copy; use with caution
           cut_node_types (list of str, optional): if present, any child nodes of a type in the list
              will be cut out in the copy; use with caution
           self_h5_file_name (string, optional): h5 file name for this model; can be passed as
              an optimisation when calling method repeatedly
           h5_uuid (uuid, optional): UUID for this model's hdf5 external part; can be passed as
              an optimisation when calling method repeatedly
           other_h5_file_name (string, optional): h5 file name for other model; can be passed as
              an optimisation when calling method repeatedly
           uuid_int (int, optional): if present, the uuid (as int) of part; if uuid already established
              use this argument as an optimisation; note: no checks for consistency are made here

        returns:
           the part name of the part in this model, after copying; may differ from requested part if
           consolidate is True; None in the case of failure

        notes:
           if the part name already exists in this model, no action is taken;
           default hdf5 file used in this model and assumed in other_model
        """

        assert other_model is not None
        if other_model is self:
            return part
        assert part is not None
        if realization is not None:
            assert isinstance(realization, int) and realization >= 0
        if force:
            assert consolidate
        if not other_h5_file_name:
            other_h5_file_name = other_model.h5_file_name()
        if not self_h5_file_name:
            self_h5_file_name = self.h5_file_name(file_must_exist = False)
        hdf5_copy_needed = not os.path.samefile(self_h5_file_name, other_h5_file_name)

        # check whether already existing in this model
        if part in self.parts_forest.keys():
            return part

        if m_c._type_of_part(other_model, part) == 'obj_EpcExternalPartReference':
            return None

        return m_f._copy_part_from_other_model(self,
                                               other_model,
                                               part,
                                               realization = realization,
                                               consolidate = consolidate,
                                               force = force,
                                               cut_refs_to_uuids = cut_refs_to_uuids,
                                               cut_node_types = cut_node_types,
                                               self_h5_file_name = self_h5_file_name,
                                               h5_uuid = h5_uuid,
                                               other_h5_file_name = other_h5_file_name,
                                               hdf5_copy_needed = hdf5_copy_needed,
                                               uuid_int = uuid_int)

    def copy_uuid_from_other_model(self,
                                   other_model,
                                   uuid,
                                   realization = None,
                                   consolidate = True,
                                   force = False,
                                   cut_refs_to_uuids = None,
                                   cut_node_types = None,
                                   self_h5_file_name = None,
                                   h5_uuid = None,
                                   other_h5_file_name = None):
        """Fully copies part for uuid in from another model, with referenced parts, hdf5 data and relationships.

        arguments:
           other model (Model): the source model from which to copy a part
           uuid (UUID): the uuid of the part in the other model to copy into this model
           realization (int, optional): if present and the part is a property, the realization
              will be set to this value, instead of the value in use in the other model if any
           consolidate (boolean, default True): if True and an equivalent part already exists in
              this model, do not duplicate but instead note uuids as equivalent
           force (boolean, default False): if True, the part itself is copied without much checking
              and all references are required to be handled by an entry in the consolidation object
           cut_refs_to_uuids (list of UUIDs, optional): if present, then xml reference nodes
              referencing any of the listed uuids are cut out in the copy; use with caution
           cut_node_types (list of str, optional): if present, any child nodes of a type in the list
              will be cut out in the copy; use with caution
           self_h5_file_name (string, optional): h5 file name for this model; can be passed as
              an optimisation when calling method repeatedly
           h5_uuid (uuid, optional): UUID for this model's hdf5 external part; can be passed as
              an optimisation when calling method repeatedly
           other_h5_file_name (string, optional): h5 file name for other model; can be passed as
              an optimisation when calling method repeatedly

        returns:
           the uuid of the part in this model, after copying; may differ from requested uuid if
           consolidate is True; None in the case of failure

        notes:
           if the part already exists in this model, no action is taken;
           default hdf5 file used in this model and assumed in other_model
        """

        part = other_model.part_for_uuid(uuid)
        if part is None:
            return None
        copied_part = self.copy_part_from_other_model(other_model,
                                                      part,
                                                      realization = realization,
                                                      consolidate = consolidate,
                                                      force = force,
                                                      cut_refs_to_uuids = cut_refs_to_uuids,
                                                      cut_node_types = cut_node_types,
                                                      self_h5_file_name = self_h5_file_name,
                                                      h5_uuid = h5_uuid,
                                                      other_h5_file_name = other_h5_file_name,
                                                      uuid_int = bu.uuid_as_int(uuid))
        if copied_part is None:
            return None
        return self.uuid_for_part(copied_part)

    def copy_all_parts_from_other_model(self, other_model, realization = None, consolidate = True):
        """Fully copies parts in from another model, with referenced parts, hdf5 data and relationships.

        arguments:
           other model (Model): the source model from which to copy parts
           realization (int, optional): if present, the realization attribute of property parts
              will be set to this value, instead of the value in use in the other model if any
           consolidate (boolean, default True): if True, where equivalent part already exists in
              this model, do not duplicate but instead note uuids as equivalent, modifying
              references and relationships of other copied parts appropriately

        notes:
           part names already existing in this model are not duplicated;
           default hdf5 file used in this model and assumed in other_model
        """

        m_f._copy_all_parts_from_other_model(self, other_model, realization = realization, consolidate = consolidate)

    def iter_objs(self, cls):
        """Iterate over all available objects of given resqpy class within the model.

        note:

           The resqpy class must expose a class attribute `resqml_type`, and must support
           being created with the signature: `obj = cls(model, uuid=uuid)`.

        example use::

           for well in model.iter_objs(cls=resqpy.well.WellboreFeature):
              print(well.title, well.uuid)

        arguments:
           cls: resqpy class to iterate

        yields:
           list of instances of cls

        :meta common:
        """

        for obj in m_c._iter_objs(self, cls):
            yield obj

    def iter_grid_connection_sets(self):
        """Yields grid connection set objects, one for each gcs in this model."""

        for gcs in list(m_c._iter_grid_connection_sets(self)):
            yield gcs

    def iter_wellbore_interpretations(self):
        """Iterable of all WellboreInterpretations associated with the model.

        yields:
           wellbore: instance of :class:`resqpy.organize.WellboreInterpretation`

        :meta common:
        """

        for wi in m_c._iter_wellbore_interpretations(self):
            yield wi

    def iter_trajectories(self):
        """Iterable of all trajectories associated with the model.

        yields:
           trajectory: instance of :class:`resqpy.well.Trajectory`

        :meta common:
        """

        for wt in m_c._iter_trajectories(self):
            yield wt

    def iter_md_datums(self):
        """Iterable of all MdDatum objects associated with the model.

        yields:
           md_datum: instance of :class:`resqpy.well.MdDatum`

        :meta common:
        """

        for mdd in m_c._iter_md_datums(self):
            yield mdd

    def iter_crs(self):
        """Iterable of all CRS objects associated with the model.

        yields:
           crs: instance of :class:`resqpy.crs.CRS`

        :meta common:
        """

        for crs in m_c._iter_crs(self):
            yield crs

    def sort_parts_list_by_timestamp(self, parts_list):
        """Returns a copy of the parts list sorted by citation block creation date, with the newest first."""

        return m_c._sort_parts_list_by_timestamp(self, parts_list)

    def check_catalogue_dictionaries(self, referred_parts_must_be_present = True, check_xml = True):
        """Checks internal consistency of catalogue dictionaries, raising assertion exception if inconsistent.

        arguments:
            referred_parts_must_be_present (bool, default True): if True, raises an exception if a referenced
                part is not present in the model (such a scenario is allowed by the RESQML standard)
            check_xml (bool, default True): if True, xml is scoured to check that references are consistent
                with the Model uuid_rels_dict internal dictionary

        note:
            this is a thorough but slow check, use sparing; primarily intended for unit tests and debugging
        """

        m_c._check_catalogue_dictionaries(self, referred_parts_must_be_present, check_xml)

    def as_graph(self, uuids_subset = None):
        """Return representation of model as nodes and edges, suitable for plotting in a graph.

        note:
           The graph can be most readily visualised with other packages such as
           NetworkX and HoloViews, which are not part of resqpy.

           For a guide to plotting graphs interactively, see:
           http://holoviews.org/user_guide/Network_Graphs.html

        example::

           # Create the nodes and edges
           nodes, edges = model.as_graph()

           # Load into a NetworkX graph
           import networkx as nx
           g = nx.Graph()
           g.add_nodes_from(nodes.items())
           g.add_edges_from(edges)

           # Import holoviews
           import holoviews as hv
           from holoviews import opts
           hv.extension('bokeh')

           # Plot
           hv.Graph.from_networkx(g, nx.layout.spring_layout).opts(
              tools=['hover'], node_color='resqml_type', cmap='Category10'
           )

        arguments:
           uuids_subset (iterable): If present, only consider uuids in this list.
              By default, use all uuids in the model.

        returns:
           2-tuple of nodes and edges:

           - nodes: dict mapping uuid to attributes (e.g. citation title)
           - edges: set of unordered pairs of uuids, representing relationships

        :meta common:
        """

        return m_c._as_graph(self, uuids_subset = uuids_subset)


def new_model(epc_file, quiet = False) -> Model:
    """Returns a new, empty Model object with basics and hdf5 ext part set up."""

    return Model(epc_file = epc_file, new_epc = True, create_basics = True, create_hdf5_ext = True, quiet = quiet)
