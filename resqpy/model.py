"""model.py: Main resqml interface module handling epc packing & unpacking and xml structures."""

version = '27th August 2021'

import logging

log = logging.getLogger(__name__)
log.debug('model.py version ' + version)

import os
import copy
import getpass
import pathlib
import shutil
import warnings
import zipfile as zf
from typing import Union, Optional, Iterable

import numpy as np
import h5py

import resqpy.olio.xml_et as rqet
import resqpy.olio.time as time
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as whdf5
import resqpy.olio.consolidation as cons
from resqpy.olio.xml_namespaces import curly_namespace as ns, namespace as ns_url

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.fault as rqf

use_version_string = False


def _pl(i, e = False):
   return '' if i == 1 else 'es' if e else 's'


class Model():
   """Class for RESQML (v2) based models.
   
   Examples:

         To open an existing dataset::

            Model(epc_file = 'filename.epc')

         To create a new, empty model ready to populate::

            Model(epc_file = 'new_file.epc', new_epc = True, create_basics = True, create_hdf5_ext = True)

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
                copy_from: Optional[str] = None):
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

      Returns:
         The newly created Model object

      :meta common:
      """

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
         self.load_epc(epc_file, full_load = full_load, epc_subdir = epc_subdir, copy_from = copy_from)
      else:
         if epc_file and new_epc:
            try:
               h5_file = epc_file[:-4] + '.h5'
               os.remove(h5_file)
               log.info('old hdf5 file deleted: ' + str(h5_file))
            except Exception:
               pass
            try:
               os.remove(epc_file)
               log.info('old epc file deleted: ' + str(epc_file))
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
      self.main_h5_uuid = None  # uuid of main hdf5 file
      # xml stuff
      self.main_tree = None
      self.main_root = None
      self.crs_uuid = None  # primary coordinate reference system for model
      self.grid_root = None  # extracted from tree as speed optimization (useful for single grid models), for 'main' grid
      self.time_series = None  # extracted as speed optimization (single time series only for now)
      self.parts_forest = {}  # dictionary keyed on part_name; mapping to (content_type, uuid, xml_tree)
      self.uuid_part_dict = {}  # dictionary keyed on uuid.int; mapping to part_name
      self.rels_present = False
      self.rels_forest = {}  # dictionary keyed on part_name; mapping to (uuid, xml_tree)
      self.other_forest = {}  # dictionary keyed on part_name; mapping to (content_type, xml_tree); used for docProps
      # grid(s): single grid models only for now
      self.grid_list = []  # list of grid.Grid objects
      self.main_grid = None  # grid.Grid object for the 'main' grid
      self.reservoir_dict = []  # todo: mapping from reservoir name (citation title) to list of grids for that reservoir
      self.consolidation = None  # Consolidation object for mapping equivalent uuids
      self.modified = False

   def parts(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             extra = {},
             related_uuid = None,
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
         extra (dictionary of key:value pairs, optional): if present, only parts which have within
            their extra metadata all the items in this argument, are included in the filtered list
         related_uuid (uuid.UUID, optional): if present, only parts which are related to this uuid
            are included in the filtered list
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
            
            model.parts(obj_type='IjkGridRepresentation', title='LGR', title_mode='starts', sort_by='title')
      
      :meta common:
      """

      if not parts_list:
         parts_list = self.list_of_parts()
      if uuid is not None:
         part_name = self.uuid_part_dict.get(bu.uuid_as_int(uuid))
         if part_name is None or part_name not in parts_list:
            return []
         parts_list = [part_name]
      if epc_subdir:
         if epc_subdir.startswith('/'):
            epc_subdir = epc_subdir[1:]
         if epc_subdir:
            if not epc_subdir.endswith('/'):
               epc_subdir += '/'
            filtered_list = []
            for part in parts_list:
               if part.startswith[epc_subdir]:
                  filtered_list.append(part)
            if len(filtered_list) == 0:
               return []
            parts_list = filtered_list
      if obj_type:
         if obj_type[0].isupper():
            obj_type = 'obj_' + obj_type
         filtered_list = []
         for part in parts_list:
            if self.parts_forest[part][0] == obj_type:
               filtered_list.append(part)
         if len(filtered_list) == 0:
            return []
         parts_list = filtered_list
      if title:
         assert title_mode in [
            'is', 'starts', 'ends', 'contains', 'is not', 'does not start', 'does not end', 'does not contain'
         ]
         if not title_case_sensitive:
            title = title.upper()
         filtered_list = []
         for part in parts_list:
            part_title = self.citation_title_for_part(part)
            if not title_case_sensitive:
               part_title = part_title.upper()
            if title_mode == 'is':
               if part_title == title:
                  filtered_list.append(part)
            elif title_mode == 'starts':
               if part_title.startswith(title):
                  filtered_list.append(part)
            elif title_mode == 'ends':
               if part_title.endswith(title):
                  filtered_list.append(part)
            elif title_mode == 'contains':
               if title in part_title:
                  filtered_list.append(part)
            if title_mode == 'is not':
               if part_title != title:
                  filtered_list.append(part)
            elif title_mode == 'does not start':
               if not part_title.startswith(title):
                  filtered_list.append(part)
            elif title_mode == 'does not end':
               if not part_title.endswith(title):
                  filtered_list.append(part)
            elif title_mode == 'does not contain':
               if title not in part_title:
                  filtered_list.append(part)
         if len(filtered_list) == 0:
            return []
         parts_list = filtered_list
      if extra:
         filtered_list = []
         for part in parts_list:
            part_extra = rqet.load_metadata_from_xml(self.root_for_part(part))
            if not part_extra:
               continue
            match = True
            for key, value in extra.items():
               if key not in part_extra or part_extra[key] != value:
                  match = False
                  break
            if match:
               filtered_list.append(part)
         if len(filtered_list) == 0:
            return []
         parts_list = filtered_list
      if related_uuid is not None:
         parts_list = self.parts_list_filtered_by_related_uuid(parts_list, related_uuid)
      if len(parts_list) == 0:
         return []
      if sort_by:
         if sort_by == 'type':
            parts_list.sort()
         elif sort_by in ['newest', 'oldest']:
            parts_list = self.sort_parts_list_by_timestamp(parts_list)
            if sort_by == 'oldest':
               parts_list.reverse()
         elif sort_by in ['uuid', 'title']:
            sort_list = []
            for index, part in enumerate(parts_list):
               if sort_by == 'uuid':
                  key = str(self.uuid_for_part(part))
               else:
                  key = self.citation_title_for_part(part)
               sort_list.append((key, index))
            sort_list.sort()
            sorted_list = []
            for _, index in sort_list:
               sorted_list.append(parts_list[index])
            parts_list = sorted_list
      return parts_list

   def part(self,
            parts_list = None,
            obj_type = None,
            uuid = None,
            title = None,
            title_mode = 'is',
            title_case_sensitive = False,
            extra = {},
            related_uuid = None,
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

      pl = self.parts(parts_list = parts_list,
                      obj_type = obj_type,
                      uuid = uuid,
                      title = title,
                      title_mode = title_mode,
                      title_case_sensitive = title_case_sensitive,
                      extra = extra,
                      related_uuid = related_uuid,
                      epc_subdir = epc_subdir)
      if len(pl) == 0:
         return None
      if len(pl) == 1 or multiple_handling == 'first':
         return pl[0]
      if multiple_handling == 'none':
         return None
      elif multiple_handling in ['newest', 'oldest']:
         sorted_list = self.sort_parts_list_by_timestamp(pl)
         if multiple_handling == 'newest':
            return sorted_list[0]
         return sorted_list[-1]
      else:
         raise ValueError('more than one part matches criteria')

   def uuids(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             extra = {},
             related_uuid = None,
             epc_subdir = None,
             sort_by = None):
      """Returns a list of uuids of parts matching all of the arguments passed.

      arguments:
         (as for parts() method)

      returns:
         list of uuids

      :meta common:
      """

      sort_by_uuid = (sort_by == 'uuid')
      if sort_by_uuid:
         sort_by = None
      pl = self.parts(parts_list = parts_list,
                      obj_type = obj_type,
                      uuid = uuid,
                      title = title,
                      title_mode = title_mode,
                      title_case_sensitive = title_case_sensitive,
                      extra = extra,
                      related_uuid = related_uuid,
                      epc_subdir = epc_subdir,
                      sort_by = sort_by)
      if len(pl) == 0:
         return []
      uuid_list = []
      for part in pl:
         uuid_list.append(self.uuid_for_part(part))
      if sort_by_uuid:
         uuid_list.sort()
      return uuid_list

   def uuid(self,
            parts_list = None,
            obj_type = None,
            uuid = None,
            title = None,
            title_mode = 'is',
            title_case_sensitive = False,
            extra = {},
            related_uuid = None,
            epc_subdir = None,
            multiple_handling = 'exception'):
      """Returns the uuid of a part matching all of the arguments passed.

      arguments:
         (as for part())

      returns:
         uuid of the single part matching all of the criteria, or None

      :meta common:
      """

      part = self.part(parts_list = parts_list,
                       obj_type = obj_type,
                       uuid = uuid,
                       title = title,
                       title_mode = title_mode,
                       title_case_sensitive = title_case_sensitive,
                       extra = extra,
                       related_uuid = related_uuid,
                       epc_subdir = epc_subdir,
                       multiple_handling = multiple_handling)
      if part is None:
         return None
      return rqet.uuid_in_part_name(part)

   def roots(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             extra = {},
             related_uuid = None,
             epc_subdir = None,
             sort_by = None):
      """Returns a list of xml root nodes of parts matching all of the arguments passed.

      arguments:
         (as for parts() method)

      returns:
         list of lxml.etree.Element objects

      :meta common:
      """

      pl = self.parts(parts_list = parts_list,
                      obj_type = obj_type,
                      uuid = uuid,
                      title = title,
                      title_mode = title_mode,
                      title_case_sensitive = title_case_sensitive,
                      extra = extra,
                      related_uuid = related_uuid,
                      epc_subdir = epc_subdir,
                      sort_by = sort_by)
      root_list = []
      for part in pl:
         root_list.append(self.root_for_part(part))
      return root_list

   def root(self,
            parts_list = None,
            obj_type = None,
            uuid = None,
            title = None,
            title_mode = 'is',
            title_case_sensitive = False,
            extra = {},
            related_uuid = None,
            epc_subdir = None,
            multiple_handling = 'exception'):
      """Returns the xml root node of a part matching all of the arguments passed.

      arguments:
         (as for part())

      returns:
         lxml.etree.Element object being the root node of the xml for the single part matching all of the criteria, or None

      :meta common:
      """

      part = self.part(parts_list = parts_list,
                       obj_type = obj_type,
                       uuid = uuid,
                       title = title,
                       title_mode = title_mode,
                       title_case_sensitive = title_case_sensitive,
                       extra = extra,
                       related_uuid = related_uuid,
                       epc_subdir = epc_subdir,
                       multiple_handling = multiple_handling)
      if part is None:
         return None
      return self.root_for_part(part)

   def titles(self,
              parts_list = None,
              obj_type = None,
              uuid = None,
              title = None,
              title_mode = 'is',
              title_case_sensitive = False,
              extra = {},
              related_uuid = None,
              epc_subdir = None,
              sort_by = None):
      """Returns a list of citation titles of parts matching all of the arguments passed.

      arguments:
         (as for parts() method)

      returns:
         list of strings being the citation titles of matching parts

      :meta common:
      """

      pl = self.parts(parts_list = parts_list,
                      obj_type = obj_type,
                      uuid = uuid,
                      title = title,
                      title_mode = title_mode,
                      title_case_sensitive = title_case_sensitive,
                      extra = extra,
                      related_uuid = related_uuid,
                      epc_subdir = epc_subdir,
                      sort_by = sort_by)
      title_list = []
      for part in pl:
         title_list.append(self.citation_title_for_part(part))
      return title_list

   def title(self,
             parts_list = None,
             obj_type = None,
             uuid = None,
             title = None,
             title_mode = 'is',
             title_case_sensitive = False,
             extra = {},
             related_uuid = None,
             epc_subdir = None,
             multiple_handling = 'exception'):
      """Returns the citation title of a part matching all of the arguments passed.

      arguments:
         (as for part())

      returns:
         string being the citation title of the single part matching all of the criteria, or None

      :meta common:
      """

      part = self.part(parts_list = parts_list,
                       obj_type = obj_type,
                       uuid = uuid,
                       title = title,
                       title_mode = title_mode,
                       title_case_sensitive = title_case_sensitive,
                       extra = extra,
                       related_uuid = related_uuid,
                       epc_subdir = epc_subdir,
                       multiple_handling = multiple_handling)
      if part is None:
         return None
      return self.citation_title_for_part(part)

   def set_modified(self):
      """Marks the model as having been modified and assigns a new uuid.

      note:
         this modification tracking functionality is not part of the resqml standard and is only loosely
         applied by the library code; not usually called directly
      """

      self.modified = True

   @property
   def crs_root(self):
      """XML node corresponding to self.crs_uuid"""

      return self.root_for_uuid(self.crs_uuid)

   def create_tree_if_none(self):
      """Checks that model has an xml tree; if not, an empty tree is created; not usually called directly."""

      if self.main_tree is None:
         self.main_tree = rqet.ElementTree()
         self.modified = True

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

      # note: epc is 'open' ZipFile handle
      if part_name.startswith('/'):
         part_name = part_name[1:]

      try:

         #        log.debug('loading part ' + part_name)
         if is_rels is None:
            is_rels = part_name.endswith('.rels')
         is_other = not is_rels and part_name.startswith('docProps')

         part_type = None
         part_uuid = None

         if is_rels:
            if part_name in self.rels_forest:
               (part_uuid, _) = self.rels_forest[part_name]
            if part_uuid is None:
               part_uuid = rqet.uuid_in_part_name(part_name)
         elif is_other:
            (part_type, _) = self.other_forest[part_name]
         else:
            (part_type, part_uuid, _) = self.parts_forest[part_name]  # part_type must already have been established

         with epc.open(part_name) as part_xml:
            part_tree = rqet.parse(part_xml)
            if is_rels:
               self.rels_forest[part_name] = (part_uuid, part_tree)
            elif is_other:
               self.other_forest[part_name] = (part_type, part_tree)
            else:
               uuid_from_tree = rqet.uuid_for_part_root(part_tree.getroot())
               if part_uuid is None:
                  part_uuid = uuid_from_tree
               elif uuid_from_tree is not None:
                  assert bu.matching_uuids(part_uuid, uuid_from_tree)
               self.parts_forest[part_name] = (part_type, part_uuid, part_tree)
               self._set_uuid_to_part(part_name)
               if self.crs_uuid is None and part_type == 'obj_LocalDepth3dCrs':  # randomly assign first crs as primary crs for model
                  self.crs_uuid = part_uuid

         return True

      except Exception:

         log.exception('(okay to continue?) failed to load part: ' + part_name)
         return False

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

      try:
         self._del_uuid_to_part(part_name)
      except Exception:
         pass
      try:
         del self.parts_forest[part_name]
      except Exception:
         pass
      try:
         del self.rels_forest[part_name]
      except Exception:
         pass
      try:
         del self.other_forest[part_name]
      except Exception:
         pass

   def remove_part_from_main_tree(self, part):
      """Removes the named part from the main (Content_Types) tree.

      note:
         not usually called directly
      """

      for child in self.main_root:
         if rqet.stripped_of_prefix(child.tag) == 'Override':
            part_name = child.attrib['PartName']
            if part_name[0] == '/':
               part_name = part_name[1:]
            if part_name == part:
               log.debug('removing part from main xml tree: ' + part)
               self.main_root.remove(child)
               break

   def tidy_up_forests(self, tidy_main_tree = True, tidy_others = False, remove_extended_core = True):
      """Removes any parts that do not have any related data in dictionaries.

      note:
         not usually called directly
      """

      deletion_list = []
      for part, info in self.parts_forest.items():
         if info == (None, None, None):
            deletion_list.append(part)
      for part in deletion_list:
         log.debug('removing part due to lack of xml tree etc.: ' + str(part))
         if tidy_main_tree:
            self.remove_part_from_main_tree(part)
         self._del_uuid_to_part(part)
         del self.parts_forest[part]
      deletion_list = []
      for part, info in self.rels_forest.items():
         if info == (None, None):
            deletion_list.append(part)
      for part in deletion_list:
         log.debug('removing rels part due to lack of xml tree etc.: ' + str(part))
         if tidy_main_tree:
            self.remove_part_from_main_tree(part)
         del self.rels_forest[part]
      if tidy_others:
         for part, info in self.other_forest.items():
            if info == (None, None):
               deletion_list.append(part)
         for part in deletion_list:
            log.debug('removing docProps part due to lack of xml tree etc.: ' + str(part))
            if tidy_main_tree:
               self.remove_part_from_main_tree(part)
            del self.other_forest[part]
      if remove_extended_core and 'docProps/extendedCore.xml' in self.other_forest:  # more trouble than it's worth
         part = 'docProps/extendedCore.xml'
         if tidy_main_tree:
            self.remove_part_from_main_tree(part)
         del self.other_forest[part]

   def load_epc(self, epc_file, full_load = True, epc_subdir = None, copy_from = None):
      """Load xml parts of model from epc file (HDF5 arrays are not loaded).

      Arguments:
         epc_file (string): the path of the epc file
         full_load (boolean): if True (recommended), the xml for each part is parsed and stored
            in a tree structure in memory; if False, only the list of parts is loaded
         epc_subdir (string or list of strings, optional): if present, only parts in the top
            level directory within the epc structure, or in the specified subdirectory(ies) are
            included in the load
         copy_from (string, optional): if present, the .epc and .h5 are copied from this source
            to epc_file (and paired .h5) prior to opening epc_file; any previous files named
            as epc_file will be overwritten

      Returns:
         None

      Note:
         when copy_from is specified, the entire contents of the source dataset are copied,
         regardless of the epc_subdir setting which only affects the subsequent load into memory
      """

      def exclude(name, epc_subdir):
         if epc_subdir is None:
            return False
         if '/' not in name:
            return False
         if name.startswith('docProps') or name.startswith('_rels'):
            return False
         if isinstance(epc_subdir, str):
            epc_subdir = [epc_subdir]
         for subdir in epc_subdir:
            if subdir.endswith('/'):
               head = subdir
            else:
               head = subdir + '/'
            if name.startswith(head):
               return False
         return True

      if not epc_file.endswith('.epc'):
         epc_file += '.epc'

      if copy_from:
         if not copy_from.endswith('.epc'):
            copy_from += '.epc'
         log.info('copying ' + copy_from + ' to ' + epc_file + ' along with paired .h5 files')
         shutil.copy(copy_from, epc_file)
         shutil.copy(copy_from[:-4] + '.h5', epc_file[:-4] + '.h5')

      log.info('loading resqml model from epc file ' + epc_file)

      if self.modified:
         log.warning('loading model from epc, discarding previous in-memory modifications')
         self.initialize()

      self.set_epc_file_and_directory(epc_file)

      with zf.ZipFile(epc_file) as epc:
         names = epc.namelist()
         for name in names:
            if exclude(name, epc_subdir):
               continue
            if name != '[Content_Types].xml':
               if name.startswith('docProps'):
                  self.other_forest[name] = (None, None)  # used for non-uuid parts, ie. docProps
               else:
                  part_uuid = rqet.uuid_in_part_name(name)
                  if '_rels' in name:
                     self.rels_forest[name] = (part_uuid, None)
                  else:
                     self.parts_forest[name] = (None, part_uuid, None)
                     self._set_uuid_to_part(name)
         with epc.open('[Content_Types].xml') as main_xml:
            self.main_tree = rqet.parse(main_xml)
            self.main_root = self.main_tree.getroot()
            for child in self.main_root:
               if rqet.stripped_of_prefix(child.tag) == 'Override':
                  attrib_dict = child.attrib
                  part_name = attrib_dict['PartName']
                  if part_name[0] == '/':
                     part_name = part_name[1:]
                  part_type = rqet.content_type(attrib_dict['ContentType'])
                  if part_name.startswith('docProps'):
                     if part_name not in self.other_forest:
                        log.warning('docProps entry in Content_Types does not exist as part in epc: ' + part_name)
                        continue
                     self.other_forest[part_name] = (part_type, None)
                  else:
                     if part_name not in self.parts_forest:
                        if epc_subdir is None:
                           log.warning('entry in Content_Types does not exist as part in epc: ' + part_name)
                        continue
                     part_uuid = self.parts_forest[part_name][1]
                     self.parts_forest[part_name] = (part_type, part_uuid, None)
                  if full_load:
                     load_success = self.load_part(epc, part_name)
                     if not load_success:
                        self.fell_part(part_name)
               elif rqet.stripped_of_prefix(child.tag) == 'Default':
                  if 'Extension' in child.attrib.keys() and child.attrib['Extension'] == 'rels':
                     assert not self.rels_present
                     self.rels_present = True
               else:
                  # todo: check standard for other valid tags
                  pass
         if self.rels_present and full_load:
            for name in names:
               if exclude(name, epc_subdir):
                  continue
               if name.startswith('_rels/'):
                  load_success = self.load_part(epc, name, is_rels = True)
                  if not load_success:
                     self.fell_part(part_name)
            if copy_from:
               self.change_filename_in_hdf5_rels(os.path.split(epc_file)[1][:-4] + '.h5')
         elif not self.rels_present:
            assert len(self.rels_forest) == 0
         if full_load:
            self.tidy_up_forests()

   def store_epc(self, epc_file = None, main_xml_name = '[Content_Types].xml', only_if_modified = False):
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

      Returns:
         None

      Note:
         the main tree, parts forest and rels forest must all be up to date before calling this method

      :meta common:
      """

      #      for prefix, uri in ns.items():
      #         et.register_namespace(prefix, uri)

      if not epc_file:
         epc_file = self.epc_file
      assert epc_file, 'no file name given or known when attempting to store epc'

      if only_if_modified and not self.modified:
         return

      log.info('storing resqml model to epc file ' + epc_file)

      assert self.main_tree is not None
      if self.main_root is None:
         self.main_root = self.main_tree.getroot()

      with zf.ZipFile(epc_file, mode = 'w') as epc:
         with epc.open(main_xml_name, mode = 'w') as main_xml:
            log.debug('Writing main xml: ' + main_xml_name)
            rqet.write_xml(main_xml, self.main_tree, standalone = 'yes')
         for part_name, (_, _, part_tree) in self.parts_forest.items():
            if part_tree is None:
               log.warning('No xml tree present to write for part: ' + part_name)
               continue
            if part_name[0] == '/':
               part_name = part_name[1:]
            with epc.open(part_name, mode = 'w') as part_xml:
               rqet.write_xml(part_xml, part_tree, standalone = None)
         for part_name, (_, part_tree) in self.other_forest.items():
            if part_tree is None:
               log.warning('No xml tree present to write for other part: ' + part_name)
               continue
            if part_name[0] == '/':
               part_name = part_name[1:]
            with epc.open(part_name, mode = 'w') as part_xml:
               rqet.write_xml(part_xml, part_tree, standalone = 'yes')
         if self.rels_present:
            for part_name, (_, part_tree) in self.rels_forest.items():
               if part_tree is None:
                  log.warning('No xml tree present to write for rels part: ' + part_name)
                  continue
               with epc.open(part_name, mode = 'w') as part_xml:
                  rqet.write_xml(part_xml, part_tree, standalone = 'yes')
         # todo: other parts (documentation etc.)
      self.set_epc_file_and_directory(epc_file)
      self.modified = False

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

      if type_of_interest and type_of_interest[0].isupper():
         type_of_interest = 'obj_' + type_of_interest

      if uuid is not None:
         part_name = self.uuid_part_dict.get(bu.uuid_as_int(uuid))
         if part_name is None or (type_of_interest is not None and
                                  (self.parts_forest[part_name][0] != type_of_interest)):
            return []
         return [part_name]

      parts_list = []
      for part_name in self.parts_forest:
         if type_of_interest is None or self.parts_forest[part_name][0] == type_of_interest:
            parts_list.append(part_name)
      return parts_list

   def list_of_parts(self, only_objects = True):
      """Return a complete list of parts."""

      pl = list(self.parts_forest.keys())
      if not only_objects:
         return pl
      obj_list = []
      for part in pl:
         dir_place = part.rfind('/')
         dir_free_part = part[dir_place + 1:]
         if dir_free_part.startswith('obj_') and not dir_free_part.startswith('obj_Epc'):
            obj_list.append(part)
      return obj_list

   def number_of_parts(self):
      """Retuns the number of parts in the model, including external parts such as the link to an hdf5 file."""

      return len(self.parts_forest)

   def part_for_uuid(self, uuid):
      """Returns the part name which has the given uuid.

      arguments:
         uuid (uuid.UUID object or string): the uuid of the part of interest

      returns:
         a string being the part name which matches the uuid, or None if not found

      :meta common:
      """

      return self.uuid_part_dict.get(bu.uuid_as_int(uuid))

   def root_for_uuid(self, uuid):
      """Returns the xml root for the part which has the given uuid.

      arguments:
         uuid (uuid.UUID object or string): the uuid of the part of interest

      returns:
         the xml root node for the part with the given uuid, or None if not found

      :meta common:
      """

      return self.root_for_part(self.part_for_uuid(uuid))

   def parts_count_by_type(self, type_of_interest = None):
      """Returns a sorted list of (type, count) for parts.

      arguments:
         type_of_interest (string, optional): if not None, the returned list only contains one pair, with
            a count for that type (resqml object class)

      returns:
         list of pairs, each being (string, int) representing part type (resqml object class without leading obj_)
         and count
      """

      # note: resqml classes start with 'obj_' whilst witsml classes don't!
      if type_of_interest and type_of_interest.startswith('obj_'):
         type_of_interest = type_of_interest[4:]

      type_list = []
      for part_name in self.parts_forest:
         part_type = self.parts_forest[part_name][0]
         if part_type is None:
            continue
         if part_type.startswith('obj_'):
            part_type = part_type[4:]
         if type_of_interest is None or part_type == type_of_interest:
            type_list.append(part_type)
      type_list.sort()
      type_list.append('END')  # simplifies termination of scan below
      result_list = []
      count = 0
      current_type = ''
      for index in range(len(type_list)):
         if type_list[index] != current_type:
            if count:
               result_list.append((current_type, count))
            current_type = type_list[index]
            count = 0
         count += 1
      return result_list

   def parts_list_filtered_by_related_uuid(self, parts_list, uuid, uuid_is_source = None):
      """From a list of parts, returns a list of those parts which have a relationship with the given uuid.

      arguments:
         parts_list (list of strings): input list of parts from which a selection is made
         uuid (uuid.UUID): the uuid of a part for which related parts are required
         uuid_is_source (boolean, default None): if None, relationships in either direction qualify;
            if True, only those where uuid is sourceObject qualify; if False, only those where
            uuid is destinationObject qualify

      returns:
         list of strings being the subset of parts_list which are related to the object with the
         given uuid

      note:
         the part to which the given uuid applies might or might not be in the input parts list;
         this method scans the relationship info for every present part, looking for uuid in rels
      """

      if not self.rels_present or parts_list is None or uuid is None:
         return None
      filtered_list = []
      this_part = self.part_for_uuid(uuid)

      if this_part is not None:
         rels_part_root = self.root_for_part(rqet.rels_part_name_for_part(this_part), is_rels = True)
         if rels_part_root is not None:
            for relation_node in rels_part_root:
               if rqet.stripped_of_prefix(relation_node.tag) != 'Relationship':
                  continue
               target_part = relation_node.attrib['Target']
               if target_part not in parts_list:
                  continue
               if uuid_is_source is not None:
                  source_dest = relation_node.attrib['Type']
                  if uuid_is_source:
                     if 'source' not in source_dest:
                        continue
                  else:
                     if 'source' in source_dest:
                        continue
               filtered_list.append(target_part)

      for part in parts_list:
         if part in filtered_list:
            continue
         rels_part_root = self.root_for_part(rqet.rels_part_name_for_part(part), is_rels = True)
         if rels_part_root is None:
            continue
         for relation_node in rels_part_root:
            if rqet.stripped_of_prefix(relation_node.tag) != 'Relationship':
               continue
            target_part = relation_node.attrib['Target']
            relation_uuid = rqet.uuid_in_part_name(target_part)
            if bu.matching_uuids(uuid, relation_uuid):
               if uuid_is_source is not None:
                  source_dest = relation_node.attrib['Type']
                  if uuid_is_source:
                     if 'source' in source_dest:
                        continue  # relation is source, so uuid is not
                  else:
                     if 'source' not in source_dest:
                        continue  # relation is not source, so uuid is
               filtered_list.append(part)
               break

      return filtered_list

   def supporting_representation_for_part(self, part):
      """Returns the uuid of the supporting representation for the part, if found, otherwise None."""

      return bu.uuid_from_string(
         rqet.find_nested_tags_text(self.root_for_part(part), ['SupportingRepresentation', 'UUID']))

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

      if parts_list is None or uuid is None:
         return None
      filtered_list = []
      for part in parts_list:
         support_ref_uuid = self.supporting_representation_for_part(part)
         if support_ref_uuid is None:
            continue
         if bu.matching_uuids(support_ref_uuid, uuid):
            filtered_list.append(part)
      return filtered_list

   def parts_list_related_to_uuid_of_type(self, uuid, type_of_interest = None):
      """Returns a list of parts of type of interest that relate to part with given uuid.

      arguments:
         uuid (uuid.UUID): the uuid of a part for which related parts are required
         type_of_interest (string): the type of parts (resqml object class) of the related
            parts of interest

      returns:
         list of strings being the part names of the type of interest, related to the uuid
      """

      parts_list = self.parts_list_of_type(type_of_interest = type_of_interest)
      return self.parts_list_filtered_by_related_uuid(parts_list, uuid)

   def external_parts_list(self):
      """Returns a list of part names for external part references.

      Returns:
         list of strings being the part names for external part references

      Note:

         in practice, external part references are only used for hdf5 files;
         furthermore, all current datasets have adopted the practice of using
         a single hdf5 file for a given epc file
      """

      return self.parts_list_of_type('obj_EpcExternalPartReference')

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

      if part_name is None:
         return None
      if is_rels is None:
         is_rels = part_name.endswith('.rels')
      if is_rels:
         return self.rels_forest[part_name][0]
      return self.parts_forest[part_name][1]

   def type_of_part(self, part_name, strip_obj = False):
      """Returns content type for the named part (does not apply to rels parts).

      arguments:
         part_name (string): the part for which the type is required
         strip_obj (boolean, default False): if True, the leading 'obj_' is removed
            from the returned string

      returns:
         string being the type (resqml object class) for the named part

      :meta common:
      """

      part_info = self.parts_forest.get(part_name)
      if part_info is None:
         return None
      obj_type = part_info[0]
      if obj_type is None or not strip_obj or not obj_type.startswith('obj_'):
         return obj_type
      return obj_type[4:]

   def type_of_uuid(self, uuid, strip_obj = False):
      """Returns content type for the uuid.

      arguments:
         uuid (uuid.UUID or str): the uuid for which the type is required
         strip_obj (boolean, default False): if True, the leading 'obj_' is removed
            from the returned string

      returns:
         string being the type (resqml object class) for the named part

      :meta common:
      """

      part_name = self.uuid_part_dict.get(bu.uuid_as_int(uuid))
      return self.type_of_part(part_name, strip_obj = strip_obj)

   def tree_for_part(self, part_name, is_rels = None):
      """Returns parsed xml tree for the named part.

      arguments:
         part_name (string): the part name for which the xml tree is required
         is_rels (boolean, optional): if True, the part is a relationship part;
            if False, it is a main part; if None, its value is determined from the part name

      returns:
         parsed xml tree (defined in lxml or ElementTree package) for the named part
      """

      if not part_name:
         return None
      if is_rels is None:
         is_rels = part_name.endswith('.rels')
      is_other = not is_rels and part_name.startswith('docProps')
      if is_rels:
         if part_name not in self.rels_forest:
            return None
         (_, tree) = self.rels_forest[part_name]
         if tree is None:
            if not self.epc_file:
               return None
            with zf.ZipFile(self.epc_file) as epc:
               load_success = self.load_part(epc, part_name, is_rels = True)
               if not load_success:
                  return None
         return self.rels_forest[part_name][1]
      elif is_other:
         if part_name not in self.other_forest:
            return None
         (_, tree) = self.other_forest[part_name]
         if tree is None:
            if not self.epc_file:
               return None
            with zf.ZipFile(self.epc_file) as epc:
               load_success = self.load_part(epc, part_name, is_rels = False)
               if not load_success:
                  return None
         return self.other_forest[part_name][1]
      else:
         if part_name not in self.parts_forest:
            return None
         (_, _, tree) = self.parts_forest[part_name]
         if tree is None:
            if not self.epc_file:
               return None
            with zf.ZipFile(self.epc_file) as epc:
               load_success = self.load_part(epc, part_name, is_rels = False)
               if not load_success:
                  return None
         return self.parts_forest[part_name][2]

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

      if not part_name:
         return None
      tree = self.tree_for_part(part_name, is_rels = is_rels)
      if tree is None:
         return None
      return tree.getroot()

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

      count = 0
      old_uuid_str = str(old_uuid)
      new_uuid_str = str(new_uuid)
      for ref_node in node.iter(ns['eml'] + 'HdfProxy'):
         try:
            uuid_node = rqet.find_tag(ref_node, 'UUID')
            if old_uuid is None or uuid_node.text == old_uuid_str:
               uuid_node.text = new_uuid_str
               count += 1
         except Exception:
            pass
      if count == 1:
         log.debug('one hdf5 reference modified')
      else:
         log.debug(str(count) + ' hdf5 references modified')
      if count > 0:
         self.set_modified()

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

      count = 0
      old_uuid_str = str(old_uuid)
      new_uuid_str = str(new_uuid)
      for ref_node in node.iter(ns['eml'] + 'PathInHdfFile'):
         try:
            uuid_place = ref_node.text.index(old_uuid_str)
            new_path_in_hdf = ref_node.text[:uuid_place] + new_uuid_str + ref_node.text[uuid_place + len(old_uuid_str):]
            log.debug('path in hdf update from: ' + ref_node.text + ' to: ' + new_path_in_hdf)
            ref_node.text = new_path_in_hdf
            count += 1
         except Exception:
            pass
      if count == 1:
         log.debug('one hdf5 reference modified')
      else:
         log.debug(str(count) + ' hdf5 references modified')
      if count > 0:
         self.set_modified()

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

      ref_node = rqet.find_tag(node, 'SupportingRepresentation')
      if ref_node is None:
         return False
      uuid_node = rqet.find_tag(ref_node, 'UUID')
      if uuid_node is None:
         return False
      if not bu.matching_uuids(uuid_node.text, old_uuid):
         return False
      uuid_node.text = str(new_uuid)
      if new_title:
         title_node = rqet.find_tag(ref_node, 'Title')
         if title_node is not None:
            title_node.text = str(new_title)
      self.set_modified()
      return True

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

      if not new_hdf5_filename and self.epc_file and self.epc_file.endswith('.epc'):
         new_hdf5_filename = os.path.split(self.epc_file)[1][:-4] + '.h5'
      count = 0
      for rel_name, entry in self.rels_forest.items():
         rel_root = entry[1].getroot()
         for child in rel_root:
            if child.attrib['Id'] == 'Hdf5File' and child.attrib['TargetMode'] == 'External':
               child.attrib['Target'] = new_hdf5_filename
               count += 1
      log.info(str(count) + ' hdf5 filename' + _pl(count) + ' set to: ' + new_hdf5_filename)
      if count > 0:
         self.set_modified()

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
            it is best to use the higher level derevied_model.copy_grid() function
      """

      old_uuid_str = str(existing_uuid)
      new_uuid_str = str(new_uuid)
      log.debug('copying xml part from uuid: ' + old_uuid_str + ' to uuid: ' + new_uuid_str)
      existing_parts_list = self.parts_list_of_type(uuid = existing_uuid)
      if len(existing_parts_list) == 0:
         log.warning('failed to find existing part for copying with uuid: ' + old_uuid_str)
         return None
      assert len(existing_parts_list) == 1, 'more than one existing part found with uuid: ' + old_uuid_str
      (part_type, old_uuid, old_tree) = self.parts_forest[existing_parts_list[0]]
      assert bu.matching_uuids(old_uuid, existing_uuid)
      new_tree = copy.deepcopy(old_tree)
      new_root = new_tree.getroot()
      part_name = rqet.patch_uuid_in_part_root(new_root, new_uuid)
      if change_hdf5_refs:
         self.change_uuid_in_hdf5_references(new_root, old_uuid_str, new_uuid_str)
      self.add_part(part_type, new_uuid, new_root, add_relationship_part = False)
      return part_name

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

      if title is not None:
         title = title.strip().upper()
      if uuid is None and not title:
         grid_root = self.root(obj_type = 'IjkGridRepresentation', title = 'ROOT', multiple_handling = 'oldest')
         if grid_root is None:
            grid_root = self.root(obj_type = 'IjkGridRepresentation')
      else:
         grid_root = self.root(obj_type = 'IjkGridRepresentation', uuid = uuid, title = title)

      assert grid_root is not None, 'IJK Grid part not found'

      return grid_root

   def citation_title_for_part(self, part):  # duplicate functionality to title_for_part()
      """Returns the citation title for the specified part.

      :meta common:
      """

      return rqet.citation_title_for_node(self.root_for_part(part))

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

      time_series_list = self.parts_list_of_type('obj_TimeSeries', uuid = uuid)
      if len(time_series_list) == 0:
         return None
      if len(time_series_list) == 1:
         return self.root_for_part(time_series_list[0])
      log.warning('selecting time series with earliest creation date')
      oldest_root = oldest_creation = None
      for ts in time_series_list:
         node = self.root_for_part(ts)
         created = rqet.creation_date_for_node(node)
         if oldest_creation is None or created < oldest_creation:
            oldest_creation = created
            oldest_root = node
      return oldest_root

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

      if grid_root is not None:
         if self.grid_root is None:
            self.grid_root = grid_root
      else:
         if self.grid_root is None:
            self.grid_root = self.root_for_ijk_grid(uuid = uuid)
         grid_root = self.grid_root
      return grid_root

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

      if uuid is None and (title is None or title.upper() == 'ROOT'):
         if self.main_grid is not None:
            if find_properties:
               self.main_grid.extract_property_collection()
            return self.main_grid
         if title is None:
            grid_root = self.resolve_grid_root()
         else:
            grid_root = self.resolve_grid_root(grid_root = self.root(obj_type = 'IjkGridRepresentation', title = title))
      else:
         grid_root = self.root(obj_type = 'IjkGridRepresentation', uuid = uuid, title = title)
      assert grid_root is not None, 'IJK Grid part not found'
      if uuid is None:
         uuid = rqet.uuid_for_part_root(grid_root)
      for grid in self.grid_list:
         if grid.root is grid_root:
            if find_properties:
               grid.extract_property_collection()
            return grid
      grid = grr.any_grid(self, uuid = uuid, find_properties = find_properties)
      assert grid is not None, 'failed to instantiate grid object'
      if find_properties:
         grid.extract_property_collection()
      self.add_grid(grid)
      return grid

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

      if check_for_duplicates:
         for g in self.grid_list:
            if bu.matching_uuids(g.uuid, grid_object.uuid):
               return
      self.grid_list.append(grid_object)

   def grid_list_uuid_list(self):
      """Returns list of uuid's for the grid objects in the cached grid list."""

      uuid_list = []
      for grid in self.grid_list:
         uuid_list.append(grid.uuid)
      return uuid_list

   def grid_for_uuid_from_grid_list(self, uuid):
      """Returns the cached grid object matching the given uuid, if found in the grid list, otherwise None."""

      for grid in self.grid_list:
         if bu.matching_uuids(uuid, grid.uuid):
            return grid
      return None

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

      if time_series_root is not None:
         return time_series_root
      if self.time_series is None:
         self.time_series = self.root_for_time_series()
      return self.time_series

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

      child = rqet.find_tag(node, tag)
      if child is None:
         return None
      assert rqet.node_type(child) == 'Hdf5Dataset'
      h5_path = rqet.find_tag(child, 'PathInHdfFile').text
      h5_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(child, ['HdfProxy', 'UUID']))
      return (h5_uuid, h5_path)

   def h5_uuid_list(self, node):
      """Returns a list of all uuids for hdf5 external part(s) referred to in recursive tree."""

      def recursive_uuid_set(node):
         uuid_set = set()
         for child in node:
            uuid_set = uuid_set.union(recursive_uuid_set(child))
         if rqet.node_type(node) == 'Hdf5Dataset':
            h5_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['HdfProxy', 'UUID']))
            uuid_set.add(h5_uuid)
         return uuid_set

      return list(recursive_uuid_set(node))

   def h5_uuid(self):
      """Returns the uuid of the 'main' hdf5 file."""

      if self.main_h5_uuid is None:
         uuid_list = None
         ext_parts = self.parts_list_of_type('EpcExternalPartReference')
         if len(ext_parts) == 1:
            self.main_h5_uuid = self.uuid_for_part(ext_parts[0])
         else:
            try:
               grid_root = self.resolve_grid_root()
               if grid_root is not None:
                  uuid_list = self.h5_uuid_list(grid_root)
            except Exception:
               uuid_list = None
            if uuid_list is None or len(uuid_list) == 0:
               for part, (_, _, tree) in self.parts_forest.items():
                  uuid_list = self.h5_uuid_list(tree.getroot())
                  if uuid_list is not None and len(uuid_list) > 0:
                     break
            if uuid_list is not None and len(uuid_list) > 0:
               self.main_h5_uuid = uuid_list[0]  # arbitrary use of first hdf5 uuid
      return self.main_h5_uuid

   def h5_file_name(self, uuid = None, override = True, file_must_exist = True):
      """Returns full path for hdf5 file with given uuid.

      arguments:
         uuid (uuid.UUID, optional): the uuid of the hdf5 external part reference for which the
            file name is required; if None, the 'main' hdf5 uuid is used
         override (boolean, default True): if True, the hdf5 filename stored in the relationship
            information is preferentially ignored and the hdf5 path is generated from the
            epc filename and directory; if False, the hdf5 filename is extracted from the
            relationship information, though the directory of the epc will still be
            preferentially used
         file_must_exist (boolean, default True): if True, the existence of the hdf5
            file is checked and None is returned if the file is not found

      returns:
         string being the full path of the hdf5 file

      notes:
         depending on the settings of the override and file_must_exist arguments, various
         file names and directories might be tested in order to determine the returned path;
         the default arguments will cause a filename based on the epc file and directory
         to be returned;
         in practice, a resqml model consists of a pair of files in the same directory,
         with names like: a.epc and a.h5;
         to allow copying, moving and renaming of files, the practical approach is simply
         to assume a one-to-one correspondence between epc and hdf5 files, and assume they
         are in the same directory
      """

      if override and self.epc_file and self.epc_file.endswith('.epc'):
         h5_full_path = self.epc_file[:-4] + '.h5'
         if not file_must_exist or os.path.exists(h5_full_path):
            return h5_full_path
      if uuid is None:
         uuid = self.h5_uuid()
      if uuid.bytes in self.h5_dict:
         return self.h5_dict[uuid.bytes]
      for rel_name in self.rels_forest:
         entry = self.rels_forest[rel_name]
         if bu.matching_uuids(uuid, entry[0]):
            rel_root = entry[1].getroot()
            for child in rel_root:
               if child.attrib['Id'] == 'Hdf5File' and child.attrib['TargetMode'] == 'External':
                  h5_full_path = None
                  target_path = rqet.strip_path(child.attrib['Target'])
                  if target_path:
                     if self.epc_directory:
                        _, target_file = os.path.split(target_path)
                        h5_full_path = os.path.join(self.epc_directory, target_file)
                     else:
                        h5_full_path = target_path
                     if file_must_exist and not os.path.exists(h5_full_path):
                        h5_full_path = None
                  if h5_full_path is None:
                     h5_full_path = child.attrib['Target']
                     if file_must_exist and not os.path.exists(h5_full_path):
                        h5_full_path = None
                  if h5_full_path is not None:
                     self.h5_dict[uuid.bytes] = h5_full_path
                  return h5_full_path
      return None

   def h5_access(self, uuid = None, mode = 'r', override = True, file_path = None):
      """Returns an open h5 file handle for the hdf5 file with the given uuid.

      arguments:
         uuid (uuid.UUID): the uuid of the hdf5 external part reference for which the
            open file handle is required; required if override is False and file_path is None
         mode (string): the hdf5 file mode ('r', 'w' or 'a') with which to open the file
         override (boolean, default True): if True, the h5 filename is generated based on
            the epc file name, rather than on the filename held in the relationships data
         file_path (string, optional): if present, is used as the hdf5 file path, otherwise
            the path will be determined based on the uuid and override arguments

      returns:
         a file handle to the opened hdf5 file

      note:
         an exception will be raised if the hdf5 file cannot be opened; note that sometimes another
         piece of code accessing the file might cause a 'resource unavailable' exception
      """

      if self.h5_currently_open_mode is not None and self.h5_currently_open_mode != mode:
         self.h5_release()
      if file_path:
         file_name = file_path
      else:
         file_name = self.h5_file_name(uuid = uuid, override = override, file_must_exist = (mode == 'r'))
      if mode == 'a' and not os.path.exists(file_name):
         mode = 'w'
      if self.h5_currently_open_path == file_name:
         return self.h5_currently_open_root
      if self.h5_currently_open_root is not None:
         self.h5_release()
      self.h5_currently_open_path = file_name
      self.h5_currently_open_mode = mode
      self.h5_currently_open_root = h5py.File(file_name, mode)  # could use try to trap file in use errors?
      return self.h5_currently_open_root

   def h5_release(self):
      """Releases (closes) the currently open hdf5 file.

      returns:
         None

      :meta common:
      """

      if self.h5_currently_open_root is not None:
         self.h5_currently_open_root.close()
         self.h5_currently_open_root = None
      self.h5_currently_open_path = None
      self.h5_currently_open_mode = None

   def h5_array_shape_and_type(self, h5_key_pair):
      """Returns the shape and dtype of the array, as stored in the hdf5 file.

      arguments:
         h5_key_pair (uuid.UUID, string): the uuid of the hdf5 external part reference and the hdf5 internal path for the array

      returns:
         (tuple of ints, type): simply the shape and dtype attributes of the referenced hdf5 array; (None, None) is returned
         if the hdf5 file is not found, or the array is not found within it
      """

      h5_root = self.h5_access(h5_key_pair[0])
      if h5_root is None:
         return (None, None)
      shape_tuple = tuple(h5_root[h5_key_pair[1]].shape)
      dtype = h5_root[h5_key_pair[1]].dtype
      return (shape_tuple, dtype)

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

      def reshaped_index(index, shape_tuple, required_shape):
         tail = len(shape_tuple) - len(index)
         if tail > 0:
            assert shape_tuple[-tail:] == required_shape[-tail:], 'not enough indices to allow reshaped indexing'
         natural = 0
         extent = 1
         for axis in range(len(shape_tuple) - tail - 1, -1, -1):
            natural += index[axis] * extent
            extent *= shape_tuple[axis]
         r_extent = np.empty(len(required_shape) - tail, dtype = int)
         r_extent[-1] = required_shape[-(tail + 1)]
         for axis in range(len(required_shape) - tail - 2, -1, -1):
            r_extent[axis] = required_shape[axis] * r_extent[axis + 1]
         r_index = np.empty(len(required_shape) - tail, dtype = int)
         for axis in range(len(r_index) - 1):
            r_index[axis], natural = divmod(natural, r_extent[axis + 1])
         r_index[-1] = natural
         return r_index

      if object is None:
         object = self

      # Check if attribute has already be cached
      if array_attribute is not None:
         existing_value = getattr(object, array_attribute, None)

         # Watch out for np.array(None): check existing_value has a valid "shape"
         if existing_value is not None and getattr(existing_value, "shape", False):
            if index is None:
               return None  # this option allows caching of array without actually referring to any element
            return existing_value[tuple(index)]

      h5_root = self.h5_access(h5_key_pair[0])
      if h5_root is None:
         return None
      if cache_array:
         shape_tuple = tuple(h5_root[h5_key_pair[1]].shape)
         if required_shape is None or shape_tuple == required_shape:
            object.__dict__[array_attribute] = np.zeros(shape_tuple, dtype = dtype)
            object.__dict__[array_attribute][:] = h5_root[h5_key_pair[1]]
         else:
            object.__dict__[array_attribute] = np.zeros(required_shape, dtype = dtype)
            object.__dict__[array_attribute][:] = np.array(h5_root[h5_key_pair[1]],
                                                           dtype = dtype).reshape(required_shape)
         self.h5_release()
         if index is None:
            return None
         return object.__dict__[array_attribute][tuple(index)]
      else:
         if index is None:
            return None
         if required_shape is None:
            result = h5_root[h5_key_pair[1]][tuple(index)]
         else:
            shape_tuple = tuple(h5_root[h5_key_pair[1]].shape)
            if shape_tuple == required_shape:
               result = h5_root[h5_key_pair[1]][tuple(index)]
            else:
               index = reshaped_index(index, required_shape, shape_tuple)
               result = h5_root[h5_key_pair[1]][tuple(index)]
         if dtype is None:
            return result
         if result.size == 1:
            if dtype is float or (isinstance(dtype, str) and dtype.startswith('float')):
               return float(result)
            elif dtype is int or (isinstance(dtype, str) and dtype.startswith('int')):
               return int(result)
            elif dtype is bool or (isinstance(dtype, str) and dtype.startswith('bool')):
               return bool(result)
         return np.array(result, dtype = dtype)

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

      h5_root = self.h5_access(h5_key_pair[0])
      return h5_root[h5_key_pair[1]][slice_tuple]

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

      h5_root = self.h5_access(h5_key_pair[0], mode = 'a')
      dset = h5_root[h5_key_pair[1]]
      dset[slice_tuple] = array_slice

   def create_root(self):
      """Initialises an empty main xml tree for model.

      note:
         not usually called directly
      """

      assert (self.main_tree is None)
      assert (self.main_root is None)
      self.main_root = rqet.Element(ns['content_types'] + 'Types')
      self.main_tree = rqet.ElementTree(element = self.main_root)

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

      use_other = (content_type == 'docProps')
      if use_other:
         if rqet.pretend_to_be_fesapi or rqet.use_fesapi_quirks:
            prefix = '/'
         else:
            prefix = ''
         part_name = prefix + 'docProps/core.xml'
         ct = 'application/vnd.openxmlformats-package.core-properties+xml'
      else:
         part_name = rqet.part_name_for_object(content_type, uuid, prefixed = False, epc_subdir = epc_subdir)
         if 'EpcExternalPartReference' in content_type:
            ct = 'application/x-eml+xml;version=2.0;type=' + content_type
         else:
            ct = 'application/x-resqml+xml;version=2.0;type=' + content_type


#      log.debug('adding part: ' + part_name)
      if isinstance(uuid, str):
         uuid = bu.uuid_from_string(uuid)
      part_tree = rqet.ElementTree(element = root)
      if use_other:
         self.other_forest[part_name] = (content_type, part_tree)
      else:
         if content_type[0].isupper():
            content_type = 'obj_' + content_type
         self.parts_forest[part_name] = (content_type, uuid, part_tree)
         self._set_uuid_to_part(part_name)
      main_ref = rqet.SubElement(self.main_root, ns['content_types'] + 'Override')
      main_ref.set('PartName', part_name)
      main_ref.set('ContentType', ct)
      if add_relationship_part and self.rels_present:
         rels_node = rqet.Element(ns['rels'] + 'Relationships')
         rels_node.text = '\n'
         rels_tree = rqet.ElementTree(element = rels_node)
         if use_other:
            rels_part_name = '_rels/.rels'
         else:
            rels_part_name = rqet.rels_part_name_for_part(part_name)
         self.rels_forest[rels_part_name] = (uuid, rels_tree)
      self.set_modified()

   def patch_root_for_part(self, part, root):
      """Updates the xml tree for the part without changing the uuid."""

      content_type, uuid, part_tree = self.parts_forest[part]
      assert bu.matching_uuids(uuid, rqet.uuid_for_part_root(root))
      part_tree = rqet.ElementTree(element = root)
      self.parts_forest[part] = (content_type, uuid, part_tree)

   def remove_part(self, part_name, remove_relationship_part = True):
      """Removes a part from the parts forest; optionally remove corresponding rels part and other relationships."""

      self._del_uuid_to_part(part_name)
      self.parts_forest.pop(part_name)
      if remove_relationship_part:
         if 'docProps' in part_name:
            rels_part_name = '_rels/.rels'
         else:
            related_parts = self.parts_list_filtered_by_related_uuid(self.list_of_parts(),
                                                                     rqet.uuid_in_part_name(part_name))
            for relative in related_parts:
               (rel_uuid, rel_tree) = self.rels_forest[rqet.rels_part_name_for_part(relative)]
               rel_root = rel_tree.getroot()
               for child in rel_root:
                  if rqet.stripped_of_prefix(child.tag) != 'Relationship':
                     continue
                  if child.attrib['Target'] == part_name:
                     rel_root.remove(child)
            rels_part_name = rqet.rels_part_name_for_part(part_name)
         self.rels_forest.pop(rels_part_name)
      self.remove_part_from_main_tree(part_name)
      self.set_modified()

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

      if flavour.startswith('obj_'):
         flavour = flavour[4:]

      node = rqet.Element(ns[name_space] + flavour)
      node.set('schemaVersion', '2.0')
      node.set('uuid', str(bu.new_uuid()))
      if is_top_lvl_obj:
         node.set(ns['xsi'] + 'type', ns[name_space] + 'obj_' + flavour)
      node.text = rqet.null_xml_text

      return node

   def referenced_node(self, ref_node, consolidate = False):
      """For a given xml reference node, returns the node for the object referred to, if present."""

      # note: the RESQML standard allows referenced objects to be missing from the package (model)

      if ref_node is None:
         return None
      #      content_type = rqet.find_tag_text(ref_node, 'ContentType')
      uuid = bu.uuid_from_string(rqet.find_tag_text(ref_node, 'UUID'))
      if uuid is None:
         return None
      #      return self.root_for_part(self.parts_list_of_type(type_of_interest = content_type, uuid = uuid))
      if consolidate and self.consolidation is not None and uuid in self.consolidation.map:
         resident_uuid = self.consolidation.map[uuid]
         if resident_uuid is None:
            return None
         node = self.root_for_part(self.part_for_uuid(resident_uuid))
         if node is not None:
            # patch resident uuid and title into ref node!
            uuid_node = rqet.find_tag(ref_node, 'UUID')
            uuid_node.text = str(resident_uuid)
            title_node = rqet.find_tag(ref_node, 'Title')
            if title_node is not None:
               title = rqet.citation_title_for_node(node)
               if title:
                  title_node.text = str(title)
      else:
         node = self.root_for_part(self.part_for_uuid(uuid))
      return node

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

      assert uuid is not None

      if flavour.startswith('obj_'):
         flavour = flavour[4:]

      if not content_type:
         content_type = 'obj_' + flavour
      else:
         if content_type[0].isupper():
            content_type = 'obj_' + content_type

      prefix = ns['eml'] if flavour == 'HdfProxy' else ns['resqml2']
      ref_node = rqet.Element(prefix + flavour)
      ref_node.set(ns['xsi'] + 'type', ns['eml'] + 'DataObjectReference')
      ref_node.text = rqet.null_xml_text

      ct_node = rqet.SubElement(ref_node, ns['eml'] + 'ContentType')
      ct_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
      if 'EpcExternalPartReference' in content_type:
         ct_node.text = 'application/x-eml+xml;version=2.0;type=' + content_type
      else:
         ct_node.text = 'application/x-resqml+xml;version=2.0;type=' + content_type

      if not title:
         title = '(title unavailable)'
      title_node = rqet.SubElement(ref_node, ns['eml'] + 'Title')
      title_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
      title_node.text = title

      uuid_node = rqet.SubElement(ref_node, ns['eml'] + 'UUID')
      uuid_node.set(ns['xsi'] + 'type', ns['eml'] + 'UuidString')
      uuid_node.text = str(uuid)

      if use_version_string:
         version_str = rqet.SubElement(ref_node, ns['eml'] + 'VersionString')  # I'm guessing what this is
         version_str.set(ns['xsi'] + 'type', ns['eml'] + 'NameString')
         version_str.text = bu.version_string(uuid)

      if root is not None:
         root.append(ref_node)

      return ref_node

   def uom_node(self, root, uom):
      """Add a generic unit of measure sub element to root.

      arguments:
         root: xml node to which unit of measure subelement (child) will be added
         uom (string): the resqml unit of measure

      returns:
         newly created unit of measure node (having already been added to root)
      """

      assert root is not None and uom is not None and len(uom)

      uom_node = rqet.SubElement(root, ns['resqml2'] + 'UOM')
      uom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlUom')
      uom_node.text = uom

      return uom_node

   def create_rels_part(self):
      """Adds a relationships reference node as a new part in the model's parts forest.

      returns:
         newly created main relationships reference xml node

      note:
         there can only be one relationships reference part in the model
      """

      rels = rqet.SubElement(self.main_root, ns['content_types'] + 'Default')
      rels.set('Extension', 'rels')
      rels.set('ContentType', 'application/vnd.openxmlformats-package.relationships+xml')
      self.rels_present = True
      self.set_modified()

      return rels

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

      if not title:
         title = '(no title)'

      citation = rqet.Element(ns['eml'] + 'Citation')
      citation.set(ns['xsi'] + 'type', ns['eml'] + 'Citation')
      citation.text = rqet.null_xml_text

      title_node = rqet.SubElement(citation, ns['eml'] + 'Title')
      title_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
      title_node.text = title

      originator_node = rqet.SubElement(citation, ns['eml'] + 'Originator')
      if originator is None:
         try:
            originator = str(getpass.getuser())
         except Exception:
            originator = 'unknown'
      originator_node.set(ns['xsi'] + 'type', ns['eml'] + 'NameString')
      originator_node.text = originator

      creation_node = rqet.SubElement(citation, ns['eml'] + 'Creation')
      creation_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
      creation_node.text = time.now()

      format_node = rqet.SubElement(citation, ns['eml'] + 'Format')
      format_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
      if rqet.pretend_to_be_fesapi:
         format_node.text = '[F2I-CONSULTING:fesapi]'
      else:
         format_node.text = 'bp:resqpy:1.0'

      # todo: add optional description field

      if root is not None:
         root.append(citation)

      return citation

   def title_for_root(self, root = None):
      """Returns the Title text from the Citation within the given root node.

      arguments:
         root: the xml node for the object for which the citation title is required

      returns:
         string being the Title text from the citation node which is a child of root,
         or None if not found

      :meta common:
      """

      title = rqet.find_tag(rqet.find_tag(root, 'Citation'), 'Title')
      if title is None:
         return None

      return title.text

   def title_for_part(self, part_name):  # duplicate functionality to citation_title_for_part()
      """Returns the Title text from the Citation for the given main part name (not for rels).

      arguments:
         part_name (string): the name of the part for which the citation title is required

      returns:
         string being the Title text from the citation node which is a child of the root xml node
         for the part, or None if not found

      :meta common:
      """

      return self.title_for_root(self.root_for_part(part_name))

   def create_unknown(self, root = None):
      """Creates an Unknown node and optionally adds as child of root.

      arguments:
         root (optional): if present, the newly created Unknown node is appended as a child
            of this xml node

      returns:
         the newly created Unknown xml node

      :meta common:
      """

      unknown = rqet.Element(ns['eml'] + 'Unknown')
      unknown.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
      unknown.text = 'Unknown'
      if root is not None:
         root.append(unknown)
      return unknown

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

      dp = rqet.Element(ns['cp'] + 'coreProperties')
      dp.text = rqet.null_xml_text

      created = rqet.SubElement(dp, ns['dcterms'] + 'created')
      created.set(ns['xsi'] + 'type', ns['dcterms'] + 'W3CDTF')  # not sure of namespace here
      created.text = time.now()

      if originator is None:
         try:
            originator = str(os.getlogin())
         except Exception:
            originator = 'unknown'
      creator = rqet.SubElement(dp, ns['dc'] + 'creator')
      creator.text = originator

      ver = rqet.SubElement(dp, ns['cp'] + 'version')
      ver.text = '1.0'

      if root is not None:
         root.append(dp)
      if add_as_part:
         self.add_part('docProps', None, dp)
         if self.rels_present:
            (_, rel_tree) = self.rels_forest['_rels/.rels']
            core_rel = rqet.SubElement(rel_tree.getroot(), ns['rels'] + 'Relationship')
            core_rel.set('Id', 'CoreProperties')
            core_rel.set('Type', ns_url['rels_md'] + 'core-properties')
            core_rel.set('Target', 'docProps/core.xml')
      return dp

   def create_crs(self,
                  add_as_part = True,
                  title = 'cell grid local CRS',
                  epsg_code = None,
                  originator = None,
                  x_offset = 0.0,
                  y_offset = 0.0,
                  z_offset = 0.0,
                  areal_rotation_radians = 0.0,
                  xy_units = 'm',
                  z_units = 'm',
                  z_inc_down = True):
      """DEPRECATED: Creates a Coordinate Reference System node and optionally adds as child of root and/or to parts forest.

      arguments:
         add_as_part (boolean, default True): if True the newly created crs node is added to the model
            as a part
         title (string): used as the Title text in the citation node
         epsg_code (integer): EPSG code of the parent coordinate reference system that this crs sits within;
            used for both projected and vertical frames of reference; if None then unknown settings are used
         originator (string, optional): the name of the human being who created the crs object;
            default is to use the login name
         x_offset, y_offset, z_offset (floats, default 0.0): the local origin within the parent coordinate
            reference system space
         areal_rotation_radians (float, default 0.0): the areal rotation of the xy axes of this crs relative
            to the parent coordinate reference system
         xy_units (string, default 'm'): the length units of x & y values in this crs; 'm' or 'ft'
         z_units (string, default 'm'): the length units of z values in this crs; 'm' or 'ft'
         z_inc_down (boolean, default True): if True, z values increase with depth; if False, z values increase
            with elevation

      returns:
         newly created coordinate reference system xml node
      """

      warnings.warn("model.create_crs is Deprecated, will be removed", DeprecationWarning)
      crs = rqc.Crs(self,
                    x_offset = x_offset,
                    y_offset = y_offset,
                    z_offset = z_offset,
                    rotation = areal_rotation_radians,
                    xy_units = xy_units,
                    z_units = z_units,
                    z_inc_down = z_inc_down,
                    epsg_code = epsg_code)

      crs_node = crs.create_xml(add_as_part = add_as_part, title = title, originator = originator)

      if self.crs_uuid is None:
         self.crs_uuid = crs.uuid

      return crs_node

   def create_crs_reference(self, crs_root = None, root = None, crs_uuid = None):
      """Creates a node refering to an existing crs node and optionally adds as child of root.

      arguments:
         crs_root: DEPRECATED, use uuid instead; the root xml node for the coordinate reference system being referenced
         root: the xml node to which the new reference node is to appended as a child (ie. the xml node
            for the object that is referring to the crs)
         crs_uuid: the uuid of the crs

      returns:
         newly created crs reference xml node
      """

      if crs_uuid is None:
         warnings.warn('use of crs_root is deprecated in Model.create_crs_reference(); use crs_uuid instead')
         crs_uuid = rqet.uuid_for_part_root(crs_root)
      else:
         crs_root = self.root_for_uuid(crs_uuid)
      assert crs_root is not None

      return self.create_ref_node('LocalCrs',
                                  rqet.find_nested_tags_text(crs_root, ['Citation', 'Title']),
                                  crs_uuid,
                                  content_type = 'obj_LocalDepth3dCrs',
                                  root = root)

   def create_md_datum_reference(self, md_datum_root, root = None):
      """Creates a node refering to an existing measured depth datum and optionally adds as child of root.

      arguments:
         md_datum_root: the root xml node for the measured depth datum being referenced
         root: the xml node to which the new reference node is to appended as a child (ie. the xml node
            for the object that is referring to the md datum)

      returns:
         newly created measured depth datum reference xml node
      """

      return self.create_ref_node('MdDatum',
                                  rqet.find_nested_tags_text(md_datum_root, ['Citation', 'Title']),
                                  bu.uuid_from_string(md_datum_root.attrib['uuid']),
                                  content_type = 'obj_MdDatum',
                                  root = root)

   def create_hdf5_ext(self, add_as_part = True, root = None, title = 'Hdf Proxy', originator = None, file_name = None):
      """Creates an hdf5 external node and optionally adds as child of root and/or to parts forest.

      arguments:
         add_as_part (boolean, default True): if True the newly created ext node is added to the model
            as a part
         root (optional, usually None): if not None, the newly created ext node is appended as a child
            of this node
         title (string): used as the Title text in the citation node, usually left at the default 'Hdf Proxy'
         originator (string, optional): the name of the human being who created the ext object;
            default is to use the login name
         file_name (string): the filename to be stored as the Target in the relationship node

      returns:
         newly created hdf5 external part xml node
      """

      ext = self.new_obj_node('EpcExternalPartReference', name_space = 'eml')

      self.create_citation(root = ext, title = title, originator = originator)

      mime_type = rqet.SubElement(ext, ns['eml'] + 'MimeType')
      mime_type.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
      mime_type.text = 'application/x-hdf5'

      if root is not None:
         root.append(ext)
      if add_as_part:
         ext_uuid = bu.uuid_from_string(ext.attrib['uuid'])
         self.add_part('obj_EpcExternalPartReference', ext_uuid, ext)
         if not file_name:
            file_name = self.h5_file_name(file_must_exist = False)
         assert file_name
         self.h5_dict[ext_uuid] = file_name
         if self.main_h5_uuid is None:
            self.main_h5_uuid = ext_uuid
         if self.rels_present and file_name:
            (uuid, rel_tree) = self.rels_forest[rqet.rels_part_name_for_part(
               rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid))]
            assert (bu.matching_uuids(uuid, ext_uuid))
            rel_node = rqet.SubElement(rel_tree.getroot(), ns['rels'] + 'Relationship')
            rel_node.set('Id', 'Hdf5File')
            rel_node.set('Type', ns_url['rels_ext'] + 'externalResource')
            rel_node.set('Target', rqet.strip_path(file_name))
            rel_node.set('TargetMode', 'External')
      return ext

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

      assert root is not None
      assert group_tail

      if group_tail[0] == '/':
         group_tail = group_tail[1:]
      if group_tail[-1] == '/':
         group_tail = group_tail[:-1]
      hdf5_path = '/RESQML/' + str(object_uuid) + '/' + group_tail

      path_node = rqet.Element(ns['eml'] + 'PathInHdfFile')
      path_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
      path_node.text = hdf5_path
      root.append(path_node)

      self.create_ref_node('HdfProxy', title, hdf5_uuid, content_type = 'obj_EpcExternalPartReference', root = root)

      return path_node

   # todo: def create_property_kind():

   def create_supporting_representation(self,
                                        grid_root = None,
                                        support_uuid = None,
                                        root = None,
                                        title = None,
                                        content_type = 'obj_IjkGridRepresentation'):
      """Craate a supporting representation reference node refering to an IjkGrid and optionally add to root.

      arguments:
         grid_root: the xml node of the grid object which is the supporting representation being
            referred to; could also be used for other classes of supporting object; this or support_uuid
            must be provided
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

      assert grid_root is not None or support_uuid is not None

      # todo: check that grid_root is for an IJK Grid
      #       (or other content type when handled, eg. blocked wells?)

      if grid_root is not None:
         uuid = rqet.uuid_for_part_root(grid_root)
         if uuid is not None:
            support_uuid = uuid
         if title is None:
            title = rqet.citation_title_for_node(grid_root)
      elif title is None:
         title = 'supporting representation'
      assert support_uuid is not None

      return self.create_ref_node('SupportingRepresentation',
                                  title,
                                  support_uuid,
                                  content_type = content_type,
                                  root = root)

   def create_source(self, source, root = None):
      """Create an extra meta data node holding information on the source of the data, optionally add to root.

      arguments:
         source (string): text describing the source of an object
         root: if not None, the newly created extra metadata node is appended as a child of this node

      returns:
         the newly created extra metadata xml node
      """

      emd_node = rqet.Element(ns['resqml2'] + 'ExtraMetadata')
      emd_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'NameValuePair')
      emd_node.text = rqet.null_xml_text

      name_node = rqet.SubElement(emd_node, ns['resqml2'] + 'Name')
      name_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
      name_node.text = 'source'

      value_node = rqet.SubElement(emd_node, ns['resqml2'] + 'Value')
      value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
      value_node.text = source

      if root is not None:
         root.append(emd_node)
      return emd_node

   def create_patch(self,
                    p_uuid,
                    ext_uuid = None,
                    root = None,
                    patch_index = 0,
                    hdf5_type = 'DoubleHdf5Array',
                    xsd_type = 'double',
                    null_value = None,
                    const_value = None,
                    const_count = None):
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

      returns:
         newly created xml node for the patch of values

      note:
         this function does not write the data to the hdf5; that should be done separately before
         calling this method;
         RESQML usually stores array data in the hdf5 file however constant arrays are flagged as
         such in the xml and no data is stored in the hdf5
      """

      if const_value is None:
         assert ext_uuid is not None
      else:
         assert const_count is not None and const_count > 0
         if hdf5_type.endswith('Hdf5Array'):
            hdf5_type = hdf5_type[:-9] + 'ConstantArray'

      lxt = str(xsd_type).lower()
      discrete = ('int' in lxt) or ('bool' in lxt)

      patch_node = rqet.Element(ns['resqml2'] + 'PatchOfValues')
      patch_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'PatchOfValues')
      patch_node.text = rqet.null_xml_text

      rep_patch_index = rqet.SubElement(patch_node, ns['resqml2'] + 'RepresentationPatchIndex')
      rep_patch_index.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
      rep_patch_index.text = str(patch_index)

      outer_values_node = rqet.SubElement(patch_node, ns['resqml2'] + 'Values')
      outer_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + hdf5_type)  # may also be constant array type
      outer_values_node.text = rqet.null_xml_text

      if discrete:
         if null_value is None:
            if str(xsd_type).startswith('u'):
               null_value = 4294967295  # 2^32 - 1, used as default even for 64 bit data!
            else:
               null_value = -1
         null_value_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'NullValue')
         null_value_node.set(ns['xsi'] + 'type', ns['xsd'] + xsd_type)
         null_value_node.text = str(null_value)

      if const_value is None:

         inner_values_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'Values')
         inner_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         inner_values_node.text = rqet.null_xml_text

         self.create_hdf5_dataset_ref(ext_uuid, p_uuid, 'values_patch{}'.format(patch_index), root = inner_values_node)

      else:

         const_value_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'Value')
         const_value_node.set(ns['xsi'] + 'type', ns['xsd'] + xsd_type)
         const_value_node.text = str(const_value)

         const_count_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'Count')
         const_count_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
         const_count_node.text = str(const_count)

      if root is not None:
         root.append(patch_node)

      return patch_node

   def create_time_series_ref(self, time_series_uuid, root = None):
      """Create a reference node to a time series, optionally add to root.

      arguments:
         time_series_uuid (uuid.UUID): the uuid of the time series part being referenced
         root (optional): if present, the newly created time series reference xml node is added
            as a child to this node

      returns:
         the newly created time series reference xml node
      """

      return self.create_ref_node('TimeSeries', 'time series', time_series_uuid, root = root)

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

      # todo: check namespaces
      p3d = rqet.SubElement(root, ns['resqml2'] + flavour)
      p3d.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3d')
      p3d.text = rqet.null_xml_text

      for axis in range(3):
         coord_node = rqet.SubElement(p3d, ns['resqml2'] + 'Coordinate' + str(axis + 1))
         coord_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
         coord_node.text = str(xyz[axis])

      return p3d

   def create_reciprocal_relationship(self, node_a, rel_type_a, node_b, rel_type_b, avoid_duplicates = True):
      """Adds a node to each of a pair of trees in the rels forest, to represent a two-way relationship.

      arguments:
         node_a: one of the two xml nodes to be related
         rel_type_a (string): the Type (role) associated with node_a in the relationship;
            usually 'sourceObject' or 'destinationObject'
         node_b: the other xml node to be related
         rel_type_a (string): the Type (role) associated with node_b in the relationship
            usually 'sourceObject' or 'destinationObject' (opposite of rel_type_a)
         avoid_duplicates (boolean, default True): if True, xml for a relationship is not added
            where it already exists; if False, a duplicate will be created in this situation

      returns:
         None
      """

      def id_str(uuid):
         stringy = str(uuid)
         if not (rqet.pretend_to_be_fesapi or rqet.use_fesapi_quirks) or not stringy[0].isdigit():
            return stringy
         return '_' + stringy

      assert (self.rels_present)

      if node_a is None or node_b is None:
         log.error('attempt to create relationship with missing object')
         return

      uuid_a = node_a.attrib['uuid']
      obj_type_a = rqet.stripped_of_prefix(rqet.content_type(node_a.attrib[ns['xsi'] + 'type']))
      part_name_a = rqet.part_name_for_object(obj_type_a, uuid_a)
      rel_part_name_a = rqet.rels_part_name_for_part(part_name_a)
      (rel_uuid_a, rel_tree_a) = self.rels_forest[rel_part_name_a]
      rel_root_a = rel_tree_a.getroot()

      uuid_b = node_b.attrib['uuid']
      obj_type_b = rqet.stripped_of_prefix(rqet.content_type(node_b.attrib[ns['xsi'] + 'type']))
      part_name_b = rqet.part_name_for_object(obj_type_b, uuid_b)
      rel_part_name_b = rqet.rels_part_name_for_part(part_name_b)
      (rel_uuid_b, rel_tree_b) = self.rels_forest[rel_part_name_b]
      rel_root_b = rel_tree_b.getroot()

      create_a = True
      if avoid_duplicates:
         existing_rel_nodes = rqet.list_of_tag(rel_root_a, 'Relationship')
         for existing in existing_rel_nodes:
            if (rqet.stripped_of_prefix(existing.attrib['Type']) == rel_type_a and
                existing.attrib['Target'] == part_name_b):
               create_a = False
               break
      if create_a:
         rel_a = rqet.SubElement(rel_root_a, ns['rels'] + 'Relationship')
         rel_a.set(
            'Id', id_str(uuid_b))  # NB: fesapi prefixes uuid with _ for some rels only (where uuid starts with a digit)
         rel_a.set('Type', ns_url['rels_ext'] + rel_type_a)
         rel_a.set('Target', part_name_b)

      create_b = True
      if avoid_duplicates:
         existing_rel_nodes = rqet.list_of_tag(rel_root_b, 'Relationship')
         for existing in existing_rel_nodes:
            if (rqet.stripped_of_prefix(existing.attrib['Type']) == rel_type_b and
                existing.attrib['Target'] == part_name_a):
               create_b = False
               break
      if create_b:
         rel_b = rqet.SubElement(rel_root_b, ns['rels'] + 'Relationship')
         rel_b.set(
            'Id', id_str(uuid_a))  # NB: fesapi prefixes uuid with _ for some rels only (where uuid starts with a digit)
         rel_b.set('Type', ns_url['rels_ext'] + rel_type_b)
         rel_b.set('Target', part_name_a)

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
         prior to adding as part
      """

      new_node = copy.deepcopy(existing_node)
      if add_as_part:
         uuid = rqet.uuid_for_part_root(new_node)
         if self.part_for_uuid(uuid) is None:
            self.add_part(rqet.node_type(new_node), uuid, new_node)
         else:
            log.warning('rejected attempt to add a duplicated part with an existing uuid')
      return new_node

   def force_consolidation_uuid_equivalence(self, immigrant_uuid, resident_uuid):
      """Forces object identified by immigrant uuid to be teated as equivalent to that with resident uuid during consolidation."""

      if self.consolidation is None:
         self.consolidation = cons.Consolidation(self)
      self.consolidation.force_uuid_equivalence(immigrant_uuid, resident_uuid)

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
                                  other_h5_file_name = None):
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

      notes:
         if the part name already exists in this model, no action is taken;
         default hdf5 file used in this model and assumed in other_model
      """

      # todo: double check behaviour around equivalent CRSes, especially any default crs in model

      assert other_model is not None
      if other_model is self:
         return
      assert part is not None
      if realization is not None:
         assert isinstance(realization, int) and realization >= 0
      if force:
         assert consolidate
      if not other_h5_file_name:
         other_h5_file_name = other_model.h5_file_name()
      if not self_h5_file_name:
         self_h5_file_name = self.h5_file_name(file_must_exist = False)

      # check whether already existing in this model
      if part in self.parts_forest.keys():
         return

      if other_model.type_of_part(part) == 'obj_EpcExternalPartReference':
         log.debug('refusing to copy hdf5 ext part from other model')
         return

      log.debug('copying part: ' + str(part))

      uuid = rqet.uuid_in_part_name(part)
      if not force:
         assert self.part_for_uuid(uuid) is None, 'part copying failure: uuid exists for different part!'

      # duplicate xml tree and add as a part
      other_root = other_model.root_for_part(part, is_rels = False)
      if other_root is None:
         log.error('failed to copy part (missing in source model?): ' + str(part))
         return

      if consolidate and not force:
         if self.consolidation is None:
            self.consolidation = cons.Consolidation(self)
         resident_uuid = self.consolidation.equivalent_uuid_for_part(part, immigrant_model = other_model)
      else:
         resident_uuid = None

      if resident_uuid is None:

         root_node = self.duplicate_node(other_root)  # adds duplicated node as part
         assert root_node is not None

         if realization is not None and rqet.node_type(root_node).endswith('Property'):
            ri_node = rqet.find_tag(root_node, 'RealizationIndex')
            if ri_node is None:
               ri_node = rqet.SubElement(root_node, ns['resqml2'] + 'RealizationIndex')
               ri_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
               ri_node.text = str(realization)
            # NB. this intentionally overwrites any pre-existing realization number
            ri_node.text = str(realization)

         # copy hdf5 data
         hdf5_internal_paths = [node.text for node in rqet.list_of_descendant_tag(other_root, 'PathInHdfFile')]
         hdf5_count = whdf5.copy_h5_path_list(other_h5_file_name, self_h5_file_name, hdf5_internal_paths, mode = 'a')

         # create relationship with hdf5 if needed and modify h5 file uuid in xml references
         if hdf5_count:
            if h5_uuid is None:
               h5_uuid = self.h5_uuid()
            if h5_uuid is None:
               self.create_hdf5_ext()
               h5_uuid = self.h5_uuid()
            self.change_hdf5_uuid_in_hdf5_references(root_node, None, h5_uuid)
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', h5_uuid, prefixed = False)
            ext_node = self.root_for_part(ext_part)
            self.create_reciprocal_relationship(root_node,
                                                'mlToExternalPartProxy',
                                                ext_node,
                                                'externalPartProxyToMl',
                                                avoid_duplicates = False)

         # cut references to objects to be excluded
         if cut_refs_to_uuids:
            rqet.cut_obj_references(root_node, cut_refs_to_uuids)

         if cut_node_types:
            rqet.cut_nodes_of_types(root_node, cut_node_types)

         # recursively copy in referenced parts where they don't already exist in this model
         for ref_node in rqet.list_obj_references(root_node):
            resident_referred_node = None
            if consolidate:
               resident_referred_node = self.referenced_node(ref_node, consolidate = True)
            if force:
               continue
            if resident_referred_node is None:
               referred_node = other_model.referenced_node(ref_node)
               if referred_node is None:
                  log.warning(
                     f'referred node not found in other model for {rqet.find_tag_text(ref_node, "Title")}; uuid: {rqet.find_tag_text(ref_node, "UUID")}'
                  )
               else:
                  referred_part = rqet.part_name_for_part_root(referred_node)
                  if other_model.type_of_part(referred_part) == 'obj_EpcExternalPartReference':
                     continue
                  if referred_part in self.list_of_parts():
                     continue
                  self.copy_part_from_other_model(other_model, referred_part, consolidate = consolidate)

         resident_uuid = uuid

      else:

         root_node = self.root_for_uuid(resident_uuid)

      # copy relationships where target part is present in this model – this part is source, then destination
      for source_flag in [True, False]:
         other_related_parts = other_model.parts_list_filtered_by_related_uuid(other_model.list_of_parts(),
                                                                               resident_uuid,
                                                                               uuid_is_source = source_flag)
         for related_part in other_related_parts:
            # log.debug('considering relationship with: ' + str(related_part))
            if not force and (related_part in self.parts_forest):
               resident_related_part = related_part
            else:
               # log.warning('skipping relationship between ' + str(part) + ' and ' + str(related_part))
               if consolidate:
                  resident_related_uuid = self.consolidation.equivalent_uuid_for_part(related_part,
                                                                                      immigrant_model = other_model)
                  if resident_related_uuid is None:
                     continue
                  resident_related_part = rqet.part_name_for_object(other_model.type_of_part(related_part),
                                                                    resident_related_uuid)
                  if resident_related_part is None:
                     continue
               else:
                  continue
            if not force and resident_related_part in self.parts_list_filtered_by_related_uuid(
                  self.list_of_parts(), resident_uuid):
               continue
            related_node = self.root_for_part(resident_related_part)
            assert related_node is not None

            if source_flag:
               sd_a, sd_b = 'sourceObject', 'destinationObject'
            else:
               sd_b, sd_a = 'sourceObject', 'destinationObject'
            self.create_reciprocal_relationship(root_node, sd_a, related_node, sd_b)

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

      assert other_model is not None and other_model is not self

      other_parts_list = other_model.parts()
      if not other_parts_list:
         log.warning('no parts found in other model for merging')
         return

      if consolidate:
         other_parts_list = cons.sort_parts_list(other_model, other_parts_list)

      self_h5_file_name = self.h5_file_name(file_must_exist = False)
      self_h5_uuid = self.h5_uuid()
      other_h5_file_name = other_model.h5_file_name()
      for part in other_parts_list:
         self.copy_part_from_other_model(other_model,
                                         part,
                                         realization = realization,
                                         consolidate = consolidate,
                                         self_h5_file_name = self_h5_file_name,
                                         h5_uuid = self_h5_uuid,
                                         other_h5_file_name = other_h5_file_name)

      if consolidate and self.consolidation is not None:
         self.consolidation.check_map_integrity()

   def iter_objs(self, cls):
      """Iterate over all available objects of given resqpy class within the model

      Note:

         The resqpy class must expose a class attribute `resqml_type`, and must support
         being created with the signature: `obj = cls(model, uuid=uuid)`.

      Example use::

         for well in model.iter_objs(cls=resqpy.well.WellboreFeature):
            print(well.title, well.uuid)

      Args:
         cls: resqpy class to iterate

      Yields:
         list of instances of cls

      :meta common:
      """

      uuids = self.uuids(obj_type = cls.resqml_type)
      for uuid in uuids:
         yield cls(self, uuid = uuid)

   def iter_grid_connection_sets(self):
      """Yields grid connection set objects, one for each gcs in this model."""

      gcs_uuids = self.uuids(obj_type = 'GridConnectionSetRepresentation')
      for gcs_uuid in gcs_uuids:
         yield rqf.GridConnectionSet(self, uuid = gcs_uuid)

   def iter_wellbore_interpretations(self):
      """ Iterable of all WellboreInterpretations associated with the model

      Yields:
         wellbore: instance of :class:`resqpy.organize.WellboreInterpretation`

      :meta common:
      """
      import resqpy.organize  # Imported here for speed, module is not always needed

      uuids = self.uuids(obj_type = 'WellboreInterpretation')
      if uuids:
         for uuid in uuids:
            yield resqpy.organize.WellboreInterpretation(self, uuid = uuid)

   def iter_trajectories(self):
      """ Iterable of all trajectories associated with the model

      Yields:
         trajectory: instance of :class:`resqpy.well.Trajectory`

      :meta common:
      """
      import resqpy.well  # Imported here for speed, module is not always needed

      uuids = self.uuids(obj_type = "WellboreTrajectoryRepresentation")
      for uuid in uuids:
         yield resqpy.well.Trajectory(self, uuid = uuid)

   def iter_md_datums(self):
      """ Iterable of all MdDatum objects associated with the model

      Yields:
         md_datum: instance of :class:`resqpy.well.MdDatum`

      :meta common:
      """
      import resqpy.well  # Imported here to avoid circular imports

      uuids = self.uuids(obj_type = 'MdDatum')
      if uuids:
         for uuid in uuids:
            datum = resqpy.well.MdDatum(self, uuid = uuid)
            yield datum

   def iter_crs(self):
      """Iterable of all CRS objects associated with the model
      
      Yields:
         crs: instance of :class:`resqpy.crs.CRS`

      :meta common:
      """
      import resqpy.crs  # Imported here for speed, module is not always needed

      uuids = self.uuids(obj_type = 'LocalDepth3dCrs') + self.uuids(obj_type = 'LocalTime3dCrs')
      if uuids:
         for uuid in uuids:
            yield resqpy.crs.Crs(self, uuid = uuid)

   def sort_parts_list_by_timestamp(self, parts_list):
      """Returns a copy of the parts list sorted by citation block creation date, with the newest first."""

      if parts_list is None:
         return None
      if len(parts_list) == 0:
         return []
      sort_list = []
      for index, part in enumerate(parts_list):
         timestamp = rqet.find_nested_tags_text(self.root_for_part(part), ['Citation', 'Creation'])
         sort_list.append((timestamp, index))
      sort_list.sort()
      results = []
      for timestamp, index in reversed(sort_list):
         results.append(parts_list[index])
      return results

   def as_graph(self, uuids_subset = None):
      """Return representation of model as nodes and edges, suitable for plotting in a graph

      Note:
         The graph can be most readily visualised with other packages such as
         NetworkX and HoloViews, which are not part of resqpy.

         For a guide to plotting graphs interactively, see:
         http://holoviews.org/user_guide/Network_Graphs.html

      Example::

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

      Args:
         uuids_subset (iterable): If present, only consider uuids in this list.
            By default, use all uuids in the model.

      Returns:
         2-tuple of nodes and edges:

         - nodes: dict mapping uuid to attributes (e.g. citation title)
         - edges: set of unordered pairs of uuids, representing relationships

      :meta common:
      """
      nodes = {}
      edges = set()

      if uuids_subset is None:
         uuids_subset = self.uuids()

      uuids_subset = set(map(str, uuids_subset))

      for uuid in uuids_subset:
         part = self.part_for_uuid(uuid)
         nodes[uuid] = dict(
            resqml_type = self.type_of_part(part, strip_obj = True),
            title = self.citation_title_for_part(part),
         )
         for rel in map(str, self.uuids(related_uuid = uuid)):
            if rel in uuids_subset:
               edges.add(frozenset([uuid, rel]))

      return nodes, edges

   def _set_uuid_to_part(self, part_name):
      """Adds an entry to the dictionary mapping from uuid to part name."""

      uuid = rqet.uuid_in_part_name(part_name)
      self.uuid_part_dict[bu.uuid_as_int(uuid)] = part_name

   def _del_uuid_to_part(self, part_name):
      """Deletes an entry from the dictionary mapping from uuid to part name."""

      uuid = rqet.uuid_in_part_name(part_name)
      try:
         del self.uuid_part_dict[bu.uuid_as_int(uuid)]
      except Exception:
         pass


def new_model(epc_file):
   """Returns a new, empty Model object with basics and hdf5 ext part set up."""

   return Model(epc_file = epc_file, new_epc = True, create_basics = True, create_hdf5_ext = True)
