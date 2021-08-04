"""well.py: resqpy well module providing trajectory, deviation survey, blocked well, wellbore frame and marker frame and md datum classes.

Example::

   # Wellbore interpretations
   for well in model.iter_wellbore_interpretations():
      print(well.title)

      for trajectory in well.iter_trajectories():
         print(trajectory.title)

         for frame in trajectory.iter_wellbore_frames():
            print(frame.title)

            # Measured depths
            mds = frame.node_mds

            # Logs
            log_collection = frame.logs
            for log in log_collection:
               values = log.values()

"""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

version = '14th July 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('well.py version ' + version)

import math as maths
import numpy as np
import pandas as pd
import lasio
import warnings
import os
# import xml.etree.ElementTree as et
# from lxml import etree as et

import resqpy.crs as crs
import resqpy.organize as rqo
import resqpy.property as rqp
import resqpy.lines as rql
import resqpy.weights_and_measures as bwam

import resqpy.olio.grid_functions as gf
import resqpy.olio.vector_utilities as vec
import resqpy.olio.intersection as intersect
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.keyword_files as kf
import resqpy.olio.wellspec_keywords as wsk
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.base import BaseResqpy

valid_md_reference_list = [
   "ground level", "kelly bushing", "mean sea level", "derrick floor", "casing flange", "arbitrary point",
   "crown valve", "rotary bushing", "rotary table", "sea floor", "lowest astronomical tide", "mean higher high water",
   "mean high water", "mean lower low water", "mean low water", "mean tide level", "kickoff point"
]

# todo: could require/maintain DeviationSurvey mds in same units as md datum object's crs vertical units?


class MdDatum(BaseResqpy):
   """Class for RESQML measured depth datum."""

   resqml_type = 'MdDatum'

   def __init__(
         self,
         parent_model,
         uuid = None,
         md_datum_root = None,
         crs_uuid = None,
         crs_root = None,  # deprecated
         location = None,
         md_reference = 'mean sea level',
         title = None,
         originator = None,
         extra_metadata = None):
      """Initialises a new MdDatum object.

      arguments:
         parent_model (model.Model object): the model which the new md datum belongs to
         uuid: If not None, load from existing object. Else, create new.
         md_datum_root (optional): DEPRECATED: the root node of the xml tree representing the md datum;
            if not None, the new md datum object is initialised based on data in the tree;
            if None, the new object is initialised from the remaining arguments
         crs_uuid (uuid.UUID): required if initialising from values
         crs_root: DEPRECATED, use crs_uuid instead; the root node of the coordinate reference system
            xml tree; ignored if uuid or md_datum_root is not None or crs_uuid is not None
         location: (triple float): the x, y, z location of the new measured depth datum;
            ignored if uuid or md_datum_root is not None
         md_reference (string): human readable resqml standard string indicating the real
            world nature of the datum, eg. 'kelly bushing'; the full list of options is
            available as the global variable valid_md_reference_list in this module;
            ignored if uuid or md_datum_root is not None
         title (str, optional): the citation title to use for a new datum;
            ignored if uuid or md_datum_root is not None
         originator (str, optional): the name of the person creating the datum, defaults to login id;
            ignored if uuid or md_datum_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the datum;
            ignored if uuid or md_datum_root is not None

      returns:
         the newly instantiated measured depth datum object

      note:
         this function does not create an xml node for the md datum; call the create_xml() method afterwards
         if initialising from data other than an existing RESQML object
      """

      if crs_root is not None:
         warnings.warn("Attribute 'crs_root' is deprecated. Use 'crs_uuid'", DeprecationWarning)
      # TODO: remove crs_root argument

      self.location = location
      self.md_reference = md_reference
      self.crs_uuid = crs_uuid

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = md_datum_root)

      # temporary code to sort out crs reference, till crs_root arg is retired
      if self.crs_uuid is None and crs_root is not None:
         self.crs_uuid = rqet.uuid_for_part_root(crs_root)

      assert self.crs_uuid is not None
      if self.root is None and (location is not None or md_reference):
         assert location is not None and md_reference
         assert md_reference in valid_md_reference_list
         assert len(location) == 3

   def _load_from_xml(self):
      md_datum_root = self.root
      assert md_datum_root is not None
      location_node = rqet.find_tag(md_datum_root, 'Location')
      self.location = (rqet.find_tag_float(location_node,
                                           'Coordinate1'), rqet.find_tag_float(location_node, 'Coordinate2'),
                       rqet.find_tag_float(location_node, 'Coordinate3'))
      self.md_reference = rqet.node_text(rqet.find_tag(md_datum_root, 'MdReference')).strip().lower()
      assert self.md_reference in valid_md_reference_list
      self.crs_uuid = self.extract_crs_uuid()

   @property
   def crs_root(self):
      """XML node corresponding to self.crs_uuid"""

      return self.model.root_for_uuid(self.crs_uuid)

   # todo: the following function is almost identical to one in the grid module: it should be made common and put in model.py

   def extract_crs_uuid(self):
      """Returns uuid for coordinate reference system, as stored in reference node of this md datum's xml tree."""

      if self.crs_uuid is not None:
         return self.crs_uuid
      crs_root = rqet.find_tag(self.root, 'LocalCrs')
      uuid_str = rqet.find_tag(crs_root, 'UUID').text
      self.crs_uuid = bu.uuid_from_string(uuid_str)
      return self.crs_uuid

   def extract_crs_root(self):
      """Returns root in parent model xml parts forest of coordinate reference system used by this md datum."""

      if self.crs_uuid is None:
         self.extract_crs_uuid()
      return self.crs_root

   def create_part(self):
      """Creates xml for this md datum object and adds to parent model as a part; returns root node for part."""

      # note: deprecated, call create_xml() directly
      assert self.root is None
      assert self.location is not None
      self.create_xml(add_as_part = True)

   def create_xml(self, add_as_part = True, add_relationships = True, title = None, originator = None):
      """Creates xml for a measured depth datum element; crs node must already exist; optionally adds as part.

         arguments:
            add_as_part (boolean, default True): if True, the newly created xml node is added as a part
               in the model
            add_relationships (boolean, default True): if True, a relationship xml part is created relating the
               new md datum part to the crs
            title (string): used as the citation Title text for the new md datum node
            originator (string, optional): the name of the human being who created the md datum part;
               default is to use the login name

         returns:
            the newly created measured depth datum xml node
      """

      md_reference = self.md_reference.lower()
      assert md_reference in valid_md_reference_list, 'invalid measured depth reference: ' + md_reference

      if title:
         self.title = title
      if not self.title:
         self.title = 'measured depth datum'

      crs_uuid = self.crs_uuid
      assert crs_uuid is not None

      datum = super().create_xml(add_as_part = False, originator = originator)

      self.model.create_solitary_point3d('Location', datum, self.location)

      md_ref = rqet.SubElement(datum, ns['resqml2'] + 'MdReference')
      md_ref.set(ns['xsi'] + 'type', ns['resqml2'] + 'MdReference')
      md_ref.text = md_reference

      self.model.create_crs_reference(crs_uuid = crs_uuid, root = datum)

      if add_as_part:
         self.model.add_part('obj_MdDatum', self.uuid, datum)
         if add_relationships:
            self.model.create_reciprocal_relationship(datum, 'destinationObject', self.crs_root, 'sourceObject')

      return datum

   def __eq__(self, other):
      """Implements equals operator. Compares class type and uuid"""

      # TODO: more detailed equality comparison
      other_uuid = getattr(other, "uuid", None)
      return isinstance(other, self.__class__) and bu.matching_uuids(self.uuid, other_uuid)


class DeviationSurvey(BaseResqpy):
   """Class for RESQML wellbore deviation survey.

   RESQML documentation:

      Specifies the station data from a deviation survey.

      The deviation survey does not provide a complete specification of the
      geometry of a wellbore trajectory. Although a minimum-curvature
      algorithm is used in most cases, the implementation varies sufficiently
      that no single algorithmic specification is available as a data transfer
      standard.

      Instead, the geometry of a RESQML wellbore trajectory is represented by
      a parametric line, parameterized by the MD.

      CRS and units of measure do not need to be consistent with the CRS and
      units of measure for wellbore trajectory representation.
   """

   resqml_type = 'DeviationSurveyRepresentation'

   def __init__(self,
                parent_model,
                uuid = None,
                title = None,
                deviation_survey_root = None,
                represented_interp = None,
                md_datum = None,
                md_uom = 'm',
                angle_uom = 'degrees',
                measured_depths = None,
                azimuths = None,
                inclinations = None,
                station_count = None,
                first_station = None,
                is_final = False,
                originator = None,
                extra_metadata = None):
      """Load or create a DeviationSurvey object.

      If uuid is given, loads from XML. Else, create new. If loading from disk, other
      parameters will be overwritten.

      Args:
         parent_model (model.Model): the model which the new survey belongs to
         uuid (uuid.UUID): If given, loads from disk. Else, creates new.
         title (str): Citation title
         deviation_survey_root: DEPCRECATED. If given, load from disk.
         represented_interp (wellbore interpretation): if present, is noted as the wellbore
            interpretation object which this deviation survey relates to
         md_datum (MdDatum): the datum that the depths for this survey are measured from
         md_uom (string, default 'm'): a resqml length unit of measure applicable to the
            measured depths; should be 'm' or 'ft'
         angle_uom (string): a resqml angle unit; should be 'dega' or 'rad'
         measured_depths (np.array): 1d array
         azimuths (np.array): 1d array
         inclindations (np.array): 1d array
         station_count (int): length of measured_depths, azimuths & inclinations
         first_station (tuple): (x, y, z) of first point in survey, in crs for md datum
         is_final (bool): whether survey is a finalised deviation survey
         originator (str): name of author
         extra_metadata (dict, optional): extra metadata key, value pairs

      Returns:
         DeviationSurvey

      Notes:
         this method does not create an xml node, nor write hdf5 arrays
      """

      self.is_final = is_final
      self.md_uom = bwam.rq_length_unit(md_uom)

      self.angles_in_degrees = angle_uom.strip().lower().startswith('deg')
      """boolean: True for degrees, False for radians (nothing else supported). Should be 'dega' or 'rad'"""

      # Array data
      self.measured_depths = _as_optional_array(measured_depths)
      self.azimuths = _as_optional_array(azimuths)
      self.inclinations = _as_optional_array(inclinations)

      if station_count is None and measured_depths is not None:
         station_count = len(measured_depths)
      self.station_count = station_count
      self.first_station = first_station

      # Referenced objects
      self.md_datum = md_datum  # md datum is an object in its own right, with a related crs!
      self.wellbore_interpretation = represented_interp

      # TODO: remove deviation_survey_root, use just uuid

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = deviation_survey_root)

   @classmethod
   def from_data_frame(cls,
                       parent_model,
                       data_frame,
                       md_datum = None,
                       md_col = 'MD',
                       azimuth_col = 'AZIM_GN',
                       inclination_col = 'INCL',
                       x_col = 'X',
                       y_col = 'Y',
                       z_col = 'Z',
                       md_uom = 'm',
                       angle_uom = 'degrees'):
      """Load MD, aximuth & inclination data from a pandas data frame.

      Args:
         parent_model (model.Model): the parent resqml model
         data_frame: a pandas dataframe holding the deviation survey data
         md_datum (MdDatum object): the datum that the depths for this survey are measured from
         md_col (string, default 'MD'): the name of the column holding measured depth values
         azimuth_col (string, default 'AZIM_GN'): the name of the column holding azimuth values relative
            to the north direction (+ve y axis) of the coordinate reference system
         inclination_col (string, default 'INCL'): the name of the column holding inclination values
         x_col (string, default 'X'): the name of the column holding an x value in the first row
         y_col (string, default 'Y'): the name of the column holding an Y value in the first row
         z_col (string, default 'Z'): the name of the column holding an z value in the first row
         md_uom (string, default 'm'): a resqml length unit of measure applicable to the
            measured depths; should be 'm' or 'ft'
         angle_uom (string, default 'degrees'): a resqml angle unit of measure applicable to both
            the azimuth and inclination data

      Returns:
         DeviationSurvey

      Note:
         The X, Y & Z columns are only used to set the first station location (from the first row)
      """

      for col in [md_col, azimuth_col, inclination_col, x_col, y_col, z_col]:
         assert col in data_frame.columns
      station_count = len(data_frame)
      assert station_count >= 2  # vertical well could be hamdled by allowing a single station in survey?
      #         self.md_uom = bwam.p_length_unit(md_uom)

      start = data_frame.iloc[0]

      return cls(parent_model = parent_model,
                 station_count = station_count,
                 md_datum = md_datum,
                 md_uom = md_uom,
                 angle_uom = angle_uom,
                 first_station = (start[x_col], start[y_col], start[z_col]),
                 measured_depths = data_frame[md_col].values,
                 azimuths = data_frame[azimuth_col].values,
                 inclinations = data_frame[inclination_col].values,
                 is_final = True)  # assume this is a finalised deviation survey

   @classmethod
   def from_ascii_file(cls,
                       parent_model,
                       deviation_survey_file,
                       comment_character = '#',
                       space_separated_instead_of_csv = False,
                       md_col = 'MD',
                       azimuth_col = 'AZIM_GN',
                       inclination_col = 'INCL',
                       x_col = 'X',
                       y_col = 'Y',
                       z_col = 'Z',
                       md_uom = 'm',
                       angle_uom = 'degrees',
                       md_datum = None):
      """Load MD, aximuth & inclination data from an ascii deviation survey file.

      Arguments:
         parent_model (model.Model): the parent resqml model
         deviation_survey_file (string): the filename of an ascii file holding the deviation survey data
         comment_character (string): the character to be treated as introducing comments
         space_separated_instead_of_csv (boolea, default False): if False, csv format expected;
            if True, columns are expected to be seperated by white space
         md_col (string, default 'MD'): the name of the column holding measured depth values
         azimuth_col (string, default 'AZIM_GN'): the name of the column holding azimuth values relative
            to the north direction (+ve y axis) of the coordinate reference system
         inclination_col (string, default 'INCL'): the name of the column holding inclination values
         x_col (string, default 'X'): the name of the column holding an x value in the first row
         y_col (string, default 'Y'): the name of the column holding an Y value in the first row
         z_col (string, default 'Z'): the name of the column holding an z value in the first row
         md_uom (string, default 'm'): a resqml length unit of measure applicable to the
            measured depths; should be 'm' or 'ft'
         angle_uom (string, default 'degrees'): a resqml angle unit of measure applicable to both
            the azimuth and inclination data
         md_datum (MdDatum object): the datum that the depths for this survey are measured from

      Returns:
         DeviationSurvey

      Note:
         The X, Y & Z columns are only used to set the first station location (from the first row)
      """

      try:
         df = pd.read_csv(deviation_survey_file,
                          comment = comment_character,
                          delim_whitespace = space_separated_instead_of_csv)
         if df is None:
            raise Exception
      except Exception:
         log.error('failed to read ascii deviation survey file ' + deviation_survey_file)
         raise

      return cls.from_data_frame(parent_model,
                                 df,
                                 md_col = md_col,
                                 azimuth_col = azimuth_col,
                                 inclination_col = inclination_col,
                                 x_col = x_col,
                                 y_col = y_col,
                                 z_col = z_col,
                                 md_uom = md_uom,
                                 angle_uom = angle_uom,
                                 md_datum = md_datum)

   def _load_from_xml(self):
      """Load attributes from xml and associated hdf5 data.

      This is invoked as part of the init method when an existing uuid is given.
      
      Returns:
         [bool]: True if sucessful
      """

      # Get node from self.uuid
      node = self.root
      assert node is not None

      # Load XML data
      self.md_uom = rqet.length_units_from_node(rqet.find_tag(node, 'MdUom', must_exist = True))
      self.angle_uom = rqet.find_tag_text(node, 'AngleUom', must_exist = True)
      self.station_count = rqet.find_tag_int(node, 'StationCount', must_exist = True)
      self.first_station = extract_xyz(rqet.find_tag(node, 'FirstStationLocation', must_exist = True))
      self.is_final = rqet.find_tag_bool(node, 'IsFinal')

      # Load HDF5 data
      mds_node = rqet.find_tag(node, 'Mds', must_exist = True)
      load_hdf5_array(self, mds_node, 'measured_depths')
      azimuths_node = rqet.find_tag(node, 'Azimuths', must_exist = True)
      load_hdf5_array(self, azimuths_node, 'azimuths')
      inclinations_node = rqet.find_tag(node, 'Inclinations', must_exist = True)
      load_hdf5_array(self, inclinations_node, 'inclinations')

      # Set related objects
      self.md_datum = self._load_related_datum()
      self.represented_interp = self._load_related_wellbore_interp()

      # Validate
      assert self.measured_depths is not None
      assert len(self.measured_depths) > 0

      return True

   def create_xml(self,
                  ext_uuid = None,
                  md_datum_root = None,
                  md_datum_xyz = None,
                  add_as_part = True,
                  add_relationships = True,
                  title = None,
                  originator = None):
      """Creates a deviation survey representation xml element from this DeviationSurvey object.

      arguments:
         ext_uuid (uuid.UUID): the uuid of the hdf5 external part holding the deviation survey arrays
         md_datum_root: the root xml node for the measured depth datum that the deviation survey depths
            are based on
         md_datum_xyz: TODO: document this
         add_as_part (boolean, default True): if True, the newly created xml node is added as a part
            in the model
         add_relationships (boolean, default True): if True, a relationship xml part is created relating the
            new deviation survey part to the measured depth datum part
         title (string): used as the citation Title text; should usually refer to the well name in a
            human readable way
         originator (string, optional): the name of the human being who created the deviation survey part;
            default is to use the login name

      returns:
         the newly created deviation survey xml node
      """

      assert self.station_count > 0

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      if md_datum_root is None:
         if self.md_datum is None:
            if md_datum_xyz is None:
               raise ValueError("Must provide a MD Datum for the DeviationSurvey")
            self.md_datum = MdDatum(self.model, location = md_datum_xyz)
         if self.md_datum.root is None:
            md_datum_root = self.md_datum.create_xml()
         else:
            md_datum_root = self.md_datum.root
      assert md_datum_root is not None

      # Create root node, write citation block
      ds_node = super().create_xml(title = title, originator = originator, add_as_part = False)

      if_node = rqet.SubElement(ds_node, ns['resqml2'] + 'IsFinal')
      if_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
      if_node.text = str(self.is_final).lower()

      sc_node = rqet.SubElement(ds_node, ns['resqml2'] + 'StationCount')
      sc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      sc_node.text = str(self.station_count)

      md_uom = rqet.SubElement(ds_node, ns['resqml2'] + 'MdUom')
      md_uom.set(ns['xsi'] + 'type', ns['eml'] + 'LengthUom')
      md_uom.text = bwam.rq_length_unit(self.md_uom)

      self.model.create_md_datum_reference(md_datum_root, root = ds_node)

      self.model.create_solitary_point3d('FirstStationLocation', ds_node, self.first_station)

      angle_uom = rqet.SubElement(ds_node, ns['resqml2'] + 'AngleUom')
      angle_uom.set(ns['xsi'] + 'type', ns['eml'] + 'PlaneAngleUom')
      if self.angles_in_degrees:
         angle_uom.text = 'dega'
      else:
         angle_uom.text = 'rad'

      mds = rqet.SubElement(ds_node, ns['resqml2'] + 'Mds')
      mds.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
      mds.text = rqet.null_xml_text

      mds_values_node = rqet.SubElement(mds, ns['resqml2'] + 'Values')
      mds_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
      mds_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Mds', root = mds_values_node)

      azimuths = rqet.SubElement(ds_node, ns['resqml2'] + 'Azimuths')
      azimuths.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
      azimuths.text = rqet.null_xml_text

      azimuths_values_node = rqet.SubElement(azimuths, ns['resqml2'] + 'Values')
      azimuths_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
      azimuths_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Azimuths', root = azimuths_values_node)

      inclinations = rqet.SubElement(ds_node, ns['resqml2'] + 'Inclinations')
      inclinations.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
      inclinations.text = rqet.null_xml_text

      inclinations_values_node = rqet.SubElement(inclinations, ns['resqml2'] + 'Values')
      inclinations_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
      inclinations_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Inclinations', root = inclinations_values_node)

      interp_root = None
      if self.wellbore_interpretation is not None:
         interp_root = self.wellbore_interpretation.root
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                    bu.uuid_from_string(interp_root.attrib['uuid']),
                                    content_type = 'obj_WellboreInterpretation',
                                    root = ds_node)

      if add_as_part:
         self.model.add_part('obj_DeviationSurveyRepresentation', self.uuid, ds_node)
         if add_relationships:
            # todo: check following relationship
            self.model.create_reciprocal_relationship(ds_node, 'destinationObject', md_datum_root, 'sourceObject')
            if interp_root is not None:
               self.model.create_reciprocal_relationship(ds_node, 'destinationObject', interp_root, 'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(ds_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

      return ds_node

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Create or append to an hdf5 file, writing datasets for the measured depths, azimuths, and inclinations."""

      # NB: array data must all have been set up prior to calling this function
      h5_reg = rwh5.H5Register(self.model)
      h5_reg.register_dataset(self.uuid, 'Mds', self.measured_depths, dtype = float)
      h5_reg.register_dataset(self.uuid, 'Azimuths', self.azimuths, dtype = float)
      h5_reg.register_dataset(self.uuid, 'Inclinations', self.inclinations, dtype = float)
      h5_reg.write(file = file_name, mode = mode)

   def _load_related_datum(self):
      """Return related MdDatum object from XML if present"""

      md_datum_uuid = bu.uuid_from_string(rqet.find_tag(rqet.find_tag(self.root, 'MdDatum'), 'UUID'))
      if md_datum_uuid is not None:
         md_datum_part = 'obj_MdDatum_' + str(md_datum_uuid) + '.xml'
         md_datum = MdDatum(self.model, md_datum_root = self.model.root_for_part(md_datum_part, is_rels = False))
      else:
         md_datum = None
      return md_datum

   def _load_related_wellbore_interp(self):
      """Return related wellbore interp object from XML if present"""

      interp_uuid = rqet.find_nested_tags_text(self.root, ['RepresentedInterpretation', 'UUID'])
      if interp_uuid is None:
         represented_interp = None
      else:
         represented_interp = rqo.WellboreInterpretation(self.model, uuid = interp_uuid)
      return represented_interp


class Trajectory(BaseResqpy):
   """Class for RESQML Wellbore Trajectory Representation (Geometry).

   note:
      resqml allows trajectory to have different crs to the measured depth datum crs;
      however, this code requires the trajectory to be in the same crs as the md datum
   """

   resqml_type = 'WellboreTrajectoryRepresentation'
   well_name = rqo._alias_for_attribute("title")

   def __init__(
         self,
         parent_model,
         trajectory_root = None,  # deprecated
         uuid = None,
         md_datum = None,
         deviation_survey = None,
         data_frame = None,
         grid = None,
         cell_kji0_list = None,
         wellspec_file = None,
         spline_mode = 'cube',
         deviation_survey_file = None,
         survey_file_space_separated = False,
         length_uom = None,
         md_domain = None,
         represented_interp = None,
         well_name = None,
         set_tangent_vectors = False,
         hdf5_source_model = None,
         originator = None,
         extra_metadata = None):
      """Creates a new trajectory object and optionally loads it from xml, deviation survey, pandas dataframe, or ascii file.

      arguments:
         parent_model (model.Model object): the model which the new trajectory belongs to
         trajectory_root (DEPRECATED): use uuid instead; the root node of an xml tree representing the trajectory;
            if not None, the new trajectory object is initialised based on the data in the tree;
            if None, one of the other arguments is used
         md_datum (MdDatum object): the datum that the depths for this trajectory are measured from;
            not used if uuid or trajectory_root is not None
         deviation_survey (DeviationSurvey object, optional): if present and uuid and trajectory_root are None
            then the trajectory is derived from the deviation survey based on minimum curvature
         data_frame (optional): a pandas dataframe with columns 'MD', 'X', 'Y' and 'Z', holding
            the measured depths, and corresponding node locations; ignored if uuid or trajectory_root is not None
         grid (grid.Grid object, optional): only required if initialising from a list of cell indices;
            ignored otherwise
         cell_kji0_list (numpy int array of shape (N, 3)): ordered list of cell indices to be visited by
            the trajectory; ignored if uuid or trajectory_root is not None
         wellspec_file (string, optional): name of an ascii file containing Nexus WELLSPEC data; well_name
            and length_uom arguments must be passed
         spline_mode (string, default 'cube'): one of 'none', 'linear', 'square', or 'cube'; affects spline
            tangent generation; only relevant if initialising from list of cells
         deviation_survey_file (string): filename of an ascii file holding the trajectory
            in a tabular form; ignored if uuid or trajectory_root is not None
         survey_file_space_separated (boolean, default False): if True, deviation survey file is
            space separated; if False, comma separated (csv); ignored unless loading from survey file
         length_uom (string, default 'm'): a resqml length unit of measure applicable to the
            measured depths; should be 'm' or 'ft'
         md_domain (string, optional): if present, must be 'logger' or 'driller'; the source of the original
            deviation data; ignored if uuid or trajectory_root is not None
         represented_interp (wellbore interpretation object, optional): if present, is noted as the wellbore
            interpretation object which this trajectory relates to; ignored if uuid or trajectory_root is not None
         well_name (string, optional): used as citation title
         set_tangent_vectors (boolean, default False): if True and tangent vectors are not loaded then they will
            be computed from the control points
         hdf5_source_model (model.Model, optional): if present this model is used to determine the hdf5 file
            name from which to load the trajectory's array data; if None, the parent_model is used as usual
         originator (str, optional): the name of the person creating the trajectory, defaults to login id;
            ignored if uuid or trajectory_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the trajectory;
            ignored if uuid or trajectory_root is not None

      returns:
         the newly created wellbore trajectory object

      notes:
         if starting from a deviation survey file, there are two routes: create a deviation survey object first,
         using the azimuth and inclination data, then generate a trajectory from that based on minimum curvature;
         or, create a trajectory directly using X, Y, Z data from the deviation survey file (ie. minimum
         curvature or other algorithm already applied externally);
         if not loading from xml, then the crs is set to that used by the measured depth datum, or if that is not
         available then the default crs for the model

      :meta common:
      """

      self.crs_uuid = None
      self.title = well_name
      self.start_md = None
      self.finish_md = None
      self.md_uom = length_uom
      self.md_domain = md_domain
      self.md_datum = md_datum  # md datum is an object in its own right, with a related crs!
      # parametric line geometry elements
      self.knot_count = None
      self.line_kind_index = None
      # 0 for vertical
      # 1 for linear spline
      # 2 for natural cubic spline
      # 3 for cubic spline
      # 4 for z linear cubic spline
      # 5 for minimum-curvature spline   # in practice this is the value actually used  in datasets
      # (-1) for null: no line
      self.measured_depths = None  # known as control point parameters in the parametric line geometry
      self.control_points = None  # xyz array of shape (knot_count, 3)
      self.tangent_vectors = None  # optional xyz tangent vector array, if present has same shape as control points)
      self.deviation_survey = deviation_survey  # optional related deviation survey
      self.wellbore_interpretation = represented_interp
      self.wellbore_feature = None
      self.feature_and_interpretation_to_be_written = False
      # todo: parent intersection for multi-lateral wells
      # todo: witsml trajectory reference (optional)

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = well_name,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = trajectory_root)

      if self.root is not None:
         return

      if set_tangent_vectors and self.knot_count > 1 and self.tangent_vectors is None:
         self.set_tangents()
      elif self.deviation_survey is not None:
         self.compute_from_deviation_survey(method = 'minimum curvature', set_tangent_vectors = set_tangent_vectors)
      elif data_frame is not None:
         self.load_from_data_frame(data_frame,
                                   md_uom = length_uom,
                                   md_datum = md_datum,
                                   set_tangent_vectors = set_tangent_vectors)
      elif cell_kji0_list is not None:
         self.load_from_cell_list(grid, cell_kji0_list, spline_mode, length_uom)
      elif wellspec_file:
         self.load_from_wellspec(grid, wellspec_file, well_name, spline_mode, length_uom)
      elif deviation_survey_file:
         self.load_from_ascii_file(deviation_survey_file,
                                   space_separated_instead_of_csv = survey_file_space_separated,
                                   md_uom = length_uom,
                                   md_datum = md_datum,
                                   title = well_name,
                                   set_tangent_vectors = set_tangent_vectors)
      # todo: create from already loaded deviation_survey node (ie. derive xyz points)

      if self.crs_uuid is None:
         if self.md_datum is not None:
            self.crs_uuid = self.md_datum.crs_uuid
         else:
            self.crs_uuid = self.model.crs_uuid

      if not self.title:
         self.title = 'well trajectory'

      if self.md_datum is None and self.control_points is not None:
         self.md_datum = MdDatum(self.model, crs_uuid = self.crs_uuid, location = self.control_points[0])

   @property
   def crs_root(self):
      """XML node corresponding to self.crs_uuid"""

      return self.model.root_for_uuid(self.crs_uuid)

   def iter_wellbore_frames(self):
      """ Iterable of all WellboreFrames associated with a trajectory

      Yields:
         frame: instance of :class:`resqpy.organize.WellboreFrame`

      :meta common:
      """
      uuids = self.model.uuids(obj_type = "WellboreFrameRepresentation", related_uuid = self.uuid)
      for uuid in uuids:
         yield WellboreFrame(self.model, uuid = uuid)

   def _load_from_xml(self):
      """Loads the trajectory object from an xml node (and associated hdf5 data)."""

      node = self.root
      assert node is not None
      self.start_md = float(rqet.node_text(rqet.find_tag(node, 'StartMd')).strip())
      self.finish_md = float(rqet.node_text(rqet.find_tag(node, 'FinishMd')).strip())
      self.md_uom = rqet.length_units_from_node(rqet.find_tag(node, 'MdUom'))
      self.md_domain = rqet.node_text(rqet.find_tag(node, 'MdDomain'))
      geometry_node = rqet.find_tag(node, 'Geometry')
      self.crs_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(geometry_node, ['LocalCrs', 'UUID']))
      self.knot_count = int(rqet.node_text(rqet.find_tag(geometry_node, 'KnotCount')).strip())
      self.line_kind_index = int(rqet.node_text(rqet.find_tag(geometry_node, 'LineKindIndex')).strip())
      mds_node = rqet.find_tag(geometry_node, 'ControlPointParameters')
      if mds_node is not None:  # not required for vertical or z linear cubic spline
         load_hdf5_array(self, mds_node, 'measured_depths')
      control_points_node = rqet.find_tag(geometry_node, 'ControlPoints')
      load_hdf5_array(self, control_points_node, 'control_points', tag = 'Coordinates')
      tangents_node = rqet.find_tag(geometry_node, 'TangentVectors')
      if tangents_node is not None:
         load_hdf5_array(self, tangents_node, 'tangent_vectors', tag = 'Coordinates')
      relatives_model = self.model  # if hdf5_source_model is None else hdf5_source_model
      # md_datum - separate part, referred to in this tree
      md_datum_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['MdDatum', 'UUID']))
      assert md_datum_uuid is not None, 'failed to fetch uuid of md datum for trajectory'
      md_datum_part = relatives_model.part_for_uuid(md_datum_uuid)
      assert md_datum_part, 'md datum part not found in model'
      self.md_datum = MdDatum(self.model, uuid = relatives_model.uuid_for_part(md_datum_part))
      ds_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['DeviationSurvey', 'UUID']))
      if ds_uuid is not None:  # this will probably not work when relatives model is different from self.model
         ds_part = rqet.part_name_for_object('obj_DeviationSurveyRepresentation_', ds_uuid)
         self.deviation_survey = DeviationSurvey(self.model,
                                                 uuid = relatives_model.uuid_for_part(ds_part, is_rels = False),
                                                 md_datum = self.md_datum)
      interp_uuid = rqet.find_nested_tags_text(node, ['RepresentedInterpretation', 'UUID'])
      if interp_uuid is None:
         self.wellbore_interpretation = None
      else:
         self.wellbore_interpretation = rqo.WellboreInterpretation(self.model, uuid = interp_uuid)

   def compute_from_deviation_survey(self,
                                     survey = None,
                                     method = 'minimum curvature',
                                     md_domain = None,
                                     set_tangent_vectors = True):
      """Derive wellbore trajectory from deviation survey azimuth and inclination data."""

      if survey is None:
         assert self.deviation_survey is not None
         survey = self.deviation_survey
      else:
         self.deviation_survey = survey

      assert method in ['minimum curvature']  # if adding other methods, set line_kind_index appropriately

      self.knot_count = survey.station_count
      assert self.knot_count >= 2  # vertical well could be hamdled by allowing a single station in survey?
      self.line_kind_index = 5  # minimum curvature spline
      self.measured_depths = survey.measured_depths.copy()
      self.md_uom = survey.md_uom
      if not self.title:
         self.title = rqet.find_nested_tags_text(survey.root_node, ['Citation', 'Title'])
      self.start_md = self.measured_depths[0]
      self.finish_md = self.measured_depths[-1]
      if md_domain is not None:
         self.md_domain = md_domain
      self.control_points = np.empty((self.knot_count, 3))
      self.control_points[0, :] = survey.first_station
      for sp in range(1, self.knot_count):
         i1 = survey.inclinations[sp - 1]
         i2 = survey.inclinations[sp]
         az1 = survey.azimuths[sp - 1]
         az2 = survey.azimuths[sp]
         delta_md = survey.measured_depths[sp] - survey.measured_depths[sp - 1]
         assert delta_md > 0.0
         if i1 == i2 and az1 == az2:
            matrix = vec.rotation_3d_matrix((180.0 - i1, -az1, 0.0))  # TODO: check sign of az1
            delta_v = vec.rotate_vector(matrix, np.array([0.0, delta_md, 0.0]))
         else:
            i1 = maths.radians(i1)
            i2 = maths.radians(i2)
            az1 = maths.radians(az1)
            az2 = maths.radians(az2)
            sin_i1 = maths.sin(i1)
            sin_i2 = maths.sin(i2)
            cos_theta = min(max(maths.cos(i2 - i1) - sin_i1 * sin_i2 * (1.0 - maths.cos(az2 - az1)), -1.0), 1.0)
            theta = maths.acos(cos_theta)
            #           theta = maths.acos(sin_i1 * sin_i2 * maths.cos(az2 - az1)  +  (maths.cos(i1) * maths.cos(i2)))
            assert theta != 0.0  # shouldn't happen as covered by if clause above
            half_rf = maths.tan(0.5 * theta) / theta
            delta_y = delta_md * half_rf * ((sin_i1 * maths.cos(az1)) + (sin_i2 * maths.cos(az2)))
            delta_x = delta_md * half_rf * ((sin_i1 * maths.sin(az1)) + (sin_i2 * maths.sin(az2)))
            delta_z = delta_md * half_rf * (maths.cos(i1) + maths.cos(i2))
            delta_v = np.array((delta_x, delta_y, delta_z))
         self.control_points[sp] = self.control_points[sp - 1] + delta_v
      self.tangent_vectors = None
      if set_tangent_vectors:
         self.set_tangents()
      self.md_datum = survey.md_datum

   def load_from_data_frame(
         self,
         data_frame,
         md_col = 'MD',
         x_col = 'X',
         y_col = 'Y',
         z_col = 'Z',
         md_uom = 'm',
         md_domain = None,
         md_datum = None,  # MdDatum object
         title = None,
         set_tangent_vectors = True):
      """Load MD and control points (xyz) data from a pandas data frame."""

      try:
         for col in [md_col, x_col, y_col, z_col]:
            assert col in data_frame.columns
         self.knot_count = len(data_frame)
         assert self.knot_count >= 2  # vertical well could be hamdled by allowing a single station in survey?
         self.line_kind_index = 5  # assume minimum curvature spline
         #         self.md_uom = bwam.p_length_unit(md_uom)
         self.md_uom = bwam.rq_length_unit(md_uom)
         start = data_frame.iloc[0]
         finish = data_frame.iloc[-1]
         if title:
            self.title = title
         self.start_md = start[md_col]
         self.finish_md = finish[md_col]
         if md_domain is not None:
            self.md_domain = md_domain
         self.measured_depths = np.empty(self.knot_count)
         self.measured_depths[:] = data_frame[md_col]
         self.control_points = np.empty((self.knot_count, 3))
         self.control_points[:, 0] = data_frame[x_col]
         self.control_points[:, 1] = data_frame[y_col]
         self.control_points[:, 2] = data_frame[z_col]
         self.tangent_vectors = None
         if set_tangent_vectors:
            self.set_tangents()
         self.md_datum = md_datum
      except Exception:
         log.exception('failed to load trajectory object from data frame')

   def load_from_cell_list(self, grid, cell_kji0_list, spline_mode = 'cube', md_uom = 'm'):
      """Loads the trajectory object based on the centre points of a list of cells."""

      assert grid is not None, 'grid argument missing for trajectory initislisation from cell list'
      assert cell_kji0_list.ndim == 2 and cell_kji0_list.shape[1] == 3
      assert spline_mode in ['none', 'linear', 'square', 'cube']

      cell_centres = grid.centre_point_list(cell_kji0_list)

      knot_count = len(cell_kji0_list) + 2
      self.line_kind_index = 5  # 5 means minimum curvature spline; todo: set to cubic spline value?
      self.md_uom = bwam.rq_length_unit(md_uom)
      self.start_md = 0.0
      points = np.empty((knot_count, 3))
      points[1:-1] = cell_centres
      points[0] = points[1]
      points[0, 2] = 0.0
      points[-1] = points[-2]
      points[-1, 2] *= 1.05
      if spline_mode == 'none':
         self.knot_count = knot_count
         self.control_points = points
      else:
         self.control_points = rql.spline(points, tangent_weight = spline_mode, min_subdivisions = 3)
         self.knot_count = len(self.control_points)
      self.set_measured_depths()

   def load_from_wellspec(self, grid, wellspec_file, well_name, spline_mode = 'cube', md_uom = 'm'):

      col_list = ['IW', 'JW', 'L']
      wellspec_dict = wsk.load_wellspecs(wellspec_file, well = well_name, column_list = col_list)

      assert len(wellspec_dict) == 1, 'no wellspec data found in file ' + wellspec_file + ' for well ' + well_name

      df = wellspec_dict[well_name]
      assert len(df) > 0, 'no rows of perforation data found in wellspec for well ' + well_name

      cell_kji0_list = np.empty((len(df), 3), dtype = int)
      cell_kji0_list[:, 0] = df['L']
      cell_kji0_list[:, 1] = df['JW']
      cell_kji0_list[:, 2] = df['IW']

      self.load_from_cell_list(grid, cell_kji0_list, spline_mode, md_uom)

   def load_from_ascii_file(self,
                            trajectory_file,
                            comment_character = '#',
                            space_separated_instead_of_csv = False,
                            md_col = 'MD',
                            x_col = 'X',
                            y_col = 'Y',
                            z_col = 'Z',
                            md_uom = 'm',
                            md_domain = None,
                            md_datum = None,
                            well_col = None,
                            title = None,
                            set_tangent_vectors = True):
      """Loads the trajectory object from an ascii file with columns for MD, X, Y & Z (and optionally WELL)."""

      if not title and not self.title:
         self.title = 'well trajectory'

      try:
         df = pd.read_csv(trajectory_file,
                          comment = comment_character,
                          delim_whitespace = space_separated_instead_of_csv)
         if df is None:
            raise Exception
      except Exception:
         log.error('failed to read ascii deviation survey file ' + str(trajectory_file))
         raise
      if well_col and well_col not in df.columns:
         log.warning('well column ' + str(well_col) + ' not found in ascii trajectory file ' + str(trajectory_file))
         well_col = None
      if well_col is None:
         for col in df.columns:
            if str(col).upper().startswith('WELL'):
               well_col = col
               break
      if title:  # filter data frame by well name
         if well_col:
            df = df[df[well_col] == title]
            if len(df) == 0:
               log.error('no data found for well ' + str(title) + ' in file ' + str(trajectory_file))
      elif well_col is not None:
         if len(set(df[well_col])) > 1:
            raise Exception(
               'attempt to set trajectory for unidentified well from ascii file holding data for multiple wells')
      self.load_from_data_frame(df,
                                md_col = md_col,
                                x_col = x_col,
                                y_col = y_col,
                                z_col = z_col,
                                md_uom = md_uom,
                                md_domain = md_domain,
                                md_datum = md_datum,
                                title = title,
                                set_tangent_vectors = set_tangent_vectors)

   def set_tangents(self, force = False, write_hdf5 = False, weight = 'cube'):
      """Calculates tangent vectors based on control points.

      arguments:
         force (boolean, default False): if False and tangent vectors already exist then the existing ones are used;
            if True or no tangents vectors exist then they are computed
         write_hdf5 (boolean, default False): if True and new tangent vectors are computed then the array is also written
            directly to the hdf5 file
         weight (string, default 'linear'): one of 'linear', 'square', 'cube'; if linear, each tangent is the mean of the
            direction vectors of the two trjectory segments which meet at the knot; the square and cube options give
            increased weight to the direction vector of shorter segments (usually better)

      returns:
         numpy float array of shape (knot_count, 3) being the tangents in xyz, 'pointing' in the direction of increased
         knot index; the tangents are also stored as an attribute of the object

      note:
         the write_hdf5() method writes all the array data for the trajectory, including the tangent vectors; only set
         the write_hdf5 argument to this method to True if the other arrays for the trajectory already exist in the hdf5 file
      """

      if self.tangent_vectors is not None and not force:
         return self.tangent_vectors
      assert self.knot_count is not None and self.knot_count >= 2
      assert self.control_points is not None and len(self.control_points) == self.knot_count

      self.tangent_vectors = rql.tangents(self.control_points, weight = weight)

      if write_hdf5:
         h5_reg = rwh5.H5Register(self.model)
         h5_reg.register_dataset(self.uuid, 'tangentVectors', self.tangent_vectors)
         h5_reg.write(file = self.model.h5_filename(), mode = 'a')

      return self.tangent_vectors

   def dataframe(self, md_col = 'MD', x_col = 'X', y_col = 'Y', z_col = 'Z'):
      """Returns a pandas data frame containing MD and control points (xyz) data.

      note:
         set md_col to None for a dataframe containing only X, Y & Z data

      :meta common:
      """

      if md_col:
         column_list = [md_col, x_col, y_col, z_col]
      else:
         column_list = [x_col, y_col, z_col]

      data_frame = pd.DataFrame(columns = column_list)
      if md_col:
         data_frame[md_col] = self.measured_depths
      data_frame[x_col] = self.control_points[:, 0]
      data_frame[y_col] = self.control_points[:, 1]
      data_frame[z_col] = self.control_points[:, 2]
      return data_frame

   def write_to_ascii_file(self,
                           trajectory_file,
                           mode = 'w',
                           space_separated_instead_of_csv = False,
                           md_col = 'MD',
                           x_col = 'X',
                           y_col = 'Y',
                           z_col = 'Z'):
      """Writes trajectory to an ascii file.

      note:
         set md_col to None for a dataframe containing only X, Y & Z data
      """

      df = self.dataframe(md_col = md_col, x_col = x_col, y_col = y_col, z_col = z_col)
      sep = ' ' if space_separated_instead_of_csv else ','
      df.to_csv(trajectory_file, sep = sep, index = False, mode = mode)

   def xyz_for_md(self, md):
      """Returns an xyz triplet corresponding to the given measured depth; uses simple linear interpolation between knots.

      args:
         md (float): measured depth for which xyz location is required; units must be those of self.md_uom

      returns:
         triple float being x, y, z coordinates of point on trajectory corresponding to given measured depth

      note:
         the algorithm uses a simple linear interpolation between neighbouring knots (control points) on the trajectory;
         if the measured depth is less than zero or greater than the finish md, a single None is returned; if the md is
         less than the start md then a linear interpolation between the md datum location and the first knot is returned

      :meta common:
      """

      def interpolate(p1, p2, f):
         return f * p2 + (1.0 - f) * p1

      def search(md, i1, i2):
         if i2 - i1 <= 1:
            if md == self.measured_depths[i1]:
               return self.control_points[i1]
            return interpolate(self.control_points[i1], self.control_points[i1 + 1], (md - self.measured_depths[i1]) /
                               (self.measured_depths[i1 + 1] - self.measured_depths[i1]))
         im = i1 + (i2 - i1) // 2
         if self.measured_depths[im] >= md:
            return search(md, i1, im)
         return search(md, im, i2)

      if md < 0.0 or md > self.finish_md or md > self.measured_depths[-1]:
         return None
      if md <= self.start_md:
         if self.start_md == 0.0:
            return self.md_datum.location
         return interpolate(np.array(self.md_datum.location), self.control_points[0], md / self.start_md)
      return search(md, 0, self.knot_count - 1)

   def splined_trajectory(self,
                          well_name,
                          min_subdivisions = 1,
                          max_segment_length = None,
                          max_degrees_per_knot = 5.0,
                          use_tangents_if_present = True,
                          store_tangents_if_calculated = True):
      """Creates and returns a new Trajectory derived as a cubic spline of this trajectory.

      arguments:
         well_name (string): the name to use as the citation title for the new trajectory
         min_subdivisions (+ve integer, default 1): the minimum number of segments in the trajectory for each
            segment in this trajectory
         max_segment_length (float, optional): if present, each segment of this trajectory is subdivided so
            that the naive subdivided length is not greater than the specified length
         max_degrees_per_knot (float, default 5.0): the maximum number of degrees
         use_tangents_if_present (boolean, default False): if True, any tangent vectors in this trajectory
            are used during splining
         store_tangents_if_calculated (boolean, default True): if True any tangents calculated by the method
            are stored in the object (causing any previous tangents to be discarded); however, the new tangents
            are not written to the hdf5 file by this method

      returns:
         Trajectory object with control points lying on a cubic spline of the points of this trajectory

      notes:
         this method is typically used to smoothe an artificial or simulator trajectory;
         measured depths are re-calculated and will differ from those in this trajectory;
         unexpected behaviour may occur if the z units are different from the xy units in the crs;
         if tangent vectors for neighbouring points in this trajectory are pointing in opposite directions,
         the resulting spline is likely to be bad;
         the max_segment_length is applied when deciding how many subdivisions to make for a segment in this
         trajectory, based on the stright line segment length; segments in the resulting spline may exceed this
         length;
         similarly max_degrees_per_knot assumes a simply bend between neighbouring knots; if the position of the
         control points results in a loop, the value may be exceeded in the spline;
         the hdf5 data for the splined trajectory is not written by this method, neither is the xml created;
         no interpretation object is created by this method
         NB: direction of tangent vectors affects results, set use_tangents_if_present = False to
         ensure locally calculated tangent vectors are used
      """

      assert self.knot_count > 1 and self.control_points is not None
      assert min_subdivisions >= 1
      assert max_segment_length is None or max_segment_length > 0.0
      assert max_degrees_per_knot is None or max_degrees_per_knot > 0.0
      if not well_name:
         well_name = self.title

      tangent_vectors = self.tangent_vectors
      if tangent_vectors is None or not use_tangents_if_present:
         tangent_vectors = rql.tangents(self.control_points, weight = 'square')
         if store_tangents_if_calculated:
            self.tangent_vectors = tangent_vectors

      spline_traj = Trajectory(self.model,
                               well_name = well_name,
                               md_datum = self.md_datum,
                               length_uom = self.md_uom,
                               md_domain = self.md_domain)
      spline_traj.line_kind_index = self.line_kind_index  # not sure how we should really be setting this
      spline_traj.crs_uuid = self.crs_uuid
      spline_traj.start_md = self.start_md
      spline_traj.deviation_survey = self.deviation_survey

      spline_traj.control_points = rql.spline(self.control_points,
                                              tangent_vectors = tangent_vectors,
                                              min_subdivisions = min_subdivisions,
                                              max_segment_length = max_segment_length,
                                              max_degrees_per_knot = max_degrees_per_knot)
      spline_traj.knot_count = len(spline_traj.control_points)

      spline_traj.set_measured_depths()

      return spline_traj

   def set_measured_depths(self):
      """Sets the measured depths from the start_md value and the control points."""

      self.measured_depths = np.empty(self.knot_count)
      self.measured_depths[0] = self.start_md
      for sk in range(1, self.knot_count):
         self.measured_depths[sk] = (self.measured_depths[sk - 1] +
                                     vec.naive_length(self.control_points[sk] - self.control_points[sk - 1]))
      self.finish_md = self.measured_depths[-1]

      return self.measured_depths

   def create_feature_and_interpretation(self):
      """Instantiate new empty WellboreFeature and WellboreInterpretation objects, if a wellboreinterpretation does not already exist.
      
      Uses the trajectory citation title as the well name
      """

      log.debug("Creating a new WellboreInterpretation..")
      log.debug(f"WellboreFeature exists: {self.wellbore_feature is not None}")
      log.debug(f"WellboreInterpretation exists: {self.wellbore_interpretation is not None}")

      if self.wellbore_interpretation is None:
         log.info(f"Creating WellboreInterpretation and WellboreFeature with name {self.title}")
         self.wellbore_feature = rqo.WellboreFeature(parent_model = self.model, feature_name = self.title)
         self.wellbore_interpretation = rqo.WellboreInterpretation(parent_model = self.model,
                                                                   wellbore_feature = self.wellbore_feature)
         self.feature_and_interpretation_to_be_written = True
      else:
         raise ValueError("Cannot add WellboreFeature, trajectory already has an associated WellboreInterpretation")

   def create_xml(self,
                  ext_uuid = None,
                  wbt_uuid = None,
                  md_datum_root = None,
                  md_datum_xyz = None,
                  add_as_part = True,
                  add_relationships = True,
                  title = None,
                  originator = None):
      """Create a wellbore trajectory representation node from a Trajectory object, optionally add as part.

         notes:
            measured depth datum xml node must be in place before calling this function;
            branching well structures (multi-laterals) are supported by the resqml standard but not yet by
            this code;
            optional witsml trajectory reference not yet supported here

      :meta common:
      """

      if title:
         self.title = title
      if not self.title:
         self.title = 'wellbore trajectory'

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      if self.feature_and_interpretation_to_be_written:
         if self.wellbore_interpretation is None:
            self.create_feature_and_interpretation()
         if self.wellbore_feature is not None:
            self.wellbore_feature.create_xml(add_as_part = add_as_part, originator = originator)
         self.wellbore_interpretation.create_xml(add_as_part = add_as_part,
                                                 add_relationships = add_relationships,
                                                 originator = originator)

      if md_datum_root is None:
         if self.md_datum is None:
            assert md_datum_xyz is not None
            self.md_datum = MdDatum(self.model, location = md_datum_xyz)
         if self.md_datum.root is None:
            md_datum_root = self.md_datum.create_xml()
         else:
            md_datum_root = self.md_datum.root

      wbt_node = super().create_xml(originator = originator, add_as_part = False)

      start_node = rqet.SubElement(wbt_node, ns['resqml2'] + 'StartMd')
      start_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
      start_node.text = str(self.start_md)

      finish_node = rqet.SubElement(wbt_node, ns['resqml2'] + 'FinishMd')
      finish_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
      finish_node.text = str(self.finish_md)

      md_uom = rqet.SubElement(wbt_node, ns['resqml2'] + 'MdUom')
      md_uom.set(ns['xsi'] + 'type', ns['eml'] + 'LengthUom')
      md_uom.text = bwam.rq_length_unit(self.md_uom)

      self.model.create_md_datum_reference(self.md_datum.root, root = wbt_node)

      if self.line_kind_index != 0:  # 0 means vertical well, which doesn't need a geometry

         # todo: check geometry elements for parametric curve flavours other than minimum curvature

         geom = rqet.SubElement(wbt_node, ns['resqml2'] + 'Geometry')
         geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'ParametricLineGeometry')
         geom.text = '\n'

         # note: resqml standard allows trajectory to be in different crs to md datum
         #       however, this module often uses the md datum crs, if the trajectory has been imported
         if self.crs_uuid is None:
            self.crs_uuid = self.md_datum.crs_uuid
         assert self.crs_uuid is not None
         self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

         kc_node = rqet.SubElement(geom, ns['resqml2'] + 'KnotCount')
         kc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         kc_node.text = str(self.knot_count)

         lki_node = rqet.SubElement(geom, ns['resqml2'] + 'LineKindIndex')
         lki_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
         lki_node.text = str(self.line_kind_index)

         cpp_node = rqet.SubElement(geom, ns['resqml2'] + 'ControlPointParameters')
         cpp_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
         cpp_node.text = rqet.null_xml_text

         cpp_values_node = rqet.SubElement(cpp_node, ns['resqml2'] + 'Values')
         cpp_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         cpp_values_node.text = rqet.null_xml_text

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'controlPointParameters', root = cpp_values_node)

         cp_node = rqet.SubElement(geom, ns['resqml2'] + 'ControlPoints')
         cp_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
         cp_node.text = rqet.null_xml_text

         cp_coords_node = rqet.SubElement(cp_node, ns['resqml2'] + 'Coordinates')
         cp_coords_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         cp_coords_node.text = rqet.null_xml_text

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'controlPoints', root = cp_coords_node)

         if self.tangent_vectors is not None:

            tv_node = rqet.SubElement(geom, ns['resqml2'] + 'TangentVectors')
            tv_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
            tv_node.text = rqet.null_xml_text

            tv_coords_node = rqet.SubElement(tv_node, ns['resqml2'] + 'Coordinates')
            tv_coords_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            tv_coords_node.text = rqet.null_xml_text

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'tangentVectors', root = tv_coords_node)

      if self.md_domain:
         domain_node = rqet.SubElement(wbt_node, ns['resqml2'] + 'MdDomain')
         domain_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'MdDomain')
         domain_node.text = self.md_domain

      if self.deviation_survey is not None:
         ds_root = self.deviation_survey.root_node
         self.model.create_ref_node('DeviationSurvey',
                                    rqet.find_tag(rqet.find_tag(ds_root, 'Citation'), 'Title').text,
                                    bu.uuid_from_string(ds_root.attrib['uuid']),
                                    content_type = 'obj_DeviationSurveyRepresentation',
                                    root = wbt_node)

      interp_root = None
      if self.wellbore_interpretation is not None:
         interp_root = self.wellbore_interpretation.root
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                    bu.uuid_from_string(interp_root.attrib['uuid']),
                                    content_type = 'obj_WellboreInterpretation',
                                    root = wbt_node)

      if add_as_part:
         self.model.add_part('obj_WellboreTrajectoryRepresentation', self.uuid, wbt_node)
         if add_relationships:
            crs_root = self.crs_root
            self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', crs_root, 'sourceObject')
            self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', self.md_datum.root, 'sourceObject')
            if self.deviation_survey is not None:
               self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', self.deviation_survey.root_node,
                                                         'sourceObject')
            if interp_root is not None:
               self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', interp_root, 'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(wbt_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

      return wbt_node

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Create or append to an hdf5 file, writing datasets for the measured depths, control points and tangent vectors.

      :meta common:
      """

      # NB: array data must all have been set up prior to calling this function
      if self.uuid is None:
         self.uuid = bu.new_uuid()

      h5_reg = rwh5.H5Register(self.model)
      h5_reg.register_dataset(self.uuid, 'controlPointParameters', self.measured_depths)
      h5_reg.register_dataset(self.uuid, 'controlPoints', self.control_points)
      if self.tangent_vectors is not None:
         h5_reg.register_dataset(self.uuid, 'tangentVectors', self.tangent_vectors)
      h5_reg.write(file = file_name, mode = mode)

   def __eq__(self, other):
      """Implements equals operator. Compares class type and uuid"""

      # TODO: more detailed equality comparison
      other_uuid = getattr(other, "uuid", None)
      return isinstance(other, self.__class__) and bu.matching_uuids(self.uuid, other_uuid)


class WellboreFrame(BaseResqpy):
   """Class for RESQML WellboreFrameRepresentation objects (supporting well log Properties)

   RESQML documentation:

      Representation of a wellbore that is organized along a wellbore trajectory by its MD values.
      RESQML uses MD values to associate properties on points and to organize association of
      properties on intervals between MD points.

   Roughly equivalent to a Techlog "dataset" object with a given depth reference.

   The `logs` attribute is a :class:`resqpy.property.WellLogCollection` of all logs in the frame.

   """

   resqml_type = 'WellboreFrameRepresentation'

   def __init__(self,
                parent_model,
                frame_root = None,
                uuid = None,
                trajectory = None,
                mds = None,
                represented_interp = None,
                title = None,
                originator = None,
                extra_metadata = None):
      """Creates a new wellbore frame object and optionally loads it from xml or list of measured depths.

      arguments:
         parent_model (model.Model object): the model which the new wellbore frame belongs to
         frame_root (optional): DEPRECATED. the root node of an xml tree representing the wellbore frame;
            if not None, the new wellbore frame object is initialised based on the data in the tree;
            if None, an empty wellbore frame object is returned
         trajectory (Trajectory object, optional): the trajectory of the well; required if loading from
            list of measured depths
         mds (optional numpy 1D array, tuple or list of floats): ordered list of measured depths which
            will constitute the frame; ignored if frame_root is not None
         represented_interp (wellbore interpretation object, optional): if present, is noted as the wellbore
            interpretation object which this frame relates to; ignored if frame_root is not None
         title (str, optional): the citation title to use for a new wellbore frame;
            ignored if uuid or frame_root is not None
         originator (str, optional): the name of the person creating the wellbore frame, defaults to login id;
            ignored if uuid or frame_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore frame;
            ignored if uuid or frame_root is not None

      returns:
         the newly created wellbore frame object

      note:
         if initialising from a list of measured depths, the wellbore trajectory object must already exist
      """

      #: Associated wellbore trajectory, an instance of :class:`resqpy.well.Trajectory`.
      self.trajectory = trajectory
      self.trajectory_uuid = None if trajectory is None else trajectory.uuid

      #: Instance of :class:`resqpy.organize.WellboreInterpretation`
      self.wellbore_interpretation = represented_interp
      self.wellbore_feature = None
      self.feature_and_interpretation_to_be_written = False

      #: number of measured depth nodes, each being an entry or exit point of trajectory with a cell
      self.node_count = None

      #: node_count measured depths (in same units and datum as trajectory) of cell entry and/or exit points
      self.node_mds = None

      #: All logs associated with the wellbore frame; an instance of :class:`resqpy.property.WellLogCollection`
      self.logs = None

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = frame_root)

      if self.root is None and trajectory is not None and mds is not None and len(mds) > 1:
         self.node_count = len(mds)
         self.node_mds = np.array(mds)
         assert self.node_mds is not None and self.node_mds.ndim == 1

      # UUID needs to have been created before LogCollection can be made
      # TODO: Figure out when this should be created, and how it is kept in sync when new logs are created
      self.logs = rqp.WellLogCollection(frame = self)

   def _load_from_xml(self):
      """Loads the wellbore frame object from an xml node (and associated hdf5 data)."""

      # NB: node is the root level xml node, not a node in the md list!

      node = self.root
      assert node is not None

      trajectory_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['Trajectory', 'UUID']))
      assert trajectory_uuid is not None, 'wellbore frame trajectory reference not found in xml'
      if self.trajectory is None:
         self.trajectory = Trajectory(self.model, uuid = trajectory_uuid)
      else:
         assert bu.matching_uuids(self.trajectory.uuid, trajectory_uuid), 'wellbore frame trajectory uuid mismatch'

      self.node_count = rqet.find_tag_int(node, 'NodeCount')
      assert self.node_count is not None, 'node count not found in xml for wellbore frame'
      assert self.node_count > 1, 'fewer than 2 nodes for wellbore frame'

      mds_node = rqet.find_tag(node, 'NodeMd')
      assert mds_node is not None, 'wellbore frame measured depths hdf5 reference not found in xml'
      load_hdf5_array(self, mds_node, 'node_mds')

      assert self.node_mds is not None and self.node_mds.ndim == 1 and self.node_mds.size == self.node_count

      interp_uuid = rqet.find_nested_tags_text(node, ['RepresentedInterpretation', 'UUID'])
      if interp_uuid is None:
         self.wellbore_interpretation = None
      else:
         self.wellbore_interpretation = rqo.WellboreInterpretation(self.model, uuid = interp_uuid)

      # Create well log collection of all log data
      self.logs = rqp.WellLogCollection(frame = self)

   def extract_crs_root(self):
      """Returns the xml root node of the coordinate reference system used by the related trajectory."""

      if self.trajectory is None:
         return None
      return self.trajectory.crs_root

   def create_feature_and_interpretation(self):
      """Instantiate new empty WellboreFeature and WellboreInterpretation objects, if a wellboreinterpretation does not already exist.
      
      Uses the wellboreframe citation title as the well name
      """
      if self.wellbore_interpretation is not None:
         log.info(f"Creating WellboreInterpretation and WellboreFeature with name {self.title}")
         self.wellbore_feature = rqo.WellboreFeature(parent_model = self.model, feature_name = self.title)
         self.wellbore_interpretation = rqo.WellboreInterpretation(parent_model = self.model,
                                                                   wellbore_feature = self.wellbore_feature)
         self.feature_and_interpretation_to_be_written = True
      else:
         log.info("WellboreInterpretation already exists")

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Create or append to an hdf5 file, writing datasets for the measured depths."""

      # NB: array data must have been set up prior to calling this function

      if self.uuid is None:
         self.uuid = bu.new_uuid()

      h5_reg = rwh5.H5Register(self.model)
      h5_reg.register_dataset(self.uuid, 'NodeMd', self.node_mds)
      h5_reg.write(file = file_name, mode = mode)

   def create_xml(self, ext_uuid = None, add_as_part = True, add_relationships = True, title = None, originator = None):
      """Create a wellbore frame representation node from this WellboreFrame object, optionally add as part.

      note:
         trajectory xml node must be in place before calling this function
      """

      assert self.trajectory is not None, 'trajectory object missing'
      assert self.trajectory.root is not None, 'trajectory xml not established'

      if self.feature_and_interpretation_to_be_written:
         if self.wellbore_interpretation is None:
            self.create_feature_and_interpretation()
         if self.wellbore_feature is not None:
            self.wellbore_feature.create_xml(add_as_part = add_as_part, originator = originator)
         self.wellbore_interpretation.create_xml(add_as_part = add_as_part,
                                                 add_relationships = add_relationships,
                                                 originator = originator)

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      if title:
         self.title = title
      if not self.title:
         self.title = 'wellbore frame'

      wf_node = super().create_xml(originator = originator, add_as_part = False)

      # wellbore frame elements

      nc_node = rqet.SubElement(wf_node, ns['resqml2'] + 'NodeCount')
      nc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      nc_node.text = str(self.node_count)

      mds_node = rqet.SubElement(wf_node, ns['resqml2'] + 'NodeMd')
      mds_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
      mds_node.text = rqet.null_xml_text

      mds_values_node = rqet.SubElement(mds_node, ns['resqml2'] + 'Values')
      mds_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
      mds_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'NodeMd', root = mds_values_node)

      traj_root = self.trajectory.root
      self.model.create_ref_node('Trajectory',
                                 rqet.find_nested_tags_text(traj_root, ['Citation', 'Title']),
                                 bu.uuid_from_string(traj_root.attrib['uuid']),
                                 content_type = 'obj_WellboreTrajectoryRepresentation',
                                 root = wf_node)

      if self.wellbore_interpretation is not None:
         interp_root = self.wellbore_interpretation.root
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                    bu.uuid_from_string(interp_root.attrib['uuid']),
                                    content_type = 'obj_WellboreInterpretation',
                                    root = wf_node)

      if add_as_part:
         self.model.add_part('obj_WellboreFrameRepresentation', self.uuid, wf_node)
         if add_relationships:
            self.model.create_reciprocal_relationship(wf_node, 'destinationObject', self.trajectory.root,
                                                      'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(wf_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')
            if self.wellbore_interpretation is not None:
               interp_root = self.wellbore_interpretation.root
               self.model.create_reciprocal_relationship(wf_node, 'destinationObject', interp_root, 'sourceObject')

      return wf_node


class BlockedWell(BaseResqpy):
   """Class for RESQML Blocked Wellbore Representation (Wells), ie cells visited by wellbore.

   RESQML documentation:

      The information that allows you to locate, on one or several grids (existing or planned),
      the intersection of volume (cells) and surface (faces) elements with a wellbore trajectory
      (existing or planned).

   note:
      measured depth data must be in same crs as those for the related trajectory
   """

   resqml_type = 'BlockedWellboreRepresentation'
   well_name = rqo._alias_for_attribute("title")

   def __init__(self,
                parent_model,
                blocked_well_root = None,
                uuid = None,
                grid = None,
                trajectory = None,
                wellspec_file = None,
                cellio_file = None,
                column_ji0 = None,
                well_name = None,
                check_grid_name = False,
                use_face_centres = False,
                represented_interp = None,
                originator = None,
                extra_metadata = None,
                add_wellspec_properties = False):
      """Creates a new blocked well object and optionally loads it from xml, or trajectory, or Nexus wellspec file.

      arguments:
         parent_model (model.Model object): the model which the new blocked well belongs to
         blocked_well_root (DEPRECATED): the root node of an xml tree representing the blocked well;
            if not None, the new blocked well object is initialised based on the data in the tree;
            if None, the other arguments are used
         grid (optional, grid.Grid object): required if intialising from a trajectory or wellspec file;
            not used if blocked_well_root is not None
         trajectory (optional, Trajectory object): the trajectory of the well, to be intersected with the grid;
            not used if blocked_well_root is not None
         wellspec_file (optional, string): filename of an ascii file holding the Nexus wellspec data;
            ignored if blocked_well_root is not None or trajectory is not None
         cellio_file (optional, string): filename of an ascii file holding the RMS exported blocked well data;
            ignored if blocked_well_root is not None or trajectory is not None or wellspec_file is not None
         column_ji0 (optional, pair of ints): column indices (j0, i0) for a 'vertical' well; ignored if
            blocked_well_root is not None or trajectory is not None or wellspec_file is not None or
            cellio_file is not None
         well_name (string): the well name as given in the wellspec or cellio file; required if loading from
            one of those files; or the name to be used as citation title for a column well
         check_grid_name (boolean, default False): if True, the GRID column of the wellspec data will be checked
            for a match with the citation title of the grid object; perforations for other grids will be skipped;
            if False, all wellspec data is assumed to relate to the grid; only relevant when loading from wellspec
         use_face_centres (boolean, default False): if True, cell face centre points are used for the entry and
            exit points when constructing the simulation trajectory; if False and ANGLA & ANGLV data are available
            then entry and exit points are constructed based on a straight line at those angles passing through
            the centre of the cell; only relevant when loading from wellspec
         represented_interp (wellbore interpretation object, optional): if present, is noted as the wellbore
            interpretation object which this frame relates to; ignored if blocked_well_root is not None
         originator (str, optional): the name of the person creating the blocked well, defaults to login id;
            ignored if uuid or blocked_well_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the blocked well;
            ignored if uuid or blocked_well_root is not None
         add_wellspec_properties (boolean or list of str, default False): if not False, and initialising from
            a wellspec file, the blocked well has its hdf5 data written and xml created and properties are
            fully created; if a list is provided the elements must be numerical wellspec column names;
            if True, all numerical columns other than the cell indices are added as properties

      returns:
         the newly created blocked well object

      notes:
         if starting from a wellspec file or column indices, a 'simulation' trajectory and md datum objects are
         constructed to go with the blocked well;
         column wells might not be truly vertical - the trajectory will consist of linear segments joining the
         centres of the k faces in the column;
         optional RESQML attributes are not handled by this code (WITSML log reference, interval stratigraphic units,
         cell fluid phase units);
         mysterious RESQML WellboreFrameIndexableElements is not used in any other RESQML classes and is therefore
         not used here

      :meta common:
      """

      self.trajectory = trajectory
      self.trajectory_to_be_written = False
      self.feature_to_be_written = False
      self.interpretation_to_be_written = False
      self.node_count = None  # number of measured depth nodes, each being an entry or exit point of trajectory with a cell
      self.node_mds = None  # node_count measured depths (in same units and datum as trajectory) of cell entry and/or exit points
      self.cell_count = None  # number of blocked intervals (<= node_count - 1)
      self.cell_indices = None  # cell_count natural cell indices, paired with non-null grid_indices
      self.grid_indices = None  # node_count-1 indices into grid list for each interval in node_mds; -1 for unblocked interval
      self.face_pair_indices = None  # entry, exit face per cell indices, -1 for Target Depth termination within a cell
      self.grid_list = [
      ]  # list of grid objects indexed by grid_indices; for now only 1 grid supported unless loading from xml
      self.wellbore_interpretation = None
      self.wellbore_feature = None

      #: All logs associated with the blockedwellbore; an instance of :class:`resqpy.property.WellIntervalPropertyCollection`
      self.logs = None
      self.cellind_null = None
      self.gridind_null = None
      self.facepair_null = None

      # face_index_map maps from (axis, p01) to face index value in range 0..5
      # this is the default as indicated on page 139 (but not p. 180) of the RESQML Usage Gude v2.0.1
      # also assumes K is generally increasing downwards
      # see DevOps backlog item 269001 discussion for more information
      #     self.face_index_map = np.array([[0, 1], [4, 2], [5, 3]], dtype = int)
      self.face_index_map = np.array([[0, 1], [2, 4], [5, 3]], dtype = int)  # order: top, base, J-, I+, J+, I-
      # and the inverse, maps from 0..5 to (axis, p01)
      #     self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 1], [2, 1], [1, 0], [2, 0]], dtype = int)
      self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [1, 1], [2, 0]], dtype = int)
      # note: the rework_face_pairs() method, below, overwrites the face indices based on I, J cell indices

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = well_name,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = blocked_well_root)

      if self.root is None:
         self.wellbore_interpretation = represented_interp
         if grid is None:
            grid = self.model.grid()
         if self.trajectory is not None:
            self.compute_from_trajectory(self.trajectory, grid)
         elif wellspec_file is not None:
            okay = self.derive_from_wellspec(wellspec_file,
                                             well_name,
                                             grid,
                                             check_grid_name = check_grid_name,
                                             use_face_centres = use_face_centres,
                                             add_properties = add_wellspec_properties)
         elif cellio_file is not None:
            okay = self.import_from_rms_cellio(cellio_file, well_name, grid)
            if not okay:
               self.node_count = 0
         elif column_ji0 is not None:
            okay = self.set_for_column(well_name, grid, column_ji0)
         self.gridind_null = -1
         self.facepair_null = -1
         self.cellind_null = -1
      # else an empty object is returned

   def _load_from_xml(self):
      """Loads the blocked wellbore object from an xml node (and associated hdf5 data)."""

      node = self.root
      assert node is not None

      trajectory_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['Trajectory', 'UUID']))
      assert trajectory_uuid is not None, 'blocked well trajectory reference not found in xml'
      if self.trajectory is None:
         self.trajectory = Trajectory(self.model, uuid = trajectory_uuid)
      else:
         assert bu.matching_uuids(self.trajectory.uuid, trajectory_uuid), 'blocked well trajectory uuid mismatch'

      self.node_count = rqet.find_tag_int(node, 'NodeCount')
      assert self.node_count is not None and self.node_count >= 2, 'problem with blocked well node count'

      mds_node = rqet.find_tag(node, 'NodeMd')
      assert mds_node is not None, 'blocked well node measured depths hdf5 reference not found in xml'
      load_hdf5_array(self, mds_node, 'node_mds')

      # Statement below has no effect, is this a bug?
      self.node_mds is not None and self.node_mds.ndim == 1 and self.node_mds.size == self.node_count

      self.cell_count = rqet.find_tag_int(node, 'CellCount')
      assert self.cell_count is not None and self.cell_count > 0

      # TODO: remove this if block once RMS export issue resolved
      if self.cell_count == self.node_count:
         extended_mds = np.empty((self.node_mds.size + 1,))
         extended_mds[:-1] = self.node_mds
         extended_mds[-1] = self.node_mds[-1] + 1.0
         self.node_mds = extended_mds
         self.node_count += 1

      assert self.cell_count < self.node_count

      ci_node = rqet.find_tag(node, 'CellIndices')
      assert ci_node is not None, 'blocked well cell indices hdf5 reference not found in xml'
      load_hdf5_array(self, ci_node, 'cell_indices', dtype = int)
      assert (self.cell_indices is not None and self.cell_indices.ndim == 1 and
              self.cell_indices.size == self.cell_count), 'mismatch in number of cell indices for blocked well'
      self.cellind_null = rqet.find_tag_int(ci_node, 'NullValue')
      if self.cellind_null is None:
         self.cellind_null = -1  # if no Null found assume -1 default

      fi_node = rqet.find_tag(node, 'LocalFacePairPerCellIndices')
      assert fi_node is not None, 'blocked well face indices hdf5 reference not found in xml'
      load_hdf5_array(self, fi_node, 'raw_face_indices', dtype = 'int')
      assert self.raw_face_indices is not None, 'failed to load face indices for blocked well'
      assert self.raw_face_indices.size == 2 * self.cell_count, 'mismatch in number of cell faces for blocked well'
      if self.raw_face_indices.ndim > 1:
         self.raw_face_indices = self.raw_face_indices.reshape((self.raw_face_indices.size,))
      mask = np.where(self.raw_face_indices == -1)
      self.raw_face_indices[mask] = 0
      self.face_pair_indices = self.face_index_inverse_map[self.raw_face_indices]
      self.face_pair_indices[mask] = (-1, -1)
      self.face_pair_indices = self.face_pair_indices.reshape((-1, 2, 2))
      del self.raw_face_indices
      self.facepair_null = rqet.find_tag_int(fi_node, 'NullValue')
      if self.facepair_null is None:
         self.facepair_null = -1

      gi_node = rqet.find_tag(node, 'GridIndices')
      assert gi_node is not None, 'blocked well grid indices hdf5 reference not found in xml'
      load_hdf5_array(self, gi_node, 'grid_indices', dtype = 'int')
      assert self.grid_indices is not None and self.grid_indices.ndim == 1 and self.grid_indices.size == self.node_count - 1
      unique_grid_indices = np.unique(self.grid_indices)  # sorted list of unique values
      self.gridind_null = rqet.find_tag_int(gi_node, 'NullValue')
      if self.gridind_null is None:
         self.gridind_null = -1  # if no Null found assume -1 default

      grid_node_list = rqet.list_of_tag(node, 'Grid')
      assert len(grid_node_list) > 0, 'blocked well grid reference(s) not found in xml'
      assert unique_grid_indices[0] >= -1 and unique_grid_indices[-1] < len(
         grid_node_list), 'blocked well grid index out of range'
      assert np.count_nonzero(self.grid_indices >= 0) == self.cell_count, 'mismatch in number of blocked well intervals'
      self.grid_list = []
      for grid_ref_node in grid_node_list:
         grid_node = self.model.referenced_node(grid_ref_node)
         assert grid_node is not None, 'grid referenced in blocked well xml is not present in model'
         grid_uuid = rqet.uuid_for_part_root(grid_node)
         grid_obj = self.model.grid(uuid = grid_uuid, find_properties = False)
         self.grid_list.append(grid_obj)

      interp_uuid = rqet.find_nested_tags_text(node, ['RepresentedInterpretation', 'UUID'])
      if interp_uuid is None:
         self.wellbore_interpretation = None
      else:
         self.wellbore_interpretation = rqo.WellboreInterpretation(self.model, uuid = interp_uuid)

      # Create blocked well log collection of all log data
      self.logs = rqp.WellIntervalPropertyCollection(frame = self)

      # Set up matches between cell_indices and grid_indices
      self.cell_grid_link = self.map_cell_and_grid_indices()

   def map_cell_and_grid_indices(self):
      """Returns a list of index values linking the grid_indices to cell_indices.

      note:
         length will match grid_indices, and will show -1 where cell is unblocked
      """

      indexmap = []
      j = 0
      for i in self.grid_indices:
         if i == -1:
            indexmap.append(-1)
         else:
            indexmap.append(j)
            j += 1
      return indexmap

   def compressed_grid_indices(self):
      """Returns a list of grid indices excluding the -1 elements (unblocked intervals).

      note:
         length will match that of cell_indices
      """

      compressed = []
      for i in self.grid_indices:
         if i >= 0:
            compressed.append(i)
      assert len(compressed) == self.cell_count
      return compressed

   def number_of_grids(self):
      """Returns the number of grids referenced by the blocked well object."""

      if self.grid_list is None:
         return 0
      return len(self.grid_list)

   def single_grid(self):
      """Asserts that exactly one grid is being referenced and returns a grid object for that grid."""

      assert len(self.grid_list) == 1, 'blocked well is not referring to exactly one grid'
      return self.grid_list[0]

   def grid_uuid_list(self):
      """Returns a list of the uuids of the grids referenced by the blocked well object.

      :meta common:
      """

      uuid_list = []
      if self.grid_list is None:
         return uuid_list
      for g in self.grid_list:
         uuid_list.append(g.uuid)
      return uuid_list

   def cell_indices_kji0(self):
      """Returns a numpy int array of shape (N, 3) of cells visited by well, for a single grid situation.

      :meta common:
      """

      grid = self.single_grid()
      return grid.denaturalized_cell_indices(self.cell_indices)

   def cell_indices_and_grid_list(self):
      """Returns a numpy int array of shape (N, 3) of cells visited by well, and a list of grid objects of length N.

      :meta common:
      """

      grid_for_cell_list = []
      grid_indices = self.compressed_grid_indices()
      assert len(grid_indices) == self.cell_count
      cell_indices = np.empty((self.cell_count, 3), dtype = int)
      for cell_number in range(self.cell_count):
         grid = self.grid_list[grid_indices[cell_number]]
         grid_for_cell_list.append(grid)
         cell_indices[cell_number] = grid.denaturalized_cell_index(self.cell_indices[cell_number])
      return cell_indices, grid_for_cell_list

   def cell_indices_for_grid_uuid(self, grid_uuid):
      """Returns a numpy int array of shape (N, 3) of cells visited by well in specified grid.

      :meta common:
      """

      if isinstance(grid_uuid, str):
         grid_uuid = bu.uuid_from_string(grid_uuid)
      ci_list, grid_list = self.cell_indices_and_grid_list()
      mask = np.zeros((len(ci_list),), dtype = bool)
      for cell_number in range(len(ci_list)):
         mask[cell_number] = bu.matching_uuids(grid_list[cell_number].uuid, grid_uuid)
      ci_selected = ci_list[mask]
      return ci_selected

   def box(self, grid_uuid = None):
      """Returns the KJI box containing the cells visited by the well, for single grid if grid_uuid is None."""

      if grid_uuid is None:
         cells_kji0 = self.cell_indices_kji0()
      else:
         cells_kji0 = self.cell_indices_for_grid_uuid(grid_uuid)

      if cells_kji0 is None or len(cells_kji0) == 0:
         return None
      well_box = np.empty((2, 3), dtype = int)
      well_box[0] = np.min(cells_kji0, axis = 0)
      well_box[1] = np.max(cells_kji0, axis = 0)
      return well_box

   def face_pair_array(self):
      """Returns numpy int array of shape (N, 2, 2) being pairs of face (axis, polarity) pairs, to go with cell_kji0_array().

      note:

         each of the N rows in the returned array is of the form:

            ((entry_face_axis, entry_face_polarity), (exit_face_axis, exit_face_polarity))

         where the axis values are in the range 0 to 2 for k, j & i respectively, and
         the polarity values are zero for the 'negative' face and 1 for the 'positive' face;
         exit values may be -1 to indicate TD within the cell (ie. no exit point)
      """
      return self.face_pair_indices

   def compute_from_trajectory(self,
                               trajectory,
                               grid,
                               active_only = False,
                               quad_triangles = True,
                               use_single_layer_tactics = True):
      """Populate this blocked wellbore object based on intersection of trajectory with cells of grid.

      arguments:
         trajectory (Trajectory object): the trajectory to intersect with the grid; control_points and crs_root attributes must
            be populated
         grid (grid.Grid object): the grid with which to intersect the trajectory
         active_only (boolean, default False): if True, only active cells are included as blocked intervals
         quad_triangles (boolean, default True): if True, 4 triangles per cell face are used for the intersection calculations;
            if False, only 2 triangles per face are used
         use_single_layer_tactics (boolean, default True): if True and the grid does not have k gaps, initial intersection
            calculations with fault planes or the outer IK & JK skin of the grid are calculated as if the grid is a single
            layer (and only after an intersection is thus found is the actual layer identified); this significantly speeds up
            computation but may cause failure in the presence of significantly non-straight pillars and could (rarely) cause
            problems where a fault plane is significantly skewed (non-planar) even if individual pillars are straight

      note:
         this method is computationally intensive and might take ~30 seconds for a tyipical grid and trajectory; large grids,
         grids with k gaps, or setting use_single_layer_tactics False will typically result in significantly longer processing time
      """

      import resqpy.grid_surface as rgs  # was causing circular import issue when at global level

      # note: see also extract_box_for_well code
      assert trajectory is not None and grid is not None
      if np.any(np.isnan(grid.points_ref(masked = False))):
         log.warning('grid does not have geometry defined everywhere: attempting fill')
         import resqpy.derived_model as rqdm
         fill_grid = rqdm.copy_grid(grid)
         fill_grid.set_geometry_is_defined(nullify_partial_pillars = True, complete_all = True)
         # note: may need to write hdf5 and create xml for fill_grid, depending on use in populate_blocked_well_from_trajectory()
         # fill_grid.write_hdf_from_caches()
         # fill_grid.create_xml
         grid = fill_grid
      assert trajectory.control_points is not None and trajectory.crs_root is not None and grid.crs_root is not None
      assert len(trajectory.control_points)

      self.trajectory = trajectory
      if not self.well_name:
         self.well_name = trajectory.title
      bw = rgs.populate_blocked_well_from_trajectory(self,
                                                     grid,
                                                     active_only = active_only,
                                                     quad_triangles = quad_triangles,
                                                     lazy = False,
                                                     use_single_layer_tactics = use_single_layer_tactics)
      if bw is None:
         raise Exception('failed to generate blocked well from trajectory with uuid: ' + str(trajectory.uuid))

      assert bw is self

   def set_for_column(self, well_name, grid, col_ji0, skip_inactive = True):
      """Populates empty blocked well for a 'vertical' well in given column; creates simulation trajectory and md datum."""

      if well_name:
         self.well_name = well_name
      col_list = ['IW', 'JW', 'L', 'ANGLA', 'ANGLV']  # NB: L is Layer, ie. k
      df = pd.DataFrame(columns = col_list)
      pinch_col = grid.pinched_out(cache_cp_array = True, cache_pinchout_array = True)[:, col_ji0[0], col_ji0[1]]
      if skip_inactive and grid.inactive is not None:
         inactive_col = grid.inactive[:, col_ji0[0], col_ji0[1]]
      else:
         inactive_col = np.zeros(grid.nk, dtype = bool)
      for k0 in range(grid.nk):
         if pinch_col[k0] or inactive_col[k0]:
            continue
         # note: leaving ANGLA & ANGLV columns as NA will cause K face centres to be used when deriving from dataframe
         row_dict = {'IW': col_ji0[1] + 1, 'JW': col_ji0[0] + 1, 'L': k0 + 1}
         df = df.append(row_dict, ignore_index = True)

      return self.derive_from_dataframe(df, self.well_name, grid, use_face_centres = True)

   def derive_from_wellspec(self,
                            wellspec_file,
                            well_name,
                            grid,
                            check_grid_name = False,
                            use_face_centres = False,
                            add_properties = True):
      """Populates empty blocked well from Nexus WELLSPEC data; creates simulation trajectory and md datum.

      args:
         wellspec_file (string): path of Nexus ascii file holding WELLSPEC keyword
         well_name (string): the name of the well as used in the wellspec data
         grid (grid.Grid object): the grid object which the cell indices in the wellspec data relate to
         check_grid_name (boolean, default False): if True, the GRID column of the wellspec data will be checked
            for a match with the citation title of the grid object; perforations for other grids will be skipped;
            if False, all wellspec data is assumed to relate to the grid
         use_face_centres (boolean, default False): if True, cell face centre points are used for the entry and
            exit points when constructing the simulation trajectory; if False and ANGLA & ANGLV data are available
            then entry and exit points are constructed based on a straight line at those angles passing through
            the centre of the cell
         add_properties (bool or list of str, default True): if True, WELLSPEC columns (other than IW, JW, L & GRID)
            are added as property parts for the blocked well; if a list is passed, it must contain a subset of the
            columns in the WELLSPEC data

      returns:
         self if successful; None otherwise

      note:
         if add_properties is True or present as a list, this method will write the hdf5, create the xml and add
         parts to the model for this blocked well and the properties
      """

      if well_name:
         self.well_name = well_name
      else:
         well_name = self.well_name

      if add_properties:
         if isinstance(add_properties, list):
            col_list = ['IW', 'JW', 'L'] + [col.upper() for col in add_properties if col not in ['IW', 'JW', 'L']]
         else:
            col_list = []
      else:
         col_list = ['IW', 'JW', 'L', 'ANGLA', 'ANGLV']
      if check_grid_name:
         grid_name = rqet.citation_title_for_node(grid.root).upper()
         if not grid_name:
            check_grid_name = False
         else:
            col_list.append('GRID')

      wellspec_dict = wsk.load_wellspecs(wellspec_file, well = well_name, column_list = col_list)

      assert len(wellspec_dict) == 1, 'no wellspec data found in file ' + wellspec_file + ' for well ' + well_name

      df = wellspec_dict[well_name]
      assert len(df) > 0, 'no rows of perforation data found in wellspec for well ' + well_name

      name_for_check = grid_name if check_grid_name else None
      return self.derive_from_dataframe(df,
                                        well_name,
                                        grid,
                                        grid_name_to_check = name_for_check,
                                        use_face_centres = use_face_centres,
                                        add_as_properties = add_properties)

   def derive_from_cell_list(self, cell_kji0_list, well_name, grid):
      """Populate empty blocked well from numpy int array of shape (N, 3) being list of cells."""

      df = pd.DataFrame(columns = ['IW', 'JW', 'L'])
      df['IW'] = cell_kji0_list[:, 2] + 1
      df['JW'] = cell_kji0_list[:, 1] + 1
      df['L'] = cell_kji0_list[:, 0] + 1

      return self.derive_from_dataframe(df, well_name, grid, use_face_centres = True)

   def derive_from_dataframe(self,
                             df,
                             well_name,
                             grid,
                             grid_name_to_check = None,
                             use_face_centres = True,
                             add_as_properties = False):
      """Populate empty blocked well from WELLSPEC-like dataframe; first columns must be IW, JW, L (i, j, k).

      note:
         if add_as_properties is True or present as a list of wellspec column names, both the blocked well and
         the properties will have their hdf5 data written, xml created and be added as parts to the model
      """

      def cell_kji0_from_df(df, df_row):
         row = df.iloc[df_row]
         if pd.isna(row[0]) or pd.isna(row[1]) or pd.isna(row[2]):
            return None
         cell_kji0 = np.empty((3,), dtype = int)
         cell_kji0[:] = row[2], row[1], row[0]
         cell_kji0[:] -= 1
         return cell_kji0

      if well_name:
         self.well_name = well_name
      else:
         well_name = self.well_name

      assert len(df) > 0, 'empty dataframe for blocked well ' + str(well_name)

      length_uom = grid.z_units()
      assert grid.xy_units() == length_uom, 'mixed length units in grid crs'

      previous_xyz = None
      trajectory_mds = []
      trajectory_points = []  # entries paired with trajectory_mds
      blocked_intervals = [
      ]  # will have one fewer entries than trajectory nodes; 0 = blocked, -1 = not blocked (for grid indices)
      blocked_cells_kji0 = []  # will have length equal to number of 0's in blocked intervals
      blocked_face_pairs = [
      ]  # same length as blocked_cells_kji0; each is ((entry axis, entry polarity), (exit axis, exit polarity))

      log.debug('wellspec dataframe for well ' + str(well_name) + ' has ' + str(len(df)) + ' row' + _pl(len(df)))

      skipped_warning_grid = None

      angles_present = ('ANGLV' in df.columns and 'ANGLA' in df.columns and not pd.isnull(df.iloc[0]['ANGLV']) and
                        not pd.isnull(df.iloc[0]['ANGLA']))

      # TODO: remove these temporary overrides
      angles_present = False
      use_face_centres = True

      if not angles_present and not use_face_centres:
         log.warning(f'ANGLV and/or ANGLA data unavailable for well {well_name}: using face centres')
         use_face_centres = True

      for i in range(len(df)):  # for each row in the dataframe for this well

         cell_kji0 = cell_kji0_from_df(df, i)
         if cell_kji0 is None:
            log.error('missing cell index in wellspec data for well ' + str(well_name) + ' row ' + str(i + 1))
            continue

         row = df.iloc[i]

         if grid_name_to_check and pd.notna(row['GRID']) and grid_name_to_check != str(row['GRID']).upper():
            other_grid = str(row['GRID'])
            if skipped_warning_grid != other_grid:
               log.warning('skipping perforation(s) in grid ' + other_grid + ' for well ' + str(well_name))
               skipped_warning_grid = other_grid
            continue
         cp = grid.corner_points(cell_kji0 = cell_kji0, cache_resqml_array = False)
         assert not np.any(np.isnan(cp)), 'missing geometry for perforation cell for well ' + str(well_name)

         if angles_present:
            log.debug('row ' + str(i) + ': using angles')
            angla = row['ANGLA']
            inclination = row['ANGLV']
            if inclination < 0.1:
               azimuth = 0.0
            else:
               i_vector = np.sum(cp[:, :, 1] - cp[:, :, 0], axis = (0, 1))
               azimuth = vec.azimuth(i_vector) - angla  # see Nexus keyword reference doc
            well_vector = vec.unit_vector_from_azimuth_and_inclination(azimuth, inclination) * 10000.0
            # todo: the following might be producing NaN's when vector passes precisely through an edge
            (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity,
             exit_xyz) = find_entry_and_exit(cp, -well_vector, well_vector, well_name)
         else:
            # fabricate entry and exit axes and polarities based on indices alone
            # note: could use geometry but here a cheap rough-and-ready approach is used
            log.debug('row ' + str(i) + ': using cell moves')
            if i == 0:
               entry_axis, entry_polarity = 0, 0  # K-
            else:
               entry_move = cell_kji0 - blocked_cells_kji0[-1]
               log.debug(f'entry move: {entry_move}')
               if entry_move[1] == 0 and entry_move[2] == 0:  # K move
                  entry_axis = 0
                  entry_polarity = 0 if entry_move[0] >= 0 else 1
               elif abs(entry_move[1]) > abs(entry_move[2]):  # J dominant move
                  entry_axis = 1
                  entry_polarity = 0 if entry_move[1] >= 0 else 1
               else:  # I dominant move
                  entry_axis = 2
                  entry_polarity = 0 if entry_move[2] >= 0 else 1
            if i == len(df) - 1:
               exit_axis, exit_polarity = entry_axis, 1 - entry_polarity
            else:
               next_cell_kji0 = cell_kji0_from_df(df, i + 1)
               if next_cell_kji0 is None:
                  exit_axis, exit_polarity = entry_axis, 1 - entry_polarity
               else:
                  exit_move = next_cell_kji0 - cell_kji0
                  log.debug(f'exit move: {exit_move}')
                  if exit_move[1] == 0 and exit_move[2] == 0:  # K move
                     exit_axis = 0
                     exit_polarity = 1 if exit_move[0] >= 0 else 0
                  elif abs(exit_move[1]) > abs(exit_move[2]):  # J dominant move
                     exit_axis = 1
                     exit_polarity = 1 if exit_move[1] >= 0 else 0
                  else:  # I dominant move
                     exit_axis = 2
                     exit_polarity = 1 if exit_move[2] >= 0 else 0

         if use_face_centres:  # override the vector based xyz entry and exit points with face centres
            if entry_axis == 0:
               entry_xyz = np.mean(cp[entry_polarity, :, :], axis = (0, 1))
            elif entry_axis == 1:
               entry_xyz = np.mean(cp[:, entry_polarity, :], axis = (0, 1))
            else:
               entry_xyz = np.mean(cp[:, :, entry_polarity], axis = (0, 1))  # entry_axis == 2, ie. I
            if exit_axis == 0:
               exit_xyz = np.mean(cp[exit_polarity, :, :], axis = (0, 1))
            elif exit_axis == 1:
               exit_xyz = np.mean(cp[:, exit_polarity, :], axis = (0, 1))
            else:
               exit_xyz = np.mean(cp[:, :, exit_polarity], axis = (0, 1))  # exit_axis == 2, ie. I

         log.debug(
            f'cell: {cell_kji0}; entry axis: {entry_axis}; polarity {entry_polarity}; exit axis: {exit_axis}; polarity {exit_polarity}'
         )

         if previous_xyz is None:  # first entry
            log.debug('adding mean sea level trajectory start')
            previous_xyz = entry_xyz.copy()
            previous_xyz[2] = 0.0  # use depth zero as md datum
            trajectory_mds.append(0.0)
            trajectory_points.append(previous_xyz)
         if not vec.isclose(previous_xyz, entry_xyz, tolerance = 0.05):  # add an unblocked interval
            log.debug('adding unblocked interval')
            trajectory_points.append(entry_xyz)
            new_md = trajectory_mds[-1] + vec.naive_length(entry_xyz - previous_xyz)  # assumes x, y & z units are same
            trajectory_mds.append(new_md)
            blocked_intervals.append(-1)  # unblocked interval
            previous_xyz = entry_xyz
         log.debug('adding blocked interval for cell kji0: ' + str(cell_kji0))
         trajectory_points.append(exit_xyz)
         new_md = trajectory_mds[-1] + vec.naive_length(exit_xyz - previous_xyz)  # assumes x, y & z units are same
         trajectory_mds.append(new_md)
         blocked_intervals.append(0)  # blocked interval
         previous_xyz = exit_xyz
         blocked_cells_kji0.append(cell_kji0)
         blocked_face_pairs.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

      blocked_count = len(blocked_cells_kji0)
      if blocked_count == 0:
         log.warning('no intervals blocked for well ' + str(well_name))
         return None
      else:
         log.info(str(blocked_count) + ' interval' + _pl(blocked_count) + ' blocked for well ' + str(well_name))

      self.node_count = len(trajectory_mds)
      self.node_mds = np.array(trajectory_mds)
      self.cell_count = len(blocked_cells_kji0)
      self.grid_indices = np.array(blocked_intervals, dtype = int)  # NB. only supporting one grid at the moment
      self.cell_indices = grid.natural_cell_indices(np.array(blocked_cells_kji0))
      self.face_pair_indices = np.array(blocked_face_pairs, dtype = int)
      self.grid_list = [grid]

      # if last segment terminates at bottom face in bottom layer, add a tail to trajectory
      if blocked_count > 0 and exit_axis == 0 and exit_polarity == 1 and cell_kji0[
            0] == grid.nk - 1 and grid.k_direction_is_down:
         tail_length = 10.0  # metres or feet
         tail_xyz = trajectory_points[-1].copy()
         tail_xyz[2] += tail_length * (1.0 if grid.z_inc_down() else -1.0)
         trajectory_points.append(tail_xyz)
         new_md = trajectory_mds[-1] + tail_length
         trajectory_mds.append(new_md)

      self.create_md_datum_and_trajectory(grid, trajectory_mds, trajectory_points, length_uom, well_name)

      if add_as_properties and len(df.columns) > 3:
         # NB: atypical writing of hdf5 data and xml creation in order to support related properties
         self.write_hdf5()
         self.create_xml()
         if isinstance(add_as_properties, list):
            for col in add_as_properties:
               assert col in df.columns[3:]  # could just skip missing columns
            property_columns = add_as_properties
         else:
            property_columns = df.columns[3:]
         self._add_df_properties(df, property_columns, length_uom = length_uom)

      return self

   def import_from_rms_cellio(self, cellio_file, well_name, grid, include_overburden_unblocked_interval = False):
      """Populates empty blocked well from RMS cell I/O data; creates simulation trajectory and md datum.

      args:
         cellio_file (string): path of RMS ascii export file holding blocked well cell I/O data; cell entry and
            exit points are expected
         well_name (string): the name of the well as used in the cell I/O file
         grid (grid.Grid object): the grid object which the cell indices in the cell I/O data relate to

      returns:
         self if successful; None otherwise
      """

      if well_name:
         self.well_name = well_name
      else:
         well_name = self.well_name

      grid_name = rqet.citation_title_for_node(grid.root)
      length_uom = grid.z_units()
      grid_z_inc_down = crs.Crs(grid.model, uuid = grid.crs_uuid).z_inc_down
      log.debug('grid z increasing downwards: ' + str(grid_z_inc_down) + '(type: ' + str(type(grid_z_inc_down)) + ')')
      cellio_z_inc_down = None

      try:
         assert ' ' not in well_name, 'cannot import for well name containing spaces'
         with open(cellio_file, 'r') as fp:
            while True:
               kf.skip_blank_lines_and_comments(fp)
               line = fp.readline()  # file format version number?
               assert line, 'well ' + str(well_name) + ' not found in file ' + str(cellio_file)
               fp.readline()  # 'Undefined'
               words = fp.readline().split()
               assert len(words), 'missing header info in cell I/O file'
               if words[0].upper() == well_name.upper():
                  break
               while not kf.blank_line(fp):
                  fp.readline()  # skip to block of data for next well
            header_lines = int(fp.readline().strip())
            for _ in range(header_lines):
               fp.readline()
            previous_xyz = None
            trajectory_mds = []
            trajectory_points = []  # entries paired with trajectory_mds
            blocked_intervals = [
            ]  # will have one fewer entries than trajectory nodes; 0 = blocked, -1 = not blocked (for grid indices)
            blocked_cells_kji0 = []  # will have length equal to number of 0's in blocked intervals
            blocked_face_pairs = [
            ]  # same length as blocked_cells_kji0; each is ((entry axis, entry polarity), (exit axis, exit polarity))

            while not kf.blank_line(fp):

               line = fp.readline()
               words = line.split()
               assert len(words) >= 9, 'not enough items on data line in cell I/O file, minimum 9 expected'
               i1, j1, k1 = int(words[0]), int(words[1]), int(words[2])
               cell_kji0 = np.array((k1 - 1, j1 - 1, i1 - 1), dtype = int)
               assert np.all(0 <= cell_kji0) and np.all(
                  cell_kji0 < grid.extent_kji), 'cell I/O cell index not within grid extent'
               entry_xyz = np.array((float(words[3]), float(words[4]), float(words[5])))
               exit_xyz = np.array((float(words[6]), float(words[7]), float(words[8])))
               if cellio_z_inc_down is None:
                  cellio_z_inc_down = bool(entry_xyz[2] + exit_xyz[2] > 0.0)
               if cellio_z_inc_down != grid_z_inc_down:
                  entry_xyz[2] = -entry_xyz[2]
                  exit_xyz[2] = -exit_xyz[2]

               cp = grid.corner_points(cell_kji0 = cell_kji0, cache_resqml_array = False)
               assert not np.any(np.isnan(
                  cp)), 'missing geometry for perforation cell(kji0) ' + str(cell_kji0) + ' for well ' + str(well_name)
               cell_centre = np.mean(cp, axis = (0, 1, 2))

               # let's hope everything is in the same coordinate reference system!
               entry_vector = 100.0 * (entry_xyz - cell_centre)
               exit_vector = 100.0 * (exit_xyz - cell_centre)
               (entry_axis, entry_polarity, facial_entry_xyz, exit_axis, exit_polarity,
                facial_exit_xyz) = find_entry_and_exit(cp, entry_vector, exit_vector, well_name)

               if previous_xyz is None:  # first entry
                  previous_xyz = entry_xyz.copy()
                  if include_overburden_unblocked_interval:
                     log.debug('adding mean sea level trajectory start')
                     previous_xyz[2] = 0.0  # use depth zero as md datum
                  trajectory_mds.append(previous_xyz[2])
                  trajectory_points.append(previous_xyz)

               if not vec.isclose(previous_xyz, entry_xyz, tolerance = 0.05):  # add an unblocked interval
                  log.debug('adding unblocked interval')
                  trajectory_points.append(entry_xyz)
                  new_md = trajectory_mds[-1] + vec.naive_length(
                     entry_xyz - previous_xyz)  # assumes x, y & z units are same
                  trajectory_mds.append(new_md)
                  blocked_intervals.append(-1)  # unblocked interval
                  previous_xyz = entry_xyz

               log.debug('adding blocked interval for cell kji0: ' + str(cell_kji0))
               trajectory_points.append(exit_xyz)
               new_md = trajectory_mds[-1] + vec.naive_length(
                  exit_xyz - previous_xyz)  # assumes x, y & z units are same
               trajectory_mds.append(new_md)
               blocked_intervals.append(0)  # blocked interval
               previous_xyz = exit_xyz
               blocked_cells_kji0.append(cell_kji0)
               blocked_face_pairs.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

            blocked_count = len(blocked_cells_kji0)
            if blocked_count == 0:
               log.warning('no intervals blocked for well ' + well_name + ' in grid ' + str(grid_name))
               return None
            else:
               log.info(
                  str(blocked_count) + ' interval' + _pl(blocked_count) + ' blocked for well ' + well_name +
                  ' in grid ' + str(grid_name))

            self.create_md_datum_and_trajectory(grid,
                                                trajectory_mds,
                                                trajectory_points,
                                                length_uom,
                                                well_name,
                                                set_depth_zero = True,
                                                set_tangent_vectors = True)

            self.node_count = len(trajectory_mds)
            self.node_mds = np.array(trajectory_mds)
            self.cell_count = len(blocked_cells_kji0)
            self.grid_indices = np.array(blocked_intervals, dtype = int)  # NB. only supporting one grid at the moment
            self.cell_indices = grid.natural_cell_indices(np.array(blocked_cells_kji0))
            self.face_pair_indices = np.array(blocked_face_pairs)
            self.grid_list = [grid]

      except Exception:
         log.exception('failed to import info for blocked well ' + str(well_name) + ' from cell I/O file ' +
                       str(cellio_file))
         return None

      return self

   def dataframe(self,
                 i_col = 'IW',
                 j_col = 'JW',
                 k_col = 'L',
                 one_based = True,
                 extra_columns_list = [],
                 ntg_uuid = None,
                 perm_i_uuid = None,
                 perm_j_uuid = None,
                 perm_k_uuid = None,
                 satw_uuid = None,
                 sato_uuid = None,
                 satg_uuid = None,
                 region_uuid = None,
                 radw = None,
                 skin = None,
                 stat = None,
                 active_only = False,
                 min_k0 = None,
                 max_k0 = None,
                 k0_list = None,
                 min_length = None,
                 min_kh = None,
                 max_depth = None,
                 max_satw = None,
                 min_sato = None,
                 max_satg = None,
                 perforation_list = None,
                 region_list = None,
                 depth_inc_down = None,
                 set_k_face_intervals_vertical = False,
                 anglv_ref = 'normal ij down',
                 angla_plane_ref = None,
                 length_mode = 'MD',
                 length_uom = None,
                 use_face_centres = False,
                 preferential_perforation = True,
                 add_as_properties = False,
                 use_properties = False):
      """Returns a pandas data frame containing WELLSPEC style data.

      arguments:
         i_col (string, default 'IW'): the column name to use for cell I index values
         j_col (string, default 'JW'): the column name to use for cell J index values
         k_col (string, default 'L'): the column name to use for cell K index values
         one_based (boolean, default True): if True, simulator protocol i, j & k values are placed in I, J & K columns;
            if False, resqml zero based values; this does not affect the interpretation of min_k0 & max_k0 arguments
         extra_columns_list (list of string, optional): list of WELLSPEC column names to include in the dataframe, from currently
            recognised values: 'GRID', 'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'RADW', 'SKIN', 'PPERF', 'RADB', 'WI', 'WBC'
         ntg_uuid (uuid.UUID, optional): the uuid of the net to gross ratio property; if present is used to downgrade the i & j
            permeabilities in the calculation of KH; ignored if 'KH' not in the extra column list and min_kh is not specified;
            the argument may also be a dictionary mapping from grid uuid to ntg uuid; if no net to gross data is provided, it
            is effectively assumed to be one (or, equivalently, the I & J permeability data is applicable to the gross rock); see
            also preferential_perforation argument which can cause adjustment of effective ntg in partially perforated cells
         perm_i_uuid (uuid.UUID or dictionary, optional): the uuid of the permeability property in the I direction;
            required if 'KH' is included in the extra columns list and min_kh is not specified; ignored otherwise;
            the argument may also be a dictionary mapping from grid uuid to perm I uuid
         perm_j_uuid (uuid.UUID, optional): the uuid (or dict) of the permeability property in the J direction;
            defaults to perm_i_uuid
         perm_k_uuid (uuid.UUID, optional): the uuid (or dict) of the permeability property in the K direction;
            defaults to perm_i_uuid
         satw_uuid (uuid.UUID, optional): the uuid of a water saturation property; required if max_satw is specified; may also
            be a dictionary mapping from grid uuid to satw uuid; ignored if max_satw is None
         sato_uuid (uuid.UUID, optional): the uuid of an oil saturation property; required if min_sato is specified; may also
            be a dictionary mapping from grid uuid to sato uuid; ignored if min_sato is None
         satg_uuid (uuid.UUID, optional): the uuid of a gas saturation property; required if max_satg is specified; may also
            be a dictionary mapping from grid uuid to satg uuid; ignored if max_satg is None
         region_uuid (uuid.UUID, optional): the uuid of a discrete or categorical property, required if region_list is not None;
            may also be a dictionary mapping from grid uuid to region uuid; ignored if region_list is None
         radw (float, optional): if present, the wellbore radius used for all perforations; must be in correct units for intended
            use of the WELLSPEC style dataframe; will default to 0.25 if 'RADW' is included in the extra column list
         skin (float, optional): if present, a skin column is included with values set to this constant
         stat (string, optional): if present, should be 'ON' or 'OFF' and is used for all perforations; will default to 'ON' if
            'STAT' is included in the extra column list
         active_only (boolean, default False): if True, only cells that are flagged in the grid object as active are included;
            if False, cells are included whether active or not
         min_k0 (int, optional): if present, perforations in layers above this are excluded (layer number will be applied
            naively to all grids – not recommended when working with more than one grid with different layering)
         max_k0 (int, optional): if present, perforations in layers below this are excluded (layer number will be applied
            naively to all grids – not recommended when working with more than one grid with different layering)
         k0_list (list of int, optional): if present, only perforations in cells in these layers are included (layer numbers
            will be applied naively to all grids – not recommended when working with more than one grid with different layering)
         min_length (float, optional): if present, a minimum length for an individual perforation interval to be included;
            units are the length units of the trajectory object unless length_uom argument is set
         min_kh (float, optional): if present, the minimum permeability x length value for which an individual interval is
            included; permeabilty uuid(s) must be supplied for the kh calculation; units of the length component are those
            of the trajectory object unless length_uom argument is set
         max_depth (float, optional): if present, rows are excluded for cells with a centre point depth greater than this value;
            max_depth should be positive downwards, with units of measure those of the grid z coordinates
         max_satw (float, optional): if present, perforations in cells where the water saturation exceeds this value will
            be excluded; satw_uuid must be supplied if this argument is present
         min_sato (float, optional): if present, perforations in cells where the oil saturation is less than this value will
            be excluded; sato_uuid must be supplied if this argument is present
         max_satg (float, optional): if present, perforations in cells where the gas saturation exceeds this value will
            be excluded; satg_uuid must be supplied if this argument is present
         perforation_list (list of (float, float), optional): if present, a list of perforated intervals; each entry is the
            start and end measured depths for a perforation; these do not need to align with cell boundaries
         region_list (list of int, optional): if present, a list of region numbers for which rows are to be included; the
            property holding the region data is identified by the region_uuid argument
         depth_inc_down (boolean, optional): if present and True, the depth values will increase with depth; if False or None,
            the direction of the depth values will be determined by the z increasing downwards indicator in the trajectory crs
         set_k_face_intervals_vertical (boolean, default False): if True, intervals with entry through K- and exit through K+
            will have angla and anglv set to 0.0 (vertical); if False angles will be computed depending on geometry
         anglv_ref (string, default 'normal ij down'): either 'gravity', 'z down' (same as gravity), 'z+', 'k down', 'k+',
            'normal ij', or 'normal ij down';
            the ANGLV angles are relative to a local (per cell) reference vector selected by this keyword
         angla_plane_ref (string, optional): string indicating normal vector defining plane onto which trajectory and I axis are
            projected for the calculation of ANGLA; options as for anglv_ref, or 'normal well i+' which results in no projection;
            defaults to the same as anglv_ref
         length_mode (string, default 'MD'): 'MD' or 'straight' indicating which length to use; 'md' takes measured depth
            difference between exit and entry; 'straight' uses a naive straight line length between entry and exit;
            this will affect values for LENGTH, KH, DEPTH, X & Y
         length_uom (string, optional): if present, either 'm' or 'ft': the length units to use for the LENGTH, KH, MD, DEPTH,
            X & Y columns if they are present in extra_columns_list; also used to interpret min_length and min_kh; if None, the
            length units of the trajectory attribute are used LENGTH, KH & MD and those of the grid are used for DEPTH, X & Y;
            RADW value, if present, is assumed to be in the correct units and is not changed; also used implicitly to determine
            conversion constant used in calculation of wellbore constant (WBC)
         use_face_centres (boolean, default False): if True, the centre points of the entry and exit faces will determine the
            vector used as the basis of ANGLA and ANGLV calculations; if False, the trajectory locations for the entry and exit
            measured depths will be used
         preferential_perforation (boolean, default True): if perforation_list is given, and KH is requested or a min_kh given,
            the perforated intervals are assumed to penetrate pay rock preferentially: an effective ntg weighting is computed
            to account for any residual non-pay perforated interval; ignored if perforation_list is None or kh values are not
            being computed
         add_as_properties (boolean or list of str, default False): if True, each column in the extra_columns_list (excluding
            GRID and STAT) is added as a property with the blocked well as supporting representation and 'cells' as the
            indexable element; any cell that is excluded from the dataframe will have corresponding entries of NaN in all the
            properties; if a list is provided it must be a subset of extra_columns_list
         use_properties (boolean or list of str, default False): if True, each column in the extra_columns_list (excluding
            GRID and STAT) is populated from a property with citation title matching the column name, if it exists

      notes:
         units of length along wellbore will be those of the trajectory's length_uom (also applies to K.H values) unless
         the length_uom argument is used;
         the constraints are applied independently for each row and a row is excluded if it fails any constraint;
         the min_k0 and max_k0 arguments do not stop later rows within the layer range from being included;
         the min_length and min_kh limits apply to individual cell intervals and thus depend on cell size;
         the water and oil saturation limits are for saturations at a single time and affect whether the interval
         is included in the dataframe – there is no functionality to support turning perforations off and on over time;
         the saturation limits do not stop deeper intervals with qualifying saturations from being included;
         the k0_list, perforation_list and region_list arguments should be set to None to disable the corresponding functionality,
         if set to an empty list, no rows will be included in the dataframe;
         if add_as_properties is True, the blocked well must already have been added as a part to the model;
         at add_as_properties and use_properties cannot both be True;
         add_as_properties and use_properties are only currently functional for single grid blocked wells;
         at present, unit conversion is not handled when using properties

      :meta common:
      """

      def prop_array(uuid_or_dict, grid):
         assert uuid_or_dict is not None and grid is not None
         if isinstance(uuid_or_dict, dict):
            prop_uuid = uuid_or_dict[grid.uuid]
         else:
            prop_uuid = uuid_or_dict  # uuid either in form of string or uuid.UUID
         return grid.property_collection.single_array_ref(uuid = prop_uuid)

      def get_ref_vector(grid, grid_crs, cell_kji0, mode):
         # gravity = np.array((0.0, 0.0, 1.0))
         if mode == 'normal well i+':
            return None  # ANGLA only: option for no projection onto a plane
         ref_vector = None
         # options for anglv or angla reference: 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down'
         cell_axial_vectors = None
         if not mode.startswith('z'):
            cell_axial_vectors = grid.interface_vectors_kji(cell_kji0)
         if mode == 'z+':
            ref_vector = np.array((0.0, 0.0, 1.0))
         elif mode == 'z down':
            if grid_crs.z_inc_down:
               ref_vector = np.array((0.0, 0.0, 1.0))
            else:
               ref_vector = np.array((0.0, 0.0, -1.0))
         elif mode in ['k+', 'k down']:
            ref_vector = vec.unit_vector(cell_axial_vectors[0])
            if mode == 'k down' and not grid.k_direction_is_down:
               ref_vector = -ref_vector
         else:  # normal to plane of ij axes
            ref_vector = vec.unit_vector(vec.cross_product(cell_axial_vectors[1], cell_axial_vectors[2]))
            if mode == 'normal ij down':
               if grid_crs.z_inc_down:
                  if ref_vector[2] < 0.0:
                     ref_vector = -ref_vector
               else:
                  if ref_vector[2] > 0.0:
                     ref_vector = -ref_vector
         if ref_vector is None or ref_vector[2] == 0.0:
            if grid_crs.z_inc_down:
               ref_vector = np.array((0.0, 0.0, 1.0))
            else:
               ref_vector = np.array((0.0, 0.0, -1.0))
         return ref_vector

      assert length_mode in ['MD', 'straight']
      assert length_uom is None or length_uom in ['m', 'ft']
      assert anglv_ref in ['gravity', 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down']
      if anglv_ref == 'gravity':
         anglv_ref = 'z down'
      if angla_plane_ref is None:
         angla_plane_ref = anglv_ref
      assert angla_plane_ref in [
         'gravity', 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down', 'normal well i+'
      ]
      if angla_plane_ref == 'gravity':
         angla_plane_ref = 'z down'
      column_list = [i_col, j_col, k_col]
      if extra_columns_list:
         for extra in extra_columns_list:
            assert extra.upper() in [
               'GRID', 'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'SKIN', 'RADW', 'PPERF', 'RADB', 'WI',
               'WBC'
            ]
            column_list.append(extra.upper())
      else:
         add_as_properties = use_properties = False
      assert not (add_as_properties and use_properties)
      pc = rqp.PropertyCollection(support = self) if use_properties else None
      pc_titles = [] if pc is None else pc.titles()
      isotropic_perm = None
      if min_length is not None and min_length <= 0.0:
         min_length = None
      if min_kh is not None and min_kh <= 0.0:
         min_kh = None
      if max_satw is not None and max_satw >= 1.0:
         max_satw = None
      if min_sato is not None and min_sato <= 0.0:
         min_sato = None
      if max_satg is not None and max_satg >= 1.0:
         max_satg = None
      doing_kh = False
      if ('KH' in column_list or min_kh is not None) and 'KH' not in pc_titles:
         assert perm_i_uuid is not None, 'KH requested (or minimum specified) without I direction permeabilty being specified'
         doing_kh = True
      if 'WBC' in column_list and 'WBC' not in pc_titles:
         assert perm_i_uuid is not None, 'WBC requested without I direction permeabilty being specified'
         doing_kh = True
      do_well_inflow = (('WI' in column_list and 'WI' not in pc_titles) or
                        ('WBC' in column_list and 'WBC' not in pc_titles) or
                        ('RADB' in column_list and 'RADB' not in pc_titles))
      if do_well_inflow:
         assert perm_i_uuid is not None, 'WI, RADB or WBC requested without I direction permeabilty being specified'
      if doing_kh or do_well_inflow:
         if perm_j_uuid is None and perm_k_uuid is None:
            isotropic_perm = True
         else:
            if perm_j_uuid is None:
               perm_j_uuid = perm_i_uuid
            if perm_k_uuid is None:
               perm_k_uuid = perm_i_uuid
            # following line assumes arguments are passed in same form; if not, some unnecessary maths might be done
            isotropic_perm = (bu.matching_uuids(perm_i_uuid, perm_j_uuid) and
                              bu.matching_uuids(perm_i_uuid, perm_k_uuid))
      if max_satw is not None:
         assert satw_uuid is not None, 'water saturation limit specified without saturation property array'
      if min_sato is not None:
         assert sato_uuid is not None, 'oil saturation limit specified without saturation property array'
      if max_satg is not None:
         assert satg_uuid is not None, 'gas saturation limit specified without saturation property array'
      if region_list is not None:
         assert region_uuid is not None, 'region list specified without region property array'
      if radw is not None and 'RADW' not in column_list:
         column_list.append('RADW')
      if radw is None:
         radw = 0.25
      if skin is not None and 'SKIN' not in column_list:
         column_list.append('SKIN')
      if skin is None:
         skin = 0.0
      if stat is not None:
         assert str(stat).upper() in ['ON', 'OFF']
         stat = str(stat).upper()
         if 'STAT' not in column_list:
            column_list.append('STAT')
      else:
         stat = 'ON'
      if 'GRID' not in column_list and self.number_of_grids() > 1:
         log.error('creating blocked well dataframe without GRID column for well that intersects more than one grid')
      if 'LENGTH' in column_list and 'PPERF' in column_list and 'KH' not in column_list and perforation_list is not None:
         log.warning(
            'both LENGTH and PPERF will include effects of partial perforation; only one should be used in WELLSPEC')
      elif (perforation_list is not None and 'LENGTH' not in column_list and 'PPERF' not in column_list and
            'KH' not in column_list and 'WBC' not in column_list):
         log.warning('perforation list supplied but no use of LENGTH, KH, PPERF nor WBC')
      if min_k0 is None:
         min_k0 = 0
      else:
         assert min_k0 >= 0
      if max_k0 is not None:
         assert min_k0 <= max_k0
      if k0_list is not None and len(k0_list) == 0:
         log.warning('no layers included for blocked well dataframe: no rows will be included')
      if perforation_list is not None and len(perforation_list) == 0:
         log.warning('empty perforation list specified for blocked well dataframe: no rows will be included')
      doing_angles = (('ANGLA' in column_list and 'ANGLA' not in pc_titles) or
                      ('ANGLV' in column_list and 'ANGLV' not in pc_titles) or doing_kh or do_well_inflow)
      doing_xyz = (('X' in column_list and 'X' not in pc_titles) or ('Y' in column_list and 'Y' not in pc_titles) or
                   ('DEPTH' in column_list and 'DEPTH' not in pc_titles))
      doing_entry_exit = doing_angles or ('LENGTH' in column_list and 'LENGTH' not in pc_titles and
                                          length_mode == 'straight')
      grid_crs_list = []
      for grid in self.grid_list:
         grid_crs = crs.Crs(self.model, uuid = grid.crs_uuid)
         grid_crs_list.append(grid_crs)
         if grid_crs.z_units != grid_crs.xy_units and (len(column_list) > 1 or (len(column_list) == 1 and
                                                                                column_list[0] != 'GRID')) is not None:
            log.error('grid ' + str(rqet.citation_title_for_node(grid.root_node)) +
                      ' has z units different to xy units: some WELLSPEC data likely to be wrong')
      k_face_check = np.zeros((2, 2), dtype = int)
      k_face_check[1, 1] = 1  # now represents entry, exit of K-, K+
      k_face_check_end = k_face_check.copy()
      k_face_check_end[1] = -1  # entry through K-, terminating (TD) within cell
      if self.trajectory is None or self.trajectory.crs_root is None:
         traj_crs = None
         traj_z_inc_down = None
      else:
         traj_crs = crs.Crs(self.trajectory.model, uuid = self.trajectory.crs_uuid)
         assert traj_crs.xy_units == traj_crs.z_units
         traj_z_inc_down = traj_crs.z_inc_down

      df = pd.DataFrame(columns = column_list)
      df = df.astype({i_col: int, j_col: int, k_col: int})

      ci = -1
      row_ci_list = []
      if self.node_count is None or self.node_count < 2:
         interval_count = 0
      else:
         interval_count = self.node_count - 1
      for interval in range(interval_count):
         if self.grid_indices[interval] < 0:
            continue  # unblocked interval
         ci += 1
         row_dict = {}
         grid = self.grid_list[self.grid_indices[interval]]
         grid_crs = grid_crs_list[self.grid_indices[interval]]
         grid_name = rqet.citation_title_for_node(grid.root).replace(' ', '_')
         natural_cell = self.cell_indices[ci]
         cell_kji0 = grid.denaturalized_cell_index(natural_cell)
         tuple_kji0 = tuple(cell_kji0)
         if max_depth is not None:
            cell_depth = grid.centre_point(cell_kji0)[2]
            if not grid_crs.z_inc_down:
               cell_depth = -cell_depth
            if cell_depth > max_depth:
               continue
         if active_only and grid.inactive is not None and grid.inactive[tuple_kji0]:
            continue
         if (min_k0 is not None and cell_kji0[0] < min_k0) or (max_k0 is not None and cell_kji0[0] > max_k0):
            continue
         if k0_list is not None and cell_kji0[0] not in k0_list:
            continue
         if region_list is not None and prop_array(region_uuid, grid)[tuple_kji0] not in region_list:
            continue
         if max_satw is not None and prop_array(satw_uuid, grid)[tuple_kji0] > max_satw:
            continue
         if min_sato is not None and prop_array(sato_uuid, grid)[tuple_kji0] < min_sato:
            continue
         if max_satg is not None and prop_array(satg_uuid, grid)[tuple_kji0] > max_satg:
            continue
         if 'PPERF' in pc_titles:
            part_perf_fraction = pc.single_array_ref(citation_title = 'PPERF')[ci]
         else:
            part_perf_fraction = 1.0
            if perforation_list is not None:
               perf_length = 0.0
               for perf_start, perf_end in perforation_list:
                  if perf_end <= self.node_mds[interval] or perf_start >= self.node_mds[interval + 1]:
                     continue
                  if perf_start <= self.node_mds[interval]:
                     if perf_end >= self.node_mds[interval + 1]:
                        perf_length += self.node_mds[interval + 1] - self.node_mds[interval]
                        break
                     else:
                        perf_length += perf_end - self.node_mds[interval]
                  else:
                     if perf_end >= self.node_mds[interval + 1]:
                        perf_length += self.node_mds[interval + 1] - perf_start
                     else:
                        perf_length += perf_end - perf_start
               if perf_length == 0.0:
                  continue
               part_perf_fraction = min(1.0, perf_length / (self.node_mds[interval + 1] - self.node_mds[interval]))
#        log.debug('kji0: ' + str(cell_kji0))
         entry_xyz = None
         exit_xyz = None
         if doing_entry_exit:
            assert self.trajectory is not None
            if use_face_centres:
               entry_xyz = grid.face_centre(cell_kji0, self.face_pair_indices[interval, 0, 0],
                                            self.face_pair_indices[interval, 0, 1])
               if self.face_pair_indices[interval, 1, 0] >= 0:
                  exit_xyz = grid.face_centre(cell_kji0, self.face_pair_indices[interval, 1, 0],
                                              self.face_pair_indices[interval, 1, 1])
               else:
                  exit_xyz = grid.face_centre(cell_kji0, self.face_pair_indices[interval, 0, 0],
                                              1 - self.face_pair_indices[interval, 0, 1])
               ee_crs = grid_crs
            else:
               entry_xyz = self.trajectory.xyz_for_md(self.node_mds[interval])
               exit_xyz = self.trajectory.xyz_for_md(self.node_mds[interval + 1])
               ee_crs = traj_crs
         if length_mode == 'MD':
            length = self.node_mds[interval + 1] - self.node_mds[interval]
            if length_uom is not None and self.trajectory is not None and length_uom != self.trajectory.md_uom:
               length = bwam.convert_lengths(length, self.trajectory.md_uom, length_uom)
         else:  # use straight line length between entry and exit
            length = vec.naive_length(np.array(exit_xyz) -
                                      np.array(entry_xyz))  # trajectory crs, unless use_face_centres!
            if length_uom is not None:
               length = bwam.convert_lengths(length, ee_crs.z_units, length_uom)
            elif self.trajectory is not None:
               length = bwam.convert_lengths(length, ee_crs.z_units, self.trajectory.md_uom)
         if perforation_list is not None:
            length *= part_perf_fraction
         if min_length is not None and length < min_length:
            continue
         sine_anglv = sine_angla = 0.0
         cosine_anglv = cosine_angla = 1.0
         xyz = (np.NaN, np.NaN, np.NaN)
         md = 0.5 * (self.node_mds[interval + 1] + self.node_mds[interval])
         anglv = pc.single_array_ref(citation_title = 'ANGLV')[ci] if 'ANGLV' in pc_titles else None
         angla = pc.single_array_ref(citation_title = 'ANGLA')[ci] if 'ANGLA' in pc_titles else None
         if doing_angles and not (set_k_face_intervals_vertical and
                                  (np.all(self.face_pair_indices[ci] == k_face_check) or
                                   np.all(self.face_pair_indices[ci] == k_face_check_end))):
            vector = vec.unit_vector(np.array(exit_xyz) - np.array(entry_xyz))  # nominal wellbore vector for interval
            if traj_z_inc_down is not None and traj_z_inc_down != grid_crs.z_inc_down:
               vector[2] = -vector[2]
            v_ref_vector = get_ref_vector(grid, grid_crs, cell_kji0, anglv_ref)
            #           log.debug('v ref vector: ' + str(v_ref_vector))
            if angla_plane_ref == anglv_ref:
               a_ref_vector = v_ref_vector
            else:
               a_ref_vector = get_ref_vector(grid, grid_crs, cell_kji0, angla_plane_ref)
            #           log.debug('a ref vector: ' + str(a_ref_vector))
            if anglv is not None:
               anglv_rad = vec.radians_from_degrees(anglv)
               cosine_anglv = maths.cos(anglv_rad)
               sine_anglv = maths.sin(anglv_rad)
            else:
               cosine_anglv = min(max(vec.dot_product(vector, v_ref_vector), -1.0), 1.0)
               anglv_rad = maths.acos(cosine_anglv)
               sine_anglv = maths.sin(anglv_rad)
               anglv = vec.degrees_from_radians(anglv_rad)
#           log.debug('anglv: ' + str(anglv))
            if anglv != 0.0:
               # project well vector and i-axis vector onto plane defined by normal vector a_ref_vector
               i_axis = grid.interface_vector(cell_kji0, 2)
               i_axis = vec.unit_vector(i_axis)
               if a_ref_vector is not None:  # project vector and i axis onto a plane
                  vector -= vec.dot_product(vector, a_ref_vector) * a_ref_vector
                  vector = vec.unit_vector(vector)
                  #                 log.debug('i axis unit vector: ' + str(i_axis))
                  i_axis -= vec.dot_product(i_axis, a_ref_vector) * a_ref_vector
                  i_axis = vec.unit_vector(i_axis)
#                 log.debug('i axis unit vector in reference plane: ' + str(i_axis))
               if angla is not None:
                  angla_rad = vec.radians_from_degrees(angla)
                  cosine_angla = maths.cos(angla_rad)
                  sine_angla = maths.sin(angla_rad)
               else:
                  cosine_angla = min(max(vec.dot_product(vector, i_axis), -1.0), 1.0)
                  angla_rad = maths.acos(cosine_angla)
                  # negate angla if vector is 'clockwise from' i_axis when viewed from above, projected in the xy plane
                  # todo: have discussion around angla sign under different ijk handedness (and z inc direction?)
                  if vec.clockwise((0.0, 0.0), i_axis, vector) > 0.0:
                     angla = -angla
                  sine_angla = maths.sin(angla_rad)
                  angla = vec.degrees_from_radians(angla_rad)


#              log.debug('angla: ' + str(angla))
         else:
            if angla is None:
               angla = 0.0
            if anglv is None:
               anglv = 0.0
         if doing_kh or do_well_inflow:
            if ntg_uuid is None:
               ntg = 1.0
               ntg_is_one = True
            else:
               ntg = prop_array(ntg_uuid, grid)[tuple_kji0]
               ntg_is_one = maths.isclose(ntg, 1.0, rel_tol = 0.001)
            if isotropic_perm and ntg_is_one:
               k_i = k_j = k_k = prop_array(perm_i_uuid, grid)[tuple_kji0]
            else:
               if preferential_perforation and not ntg_is_one:
                  if part_perf_fraction <= ntg:
                     ntg = 1.0  # effective ntg when perforated intervals are in pay
                  else:
                     ntg /= part_perf_fraction  # adjusted ntg when some perforations in non-pay
               # todo: check netgross facet type in property perm i & j parts: if set to gross then don't multiply by ntg below
               k_i = prop_array(perm_i_uuid, grid)[tuple_kji0] * ntg
               k_j = prop_array(perm_j_uuid, grid)[tuple_kji0] * ntg
               k_k = prop_array(perm_k_uuid, grid)[tuple_kji0]
         if doing_kh:
            if isotropic_perm and ntg_is_one:
               kh = length * prop_array(perm_i_uuid, grid)[tuple_kji0]
            else:
               if np.isnan(k_i) or np.isnan(k_j):
                  kh = 0.0
               elif anglv == 0.0:
                  kh = length * maths.sqrt(k_i * k_j)
               elif np.isnan(k_k):
                  kh = 0.0
               else:
                  k_e = maths.pow(k_i * k_j * k_k, 1.0 / 3.0)
                  if k_e == 0.0:
                     kh = 0.0
                  else:
                     l_i = length * maths.sqrt(k_e / k_i) * sine_anglv * cosine_angla
                     l_j = length * maths.sqrt(k_e / k_j) * sine_anglv * sine_angla
                     l_k = length * maths.sqrt(k_e / k_k) * cosine_anglv
                     l_p = maths.sqrt(l_i * l_i + l_j * l_j + l_k * l_k)
                     kh = k_e * l_p
            if min_kh is not None and kh < min_kh:
               continue
         elif 'KH' in pc_titles:
            kh = pc.single_array_ref(citation_title = 'KH')[ci]
         else:
            kh = None
         if 'LENGTH' in pc_titles:
            length = pc.single_array_ref(citation_title = 'LENGTH')[ci]
         if 'RADW' in pc_titles:
            radw = pc.single_array_ref(citation_title = 'RADW')[ci]
         assert radw > 0.0
         if 'SKIN' in pc_titles:
            skin = pc.single_array_ref(citation_title = 'SKIN')[ci]
         radb = wi = wbc = None
         if 'RADB' in pc_titles:
            radb = pc.single_array_ref(citation_title = 'RADB')[ci]
         if 'WI' in pc_titles:
            wi = pc.single_array_ref(citation_title = 'WI')[ci]
         if 'WBC' in pc_titles:
            wbc = pc.single_array_ref(citation_title = 'WBC')[ci]
         if do_well_inflow:
            if isotropic_perm and ntg_is_one:
               k_ei = k_ej = k_ek = k_i
               radw_e = radw
            else:
               k_ei = maths.sqrt(k_j * k_k)
               k_ej = maths.sqrt(k_i * k_k)
               k_ek = maths.sqrt(k_i * k_j)
               r_wi = 0.0 if k_ei == 0.0 else 0.5 * radw * (maths.sqrt(k_ei / k_j) + maths.sqrt(k_ei / k_k))
               r_wj = 0.0 if k_ej == 0.0 else 0.5 * radw * (maths.sqrt(k_ej / k_i) + maths.sqrt(k_ej / k_k))
               r_wk = 0.0 if k_ek == 0.0 else 0.5 * radw * (maths.sqrt(k_ek / k_i) + maths.sqrt(k_ek / k_j))
               rwi = r_wi * sine_anglv * cosine_angla
               rwj = r_wj * sine_anglv * sine_angla
               rwk = r_wk * cosine_anglv
               radw_e = maths.sqrt(rwi * rwi + rwj * rwj + rwk * rwk)
               if radw_e == 0.0:
                  radw_e = radw  # no permeability in this situation anyway
            cell_axial_vectors = grid.interface_vectors_kji(cell_kji0)
            d2 = np.empty(3)
            for axis in range(3):
               d2[axis] = np.sum(cell_axial_vectors[axis] * cell_axial_vectors[axis])
            r_bi = 0.0 if k_ei == 0.0 else 0.14 * maths.sqrt(k_ei * (d2[1] / k_j + d2[0] / k_k))
            r_bj = 0.0 if k_ej == 0.0 else 0.14 * maths.sqrt(k_ej * (d2[2] / k_i + d2[0] / k_k))
            r_bk = 0.0 if k_ek == 0.0 else 0.14 * maths.sqrt(k_ek * (d2[2] / k_i + d2[1] / k_j))
            rbi = r_bi * sine_anglv * cosine_angla
            rbj = r_bj * sine_anglv * sine_angla
            rbk = r_bk * cosine_anglv
            radb_e = maths.sqrt(rbi * rbi + rbj * rbj + rbk * rbk)
            if radb is None:
               radb = radw * radb_e / radw_e
            if wi is None:
               wi = 0.0 if radb <= 0.0 else 2.0 * maths.pi / (maths.log(radb / radw) + skin)
            if 'WBC' in column_list and wbc is None:
               conversion_constant = 8.5270171e-5 if length_uom == 'm' else 0.006328286
               wbc = conversion_constant * kh * wi  # note: pperf aleady accounted for in kh
         if doing_xyz:
            if length_mode == 'MD' and self.trajectory is not None:
               xyz = self.trajectory.xyz_for_md(md)
               if length_uom is not None and length_uom != self.trajectory.md_uom:
                  bwam.convert_lengths(xyz, traj_crs.z_units, length_uom)
               if depth_inc_down and traj_z_inc_down is False:
                  xyz[2] = -xyz[2]
            else:
               xyz = 0.5 * (np.array(exit_xyz) + np.array(entry_xyz))
               if length_uom is not None and length_uom != ee_crs.z_units:
                  bwam.convert_lengths(xyz, ee_crs.z_units, length_uom)
               if depth_inc_down and ee_crs.z_inc_down is False:
                  xyz[2] = -xyz[2]
         xyz = np.array(xyz)
         if 'X' in pc_titles:
            xyz[0] = pc.single_array_ref(citation_title = 'X')[ci]
         if 'Y' in pc_titles:
            xyz[1] = pc.single_array_ref(citation_title = 'Y')[ci]
         if 'DEPTH' in pc_titles:
            xyz[2] = pc.single_array_ref(citation_title = 'DEPTH')[ci]
         if length_uom is not None and self.trajectory is not None and length_uom != self.trajectory.md_uom:
            md = bwam.convert_lengths(md, self.trajectory.md_uom, length_uom)
         if 'MD' in pc_titles:
            md = pc.single_array_ref(citation_title = 'MD')[ci]
         for col_index in range(len(column_list)):
            column = column_list[col_index]
            if col_index < 3:
               if one_based:
                  row_dict[column] = cell_kji0[2 - col_index] + 1
               else:
                  row_dict[column] = cell_kji0[2 - col_index]
            elif column == 'GRID':
               row_dict['GRID'] = grid_name  # todo: worry about spaces and quotes
            elif column == 'RADW':
               row_dict['RADW'] = radw
            elif column == 'SKIN':
               row_dict['SKIN'] = skin
            elif column == 'ANGLA':
               row_dict['ANGLA'] = angla
            elif column == 'ANGLV':
               row_dict['ANGLV'] = anglv
            elif column == 'LENGTH':
               # note: length units are those of trajectory length uom if length mode is MD and length_uom is None
               row_dict['LENGTH'] = length
            elif column == 'KH':
               row_dict['KH'] = kh
            elif column == 'DEPTH':
               row_dict['DEPTH'] = xyz[2]
            elif column == 'MD':
               row_dict['MD'] = md
            elif column == 'X':
               row_dict['X'] = xyz[0]
            elif column == 'Y':
               row_dict['Y'] = xyz[1]
            elif column == 'STAT':
               row_dict['STAT'] = stat
            elif column == 'PPERF':
               row_dict['PPERF'] = part_perf_fraction
            elif column == 'RADB':
               row_dict['RADB'] = radb
            elif column == 'WI':
               row_dict['WI'] = wi
            elif column == 'WBC':  # note: not a valid WELLSPEC column name
               row_dict['WBC'] = wbc
         df = df.append(row_dict, ignore_index = True)
         row_ci_list.append(ci)

      if add_as_properties:
         if isinstance(add_as_properties, list):
            for col in add_as_properties:
               assert col in extra_columns_list
            property_columns = add_as_properties
         else:
            property_columns = extra_columns_list
         self._add_df_properties(df, property_columns, row_ci_list = row_ci_list, length_uom = length_uom)

      return df

   def _add_df_properties(self, df, columns, row_ci_list = None, length_uom = None):
      # creates a property part for each named column, based on the dataframe values
      # column name used as the citation title
      # self must already exist as a part in the model
      # currently only handles single grid situations
      # todo: rewrite to add separate property objects for each grid references by the blocked well
      log.debug('_add_df_props: df:')
      log.debug(f'\n{df}')
      log.debug(f'columns: {columns}')
      assert len(self.grid_list) == 1
      if columns is None or len(columns) == 0 or len(df) == 0:
         return
      if row_ci_list is None:
         row_ci_list = np.arange(self.cell_count)
      assert len(row_ci_list) == len(df)
      if length_uom is None:
         length_uom = self.trajectory.md_uom
      extra_pc = rqp.PropertyCollection()
      extra_pc.set_support(support = self)
      ci_map = np.array(row_ci_list, dtype = int)
      for e in columns:
         extra = e.upper()
         if extra in ['GRID', 'STAT']:
            continue  # todo: other non-numeric columns may need to be added to this list
         pk = 'continuous'
         uom = 'Euc'
         if extra in ['ANGLA', 'ANGLV']:
            uom = 'dega'
            # neither azimuth nor dip are correct property kinds; todo: create local property kinds
            pk = 'azimuth' if extra == 'ANGLA' else 'inclination'
         elif extra in ['LENGTH', 'MD', 'X', 'Y', 'DEPTH', 'RADW']:
            if length_uom is None or length_uom == 'Euc':
               if extra in ['LENGTH', 'MD']:
                  uom = self.trajectory.md_uom
               elif extra in ['X', 'Y', 'RADW']:
                  uom = self.grid_list[0].xy_units()
               else:
                  uom = self.grid_list[0].z_units()
            else:
               uom = length_uom
            if extra == 'DEPTH':
               pk = 'depth'
            else:
               pk = 'length'
         elif extra == 'KH':
            uom = 'mD.' + length_uom
            pk = 'permeability length'
         elif extra == 'PPERF':
            uom = length_uom + '/' + length_uom
         else:
            uom = 'Euc'
         # 'SKIN': use defaults for now; todo: create local property kind for skin
         expanded = np.full(self.cell_count, np.NaN)
         expanded[ci_map] = df[extra]
         extra_pc.add_cached_array_to_imported_list(expanded,
                                                    'blocked well dataframe',
                                                    extra,
                                                    discrete = False,
                                                    uom = uom,
                                                    property_kind = pk,
                                                    local_property_kind_uuid = None,
                                                    facet_type = None,
                                                    facet = None,
                                                    realization = None,
                                                    indexable_element = 'cells',
                                                    count = 1)
      extra_pc.write_hdf5_for_imported_list()
      extra_pc.create_xml_for_imported_list_and_add_parts_to_model()

   def static_kh(self,
                 ntg_uuid = None,
                 perm_i_uuid = None,
                 perm_j_uuid = None,
                 perm_k_uuid = None,
                 satw_uuid = None,
                 sato_uuid = None,
                 satg_uuid = None,
                 region_uuid = None,
                 active_only = False,
                 min_k0 = None,
                 max_k0 = None,
                 k0_list = None,
                 min_length = None,
                 min_kh = None,
                 max_depth = None,
                 max_satw = None,
                 min_sato = None,
                 max_satg = None,
                 perforation_list = None,
                 region_list = None,
                 set_k_face_intervals_vertical = False,
                 anglv_ref = 'gravity',
                 angla_plane_ref = None,
                 length_mode = 'MD',
                 length_uom = None,
                 use_face_centres = False,
                 preferential_perforation = True):
      """Returns the total static K.H (permeability x height); length units are those of trajectory md_uom unless length_upm is set.

      note:
         see doc string for dataframe() method for argument descriptions; perm_i_uuid required
      """

      df = self.dataframe(i_col = 'I',
                          j_col = 'J',
                          k_col = 'K',
                          one_based = False,
                          extra_columns_list = ['KH'],
                          ntg_uuid = ntg_uuid,
                          perm_i_uuid = perm_i_uuid,
                          perm_j_uuid = perm_j_uuid,
                          perm_k_uuid = perm_k_uuid,
                          satw_uuid = satw_uuid,
                          sato_uuid = sato_uuid,
                          satg_uuid = satg_uuid,
                          region_uuid = region_uuid,
                          active_only = active_only,
                          min_k0 = min_k0,
                          max_k0 = max_k0,
                          k0_list = k0_list,
                          min_length = min_length,
                          min_kh = min_kh,
                          max_depth = max_depth,
                          max_satw = max_satw,
                          min_sato = min_sato,
                          max_satg = max_satg,
                          perforation_list = perforation_list,
                          region_list = region_list,
                          set_k_face_intervals_vertical = set_k_face_intervals_vertical,
                          anglv_ref = anglv_ref,
                          angla_plane_ref = angla_plane_ref,
                          length_mode = length_mode,
                          length_uom = length_uom,
                          use_face_centres = use_face_centres,
                          preferential_perforation = preferential_perforation)

      return sum(df['KH'])

   def write_wellspec(self,
                      wellspec_file,
                      well_name = None,
                      mode = 'a',
                      extra_columns_list = [],
                      ntg_uuid = None,
                      perm_i_uuid = None,
                      perm_j_uuid = None,
                      perm_k_uuid = None,
                      satw_uuid = None,
                      sato_uuid = None,
                      satg_uuid = None,
                      region_uuid = None,
                      radw = None,
                      skin = None,
                      stat = None,
                      active_only = False,
                      min_k0 = None,
                      max_k0 = None,
                      k0_list = None,
                      min_length = None,
                      min_kh = None,
                      max_depth = None,
                      max_satw = None,
                      min_sato = None,
                      max_satg = None,
                      perforation_list = None,
                      region_list = None,
                      set_k_face_intervals_vertical = False,
                      depth_inc_down = True,
                      anglv_ref = 'gravity',
                      angla_plane_ref = None,
                      length_mode = 'MD',
                      length_uom = None,
                      preferential_perforation = True,
                      space_instead_of_tab_separator = True,
                      align_columns = True,
                      preceeding_blank_lines = 0,
                      trailing_blank_lines = 0,
                      length_uom_comment = False,
                      write_nexus_units = True,
                      float_format = '5.3'):
      """Writes Nexus WELLSPEC keyword to an ascii file.

      returns:
         pandas DataFrame containing data that has been written to the wellspec file

      note:
         see doc string for dataframe() method for most of the argument descriptions;
         align_columns and float_format arguments are deprecated and no longer used
      """

      def tidy_well_name(well_name):
         nexus_friendly = ''
         previous_underscore = False
         for ch in well_name:
            if not 32 <= ord(ch) < 128 or ch in ' ,!*#':
               ch = '_'
            if not (previous_underscore and ch == '_'):
               nexus_friendly += ch
            previous_underscore = (ch == '_')
         if not nexus_friendly:
            well_name = 'WELL_X'
         return nexus_friendly

      def is_float_column(col_name):
         if col_name.upper() in ['ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y', 'SKIN', 'RADW', 'PPERF']:
            return True
         return False

      def is_int_column(col_name):
         if col_name.upper() in ['IW', 'JW', 'L']:
            return True
         return False

      assert wellspec_file, 'no output file specified to write WELLSPEC to'

      col_width_dict = {
         'IW': 4,
         'JW': 4,
         'L': 4,
         'ANGLA': 8,
         'ANGLV': 8,
         'LENGTH': 8,
         'KH': 10,
         'DEPTH': 10,
         'MD': 10,
         'X': 8,
         'Y': 12,
         'SKIN': 7,
         'RADW': 5,
         'PPERF': 5
      }

      if not well_name:
         if self.well_name:
            well_name = self.well_name
         elif self.root is not None:
            well_name = rqet.citation_title_for_node(self.root)
         elif self.wellbore_interpretation is not None:
            well_name = self.wellbore_interpretation.title
         elif self.trajectory is not None:
            well_name = self.trajectory.title
         else:
            log.warning('no well name identified for use in WELLSPEC')
            well_name = 'WELLNAME'
      well_name = tidy_well_name(well_name)

      df = self.dataframe(one_based = True,
                          extra_columns_list = extra_columns_list,
                          ntg_uuid = ntg_uuid,
                          perm_i_uuid = perm_i_uuid,
                          perm_j_uuid = perm_j_uuid,
                          perm_k_uuid = perm_k_uuid,
                          satw_uuid = satw_uuid,
                          sato_uuid = sato_uuid,
                          satg_uuid = satg_uuid,
                          region_uuid = region_uuid,
                          radw = radw,
                          skin = skin,
                          stat = stat,
                          active_only = active_only,
                          min_k0 = min_k0,
                          max_k0 = max_k0,
                          k0_list = k0_list,
                          min_length = min_length,
                          min_kh = min_kh,
                          max_depth = max_depth,
                          max_satw = max_satw,
                          min_sato = min_sato,
                          max_satg = max_satg,
                          perforation_list = perforation_list,
                          region_list = region_list,
                          depth_inc_down = depth_inc_down,
                          set_k_face_intervals_vertical = set_k_face_intervals_vertical,
                          anglv_ref = anglv_ref,
                          angla_plane_ref = angla_plane_ref,
                          length_mode = length_mode,
                          length_uom = length_uom,
                          preferential_perforation = preferential_perforation)

      sep = ' ' if space_instead_of_tab_separator else '\t'

      with open(wellspec_file, mode = mode) as fp:
         for _ in range(preceeding_blank_lines):
            fp.write('\n')
         if write_nexus_units:
            if length_uom == 'm':
               fp.write('METRIC\n\n')
            elif length_uom == 'ft':
               fp.write('ENGLISH\n\n')
         if length_uom_comment and self.trajectory is not None and ('LENGTH' in extra_columns_list or
                                                                    'MD' in extra_columns_list or
                                                                    'KH' in extra_columns_list):
            fp.write(f'! Length units along wellbore: {self.trajectory.md_uom if length_uom is None else length_uom}\n')
         fp.write('WELLSPEC ' + str(well_name) + '\n')
         for col_name in df.columns:
            if col_name in col_width_dict:
               width = col_width_dict[col_name]
            else:
               width = 10
            form = '{0:>' + str(width) + '}'
            fp.write(sep + form.format(col_name))
         fp.write('\n')
         for row_info in df.iterrows():
            row = row_info[1]
            for col_name in df.columns:
               try:
                  if col_name in col_width_dict:
                     width = col_width_dict[col_name]
                  else:
                     width = 10
                  if is_float_column(col_name):
                     form = '{0:>' + str(width) + '.3f}'
                     fp.write(sep + form.format(float(row[col_name])))
                  else:
                     form = '{0:>' + str(width) + '}'
                     if is_int_column(col_name):
                        fp.write(sep + form.format(int(row[col_name])))
                     else:
                        fp.write(sep + form.format(str(row[col_name])))
               except Exception:
                  fp.write(sep + str(row[col_name]))
            fp.write('\n')
         for _ in range(trailing_blank_lines):
            fp.write('\n')

      return df

   def kji0_marker(self, active_only = True):
      """Convenience method returning (k0, j0, i0), grid_uuid of first blocked interval."""

      cells, grids = self.cell_indices_and_grid_list()
      if cells is None or grids is None or len(grids) == 0:
         return None, None, None, None
      return cells[0], grids[0].uuid

   def xyz_marker(self, active_only = True):
      """Convenience method returning (x, y, z), crs_uuid of perforation in first blocked interval.

      notes:
         active_only argument not yet in use;
         returns None, None if no blocked interval found
      """

      cells, grids = self.cell_indices_and_grid_list()
      if cells is None or grids is None or len(grids) == 0:
         return None, None
      node_index = 0
      while node_index < self.node_count - 1 and self.grid_indices[node_index] == -1:
         node_index += 1
      if node_index >= self.node_count - 1:
         return None, None
      md = 0.5 * (self.node_mds[node_index] + self.node_mds[node_index + 1])
      xyz = self.trajectory.xyz_for_md(md)
      return xyz, rqet.uuid_for_part_root(self.trajectory.crs_root)

   def create_feature_and_interpretation(self, shared_interpretation = True):
      """Instantiate new empty WellboreFeature and WellboreInterpretation objects
      
      Uses the Blocked well citation title as the well name
      """
      if self.trajectory is not None:
         traj_interp_uuid = self.model.uuid(obj_type = 'WellboreInterpretation', related_uuid = self.trajectory.uuid)
         if traj_interp_uuid is not None:
            if shared_interpretation:
               self.wellbore_interpretation = rqo.WellboreInterpretation(parent_model = self.model,
                                                                         uuid = traj_interp_uuid)
            traj_feature_uuid = self.model.uuid(obj_type = 'WellboreFeature', related_uuid = traj_interp_uuid)
            if traj_feature_uuid is not None:
               self.wellbore_feature = rqo.WellboreFeature(parent_model = self.model, uuid = traj_feature_uuid)
      if self.wellbore_feature is None:
         self.wellbore_feature = rqo.WellboreFeature(parent_model = self.model, feature_name = self.trajectory.title)
         self.feature_to_be_written = True
      if self.wellbore_interpretation is None:
         self.wellbore_interpretation = rqo.WellboreInterpretation(parent_model = self.model,
                                                                   wellbore_feature = self.wellbore_feature)
         if self.trajectory.wellbore_interpretation is None and shared_interpretation:
            self.trajectory.wellbore_interpretation = self.wellbore_interpretation
         self.interpretation_to_be_written = True

   def create_md_datum_and_trajectory(self,
                                      grid,
                                      trajectory_mds,
                                      trajectory_points,
                                      length_uom,
                                      well_name,
                                      set_depth_zero = False,
                                      set_tangent_vectors = False,
                                      create_feature_and_interp = True):
      """Creates an Md Datum object and a (simulation) Trajectory object for this blocked well.

      note:
         not usually called directly; used by import methods
      """

      # create md datum node for synthetic trajectory, using crs for grid
      datum_location = trajectory_points[0].copy()
      if set_depth_zero:
         datum_location[2] = 0.0
      datum = MdDatum(self.model, crs_uuid = grid.crs_uuid, location = datum_location, md_reference = 'mean sea level')

      # create synthetic trajectory object, using crs for grid
      trajectory_mds_array = np.array(trajectory_mds)
      trajectory_xyz_array = np.array(trajectory_points)
      trajectory_df = pd.DataFrame({
         'MD': trajectory_mds_array,
         'X': trajectory_xyz_array[..., 0],
         'Y': trajectory_xyz_array[..., 1],
         'Z': trajectory_xyz_array[..., 2]
      })
      self.trajectory = Trajectory(self.model,
                                   md_datum = datum,
                                   data_frame = trajectory_df,
                                   length_uom = length_uom,
                                   well_name = well_name,
                                   set_tangent_vectors = set_tangent_vectors)
      self.trajectory_to_be_written = True

      if create_feature_and_interp:
         self.create_feature_and_interpretation()

   def create_xml(self,
                  ext_uuid = None,
                  create_for_trajectory_if_needed = True,
                  add_as_part = True,
                  add_relationships = True,
                  title = None,
                  originator = None):
      """Create a blocked wellbore representation node from this BlockedWell object, optionally add as part.

      note:
         trajectory xml node must be in place before calling this function;
         witsml log reference, interval stratigraphic units, and cell fluid phase units not yet supported

      :meta common:
      """

      assert self.trajectory is not None, 'trajectory object missing'

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      if title:
         self.title = title
      if not self.title:
         self.title = 'blocked well'

      if self.feature_to_be_written:
         if self.wellbore_feature is None:
            self.create_feature_and_interpretation()
         self.wellbore_feature.create_xml(add_as_part = add_as_part, originator = originator)
      if self.interpretation_to_be_written:
         if self.wellbore_interpretation is None:
            self.create_feature_and_interpretation()
         self.wellbore_interpretation.create_xml(add_as_part = add_as_part,
                                                 title_suffix = None,
                                                 add_relationships = add_relationships,
                                                 originator = originator)

      if create_for_trajectory_if_needed and self.trajectory_to_be_written and self.trajectory.root is None:
         md_datum_root = self.trajectory.md_datum.create_xml(add_as_part = add_as_part,
                                                             add_relationships = add_relationships,
                                                             title = str(self.title),
                                                             originator = originator)
         self.trajectory.create_xml(ext_uuid,
                                    md_datum_root = md_datum_root,
                                    add_as_part = add_as_part,
                                    add_relationships = add_relationships,
                                    title = title,
                                    originator = originator)

      assert self.trajectory.root is not None, 'trajectory xml not established'

      bw_node = super().create_xml(title = title, originator = originator, add_as_part = False)

      # wellbore frame elements

      nc_node = rqet.SubElement(bw_node, ns['resqml2'] + 'NodeCount')
      nc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      nc_node.text = str(self.node_count)

      mds_node = rqet.SubElement(bw_node, ns['resqml2'] + 'NodeMd')
      mds_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
      mds_node.text = rqet.null_xml_text

      mds_values_node = rqet.SubElement(mds_node, ns['resqml2'] + 'Values')
      mds_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      mds_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'NodeMd', root = mds_values_node)

      traj_root = self.trajectory.root
      self.model.create_ref_node('Trajectory',
                                 rqet.find_nested_tags_text(traj_root, ['Citation', 'Title']),
                                 bu.uuid_from_string(traj_root.attrib['uuid']),
                                 content_type = 'obj_WellboreTrajectoryRepresentation',
                                 root = bw_node)

      # remaining blocked wellbore elements

      cc_node = rqet.SubElement(bw_node, ns['resqml2'] + 'CellCount')
      cc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
      cc_node.text = str(self.cell_count)

      cis_node = rqet.SubElement(bw_node, ns['resqml2'] + 'CellIndices')
      cis_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
      cis_node.text = rqet.null_xml_text

      cnull_node = rqet.SubElement(cis_node, ns['resqml2'] + 'NullValue')
      cnull_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
      cnull_node.text = str(self.cellind_null)

      cis_values_node = rqet.SubElement(cis_node, ns['resqml2'] + 'Values')
      cis_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      cis_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'CellIndices', root = cis_values_node)

      gis_node = rqet.SubElement(bw_node, ns['resqml2'] + 'GridIndices')
      gis_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
      gis_node.text = rqet.null_xml_text

      gnull_node = rqet.SubElement(gis_node, ns['resqml2'] + 'NullValue')
      gnull_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
      gnull_node.text = str(self.gridind_null)

      gis_values_node = rqet.SubElement(gis_node, ns['resqml2'] + 'Values')
      gis_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      gis_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'GridIndices', root = gis_values_node)

      fis_node = rqet.SubElement(bw_node, ns['resqml2'] + 'LocalFacePairPerCellIndices')
      fis_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
      fis_node.text = rqet.null_xml_text

      fnull_node = rqet.SubElement(fis_node, ns['resqml2'] + 'NullValue')
      fnull_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
      fnull_node.text = str(self.facepair_null)

      fis_values_node = rqet.SubElement(fis_node, ns['resqml2'] + 'Values')
      fis_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
      fis_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'LocalFacePairPerCellIndices', root = fis_values_node)

      for grid in self.grid_list:

         grid_root = grid.root
         self.model.create_ref_node('Grid',
                                    rqet.find_nested_tags_text(grid_root, ['Citation', 'Title']),
                                    bu.uuid_from_string(grid_root.attrib['uuid']),
                                    content_type = 'obj_IjkGridRepresentation',
                                    root = bw_node)

      interp_root = None
      if self.wellbore_interpretation is not None:
         interp_root = self.wellbore_interpretation.root
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                    bu.uuid_from_string(interp_root.attrib['uuid']),
                                    content_type = 'obj_WellboreInterpretation',
                                    root = bw_node)

      if add_as_part:
         self.model.add_part('obj_BlockedWellboreRepresentation', self.uuid, bw_node)
         if add_relationships:
            self.model.create_reciprocal_relationship(bw_node, 'destinationObject', self.trajectory.root,
                                                      'sourceObject')

            for grid in self.grid_list:
               self.model.create_reciprocal_relationship(bw_node, 'destinationObject', grid.root, 'sourceObject')
            if interp_root is not None:
               self.model.create_reciprocal_relationship(bw_node, 'destinationObject', interp_root, 'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(bw_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

      return bw_node

   def write_hdf5(self, file_name = None, mode = 'a', create_for_trajectory_if_needed = True):
      """Create or append to an hdf5 file, writing datasets for the measured depths, grid, cell & face indices.

      :meta common:
      """

      # NB: array data must all have been set up prior to calling this function

      if self.uuid is None:
         self.uuid = bu.new_uuid()

      h5_reg = rwh5.H5Register(self.model)

      if create_for_trajectory_if_needed and self.trajectory_to_be_written:
         self.trajectory.write_hdf5(file_name, mode = mode)
         mode = 'a'

      h5_reg.register_dataset(self.uuid, 'NodeMd', self.node_mds)
      h5_reg.register_dataset(self.uuid, 'CellIndices', self.cell_indices)  # could use int32?
      h5_reg.register_dataset(self.uuid, 'GridIndices', self.grid_indices)  # could use int32?
      # convert face index pairs from [axis, polarity] back to strange local face numbering
      mask = (self.face_pair_indices.flatten() == -1).reshape((-1, 2))  # 2nd axis is (axis, polarity)
      masked_face_indices = np.where(mask, 0, self.face_pair_indices.reshape((-1, 2)))  # 2nd axis is (axis, polarity)
      # using flat array for raw_face_indices array
      # other resqml writing code might use an array with one int per entry point and one per exit point, with 2nd axis as (entry, exit)
      raw_face_indices = np.where(mask[:, 0], -1, self.face_index_map[masked_face_indices[:, 0],
                                                                      masked_face_indices[:, 1]].flatten()).reshape(-1)

      h5_reg.register_dataset(self.uuid, 'LocalFacePairPerCellIndices', raw_face_indices)  # could use uint8?

      h5_reg.write(file = file_name, mode = mode)


class WellboreMarkerFrame(BaseResqpy):
   """Class to handle RESQML WellBoreMarkerFrameRepresentation objects.

   note:
      measured depth data must be in same crs as those for the related trajectory
   """

   resqml_type = 'WellboreMarkerFrameRepresentation'

   def __init__(self,
                parent_model,
                wellbore_marker_frame_root = None,
                uuid = None,
                trajectory = None,
                title = None,
                originator = None,
                extra_metadata = None):
      """Creates a new wellbore marker object and optionally loads it from xml, or trajectory, or Nexus wellspec file.

      arguments:
         parent_model (model.Model object): the model which the new blocked well belongs to
         wellbore_marker_root (DEPRECATED): the root node of an xml tree representing the wellbore marker;
         trajectory (optional, Trajectory object): the trajectory of the well, to be intersected with the grid;
            not used if wellbore_marker_root is not None;
         title (str, optional): the citation title to use for a new wellbore marker frame;
            ignored if uuid or wellbore_marker_frame_root is not None
         originator (str, optional): the name of the person creating the wellbore marker frame, defaults to login id;
            ignored if uuid or wellbore_marker_frame_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore marker frame;
            ignored if uuid or wellbore_marker_frame_root is not None

      returns:
         the newly created wellbore framework marker object
      """

      self.trajectory = None
      self.node_count = None  # number of measured depth nodes, each being for a marker
      self.node_mds = None  # node_count measured depths (in same units and datum as trajectory) of markers
      self.wellbore_marker_list = [
      ]  # list of markers, each: (marker UUID, geologic boundary, marker citation title, interp. object)
      if self.trajectory is not None:
         self.trajectory = trajectory

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = wellbore_marker_frame_root)

   def get_trajectory_obj(self, trajectory_uuid):
      """Returns a trajectory object.

      arguments:
         trajectory_uuid (string or uuid.UUID): the uuid of the trajectory for which a Trajectory object is required

      returns:
         well.Trajectory object

      note:
         this method is not usually called directly
      """

      if trajectory_uuid is None:
         log.error('no trajectory was found')
         return None
      else:
         # create new trajectory object
         trajectory_root_node = self.model.root_for_uuid(trajectory_uuid)
         assert trajectory_root_node is not None, 'referenced wellbore trajectory missing from model'
         return Trajectory(self.model, trajectory_root = trajectory_root_node)

   def get_interpretation_obj(self, interpretation_uuid, interp_type = None):
      """Creates an interpretation object; returns a horizon or fault interpretation object.

      arguments:
         interpretation_uiud (string or uuid.UUID): the uuid of the required interpretation object
         interp_type (string, optional): 'HorizonInterpretation' or 'FaultInterpretation' (optionally
            prefixed with `obj_`); if None, the type is inferred from the xml for the given uuid

      returns:
         organization.HorizonInterpretation or organization.FaultInterpretation object

      note:
         this method is not usually called directly
      """

      assert interpretation_uuid is not None, 'interpretation uuid argument missing'

      interpretation_root_node = self.model.root_for_uuid(interpretation_uuid)

      if not interp_type:
         interp_type = rqet.node_type(interpretation_root_node)

      if not interp_type.startswith('obj_'):
         interp_type = 'obj_' + interp_type

      if interp_type == 'obj_HorizonInterpretation':
         # create new horizon interpretation object
         return rqo.HorizonInterpretation(self.model, root_node = interpretation_root_node)

      elif interp_type == 'obj_FaultInterpretation':
         # create new fault interpretation object
         return rqo.FaultInterpretation(self.model, root_node = interpretation_root_node)

      elif interp_type == 'obj_GeobodyInterpretation':
         # create new geobody interpretation object
         return rqo.GeobodyInterpretation(self.model, root_node = interpretation_root_node)
      else:
         # No interpretation for the marker
         return None
         # log.error('interpretation type not recognized: ' + str(interp_type))

   def _load_from_xml(self):
      """Loads the wellbore marker frame object from an xml node (and associated hdf5 data).

      note:
         this method is not usually called directly
      """

      wellbore_marker_frame_root = self.root
      assert wellbore_marker_frame_root is not None

      if self.trajectory is None:
         self.trajectory = self.get_trajectory_obj(
            rqet.find_nested_tags_text(wellbore_marker_frame_root, ['Trajectory', 'UUID']))

      # list of Wellbore markers, each: (marker UUID, geologic boundary, marker citation title, interp. object)
      self.wellbore_marker_list = []
      for tag in rqet.list_of_tag(wellbore_marker_frame_root, 'WellboreMarker'):
         interp_tag = rqet.content_type(rqet.find_nested_tags_text(tag, ['Interpretation', 'ContentType']))
         if interp_tag is not None:
            interp_obj = self.get_interpretation_obj(rqet.find_nested_tags_text(tag, ['Interpretation', 'UUID']),
                                                     interp_tag)
         else:
            interp_obj = None
         self.wellbore_marker_list.append(
            (str(rqet.uuid_for_part_root(tag)), rqet.find_tag_text(tag, 'GeologicBoundaryKind'),
             rqet.find_nested_tags_text(tag, ['Citation', 'Title']), interp_obj))

      self.node_count = rqet.find_tag_int(wellbore_marker_frame_root, 'NodeCount')
      load_hdf5_array(self, rqet.find_tag(wellbore_marker_frame_root, 'NodeMd'), "node_mds", tag = 'Values')
      if self.node_count != len(self.node_mds):
         log.error('node count does not match hdf5 array')

      if len(self.wellbore_marker_list) != self.node_count:
         log.error('wellbore marker list does not contain correct node count')

   def dataframe(self):
      """Returns a pandas dataframe with columns X, Y, Z, MD, Type, Surface, Well."""

      # todo: handle fractures and geobody boundaries as well as horizons and faults

      xyz = np.empty((self.node_count, 3))
      type_list = []
      surface_list = []
      well_list = []

      for i in range(self.node_count):
         _, boundary_kind, title, interp = self.wellbore_marker_list[i]
         if interp:
            if boundary_kind == 'horizon':
               feature_name = rqo.GeneticBoundaryFeature(self.model, root_node = interp.feature_root).feature_name
            elif boundary_kind == 'fault':
               feature_name = rqo.TectonicBoundaryFeature(self.model, root_node = interp.feature_root).feature_name
            elif boundary_kind == 'geobody':
               feature_name = rqo.GeneticBoundaryFeature(self.model, root_node = interp.feature_root).feature_name
            else:
               assert False, 'unexpected boundary kind'
         else:
            feature_name = title
         boundary_kind = boundary_kind[0].upper() + boundary_kind[1:]
         feature_name = '"' + feature_name + '"'
         xyz[i] = self.trajectory.xyz_for_md(self.node_mds[i])
         type_list.append(boundary_kind)
         surface_list.append(feature_name)
         if self.trajectory.wellbore_interpretation is None:
            well_name = '"' + self.trajectory.title + '"'  # todo: trace through wellbore interp to wellbore feature name
         else:
            well_name = '"' + self.trajectory.wellbore_interpretation.title + '"'  # use wellbore_interpretation title instead, RMS exports have feature_name as "Wellbore feature"
            # well_name = '"' + self.trajectory.wellbore_interpretation.wellbore_feature.feature_name + '"'
         well_list.append(well_name)

      return pd.DataFrame({
         'X': xyz[:, 0],
         'Y': xyz[:, 1],
         'Z': xyz[:, 2],
         'MD': self.node_mds,
         'Type': type_list,
         'Surface': surface_list,
         'Well': well_list
      })

   def create_xml(self,
                  ext_uuid = None,
                  add_as_part = True,
                  add_relationships = True,
                  wellbore_marker_list = None,
                  title = 'wellbore marker framework',
                  originator = None):

      assert type(add_as_part) is bool

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      wbm_node = super().create_xml(originator = originator, add_as_part = False)

      nodeCount = rqet.SubElement(wbm_node, ns['resqml2'] + 'NodeCount')
      nodeCount.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      nodeCount.text = str(self.node_count)

      nodeMd = rqet.SubElement(wbm_node, ns['resqml2'] + 'NodeMd')
      nodeMd.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
      nodeMd.text = rqet.null_xml_text

      md_values_node = rqet.SubElement(nodeMd, ns['resqml2'] + 'Values')
      md_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
      md_values_node.text = rqet.null_xml_text

      self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'mds', root = md_values_node)

      if self.trajectory is not None:
         traj_root = self.trajectory.root
         self.model.create_ref_node('Trajectory',
                                    rqet.find_tag(rqet.find_tag(traj_root, 'Citation'), 'Title').text,
                                    bu.uuid_from_string(traj_root.attrib['uuid']),
                                    content_type = 'obj_WellboreTrajectoryRepresentation',
                                    root = wbm_node)
      else:
         log.error('trajectory object is missing and must be included')

      # fill wellbore marker
      for marker in self.wellbore_marker_list:

         wbm_node_obj = self.model.new_obj_node('WellboreMarker', is_top_lvl_obj = False)
         wbm_node_obj.set('uuid', marker[0])
         wbm_node.append(wbm_node_obj)
         wbm_gb_node = rqet.SubElement(wbm_node_obj, ns['resqml2'] + 'GeologicBoundaryKind')
         wbm_gb_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
         wbm_gb_node.text = str(marker[1])

         interp = marker[3]
         if interp is not None:
            interp_root = marker[3].root
            if 'HorizonInterpretation' in str(type(marker[3])):
               self.model.create_ref_node('Interpretation',
                                          rqet.find_tag(rqet.find_tag(interp_root, 'Citation'), 'Title').text,
                                          bu.uuid_from_string(interp_root.attrib['uuid']),
                                          content_type = 'obj_HorizonInterpretation',
                                          root = wbm_node_obj)

            elif 'FaultInterpretation' in str(type(marker[3])):
               self.model.create_ref_node('Interpretation',
                                          rqet.find_tag(rqet.find_tag(interp_root, 'Citation'), 'Title').text,
                                          bu.uuid_from_string(interp_root.attrib['uuid']),
                                          content_type = 'obj_FaultInterpretation',
                                          root = wbm_node_obj)

      # add as part
      if add_as_part:
         self.model.add_part('obj_WellboreMarkerFrameRepresentation', self.uuid, wbm_node)

         if add_relationships:
            self.model.create_reciprocal_relationship(wbm_node, 'destinationObject', self.trajectory.root,
                                                      'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(wbm_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

            for marker in self.wellbore_marker_list:
               self.model.create_reciprocal_relationship(wbm_node, 'destinationObject', marker[3].root, 'sourceObject')

      return wbm_node

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Writes the hdf5 array associated with this object (the measured depth data).

      arguments:
         file_name (string): the name of the hdf5 file, or None, in which case the model's default will be used
         mode (string, default 'a'): the write mode for the hdf5, either 'w' or 'a'
      """

      h5_reg = rwh5.H5Register(self.model)
      h5_reg.register_dataset(self.uuid, 'Mds', self.node_mds)
      h5_reg.write(file = file_name, mode = mode)

   def find_marker_from_interp(self, interpetation_obj = None, uuid = None):
      """Find wellbore marker by interpretation; can pass object or uuid.

      arguments:
         interpretation_obj (organize.HorizonInterpretation or organize.FaultInterpretation object, optional):
            if present, the first (smallest md) marker relating to this interpretation object is returned
         uuid (string or uuid.UUID): if present, the uuid of the interpretation object of interest; ignored if
            interpretation_obj is not None

      returns:
         tuple, list of tuples or None; tuple is (marker UUID, geologic boundary, marker citation title, interp. object)

      note:
         if no arguments are passed, then a list of wellbore markers is returned;
         if no marker is found for the interpretation object, None is returned
      """

      if interpetation_obj is None and uuid is None:
         return self.wellbore_marker_list

      if interpetation_obj is not None:
         uuid = interpetation_obj.uuid

      for marker in self.wellbore_marker_list:
         if bu.matching_uuids(marker[3].uuid, uuid):
            return marker

      return None

   def get_marker_count(self):
      '''Retruns number of wellbore markers'''

      return len(self.wellbore_marker_list)

   def find_marker_from_index(self, idx):
      '''Returns wellbore marker by index'''

      return self.wellbore_marker_list[idx - 1]


def add_las_to_trajectory(las: lasio.LASFile, trajectory, realization = None, check_well_name = False):
   """Creates a WellLogCollection and WellboreFrame from a LAS file.

   Note:
      In this current implementation, the first curve in the las object must be
      Measured Depths, not e.g. TVDSS.

   Arguments:
      las: an lasio.LASFile object
      trajectory: an instance of :class:`resqpy.well.Trajectory` .
      realization (integer): if present, the single realisation (within an ensemble)
         that this collection is for
      check_well_name (bool): if True, raise warning if LAS well name does not match
         existing wellborefeature citation title

   Returns:
      collection, well_frame: instances of :class:`resqpy.property.WellLogCollection`
         and :class:`resqpy.well.WellboreFrame`

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

   well_frame = WellboreFrame(
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


def add_logs_from_cellio(blockedwell, cellio):
   """Creates a WellIntervalPropertyCollection for a given BlockedWell, using a given cell I/O file.

   Arguments:
      blockedwell: a resqml blockedwell object
      cellio: an ascii file exported from RMS containing blocked well geometry and logs. Must contain columns i_index, j_index and k_index, plus additional columns for logs to be imported.
   """
   # Get the initial variables from the blocked well
   assert isinstance(blockedwell, BlockedWell), 'Not a blocked wellbore object'
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
   Arguments:
      line: a string from a cell I/O file, containing the column (log) name, type and categorical information
      model: the model to add the StringTableLookup to
   Returns:
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
   for existing in model.parts_list_of_type('obj_StringTableLookup'):
      table = rqp.StringLookup(parent_model = model, root_node = model.root_for_part(existing))
      if table.title == title:
         if table.str_dict == lookup_dict:
            return table.uuid  # If the exact table exists, reuse it by returning the uuid

   # If no matching StringLookupTable exists, make a new one and return the uuid
   lookup = rqp.StringLookup(parent_model = model, int_to_str_dict = lookup_dict, title = title)
   lookup.create_xml(add_as_part = True)
   return lookup.uuid


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
      tuple of lists of objects: (feature_list, interpretation_list, trajectory_list, md_datum_list),

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
      crs_root = model.crs_root
   else:
      crs_root = model.root_for_uuid(crs_uuid)
   assert crs_root is not None, 'coordinate reference system not found when trying to add wells'

   try:
      df = pd.read_csv(trajectory_file, comment = comment_character, delim_whitespace = space_separated_instead_of_csv)
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

   feature_list = interpretation_list = trajectory_list = md_datum_list = []

   for well_name in unique_wells:

      log.debug('importing well: ' + str(well_name))
      # create single well data frame (assumes measured depths increasing)
      well_df = df[df[well_col] == well_name]
      # create a measured depth datum for the well and add as part
      first_row = well_df.iloc[0]
      if first_row[md_col] == 0.0:
         md_datum = MdDatum(model,
                            crs_root = crs_root,
                            location = (first_row[x_col], first_row[y_col], first_row[z_col]))
      else:
         md_datum = MdDatum(model, crs_root = crs_root,
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
      trajectory = Trajectory(model,
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
      elif isinstance(well_object, Trajectory):
         obj_type = 'WellboreTrajectoryRepresentation'
         traj_root = well_object.root
      elif isinstance(well_object, rqo.WellboreFeature):
         obj_type = 'WellboreFeature'
      elif isinstance(well_object, rqo.WellboreInterpretation):
         obj_type = 'WellboreInterpretation'
      elif isinstance(well_object, BlockedWell):
         obj_type = 'BlockedWellboreRepresentation'
         if well_object.trajectory is not None:
            traj_root = well_object.trajectory.root
      elif isinstance(well_object, WellboreMarkerFrame):  # note: trajectory might be None
         obj_type = 'WellboreMarkerFrameRepresentation'
         if well_object.trajectory is not None:
            traj_root = well_object.trajectory.root
      elif isinstance(well_object, WellboreFrame):  # note: trajectory might be None
         obj_type = 'WellboreFrameRepresentation'
         if well_object.trajectory is not None:
            traj_root = well_object.trajectory.root
      elif isinstance(well_object, DeviationSurvey):
         obj_type = 'DeviationSurveyRepresentation'
      elif isinstance(well_object, MdDatum):
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

   return rqet.citation_title_for_node(best_root_for_object(well_object, model = model))


def add_blocked_wells_from_wellspec(model, grid, wellspec_file):
   """Add a blocked well for each well in a Nexus WELLSPEC file.

   arguments:
      model (model.Model object): model to which blocked wells are added
      grid (grid.Grid object): grid against which wellspec data will be interpreted
      wellspec_file (string): path of ascii file holding Nexus WELLSPEC keyword and data

   returns:
      int: count of number of blocked wells created

   notes:
      this function appends to the hdf5 file and creates xml for the blocked wells (but does not store epc);
      'simulation' trajectory and measured depth datum objects will also be created
   """

   well_list_dict = wsk.load_wellspecs(wellspec_file, column_list = None)

   count = 0
   for well in well_list_dict:
      log.info('processing well: ' + str(well))
      bw = BlockedWell(model,
                       grid = grid,
                       wellspec_file = wellspec_file,
                       well_name = well,
                       check_grid_name = True,
                       use_face_centres = True)
      if not bw.node_count:  # failed to load from wellspec, eg. because of no perforations in grid
         log.warning('no wellspec data loaded for well: ' + str(well))
         continue
      bw.write_hdf5(model.h5_file_name(), mode = 'a', create_for_trajectory_if_needed = True)
      bw.create_xml(model.h5_uuid(), title = well)
      count += 1

   log.info(f'{count} blocked wells created based on wellspec file: {wellspec_file}')


def extract_xyz(xyz_node):
   """Extracts an x,y,z coordinate from a solitary point xml node.

      argument:
         xyz_node: the xml node representing the solitary point (in 3D space)

      returns:
         triple float: (x, y, z) coordinates as a tuple
   """

   if xyz_node is None:
      return None
   xyz = np.zeros(3)
   for axis in range(3):
      xyz[axis] = rqet.find_tag_float(xyz_node, 'Coordinate' + str(axis + 1), must_exist = True)
   return tuple(xyz)


def well_names_in_cellio_file(cellio_file):
   """Returns a list of well names as found in the RMS blocked well export cell I/O file."""

   well_list = []
   with open(cellio_file, 'r') as fp:
      while True:
         kf.skip_blank_lines_and_comments(fp)
         line = fp.readline()  # file format version number?
         if line == '':
            break  # end of file
         fp.readline()  # 'Undefined'
         words = fp.readline().split()
         assert len(words), 'missing header info (well name) in cell I/O file'
         well_list.append(words[0])
         while not kf.blank_line(fp):
            fp.readline()  # skip to block of data for next well
   return well_list


# 'private' functions


def load_hdf5_array(object, node, array_attribute, tag = 'Values', dtype = 'float', model = None):
   """Loads the property array data as an attribute of object, from the hdf5 referenced in xml node.

      :meta private:
   """

   assert (rqet.node_type(node) in ['DoubleHdf5Array', 'IntegerHdf5Array', 'Point3dHdf5Array'])
   if model is None:
      model = object.model
   h5_key_pair = model.h5_uuid_and_path_for_node(node, tag = tag)
   if h5_key_pair is None:
      return None
   return model.h5_array_element(h5_key_pair,
                                 index = None,
                                 cache_array = True,
                                 dtype = dtype,
                                 object = object,
                                 array_attribute = array_attribute)


def find_entry_and_exit(cp, entry_vector, exit_vector, well_name):
   """Returns (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz).

      :meta private:
   """

   cell_centre = np.mean(cp, axis = (0, 1, 2))
   face_triangles = gf.triangles_for_cell_faces(cp).reshape(-1, 3, 3)  # flattened first index 4 values per face
   entry_points = intersect.line_triangles_intersects(cell_centre, entry_vector, face_triangles, line_segment = True)
   entry_axis = entry_polarity = entry_xyz = exit_xyz = None
   for t in range(24):
      if not np.any(np.isnan(entry_points[t])):
         entry_xyz = entry_points[t]
         entry_axis = t // 8
         entry_polarity = (t - 8 * entry_axis) // 4
         break
   assert entry_axis is not None, 'failed to find entry face for a perforation in well ' + str(well_name)
   exit_points = intersect.line_triangles_intersects(cell_centre, exit_vector, face_triangles, line_segment = True)
   exit_axis = exit_polarity = None
   for t in range(24):
      if not np.any(np.isnan(exit_points[t])):
         exit_xyz = entry_points[t]
         exit_axis = t // 8
         exit_polarity = (t - 8 * exit_axis) // 4
         break
   assert exit_axis is not None, 'failed to find exit face for a perforation in well ' + str(well_name)

   return (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz)


def _as_optional_array(arr):
   """If not None, cast as numpy array.

   Casting directly to an array can be problematic:
   np.array(None) creates an unsized array, which is potentially confusing.
   """
   if arr is None:
      return None
   else:
      return np.array(arr)


def _pl(i, e = False):
   return '' if i == 1 else 'es' if e else 's'
