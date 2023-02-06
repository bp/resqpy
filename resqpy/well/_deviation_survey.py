"""DeviationSurvey class."""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import pandas as pd

import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.weights_and_measures as bwam
import resqpy.well
import resqpy.well._md_datum as rqmdd
import resqpy.well.well_utils as rqwu
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


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
                 represented_interp = None,
                 md_datum = None,
                 md_uom = 'm',
                 angle_uom = 'dega',
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

        arguments:
           parent_model (model.Model): the model which the new survey belongs to
           uuid (uuid.UUID): If given, loads from disk. Else, creates new.
           title (str): Citation title
           represented_interp (wellbore interpretation): if present, is noted as the wellbore
              interpretation object which this deviation survey relates to
           md_datum (MdDatum): the datum that the depths for this survey are measured from
           md_uom (string, default 'm'): a resqml length unit of measure applicable to the
              measured depths; should be 'm' or 'ft'
           angle_uom (string): a resqml angle unit; should be 'dega' or 'rad'
           measured_depths (np.array): 1d array
           azimuths (np.array): 1d array
           inclinations (np.array): 1d array
           station_count (int): length of measured_depths, azimuths & inclinations
           first_station (tuple): (x, y, z) of first point in survey, in crs for md datum
           is_final (bool): whether survey is a finalised deviation survey
           originator (str): name of author
           extra_metadata (dict, optional): extra metadata key, value pairs

        returns:
           DeviationSurvey

        notes:
           this method does not create an xml node, nor write hdf5 arrays

        :meta common:
        """

        self.is_final = is_final
        self.md_uom = bwam.rq_length_unit(md_uom)

        self.angles_in_degrees = angle_uom.strip().lower().startswith('deg')
        """boolean: True for degrees, False for radians (nothing else supported). Should be 'dega' or 'rad'"""

        # Array data
        self.measured_depths = rqwu._as_optional_array(measured_depths)
        self.azimuths = rqwu._as_optional_array(azimuths)
        self.inclinations = rqwu._as_optional_array(inclinations)

        if station_count is None and measured_depths is not None:
            station_count = len(measured_depths)
        self.station_count = station_count
        self.first_station = first_station

        # Referenced objects
        self.md_datum = md_datum  # md datum is an object in its own right, with a related crs!
        self.wellbore_interpretation = represented_interp

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

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
                        angle_uom = 'dega'):
        """Load MD, aximuth & inclination data from a pandas data frame.

        arguments:
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
           angle_uom (string, default 'dega'): a resqml angle unit of measure applicable to both
              the azimuth and inclination data

        returns:
           DeviationSurvey

        note:
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
                        angle_uom = 'dega',
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
           angle_uom (string, default 'dega'): a resqml angle unit of measure applicable to both
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
           [bool]: True if successful
        """

        # Get node from self.uuid
        node = self.root
        assert node is not None

        # Load XML data
        self.md_uom = rqet.length_units_from_node(rqet.find_tag(node, 'MdUom', must_exist = True))
        self.angle_uom = rqet.find_tag_text(node, 'AngleUom', must_exist = True)
        self.station_count = rqet.find_tag_int(node, 'StationCount', must_exist = True)
        self.first_station = rqwu.extract_xyz(rqet.find_tag(node, 'FirstStationLocation', must_exist = True))
        self.is_final = rqet.find_tag_bool(node, 'IsFinal')

        # Load HDF5 data
        mds_node = rqet.find_tag(node, 'Mds', must_exist = True)
        rqwu.load_hdf5_array(self, mds_node, 'measured_depths')
        azimuths_node = rqet.find_tag(node, 'Azimuths', must_exist = True)
        rqwu.load_hdf5_array(self, azimuths_node, 'azimuths')
        inclinations_node = rqet.find_tag(node, 'Inclinations', must_exist = True)
        rqwu.load_hdf5_array(self, inclinations_node, 'inclinations')

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

        md_datum_root = self.__get_md_datum_root(md_datum_root = md_datum_root, md_datum_xyz = md_datum_xyz)

        # Create root node, write citation block
        ds_node = super().create_xml(title = title, originator = originator, add_as_part = False)

        mds_values_node, azimuths_values_node, inclinations_values_node = self.__add_sub_elements_to_root_node(
            ds_node = ds_node)

        self.model.create_md_datum_reference(md_datum_root, root = ds_node)
        self.model.create_solitary_point3d('FirstStationLocation', ds_node, self.first_station)
        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Mds', root = mds_values_node)
        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Azimuths', root = azimuths_values_node)
        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'Inclinations', root = inclinations_values_node)
        interp_root = self.__create_wellbore_interpretation_ref_node(ds_node = ds_node)
        self.__add_as_part_and_add_relationships(ds_node = ds_node,
                                                 md_datum_root = md_datum_root,
                                                 interp_root = interp_root,
                                                 ext_uuid = ext_uuid,
                                                 add_as_part = add_as_part,
                                                 add_relationships = add_relationships)

        return ds_node

    def __get_md_datum_root(self, md_datum_root, md_datum_xyz):
        """ Ensures that the root node for the MdDatum object that the DeviationSurvey depths are based on exists.

        If not, a root node will be created and returned.
        """

        if md_datum_root is None:
            if self.md_datum is None:
                if md_datum_xyz is None:
                    raise ValueError("Must provide a MD Datum for the DeviationSurvey")
                self.md_datum = rqmdd.MdDatum(self.model, location = md_datum_xyz)
            if self.md_datum.root is None:
                md_datum_root = self.md_datum.create_xml()
            else:
                md_datum_root = self.md_datum.root
        assert md_datum_root is not None
        return md_datum_root

    def __add_sub_elements_to_root_node(self, ds_node):
        """Appends sub-elements to the DeviationSurvey object's root node."""

        if_node = rqet.SubElement(ds_node, ns['resqml2'] + 'IsFinal')
        if_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
        if_node.text = str(self.is_final).lower()

        sc_node = rqet.SubElement(ds_node, ns['resqml2'] + 'StationCount')
        sc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        sc_node.text = str(self.station_count)

        md_uom = rqet.SubElement(ds_node, ns['resqml2'] + 'MdUom')
        md_uom.set(ns['xsi'] + 'type', ns['eml'] + 'LengthUom')
        md_uom.text = bwam.rq_length_unit(self.md_uom)

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

        azimuths = rqet.SubElement(ds_node, ns['resqml2'] + 'Azimuths')
        azimuths.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
        azimuths.text = rqet.null_xml_text

        azimuths_values_node = rqet.SubElement(azimuths, ns['resqml2'] + 'Values')
        azimuths_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
        azimuths_values_node.text = rqet.null_xml_text

        inclinations = rqet.SubElement(ds_node, ns['resqml2'] + 'Inclinations')
        inclinations.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
        inclinations.text = rqet.null_xml_text

        inclinations_values_node = rqet.SubElement(inclinations, ns['resqml2'] + 'Values')
        inclinations_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
        inclinations_values_node.text = rqet.null_xml_text

        return mds_values_node, azimuths_values_node, inclinations_values_node

    def __create_wellbore_interpretation_ref_node(self, ds_node):
        """Create a reference node for the WellboreInterpretation object and add to the DeviationSurvey root node. """

        interp_root = None
        if self.wellbore_interpretation is not None:
            interp_root = self.wellbore_interpretation.root
            self.model.create_ref_node('RepresentedInterpretation',
                                       rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                       bu.uuid_from_string(interp_root.attrib['uuid']),
                                       content_type = 'obj_WellboreInterpretation',
                                       root = ds_node)
        return interp_root

    def __add_as_part_and_add_relationships(self, ds_node, md_datum_root, interp_root, ext_uuid, add_as_part,
                                            add_relationships):
        """Add the newly created DeviationSurvey object's root node as a part in the model and add reciprocal

        relationships.
        """

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

    def write_hdf5(self, file_name = None, mode = 'a'):
        """Create or append to an hdf5 file, writing datasets for the measured depths, azimuths, and inclinations."""

        # NB: array data must all have been set up prior to calling this function
        h5_reg = rwh5.H5Register(self.model)
        h5_reg.register_dataset(self.uuid, 'Mds', self.measured_depths, dtype = float)
        h5_reg.register_dataset(self.uuid, 'Azimuths', self.azimuths, dtype = float)
        h5_reg.register_dataset(self.uuid, 'Inclinations', self.inclinations, dtype = float)
        h5_reg.write(file = file_name, mode = mode)

    def _load_related_datum(self):
        """Return related MdDatum object from XML if present."""

        md_datum_uuid = bu.uuid_from_string(rqet.find_tag(rqet.find_tag(self.root, 'MdDatum'), 'UUID'))
        if md_datum_uuid is not None:
            md_datum_part = 'obj_MdDatum_' + str(md_datum_uuid) + '.xml'
            md_datum = rqmdd.MdDatum(self.model,
                                     md_datum_root = self.model.root_for_part(md_datum_part, is_rels = False))
        else:
            md_datum = None
        return md_datum

    def _load_related_wellbore_interp(self):
        """Return related wellbore interp object from XML if present."""

        interp_uuid = rqet.find_nested_tags_text(self.root, ['RepresentedInterpretation', 'UUID'])
        if interp_uuid is None:
            represented_interp = None
        else:
            represented_interp = rqo.WellboreInterpretation(self.model, uuid = interp_uuid)
        return represented_interp
