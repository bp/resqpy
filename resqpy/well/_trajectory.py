"""Trajectory class."""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np
import pandas as pd
from functools import partial

import resqpy.lines as rql
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.wellspec_keywords as wsk
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.weights_and_measures as bwam
import resqpy.well
import resqpy.well._md_datum as rqmdd
import resqpy.well._wellbore_frame as rqwbf
import resqpy.well._deviation_survey as rqds
import resqpy.well.well_utils as rqwu
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class Trajectory(BaseResqpy):
    """Class for RESQML Wellbore Trajectory Representation (Geometry).

    note:
       resqml allows trajectory to have different crs to the measured depth datum crs;
       however, this code requires the trajectory to be in the same crs as the md datum
    """

    resqml_type = 'WellboreTrajectoryRepresentation'
    well_name = rqo.alias_for_attribute("title")

    def __init__(self,
                 parent_model,
                 uuid = None,
                 crs_uuid = None,
                 md_datum = None,
                 deviation_survey = None,
                 data_frame = None,
                 grid = None,
                 cell_kji0_list = None,
                 wellspec_file = None,
                 spline_mode = 'cube',
                 ascii_trajectory_file = None,
                 survey_file_space_separated = False,
                 length_uom = None,
                 md_domain = None,
                 represented_interp = None,
                 well_name = None,
                 set_tangent_vectors = False,
                 hdf5_source_model = None,
                 originator = None,
                 extra_metadata = None):
        """Creates a new trajectory object and optionally loads it from xml, deviation survey, pandas dataframe, or

        ascii file.

        arguments:
           parent_model (model.Model object): the model which the new trajectory belongs to
           uuid (UUID, optional): if present, the Trajectory is initialised from xml for an existing RESQML object
              and the remaining arguments are mostly ignored
           crs_uuid (UUID, optional): the uuid of a Crs object to use when generating a new trajectory
           md_datum (MdDatum object): the datum that the depths for this trajectory are measured from;
              not used if uuid is not None
           deviation_survey (DeviationSurvey object, optional): if present and uuid is None
              then the trajectory is derived from the deviation survey based on minimum curvature
           data_frame (optional): a pandas dataframe with columns 'MD', 'X', 'Y' and 'Z', holding
              the measured depths, and corresponding node locations; ignored if uuid is not None
           grid (grid.Grid object, optional): only required if initialising from a list of cell indices;
              ignored otherwise
           cell_kji0_list (numpy int array of shape (N, 3)): ordered list of cell indices to be visited by
              the trajectory; ignored if uuid is not None
           wellspec_file (string, optional): name of an ascii file containing Nexus WELLSPEC data; well_name
              and length_uom arguments must be passed
           spline_mode (string, default 'cube'): one of 'none', 'linear', 'square', or 'cube'; affects spline
              tangent generation; only relevant if initialising from list of cells
           ascii_trajectory_file (string): filename of an ascii file holding the trajectory
              in a tabular form; ignored if uuid is not None
           survey_file_space_separated (boolean, default False): if True, deviation survey file is
              space separated; if False, comma separated (csv); ignored unless loading from survey file
           length_uom (string, default 'm'): a resqml length unit of measure applicable to the
              measured depths; should be 'm' or 'ft'
           md_domain (string, optional): if present, must be 'logger' or 'driller'; the source of the original
              deviation data; ignored if uuid is not None
           represented_interp (wellbore interpretation object, optional): if present, is noted as the wellbore
              interpretation object which this trajectory relates to; ignored if uuid is not None
           well_name (string, optional): used as citation title
           set_tangent_vectors (boolean, default False): if True and tangent vectors are not loaded then they will
              be computed from the control points
           hdf5_source_model (model.Model, optional): if present this model is used to determine the hdf5 file
              name from which to load the trajectory's array data; if None, the parent_model is used as usual
           originator (str, optional): the name of the person creating the trajectory, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the trajectory;
              ignored if uuid is not None

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

        self.crs_uuid = crs_uuid
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
                         extra_metadata = extra_metadata)

        if self.root is not None:
            if set_tangent_vectors and type(self.knot_count) is int and self.tangent_vectors is None:
                if self.knot_count > 1:
                    self.set_tangents()
            return

        # Using dictionary mapping to replicate a switch statement. The init_function key is chosen based on the
        # data source and the correct function is then called based on the init_function_dict
        init_function_dict = {
            'deviation_survey':
                partial(self.compute_from_deviation_survey,
                        method = 'minimum curvature',
                        set_tangent_vectors = set_tangent_vectors),
            'data_frame':
                partial(self.load_from_data_frame,
                        data_frame,
                        md_uom = length_uom,
                        md_datum = md_datum,
                        set_tangent_vectors = set_tangent_vectors),
            'cell_kji0_list':
                partial(self.load_from_cell_list, grid, cell_kji0_list, spline_mode, length_uom),
            'wellspec_file':
                partial(self.load_from_wellspec, grid, wellspec_file, well_name, spline_mode, length_uom),
            'ascii_trajectory_file':
                partial(self.load_from_ascii_file,
                        ascii_trajectory_file,
                        space_separated_instead_of_csv = survey_file_space_separated,
                        md_uom = length_uom,
                        md_datum = md_datum,
                        title = well_name,
                        set_tangent_vectors = set_tangent_vectors)
        }

        chosen_init_method = self.__choose_init_method(data_frame = data_frame,
                                                       cell_kji0_list = cell_kji0_list,
                                                       wellspec_file = wellspec_file,
                                                       deviation_survey_file = ascii_trajectory_file)

        try:
            init_function_dict[chosen_init_method]()
        except KeyError:
            log.warning('invalid combination of input arguments specified')

        # todo: create from already loaded deviation_survey node (ie. derive xyz points)

        if self.crs_uuid is None:
            if self.md_datum is not None:
                self.crs_uuid = self.md_datum.crs_uuid
            else:
                self.crs_uuid = self.model.crs_uuid

        if not self.title:
            self.title = 'well trajectory'

        if self.md_datum is None and self.control_points is not None:
            self.md_datum = rqmdd.MdDatum(self.model, crs_uuid = self.crs_uuid, location = self.control_points[0])

        if set_tangent_vectors and type(self.knot_count) is int and self.tangent_vectors is None:
            if self.knot_count > 1:
                self.set_tangents()

    def __choose_init_method(self, data_frame, cell_kji0_list, wellspec_file, deviation_survey_file):
        """Choose an init method based on data source."""

        if self.deviation_survey is not None:
            return 'deviation_survey'
        elif data_frame is not None:
            return 'data_frame'
        elif cell_kji0_list is not None:
            return 'cell_kji0_list'
        elif wellspec_file:
            return 'wellspec_file'
        elif deviation_survey_file:
            return 'ascii_trajectory_file'
        else:
            return None

    def iter_wellbore_frames(self):
        """Iterable of all WellboreFrames associated with a trajectory.

        Yields:
           frame: instance of :class:`resqpy.organize.WellboreFrame`

        :meta common:
        """
        uuids = self.model.uuids(obj_type = "WellboreFrameRepresentation", related_uuid = self.uuid)
        for uuid in uuids:
            yield rqwbf.WellboreFrame(self.model, uuid = uuid)

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
            rqwu.load_hdf5_array(self, mds_node, 'measured_depths')
        control_points_node = rqet.find_tag(geometry_node, 'ControlPoints')
        rqwu.load_hdf5_array(self, control_points_node, 'control_points', tag = 'Coordinates')
        tangents_node = rqet.find_tag(geometry_node, 'TangentVectors')
        if tangents_node is not None:
            rqwu.load_hdf5_array(self, tangents_node, 'tangent_vectors', tag = 'Coordinates')
        relatives_model = self.model  # if hdf5_source_model is None else hdf5_source_model
        # md_datum - separate part, referred to in this tree
        md_datum_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['MdDatum', 'UUID']))
        assert md_datum_uuid is not None, 'failed to fetch uuid of md datum for trajectory'
        md_datum_part = relatives_model.part_for_uuid(md_datum_uuid)
        assert md_datum_part, 'md datum part not found in model'
        self.md_datum = rqmdd.MdDatum(self.model, uuid = relatives_model.uuid_for_part(md_datum_part))
        ds_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['DeviationSurvey', 'UUID']))
        if ds_uuid is not None:  # this will probably not work when relatives model is different from self.model
            ds_uuid = self.model.uuid(obj_type = 'DeviationSurveyRepresentation',
                                      uuid = ds_uuid)  # check part is present
            if ds_uuid is not None:
                self.deviation_survey = rqds.DeviationSurvey(self.model, uuid = ds_uuid)
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
            self.title = rqet.find_nested_tags_text(survey.root, ['Citation', 'Title'])
        self.start_md = self.measured_depths[0]
        self.finish_md = self.measured_depths[-1]
        if md_domain is not None:
            self.md_domain = md_domain
        self.control_points = np.empty((self.knot_count, 3))
        self.control_points[0, :] = survey.first_station
        self.__calculate_trajectory_from_inclination_and_azimuth(survey = survey)
        self.tangent_vectors = None
        if set_tangent_vectors:
            self.set_tangents()
        self.md_datum = survey.md_datum

    def __calculate_trajectory_from_inclination_and_azimuth(self, survey):
        """Calculate well trajectory from inclination and azimuth data."""

        for sp in range(1, self.knot_count):
            i1 = survey.inclinations[sp - 1]
            i2 = survey.inclinations[sp]
            az1 = survey.azimuths[sp - 1]
            az2 = survey.azimuths[sp]
            delta_md = survey.measured_depths[sp] - survey.measured_depths[sp - 1]
            assert delta_md > 0.0
            if i1 == i2 and az1 == az2:
                delta_v = delta_md * vec.unit_vector_from_azimuth_and_inclination(az1, i1)
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
        cell_kji0_list = np.array(cell_kji0_list, dtype = int)
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
        """Sets the trajectory data based on visiting the cells identified in a Nexus wellspec keyword."""

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

        well_col = self.__check_well_col(df = df, trajectory_file = trajectory_file, well_col = well_col)
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

    @staticmethod
    def __check_well_col(
        df,
        trajectory_file,
        well_col = None,
    ):
        """Verifies that a valid well_col has been supplied or can be found in the dataframe that has been generated

        from the trajectory file.
        """

        if well_col and well_col not in df.columns:
            log.warning('well column ' + str(well_col) + ' not found in ascii trajectory file ' + str(trajectory_file))
            well_col = None
        if well_col is None:
            for col in df.columns:
                if str(col).upper().startswith('WELL'):
                    well_col = col
                    break
        return well_col

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
        """Returns an xyz triplet corresponding to the given measured depth; uses simple linear interpolation between

        knots.

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
                return interpolate(self.control_points[i1], self.control_points[i1 + 1],
                                   (md - self.measured_depths[i1]) /
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
        """Instantiate new empty WellboreFeature and WellboreInterpretation objects, if a wellboreinterpretation does

        not already exist.

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

        self.__create_wellbore_feature_and_interpretation_xml(add_as_part = add_as_part,
                                                              add_relationships = add_relationships,
                                                              originator = originator)

        if md_datum_root is None:
            md_datum_root = self.__create_md_datum_root(md_datum_xyz = md_datum_xyz)

        wbt_node = super().create_xml(originator = originator, add_as_part = False)

        self.__create_wbt_node_non_geometry_sub_elements(wbt_node = wbt_node)

        self.model.create_md_datum_reference(self.md_datum.root, root = wbt_node)

        if self.line_kind_index != 0:  # 0 means vertical well, which doesn't need a geometry

            # todo: check geometry elements for parametric curve flavours other than minimum curvature

            geom, kc_node, lki_node, cpp_node, cpp_values_node, cp_node, cp_coords_node, tv_node, tv_coords_node = self.__create_wbt_node_geometry_sub_elements(
                wbt_node = wbt_node)

            self.__get_crs_uuid()

            self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'controlPointParameters', root = cpp_values_node)

            self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'controlPoints', root = cp_coords_node)

            if (tv_node is not None) & (tv_coords_node is not None):
                self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'tangentVectors', root = tv_coords_node)

        self.__create_deviation_survey_reference_node(wbt_node = wbt_node)

        interp_root = self.__create_wellbore_interpretation_reference_node(wbt_node = wbt_node)

        self.__add_as_part_and_add_relationships(wbt_node = wbt_node,
                                                 interp_root = interp_root,
                                                 add_as_part = add_as_part,
                                                 add_relationships = add_relationships,
                                                 ext_uuid = ext_uuid)

        return wbt_node

    def __create_wellbore_feature_and_interpretation_xml(self, add_as_part, add_relationships, originator):
        """ Create root node for WellboreFeature and WellboreInterpretation objects."""

        if self.feature_and_interpretation_to_be_written:
            if self.wellbore_interpretation is None:
                self.create_feature_and_interpretation()
            if self.wellbore_feature is not None:
                self.wellbore_feature.create_xml(add_as_part = add_as_part, originator = originator)
            self.wellbore_interpretation.create_xml(add_as_part = add_as_part,
                                                    add_relationships = add_relationships,
                                                    originator = originator)

    def __create_md_datum_root(self, md_datum_xyz):
        """ Create the root node for the MdDatum object."""

        if self.md_datum is None:
            assert md_datum_xyz is not None
            self.md_datum = rqmdd.MdDatum(self.model, location = md_datum_xyz)
        if self.md_datum.root is None:
            md_datum_root = self.md_datum.create_xml()
        else:
            md_datum_root = self.md_datum.root
        return md_datum_root

    def __create_wbt_node_non_geometry_sub_elements(self, wbt_node):
        """ Append sub-elements to the Trajectory object's root node that are unrelated to well geometry."""

        start_node = rqet.SubElement(wbt_node, ns['resqml2'] + 'StartMd')
        start_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
        start_node.text = str(self.start_md)

        finish_node = rqet.SubElement(wbt_node, ns['resqml2'] + 'FinishMd')
        finish_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
        finish_node.text = str(self.finish_md)

        md_uom = rqet.SubElement(wbt_node, ns['resqml2'] + 'MdUom')
        md_uom.set(ns['xsi'] + 'type', ns['eml'] + 'LengthUom')
        md_uom.text = bwam.rq_length_unit(self.md_uom)

        if self.md_domain:
            domain_node = rqet.SubElement(wbt_node, ns['resqml2'] + 'MdDomain')
            domain_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'MdDomain')
            domain_node.text = self.md_domain

    def __create_wbt_node_geometry_sub_elements(self, wbt_node):
        """ Append sub-elements to the Trajectory object's root node that are related to well geometry."""

        geom = rqet.SubElement(wbt_node, ns['resqml2'] + 'Geometry')
        geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'ParametricLineGeometry')
        geom.text = '\n'

        # note: resqml standard allows trajectory to be in different crs to md datum
        #       however, this module often uses the md datum crs, if the trajectory has been imported

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

        cp_node = rqet.SubElement(geom, ns['resqml2'] + 'ControlPoints')
        cp_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
        cp_node.text = rqet.null_xml_text

        cp_coords_node = rqet.SubElement(cp_node, ns['resqml2'] + 'Coordinates')
        cp_coords_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        cp_coords_node.text = rqet.null_xml_text

        if self.tangent_vectors is not None:
            tv_node = rqet.SubElement(geom, ns['resqml2'] + 'TangentVectors')
            tv_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
            tv_node.text = rqet.null_xml_text

            tv_coords_node = rqet.SubElement(tv_node, ns['resqml2'] + 'Coordinates')
            tv_coords_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            tv_coords_node.text = rqet.null_xml_text
        else:
            tv_node = None
            tv_coords_node = None

        return geom, kc_node, lki_node, cpp_node, cpp_values_node, cp_node, cp_coords_node, tv_node, tv_coords_node

    def __get_crs_uuid(self):
        """ Assign the same crs uuid to the Trajectory object.

        note: this assumes that the crs is the same for the MdDatum object and the Trajectory object.
        """

        if self.crs_uuid is None:
            self.crs_uuid = self.md_datum.crs_uuid
        assert self.crs_uuid is not None

    def __create_deviation_survey_reference_node(self, wbt_node):
        """ Create a reference node to a DeviationSurvey object and append it to the WellboreTrajectory

        object's root node.
        """

        if self.deviation_survey is not None:
            ds_root = self.deviation_survey.root_node
            self.model.create_ref_node('DeviationSurvey',
                                       rqet.find_tag(rqet.find_tag(ds_root, 'Citation'), 'Title').text,
                                       bu.uuid_from_string(ds_root.attrib['uuid']),
                                       content_type = 'obj_DeviationSurveyRepresentation',
                                       root = wbt_node)

    def __create_wellbore_interpretation_reference_node(self, wbt_node):
        """Create a reference node to a WellboreInterpretation object and append it to the WellboreTrajectory

        object's root node.
        """

        interp_root = None
        if self.wellbore_interpretation is not None:
            interp_root = self.wellbore_interpretation.root
            self.model.create_ref_node('RepresentedInterpretation',
                                       rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                       bu.uuid_from_string(interp_root.attrib['uuid']),
                                       content_type = 'obj_WellboreInterpretation',
                                       root = wbt_node)
        return interp_root

    def __add_as_part_and_add_relationships(self, wbt_node, interp_root, add_as_part, add_relationships, ext_uuid):
        """Add the newly created Trajectory object's root node as a part in the model and add reciprocal

        relationships.
        """

        if add_as_part:
            self.model.add_part('obj_WellboreTrajectoryRepresentation', self.uuid, wbt_node)
            if add_relationships:
                crs_root = self.model.root_for_uuid(self.crs_uuid)
                self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', crs_root, 'sourceObject')
                self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', self.md_datum.root,
                                                          'sourceObject')
                if self.deviation_survey is not None:
                    self.model.create_reciprocal_relationship(wbt_node, 'destinationObject',
                                                              self.deviation_survey.root_node, 'sourceObject')
                if interp_root is not None:
                    self.model.create_reciprocal_relationship(wbt_node, 'destinationObject', interp_root,
                                                              'sourceObject')
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(wbt_node, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

    def write_hdf5(self, file_name = None, mode = 'a'):
        """Create or append to an hdf5 file, writing datasets for the measured depths, control points and tangent

        vectors.

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
        """Implements equals operator.

        Compares class type and uuid
        """

        # TODO: more detailed equality comparison
        other_uuid = getattr(other, "uuid", None)
        return isinstance(other, self.__class__) and bu.matching_uuids(self.uuid, other_uuid)
