"""Classes for storing and retrieving dataframes as RESQML objects.

Note that this module uses the obj_Grid2dRepresentation class in a way that was not envisaged
when the RESQML standard was defined; software that does not use resqpy is unlikely to be
able to do much with data stored in this way
"""

import logging

log = logging.getLogger(__name__)

import warnings
import numpy as np
import pandas as pd

import resqpy.crs as rqc
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
import resqpy.surface as rqs
import resqpy.time_series as rqts

# todo: add support for building an ensemble of dataframes using the same mesh support


class DataFrame:
    """Class for storing and retrieving a pandas dataframe of numerical data as a RESQML property.

    notes:
       actual values are stored either as z values in a Mesh (Grid2d) object, or as a property on
       such a mesh when multiple raalizations are in use; a regular Mesh object is created to act
       as a supporting representation; columns are mapped onto I and rows onto J; if a property is
       used then the indexable elements are 'nodes'; column titles are stored in a related StringLookup
       object, indexed by column number; column units are optionally treated in the same way (uom for
       the property is generally set to Euc); all values are stored as floats; use the derived TimeTable
       class if rows relate to steps in a TimeSeries; use the derived RelPerm class if the rows relate to
       relative permeability data
    """

    def __init__(self,
                 model,
                 uuid = None,
                 df = None,
                 uom_list = None,
                 realization = None,
                 title = 'dataframe',
                 column_lookup_uuid = None,
                 uom_lookup_uuid = None,
                 extra_metadata = None):
        """Create a new Dataframe object from either a previously stored property or a pandas dataframe.

        arguments:
           model (model.Model): the model to which the new Dataframe will be attached
           uuid (uuid.UUID, optional): the uuid of an existing Grid2dRepresentation
              object acting as support for a dataframe property (or holding the dataframe as z values)
           df (pandas.DataFrame, optional): a dataframe from which the new Dataframe is to be created;
              if both uuid and df are supplied, realization must not be None and a new
              realization property will be created
           uom_list (list of str, optional): a list holding the units of measure for each
              column; if present, length of list must match number of columns in df; ignored if
              uuid is not None
           realization (int, optional): if present, the realization number of the RESQML property
              holding the dataframe
           title (str, default 'dataframe'): used as the citation title for the Mesh (and property);
              ignored if uuid is not None
           column_lookup_uuid (uuid, optional): if present, the uuid of a string lookup table holding
              the column names; if present, the contents and order of the table must match the columns
              in the dataframe; if absent, a new lookup table will be created
           uom_lookup_uuid (uuid, optional): if present, the uuid of a string lookup table holding
              the units of measure for each column; if None and uom_list is present, a new table
              will be created; if both uom_list and uom_lookup_uuid are present, their contents
              must match
           extra_metadata (dict, optional): if present, a dictionary of extra metadata items, str: str;
              ignored if uuid is not None

        returns:
           a newly created Dataframe object

        notes:
           when initialising from an existing RESQML object, the supporting mesh and its property should
           have been originally created using this class; when working with ensembles, each object of this
           class will only handle the data for one realization, though they may share a common support;
           if both a uuid and a df are provided, a realization number must also be given and the dataframe
           is used to create a new realization similar to that identified by the uuid
        """

        assert uuid is not None or df is not None
        assert uuid is None or df is None or realization is not None

        self.model = model
        self.df = None
        self.n_rows = self.n_cols = 0
        self.uom_list = None
        self.realization = realization
        self.title = title
        self.mesh = None  # only generated when needed for write_hdf5(), create_xml()
        self.pc = None  # property collection; only generated when needed for write_hdf5(), create_xml()
        self.column_lookup_uuid = column_lookup_uuid
        self.column_lookup = None  # string lookup table mapping column index (0 based) to column name
        self.uom_lookup_uuid = uom_lookup_uuid
        self.uom_lookup = None  # string lookup table mapping column index (0 based) to uom
        self.extra_metadata = extra_metadata

        if uuid is not None:
            support_root = model.root_for_uuid(uuid)
            assert rqet.node_type(support_root) == 'obj_Grid2dRepresentation'
            self.mesh = rqs.Mesh(self.model, uuid = uuid)
            self.extra_metadata = self.mesh.extra_metadata
            assert 'dataframe' in self.extra_metadata and self.extra_metadata['dataframe'] == 'true'
            self.title = self.mesh.title
            self.n_rows, self.n_cols = self.mesh.nj, self.mesh.ni
            cl_uuid = self.model.uuid(obj_type = 'StringTableLookup', related_uuid = uuid, title = 'dataframe columns')
            assert cl_uuid is not None, 'column name lookup table not found for dataframe'
            self.column_lookup = rqp.StringLookup(self.model, uuid = cl_uuid)
            self.column_lookup_uuid = self.column_lookup.uuid
            assert self.column_lookup.length() == self.n_cols
            ul_uuid = self.model.uuid(obj_type = 'StringTableLookup', related_uuid = uuid, title = 'dataframe units')
            if ul_uuid is not None:
                self.uom_lookup = rqp.StringLookup(self.model, uuid = ul_uuid)
                self.uom_lookup_uuid = self.uom_lookup.uuid
                self.uom_list = self.uom_lookup.get_list()
            da = self.mesh.full_array_ref()[..., 2]  # dataframe data as 2D numpy array, defaulting to z values in mesh
            existing_pc = rqp.PropertyCollection(support = self.mesh)
            existing_count = 0 if existing_pc is None else existing_pc.number_of_parts()
            if df is None:  # existing dara, either in mesh or property
                if existing_count > 0:  # use property data instead of z values
                    if existing_count == 1:
                        if self.realization is not None:
                            assert existing_pc.realization_for_part(existing_pc.singleton()) == self.realization
                    else:
                        assert self.realization is not None, 'no realization specified when accessing ensemble dataframe'
                    da = existing_pc.single_array_ref(realization = self.realization)
                    assert da is not None and da.ndim == 2 and da.shape == (self.n_rows, self.n_cols)
                else:
                    assert realization is None
                self.df = pd.DataFrame(da, columns = self.column_lookup.get_list())
            else:  # both uuid and df supplied: add a new realisation
                if existing_count > 0:
                    assert existing_pc.singleton(
                        realization = self.realization) is None, 'dataframe realization already exists'
                self.df = df.copy()
                assert len(self.df) == self.n_rows
                assert len(self.df.columns) == self.n_rows
        else:
            assert df is not None, 'no dataframe provided when instantiating DataFrame object'
            self.df = df.copy()
            # todo: check data type of columns – restrict to numerical data
            self.n_rows = len(self.df)
            self.n_cols = len(self.df.columns)
            if column_lookup_uuid is not None:
                self.column_lookup = rqp.StringLookup(self.model, uuid = column_lookup_uuid)
                assert self.column_lookup is not None
                assert self.column_lookup.length() == self.n_cols
                assert all(self.df.columns == self.column_lookup.get_list())  # exact match of column names required!
            if uom_lookup_uuid is not None:
                self.uom_lookup = rqp.StringLookup(self.model, uuid = uom_lookup_uuid)
                assert self.uom_lookup is not None
            if uom_list is not None:
                assert len(uom_list) == self.n_cols
                self.uom_list = uom_list.copy()
                if self.uom_lookup is not None:
                    assert self.uom_list == self.uom_lookup.get_list()
            elif self.uom_lookup is not None:
                self.uom_list = self.uom_lookup.get_list()

    def dataframe(self):
        """Returns the Dataframe as a pandas DataFrame."""

        return self.df

    def column_uom(self, col_index):
        """Returns units of measure for the specified column, or Euc if no units present."""

        if self.units_table is None:
            return 'Euc'
        assert 0 <= col_index < self.n_cols, 'column index out of range'
        return self.units_table.get_string(col_index)

    def write_hdf5_and_create_xml(self):
        """Write dataframe data to hdf5 file and create xml for RESQML objects to represent dataframe."""

        self._set_mesh_from_df()  # writes hdf5 data and creates xml for mesh (and property)

        if self.column_lookup is None:
            self.column_lookup = rqp.StringLookup(self.model,
                                                  int_to_str_dict = dict(enumerate(self.df.columns)),
                                                  title = 'dataframe columns')
            self.column_lookup_uuid = self.column_lookup.uuid
            sl_node = self.column_lookup.create_xml()
        else:
            sl_node = self.column_lookup.root
        if sl_node is not None:
            self.model.create_reciprocal_relationship(self.mesh.root, 'destinationObject', sl_node, 'sourceObject')

        if self.uom_list and self.uom_lookup is None:
            self.uom_lookup = rqp.StringLookup(self.model,
                                               int_to_str_dict = dict(enumerate(self.uom_list)),
                                               title = 'dataframe units')
            self.uom_lookup_uuid = self.uom_lookup.uuid
            ul_node = self.uom_lookup.create_xml()
        elif self.uom_lookup is not None:
            ul_node = self.uom_lookup.root
        else:
            ul_node = None
        if ul_node is not None:
            self.model.create_reciprocal_relationship(self.mesh.root, 'destinationObject', ul_node, 'sourceObject')

    def _set_mesh_from_df(self):
        """Creates Mesh object; called before writing to hdf5 or creating xml."""
        # note: actual data is stored in related Property if realization number is present, directly in Mesh otherwise

        assert self.n_rows == len(self.df)
        assert self.n_cols == len(self.df.columns)

        if self.mesh is None:
            origin = (0.0, 0.0, 0.0)
            dxyz_dij = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            crs_uuids = self.model.uuids(obj_type = 'LocalDepth3dCrs')
            if len(crs_uuids) == 0:
                crs = rqc.Crs(self.model)
                crs.create_xml()
                crs_uuid = crs.uuid
            else:  # use any available crs
                crs_uuid = crs_uuids[0]
            if self.realization is None:
                self.mesh = rqs.Mesh(self.model,
                                     mesh_flavour = 'reg&z',
                                     ni = self.n_cols,
                                     nj = self.n_rows,
                                     dxyz_dij = dxyz_dij,
                                     origin = origin,
                                     z_values = np.array(self.df),
                                     crs_uuid = crs_uuid)
            else:
                self.mesh = rqs.Mesh(self.model,
                                     mesh_flavour = 'regular',
                                     ni = self.n_cols,
                                     nj = self.n_rows,
                                     dxyz_dij = dxyz_dij,
                                     origin = origin,
                                     crs_uuid = crs_uuid)
            self.mesh.write_hdf5()
            mesh_root = self.mesh.create_xml(title = self.title)
            rqet.create_metadata_xml(mesh_root, {'dataframe': 'true'})
            if self.realization is not None:
                self.pc = rqp.PropertyCollection()
                self.pc.set_support(support = self.mesh)
                dataframe_pk_uuid = self.model.uuid(obj_type = 'PropertyKind', title = 'dataframe')
                if dataframe_pk_uuid is None:
                    dataframe_pk = rqp.PropertyKind(self.model, title = 'dataframe', example_uom = 'Euc')
                    dataframe_pk.create_xml()
                    dataframe_pk_uuid = dataframe_pk.uuid
                self.pc.add_cached_array_to_imported_list(np.array(self.df),
                                                          'dataframe',
                                                          self.title,
                                                          uom = 'Euc',
                                                          property_kind = 'dataframe',
                                                          local_property_kind_uuid = dataframe_pk_uuid,
                                                          realization = self.realization,
                                                          indexable_element = 'nodes')
                self.pc.write_hdf5_for_imported_list()
                self.pc.create_xml_for_imported_list_and_add_parts_to_model()


class TimeTable(DataFrame):
    """Class for storing and retrieving a pandas dataframe where rows relate to steps in a time series.

    note:
       inherits from DataFrame class
    """

    def __init__(self,
                 model,
                 uuid = None,
                 df = None,
                 uom_list = None,
                 realization = None,
                 time_series = None,
                 title = 'timetable',
                 column_lookup_uuid = None,
                 uom_lookup_uuid = None):
        """Create a new TimeTable object from either a previously stored property or a pandas dataframe.

        arguments:
           time_series (resqpy.time_series.TimeSeries): the time series which rows in the dataframe relate to;
              required if initialising from a dataframe, ignored otherwise

        note:
           see DataFrame class docstring for details of other arguments
        """

        # todo: add option to set up time series from a column in the dataframe?

        assert uuid is not None or (df is not None and time_series is not None)

        super().__init__(model,
                         uuid = uuid,
                         df = df,
                         uom_list = uom_list,
                         realization = realization,
                         title = title,
                         column_lookup_uuid = column_lookup_uuid,
                         uom_lookup_uuid = uom_lookup_uuid)
        if uuid is not None:
            ts_uuid = self.model.uuid(obj_type = 'TimeSeries', related_uuid = self.mesh.uuid)
            assert ts_uuid is not None, 'no time series related to mesh holding dataframe'
            self.ts = rqts.TimeSeries(self.model, uuid = ts_uuid)
        else:
            assert time_series is not None
            assert time_series.number_of_timestamps() == self.n_rows
            self.ts = time_series

    def time_series(self):
        """Returns the TimeSeries object in use by the time table."""

        return self.ts

    def write_hdf5_and_create_xml(self):
        """Write time table data to hdf5 file and create xml for RESQML objects to represent dataframe."""

        super().write_hdf5_and_create_xml()
        # note: time series xml must be created before calling this method
        self.model.create_reciprocal_relationship(self.mesh.root, 'destinationObject', self.ts.root, 'sourceObject')


def dataframe_parts_in_model(model, timetables = None, title = None, related_uuid = None):
    """Returns list of part names within model that are representing DataFrame support objects.

    arguments:
       model (model.Model): the model to be inspected for dataframes
       timetables (boolean or None): if True, only TimeTable dataframe parts will be included; if False
          only DataFrame parts that are not representing TimeTable objects will be included; if None,
          both parts for both types of dataframe will be included
       title (str, optional): if present, only parts with a citation title exactly matching will be
          included
       related_uuid (uuid, optional): if present, only parts relating to this uuid are included

    returns:
       list of str, each element in the list is a part name, within model, which is representing the
       support for a DataFrame object
    """

    df_parts_list = model.parts(obj_type = 'Grid2dRepresentation',
                                title = title,
                                extra = {'dataframe': 'true'},
                                related_uuid = related_uuid)

    if timetables is not None:
        filtered_list = []
        for df_part in df_parts_list:
            is_tt = (model.part(obj_type = 'TimeSeries', related_uuid = model.uuid_for_part(df_part)) is not None)
            if timetables == is_tt:
                filtered_list.append(df_part)
        df_parts_list = filtered_list

    return df_parts_list


def timetable_parts_in_model(model, title = None, related_uuid = None):
    """Returns list of part names within model that are representing TimeTable dataframe support objects.

    arguments:
       model (model.Model): the model to be inspected for dataframes
       title (str, optional): if present, only parts with a citation title exactly matching will be
          included
       related_uuid (uuid, optional): if present, only parts relating to this uuid are included

    returns:
       list of str, each element in the list is a part name, within model, which is representing the support
       for a TimeTable object
    """

    return dataframe_parts_in_model(model, timetables = True, title = title, related_uuid = related_uuid)


def dataframe_for_title(model, title, realization = None):
    """Returns a DataFrame object loaded from model, with given title (optionally for given realization)."""

    df_parts = dataframe_parts_in_model(model, title = title)
    if df_parts is None or len(df_parts) == 0:
        return None
    assert len(df_parts) == 1
    return DataFrame(model, uuid = model.uuid_for_part(df_parts[0]), realization = realization)


def timetable_for_title(model, title, realization = None):
    """Returns a TimeTable object loaded from model, with given title (optionally for given realization)."""

    tt_parts = timetable_parts_in_model(model, title = title)
    if tt_parts is None or len(tt_parts) == 0:
        return None
    assert len(tt_parts) == 1
    return TimeTable(model, uuid = model.uuid_for_part(tt_parts[0]), realization = realization)
