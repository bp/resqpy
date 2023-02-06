"""Class for a collection of well logs"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import lasio
from datetime import datetime

import resqpy.property
import resqpy.weights_and_measures as bwam

import resqpy.property.property_collection as rqp_pc
import resqpy.property.property_common as rqp_c
import resqpy.property.well_log as rqp_wl


class WellLogCollection(rqp_pc.PropertyCollection):
    """Class for RESQML Property collection for a Wellbore Frame (ie well logs), inheriting from PropertyCollection."""

    def __init__(self, frame = None, property_set_root = None, realization = None):
        """Creates a new property collection related to a wellbore frame.

        arguments:
           frame (well.WellboreFrame object, optional): must be present unless creating a completely blank, empty collection.
              See :class:`resqpy.well.WellboreFrame`
           property_set_root (optional): if present, the collection is populated with the properties defined in the xml tree
              of the property set; frame must not be None when using this argument
           realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
              if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
              realisation number), or is for properties that do not have multiple realizations

        returns:
           the new WellLogCollection object

        note:
           usually a wellbore frame should be passed, however a completely blank collection may be created prior to using
           collection inheritance methods to populate from another collection, in which case the frame can be lazily left
           as None here;
           for actual well logs, the realization argument will usually be None; for synthetic logs created from an ensemble
           it may be of use
        """

        super().__init__(support = frame, property_set_root = property_set_root, realization = realization)

    def add_log(self, title, data, unit, discrete = False, realization = None, write = True, source_info = ''):
        """Add a well log to the collection, and optionally save to HDF / XML.

        Note:
           If write=False, the data are not written to the model and are saved to be written later.
           To write the data, you can subsequently call::

              logs.write_hdf5_for_imported_list()
              logs.create_xml_for_imported_list_and_add_parts_to_model()

        arguments:
           title (str): Name of log, typically the mnemonic
           data (array-like): log data to write. Must have same length as frame MDs
           unit (str): Unit of measure
           discrete (bool): by default False, i.e. continuous
           realization (int): If given, assign data to a realisation.
           write (bool): If True, write XML and HDF5.
           source_info (str): curve description or other human readable text

        returns:
           uuids: list of uuids of newly added properties. Only returned if write=True.
        """
        # Validate
        if self.support is None:
            raise ValueError('Supporting WellboreFrame not present')
        if len(data) != self.support.node_count:
            raise ValueError(f'Data mismatch: data length={len(data)}, but MD node count={self.support.node_count}')

        # Infer valid RESQML properties
        # TODO: Store orginal unit somewhere if it's not a valid RESQML unit
        uom = bwam.rq_uom(unit)
        property_kind, facet_type, facet = rqp_c.infer_property_kind(title, uom)

        # Add to the "import list"
        self.add_cached_array_to_imported_list(
            cached_array = np.array(data),
            source_info = source_info,
            keyword = title,
            discrete = discrete,
            uom = uom,
            property_kind = property_kind,
            facet_type = facet_type,
            facet = facet,
            realization = realization,
        )

        if write:
            self.write_hdf5_for_imported_list()
            return self.create_xml_for_imported_list_and_add_parts_to_model()
        else:
            return None

    def iter_logs(self):
        """Generator that yields component Log objects.

        Yields:
           instances of :class:`resqpy.property.WellLog` .

        Example::

           for log in log_collection.logs():
              print(log.title)
        """

        return (rqp_wl.WellLog(collection = self, uuid = uuid) for uuid in self.uuids())

    def to_df(self, include_units = False):
        """Return pandas dataframe of log data.

        arguments:
           include_units (bool): include unit in column names
        """

        assert self.support is not None

        # Get MD values
        md_values = self.support.node_mds
        assert md_values.ndim == 1, 'measured depths not 1D numpy array'

        # Get logs
        data = {}
        for log in self.iter_logs():

            col_name = log.title
            if include_units and log.uom:
                col_name += f' ({log.uom})'

            values = log.values()
            if values.ndim > 1:
                raise NotImplementedError('Multidimensional logs not yet supported in pandas')

            data[col_name] = values

        df = pd.DataFrame(data = data, index = md_values)
        return df

    def to_las(self):
        """Return a lasio.LASFile object, which can then be written to disk.

        Example::

           las = collection.to_las()
           las.write('example_logs.las', version=2)
        """
        las = lasio.LASFile()

        las.well.WELL = str(self.support.wellbore_interpretation.title)
        las.well.DATE = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        # todo: Get UWI from somewhere
        # las.well.UWI = uwi

        # Lookup depths from associated WellboreFrame and Trajectory
        md_values = self.support.node_mds
        md_unit = self.support.trajectory.md_uom

        # Measured depths should be first column in LAS file
        # todo: include datum information in description
        las.append_curve('MD', md_values, unit = md_unit)

        for well_log in self.iter_logs():
            name = well_log.title
            unit = well_log.uom
            values = well_log.values()
            if values.ndim > 1:
                raise NotImplementedError('Multidimensional logs not yet supported in pandas')
            assert len(values) > 0
            log.debug(f"Writing log {name} of length {len(values)} and shape {values.shape}")
            las.append_curve(name, values, unit = unit, descr = None)
        return las

    def set_wellbore_frame(self, frame):
        """Sets the supporting representation object for the property collection to be the given wellbore frame object.

        note:
           this method does not need to be called if the wellbore frame was identified at the time the collection
           was initialised
        """

        self.set_support(support = frame)
