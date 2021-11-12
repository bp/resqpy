"""well_functions.py: resqpy well module providing trajectory, deviation survey, blocked well, wellbore frame and marker frame and md datum classes.

"""

# todo: create a trajectory from a deviation survey, assuming minimum curvature

version = '10th November 2021'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('well_functions.py version ' + version)

import resqpy.olio.xml_et as rqet


# def load_hdf5_array(object, node, array_attribute, tag = 'Values', dtype = 'float', model = None):
#     """Loads the property array data as an attribute of object, from the hdf5 referenced in xml node.
#
#     :meta private:
#     """
#
#     assert (rqet.node_type(node) in ['DoubleHdf5Array', 'IntegerHdf5Array', 'Point3dHdf5Array'])
#     if model is None:
#         model = object.model
#     h5_key_pair = model.h5_uuid_and_path_for_node(node, tag = tag)
#     if h5_key_pair is None:
#         return None
#     return model.h5_array_element(h5_key_pair,
#                                   index = None,
#                                   cache_array = True,
#                                   dtype = dtype,
#                                   object = object,
#                                   array_attribute = array_attribute)
