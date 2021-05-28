"""Base class for all RESQML objects"""

import resqpy.olio.uuid as bu


class BaseResqml:
    """Base class wfor all RESQML objects"""
    
    def __init__(self, parent_model, root_node=None):
        """Base class for RESQML objects

        Args:
            parent_model (model.Model): Model containing object
        """
        self.model = parent_model
        self.root_node = root_node
        self.uuid = bu.new_uuid()
