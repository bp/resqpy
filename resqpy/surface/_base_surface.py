"""_base_surface.py: base_surface class based on resqml standard."""

import logging

log = logging.getLogger(__name__)

import resqpy.organize as rqo
from resqpy.olio.base import BaseResqpy


class BaseSurface(BaseResqpy):
    """Base class to implement shared methods for other classes in this module."""

    def create_interpretation_and_feature(self,
                                          kind = 'horizon',
                                          name = None,
                                          interp_title_suffix = None,
                                          is_normal = True):
        """Creates xml and objects for a represented interpretaion and interpreted feature, if not already present."""

        assert kind in ['horizon', 'fault', 'fracture', 'geobody boundary']
        assert name or self.title, 'title missing'
        if not name:
            name = self.title

        if self.represented_interpretation_root is not None:
            log.debug(f'represented interpretation already exisrs for surface {self.title}')
            return
        if kind in ['horizon', 'geobody boundary']:
            feature = rqo.GeneticBoundaryFeature(self.model, kind = kind, feature_name = name)
            feature.create_xml()
            if kind == 'horizon':
                interp = rqo.HorizonInterpretation(self.model, genetic_boundary_feature = feature, domain = 'depth')
            else:
                interp = rqo.GeobodyBoundaryInterpretation(self.model,
                                                           genetic_boundary_feature = feature,
                                                           domain = 'depth')
        elif kind in ['fault', 'fracture']:
            feature = rqo.TectonicBoundaryFeature(self.model, kind = kind, feature_name = name)
            feature.create_xml()
            interp = rqo.FaultInterpretation(self.model,
                                             is_normal = is_normal,
                                             tectonic_boundary_feature = feature,
                                             domain = 'depth')  # might need more arguments
        else:
            log.critical('code failure')
        interp_root = interp.create_xml(title_suffix = interp_title_suffix)
        self.set_represented_interpretation_root(interp_root)
