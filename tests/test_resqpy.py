import pytest
from resqpy import model


def test_empty_model():
   _ = model.Model()
   return


def test_all_imports():
   from resqpy import crs, derived_model, fault, grid, grid_surface, lines, organize
   from resqpy import property, rq_import, surface, time_series, well
   #    from resqpy.olio import *
   return
