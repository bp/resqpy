# tests for resqpy.olio.fine_coarse class and functions

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.fine_coarse as rqfc

def test_fine_coarse():

   fc1 = rqfc.FineCoarse((3, 4, 5), (3, 4, 5))
   assert fc1 is not None
   fc1.set_all_ratios_constant()
   fc1.assert_valid()
   assert fc1.ratios((2, 2, 2)) == (1, 1, 1)

   fc2 = rqfc.FineCoarse((9, 16, 10), (3, 4, 5))
   assert fc2 is not None
   fc2.set_all_ratios_constant()
   fc2.set_all_proprtions_equal()
   fc2.assert_valid()
   assert fc2.ratios((2, 2, 2)) == (3, 4, 2)

   assert fc2.coarse_for_fine_kji0((5, 5, 5)) == (1, 1, 2)
   assert fc2.coarse_for_fine_axial(1, 13) == 3
   assert fc2.fine_base_for_coarse_axial(0, 1) == 4
   assert fc2.fine_base_for_coarse((2, 2, 2)) == (6, 8, 4)
   assert np.all(fc2.fine_box_for_coarse((1, 2, 3)) == np.array([[3, 8, 6], [5, 11, 7]], dtype = int))

   assert_array_almost_equal(fc2.proportion(1, 1), (0.25, 0.25, 0.25, 0.25))
   assert_array_almost_equal(fc2.interpolation(1, 1), (0.0, 0.25, 0.5, 0.75))

   # todo: several more methods to test


def test_fine_coarse_functions():
   assert rqfc.axis_for_letter('k') == 0
   assert rqfc.axis_for_letter('J') == 1
   assert rqfc.axis_for_letter('i') == 2
   assert rqfc.letter_for_axis(0) == 'K'
   assert rqfc.letter_for_axis(2) == 'I'
