# test olio box_utilities functions

import pytest
import numpy as np

import resqpy.olio.box_utilities as bx


def test_box_utilities():

   # some example boxes to test against, together with expected extents
   b1 = np.array([[0, 0, 0], [3, 4, 5]], dtype = int)
   e1 = np.array([4, 5, 6], dtype = int)
   b2 = np.array([[1, 1, 1], [2, 3, 4]], dtype = int)
   e2 = np.array([2, 3, 4], dtype = int)
   b3 = np.array([[5, 3, 2], [5, 5, 5]], dtype = int)
   e3 = np.array([1, 3, 4], dtype = int)
   b4 = np.array([[3, 3, 3], [4, 4, 4]], dtype = int)
   e4 = np.array([2, 2, 2], dtype = int)

   # check extents and logical 'volume' of boxes
   for b, e in zip((b1, b2, b3, b4), (e1, e2, e3, e4)):
      assert np.all(bx.extent_of_box(b) == e)
      assert bx.volume_of_box(b) == e[0] * e[1] * e[2]

   # check logically central cell indices against these expected values
   c1 = np.array([1, 2, 2], dtype = int)
   c2 = np.array([1, 2, 2], dtype = int)
   c3 = np.array([5, 4, 3], dtype = int)
   c4 = np.array([3, 3, 3], dtype = int)
   for b, c in zip((b1, b2, b3, b4), (c1, c2, c3, c4)):
      assert np.all(bx.central_cell(b) == c)

   # test functions for converting to and from simulator format strings
   assert bx.string_iijjkk1_for_box_kji0(b2) == '[2:5, 2:4, 2:3]'
   assert bx.spaced_string_iijjkk1_for_box_kji0(b1) == '1 6  1 5  1 4'
   assert bx.spaced_string_iijjkk1_for_box_kji0(b3, colon_separator = '-') == '3-6  4-6  6-6'
   assert np.all(b4 == bx.box_kji0_from_words_iijjkk1(bx.spaced_string_iijjkk1_for_box_kji0(b4).split()))

   # test cell inclusion function
   assert bx.cell_in_box(np.array([0, 0, 0], dtype = int), b1)
   assert bx.cell_in_box(np.array([2, 4, 3], dtype = int), b1)
   assert not bx.cell_in_box(np.array([7, 7, 7], dtype = int), b1)
   assert not bx.cell_in_box(np.array([1, 5, 3], dtype = int), b1)
   assert not bx.cell_in_box(np.array([0, 0, 0], dtype = int), b2)

   # check behaviour of box validity function, which compares the box against an extent (of a grid)
   assert bx.valid_box(b1, e1)
   assert bx.valid_box(b3, np.array([7, 8, 9], dtype = int))
   assert bx.valid_box(b3, np.array([6, 6, 6], dtype = int))
   assert not bx.valid_box(b2, np.array([5, 3, 7], dtype = int))

   # test the function for creating a box containing a single cell
   assert np.all(
      bx.single_cell_box(np.array([8, 0, 10], dtype = int)) == np.array([[8, 0, 10], [8, 0, 10]], dtype = int))

   # test the function for generating the box containing a full grid (from its extent)
   assert np.all(bx.full_extent_box0(e1) == b1)
   assert np.all(bx.full_extent_box0(np.array([7, 5, 3], dtype = int)) == np.array([[0, 0, 0], [6, 4, 2]], dtype = int))

   # check the function that finds the union of two boxes (the smallest box containing both the boxes)
   assert np.all(bx.union(b1, b2) == b1)
   assert np.all(bx.union(b2, b3) == np.array([[1, 1, 1], [5, 5, 5]], dtype = int))
   assert np.all(bx.union(b3, b4) == np.array([[3, 3, 2], [5, 5, 5]], dtype = int))
   for ba, bb in ((b1, b2), (b4, b1), (b2, b4)):
      assert np.all(bx.union(ba, bb) == bx.union(bb, ba))

   # test function for returning parent grid cell indices from a box and local indices within the box
   cell = np.array((0, 2, 3), dtype = int)
   assert np.all(bx.parent_cell_from_local_box_cell(b1, cell) == cell)
   assert np.all(bx.parent_cell_from_local_box_cell(b2, cell) == cell + 1)
   assert np.all(bx.parent_cell_from_local_box_cell(b3, cell) == (5, 5, 5))

   # test the inverse function which returns local cell indices within a box, given the parent cell indices
   for b in (b1, b2, b3):
      assert np.all(bx.local_box_cell_from_parent_cell(b, bx.parent_cell_from_local_box_cell(b, cell)) == cell)

   # the parent cell is outside the box in the following test
   assert bx.local_box_cell_from_parent_cell(b3, cell) is None

   # test the function that returns a boolean indicating whether two boxes have any overlap
   assert bx.boxes_overlap(b1, b2)
   assert bx.boxes_overlap(b2, b1)
   for ba, bb in ((b1, b3), (b2, b4), (b3, b4)):
      assert not bx.boxes_overlap(ba, bb)
      assert not bx.boxes_overlap(bb, ba)

   by = b1 + 2
   assert bx.boxes_overlap(b1, by)
   assert bx.boxes_overlap(by, b1)

   bz = b4 + 3
   assert not bx.boxes_overlap(b1, bz)
   assert not bx.boxes_overlap(bz, b2)

   # todo: test box trimming functions
