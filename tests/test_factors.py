# tests functions in resqpy.olio.factors module

import pytest

import resqpy.olio.factors as f


def test_factors():
   assert f.factorize(20) == [2, 2, 5]
   assert f.factorize(29) == [29]
   assert f.all_factors_from_primes(f.factorize(12)) == [1, 2, 3, 4, 6, 12]
   f60 = f.all_factors(60)
   assert f60 == [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
   f6 = f.all_factors(6)
   assert f6 == [1, 2, 3, 6]
   f.remove_subset(f60, f6)
   assert f60 == [4, 5, 10, 12, 15, 20, 30, 60]
