"""Module providing wrapper for random number generator seeding functions."""

import numpy
import random


def seed(seed, package = 'all'):
    """Set seed for random number generator of one or more packages, to allow for repeatable behaviour.

    arguments:
       seed (int): the value to use to seed the random number generator(s); a value of None will
          generally result in an unrepeatable sequence
       package (string or list of strings): one or more of known packages: 'random' and 'numpy' at present;
          passing 'all' will cause all packages known to have a random number generator to be re-seeded
    """

    known_list = ['random', 'numpy']

    if isinstance(package, str):
        if package == 'all':
            package = known_list
        else:
            package = [package]

    assert isinstance(package, list)

    for pack in package:
        assert pack in known_list, 'unknown package for random number seeding: ' + str(pack)

        if pack == 'random':
            random.seed(seed)
        elif pack == 'numpy':
            numpy.random.seed(seed = seed)
