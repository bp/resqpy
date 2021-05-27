# resqpy: Python API for working with RESQML models

[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bp/resqpy/blob/master/LICENSE)
[![Python CI](https://github.com/bp/resqpy/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/bp/resqpy/actions/workflows/ci-tests.yml)
[![PyPI](https://badge.fury.io/py/resqpy.svg)](https://badge.fury.io/py/resqpy)
[![Documentation Status](https://readthedocs.org/projects/resqpy/badge/?version=latest)](https://resqpy.readthedocs.io/en/latest/?badge=latest)

## Introduction

This repository contains pure python modules which provide a programming
interface (API) for reading, writing, and modifying reservoir models in the
RESQML format.

See the full documentation [here](https://resqpy.readthedocs.io).

### Current capabilities

Specialized classes are only available for a subset of the RESQML high level
object classes:

- Grids: IjkGridRepresentation
- Wells: WellboreTrajectoryRepresentation, DeviationSurveyRepresentation,
  MdDatum, BlockedWellboreRepresentation, WellboreFrameRepresentation,
  WellboreMarkerFrameRepresentation, WellboreInterpretation, WellboreFeature
- Properties for Grids, Wells etc: ContinuousProperty, DiscreteProperty,
  CatagoricalProperty, PropertyKind, PropertySet, StringTableLookup
- Surfaces: TriangulatedSetRepresentation, Grid2dRepresentation,
  PointSetRepresentation, HorizonInterpretation, GeneticBoundaryFeature
- Faults: GridConnectionSetRepresentation, FaultInterpretation,
  TectonicBoundaryFeature
- Lines: PolylineRepresentation, PolylineSetRepresentation
- Other: TimeSeries, EpcExternalPartReference, various other Interpretation and
  Feature classes

Furthermore, not all variations of these object types are supported; for
example, radial IJK grids are not yet catered for, although the RESQML standard
does allow for such grids.

It is envisaged that the code base will be expanded to include other classes of
object and more fully cover the options permitted by the RESQML standard.

Modification functionality at the moment focuses on changes to grid geometry.

### Installation

Install from source in "editable" mode with:

```bash
pip install -e /path/to/repo/
```

## Contributing

### Repository structure

- `resqpy`: high level modules providing classes for main RESQML object types
  and high level modification functions
- `resqpy/olio`: low level modules, not often imported directly by calling code
- `tests`: unit tests

### Unit tests

Run the test suite locally with:

```bash
pytest tests/
```
