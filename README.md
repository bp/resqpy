# resqpy: Python API for working with RESQML models

[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bp/resqpy/blob/master/LICENSE)

## Introduction

This repository contains pure python modules which provide a programming
interface (API) for reading, writing, and modifying reservoir models in the
RESQML format.

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

### Documentation

Build locally with:

```bash
sphinx-build docs docs/html
```

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

Run locally with:

```bash
pytest tests/
```

### Making a release

To make a release at a given commit, simply make a git tag:

```bash
git tag v0.0.0
git push origin v0.0.0
```

The tag must have the prefix `v` and have the form `MAJOR.MINOR.PATCH`.

Following [semantic versioning](https://semver.org/), increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.
