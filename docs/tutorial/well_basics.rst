A first look at Well Objects
============================

**This page is under development.**

This tutorial introduces the classes relating to wells and goes into more detail for some of the basic ones. Other tutorials will cover the remaining well classes in depth.

The RESQML classes for wells
----------------------------

The RESQML standard contains several classes of object that relate to wells. Each of these has an equivalent resqpy class, named in parentheses in this list:

* MdDatum (MdDatum) - a simple class holding a datum location for measured depths
* DeviationSurveyRepresentation (DeviationSurvey) - inclination and azimuth at given measured depths
* WellboreTrajectoryRepresentation (Trajectory) - xyz coordinates at given measured depths
* WellboreFrameRepresentation (WellboreFrame) - list of measured depths supporting well log properties
* WellboreMarkerFrameRepresentation (WellboreMarkerFrame) - list of picks (well markers)
* BlockedWellboreRepresentation (BlockedWell) - list of cells visited or perforated by a well

The resqpy WellboreFrame and BlockedWell support related properties, which can be handled with the PropertyCollection and/or Property classes. However, for working with well logs, the resqpy property module includes the following classes for convenience:

* (WellLogCollection) - for managing logs, including a method for exporting in LAS format
* (WellLog) - for simpler access to a single log

RESQML also has organisational classes relating to wells:

* WellboreFeature (WellboreFeature) - a named entity representing a real, planned or conceptual well
* WellboreInterpretation (WellboreInterpretation) - one possible incarnation of a wellbore feature

There are various relationships between these classes. For example, a deviation survey or a trajectory must refer to a measured depth datum, and a blocked well must refer to a trajectory. Any of the representation objects can relate to a wellbore interpretation, which in turn must relate to a wellbore feature. The use of these optional organisational objects is encouraged and some software requires them to be present.

The measured depth datum class: MdDatum
---------------------------------------

