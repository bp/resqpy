"""A simple dictionary mapping resqml class names to more readable names."""

class_dict = {
    'obj_LocalDepth3dCrs': 'Coordinate Reference System (z is length)',
    'obj_LocalTime3dCrs': 'Coordinate Reference System (z is time)',
    'obj_IjkGridRepresentation': 'Grid (IJK)',
    'obj_UnstructuredColumnLayerGridRepresentation': 'Grid (Column Layer)',
    'obj_UnstructuredGridRepresentation': 'Grid (Unstructured)',
    'obj_GridConnectionSetRepresentation': 'Grid Connection Set',
    'obj_TruncatedIjkGridRepresentation': 'Grid (Truncated IJK)',
    'obj_TruncatedUnstructuredColumnLayerGridRepresentation': 'Grid (Truncated Column Layer)',
    'obj_GpGridRepresentation': 'Grid (General Purpose)',
    'obj_LocalGridSet': 'Local Grid Set',
    'obj_EpcExternalPartReference': 'HDF5 Reference',
    'obj_PropertyKind': 'Kind of Property',
    'obj_CategoricalProperty': 'Property (Categorical)',
    'obj_DiscreteProperty': 'Property (Discrete)',
    'obj_ContinuousProperty': 'Property (Continuous)',
    'obj_CategoricalPropertySeries': 'Property Series (Categorical)',
    'obj_DiscretePropertySeries': 'Property Series (Discrete)',
    'obj_ContinuousPropertySeries': 'Property Series (Continuous)',
    'obj_PropertySet': 'Property Set',
    'obj_TimeSeries': 'Time Series',
    'obj_WellboreTrajectoryRepresentation': 'Wellbore Trajectory',
    'obj_BlockedWellboreRepresentation': 'Blocked Wellbore',
    'obj_WellboreInterpretation': 'Wellbore Interpretation',
    'obj_WellboreFeature': 'Wellbore Feature',
    'obj_WellboreFrameRepresentation': 'Wellbore Frame',
    'obj_WellboreMarkerFrameRepresentation': 'Wellbore Marker Frame',
    'obj_DeviationSurveyRepresentation': 'Deviation Survey',
    'obj_MdDatum': 'Measured Depth Datum',
    # geo classes
    'obj_TectonicBoundaryFeature': 'Tectonic Boundary Feature (fault or fracture)',
    'obj_Grid2dRepresentation': 'Mesh (Grid 2D)',
    'obj_Grid2dSetRepresentation': 'Mesh Set (Grid 2D set)',
    'obj_SealedSurfaceFrameworkRepresentation': 'Sealed Surface Framework',
    'obj_SealedVolumeFrameworkRepresentation': 'Sealed Volume Framework',
    'obj_NonSealedSurfaceFrameworkRepresentation': 'Non-sealed Surface Framework',
    'obj_SeismicLineSetFeature': 'Seismic Line Set Feature',
    'obj_SeismicLineFeature': 'Seismic Line Feature',
    'obj_SeismicLatticeFeature': 'Seismic Lattice Feature',
    'obj_GeneticBoundaryFeature': 'Genetic Boundary Feature (horizon or geobody)',
    'obj_FaultInterpretation': 'Fault Interpretation',
    'obj_HorizonInterpretation': 'Horizon Interpretation',
    'obj_GenericFeatureInterpretation': 'Generic Feature Interpretation',
    'obj_EarthModelInterpretation': 'Earth Model Interpretation',
    'obj_OrganizationFeature': 'Organization Feature',
    'obj_StructuralOrganizationInterpretation': 'Structural Organization Interpretation',
    'obj_BoundaryFeature': 'Boundary Feature',
    'obj_BoundaryFeatureInterpretation': 'Boundary Feature Interpretation',
    'obj_FluidBoundaryFeature': 'Fluid Boundary Feature',
    'obj_FrontierFeature': 'Frontier Feature (area of interest boundary)',
    'obj_GeobodyFeature': 'Geobody Feature',
    'obj_GeobodyInterpretation': 'Geobody Interpretation',
    'obj_GeobodyBoundaryInterpretation': 'Geobody Boundary Interpretation',
    'obj_GeologicUnitFeature': 'Geological Unit Feature',
    'obj_GeologicUnitInterpretation': 'Geological Unit Interpretation',
    'obj_RockFluidOrganizationInterpretation': 'Rock Fluid Organization Interpretation',
    'obj_RockFluidUnitFeature': 'Rock Fluid Unit Feature',
    'obj_RockFluidUnitInterpretation': 'Rock Fluid Unit Interpretation',
    # stratigraphy classes
    'obj_GlobalChronostratigraphicColumn': 'Global Chronostratigraphic Column',
    'obj_StratigraphicColumn': 'Stratigraphic Column',
    'obj_StratigraphicColumnRankInterpretation': 'Stratigraphic Column Rank Interpretation',
    'obj_StratigraphicOccurrenceInterpretation': 'Stratigraphic Occurrence Interpretation',
    'obj_StratigraphicUnitFeature': 'Stratigraphic Unit Feature',
    'obj_StratigraphicUnitInterpretation': 'Stratigraphic Unit Interpretation',
    # low level classes
    'obj_SubRepresentation': 'Sub Representation',
    'obj_RepresentationSetRepresentation': 'Representation Set',
    'obj_StringTableLookup': 'String Lookup Table',
    'obj_DoubleTableLookup': 'Double (real) Lookup Table',
    'obj_TriangulatedSetRepresentation': 'Triangulated Set',
    'obj_PolylineSetRepresentation': 'Poly Line Set',
    'obj_PolylineRepresentation': 'Poly Line',
    'obj_PointSetRepresentation': 'Point Set',
    'obj_PlaneSetRepresentation': 'Plane Set',
    'obj_PointsProperty': 'Points Property',
    'obj_RedefinedGeometryRepresentation': 'Redefined Geometry Representation',
    # doc props bits
    'application/vnd.openxmlformats-package.core-properties+xml': 'Documentation (core properties)',
    'application/x-extended-core-properties+xml': 'Documentation (extended properties)',
    # other
    'obj_Activity': 'Activity',
    'obj_ActivityTemplate': 'Activity Template',
    'obj_CommentProperty': 'Comment Property',
    'obj_CommentPropertySeries': 'Comment Property Series',
    'obj_RepresentationIdentitySet': 'Representation Identity Set',
    'obj_StreamlinesFeature': 'Streamlines Feature',
    'obj_StreamlinesRepresentation': 'Streamlines'
}


def readable_class(class_name):
    """Given a resqml object class name as a string, returns a more human readable string.

    argument:
       class_name (string): the resqml class name, eg. 'obj_IjkGridRepresentation'

    returns:
       a human readable version of the class name, eg. 'Grid (IJK)'
    """

    if class_name in class_dict.keys():
        return class_dict[class_name]
    if class_name.startswith('obj_'):
        return class_name[4:]
    elif 'obj_' + class_name in class_dict.keys():
        return class_dict['obj_' + class_name]
    return class_name
