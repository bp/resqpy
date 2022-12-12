"""Submodule containing functions to create a grid xml file."""

import logging

log = logging.getLogger(__name__)

import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
from resqpy.olio.xml_namespaces import curly_namespace as ns

# these booleans control expansion of constant arrays
always_write_pillar_geometry_is_defined_array = False
always_write_cell_geometry_is_defined_array = False


def _create_grid_xml(grid,
                     ijk,
                     ext_uuid = None,
                     add_as_part = True,
                     add_relationships = True,
                     write_active = True,
                     write_geometry = True,
                     use_lattice = False):
    """Function that returns an xml representation containing grid information"""

    if grid.grid_representation and not write_geometry:
        rqet.create_metadata_xml(node = ijk, extra_metadata = {'grid_flavour': grid.grid_representation})

    if grid.represented_interpretation_uuid is not None:
        interp_title = grid.model.citation_title_for_part(grid.model.part_for_uuid(
            grid.represented_interpretation_uuid))
        grid.model.create_ref_node('RepresentedInterpretation',
                                   interp_title,
                                   grid.represented_interpretation_uuid,
                                   content_type = grid.model.type_of_uuid(grid.represented_interpretation_uuid),
                                   root = ijk)

    ni_node = rqet.SubElement(ijk, ns['resqml2'] + 'Ni')
    ni_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
    ni_node.text = str(grid.extent_kji[2])

    nj_node = rqet.SubElement(ijk, ns['resqml2'] + 'Nj')
    nj_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
    nj_node.text = str(grid.extent_kji[1])

    nk_node = rqet.SubElement(ijk, ns['resqml2'] + 'Nk')
    nk_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
    nk_node.text = str(grid.extent_kji[0])

    if grid.k_gaps:
        kg_node = rqet.SubElement(ijk, ns['resqml2'] + 'KGaps')
        kg_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'KGaps')
        kg_node.text = '\n'

        kgc_node = rqet.SubElement(kg_node, ns['resqml2'] + 'Count')
        kgc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        kgc_node.text = str(grid.k_gaps)

        assert grid.k_gap_after_array.ndim == 1 and grid.k_gap_after_array.size == grid.nk - 1

        kgal_node = rqet.SubElement(kg_node, ns['resqml2'] + 'GapAfterLayer')
        kgal_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
        kgal_node.text = '\n'

        kgal_values = rqet.SubElement(kgal_node, ns['resqml2'] + 'Values')
        kgal_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        kgal_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid, grid.uuid, 'GapAfterLayer', root = kgal_values)

    if grid.stratigraphic_column_rank_uuid is not None and grid.stratigraphic_units is not None:
        assert grid.model.type_of_uuid(
            grid.stratigraphic_column_rank_uuid) == 'obj_StratigraphicColumnRankInterpretation'

        strata_node = rqet.SubElement(ijk, ns['resqml2'] + 'IntervalStratigraphicUnits')
        strata_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntervalStratigraphicUnits')
        strata_node.text = '\n'

        ui_node = rqet.SubElement(strata_node, ns['resqml2'] + 'UnitIndices')
        ui_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        ui_node.text = '\n'

        ui_null = rqet.SubElement(ui_node, ns['resqml2'] + 'NullValue')
        ui_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        ui_null.text = '-1'

        ui_values = rqet.SubElement(ui_node, ns['resqml2'] + 'Values')
        ui_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        ui_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid, grid.uuid, 'unitIndices', root = ui_values)

        grid.model.create_ref_node('StratigraphicOrganization',
                                   grid.model.title(uuid = grid.stratigraphic_column_rank_uuid),
                                   grid.stratigraphic_column_rank_uuid,
                                   content_type = 'StratigraphicColumnRankInterpretation',
                                   root = strata_node)

    if grid.parent_window is not None:

        pw_node = rqet.SubElement(ijk, ns['resqml2'] + 'ParentWindow')
        pw_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IjkParentWindow')
        pw_node.text = '\n'

        assert grid.parent_grid_uuid is not None
        parent_grid_root = grid.model.root(uuid = grid.parent_grid_uuid)
        if parent_grid_root is None:
            pg_title = 'ParentGrid'
        else:
            pg_title = rqet.citation_title_for_node(parent_grid_root)
        grid.model.create_ref_node('ParentGrid',
                                   pg_title,
                                   grid.parent_grid_uuid,
                                   content_type = 'obj_IjkGridRepresentation',
                                   root = pw_node)

        for axis in range(3):

            regrid_node = rqet.SubElement(pw_node, 'KJI'[axis] + 'Regrid')
            regrid_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Regrid')
            regrid_node.text = '\n'

            if grid.is_refinement:
                if grid.parent_window.within_coarse_box is None:
                    iiopg = 0  # InitialIndexOnParentGrid
                else:
                    iiopg = grid.parent_window.within_coarse_box[0, axis]
            else:
                if grid.parent_window.within_fine_box is None:
                    iiopg = 0
                else:
                    iiopg = grid.parent_window.within_fine_box[0, axis]
            iiopg_node = rqet.SubElement(regrid_node, ns['resqml2'] + 'InitialIndexOnParentGrid')
            iiopg_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
            iiopg_node.text = str(iiopg)

            if grid.parent_window.fine_extent_kji[axis] == grid.parent_window.coarse_extent_kji[axis]:
                continue  # one-to-noe mapping

            intervals_node = rqet.SubElement(regrid_node, ns['resqml2'] + 'Intervals')
            intervals_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Intervals')
            intervals_node.text = '\n'

            if grid.parent_window.constant_ratios[axis] is not None:
                interval_count = 1
            else:
                if grid.is_refinement:
                    interval_count = grid.parent_window.coarse_extent_kji[axis]
                else:
                    interval_count = grid.parent_window.fine_extent_kji[axis]
            ic_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'IntervalCount')
            ic_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            ic_node.text = str(interval_count)

            pcpi_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'ParentCountPerInterval')
            pcpi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            pcpi_node.text = '\n'

            pcpi_values = rqet.SubElement(pcpi_node, ns['resqml2'] + 'Values')
            pcpi_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            pcpi_values.text = '\n'

            grid.model.create_hdf5_dataset_ref(ext_uuid,
                                               grid.uuid,
                                               'KJI'[axis] + 'Regrid/ParentCountPerInterval',
                                               root = pcpi_values)

            ccpi_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'ChildCountPerInterval')
            ccpi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            ccpi_node.text = '\n'

            ccpi_values = rqet.SubElement(ccpi_node, ns['resqml2'] + 'Values')
            ccpi_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            ccpi_values.text = '\n'

            grid.model.create_hdf5_dataset_ref(ext_uuid,
                                               grid.uuid,
                                               'KJI'[axis] + 'Regrid/ChildCountPerInterval',
                                               root = ccpi_values)

            if grid.is_refinement and not grid.parent_window.equal_proportions[axis]:
                ccw_node = rqet.SubElement(intervals_node, ns['resqml2'] + 'ChildCellWeights')
                ccw_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
                ccw_node.text = rqet.null_xml_text

                ccw_values_node = rqet.SubElement(ccw_node, ns['resqml2'] + 'Values')
                ccw_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
                ccw_values_node.text = rqet.null_xml_text

                grid.model.create_hdf5_dataset_ref(ext_uuid,
                                                   grid.uuid,
                                                   'KJI'[axis] + 'Regrid/ChildCellWeights',
                                                   root = ccw_values_node)

        # todo: handle omit and cell overlap functionality as part of parent window refining or coarsening

    if write_geometry:
        __add_geometry_xml(ext_uuid, grid, ijk, use_lattice)

    if add_as_part:
        __add_as_part(add_relationships, ext_uuid, grid, ijk, write_geometry)

    if (write_active and grid.active_property_uuid is not None and
            grid.model.part(uuid = grid.active_property_uuid) is None):
        # TODO: replace following with call to rprop.write_hdf5_and_create_xml_for_active_property()
        active_collection = rprop.PropertyCollection()
        active_collection.set_support(support = grid)
        active_collection.create_xml(None,
                                     None,
                                     'ACTIVE',
                                     'active',
                                     p_uuid = grid.active_property_uuid,
                                     discrete = True,
                                     add_min_max = False,
                                     find_local_property_kinds = True)

    return ijk


def __add_as_part(add_relationships, ext_uuid, grid, ijk, write_geometry):
    grid.model.add_part('obj_IjkGridRepresentation', grid.uuid, ijk)
    if add_relationships:
        if grid.stratigraphic_column_rank_uuid is not None and grid.stratigraphic_units is not None:
            grid.model.create_reciprocal_relationship(ijk, 'destinationObject',
                                                      grid.model.root_for_uuid(grid.stratigraphic_column_rank_uuid),
                                                      'sourceObject')
        if write_geometry:
            # create 2 way relationship between IjkGrid and Crs
            if grid.crs is None:
                crs_root = grid.model.root_for_uuid(grid.crs_uuid)
            else:
                crs_root = grid.crs.root
            grid.model.create_reciprocal_relationship(ijk, 'destinationObject', crs_root, 'sourceObject')
            # create 2 way relationship between IjkGrid and Ext
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = grid.model.root_for_part(ext_part)
            grid.model.create_reciprocal_relationship(ijk, 'mlToExternalPartProxy', ext_node, 'externalPartProxyToMl')
        # create relationship with parent grid
        if grid.parent_window is not None and grid.parent_grid_uuid is not None:
            grid.model.create_reciprocal_relationship(ijk, 'destinationObject',
                                                      grid.model.root_for_uuid(grid.parent_grid_uuid), 'sourceObject')

        if grid.represented_interpretation_uuid is not None:
            grid.model.create_reciprocal_relationship(ijk, 'destinationObject',
                                                      grid.model.root_for_uuid(grid.represented_interpretation_uuid),
                                                      'sourceObject')


def __add_geometry_xml(ext_uuid, grid, ijk, use_lattice):
    geom = rqet.SubElement(ijk, ns['resqml2'] + 'Geometry')
    geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'IjkGridGeometry')
    geom.text = '\n'
    # the remainder of this function is populating the geometry node
    grid.model.create_crs_reference(crs_uuid = grid.crs_uuid, root = geom)
    k_dir = rqet.SubElement(geom, ns['resqml2'] + 'KDirection')
    k_dir.set(ns['xsi'] + 'type', ns['resqml2'] + 'KDirection')
    if grid.k_direction_is_down:
        k_dir.text = 'down'
    else:
        k_dir.text = 'up'
    handed = rqet.SubElement(geom, ns['resqml2'] + 'GridIsRighthanded')
    handed.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
    handed.text = str(grid.grid_is_right_handed).lower()
    p_shape = rqet.SubElement(geom, ns['resqml2'] + 'PillarShape')
    p_shape.set(ns['xsi'] + 'type', ns['resqml2'] + 'PillarShape')
    p_shape.text = grid.pillar_shape

    if use_lattice:
        grid._add_geom_points_xml(geom, ext_uuid)  # usually calls _add_pillar_points_xml(), below
    else:
        _add_pillar_points_xml(grid, geom, ext_uuid)

    if grid.time_index is not None:

        assert grid.time_series_uuid is not None

        ti_node = rqet.SubElement(geom, ns['resqml2'] + 'TimeIndex')
        ti_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeIndex')
        ti_node.text = '\n'

        index_node = rqet.SubElement(ti_node, ns['resqml2'] + 'Index')
        index_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        index_node.text = str(grid.time_index)

        grid.model.create_ref_node('TimeSeries',
                                   grid.model.title(uuid = grid.time_series_uuid),
                                   grid.time_series_uuid,
                                   content_type = 'obj_TimeSeries',
                                   root = ti_node)


def _add_pillar_points_xml(grid, geom, ext_uuid):
    points_node = rqet.SubElement(geom, ns['resqml2'] + 'Points')
    points_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
    points_node.text = '\n'
    coords = rqet.SubElement(points_node, ns['resqml2'] + 'Coordinates')
    coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
    coords.text = '\n'
    grid.model.create_hdf5_dataset_ref(ext_uuid, grid.uuid, 'Points', root = coords)

    if always_write_pillar_geometry_is_defined_array or not grid.geometry_defined_for_all_pillars(cache_array = True):

        pillar_def = rqet.SubElement(geom, ns['resqml2'] + 'PillarGeometryIsDefined')
        pillar_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
        pillar_def.text = '\n'

        pd_values = rqet.SubElement(pillar_def, ns['resqml2'] + 'Values')
        pd_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        pd_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid, grid.uuid, 'PillarGeometryIsDefined', root = pd_values)

    else:

        _add_constant_pillar_geometry_is_defined(geom, grid.extent_kji)

    if always_write_cell_geometry_is_defined_array or not grid.geometry_defined_for_all_cells(cache_array = True):

        cell_def = rqet.SubElement(geom, ns['resqml2'] + 'CellGeometryIsDefined')
        cell_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanHdf5Array')
        cell_def.text = '\n'

        cd_values = rqet.SubElement(cell_def, ns['resqml2'] + 'Values')
        cd_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        cd_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid, grid.uuid, 'CellGeometryIsDefined', root = cd_values)

    else:

        _add_constant_cell_geometry_is_defined(geom, grid.extent_kji)

    if grid.has_split_coordinate_lines:

        scl = rqet.SubElement(geom, ns['resqml2'] + 'SplitCoordinateLines')
        scl.set(ns['xsi'] + 'type', ns['resqml2'] + 'ColumnLayerSplitCoordinateLines')
        scl.text = '\n'

        scl_count = rqet.SubElement(scl, ns['resqml2'] + 'Count')
        scl_count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        scl_count.text = str(grid.split_pillars_count)

        pi_node = rqet.SubElement(scl, ns['resqml2'] + 'PillarIndices')
        pi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        pi_node.text = '\n'

        pi_null = rqet.SubElement(pi_node, ns['resqml2'] + 'NullValue')
        pi_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        pi_null.text = str((grid.extent_kji[1] + 1) * (grid.extent_kji[2] + 1))

        pi_values = rqet.SubElement(pi_node, ns['resqml2'] + 'Values')
        pi_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        pi_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid, grid.uuid, 'PillarIndices', root = pi_values)

        cpscl = rqet.SubElement(scl, ns['resqml2'] + 'ColumnsPerSplitCoordinateLine')
        cpscl.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlJaggedArray')
        cpscl.text = '\n'

        elements = rqet.SubElement(cpscl, ns['resqml2'] + 'Elements')
        elements.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        elements.text = '\n'

        el_null = rqet.SubElement(elements, ns['resqml2'] + 'NullValue')
        el_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        el_null.text = str(grid.extent_kji[1] * grid.extent_kji[2])

        el_values = rqet.SubElement(elements, ns['resqml2'] + 'Values')
        el_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        el_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid,
                                           grid.uuid,
                                           'ColumnsPerSplitCoordinateLine/elements',
                                           root = el_values)

        c_length = rqet.SubElement(cpscl, ns['resqml2'] + 'CumulativeLength')
        c_length.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
        c_length.text = '\n'

        cl_null = rqet.SubElement(c_length, ns['resqml2'] + 'NullValue')
        cl_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        cl_null.text = '0'

        cl_values = rqet.SubElement(c_length, ns['resqml2'] + 'Values')
        cl_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        cl_values.text = '\n'

        grid.model.create_hdf5_dataset_ref(ext_uuid,
                                           grid.uuid,
                                           'ColumnsPerSplitCoordinateLine/cumulativeLength',
                                           root = cl_values)


def _add_constant_pillar_geometry_is_defined(geom, extent_kji):

    pillar_def = rqet.SubElement(geom, ns['resqml2'] + 'PillarGeometryIsDefined')
    pillar_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanConstantArray')
    pillar_def.text = '\n'

    pd_value = rqet.SubElement(pillar_def, ns['resqml2'] + 'Value')
    pd_value.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
    pd_value.text = 'true'

    pd_count = rqet.SubElement(pillar_def, ns['resqml2'] + 'Count')
    pd_count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
    pd_count.text = str((extent_kji[1] + 1) * (extent_kji[2] + 1))


def _add_constant_cell_geometry_is_defined(geom, extent_kji):

    cell_def = rqet.SubElement(geom, ns['resqml2'] + 'CellGeometryIsDefined')
    cell_def.set(ns['xsi'] + 'type', ns['resqml2'] + 'BooleanConstantArray')
    cell_def.text = '\n'

    cd_value = rqet.SubElement(cell_def, ns['resqml2'] + 'Value')
    cd_value.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
    cd_value.text = 'true'

    cd_count = rqet.SubElement(cell_def, ns['resqml2'] + 'Count')
    cd_count.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
    cd_count.text = str(extent_kji[0] * extent_kji[1] * extent_kji[2])
