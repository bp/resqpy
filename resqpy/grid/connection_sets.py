"""A submodule containing functions relating to grid connection sets."""
import logging

log = logging.getLogger(__name__)

import resqpy.fault as rqf
import resqpy.olio.transmission as rqtr
import resqpy.property as rprop


def fault_connection_set(grid,
                         skip_inactive = True,
                         compute_transmissibility = False,
                         add_to_model = False,
                         realization = None,
                         inherit_features_from = None,
                         title = 'fault juxtaposition set'):
    """Returns (and caches) a GridConnectionSet representing juxtaposition across faces with split pillars.

    arguments:
       skip_inactive (boolean, default True): if True, then cell face pairs involving an inactive cell will
          be omitted from the results
       compute_transmissibilities (boolean, default False): if True, then transmissibilities will be computed
          for the cell face pairs (unless already existing as a cached attribute of the grid)
       add_to_model (boolean, default False): if True, the connection set is written to hdf5 and xml is created;
          if compute_transmissibilty is True then the transmissibility property is also added
       realization (int, optional): if present, is used as the realization number when adding transmissibility
          property to model; ignored if compute_transmissibility is False
       inherit_features_from (GridConnectionSet, optional): if present, the features (named faults) are
          inherited from this grid connection set based on a match of either cell face in a juxtaposed pair
       title (string, default 'fault juxtaposition set'): the citation title to use if adding to model

    returns:
       GridConnectionSet, numpy float array of shape (count,) transmissibilities (or None), where count is the
       number of cell face pairs in the grid connection set, which contains entries for all juxtaposed faces
       with a split pillar as an edge; if the grid does not have split pillars (ie. is unfaulted) or there
       are no qualifying connections, (None, None) is returned
    """

    if not hasattr(grid, 'fgcs') or grid.fgcs_skip_inactive != skip_inactive:
        grid.fgcs, grid.fgcs_fractional_area = rqtr.fault_connection_set(grid, skip_inactive = skip_inactive)
        grid.fgcs_skip_inactive = skip_inactive

    if grid.fgcs is None:
        return None, None

    new_tr = False
    if compute_transmissibility and not hasattr(grid, 'array_fgcs_transmissibility'):
        grid.array_fgcs_transmissibility = grid.fgcs.tr_property_array(grid.fgcs_fractional_area)
        new_tr = True

    tr = grid.array_fgcs_transmissibility if hasattr(grid, 'array_fgcs_transmissibility') else None

    if inherit_features_from is not None:
        grid.fgcs.inherit_features(inherit_features_from)
        grid.fgcs.clean_feature_list()

    if add_to_model:
        if grid.model.uuid(uuid = grid.fgcs.uuid) is None:
            grid.fgcs.write_hdf5()
            grid.fgcs.create_xml(title = title)
        if new_tr:
            tr_pc = rprop.PropertyCollection()
            tr_pc.set_support(support = grid.fgcs)
            tr_pc.add_cached_array_to_imported_list(
                grid.array_fgcs_transmissibility,
                'computed for faces with split pillars',
                'fault transmissibility',
                discrete = False,
                uom = 'm3.cP/(kPa.d)' if grid.xy_units() == 'm' else 'bbl.cP/(psi.d)',
                property_kind = 'transmissibility',
                realization = realization,
                indexable_element = 'faces',
                count = 1)
            tr_pc.write_hdf5_for_imported_list()
            tr_pc.create_xml_for_imported_list_and_add_parts_to_model()

    return grid.fgcs, tr


def pinchout_connection_set(grid,
                            skip_inactive = True,
                            compute_transmissibility = False,
                            add_to_model = False,
                            realization = None):
    """Returns (and caches) a GridConnectionSet representing juxtaposition across pinched out cells.

    arguments:
       skip_inactive (boolean, default True): if True, then cell face pairs involving an inactive cell will
          be omitted from the results
       compute_transmissibilities (boolean, default False): if True, then transmissibilities will be computed
          for the cell face pairs (unless already existing as a cached attribute of the grid)
       add_to_model (boolean, default False): if True, the connection set is written to hdf5 and xml is created;
          if compute_transmissibilty is True then the transmissibility property is also added
       realization (int, optional): if present, is used as the realization number when adding transmissibility
          property to model; ignored if compute_transmissibility is False

    returns:
       GridConnectionSet, numpy float array of shape (count,) transmissibilities (or None), where count is the
       number of cell face pairs in the grid connection set, which contains entries for all juxtaposed K faces
       separated logically by pinched out (zero thickness) cells; if there are no pinchouts (or no qualifying
       connections) then (None, None) will be returned
    """

    if not hasattr(grid, 'pgcs') or grid.pgcs_skip_inactive != skip_inactive:
        grid.pgcs = rqf.pinchout_connection_set(grid, skip_inactive = skip_inactive)
        grid.pgcs_skip_inactive = skip_inactive

    if grid.pgcs is None:
        return None, None

    new_tr = False
    if compute_transmissibility and not hasattr(grid, 'array_pgcs_transmissibility'):
        grid.array_pgcs_transmissibility = grid.pgcs.tr_property_array()
        new_tr = True

    tr = grid.array_pgcs_transmissibility if hasattr(grid, 'array_pgcs_transmissibility') else None

    if add_to_model:
        if grid.model.uuid(uuid = grid.pgcs.uuid) is None:
            grid.pgcs.write_hdf5()
            grid.pgcs.create_xml()
        if new_tr:
            tr_pc = rprop.PropertyCollection()
            tr_pc.set_support(support = grid.pgcs)
            tr_pc.add_cached_array_to_imported_list(
                tr,
                'computed for faces across pinchouts',
                'pinchout transmissibility',
                discrete = False,
                uom = 'm3.cP/(kPa.d)' if grid.xy_units() == 'm' else 'bbl.cP/(psi.d)',
                property_kind = 'transmissibility',
                realization = realization,
                indexable_element = 'faces',
                count = 1)
            tr_pc.write_hdf5_for_imported_list()
            tr_pc.create_xml_for_imported_list_and_add_parts_to_model()

    return grid.pgcs, tr


def k_gap_connection_set(grid,
                         skip_inactive = True,
                         compute_transmissibility = False,
                         add_to_model = False,
                         realization = None,
                         tolerance = 0.001):
    """Returns (and caches) a GridConnectionSet representing juxtaposition across zero thickness K gaps.

    arguments:
       skip_inactive (boolean, default True): if True, then cell face pairs involving an inactive cell will
          be omitted from the results
       compute_transmissibilities (boolean, default False): if True, then transmissibilities will be computed
          for the cell face pairs (unless already existing as a cached attribute of the grid)
       add_to_model (boolean, default False): if True, the connection set is written to hdf5 and xml is created;
          if compute_transmissibilty is True then the transmissibility property is also added
       realization (int, optional): if present, is used as the realization number when adding transmissibility
          property to model; ignored if compute_transmissibility is False
       tolerance (float, default 0.001): the maximum K gap thickness that will be 'bridged' by a connection;
          units are implicitly those of the z units in the grid's coordinate reference system

    returns:
       GridConnectionSet, numpy float array of shape (count,) transmissibilities (or None), where count is the
       number of cell face pairs in the grid connection set, which contains entries for all juxtaposed K faces
       separated logically by pinched out (zero thickness) cells; if there are no pinchouts (or no qualifying
       connections) then (None, None) will be returned

    note:
       if cached values are found they are returned regardless of the specified tolerance
    """

    if not hasattr(grid, 'kgcs') or grid.kgcs_skip_inactive != skip_inactive:
        grid.kgcs = rqf.k_gap_connection_set(grid, skip_inactive = skip_inactive, tolerance = tolerance)
        grid.kgcs_skip_inactive = skip_inactive

    if grid.kgcs is None:
        return None, None

    new_tr = False
    if compute_transmissibility and not hasattr(grid, 'array_kgcs_transmissibility'):
        grid.array_kgcs_transmissibility = grid.kgcs.tr_property_array()
        new_tr = True

    tr = grid.array_kgcs_transmissibility if hasattr(grid, 'array_kgcs_transmissibility') else None

    if add_to_model:
        if grid.model.uuid(uuid = grid.kgcs.uuid) is None:
            grid.kgcs.write_hdf5()
            grid.kgcs.create_xml()
        if new_tr:
            tr_pc = rprop.PropertyCollection()
            tr_pc.set_support(support = grid.kgcs)
            tr_pc.add_cached_array_to_imported_list(
                tr,
                'computed for faces across zero thickness K gaps',
                'K gap transmissibility',
                discrete = False,
                uom = 'm3.cP/(kPa.d)' if grid.xy_units() == 'm' else 'bbl.cP/(psi.d)',
                property_kind = 'transmissibility',
                realization = realization,
                indexable_element = 'faces',
                count = 1)
            tr_pc.write_hdf5_for_imported_list()
            tr_pc.create_xml_for_imported_list_and_add_parts_to_model()

    return grid.kgcs, tr
