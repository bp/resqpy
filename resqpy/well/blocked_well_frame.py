"""Module holding functions relating to the mapping of welbore frame properties onto blocked wells."""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.property as rqp
import resqpy.well as rqw


def blocked_well_frame_contributions_list(bw, wbf):
    """Returns wellbore frame contributions to each cell of a blocked well.

    arguments:
        bw (BlockedWell): the blocked well to map the wellbore frame onto
        wbf (WellboreFrame): the wellbore frame to map to the blocked well

    returns:
        list of list of (int, float, float) with one entry per blocked well cell, and each
            entry being a list of (wellbore frame interval index,
                                   fraction of wellbore frame interval in cell,
                                   fraction of cell's wellbore interval in wellbore frame interval)
    """

    wb_fraction_of_wbf = []
    for wb_ii in range(bw.node_count - 1):
        if bw.grid_indices[wb_ii] < 0:
            continue
        cell_entry_md = bw.node_mds[wb_ii]
        cell_exit_md = bw.node_mds[wb_ii + 1]
        cell_dmd = cell_exit_md - cell_entry_md  # wellbore interval length within cell
        entry_wbf_i, entry_wbf_fr = wbf.interval_for_md(bw.node_mds[wb_ii])
        exit_wbf_i, exit_wbf_fr = wbf.interval_for_md(bw.node_mds[wb_ii + 1])
        if (exit_wbf_i == -1 and exit_wbf_fr < 0.5) or (entry_wbf_i == -1 and entry_wbf_fr > 0.5):
            wb_fraction_of_wbf.append([])  # cell is above or below all frame intervals
            continue
        if entry_wbf_i == -1:
            assert entry_wbf_fr < 0.5
            entry_wbf_i = 0
            entry_wbf_fr = 0.0
        if exit_wbf_i == -1:
            assert exit_wbf_fr > 0.5
            exit_wbf_i = wbf.node_count - 2
            exit_wbf_fr = 1.0
        assert exit_wbf_i >= entry_wbf_i
        if exit_wbf_i == entry_wbf_i:  # single frame interval contributing to this cell
            assert exit_wbf_fr >= entry_wbf_fr
            if exit_wbf_fr == entry_wbf_fr:  # kissing point
                wb_fraction_of_wbf.append([])
                continue
            cell_fr = (min(cell_exit_md, wbf.node_mds[exit_wbf_i + 1]) -
                       max(cell_entry_md, wbf.node_mds[entry_wbf_i])) / cell_dmd
            assert 0.0 <= cell_fr <= 1.0 + 1.0e-6
            wb_fraction_of_wbf.append([(entry_wbf_i, exit_wbf_fr - entry_wbf_fr, cell_fr)])
            continue
        # more than one frame interval contributes to this cell
        cell_fr = (wbf.node_mds[entry_wbf_i + 1] - max(cell_entry_md, wbf.node_mds[entry_wbf_i])) / cell_dmd
        assert 0.0 <= cell_fr <= 1.0
        contrib_list = [(entry_wbf_i, 1.0 - entry_wbf_fr, cell_fr)]
        for wbf_i in range(entry_wbf_i + 1, exit_wbf_i):  # these frame intervals entirely within cell
            contrib_list.append((wbf_i, 1.0, (wbf.node_mds[wbf_i + 1] - wbf.node_mds[wbf_i]) / cell_dmd))
        cell_fr = (min(cell_exit_md, wbf.node_mds[exit_wbf_i + 1]) - wbf.node_mds[exit_wbf_i]) / cell_dmd
        assert 0.0 <= cell_fr <= 1.0
        contrib_list.append((exit_wbf_i, exit_wbf_fr, cell_fr))
        wb_fraction_of_wbf.append(contrib_list)

    return wb_fraction_of_wbf


def add_blocked_well_properties_from_wellbore_frame(bw,
                                                    frame_uuid = None,
                                                    property_kinds_list = None,
                                                    realization = None,
                                                    set_length = None,
                                                    set_perforation_fraction = None,
                                                    set_frame_interval = False):
    """Add properties to this blocked well by derivation from wellbore frame intervals properties.

    arguments:
        bw (BlockedWell): the blocked well to add properties to
        frame_uuid (UUID, optional): the uuid of the wellbore frame to source properties from; if None, a
            solitary wellbore frame relating to the same trajectory as the blocked well will be used
        property_kinds_list (list of str, optional): if present, a list of handled property kinds which
            are to be set from the wellbore frame properties; if None, any handled property kinds that
            are present for the wellbore frame will be used
        realization (int, optional): if present, wellbore frame properties will be filtered by this
            realization number and it will be assigned to the blocked well properties that are created
        set_length (bool, optional): if True, a length property will be generated based on active
            measured depth intervals; if None, will be set True if length is in list of property kinds
            being processed
        set_perforation_fraction (bool, optional): if True, a perforation fraction property will be
            created based on the fraction of the measured depth within a blocked well cell that is
            flagged as active, ie. perforated at some time; if None, it will be created only if length
            and permeability thickness are both absent
        set_frame_interval (bool, default False): if True, a static discrete property holding the index
            of the dominant active wellbore frame interval (per blocked well cell) is created

    returns:
        list of uuids of created property parts (does not include any copied time series object)

    notes:
        this method is designed to set up some blocked well properties based on similar properties
        already established on a special wellbore frame, mainly for perforations and open hole completions;
        frame_uuid should be specified if there are well logs in the dataset, or other wellbore frames;
        if a permeability thickness property is being set based on a wellbore frame property, the value
        is divided between blocked well cells based solely on measured depth interval lengths, without
        reference to grid properties such as net to gross ratio or permeability;
        titles will be the same as those used in the frame properties, and 'PPERF' for partial
        perforation;
        if set_frame_interval is True, the resulting property will be given a soft relationship with the
        wellbore frame (in addition to its supporting representation reference relationship with the
        blocked well); a null value of -1 is used where no active frame interval is present in a cell;
        units of measure will also be the same as those in the wellbore frame;
        this method only supports single grid blocked wells at present;
        blocked well and wellbore frame must be in the same model
    """

    # note: 'active' ia a static property kind used to indicate that a cell is perforated or otherwise
    # open to flow at some time, so needs to appear somewhere in well specification data for flow simulation;
    # 'well connection open' is a dynamic indicator used to identify time periods when a connection is open
    handled_property_kinds = [
        'active', 'well connection open', 'length', 'permeability length', 'skin', 'wellbore radius',
        'nonDarcy flow coefficient'
    ]

    def pk_pc_and_ts(pc, pk):
        """Returns a selective property collection based on property kind and a time series uuid if used."""
        pk_pc = rqp.selective_version_of_collection(pc, property_kind = pk)
        ts_uuids = pk_pc.time_series_uuid_list(sort_list = False)
        assert 0 <= len(ts_uuids) <= 1
        ts_uuid = None if len(ts_uuids) == 0 else ts_uuids[0]
        assert ts_uuid is not None or pk_pc.number_of_parts() == 1
        return pk_pc, ts_uuid

    assert len(bw.grid_list) == 1,  \
        'blocked well references more than one grid; not supported for frame properties'

    if frame_uuid is None:  # note: there could be multiple wellbore frames if well logs are in the dataset
        frame_uuid = bw.model.uuid(obj_type = 'WellboreFrameRepresentation', related_uuid = bw.trajectory.uuid)
    assert frame_uuid is not None, f'wellbore frame neither specified nor found for trajectory: {bw.trajectory.title}'

    if property_kinds_list is not None:
        for pk in property_kinds_list:
            assert pk in handled_property_kinds, f'blocked well from frame property kind not supported: {pk}'

    # establish a property collection for this blocked well
    wb_pc = bw.extract_property_collection()

    # load wellbore frame and its properties
    wbf = rqw.WellboreFrame(bw.model, uuid = frame_uuid)
    wbf_pc = wbf.extract_property_collection()
    wbf_pc = rqp.selective_version_of_collection(wbf_pc, indexable = 'intervals', realization = realization)
    if wbf_pc.number_of_parts() == 0:
        log.warning(f'no intervals properties found for wellbore frame: {wbf.title}; realization: {realization}')
        return []

    # determine the actual property kinds list that will be worked with
    pk_list = []
    wbf_pk_list = wbf_pc.property_kind_list(sort_list = False)
    for pk in wbf_pk_list:
        if ((property_kinds_list is None and pk in handled_property_kinds) or
            (property_kinds_list is not None and pk in property_kinds_list)):
            pk_list.append(pk)
    if len(pk_list) == 0:
        log.warning('no appropriate property kinds found for wellbore frame: {wbf.title}')

    if set_length is None:
        set_length = ('length' in pk_list)

    if set_perforation_fraction is None:
        set_perforation_fraction = (not set_length and 'permeability length' not in pk_list)

    log.debug(f'working list of property kinds for: {wbf.title}: {pk_list}')

    # find mapping of wellbpre frame onto blocked well cells, based on measured depths
    # list of entry per cell:
    # entry is list of triplets of (frame interval index, fraction of frame interval, fraction of cell md interval)
    wb_fraction_of_wbf = bw.frame_contributions_list(wbf)
    assert len(wb_fraction_of_wbf) == bw.cell_count

    # inherit static properties from wellbore frame to blocked well; resampling mehtod depends on property kind
    # TODO: make length and perforation fraction properties dynamic
    wb_active_array = None
    if 'active' in pk_list or set_length or set_perforation_fraction or set_frame_interval:
        wbf_p = wbf_pc.singleton(property_kind = 'active')
        assert wbf_p is not None, 'problem with active wellbore frame property'
        wbf_a = wbf_pc.cached_part_array_ref(wbf_p)
        wb_a = np.zeros(bw.cell_count, dtype = bool)
        length = np.zeros(bw.cell_count, dtype = float)
        pperf = np.zeros(bw.cell_count, dtype = float)
        dominant_wbf_interval = np.full(bw.cell_count, -1, dtype = int)
        ci = 0
        for wb_ii in range(bw.node_count - 1):
            if bw.grid_indices[wb_ii] < 0:
                continue
            i = ci
            ci += 1
            wbf_contributions = wb_fraction_of_wbf[i]
            max_active_cell_fr = 0.0
            max_active_wbf_ii = None
            for (wbf_ii, _, cell_fr) in wbf_contributions:
                if wbf_a[wbf_ii]:
                    wb_a[i] = True  # any contributing frame interval active makes cell active
                    pperf[i] += cell_fr
                    if cell_fr > max_active_cell_fr:
                        max_active_cell_fr = cell_fr
                        max_active_wbf_ii = wbf_ii
            if pperf[i] > 1.0:
                pperf[i] = 1.0
            if max_active_wbf_ii is not None:
                dominant_wbf_interval[i] = max_active_wbf_ii
            length[i] = pperf[i] * (bw.node_mds[wb_ii + 1] - bw.node_mds[wb_ii])
        wb_active_array = wb_a.copy()
        if set_frame_interval:
            # position this first to make it easy to get its uuid, for adding a soft relationship
            wb_pc.add_cached_array_to_imported_list(dominant_wbf_interval,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    'wellbore frame interval',
                                                    discrete = True,
                                                    property_kind = 'wellbore frame interval',
                                                    null_value = -1,
                                                    realization = realization,
                                                    indexable_element = 'cells')
        if 'active' in pk_list:
            wb_pc.add_cached_array_to_imported_list(wb_a,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    wbf_pc.citation_title_for_part(wbf_p),
                                                    discrete = True,
                                                    property_kind = 'active',
                                                    realization = realization,
                                                    indexable_element = 'cells')
        if set_length:
            md_uom = bw.trajectory.md_uom
            wb_pc.add_cached_array_to_imported_list(length,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    'LENGTH',
                                                    discrete = False,
                                                    uom = f'{md_uom}',
                                                    property_kind = 'length',
                                                    realization = realization,
                                                    indexable_element = 'cells')
        if set_perforation_fraction:
            md_uom = bw.trajectory.md_uom
            md_uom = 'ft' if md_uom.startswith('ft') else 'm'
            wb_pc.add_cached_array_to_imported_list(pperf,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    'PPERF',
                                                    discrete = False,
                                                    uom = f'{md_uom}/{md_uom}',
                                                    property_kind = 'perforation fraction',
                                                    realization = realization,
                                                    indexable_element = 'cells')

    if 'wellbore radius' in pk_list:
        wbf_p = wbf_pc.singleton(property_kind = 'wellbore radius')
        assert wbf_p is not None, 'problem with wellbore radius wellbore frame property'
        wbf_a = wbf_pc.cached_part_array_ref(wbf_p)
        wb_a = np.full(bw.cell_count, np.NaN, dtype = float)
        for i, wbf_contributions in enumerate(wb_fraction_of_wbf):
            if len(wbf_contributions) == 0:
                continue  # todo: could try to inherit wellbore radius from above or below?
            if np.all(np.isnan(wbf_a[wbf_contributions[0][0]:wbf_contributions[-1][0] + 1])):
                continue
            # use a maximum wellbore radius for the whole cell; over-optimistic in case of variation
            max_r = np.nanmax(wbf_a[wbf_contributions[0][0]:wbf_contributions[-1][0] + 1])
            if np.nanmin(wbf_a[wbf_contributions[0][0]:wbf_contributions[-1][0] + 1]) < max_r - 1.0e-6:
                log.warning(
                    f'radius of wellbore for {wbf.title} changes within cell {i} for blocked wellbore {bw.title}')
            # todo: find modal average or measured length weighted mean?; for now, use topmost (largest) radius
            wb_a[i] = max_r
        wb_pc.add_cached_array_to_imported_list(wb_a,
                                                f'derived from wellbore frame {wbf.title}',
                                                wbf_pc.citation_title_for_part(wbf_p),
                                                discrete = False,
                                                uom = wbf_pc.uom_for_part(wbf_p),
                                                property_kind = 'wellbore radius',
                                                realization = realization,
                                                indexable_element = 'cells')

    # if any dynamic, copy time series (possibly reusing an existing one, as likely to be shared with other wells)
    if 'well connection open' in pk_list:
        wco_pc, ts_uuid = pk_pc_and_ts(wbf_pc, 'well connection open')
        for wco_part in wco_pc.parts():
            wbf_wco = wco_pc.cached_part_array_ref(wco_part)
            assert wbf_wco is not None
            ti = wco_pc.time_index_for_part(wco_part)
            wb_wco = np.zeros(bw.cell_count, dtype = bool)
            for i, wbf_contributions in enumerate(wb_fraction_of_wbf):
                if len(wbf_contributions) == 0:
                    continue
                some_active_closed = False
                for (wbf_ii, _, _) in wbf_contributions:
                    if wbf_wco[wbf_ii]:
                        wb_wco[i] = True  # any contributing frame interval active makes cell active
                    elif wb_active_array is not None and wb_active_array[i]:
                        some_active_closed = True
                if wb_wco[i] and some_active_closed:
                    log.error('unsyncronised opening of active intervals ' +
                              f'at time index {ti} within cell {i} in blocked well {bw.title}')
            wb_pc.add_cached_array_to_imported_list(wb_wco,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    wco_pc.citation_title_for_part(wco_part),
                                                    discrete = True,
                                                    property_kind = 'well connection open',
                                                    time_index = ti,
                                                    time_series_uuid = ts_uuid,
                                                    realization = realization,
                                                    indexable_element = 'cells')

    if 'permeability length' in pk_list:
        kh_pc, ts_uuid = pk_pc_and_ts(wbf_pc, 'permeability length')
        kh_uom = None
        for kh_part in kh_pc.parts():
            wbf_kh = kh_pc.cached_part_array_ref(kh_part)
            assert wbf_kh is not None
            if kh_uom is None:
                kh_uom = kh_pc.uom_for_part(kh_part)
            else:
                assert kh_pc.uom_for_part(kh_part) == kh_uom, 'mixed permeability length units of measure'
            wb_kh = np.zeros(bw.cell_count, dtype = float)
            for i, wbf_contributions in enumerate(wb_fraction_of_wbf):
                if len(wbf_contributions) == 0:
                    continue
                for (wbf_ii, wbf_fr, _) in wbf_contributions:
                    if not np.isnan(wbf_kh[wbf_ii]):
                        wb_kh[i] += wbf_kh[wbf_ii] * wbf_fr
            wb_pc.add_cached_array_to_imported_list(wb_kh,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    kh_pc.citation_title_for_part(kh_part),
                                                    discrete = False,
                                                    uom = kh_uom,
                                                    property_kind = 'permeability length',
                                                    time_index = kh_pc.time_index_for_part(kh_part),
                                                    time_series_uuid = ts_uuid,
                                                    realization = realization,
                                                    indexable_element = 'cells')

    if 'skin' in pk_list:
        # note: here the skin contributions are weighted by fraction of measured length within a cell
        # this will not give a technically correct result if the permeability varies between frame intervals
        skin_pc, ts_uuid = pk_pc_and_ts(wbf_pc, 'skin')
        skin_uom = None  # probably always Euc
        for skin_part in skin_pc.parts():
            wbf_skin = skin_pc.cached_part_array_ref(skin_part)
            assert wbf_skin is not None
            if skin_uom is None:
                skin_uom = skin_pc.uom_for_part(skin_part)
            else:
                assert skin_pc.uom_for_part(skin_part) == skin_uom, 'mixed skin units of measure'
            wb_skin = np.full(bw.cell_count, np.NaN, dtype = float)
            for i, wbf_contributions in enumerate(wb_fraction_of_wbf):
                if len(wbf_contributions) == 0:
                    continue
                fr_sum = 0.0
                for (wbf_ii, _, cell_fr) in wbf_contributions:
                    if np.isnan(wbf_skin[wbf_ii]):
                        continue
                    if np.isnan(wb_skin[i]):
                        wb_skin[i] = 0.0
                    wb_skin[i] += wbf_skin[wbf_ii] * cell_fr
                    fr_sum += cell_fr
                if fr_sum > 0.0:
                    wb_skin[i] /= fr_sum
            wb_pc.add_cached_array_to_imported_list(wb_skin,
                                                    f'derived from wellbore frame {wbf.title}',
                                                    skin_pc.citation_title_for_part(skin_part),
                                                    discrete = False,
                                                    uom = skin_uom,
                                                    property_kind = 'skin',
                                                    time_index = skin_pc.time_index_for_part(skin_part),
                                                    time_series_uuid = ts_uuid,
                                                    realization = realization,
                                                    indexable_element = 'cells')

    # TODO: non Darcy flow coefficient
    # D factor is complicated: it appears in the inflow equation as a rate dependent adjustment to skin

    wb_pc.write_hdf5_for_imported_list()
    uuids = wb_pc.create_xml_for_imported_list_and_add_parts_to_model()

    if set_frame_interval:
        bw.model.create_reciprocal_relationship_uuids(uuids[0], 'sourceObject', frame_uuid, 'destinationObject')

    return uuids
