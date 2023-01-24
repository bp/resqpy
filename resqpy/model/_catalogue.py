"""_catalogue.py: RESQML parts (high level objects) catalogue functions."""

import logging

log = logging.getLogger(__name__)

import zipfile as zf

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


def _parts(model,
           parts_list = None,
           obj_type = None,
           uuid = None,
           title = None,
           title_mode = 'is',
           title_case_sensitive = False,
           metadata = {},
           extra = {},
           related_uuid = None,
           related_mode = None,
           epc_subdir = None,
           sort_by = None):
    """Returns a list of parts matching all of the arguments passed."""

    if not parts_list:
        parts_list = _list_of_parts(model)
    if uuid is not None:
        part_name = model.uuid_part_dict.get(bu.uuid_as_int(uuid))
        if part_name is None or part_name not in parts_list:
            return []
        parts_list = [part_name]
    if epc_subdir:
        parts_list = _filtered_by_epc_subdir(model, parts_list, epc_subdir)
    if obj_type:
        if obj_type[0].isupper():
            obj_type = 'obj_' + obj_type
        filtered_list = []
        for part in parts_list:
            if model.parts_forest[part][0] == obj_type:
                filtered_list.append(part)
        if len(filtered_list) == 0:
            return []
        parts_list = filtered_list
    if title:
        parts_list = _filtered_by_title(model, parts_list, title, title_mode, title_case_sensitive)
    if metadata:
        parts_list = _filtered_by_metadata(model, parts_list, metadata)
    if extra:
        parts_list = _filtered_by_extra(model, parts_list, extra)
    if related_uuid is not None:
        parts_list = _parts_list_filtered_by_related_uuid(model, parts_list, related_uuid, related_mode = related_mode)
    if sort_by and len(parts_list):
        parts_list = _sorted_parts_list(model, parts_list, sort_by)
    return parts_list


def _part(model,
          parts_list = None,
          obj_type = None,
          uuid = None,
          title = None,
          title_mode = 'is',
          title_case_sensitive = False,
          metadata = {},
          extra = {},
          related_uuid = None,
          related_mode = None,
          epc_subdir = None,
          multiple_handling = 'exception'):
    """Returns the name of a part matching all of the arguments passed."""

    pl = _parts(model,
                parts_list = parts_list,
                obj_type = obj_type,
                uuid = uuid,
                title = title,
                title_mode = title_mode,
                title_case_sensitive = title_case_sensitive,
                metadata = metadata,
                extra = extra,
                related_uuid = related_uuid,
                related_mode = related_mode,
                epc_subdir = epc_subdir)
    if len(pl) == 0:
        return None
    if len(pl) == 1 or multiple_handling == 'first':
        return pl[0]
    if multiple_handling == 'none':
        return None
    elif multiple_handling in ['newest', 'oldest']:
        sorted_list = _sort_parts_list_by_timestamp(model, pl)
        if multiple_handling == 'newest':
            return sorted_list[0]
        return sorted_list[-1]
    else:
        raise ValueError('more than one part matches criteria')


def _uuids(model,
           parts_list = None,
           obj_type = None,
           uuid = None,
           title = None,
           title_mode = 'is',
           title_case_sensitive = False,
           metadata = {},
           extra = {},
           related_uuid = None,
           related_mode = None,
           epc_subdir = None,
           sort_by = None):
    """Returns a list of uuids of parts matching all of the arguments passed."""

    sort_by_uuid = (sort_by == 'uuid')
    if sort_by_uuid:
        sort_by = None
    pl = _parts(model,
                parts_list = parts_list,
                obj_type = obj_type,
                uuid = uuid,
                title = title,
                title_mode = title_mode,
                title_case_sensitive = title_case_sensitive,
                metadata = metadata,
                extra = extra,
                related_uuid = related_uuid,
                related_mode = related_mode,
                epc_subdir = epc_subdir,
                sort_by = sort_by)
    if len(pl) == 0:
        return []
    uuid_list = []
    for part in pl:
        uuid_list.append(_uuid_for_part(model, part))
    if sort_by_uuid:
        uuid_list.sort()
    return uuid_list


def _uuid(model,
          parts_list = None,
          obj_type = None,
          uuid = None,
          title = None,
          title_mode = 'is',
          title_case_sensitive = False,
          metadata = {},
          extra = {},
          related_uuid = None,
          related_mode = None,
          epc_subdir = None,
          multiple_handling = 'exception'):
    """Returns the uuid of a part matching all of the arguments passed."""

    part = _part(model,
                 parts_list = parts_list,
                 obj_type = obj_type,
                 uuid = uuid,
                 title = title,
                 title_mode = title_mode,
                 title_case_sensitive = title_case_sensitive,
                 metadata = metadata,
                 extra = extra,
                 related_uuid = related_uuid,
                 related_mode = related_mode,
                 epc_subdir = epc_subdir,
                 multiple_handling = multiple_handling)
    if part is None:
        return None
    return rqet.uuid_in_part_name(part)


def _roots(model,
           parts_list = None,
           obj_type = None,
           uuid = None,
           title = None,
           title_mode = 'is',
           title_case_sensitive = False,
           metadata = {},
           extra = {},
           related_uuid = None,
           related_mode = None,
           epc_subdir = None,
           sort_by = None):
    """Returns a list of xml root nodes of parts matching all of the arguments passed."""

    pl = _parts(model,
                parts_list = parts_list,
                obj_type = obj_type,
                uuid = uuid,
                title = title,
                title_mode = title_mode,
                title_case_sensitive = title_case_sensitive,
                metadata = metadata,
                extra = extra,
                related_uuid = related_uuid,
                related_mode = related_mode,
                epc_subdir = epc_subdir,
                sort_by = sort_by)
    root_list = []
    for part in pl:
        root_list.append(_root_for_part(model, part))
    return root_list


def _root(model,
          parts_list = None,
          obj_type = None,
          uuid = None,
          title = None,
          title_mode = 'is',
          title_case_sensitive = False,
          metadata = {},
          extra = {},
          related_uuid = None,
          related_mode = None,
          epc_subdir = None,
          multiple_handling = 'exception'):
    """Returns the xml root node of a part matching all of the arguments passed."""

    part = _part(model,
                 parts_list = parts_list,
                 obj_type = obj_type,
                 uuid = uuid,
                 title = title,
                 title_mode = title_mode,
                 title_case_sensitive = title_case_sensitive,
                 metadata = metadata,
                 extra = extra,
                 related_uuid = related_uuid,
                 related_mode = related_mode,
                 epc_subdir = epc_subdir,
                 multiple_handling = multiple_handling)
    if part is None:
        return None
    return _root_for_part(model, part)


def _titles(model,
            parts_list = None,
            obj_type = None,
            uuid = None,
            title = None,
            title_mode = 'is',
            title_case_sensitive = False,
            metadata = {},
            extra = {},
            related_uuid = None,
            related_mode = None,
            epc_subdir = None,
            sort_by = None):
    """Returns a list of citation titles of parts matching all of the arguments passed."""

    pl = _parts(model,
                parts_list = parts_list,
                obj_type = obj_type,
                uuid = uuid,
                title = title,
                title_mode = title_mode,
                title_case_sensitive = title_case_sensitive,
                metadata = metadata,
                extra = extra,
                related_uuid = related_uuid,
                related_mode = related_mode,
                epc_subdir = epc_subdir,
                sort_by = sort_by)
    title_list = []
    for part in pl:
        title_list.append(_citation_title_for_part(model, part))
    return title_list


def _title(model,
           parts_list = None,
           obj_type = None,
           uuid = None,
           title = None,
           title_mode = 'is',
           title_case_sensitive = False,
           metadata = {},
           extra = {},
           related_uuid = None,
           related_mode = None,
           epc_subdir = None,
           multiple_handling = 'exception'):
    """Returns the citation title of a part matching all of the arguments passed."""

    part = _part(model,
                 parts_list = parts_list,
                 obj_type = obj_type,
                 uuid = uuid,
                 title = title,
                 title_mode = title_mode,
                 title_case_sensitive = title_case_sensitive,
                 metadata = metadata,
                 extra = extra,
                 related_uuid = related_uuid,
                 related_mode = related_mode,
                 epc_subdir = epc_subdir,
                 multiple_handling = multiple_handling)
    if part is None:
        return None
    return _citation_title_for_part(model, part)


def _parts_list_of_type(model, type_of_interest = None, uuid = None):
    """Returns a list of part names for parts of type of interest, optionally matching a uuid."""

    if type_of_interest and type_of_interest[0].isupper():
        type_of_interest = 'obj_' + type_of_interest

    if uuid is not None:
        part_name = model.uuid_part_dict.get(bu.uuid_as_int(uuid))
        if part_name is None or (type_of_interest is not None and
                                 (model.parts_forest[part_name][0] != type_of_interest)):
            return []
        return [part_name]

    parts_list = []
    for part_name in model.parts_forest:
        if type_of_interest is None or model.parts_forest[part_name][0] == type_of_interest:
            parts_list.append(part_name)
    return parts_list


def _list_of_parts(model, only_objects = True):
    """Return a complete list of parts."""

    pl = list(model.parts_forest.keys())
    if not only_objects:
        return pl
    obj_list = []
    for part in pl:
        dir_place = part.rfind('/')
        dir_free_part = part[dir_place + 1:]
        if dir_free_part.startswith('obj_') and not dir_free_part.startswith('obj_Epc'):
            obj_list.append(part)
    return obj_list


def _number_of_parts(model):
    """Retuns the number of parts in the model, including external parts such as the link to an hdf5 file."""

    return len(model.parts_forest)


def _part_for_uuid(model, uuid):
    """Returns the part name which has the given uuid."""

    return model.uuid_part_dict.get(bu.uuid_as_int(uuid))


def _root_for_uuid(model, uuid):
    """Returns the xml root for the part which has the given uuid."""

    return _root_for_part(model, _part_for_uuid(model, uuid))


def _parts_count_by_type(model, type_of_interest = None):
    """Returns a sorted list of (type, count) for parts."""

    # note: resqml classes start with 'obj_' whilst witsml classes don't!
    if type_of_interest and type_of_interest.startswith('obj_'):
        type_of_interest = type_of_interest[4:]

    type_list = []
    for part_name in model.parts_forest:
        part_type = model.parts_forest[part_name][0]
        if part_type is None:
            continue
        if part_type.startswith('obj_'):
            part_type = part_type[4:]
        if type_of_interest is None or part_type == type_of_interest:
            type_list.append(part_type)
    type_list.sort()
    type_list.append('END')  # simplifies termination of scan below
    result_list = []
    count = 0
    current_type = ''
    for index in range(len(type_list)):
        if type_list[index] != current_type:
            if count:
                result_list.append((current_type, count))
            current_type = type_list[index]
            count = 0
        count += 1
    return result_list


def _parts_list_filtered_by_related_uuid(model, parts_list, uuid, uuid_is_source = None, related_mode = None):
    """From a list of parts, returns a list of those parts which have a relationship with the given uuid."""

    if not model.rels_present or parts_list is None or uuid is None:
        return None

    relations = model.uuid_rels_dict.get(bu.uuid_from_string(uuid).int)
    if relations is None:
        return []
    if related_mode is None:
        all_relations = relations[0] | relations[1] | relations[2]
    else:
        all_relations = relations[related_mode]

    filtered_list = []
    for part in parts_list:
        part_uuid_int = model.uuid_for_part(part).int
        if part_uuid_int in all_relations:
            filtered_list.append(part)

    return filtered_list


def _supporting_representation_for_part(model, part):
    """Returns the uuid of the supporting representation for the part, if found, otherwise None."""

    return bu.uuid_from_string(
        rqet.find_nested_tags_text(_root_for_part(model, part), ['SupportingRepresentation', 'UUID']))


def _parts_list_filtered_by_supporting_uuid(model, parts_list, uuid):
    """From a list of parts, returns a list of those parts which have the given uuid as supporting representation."""

    if parts_list is None or uuid is None:
        return []

    relations = model.uuid_rels_dict.get(bu.uuid_from_string(uuid).int)
    if relations is None:
        return []

    filtered_list = []
    for part in parts_list:
        part_uuid_int = model.uuid_for_part(part).int
        if part_uuid_int not in relations[1]:
            continue
        support_ref_uuid = _supporting_representation_for_part(model, part)
        if support_ref_uuid is None:
            continue
        if bu.matching_uuids(support_ref_uuid, uuid):
            filtered_list.append(part)
    return filtered_list


def _parts_list_related_to_uuid_of_type(model, uuid, type_of_interest = None):
    """Returns a list of parts of type of interest that relate to part with given uuid."""

    parts_list = _parts_list_of_type(model, type_of_interest = type_of_interest)
    return _parts_list_filtered_by_related_uuid(model, parts_list, uuid)


def _external_parts_list(model):
    """Returns a list of part names for external part references."""

    return _parts_list_of_type(model, 'obj_EpcExternalPartReference')


def _uuid_for_part(model, part_name, is_rels = None):
    """Returns the uuid for the named part."""

    if part_name is None:
        return None
    if is_rels is None:
        is_rels = part_name.endswith('.rels')
    if is_rels:
        return model.rels_forest[part_name][0]
    return model.parts_forest[part_name][1]


def _type_of_part(model, part_name, strip_obj = False):
    """Returns content type for the named part (does not apply to rels parts)."""

    part_info = model.parts_forest.get(part_name)
    if part_info is None:
        return None
    obj_type = part_info[0]
    if obj_type is None or not strip_obj or not obj_type.startswith('obj_'):
        return obj_type
    return obj_type[4:]


def _type_of_uuid(model, uuid, strip_obj = False):
    """Returns content type for the uuid."""

    part_name = model.uuid_part_dict.get(bu.uuid_as_int(uuid))
    return _type_of_part(model, part_name, strip_obj = strip_obj)


def _tree_for_part(model, part_name, is_rels = None):
    """Returns parsed xml tree for the named part."""

    if not part_name:
        return None
    if is_rels is None:
        is_rels = part_name.endswith('.rels')
    is_other = not is_rels and part_name.startswith('docProps')
    if is_rels:
        if part_name not in model.rels_forest:
            return None
        (_, tree) = model.rels_forest[part_name]
        if tree is None:
            if not model.epc_file:
                return None
            with zf.ZipFile(model.epc_file) as epc:
                load_success = model.load_part(epc, part_name, is_rels = True)
                if not load_success:
                    return None
        return model.rels_forest[part_name][1]
    elif is_other:
        if part_name not in model.other_forest:
            return None
        (_, tree) = model.other_forest[part_name]
        if tree is None:
            if not model.epc_file:
                return None
            with zf.ZipFile(model.epc_file) as epc:
                load_success = model.load_part(epc, part_name, is_rels = False)
                if not load_success:
                    return None
        return model.other_forest[part_name][1]
    else:
        if part_name not in model.parts_forest:
            return None
        (_, _, tree) = model.parts_forest[part_name]
        if tree is None:
            if not model.epc_file:
                return None
            with zf.ZipFile(model.epc_file) as epc:
                load_success = model.load_part(epc, part_name, is_rels = False)
                if not load_success:
                    return None
        return model.parts_forest[part_name][2]


def _root_for_part(model, part_name, is_rels = None):
    """Returns root of parsed xml tree for the named part."""

    if not part_name:
        return None
    tree = _tree_for_part(model, part_name, is_rels = is_rels)
    if tree is None:
        return None
    return tree.getroot()


def _citation_title_for_part(model, part):  # duplicate functionality to title_for_part()
    """Returns the citation title for the specified part."""

    title = rqet.citation_title_for_node(_root_for_part(model, part))
    if title is None:
        title = ''
    return title


def _root_for_time_series(model, uuid = None):
    """Return root for time series part."""

    time_series_list = _parts_list_of_type(model, 'obj_TimeSeries', uuid = uuid)
    if len(time_series_list) == 0:
        return None
    if len(time_series_list) == 1:
        return _root_for_part(model, time_series_list[0])
    log.warning('selecting time series with earliest creation date')
    oldest_root = oldest_creation = None
    for ts in time_series_list:
        node = _root_for_part(model, ts)
        created = rqet.creation_date_for_node(node)
        if oldest_creation is None or created < oldest_creation:
            oldest_creation = created
            oldest_root = node
    return oldest_root


def _resolve_time_series_root(model, time_series_root = None):
    """If time_series_root is None, finds the root for a time series in the model."""

    if time_series_root is not None:
        return time_series_root
    if model.time_series is None:
        model.time_series = _root_for_time_series(model)
    return model.time_series


def _title_for_root(model, root = None):
    """Returns the Title text from the Citation within the given root node."""

    title = rqet.find_tag(rqet.find_tag(root, 'Citation'), 'Title')
    if title is None:
        return None

    return title.text


def _title_for_part(model, part_name):  # duplicate functionality to citation_title_for_part()
    """Returns the Title text from the Citation for the given main part name (not for rels)."""

    return _title_for_root(model, _root_for_part(model, part_name))


def _iter_objs(model, cls):
    """Iterate over all available objects of given resqpy class within the model."""

    uuids = _uuids(model, obj_type = cls.resqml_type)
    for uuid in uuids:
        yield cls(model, uuid = uuid)


def _iter_grid_connection_sets(model):
    """Yields grid connection set objects, one for each gcs in this model."""

    import resqpy.fault as rqf  # imported here for speed, module is not always needed

    gcs_uuids = _uuids(model, obj_type = 'GridConnectionSetRepresentation')
    for gcs_uuid in gcs_uuids:
        yield rqf.GridConnectionSet(model, uuid = gcs_uuid)


def _iter_wellbore_interpretations(model):
    """Iterable of all WellboreInterpretations associated with the model."""

    import resqpy.organize as rqo  # imported here for speed, module is not always needed

    uuids = _uuids(model, obj_type = 'WellboreInterpretation')
    if uuids:
        for uuid in uuids:
            yield rqo.WellboreInterpretation(model, uuid = uuid)


def _iter_trajectories(model):
    """Iterable of all trajectories associated with the model."""

    import resqpy.well as rqw  # imported here for speed, module is not always needed

    uuids = _uuids(model, obj_type = "WellboreTrajectoryRepresentation")
    for uuid in uuids:
        yield rqw.Trajectory(model, uuid = uuid)


def _iter_md_datums(model):
    """Iterable of all MdDatum objects associated with the model."""

    import resqpy.well as rqw  # imported here for speed, module is not always needed

    uuids = _uuids(model, obj_type = 'MdDatum')
    if uuids:
        for uuid in uuids:
            datum = rqw.MdDatum(model, uuid = uuid)
            yield datum


def _iter_crs(model):
    """Iterable of all CRS objects associated with the model."""

    import resqpy.crs as rqc  # imported here for speed, module is not always needed

    uuids = _uuids(model, obj_type = 'LocalDepth3dCrs') + _uuids(model, obj_type = 'LocalTime3dCrs')
    if uuids:
        for uuid in uuids:
            yield rqc.Crs(model, uuid = uuid)


def _sort_parts_list_by_timestamp(model, parts_list):
    """Returns a copy of the parts list sorted by citation block creation date, with the newest first."""

    if parts_list is None:
        return None
    if len(parts_list) == 0:
        return []
    sort_list = []
    for index, part in enumerate(parts_list):
        timestamp = rqet.find_nested_tags_text(_root_for_part(model, part), ['Citation', 'Creation'])
        sort_list.append((timestamp, index))
    sort_list.sort()
    results = []
    for timestamp, index in reversed(sort_list):
        results.append(parts_list[index])
    return results


def _as_graph(model, uuids_subset = None):
    """Return representation of model as nodes and edges, suitable for plotting in a graph."""

    nodes = {}
    edges = set()

    if uuids_subset is None:
        uuids_subset = _uuids(model)

    uuids_subset = set(map(str, uuids_subset))

    for uuid in uuids_subset:
        part = _part_for_uuid(model, uuid)
        nodes[uuid] = dict(
            resqml_type = _type_of_part(model, part, strip_obj = True),
            title = _citation_title_for_part(model, part),
        )
        for rel in map(str, _uuids(model, related_uuid = uuid)):
            if rel in uuids_subset:
                edges.add(frozenset([uuid, rel]))

    return nodes, edges


def _filtered_by_epc_subdir(model, parts_list, epc_subdir):
    if epc_subdir.startswith('/'):
        epc_subdir = epc_subdir[1:]
    if epc_subdir:
        if not epc_subdir.endswith('/'):
            epc_subdir += '/'
        filtered_list = []
        for part in parts_list:
            if part.startswith[epc_subdir]:
                filtered_list.append(part)
        return filtered_list
    else:
        return parts_list


def _filtered_by_title(model, parts_list, title, title_mode, title_case_sensitive):
    assert title_mode in [
        'is', 'starts', 'ends', 'contains', 'is not', 'does not start', 'does not end', 'does not contain'
    ]
    if not title_case_sensitive:
        title = title.upper()
    filtered_list = []
    for part in parts_list:
        part_title = _citation_title_for_part(model, part)
        if not title_case_sensitive:
            part_title = part_title.upper()
        if title_mode == 'is':
            if part_title == title:
                filtered_list.append(part)
        elif title_mode == 'starts':
            if part_title.startswith(title):
                filtered_list.append(part)
        elif title_mode == 'ends':
            if part_title.endswith(title):
                filtered_list.append(part)
        elif title_mode == 'contains':
            if title in part_title:
                filtered_list.append(part)
        if title_mode == 'is not':
            if part_title != title:
                filtered_list.append(part)
        elif title_mode == 'does not start':
            if not part_title.startswith(title):
                filtered_list.append(part)
        elif title_mode == 'does not end':
            if not part_title.endswith(title):
                filtered_list.append(part)
        elif title_mode == 'does not contain':
            if title not in part_title:
                filtered_list.append(part)
    return filtered_list


def _filtered_by_metadata(model, parts_list, metadata):
    filtered_list = []
    for part in parts_list:
        root = _root_for_part(model, part)
        match = True
        for key, value in metadata.items():
            node = rqet.find_tag(root, str(key))
            if node is None or node.text != str(value):
                match = False
                break
        if match:
            filtered_list.append(part)
    return filtered_list


def _filtered_by_extra(model, parts_list, extra):
    filtered_list = []
    for part in parts_list:
        part_extra = rqet.load_metadata_from_xml(_root_for_part(model, part))
        if not part_extra:
            continue
        match = True
        for key, value in extra.items():
            if key not in part_extra or part_extra[key] != value:
                match = False
                break
        if match:
            filtered_list.append(part)
    return filtered_list


def _sorted_parts_list(model, parts_list, sort_by):
    if sort_by == 'type':
        sorted_list = sorted(parts_list)
    elif sort_by in ['newest', 'oldest']:
        sorted_list = _sort_parts_list_by_timestamp(model, parts_list)
        if sort_by == 'oldest':
            sorted_list.reverse()
    elif sort_by in ['uuid', 'title']:
        sort_list = []
        for index, part in enumerate(parts_list):
            if sort_by == 'uuid':
                key = str(_uuid_for_part(model, part))
            else:
                key = _citation_title_for_part(model, part)
            sort_list.append((key, index))
        sort_list.sort()
        sorted_list = []
        for _, index in sort_list:
            sorted_list.append(parts_list[index])
    return sorted_list


def _uuids_as_int_related_to_uuid(model, uuid):
    if uuid is None:
        return None
    uuid_int = bu.uuid_as_int(uuid)
    relatives = model.uuid_rels_dict.get(uuid_int)
    if relatives is None:
        return None
    return relatives[0] | relatives[1] | relatives[2]


def _uuids_as_int_referenced_by_uuid(model, uuid):
    if uuid is None:
        return None
    uuid_int = bu.uuid_as_int(uuid)
    relatives = model.uuid_rels_dict.get(uuid_int)
    if relatives is None:
        return None
    return relatives[0]


def _uuids_as_int_referencing_uuid(model, uuid):
    if uuid is None:
        return None
    uuid_int = bu.uuid_as_int(uuid)
    relatives = model.uuid_rels_dict.get(uuid_int)
    if relatives is None:
        return None
    return relatives[1]


def _uuids_as_int_softly_related_to_uuid(model, uuid):
    if uuid is None:
        return None
    uuid_int = bu.uuid_as_int(uuid)
    relatives = model.uuid_rels_dict.get(uuid_int)
    if relatives is None:
        return None
    return relatives[2]


def _check_catalogue_dictionaries(model, referred_parts_must_be_present, check_xml):
    for uuid_int, part in model.uuid_part_dict.items():
        assert uuid_int is not None and part
        assert bu.is_uuid(uuid_int)
        assert part in model.parts_forest
    for part in model.parts_forest:
        if part.startswith('obj_') and 'EpcExternal' not in part:
            assert part in model.uuid_part_dict.values()
    for uuid_int, relatives in model.uuid_rels_dict.items():
        assert uuid_int in model.uuid_part_dict
        if referred_parts_must_be_present:
            for ref_uuid_int in relatives[0] | relatives[1] | relatives[2]:
                assert ref_uuid_int in model.uuid_part_dict
                assert ref_uuid_int != uuid_int
            for ref_uuid_int in relatives[0]:
                assert uuid_int in model.uuid_rels_dict[ref_uuid_int][1]
                assert ref_uuid_int not in relatives[1] | relatives[2]
            for ref_uuid_int in relatives[1]:
                assert uuid_int in model.uuid_rels_dict[ref_uuid_int][0]
                assert ref_uuid_int not in relatives[0] | relatives[2]
            for ref_uuid_int in relatives[2]:
                assert ref_uuid_int not in relatives[0] | relatives[1]
    if check_xml:
        for uuid_int, part in model.uuid_part_dict.items():
            root = model.parts_forest[part][2].getroot()
            assert root is not None
            ref_nodes = rqet.list_obj_references(root, skip_hdf5 = True)
            relatives = model.uuid_rels_dict.get(uuid_int)
            if relatives is None:
                assert len(ref_nodes) == 0
                continue
            ref_uuid_ints_from_dict = relatives[0]
            assert len(ref_nodes) == len(ref_uuid_ints_from_dict)
            ref_node_uuid_ints = []
            for ref_node in ref_nodes:
                uuid_str = rqet.find_tag_text(ref_node, 'UUID')
                uuid = bu.uuid_from_string(uuid_str)
                assert uuid is not None
                assert uuid.int in ref_uuid_ints_from_dict
                ref_node_uuid_ints.append(uuid.int)
            for ref_uuid_int in ref_uuid_ints_from_dict:
                assert ref_uuid_int in ref_node_uuid_ints
