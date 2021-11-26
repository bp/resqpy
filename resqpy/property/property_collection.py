"""propertycollection.py: class handling collections of RESQML properties for grids, wellbore frames, grid connection sets etc."""

version = '23rd November 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('property.py version ' + version)

import numpy as np
import numpy.ma as ma

import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.time_series as rts
from resqpy.olio.xml_namespaces import curly_namespace as ns

from .property_kind import PropertyKind
from .string_lookup import StringLookup
from .property_common import same_property_kind, supported_property_kind_list, property_kind_and_facet_from_keyword, dtype_flavour, _cache_name, _cache_name_for_uuid, selective_version_of_collection, guess_uom


class PropertyCollection():
    """Class for RESQML Property collection for any supporting representation (or mix of supporting representations).

    notes:
       this is a base class inherited by GridPropertyCollection and WellLogCollection (and others to follow), application
       code usually works with the derived classes;
       RESQML caters for three simple types of numerical property: Continuous (ie. real data, aka floating point);
       Discrete (ie. integer data, or boolean); Categorical (integer data, usually non-negative, with an associated
       look-up table to convert to a string); Points properties are for storing xyz values; resqpy does not currently
       support Comment properties
    """

    def __init__(self, support = None, property_set_root = None, realization = None):
        """Initialise an empty Property Collection, optionally populate properties from a supporting representation.

        arguments:
           support (optional): a grid.Grid object or a well.WellboreFrame object which belongs to a resqpy.Model which includes
              associated properties; if this argument is given, and property_set_root is None, the properties in the support's
              parent model which are for this representation (ie. have this grid or wellbore frame as the supporting representation)
              are added to this collection as part of the initialisation
           property_set_root (optional): if present, the collection is populated with the properties defined in the xml tree
              of the property set
           realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
              if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
              realisation number), or is for properties that do not have multiple realizations

        note:
           at present, if the collection is being initialised from a property set, the support argument must also be specified;
           also for now, if not initialising from a property set, all properties related to the support are included, whether
           the relationship is supporting representation or some other relationship;
           the full handling of RESQML property sets and property series is still under development

        :meta common:
        """

        assert property_set_root is None or support is not None,  \
           'support (grid, wellbore frame, blocked well, mesh, or grid connection set) must be specified when populating property collection from property set'

        self.dict = {}  # main dictionary of model property parts which are members of the collection
        # above is mapping from part_name to:
        # (realization, support, uuid, xml_node, continuous, count, indexable, prop_kind, facet_type, facet, citation_title,
        #   time_series_uuid, time_index, min, max, uom, string_lookup_uuid, property_kind_uuid, extra_metadata, null_value,
        #     const_value, points)
        #  0            1        2     3         4           5      6          7          8           9      10
        #   11                12          13   14   15   16                  17                  18              19
        #     20           21
        # note: grid is included to allow for super-collections covering more than one grid
        # todo: replace items 8 & 9 with a facet dictionary (to allow for multiple facets)
        self.model = None
        self.support = None
        self.support_root = None
        self.support_uuid = None
        self.property_set_root = None
        self.time_set_kind_attr = None
        self.has_single_property_kind_flag = None
        self.has_single_uom_flag = None
        self.has_single_indexable_element_flag = None
        self.has_multiple_realizations_flag = None
        self.parent_set_root = None
        self.realization = realization  # model realization number within an ensemble
        self.null_value = None
        self.imported_list = [
        ]  # list of (uuid, file_name, keyword, cached_name, discrete, uom, time_index, null_value,
        #                                   min_value, max_value, property_kind, facet_type, facet, realization,
        #                                   indexable_element, count, local_property_kind_uuid, const_value, points)
        if support is not None:
            self.model = support.model
            self.set_support(support = support)
            assert self.model is not None
            # assert self.support_root is not None
            assert self.support_uuid is not None
            if property_set_root is None:
                # todo: make more rigorous by looking up supporting representation node uuids
                props_list = self.model.parts_list_of_type(type_of_interest = 'obj_DiscreteProperty')
                discrete_props_list = self.model.parts_list_filtered_by_supporting_uuid(props_list, self.support_uuid)
                self.add_parts_list_to_dict(discrete_props_list)
                props_list = self.model.parts_list_of_type(type_of_interest = 'obj_CategoricalProperty')
                categorical_props_list = self.model.parts_list_filtered_by_supporting_uuid(
                    props_list, self.support_uuid)
                self.add_parts_list_to_dict(categorical_props_list)
                props_list = self.model.parts_list_of_type(type_of_interest = 'obj_ContinuousProperty')
                continuous_props_list = self.model.parts_list_filtered_by_supporting_uuid(props_list, self.support_uuid)
                self.add_parts_list_to_dict(continuous_props_list)
                props_list = self.model.parts_list_of_type(type_of_interest = 'obj_PointsProperty')
                points_props_list = self.model.parts_list_filtered_by_supporting_uuid(props_list, self.support_uuid)
                self.add_parts_list_to_dict(points_props_list)
            else:
                self.populate_from_property_set(property_set_root)

    def set_support(self, support_uuid = None, support = None, model = None, modify_parts = True):
        """Sets the supporting object associated with this collection if not done so at initialisation.

        Does not load properties.

        Arguments:
           support_uuid: the uuid of the supporting representation which the properties in this collection are for
           support: a grid.Grid, unstructured.UnstructuredGrid (or derived class), well.WellboreFrame, well.BlockedWell,
              surface.Mesh, or fault.GridConnectionSet object which the properties in this collection are for
           model (model.Model object, optional): if present, the model associated with this collection is set to this;
              otherwise the model is assigned from the supporting object
           modify_parts (boolean, default True): if True, any parts already in this collection have their individual
              support_uuid set
        """

        # when at global level was causing circular reference loading issues as grid imports this module
        import resqpy.fault as rqf
        import resqpy.grid as grr
        import resqpy.surface as rqs
        import resqpy.unstructured as rug
        import resqpy.well as rqw

        # todo: check uuid's of individual parts' supports match that of support being set for whole collection

        if model is None and support is not None:
            model = support.model
        if model is None:
            model = self.model
        else:
            self.model = model

        if support_uuid is None and support is not None:
            support_uuid = support.uuid

        if support_uuid is None:
            if self.support_uuid is not None:
                log.warning('clearing supporting representation for property collection')
            # self.model = None
            self.support = None
            self.support_root = None
            self.support_uuid = None

        else:
            assert model is not None, 'model not established when setting support for property collection'
            if self.support_uuid is not None and not bu.matching_uuids(support_uuid, self.support_uuid):
                log.warning('changing supporting representation for property collection')
            self.support_uuid = support_uuid
            self.support = support
            if self.support is None:
                support_part = model.part_for_uuid(support_uuid)
                assert support_part is not None, 'supporting representation part missing in model'
                self.support_root = model.root_for_part(support_part)
                support_type = model.type_of_part(support_part)
                assert support_type is not None
                if support_type == 'obj_IjkGridRepresentation':
                    self.support = grr.any_grid(model, uuid = self.support_uuid, find_properties = False)
                elif support_type == 'obj_WellboreFrameRepresentation':
                    self.support = rqw.WellboreFrame(model, uuid = self.support_uuid)
                elif support_type == 'obj_BlockedWellboreRepresentation':
                    self.support = rqw.BlockedWell(model, uuid = self.support_uuid)
                elif support_type == 'obj_Grid2dRepresentation':
                    self.support = rqs.Mesh(model, uuid = self.support_uuid)
                elif support_type == 'obj_GridConnectionSetRepresentation':
                    self.support = rqf.GridConnectionSet(model, uuid = self.support_uuid)
                elif support_type == 'obj_UnstructuredGridRepresentation':
                    self.support = rug.UnstructuredGrid(model,
                                                        uuid = self.support_uuid,
                                                        geometry_required = False,
                                                        find_properties = False)
                else:
                    raise TypeError('unsupported property supporting representation class: ' + str(support_type))
            else:
                if type(self.support) in [
                        grr.Grid, grr.RegularGrid, rqw.WellboreFrame, rqw.BlockedWell, rqs.Mesh, rqf.GridConnectionSet,
                        rug.UnstructuredGrid, rug.HexaGrid, rug.TetraGrid, rug.PrismGrid, rug.VerticalPrismGrid,
                        rug.PyramidGrid
                ]:
                    self.support_root = self.support.root
                else:
                    raise TypeError('unsupported property supporting representation class: ' + str(type(self.support)))
            if modify_parts:
                for (part, info) in self.dict.items():
                    if info[1] is not None:
                        modified = list(info)
                        modified[1] = support_uuid
                        self.dict[part] = tuple(modified)

    def supporting_shape(self, indexable_element = None, direction = None):
        """Return the shape of the supporting representation with respect to the given indexable element

        arguments:
           indexable_element (string, optional): if None, a hard-coded default depending on the supporting representation class
              will be used
           direction (string, optional): must be passed if required for the combination of support class and indexable element;
              currently only used for Grid faces.

        returns:
           list of int, being required shape of numpy array, or None if not coded for

        note:
           individual property arrays will only match this shape if they have the same indexable element and a count of one
        """

        # when at global level was causing circular reference loading issues as grid imports this module
        import resqpy.fault as rqf
        import resqpy.grid as grr
        import resqpy.surface as rqs
        import resqpy.unstructured as rug
        import resqpy.well as rqw

        support = self.support

        if isinstance(support, grr.Grid):
            shape_list = _supporting_shape_grid(support, indexable_element, direction)

        elif isinstance(support, rqw.WellboreFrame):
            shape_list = _supporting_shape_wellboreframe(support, indexable_element)

        elif isinstance(support, rqw.BlockedWell):
            shape_list = _supporting_shape_blockedwell(support, indexable_element)

        elif isinstance(support, rqs.Mesh):
            shape_list = _supporting_shape_mesh(support, indexable_element)

        elif isinstance(support, rqf.GridConnectionSet):
            shape_list = _supporting_shape_gridconnectionset(support, indexable_element)

        elif type(support) in [rug.UnstructuredGrid, rug.HexaGrid, rug.TetraGrid,
                               rug.PrismGrid, rug.VerticalPrismGrid, rug.PyramidGrid]:
            shape_list, support = _supporting_shape_other(support, indexable_element)

        else:
            raise Exception(f'unsupported support class {type(support)} for property')

        return shape_list

    def populate_from_property_set(self, property_set_root):
        """Populates this (newly created) collection based on xml members of property set."""

        assert property_set_root is not None, 'missing property set xml root'
        assert self.model is not None and self.support is not None, 'set support for collection before populating from property set'

        self.property_set_root = property_set_root
        self.time_set_kind_attr = rqet.find_tag_text(property_set_root, 'TimeSetKind')
        self.has_single_property_kind_flag = rqet.find_tag_bool(property_set_root, 'HasSinglePropertyKind')
        self.has_multiple_realizations_flag = rqet.find_tag_bool(property_set_root, 'HasMultipleRealizations')
        parent_set_ref_root = rqet.find_tag(property_set_root, 'ParentSet')  # at most one parent set handled here
        if parent_set_ref_root is not None:
            self.parent_set_root = self.model.referenced_node(parent_set_ref_root)
        # loop over properties in property set xml, adding parts to main dictionary
        for child in property_set_root:
            if rqet.stripped_of_prefix(child.tag) != 'Properties':
                continue
            property_root = self.model.referenced_node(child)
            if property_root is None:
                log.warning('property set member missing from resqml dataset')
                continue
            self.add_part_to_dict(rqet.part_name_for_part_root(property_root))

    def set_realization(self, realization):
        """Sets the model realization number (within an ensemble) for this collection.

        argument:
           realization (non-negative integer): the realization number of the whole collection within an ensemble of
                   collections

        note:
           the resqml Property classes allow for a realization index to be associated with an individual property
           array; this module supports this by associating a realization number (equivalent to realization index) for
           each part (ie. for each property array); however, the collection can be given a realization number which is
           then applied to each member of the collection as it is added, if no part-specific realization number is
           provided
        """

        # the following assertion might need to be downgraded to a warning, to allow for reassignment of realization numbers
        assert self.realization is None
        self.realization = realization

    def add_part_to_dict(self, part, continuous = None, realization = None, trust_uom = True):
        """Add the named part to the dictionary for this collection.

        arguments:
           part (string): the name of a part (which exists in the support's parent model) to be added to this collection
           continuous (boolean, optional): whether the property is of a continuous (real) kind; if not None,
                    is checked against the property's type and an assertion error is raised if there is a mismatch;
                    should be None or True for Points properties
           realization (integer, optional): if present, must match this collection's realization number if that is
                    not None; if this argument is None then the part is assigned the realization number associated
                    with this collection as a whole; if the xml for the part includes a realization index then that
                    overrides these other sources to become the realization number
           trust_uom (boolean, default True): if True, the uom stored in the part's xml is used as the part's uom
                    in this collection; if False and the uom in the xml is an empty string or 'Euc', then the
                    part's uom in this collection is set to a guess based on the property kind and min & max values;
                    note that this guessed value is not used to overwrite the value in the xml
        """

        if part is None:
            return
        assert part not in self.dict
        uuid = self.model.uuid_for_part(part, is_rels = False)
        assert uuid is not None
        xml_node = self.model.root_for_part(part, is_rels = False)
        assert xml_node is not None

        realization = self._add_part_to_dict_get_realization(realization, xml_node)
        type, continuous, points, string_lookup_uuid, sl_ref_node = self._add_part_to_dict_get_type_details(
            part, continuous, xml_node)
        extra_metadata = rqet.load_metadata_from_xml(xml_node)
        citation_title = rqet.find_tag(rqet.find_tag(xml_node, 'Citation'), 'Title').text
        count, indexable = _add_part_to_dict_get_count_and_indexable(xml_node)
        property_kind, property_kind_uuid, lpk_node = _add_part_to_dict_get_property_kind(xml_node, citation_title)
        facet_type, facet = _add_part_to_dict_get_facet(xml_node)
        time_series_uuid, time_index = _add_part_to_dict_get_timeseries(xml_node)
        minimum, maximum = _add_part_to_dict_get_minmax(xml_node)
        support_uuid = self._add_part_to_dict_get_support_uuid(part)
        uom = self._add_part_to_dict_get_uom(part, continuous, xml_node, trust_uom, property_kind, minimum, maximum, facet, facet_type)
        null_value, const_value = _add_part_to_dict_get_null_constvalue_points(xml_node, continuous, points)

        self.dict[part] = (realization, support_uuid, uuid, xml_node, continuous, count, indexable, property_kind,
                           facet_type, facet, citation_title, time_series_uuid, time_index, minimum, maximum, uom,
                           string_lookup_uuid, property_kind_uuid, extra_metadata, null_value, const_value, points)

    def add_parts_list_to_dict(self, parts_list):
        """Add all the parts named in the parts list to the dictionary for this collection.

        argument:
           parts_list: a list of strings, each being the name of a part in the support's parent model

        note:
           the add_part_to_dict() function is called for each part in the list
        """

        for part in parts_list:
            self.add_part_to_dict(part)

    def remove_part_from_dict(self, part):
        """Remove the named part from the dictionary for this collection.

        argument:
           part (string): the name of a part which might be in this collection, to be removed

        note:
           if the part is not in the collection, no action is taken and no exception is raised
        """

        if part is None:
            return
        if part not in self.dict:
            return
        del self.dict[part]

    def remove_parts_list_from_dict(self, parts_list):
        """Remove all the parts named in the parts list from the dictionary for this collection.

        argument:
           parts_list: a list of strings, each being the name of a part which might be in the collection

        note:
           the remove_part_from_dict() function is called for each part in the list
        """

        for part in parts_list:
            self.remove_part_from_dict(part)

    def inherit_imported_list_from_other_collection(self, other, copy_cached_arrays = True, exclude_inactive = False):
        """Extends this collection's imported list with items from other's imported list.

        arguments:
           other: another PropertyCollection object with some imported arrays
           copy_cached_arrays (boolean, default True): if True, arrays cached with the other
              collection are copied and cached with this collection
           exclude_inactive (boolean, default False): if True, any item in the other imported list
              which has INACTIVE or ACTIVE as the keyword is excluded from the inheritance

        note:
           the imported list is a list of cached imported arrays with basic info for each array;
           it is used as a staging post before fully incorporating the imported arrays as parts
           of the support's parent model and writing the arrays to the hdf5 file
        """

        # note: does not inherit parts
        if exclude_inactive:
            other_list = []
            for imp in other.imported_list:
                if imp[2].upper() not in ['INACTIVE', 'ACTIVE']:
                    other_list.append(imp)
        else:
            other_list = other.imported_list
        self.imported_list += other_list
        if copy_cached_arrays:
            for imp in other_list:
                if imp[17] is not None:
                    continue  # constant array
                cached_name = imp[3]
                self.__dict__[cached_name] = other.__dict__[cached_name].copy()

    def inherit_parts_from_other_collection(self, other, ignore_clashes = False):
        """Adds all the parts in the other PropertyCollection to this one.

        Arguments:
           other: another PropertyCollection object related to the same support as this collection
           ignore_clashes (boolean, default False): if False, any part in other which is already in
              this collection will result in an assertion error; if True, such duplicates are
              simply skipped without modifying the existing part in this collection
        """

        assert self.support_uuid is None or other.support_uuid is None or bu.matching_uuids(
            self.support_uuid, other.support_uuid)
        if self.support_uuid is None and self.number_of_parts() == 0 and other.support_uuid is not None:
            self.set_support(support_uuid = other.support_uuid, support = other.support)
        if self.realization is not None and other.realization is not None:
            assert self.realization == other.realization
        for (part, info) in other.dict.items():
            if part in self.dict.keys():
                if ignore_clashes:
                    continue
                assert False, 'attempt to inherit a part which already exists in property collection: ' + part
            self.dict[part] = info

    def add_to_imported_list_sampling_other_collection(self, other, flattened_indices):
        """Makes cut down copies of parts from other collection, using indices, and adds to imported list.

        arguments:
          other (PropertyCollection): the source collection whose arrays will be sampled
          flattened_indices (1D numpy int array): the indices (in flattened space) of the elements to be copied

        notes:
          the values in flattened_indices refer to the source (other) array elements, after flattening;
          the size of flatted_indices must match the size of the target (self) supporting shape; where different
          indexable elements are at play, with different implicit sizes, make selective copies of other and
          call this method once for each group of differently sized properties; for very large collections
          it might also be necessary to divide the work into smaller groups to reduce memory usage;
          this method does not write to hdf5 nor create xml – use the usual methods for further processing
          of the imported list
        """

        source = 'sampled'
        if other.support is not None:
            source += ' from property for ' + str(other.support.title)
        for (part, info) in other.dict.items():
            target_shape = self.supporting_shape(indexable_element = other.indexable_for_part(part),
                                                 direction = other._part_direction(part))
            assert np.prod(target_shape) == flattened_indices.size
            a = other.cached_part_array_ref(part).flatten()[flattened_indices].reshape(target_shape)
            self.add_cached_array_to_imported_list(a,
                                                   source,
                                                   info[10],
                                                   discrete = not info[4],
                                                   uom = info[15],
                                                   time_index = info[12],
                                                   null_value = info[19],
                                                   property_kind = info[7],
                                                   local_property_kind_uuid = info[17],
                                                   facet_type = info[8],
                                                   facet = info[9],
                                                   realization = info[0],
                                                   indexable_element = info[6],
                                                   count = info[5],
                                                   const_value = info[20],
                                                   points = info[21])

    def inherit_parts_selectively_from_other_collection(
            self,
            other,
            realization = None,
            support_uuid = None,
            grid = None,  # for backward compatibility
            uuid = None,
            continuous = None,
            count = None,
            points = None,
            indexable = None,
            property_kind = None,
            facet_type = None,
            facet = None,
            citation_title = None,
            citation_title_match_starts_with = False,
            time_series_uuid = None,
            time_index = None,
            uom = None,
            string_lookup_uuid = None,
            categorical = None,
            ignore_clashes = False):
        """Adds those parts from the other PropertyCollection which match all arguments that are not None.

        arguments:
           other: another PropertyCollection object related to the same support as this collection
           citation_title_match_starts_with (boolean, default False): if True, any citation title that
              starts with the given citation_title argument will be deemed to have passed that filter
           ignore_clashes (boolean, default False): if False, any part in other which passes the filters
              yet is already in this collection will result in an assertion error; if True, such duplicates
              are simply skipped without modifying the existing part in this collection

        Other optional arguments (realization, grid, uuid, continuous, count, points, indexable, property_kind, facet_type,
        facet, citation_title, time_series_uuid, time_index, uom, string_lookup_uuid, categorical):

        For each of these arguments: if None, then all members of collection pass this filter;
        if not None then only those members with the given value pass this filter;
        finally, the filters for all the attributes must be passed for a given member (part)
        to be inherited.

        note:
           the grid argument is maintained for backward compatibility; it is treated synonymously with support
           which takes precendence; the categorical boolean argument can be used to filter only Categorical
           (or non-Categorical) properties
        """

        #      log.debug('inheriting parts selectively')
        self._set_support_and_model_from_collection(other, support_uuid, grid)

        if self.realization is not None and other.realization is not None:
            assert self.realization == other.realization
        if time_index is not None:
            assert time_index >= 0

        for (part, info) in other.dict.items():
            self._add_selected_part_from_other_dict(part, other, realization, support_uuid, uuid, continuous,
                                                    categorical, count, points, indexable, property_kind, facet_type,
                                                    facet, citation_title, citation_title_match_starts_with,
                                                    time_series_uuid, time_index, string_lookup_uuid, ignore_clashes)

    def inherit_similar_parts_for_time_series_from_other_collection(self,
                                                                    other,
                                                                    example_part,
                                                                    citation_title_match_starts_with = False,
                                                                    ignore_clashes = False):
        """Adds the example part from other collection and any other parts for the same property at different times.

        arguments:
           other: another PropertyCollection object related to the same grid as this collection, from which to inherit
           example_part (string): the part name of an example member of other (which has an associated time_series)
           citation_title_match_starts_with (booleam, default False): if True, any citation title that starts with
              the example's citation_title after removing trailing digits, will be deemed to have passed that filter
           ignore_clashes (boolean, default False): if False, any part in other which passes the filters
              yet is already in this collection will result in an assertion error; if True, such duplicates
              are simply skipped without modifying the existing part in this collection

        note:
           at present, the citation title must match (as well as the other identifying elements) for a part to be
           inherited
        """

        assert other is not None
        assert other.part_in_collection(example_part)
        time_series_uuid = other.time_series_uuid_for_part(example_part)
        assert time_series_uuid is not None
        title = other.citation_title_for_part(example_part)
        if citation_title_match_starts_with:
            while title and title[-1].isdigit():
                title = title[:-1]
        self.inherit_parts_selectively_from_other_collection(
            other,
            realization = other.realization_for_part(example_part),
            support_uuid = other.support_uuid_for_part(example_part),
            continuous = other.continuous_for_part(example_part),
            points = other.points_for_part(example_part),
            indexable = other.indexable_for_part(example_part),
            property_kind = other.property_kind_for_part(example_part),
            facet_type = other.facet_type_for_part(example_part),
            facet = other.facet_for_part(example_part),
            citation_title = title,
            citation_title_match_starts_with = citation_title_match_starts_with,
            time_series_uuid = time_series_uuid,
            ignore_clashes = ignore_clashes)

    def inherit_similar_parts_for_facets_from_other_collection(self,
                                                               other,
                                                               example_part,
                                                               citation_title_match_starts_with = False,
                                                               ignore_clashes = False):
        """Adds the example part from other collection and any other parts for same property with different facets.

        arguments:
           other: another PropertyCollection object related to the same grid as this collection, from which to inherit
           example_part (string): the part name of an example member of other
           citation_title_match_starts_with (booleam, default False): if True, any citation title that starts with
              the example's citation_title after removing trailing digits, will be deemed to have passed that filter
           ignore_clashes (boolean, default False): if False, any part in other which passes the filters
              yet is already in this collection will result in an assertion error; if True, such duplicates
              are simply skipped without modifying the existing part in this collection

        note:
           at present, the citation title must match (as well as the other identifying elements) for a part to be
           inherited
        """

        # the following RESQML limitation could be reported as a warning here, instead of an assertion
        assert not other.points_for_part(example_part), 'facets not allowed for RESQML Points properties'
        assert other is not None
        assert other.part_in_collection(example_part)
        title = other.citation_title_for_part(example_part)
        if citation_title_match_starts_with:
            while title and title[-1].isdigit():
                title = title[:-1]
        self.inherit_parts_selectively_from_other_collection(
            other,
            realization = other.realization_for_part(example_part),
            support_uuid = other.support_uuid_for_part(example_part),
            continuous = other.continuous_for_part(example_part),
            points = False,
            indexable = other.indexable_for_part(example_part),
            property_kind = other.property_kind_for_part(example_part),
            citation_title = title,
            time_series_uuid = other.time_series_uuid_for_part(example_part),
            time_index = other.time_index_for_part(example_part),
            ignore_clashes = ignore_clashes)

    def inherit_similar_parts_for_realizations_from_other_collection(self,
                                                                     other,
                                                                     example_part,
                                                                     citation_title_match_starts_with = False,
                                                                     ignore_clashes = False):
        """Add the example part from other collection and any other parts for same property with different realizations.

        arguments:
           other: another PropertyCollection object related to the same support as this collection, from which to inherit
           example_part (string): the part name of an example member of other
           citation_title_match_starts_with (booleam, default False): if True, any citation title that starts with
              the example's citation_title after removing trailing digits, will be deemed to have passed that filter
           ignore_clashes (boolean, default False): if False, any part in other which passes the filters
              yet is already in this collection will result in an assertion error; if True, such duplicates
              are simply skipped without modifying the existing part in this collection

        note:
           at present, the citation title must match (as well as the other identifying elements) for a part to be
           inherited
        """

        assert other is not None
        assert other.part_in_collection(example_part)
        title = other.citation_title_for_part(example_part)
        if citation_title_match_starts_with:
            while title and title[-1].isdigit():
                title = title[:-1]
        self.inherit_parts_selectively_from_other_collection(
            other,
            realization = None,
            support_uuid = other.support_uuid_for_part(example_part),
            continuous = other.continuous_for_part(example_part),
            points = other.points_for_part(example_part),
            indexable = other.indexable_for_part(example_part),
            property_kind = other.property_kind_for_part(example_part),
            facet_type = other.facet_type_for_part(example_part),
            facet = other.facet_for_part(example_part),
            citation_title = title,
            citation_title_match_starts_with = citation_title_match_starts_with,
            time_series_uuid = other.time_series_uuid_for_part(example_part),
            time_index = other.time_index_for_part(example_part),
            ignore_clashes = ignore_clashes)

    def number_of_imports(self):
        """Returns the number of property arrays in the imported list for this collection.

        returns:
           count of number of cached property arrays in the imported list for this collection (non-negative integer)

        note:
           the importation list is cleared after creation of xml trees for the imported properties, so this
           function will return zero at that point, until a new list of imports is built up
        """

        return len(self.imported_list)

    def parts(self):
        """Return list of parts in this collection.

        returns:
           list of part names (strings) being the members of this collection; there is one part per property array

        :meta common:
        """

        return list(self.dict.keys())

    def uuids(self):
        """Return list of uuids in this collection.

        returns:
           list of uuids being the members of this collection; there is one uuid per property array

        :meta common:
        """
        return [self.model.uuid_for_part(p) for p in self.dict.keys()]

    def selective_parts_list(
            self,
            realization = None,
            support = None,  # maintained for backward compatibility
            support_uuid = None,
            grid = None,  # maintained for backward compatibility
            continuous = None,
            points = None,
            count = None,
            indexable = None,
            property_kind = None,
            facet_type = None,
            facet = None,
            citation_title = None,
            time_series_uuid = None,
            time_index = None,
            uom = None,
            string_lookup_uuid = None,
            categorical = None):
        """Returns a list of parts filtered by those arguments which are not None.

        All arguments are optional.

        For each of these arguments: if None, then all members of collection pass this filter;
        if not None then only those members with the given value pass this filter;
        finally, the filters for all the attributes must be passed for a given member (part)
        to be included in the returned list of parts

        returns:
           list of part names (strings) of those parts which match any selection arguments which are not None

        note:
           the support and grid keyword arguments are maintained for backward compatibility;
           support_uuid takes precedence over support and both take precedence over grid

        :meta common:
        """

        if support is None:
            support = grid
        if support_uuid is None and support is not None:
            support_uuid = support.uuid

        temp_collection = selective_version_of_collection(self,
                                                          realization = realization,
                                                          support_uuid = support_uuid,
                                                          continuous = continuous,
                                                          points = points,
                                                          count = count,
                                                          indexable = indexable,
                                                          property_kind = property_kind,
                                                          facet_type = facet_type,
                                                          facet = facet,
                                                          citation_title = citation_title,
                                                          time_series_uuid = time_series_uuid,
                                                          time_index = time_index,
                                                          uom = uom,
                                                          categorical = categorical,
                                                          string_lookup_uuid = string_lookup_uuid)
        parts_list = temp_collection.parts()
        return parts_list

    def singleton(
            self,
            realization = None,
            support = None,  # for backward compatibility
            support_uuid = None,
            grid = None,  # for backward compatibility
            uuid = None,
            continuous = None,
            points = None,
            count = None,
            indexable = None,
            property_kind = None,
            facet_type = None,
            facet = None,
            citation_title = None,
            time_series_uuid = None,
            time_index = None,
            uom = None,
            string_lookup_uuid = None,
            categorical = None):
        """Returns a single part selected by those arguments which are not None.

        For each argument: if None, then all members of collection pass this filter;
        if not None then only those members with the given value pass this filter;
        finally, the filters for all the attributes must be passed for a given member (part)
        to be selected

        returns:
           part name (string) of the part which matches all selection arguments which are not None;
           returns None if no parts match; raises an assertion error if more than one part matches

        :meta common:
        """

        if support is None:
            support = grid
        if support_uuid is None and support is not None:
            support_uuid = support.uuid

        temp_collection = selective_version_of_collection(self,
                                                          realization = realization,
                                                          support_uuid = support_uuid,
                                                          uuid = uuid,
                                                          continuous = continuous,
                                                          points = points,
                                                          count = count,
                                                          indexable = indexable,
                                                          property_kind = property_kind,
                                                          facet_type = facet_type,
                                                          facet = facet,
                                                          citation_title = citation_title,
                                                          time_series_uuid = time_series_uuid,
                                                          time_index = time_index,
                                                          uom = uom,
                                                          string_lookup_uuid = string_lookup_uuid,
                                                          categorical = categorical)
        parts_list = temp_collection.parts()
        if len(parts_list) == 0:
            return None
        assert len(parts_list) == 1, 'More than one property part matches selection criteria'
        return parts_list[0]

    def single_array_ref(
            self,
            realization = None,
            support = None,  # for backward compatibility
            support_uuid = None,
            grid = None,  # for backward compatibility
            uuid = None,
            continuous = None,
            points = None,
            count = None,
            indexable = None,
            property_kind = None,
            facet_type = None,
            facet = None,
            citation_title = None,
            time_series_uuid = None,
            time_index = None,
            uom = None,
            string_lookup_uuid = None,
            categorical = None,
            dtype = None,
            masked = False,
            exclude_null = False):
        """Returns the array of data for a single part selected by those arguments which are not None.

        arguments:
           dtype (optional, default None): the element data type of the array to be accessed, eg 'float' or 'int';
              if None (recommended), the dtype of the returned numpy array matches that in the hdf5 dataset
           masked (boolean, optional, default False): if True, a masked array is returned instead of a simple
              numpy array; the mask is set to the inactive array attribute of the grid object
           exclude_null (boolean, default False): it True and masked is True, elements holding the null value
              will also be masked out

        Other optional arguments:
        realization, support, support_uuid, grid, continuous, points, count, indexable, property_kind, facet_type, facet,
        citation_title, time_series_uuid, time_index, uom, string_lookup_id, categorical:

        For each of these arguments: if None, then all members of collection pass this filter;
        if not None then only those members with the given value pass this filter;
        finally, the filters for all the attributes must be passed for a given member (part)
        to be selected

        returns:
           reference to a cached numpy array containing the actual property data for the part which matches all
           selection arguments which are not None

        notes:
           returns None if no parts match; raises an assertion error if more than one part matches;
           multiple calls will return the same cached array so calling code should copy if duplication is needed;
           support and grid arguments are for backward compatibilty: support_uuid takes precedence over support and
           both take precendence over grid

        :meta common:
        """

        if support is None:
            support = grid
        if support_uuid is None and support is not None:
            support_uuid = support.uuid

        part = self.singleton(realization = realization,
                              support_uuid = support_uuid,
                              uuid = uuid,
                              continuous = continuous,
                              points = points,
                              count = count,
                              indexable = indexable,
                              property_kind = property_kind,
                              facet_type = facet_type,
                              facet = facet,
                              citation_title = citation_title,
                              time_series_uuid = time_series_uuid,
                              time_index = time_index,
                              uom = uom,
                              string_lookup_uuid = string_lookup_uuid,
                              categorical = categorical)
        if part is None:
            return None
        return self.cached_part_array_ref(part, dtype = dtype, masked = masked, exclude_null = exclude_null)

    def number_of_parts(self):
        """Returns the number of parts (properties) in this collection.

        returns:
           count of the number of parts (members) in this collection; there is one part per property array (non-negative integer)

        :meta common:
        """

        return len(self.dict)

    def part_in_collection(self, part):
        """Returns True if named part is member of this collection; otherwise False.

        arguments:
           part (string): part name to be tested for membership of this collection

        returns:
           boolean
        """

        return part in self.dict

    # 'private' function for accessing an element from the tuple for the part
    # the main dictionary maps from part name to a tuple of information
    # this function simply extracts one element of the tuple in a way that returns None if the part is awol
    def element_for_part(self, part, index):
        if part not in self.dict:
            return None
        return self.dict[part][index]

    # 'private' function for returning a list of unique values for an element from the tuples within the collection
    # excludes None from list
    def unique_element_list(self, index, sort_list = True):
        s = set()
        for _, t in self.dict.items():
            e = t[index]
            if e is not None:
                s = s.union({e})
        result = list(s)
        if sort_list:
            result.sort()
        return result

    def part_str(self, part, include_citation_title = True):
        """Returns a human-readable string identifying the part.

        arguments:
           part (string): the part name for which a displayable string is required
           include_citation_title (boolean, default True): if True, the citation title for the part is
              included in parenthesis at the end of the returned string; otherwise it does not appear

        returns:
           a human readable string consisting of the property kind, the facet (if present), the
           time index (if applicable), and the citation title (if requested)

        note:
           the time index is labelled 'timestep' in the returned string; however, resqml differentiates
           between the simulator timestep number and a time index into a time series; at present this
           module conflates the two

        :meta common:
        """

        text = self.property_kind_for_part(part)
        facet = self.facet_for_part(part)
        if facet:
            text += ': ' + facet
        time_index = self.time_index_for_part(part)
        if time_index is not None:
            text += '; timestep: ' + str(time_index)
        if include_citation_title:
            title = self.citation_title_for_part(part)
            if title:
                text += ' (' + title + ')'
        return text

    def part_filename(self, part):
        """Returns a string which can be used as the starting point of a filename relating to part.

        arguments:
           part (string): the part name for which a partial filename is required

        returns:
           a string suitable as the basis of a filename for the part (typically used when exporting)
        """

        text = self.property_kind_for_part(part).replace(' ', '_')
        facet = self.facet_for_part(part)
        if facet:
            text += '_' + facet  # could insert facet_type prior to this
        time_index = self.time_index_for_part(part)
        if time_index is not None:
            text += '_ts_' + str(time_index)
        return text

    def realization_for_part(self, part):
        """Returns realization number (within ensemble) that the property relates to.

        arguments:
           part (string): the part name for which the realization number (realization index) is required

        returns:
           integer or None

        :meta common:
        """

        return self.element_for_part(part, 0)

    def realization_list(self, sort_list = True):
        """Returns a list of unique realization numbers present in the collection."""

        return self.unique_element_list(0)

    def support_uuid_for_part(self, part):
        """Returns supporting representation object's uuid that the property relates to.

        arguments:
           part (string): the part name for which the related support object uuid is required

        returns:
           uuid.UUID object (or string representation thereof)
        """

        return self.element_for_part(part, 1)

    def grid_for_part(self, part):
        """Returns grid object that the property relates to.

        arguments:
           part (string): the part name for which the related grid object is required

        returns:
           grid.Grid object reference

        note:
           this method maintained for backward compatibility and kept in base PropertyClass
           for pragmatic reasons (rather than being method in GridPropertyCollection)
        """

        import resqpy.grid as grr

        support_uuid = self.support_uuid_for_part(part)
        if support_uuid is None:
            return None
        if bu.matching_uuids(self.support_uuid, support_uuid):
            return self.support
        assert self.model is not None
        part = self.model.part_for_uuid(support_uuid)
        assert part is not None and self.model.type_of_part(part) in [
            'obj_IjkGridRepresentation', 'obj_UnstructuredGridRepresentation'
        ]
        return grr.any_grid(self.model, uuid = support_uuid, find_properties = False)

    def uuid_for_part(self, part):
        """Returns UUID object for the property part.

        arguments:
           part (string): the part name for which the UUID is required

        returns:
           uuid.UUID object reference; use str(uuid_for_part()) to convert to string

        :meta common:
        """

        return self.element_for_part(part, 2)

    def node_for_part(self, part):
        """Returns the xml node for the property part.

        arguments:
           part (string): the part name for which the xml node is required

        returns:
           xml Element object reference for the main xml node for the part
        """

        return self.element_for_part(part, 3)

    def extra_metadata_for_part(self, part):
        """Returns the extra_metadata dictionary for the part.

        arguments:
           part (string): the part name for which the xml node is required

        returns:
           dictionary containing extra_metadata for part
        """
        try:
            meta = self.element_for_part(part, 18)
        except Exception:
            pass
            meta = {}
        return meta

    def null_value_for_part(self, part):
        """Returns the null value for the (discrete) property part; np.NaN for continuous parts.

        arguments:
           part (string): the part name for which the null value is required

        returns:
           int or np.NaN
        """

        if self.continuous_for_part(part):
            return np.NaN
        return self.element_for_part(part, 19)

    def continuous_for_part(self, part):
        """Returns True if the property is continuous (including points); False if it is discrete (or categorical).

        arguments:
           part (string): the part name for which the continuous versus discrete flag is required

        returns:
           True if the part is representing a continuous (or points) property, ie. the array elements are
           real numbers (float); False if the part is representing a discrete property or a categorical property,
           ie the array elements are integers (or boolean)

        note:
           RESQML differentiates between discrete and categorical properties; discrete properties are
           unbounded integers where the values have numerical significance (eg. could be added together),
           whilst categorical properties have an associated dictionary mapping from a finite set of integer
           key values onto strings (eg. {1: 'background', 2: 'channel sand', 3: 'mud drape'}); however, this
           module treats categorical properties as a special case of discrete properties

        :meta common:
        """

        return self.element_for_part(part, 4)

    def points_for_part(self, part):
        """Returns True if the property is a points property; False otherwise.

        arguments:
           part (string): the part name for which the points flag is required

        returns:
           True if the part is representing a points property, ie. the array has an extra dimension of extent 3
           covering the xyz axes; False if the part is representing a non-points property
        """

        return self.element_for_part(part, 21)

    def all_continuous(self):
        """Returns True if all the parts are for continuous (real) properties (includes points)."""

        unique_elements = self.unique_element_list(4, sort_list = False)
        if len(unique_elements) != 1:
            return False
        return unique_elements[0]

    def all_discrete(self):
        """Returns True if all the parts are for discrete or categorical (integer) properties."""

        unique_elements = self.unique_element_list(4, sort_list = False)
        if len(unique_elements) != 1:
            return False
        return not unique_elements[0]

    def count_for_part(self, part):
        """Returns the Count value for the property part; usually 1.

        arguments:
           part (string): the part name for which the count is required

        returns:
           integer reflecting the count attribute for the part (usually one); if greater than one,
           the array has an extra axis, cycling fastest, having this extent

        note:
           this mechanism allows a vector of values to be associated with a single indexable element
           in the supporting representation
        """

        return self.element_for_part(part, 5)

    def all_count_one(self):
        """Returns True if the low level Count value is 1 for all the parts in the collection."""

        unique_elements = self.unique_element_list(5, sort_list = False)
        if len(unique_elements) != 1:
            return False
        return unique_elements[0] == 1

    def indexable_for_part(self, part):
        """Returns the text of the IndexableElement for the property part; usually 'cells' for grid properties.

        arguments:
           part (string): the part name for which the indexable element is required

        returns:
           string, usually 'cells' when the supporting representation is a grid or 'nodes' when a wellbore frame

        note:
           see tail of Representations.xsd for overview of indexable elements usable for other object classes

        :meta common:
        """

        return self.element_for_part(part, 6)

    def unique_indexable_element_list(self, sort_list = False):
        """Returns a list of unique values for the IndexableElement of the property parts in the collection."""

        return self.unique_element_list(6, sort_list = sort_list)

    def property_kind_for_part(self, part):
        """Returns the resqml property kind for the property part.

        arguments:
           part (string): the part name for which the property kind is required

        returns:
           standard resqml property kind or local property kind for this part, as a string, eg. 'porosity'

        notes:
           see attributes of this module named supported_property_kind_list and supported_local_property_kind_list
           for the property kinds which this module can relate to simulator keywords (Nexus); however, other property
           kinds should be handled okay in a generic way;
           for bespoke (local) property kinds, this is the property kind title as stored in the xml reference node

        :meta common:
        """

        return self.element_for_part(part, 7)

    def property_kind_list(self, sort_list = True):
        """Returns a list of unique property kinds found amongst the parts of the collection."""

        return self.unique_element_list(7, sort_list = sort_list)

    def local_property_kind_uuid(self, part):
        """Returns the uuid of the bespoke (local) property kind for this part, or None for a standard property kind."""

        return self.element_for_part(part, 17)

    def facet_type_for_part(self, part):
        """If relevant, returns the resqml Facet Facet for the property part, eg. 'direction'; otherwise None.

        arguments:
           part (string): the part name for which the facet type is required

        returns:
           standard resqml facet type for this part (string), or None

        notes:
           resqml refers to Facet Facet and Facet Value; the equivalents in this module are facet_type and facet;
           the resqml standard allows a property to have any number of facets; this module currently limits a
           property to having at most one facet; the facet_type and facet should be either both None or both not None

        :meta common:
        """

        return self.element_for_part(part, 8)

    def facet_type_list(self, sort_list = True):
        """Returns a list of unique facet types found amongst the parts of the collection."""

        return self.unique_element_list(8, sort_list = sort_list)

    def facet_for_part(self, part):
        """If relevant, returns the resqml Facet Value for the property part, eg. 'I'; otherwise None.

        arguments:
           part (string): the part name for which the facet value is required

        returns:
           facet value for this part (string), for the facet type returned by the facet_type_for_part() function,
           or None

        see notes for facet_type_for_part()

        :meta common:
        """

        return self.element_for_part(part, 9)

    def facet_list(self, sort_list = True):
        """Returns a list of unique facet values found amongst the parts of the collection."""

        return self.unique_element_list(9, sort_list = sort_list)

    def citation_title_for_part(self, part):
        """Returns the citation title for the property part.

        arguments:
           part (string): the part name for which the citation title is required

        returns:
           citation title (string) for this part

        note:
           for simulation grid properties, the citation title is often a property keyword specific to a simulator

        :meta common:
        """

        return self.element_for_part(part, 10)

    def title_for_part(self, part):
        """Synonymous with citation_title_for_part()."""

        return self.citation_title_for_part(part)

    def titles(self):
        """Returns a list of citation titles for the parts in the collection."""

        return [self.citation_title_for_part(p) for p in self.parts()]

    def time_series_uuid_for_part(self, part):
        """If the property has an associated time series (is not static), returns the uuid for the time series.

        arguments:
           part (string): the part name for which the time series uuid is required

        returns:
           time series uuid (uuid.UUID) for this part
        """

        return self.element_for_part(part, 11)

    def time_series_uuid_list(self, sort_list = True):
        """Returns a list of unique time series uuids found amongst the parts of the collection."""

        return self.unique_element_list(11, sort_list = sort_list)

    def time_index_for_part(self, part):
        """If the property has an associated time series (is not static), returns the time index within the time series.

        arguments:
           part (string): the part name for which the time index is required

        returns:
           time index (integer) for this part

        :meta common:
        """

        return self.element_for_part(part, 12)

    def time_index_list(self, sort_list = True):
        """Returns a list of unique time indices found amongst the parts of the collection."""

        return self.unique_element_list(12, sort_list = sort_list)

    def minimum_value_for_part(self, part):
        """Returns the minimum value for the property part, as stored in the xml.

        arguments:
           part (string): the part name for which the minimum value is required

        returns:
           minimum value (as float or int) for this part, or None if metadata item is not set

        note:
           this method merely returns the minimum value recorded in the xml for the property, it does not check
           the array data

        :meta common:
        """

        mini = self.element_for_part(part, 13)
        if mini:
            if self.continuous_for_part(part):
                mini = float(mini)
            else:
                mini = int(mini)
        return mini

    def maximum_value_for_part(self, part):
        """Returns the maximum value for the property part, as stored in the xml.

        arguments:
           part (string): the part name for which the maximum value is required

        returns:
           maximum value (as float ir int) for this part, or None if metadata item is not set

        note:
           this method merely returns the maximum value recorded in the xml for the property, it does not check
           the array data

        :meta common:
        """

        maxi = self.element_for_part(part, 14)
        if maxi:
            if self.continuous_for_part(part):
                maxi = float(maxi)
            else:
                maxi = int(maxi)
        return maxi

    def patch_min_max_for_part(self, part, minimum = None, maximum = None, model = None):
        """Updates the minimum and/ox maximum values stored in the metadata, optionally updating xml tree too.

        arguments:
           part (str): the part name of the property
           minimum (float or int, optional): the new minimum value to be set in the metadata (unchanged if None)
           maximum (float or int, optional): the new maximum value to be set in the metadata (unchanged if None)
           model (model.Model, optional): if present and containing xml for the part, that xml is also patched

        notes:
           this method is rarely needed: only if a property array is being re-populated after being initialised
           with temporary values; the xml tree for the part in the model will only be updated where the minimum
           and/or maximum nodes already exist in the tree
        """

        info = list(self.dict[part])
        if minimum is not None:
            info[13] = minimum
        if maximum is not None:
            info[14] = maximum
        self.dict[part] = tuple(info)
        if model is not None:
            p_root = model.root_for_part(part)
            if p_root is not None:
                if minimum is not None:
                    min_node = rqet.find_tag(p_root, 'MinimumValue')
                    if min_node is not None:
                        min_node.text = str(minimum)
                        model.set_modified()
                if maximum is not None:
                    max_node = rqet.find_tag(p_root, 'MaximumValue')
                    if max_node is not None:
                        max_node.text = str(maximum)
                        model.set_modified()

    def uom_for_part(self, part):
        """Returns the resqml units of measure for the property part.

        arguments:
           part (string): the part name for which the units of measure is required

        returns:
           resqml units of measure (string) for this part

        :meta common:
        """

        # NB: this field is not set correctly in data sets generated by DGI
        return self.element_for_part(part, 15)

    def uom_list(self, sort_list = True):
        """Returns a list of unique units of measure found amongst the parts of the collection."""

        return self.unique_element_list(15, sort_list = sort_list)

    def string_lookup_uuid_for_part(self, part):
        """If the property has an associated string lookup (is categorical), return the uuid.

        arguments:
           part (string): the part name for which the string lookup uuid is required

        returns:
           string lookup uuid (uuid.UUID) for this part
        """

        return self.element_for_part(part, 16)

    def string_lookup_for_part(self, part):
        """Returns a StringLookup object for the part, if it has a string lookup uuid, otherwise None."""

        sl_uuid = self.string_lookup_uuid_for_part(part)
        if sl_uuid is None:
            return None
        sl_root = self.model.root_for_uuid(sl_uuid)
        assert sl_root is not None, 'string table lookup referenced by property is not present in model'
        return StringLookup(self.model, sl_root)

    def string_lookup_uuid_list(self, sort_list = True):
        """Returns a list of unique string lookup uuids found amongst the parts of the collection."""

        return self.unique_element_list(16, sort_list = sort_list)

    def part_is_categorical(self, part):
        """Returns True if the property is categorical (not conintuous and has an associated string lookup).

        :meta common:
        """

        return not self.continuous_for_part(part) and self.string_lookup_uuid_for_part(part) is not None

    def constant_value_for_part(self, part):
        """Returns the value (float or int) of a constant array part, or None for an hdf5 array.

        note:
           a constant array can optionally be expanded and written to the hdf5, in which case it will
           not have a constant value assigned when the dataset is read from file
        :meta common:
        """

        return self.element_for_part(part, 20)

    def override_min_max(self, part, min_value, max_value):
        """Sets the minimum and maximum values in the metadata for the part.

        arguments:
           part (string): the part name for which the minimum and maximum values are to be set
           min_value (float or int or string): the minimum value to be stored in the metadata
           max_value (float or int or string): the maximum value to be stored in the metadata

        note:
           this function is typically called if the existing min & max metadata is missing or
           distrusted; the min and max values passed in are typically the result of numpy
           min and max function calls (possibly skipping NaNs) on the property array or
           a version of it masked for inactive cells
        """

        if part not in self.dict:
            return
        property_part = list(self.dict[part])
        property_part[13] = min_value
        property_part[14] = max_value
        self.dict[part] = tuple(property_part)

    def establish_time_set_kind(self):
        """Re-evaulates the time set kind attribute.

        Based on all properties having same time index in the same time series.
        """
        self.time_set_kind_attr = 'single time'
        #  note: other option of 'equivalent times' not catered for in this code
        common_time_index = None
        common_time_series_uuid = None
        for part in self.parts():
            part_time_index = self.time_index_for_part(part)
            if part_time_index is None:
                self.time_set_kind_attr = 'not a time set'
                break
            if common_time_index is None:
                common_time_index = part_time_index
            elif common_time_index != part_time_index:
                self.time_set_kind_attr = 'not a time set'
                break
            part_ts_uuid = self.time_series_uuid_for_part(part)
            if part_ts_uuid is None:
                self.time_set_kind_attr = 'not a time set'
                break
            if common_time_series_uuid is None:
                common_time_series_uuid = part_ts_uuid
            elif not bu.matching_uuids(common_time_series_uuid, part_ts_uuid):
                self.time_set_kind_attr = 'not a time set'
                break
        return self.time_set_kind_attr

    def time_set_kind(self):
        """Returns the time set kind attribute.

        Based on all properties having same time index in the same time series.
        """
        if self.time_set_kind_attr is None:
            self.establish_time_set_kind()
        return self.time_set_kind_attr

    def establish_has_single_property_kind(self):
        """Re-evaluates the has single property kind attribute.
        
        Depends on whether all properties are of the same kind.
        """
        self.has_single_property_kind_flag = True
        common_property_kind = None
        for part in self.parts():
            part_kind = self.property_kind_for_part(part)
            if common_property_kind is None:
                common_property_kind = part_kind
            elif part_kind != common_property_kind:
                self.has_single_property_kind_flag = False
                break
        return self.has_single_property_kind_flag

    def has_single_property_kind(self):
        """Return the has single property kind flag depending on whether all properties are of the same kind."""

        if self.has_single_property_kind_flag is None:
            self.establish_has_single_property_kind()
        return self.has_single_property_kind_flag

    def establish_has_single_indexable_element(self):
        """Re-evaluate the has single indexable element attribute.

        Depends on whether all properties have the same.
        """
        self.has_single_indexable_element_flag = True
        common_ie = None
        for part in self.parts():
            ie = self.indexable_for_part(part)
            if common_ie is None:
                common_ie = ie
            elif ie != common_ie:
                self.has_single_indexable_element_flag = False
                break
        return self.has_single_indexable_element_flag

    def has_single_indexable_element(self):
        """Returns the has single indexable element flag depending on whether all properties have the same."""

        if self.has_single_indexable_element_flag is None:
            self.establish_has_single_indexable_element()
        return self.has_single_indexable_element_flag

    def establish_has_multiple_realizations(self):
        """Re-evaluates the has multiple realizations attribute.

        Based on whether properties belong to more than one realization.
        """
        self.has_multiple_realizations_flag = False
        common_realization = None
        for part in self.parts():
            part_realization = self.realization_for_part(part)
            if part_realization is None:
                continue
            if common_realization is None:
                common_realization = part_realization
                continue
            if part_realization != common_realization:
                self.has_multiple_realizations_flag = True
                self.realization = None  # override single realization number applicable to whole collection
                break
        if not self.has_multiple_realizations_flag and common_realization is not None:
            self.realization = common_realization
        return self.has_multiple_realizations_flag

    def has_multiple_realizations(self):
        """Returns the has multiple realizations flag based on whether properties belong to more than one realization.

        :meta common:
        """

        if self.has_multiple_realizations_flag is None:
            self.establish_has_multiple_realizations()
        return self.has_multiple_realizations_flag

    def establish_has_single_uom(self):
        """Re-evaluates the has single uom attribute depending on whether all properties have the same units of measure."""

        self.has_single_uom_flag = True
        common_uom = None
        for part in self.parts():
            part_uom = self.uom_for_part(part)
            if common_uom is None:
                common_uom = part_uom
            elif part_uom != common_uom:
                self.has_single_uom_flag = False
                break
        if common_uom is None:
            self.has_single_uom_flag = True  # all uoms are None (probably discrete properties)
        return self.has_single_uom_flag

    def has_single_uom(self):
        """Returns the has single uom flag depending on whether all properties have the same units of measure."""

        if self.has_single_uom_flag is None:
            self.establish_has_single_uom()
        return self.has_single_uom_flag

    def assign_realization_numbers(self):
        """Assigns a distinct realization number to each property, after checking for compatibility.

        note:
           this method does not modify realization information in any established xml; it is intended primarily as
           a convenience to allow realization based processing of any collection of compatible properties
        """

        assert self.has_single_property_kind(), 'attempt to assign realization numbers to properties of differing kinds'
        assert self.has_single_indexable_element(
        ), 'attempt to assign realizations to properties of differing indexable elements'
        assert self.has_single_uom(
        ), 'attempt to assign realization numbers to properties with differing units of measure'

        new_dict = {}
        realization = 0
        for key, entry in self.dict.items():
            entry_list = list(entry)
            entry_list[0] = realization
            new_dict[key] = tuple(entry_list)
            realization += 1
        self.dict = new_dict
        self.has_multiple_realizations_flag = (realization > 1)

    def masked_array(self, simple_array, exclude_inactive = True, exclude_value = None, points = False):
        """Returns a masked version of simple_array, using inactive mask associated with support for this property collection.

        arguments:
           simple_array (numpy array): an unmasked numpy array with the same shape as property arrays for the support
              (and indexable element) associated with this collection
           exclude_inactive (boolean, default True): elements which are flagged as inactive in the supporting representation
              are masked out if this argument is True
           exclude_value (float or int, optional): if present, elements which match this value are masked out; if not None
              then usually set to np.NaN for continuous data or null_value_for_part() for discrete data
           points (boolean, default False): if True, the simple array is expected to have an extra dimension of extent 3,
              relative to the inactive attribute of the support

        returns:
           a masked version of the array, with the mask set to exclude cells which are inactive in the support

        notes:
           when requesting a reference to a cached copy of a property array (using other functions), a masked argument
           can be used to apply the inactive mask; this function is therefore rarely needed by calling code (it is used
           internally by this module); the simple_array need not be part of this collection
        """

        mask = None
        if (exclude_inactive and self.support is not None and hasattr(self.support, 'inactive') and
                self.support.inactive is not None):
            if not points:
                if self.support.inactive.shape == simple_array.shape:
                    mask = self.support.inactive
            else:
                assert simple_array.ndim > 1 and simple_array.shape[-1] == 3
                if (self.support.inactive.ndim + 1 == simple_array.ndim and
                        self.support.inactive.shape == tuple(simple_array.shape[:-1])):
                    mask = np.empty(simple_array.shape, dtype = bool)
                    mask[:] = self.support.inactive[:, np.newaxis]
        if exclude_value:
            null_mask = (simple_array == exclude_value)
            if mask is None:
                mask = null_mask
            else:
                mask = np.logical_or(mask, null_mask)
        if mask is None:
            mask = ma.nomask
        return ma.masked_array(simple_array, mask = mask)

    def h5_key_pair_for_part(self, part):
        """Return hdf5 key pair (ext uuid, internal path) for the part."""

        # note: this method does not currently support all the possible tag values for different instances
        # of the RESQML abstract arrays

        model = self.model
        part_node = self.node_for_part(part)
        if part_node is None:
            return None
        if self.points_for_part(part):
            patch_list = rqet.list_of_tag(part_node, 'PatchOfPoints')
            assert len(patch_list) == 1  # todo: handle more than one patch of points
            first_values_node = rqet.find_tag(patch_list[0], 'Points')
            tag = 'Coordinates'
        else:
            patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
            assert len(patch_list) == 1  # todo: handle more than one patch of values
            first_values_node = rqet.find_tag(patch_list[0], 'Values')
            tag = 'Values'
        if first_values_node is None:
            return None  # could treat as fatal error
        return model.h5_uuid_and_path_for_node(first_values_node, tag = tag)

    def cached_part_array_ref(self, part, dtype = None, masked = False, exclude_null = False):
        """Returns a numpy array containing the data for the property part; the array is cached in this collection.

        arguments:
           part (string): the part name for which the array reference is required
           dtype (optional, default None): the element data type of the array to be accessed, eg 'float' or 'int';
              if None (recommended), the dtype of the returned numpy array matches that in the hdf5 dataset
           masked (boolean, default False): if True, a masked array is returned instead of a simple numpy array;
              the mask is set to the inactive array attribute of the support object if present
           exclude_null (boolean, default False): if True, and masked is also True, then elements of the array
              holding the null value will also be masked out

        returns:
           reference to a cached numpy array containing the actual property data; multiple calls will return
           the same cached array so calling code should copy if duplication is needed

        notes:
           this function is the usual way to get at the actual property array; at present, the funtion only works
           if the entire array is stored as a single patch in the hdf5 file (resqml allows multiple patches per
           array); the masked functionality can be used to apply a common mask, stored in the supporting
           representation object with the attribute name 'inactive', to multiple properties (this will only work
           if the indexable element is set to the typical value for the class of supporting representation, eg.
           'cells' for grid objects); if exclude_null is set True then null value elements will also be masked out
           (as long as masked is True); however, it is recommended simply to use np.NaN values in floating point
           property arrays if the commonality is not needed

        :meta common:
        """

        model = self.model
        cached_array_name = _cache_name(part)
        if cached_array_name is None:
            return None

        if not hasattr(self, cached_array_name):
            self._cached_part_array_ref_get_array(part, dtype, model, cached_array_name)

        if masked:
            exclude_value = self.null_value_for_part(part) if exclude_null else None
            return self.masked_array(self.__dict__[cached_array_name], exclude_value = exclude_value)
        else:
            return self.__dict__[cached_array_name]

    def h5_slice(self, part, slice_tuple):
        """Returns a subset of the array for part, without loading the whole array.

        arguments:
           part (string): the part name for which the array slice is required
           slice_tuple (tuple of slice objects): each element should be constructed using the python built-in
              function slice()

        returns:
           numpy array that is a hyper-slice of the hdf5 array, with the same ndim as the source hdf5 array

        note:
           this method always fetches from the hdf5 file and does not attempt local caching; the whole array
           is not loaded; all axes continue to exist in the returned array, even where the sliced extent of
           an axis is 1
        """

        h5_key_pair = self.h5_key_pair_for_part(part)
        if h5_key_pair is None:
            return None
        return self.model.h5_array_slice(h5_key_pair, slice_tuple)

    def h5_overwrite_slice(self, part, slice_tuple, array_slice, update_cache = True):
        """Overwrites a subset of the array for part, in the hdf5 file.

        arguments:
           part (string): the part name for which the array slice is to be overwritten
           slice_tuple (tuple of slice objects): each element should be constructed using the python built-in
              function slice()
           array_slice (numpy array of shape to match slice_tuple): the data to be written
           update_cache (boolean, default True): if True and the part is currently cached within this
              PropertyCollection, then the cached array is also updated; if False, the part is uncached

        notes:
           this method naively writes the slice to hdf5 without using mpi to look after parallel writes;
           if a cached copy of the array is updated, this is in an unmasked form; if calling code has a
           reterence to a masked version of the array then the mask will not be updated by this method;
           if the part is not currently cached, this method will not cause it to become cached,
           regardless of the update_cache argument
        """

        h5_key_pair = self.h5_key_pair_for_part(part)
        assert h5_key_pair is not None
        self.model.h5_overwrite_array_slice(h5_key_pair, slice_tuple, array_slice)
        cached_array_name = _cache_name(part)
        if cached_array_name is None:
            return
        if hasattr(self, cached_array_name):
            if update_cache:
                self.__dict__[cached_array_name][slice_tuple] = array_slice
            else:
                delattr(self, cached_array_name)

    def shape_and_type_of_part(self, part):
        """Returns shape tuple and element type of cached or hdf5 array for part."""

        model = self.model
        cached_array_name = _cache_name(part)
        if cached_array_name is None:
            return None, None

        if hasattr(self, cached_array_name):
            return tuple(self.__dict__[cached_array_name].shape), self.__dict__[cached_array_name].dtype

        part_node = self.node_for_part(part)
        if part_node is None:
            return None, None

        if self.constant_value_for_part(part) is not None:
            assert not self.points_for_part(part), 'constant array not supported for points property'
            assert self.support is not None
            shape = self.supporting_shape(indexable_element = self.indexable_for_part(part),
                                          direction = self._part_direction(part))
            assert shape is not None
            return shape, (float if self.continuous_for_part(part) else int)

        if self.points_for_part(part):
            patch_list = rqet.list_of_tag(part_node, 'PatchOfpoints')
            assert len(patch_list) == 1  # todo: handle more than one patch of points
            h5_key_pair = model.h5_uuid_and_path_for_node(rqet.find_tag(patch_list[0], tag = 'Coordinates'))
        else:
            patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
            assert len(patch_list) == 1  # todo: handle more than one patch of values
            h5_key_pair = model.h5_uuid_and_path_for_node(rqet.find_tag(patch_list[0], tag = 'Values'))
        if h5_key_pair is None:
            return None, None
        return model.h5_array_shape_and_type(h5_key_pair)

    def facets_array_ref(self, use_32_bit = False, indexable_element = None):  # todo: add masked argument
        """Returns a +1D array of all parts with first axis being over facet values; Use facet_list() for lookup.

        arguments:
           use_32_bit (boolean, default False): if True, the resulting numpy array will use a 32 bit dtype; if False, 64 bit
           indexable_element (string, optional): the indexable element for the properties in the collection; if None, will
              be determined from the data

        returns:
           numpy array containing all the data in the collection, the first axis being over facet values and the rest of
           the axes matching the shape of the individual property arrays

        notes:
           the property collection should be constructed so as to hold a suitably coherent set of properties before
           calling this method;
           the facet_list() method will return the facet values that correspond to slices in the first axis of the
           resulting array
        """

        assert self.support is not None, 'attempt to build facets array for property collection without supporting representation'
        assert self.number_of_parts() > 0, 'attempt to build facets array for empty property collection'
        assert self.has_single_property_kind(
        ), 'attempt to build facets array for collection containing multiple property kinds'
        assert self.has_single_indexable_element(
        ), 'attempt to build facets array for collection containing a variety of indexable elements'
        assert self.has_single_uom(
        ), 'attempt to build facets array for collection containing multiple units of measure'

        #  could check that facet_type_list() has exactly one value
        facet_list = self.facet_list(sort_list = True)
        facet_count = len(facet_list)
        assert facet_count > 0, 'no facets found in property collection'
        assert self.number_of_parts() == facet_count, 'collection covers more than facet variability'

        continuous = self.all_continuous()
        if not continuous:
            assert self.all_discrete(), 'mixture of continuous and discrete properties in collection'

        if indexable_element is None:
            indexable_element = self.indexable_for_part(self.parts()[0])

        dtype = dtype_flavour(continuous, use_32_bit)
        shape_list = self.supporting_shape(indexable_element = indexable_element)
        shape_list.insert(0, facet_count)

        a = np.zeros(shape_list, dtype = dtype)

        for part in self.parts():
            facet_index = facet_list.index(self.facet_for_part(part))
            pa = self.cached_part_array_ref(part, dtype = dtype)
            a[facet_index] = pa
            self.uncache_part_array(part)

        return a

    def realizations_array_ref(self,
                               use_32_bit = False,
                               fill_missing = True,
                               fill_value = None,
                               indexable_element = None):
        """Returns a +1D array of all parts with first axis being over realizations.

        arguments:
           use_32_bit (boolean, default False): if True, the resulting numpy array will use a 32 bit dtype; if False, 64 bit
           fill_missing (boolean, default True): if True, the first axis of the resulting numpy array will range from 0 to
              the maximum realization number present and slices for any missing realizations will be filled with fill_value;
              if False, the extent of the first axis will only cpver the number pf realizations actually present (see also notes)
           fill_value (int or float, optional): the value to use for missing realization slices; if None, will default to
              np.NaN if data is continuous, -1 otherwise; irrelevant if fill_missing is False
           indexable_element (string, optional): the indexable element for the properties in the collection; if None, will
              be determined from the data

        returns:
           numpy array containing all the data in the collection, the first axis being over realizations and the rest of
           the axes matching the shape of the individual property arrays

        notes:
           the property collection should be constructed so as to hold a suitably coherent set of properties before
           calling this method;
           if fill_missing is False, the realization axis indices range from zero to the number of realizations present;
           if True, the realization axis indices range from zero to the maximum realization number and slices for missing
           realizations will be filled with the fill_value

        :meta common:
        """

        assert self.support is not None, 'attempt to build realizations array for property collection without supporting representation'
        assert self.number_of_parts() > 0, 'attempt to build realizations array for empty property collection'
        assert self.has_single_property_kind(
        ), 'attempt to build realizations array for collection with multiple property kinds'
        assert self.has_single_indexable_element(
        ), 'attempt to build realizations array for collection containing a variety of indexable elements'
        assert self.has_single_uom(
        ), 'attempt to build realizations array for collection containing multiple units of measure'

        r_list = self.realization_list(sort_list = True)
        assert self.number_of_parts() == len(r_list), 'collection covers more than realizations of a single property'

        continuous = self.all_continuous()
        if not continuous:
            assert self.all_discrete(), 'mixture of continuous and discrete properties in collection'

        if fill_value is None:
            fill_value = np.NaN if continuous else -1

        if indexable_element is None:
            indexable_element = self.indexable_for_part(self.parts()[0])

        if fill_missing:
            r_extent = r_list[-1] + 1
        else:
            r_extent = len(r_list)

        dtype = dtype_flavour(continuous, use_32_bit)
        # todo: handle direction dependent shapes
        shape_list = self.supporting_shape(indexable_element = indexable_element)
        shape_list.insert(0, r_extent)
        if self.points_for_part(self.parts()[0]):
            shape_list.append(3)

        a = np.full(shape_list, fill_value, dtype = dtype)

        if fill_missing:
            for part in self.parts():
                realization = self.realization_for_part(part)
                assert realization is not None and 0 <= realization < r_extent, 'realization missing (or out of range?)'
                pa = self.cached_part_array_ref(part, dtype = dtype)
                a[realization] = pa
                self.uncache_part_array(part)
        else:
            for index in range(len(r_list)):
                realization = r_list[index]
                part = self.singleton(realization = realization)
                pa = self.cached_part_array_ref(part, dtype = dtype)
                a[index] = pa
                self.uncache_part_array(part)

        return a

    def time_series_array_ref(self,
                              use_32_bit = False,
                              fill_missing = True,
                              fill_value = None,
                              indexable_element = None):
        """Returns a +1D array of all parts with first axis being over time indices.

        arguments:
           use_32_bit (boolean, default False): if True, the resulting numpy array will use a 32 bit dtype; if False, 64 bit
           fill_missing (boolean, default True): if True, the first axis of the resulting numpy array will range from 0 to
              the maximum time index present and slices for any missing indices will be filled with fill_value; if False,
              the extent of the first axis will only cpver the number pf time indices actually present (see also notes)
           fill_value (int or float, optional): the value to use for missing time index slices; if None, will default to
              np.NaN if data is continuous, -1 otherwise; irrelevant if fill_missing is False
           indexable_element (string, optional): the indexable element for the properties in the collection; if None, will
              be determined from the data

        returns:
           numpy array containing all the data in the collection, the first axis being over time indices and the rest of
           the axes matching the shape of the individual property arrays

        notes:
           the property collection should be constructed so as to hold a suitably coherent set of properties before
           calling this method;
           if fill_missing is False, the time axis indices range from zero to the number of time indices present,
           with the list of tine index values available by calling the method time_index_list(sort_list = True);
           if fill_missing is True, the time axis indices range from zero to the maximum time index and slices for
           missing time indices will be filled with the fill_value

        :meta common:
        """

        assert self.support is not None, 'attempt to build time series array for property collection without supporting representation'
        assert self.number_of_parts() > 0, 'attempt to build time series array for empty property collection'
        assert self.has_single_property_kind(
        ), 'attempt to build time series array for collection with multiple property kinds'
        assert self.has_single_indexable_element(
        ), 'attempt to build time series array for collection containing a variety of indexable elements'
        assert self.has_single_uom(
        ), 'attempt to build time series array for collection containing multiple units of measure'

        ti_list = self.time_index_list(sort_list = True)
        assert self.number_of_parts() == len(ti_list), 'collection covers more than time indices of a single property'

        continuous = self.all_continuous()
        if not continuous:
            assert self.all_discrete(), 'mixture of continuous and discrete properties in collection'

        if fill_value is None:
            fill_value = np.NaN if continuous else -1

        if indexable_element is None:
            indexable_element = self.indexable_for_part(self.parts()[0])

        if fill_missing:
            ti_extent = ti_list[-1] + 1
        else:
            ti_extent = len(ti_list)

        dtype = dtype_flavour(continuous, use_32_bit)
        # todo: handle direction dependent shapes
        shape_list = self.supporting_shape(indexable_element = indexable_element)
        shape_list.insert(0, ti_extent)
        if self.points_for_part(self.parts()[0]):
            shape_list.append(3)

        a = np.full(shape_list, fill_value, dtype = dtype)

        if fill_missing:
            for part in self.parts():
                time_index = self.time_index_for_part(part)
                assert time_index is not None and 0 <= time_index < ti_extent, 'time index missing (or out of range?)'
                pa = self.cached_part_array_ref(part, dtype = dtype)
                a[time_index] = pa
                self.uncache_part_array(part)
        else:
            for index in range(len(ti_list)):
                time_index = ti_list[index]
                part = self.singleton(time_index = time_index)
                pa = self.cached_part_array_ref(part, dtype = dtype)
                a[index] = pa
                self.uncache_part_array(part)

        return a

    def combobulated_face_array(self, resqml_a):
        """Returns a logically ordered copy of RESQML faces-per-cell property array resqml_a.

        argument:
           resqml_a (numpy array of shape (..., 6): a RESQML property array with indexable element faces per cell

        returns:
           numpy array of shape (..., 3, 2) where the 3 covers K,J,I and the 2 the -/+ face polarities being a resqpy logically
              arranged copy of resqml_a

        notes:
           this method is for properties of IJK grids only;
           RESQML documentation is not entirely clear about the required ordering of -I, +I, -J, +J faces;
           current implementation assumes count = 1 for the property;
           does not currently support points properties
        """

        assert resqml_a.shape[-1] == 6

        resqpy_a_shape = tuple(list(resqml_a.shape[:-1]) + [3, 2])
        resqpy_a = np.empty(resqpy_a_shape, dtype = resqml_a.dtype)

        for axis in range(3):
            for polarity in range(2):
                resqpy_a[..., axis, polarity] = resqml_a[..., self.face_index_map[axis, polarity]]

        return resqpy_a

    def discombobulated_face_array(self, resqpy_a):
        """Return logical face property array a, re-ordered and reshaped regarding the six facial directions.

        argument:
           resqpy_a (numpy array of shape (..., 3, 2)): the penultimate array axis represents K,J,I and the final axis is -/+ face
              polarity; the resqpy logically arranged property array to be converted to illogical RESQML ordering and shape

        returns:
           numpy array of shape (..., 6) being a copy of resqpy_a with slices reordered before collapsing the last 2 axes into 1;
              ready to be stored as a RESQML property array with indexable element faces per cell

        notes:
           this method is for properties of IJK grids only;
           RESQML documentation is not entirely clear about the required ordering of -I, +I, -J, +J faces;
           current implementation assumes count = 1 for the property;
           does not currently support points properties
        """

        assert resqpy_a.ndim >= 2 and resqpy_a.shape[-2] == 3 and resqpy_a.shape[-1] == 2

        resqml_a_shape = tuple(list(resqpy_a.shape[:-2]) + [6])
        resqml_a = np.empty(resqml_a_shape, dtype = resqpy_a.dtype)

        for face in range(6):
            axis, polarity = self.face_index_inverse_map[face]
            resqml_a[..., face] = resqpy_a[..., axis, polarity]

        return resqml_a

    def cached_normalized_part_array_ref(self,
                                         part,
                                         masked = False,
                                         use_logarithm = False,
                                         discrete_cycle = None,
                                         trust_min_max = False):
        """DEPRECATED: replaced with normalized_part_array() method."""

        return self.normalized_part_array(part,
                                          masked = masked,
                                          use_logarithm = use_logarithm,
                                          discrete_cycle = discrete_cycle,
                                          trust_min_max = trust_min_max)

    def normalized_part_array(self,
                              part,
                              masked = False,
                              use_logarithm = False,
                              discrete_cycle = None,
                              trust_min_max = False,
                              fix_zero_at = None):
        """Return data normalised to between 0 and 1, along with min and max value.

        arguments:
           part (string): the part name for which the normalized array reference is required
           masked (boolean, optional, default False): if True, the masked version of the property array is used to
              determine the range of values to map onto the normalized range of 0 to 1 (the mask removes inactive cells
              from having any impact); if False, the values of inactive cells are included in the operation; the returned
              normalized array is masked or not depending on this argument
           use_logarithm (boolean, optional, default False): if False, the property values are linearly mapped to the
              normalized range; if True, the logarithm (base 10) of the property values are mapped to the normalized range
           discrete_cycle (positive integer, optional, default None): if a value is supplied and the property array
              contains integer data (discrete or categorical), the modulus of the property values are calculated
              against this value before conversion to floating point and mapping to the normalized range
           trust_min_max (boolean, optional, default False): if True, the minimum and maximum values from the property's
              metadata is used as the range of the property values; if False, the values are determined using numpy
              min and max operations
           fix_zero_at (float, optional): if present, a value between 0.0 and 1.0 (typically 0.0 or 0.5) to pin zero at

        returns:
           (normalized_array, min_value, max_value) where:
           normalized_array is a numpy array of floats, masked or unmasked depending on the masked argument, with values
           ranging between 0 and 1; in the case of a masked array the values for excluded cells are meaningless and may
           lie outside the range 0 to 1
           min_value and max_value: the property values that have been mapped to 0 and 1 respectively

        notes:
           this function is typically used to map property values onto the range required for colouring in;
           in case of failure, (None, None, None) is returned;
           if use_logarithm is True, the min_value and max_value returned are the log10 values, not the original
           property values;
           also, if use logarithm is True and the minimum property value is not greater than zero, then values less than
           0.0001 are set to 0.0001, prior to taking the logarithm;
           fix_zero_at is mutually incompatible with use_logarithm; to force the normalised data to have a true zero,
           set fix_zero_at to 0.0; for divergent data fixing zero at 0.5 will often be appropriate;
           fixing zero at 0.0 or 1.0 may result in normalised values being clipped;
           for floating point data, NaN values will be handled okay; if all data are NaN, (None, NaN, NaN) is returned;
           for integer data, null values are not currently supported (though the RESQML metadata can hold a null value);
           the masked argument is most applicable to properties for grid objects; note that NaN values are excluded when
           determining the min and max regardless of the value of the masked argument;
           not applicable to points properties
        """

        assert not self.points_for_part(part), 'property normalisation not available for points properties'
        assert fix_zero_at is None or not use_logarithm

        p_array = self.cached_part_array_ref(part, masked = masked)

        if p_array is None:
            return None, None, None

        min_value, max_value = self._normalized_part_array_get_minmax(trust_min_max, part, p_array, masked)

        if min_value is None or max_value is None:
            return None, min_value, max_value

        min_value, max_value, p_array = _normalized_part_array_apply_discrete_cycle(discrete_cycle, p_array, min_value, max_value)
        min_value, max_value = _normalized_part_array_nan_if_masked(min_value, max_value, masked)

        if min_value == np.nan or max_value == np.nan:
            return None, min_value, max_value
        if max_value < min_value:
            return None, min_value, max_value

        n_prop = p_array.astype(float)
        if use_logarithm:
            min_value, max_value = _normalized_part_array_use_logarithm(min_value, n_prop, masked)
            if min_value == np.nan or max_value == np.nan:
                return None, min_value, max_value

        if fix_zero_at is not None:
            min_value, max_value, n_prop = _normalized_part_array_fix_zero_at(min_value, max_value, n_prop, fix_zero_at)

        if max_value == min_value:
            n_prop[:] = 0.5
            return n_prop, min_value, max_value

        return (n_prop - min_value) / (max_value - min_value), min_value, max_value

    def uncache_part_array(self, part):
        """Removes the cached copy of the array of data for the named property part.

        argument:
           part (string): the part name for which the cached array is to be removed

        note:
           this function applies a python delattr() which will mark the array as no longer being in use
           here; however, actual freeing of the memory only happens when all other references to the
           array are released
        """

        cached_array_name = _cache_name(part)
        if cached_array_name is not None and hasattr(self, cached_array_name):
            delattr(self, cached_array_name)

    def add_cached_array_to_imported_list(self,
                                          cached_array,
                                          source_info,
                                          keyword,
                                          discrete = False,
                                          uom = None,
                                          time_index = None,
                                          null_value = None,
                                          property_kind = None,
                                          local_property_kind_uuid = None,
                                          facet_type = None,
                                          facet = None,
                                          realization = None,
                                          indexable_element = None,
                                          count = 1,
                                          const_value = None,
                                          points = False):
        """Caches array and adds to the list of imported properties (but not to the collection dict).

        arguments:
           cached_array: a numpy array to be added to the imported list for this collection (prior to being added
              as a part); for a constant array set cached_array to None (and use const_value)
           source_info (string): typically the name of a file from which the array has been read but can be any
              information regarding the source of the data
           keyword (string): this will be used as the citation title when a part is generated for the array
           discrete (boolean, optional, default False): if True, the array should contain integer (or boolean)
              data; if False, float
           uom (string, optional, default None): the resqml units of measure for the data
           time_index (integer, optional, default None): if not None, the time index to be used when creating
              a part for the array
           null_value (int or float, optional, default None): if present, this is used in the metadata to
              indicate that this value is to be interpreted as a null value wherever it appears in the data
           property_kind (string): resqml property kind, or None
           local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
           facet_type (string): resqml facet type, or None
           facet (string): resqml facet, or None
           realization (int): realization number, or None
           indexable_element (string, optional): the indexable element in the supporting representation
           count (int, default 1): the number of values per indexable element; if greater than one then this
              must be the fastest cycling axis in the cached array, ie last index
           const_value (int, float or bool, optional): the value with which a constant array is filled;
              required if cached_array is None, must be None otherwise
           points (bool, default False): if True, this is a points property with an extra dimension of extent 3

        returns:
           uuid of nascent property object

        note:
           the process of importing property arrays follows these steps:
           1. read (or generate) array of data into a numpy array in memory (cache)
           2. add to the imported list using this function (which also takes takes note of some metadata)
           3. write imported list arrays to hdf5 file
           4. create resqml xml nodes for imported list arrays and add as parts to model
           5. include newly added parts in collection

        :meta common:
        """

        assert (cached_array is not None and const_value is None) or (cached_array is None and const_value is not None)
        assert not points or not discrete
        assert count > 0
        if self.imported_list is None:
            self.imported_list = []

        uuid = bu.new_uuid()
        cached_name = _cache_name_for_uuid(uuid)
        if cached_array is not None:
            self.__dict__[cached_name] = cached_array
            zorro = self.masked_array(cached_array, exclude_value = null_value)
            if not discrete and np.all(np.isnan(zorro)):
                min_value = max_value = None
            elif discrete:
                min_value = int(np.nanmin(zorro))
                max_value = int(np.nanmax(zorro))
            else:
                min_value = np.nanmin(zorro)
                max_value = np.nanmax(zorro)
            if min_value is ma.masked or min_value == np.NaN:
                min_value = None
            if max_value is ma.masked or max_value == np.NaN:
                max_value = None
        else:
            if const_value == null_value or (not discrete and np.isnan(const_value)):
                min_value = max_value = None
            else:
                min_value = max_value = const_value
        self.imported_list.append((uuid, source_info, keyword, cached_name, discrete, uom, time_index, null_value,
                                   min_value, max_value, property_kind, facet_type, facet, realization,
                                   indexable_element, count, local_property_kind_uuid, const_value, points))
        return uuid

    def remove_cached_imported_arrays(self):
        """Removes any cached arrays that are mentioned in imported list."""

        for imported in self.imported_list:
            cached_name = imported[3]
            if hasattr(self, cached_name):
                delattr(self, cached_name)

    def remove_cached_part_arrays(self):
        """Removes any cached arrays for parts of the collection."""

        for part in self.dict:
            self.uncache_part_array(part)

    def remove_all_cached_arrays(self):
        """Removes any cached arrays for parts or mentioned in imported list."""

        self.remove_cached_imported_arrays()
        self.remove_cached_part_arrays()

    def write_hdf5_for_imported_list(self, file_name = None, mode = 'a', expand_const_arrays = False):
        """Create or append to an hdf5 file, writing datasets for the imported arrays.

        arguments:
           file_name (str, optional): if present, this hdf5 filename will override the default
           mode (str, default 'a'): the mode to open the hdf5 file in, either 'a' (append), or 'w' (overwrite)
           expand_const_arrays (boolean, default False): if True, constant arrays will be written in full to
              the hdf5 file and the same argument should be used when creating the xml

        :meta common:
        """

        # NB: imported array data must all have been cached prior to calling this function
        assert self.imported_list is not None
        h5_reg = rwh5.H5Register(self.model)
        for ei, entry in enumerate(self.imported_list):
            if entry[17] is not None:  # array has constant value
                if not expand_const_arrays:
                    continue  # constant array – handled entirely in xml
                uuid = entry[0]
                cached_name = _cache_name_for_uuid(uuid)
                assert self.support is not None
                # note: will not handle direction dependent shapes
                shape = self.supporting_shape(indexable_element = entry[14])
                value = float(entry[17]) if isinstance(entry[17], str) else entry[17]
                self.__dict__[cached_name] = np.full(shape, value)
            else:
                uuid = entry[0]
                cached_name = entry[3]
            tail = 'points_patch0' if entry[18] else 'values_patch0'
            h5_reg.register_dataset(uuid, tail, self.__dict__[cached_name])
        h5_reg.write(file = file_name, mode = mode)

    def write_hdf5_for_part(self, part, file_name = None, mode = 'a'):
        """Create or append to an hdf5 file, writing dataset for the specified part."""

        if self.constant_value_for_part(part) is not None:
            return
        h5_reg = rwh5.H5Register(self.model)
        a = self.cached_part_array_ref(part)
        tail = 'points_patch0' if self.points_for_part(part) else 'values_patch0'
        h5_reg.register_dataset(self.uuid_for_part(part), tail, a)
        h5_reg.write(file = file_name, mode = mode)

    def create_xml_for_imported_list_and_add_parts_to_model(self,
                                                            ext_uuid = None,
                                                            support_uuid = None,
                                                            time_series_uuid = None,
                                                            selected_time_indices_list = None,
                                                            string_lookup_uuid = None,
                                                            property_kind_uuid = None,
                                                            find_local_property_kinds = True,
                                                            expand_const_arrays = False,
                                                            extra_metadata = {}):
        """Add imported or generated grid property arrays as parts in parent model, creating xml.
        
        hdf5 should already have been written.

        arguments:
           ext_uuid: uuid for the hdf5 external part, which must be known to the model's hdf5 dictionary
           support_uuid (optional): the uuid of the supporting representation that the imported properties relate to
           time_series_uuid (optional): the uuid of the full or reduced time series for which any recurrent properties'
              timestep numbers can be used as a time index; in the case of a reduced series, the selected_time_indices_list
              argument must be passed and the properties timestep numbers are found in the list with the position yielding
              the time index for the reduced list; time_series_uuid should be present if there are any recurrent properties
              in the imported list
           selected_time_indices_list (list of int, optional): if time_series_uuid is for a reduced time series then this
              argument must be present and its length must match the number of timestamps in the reduced series; the values
              in the list are indices in the full time series
           string_lookup_uuid (optional): if present, the uuid of the string table lookup which any non-continuous
              properties relate to (ie. they are all taken to be categorical)
           property_kind_uuid (optional): if present, the uuid of the bespoke (local) property kind for all the
              property arrays in the imported list (except those with an individual local property kind uuid)
           find_local_property_kinds (boolean, default True): if True, local property kind uuids need not be provided as
              long as the property kinds are set to match the titles of the appropriate local property kind objects
           expand_const_arrays (boolean, default False): if True, the hdf5 write must also have been called with the
              same argument and the xml will treat the constant arrays as normal arrays
           extra_metadata (optional): if present, a dictionary of extra metadata to be added for the part

        returns:
           list of uuid.UUID, being the uuids of the newly added property parts

        notes:
           the imported list should have been built up, and associated hdf5 arrays written, before calling this method;
           the imported list is cleared as a deliberate side-effect of this method (so a new set of imports can be
           started hereafter);
           discrete and categorical properties cannot be mixed in the same import list - process as separate lists;
           all categorical properties in the import list must refer to the same string table lookup;
           when importing categorical properties, establish the xml for the string table lookup before calling this method;
           if importing properties of a bespoke (local) property kind, ensure the property kind objects exist as parts in
           the model before calling this method

        :meta common:
        """

        if self.imported_list is None:
            return []
        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()
        prop_parts_list = []
        uuid_list = []
        for (p_uuid, p_file_name, p_keyword, p_cached_name, p_discrete, p_uom, p_time_index, p_null_value, p_min_value,
             p_max_value, property_kind, facet_type, facet, realization, indexable_element, count,
             local_property_kind_uuid, const_value, points) in self.imported_list:
            log.debug('processing imported property ' + str(p_keyword))
            assert not points or not p_discrete
            if local_property_kind_uuid is None:
                local_property_kind_uuid = property_kind_uuid
            if property_kind is None:
                if local_property_kind_uuid is not None:
                    # note: requires local property kind to be present
                    property_kind = self.model.title(uuid = local_property_kind_uuid)
                else:
                    # todo: only if None in ab_property_list
                    (property_kind, facet_type, facet) = property_kind_and_facet_from_keyword(p_keyword)
            if property_kind is None:
                # todo: the following are abstract standard property kinds, which shouldn't really have data directly associated with them
                if p_discrete:
                    if string_lookup_uuid is not None:
                        property_kind = 'categorical'
                    else:
                        property_kind = 'discrete'
                elif points:
                    property_kind = 'length'
                else:
                    property_kind = 'continuous'
            if hasattr(self, p_cached_name):
                p_array = self.__dict__[p_cached_name]
            else:
                p_array = None
            if points or property_kind == 'categorical':
                add_min_max = False
            elif local_property_kind_uuid is not None and string_lookup_uuid is not None:
                add_min_max = False
            else:
                add_min_max = True
            if selected_time_indices_list is not None and p_time_index is not None:
                p_time_index = selected_time_indices_list.index(p_time_index)
            p_node = self.create_xml(
                ext_uuid = ext_uuid,
                property_array = p_array,
                title = p_keyword,
                property_kind = property_kind,
                support_uuid = support_uuid,
                p_uuid = p_uuid,
                facet_type = facet_type,
                facet = facet,
                discrete = p_discrete,  # todo: time series bits
                time_series_uuid = time_series_uuid,
                time_index = p_time_index,
                uom = p_uom,
                null_value = p_null_value,
                originator = None,
                source = p_file_name,
                add_as_part = True,
                add_relationships = True,
                add_min_max = add_min_max,
                min_value = p_min_value,
                max_value = p_max_value,
                realization = realization,
                string_lookup_uuid = string_lookup_uuid,
                property_kind_uuid = local_property_kind_uuid,
                indexable_element = indexable_element,
                count = count,
                points = points,
                find_local_property_kinds = find_local_property_kinds,
                extra_metadata = extra_metadata,
                const_value = const_value,
                expand_const_arrays = expand_const_arrays)
            if p_node is not None:
                prop_parts_list.append(rqet.part_name_for_part_root(p_node))
                uuid_list.append(rqet.uuid_for_part_root(p_node))
        self.add_parts_list_to_dict(prop_parts_list)
        self.imported_list = []
        return uuid_list

    def create_xml(self,
                   ext_uuid,
                   property_array,
                   title,
                   property_kind,
                   support_uuid = None,
                   p_uuid = None,
                   facet_type = None,
                   facet = None,
                   discrete = False,
                   time_series_uuid = None,
                   time_index = None,
                   uom = None,
                   null_value = None,
                   originator = None,
                   source = None,
                   add_as_part = True,
                   add_relationships = True,
                   add_min_max = True,
                   min_value = None,
                   max_value = None,
                   realization = None,
                   string_lookup_uuid = None,
                   property_kind_uuid = None,
                   find_local_property_kinds = True,
                   indexable_element = None,
                   count = 1,
                   points = False,
                   extra_metadata = {},
                   const_value = None,
                   expand_const_arrays = False):
        """Create a property xml node for a single property related to a given supporting representation node.

        arguments:
           ext_uuid (uuid.UUID): the uuid of the hdf5 external part
           property_array (numpy array): the actual property array (used to populate xml min & max values);
              may be None if min_value and max_value are passed or add_min_max is False
           title (string): used for the citation Title text for the property; often set to a simulator keyword for
              grid properties
           property_kind (string): the resqml property kind of the property; in the case of a bespoke (local)
              property kind, this is used as the title in the local property kind reference and the
              property_kind_uuid argument must also be passed or find_local_property_kinds set True
           support_uuid (uuid.UUID, optional): if None, the support for the collection is used
           p_uuid (uuid.UUID, optional): if None, a new uuid is generated for the property; otherwise this
              uuid is used
           facet_type (string, optional): if present, a resqml facet type whose value is supplied in the facet argument
           facet (string, optional): required if facet_type is supplied; the value of the facet
           discrete (boolean, default False): if True, a discrete or categorical property node is created (depending
              on whether string_lookup_uuid is None or present); if False (default), a continuous property node is created
           time_series_uuid (uuid.UUID, optional): if present, the uuid of the time series that this (recurrent)
              property relates to
           time_index (int, optional): if time_series_uuid is not None, this argument is required and provides
              the time index into the time series for this property array
           uom (string): the resqml unit of measure for the property (only used for continuous properties)
           null_value (optional): the value that is to be interpreted as null if it appears in the property array
           originator (string, optional): the name of the human being who created the property object;
              default is to use the login name
           source (string, optional): if present, an extra metadata node is added as a child to the property
              node, with this string indicating the source of the property data
           add_as_part (boolean, default True): if True, the newly created xml node is added as a part
              in the model
           add_relationships (boolean, default True): if True, relationship xml parts are created relating the
              new property part to: the support, the hdf5 external part; and the time series part (if applicable)
           add_min_max (boolean, default True): if True, min and max values are included as children in the
              property node
           min_value (optional): if present and add_min_max is True, this is used as the minimum value (otherwise it
              is calculated from the property array)
           max_value (optional): if present and add_min_max is True, this is used as the maximum value (otherwise it
              is calculated from the property array)
           realization (int, optional): if present, is used as the realization number in the property node; if None,
              no realization child is created
           string_lookup_uuid (optional): if present, and discrete is True, a categorical property node is created
              which refers to this string table lookup
           property_kind_uuid (optional): if present, the property kind is a local property kind; must be None for a
              standard property kind
           find_local_property_kinds (boolean, default True): if True and property_kind is not in standard supported
              property kind list and property_kind_uuid is None, the citation titles of PropertyKind objects in the
              model are compared with property_kind and if a match is found, that local property kind is used
           indexable_element (string, optional): if present, is used as the indexable element in the property node;
              if None, 'cells' are used for grid properties and 'nodes' for wellbore frame properties
           count (int, default 1): the number of values per indexable element; if greater than one then this axis
              must cycle fastest in the array, ie. be the last index
           points (bool, default False): if True, this is a points property
           extra_metadata (dictionary, optional): if present, adds extra metadata in the xml
           const_value (float or int, optional): if present, create xml for a constant array filled with this value
           expand_const_arrays (boolean, default False): if True, the hdf5 write must also have been called with the
              same argument and the xml will treat a constant array as a normal array

        returns:
           the newly created property xml node

        notes:
           this function doesn't write the actual array data to the hdf5 file: that should be done
           before calling this function;
           this code (and elsewhere) only supports at most one facet per property, though the RESQML standard
           allows for multiple facets;
           RESQML does not allow facets for points properties;
           if the xml has not been created for the support object, then xml will not be created for relationships
           between the properties and the supporting representation
        """

        #      log.debug('creating property node for ' + title)
        # currently assumes discrete properties to be 32 bit integers and continuous to be 64 bit reals
        # also assumes property_kind is one of the standard resqml property kinds; todo: allow local p kind node as optional arg
        assert not discrete or not points
        assert not points or const_value is None
        assert not points or facet_type is None
        assert self.model is not None

        if null_value is not None:
            self.null_value = null_value

        if support_uuid is None:
            support_uuid = self.support_uuid
        assert support_uuid is not None
        support_root = self.model.root_for_uuid(support_uuid)
        # assert support_root is not None

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()

        if expand_const_arrays:
            const_value = None

        support_type = self.model.type_of_part(self.model.part_for_uuid(support_uuid))

        indexable_element = _get_indexable_element(indexable_element, support_type)

        direction = None if facet_type is None or facet_type != 'direction' else facet

        if self.support is not None:
            self._check_shape_list(indexable_element, direction, property_array, points, count)

        # todo: assertions:
        #    numpy data type matches discrete flag (and assumptions about precision)
        #    uom are valid units for property_kind
        assert property_kind, 'missing property kind when creating xml for property'

        self._get_property_type_details(discrete, string_lookup_uuid, points)

        p_node, p_uuid = self._create_xml_get_p_node(p_uuid)

        self.model.create_citation(root = p_node, title = title, originator = originator)

        rqet.create_metadata_xml(node = p_node, extra_metadata = extra_metadata)

        if source is not None and len(source) > 0:
            self.model.create_source(source = source, root = p_node)

        count_node = rqet.SubElement(p_node, ns['resqml2'] + 'Count')
        count_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        count_node.text = str(count)

        ie_node = rqet.SubElement(p_node, ns['resqml2'] + 'IndexableElement')
        ie_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IndexableElements')
        ie_node.text = indexable_element

        self._create_xml_realization_node(realization, p_node)

        related_time_series_node = self._create_xml_time_series_node(time_series_uuid, time_index, p_node, support_uuid,
                                                                     support_type, support_root)

        self._create_xml_property_kind(p_node, find_local_property_kinds, property_kind, uom, discrete,
                                       property_kind_uuid)
        self._create_xml_patch_node(p_node, points, const_value, indexable_element, direction, p_uuid, ext_uuid)
        _create_xml_facet_node(facet_type, facet, p_node)

        if add_min_max:
            # todo: use active cell mask on numpy min and max operations; exclude null values on discrete min max
            min_value, max_value = self._get_property_array_min_max_value(property_array, const_value, discrete,
                                                                          min_value, max_value)

            self._create_xml_property_min_max(p_node, min_value, max_value)  # TODO: continue here!

        sl_root = None
        if discrete:
            sl_root = self._create_xml_lookup_node(p_node, string_lookup_uuid)

        else:  # continuous
            self._create_xml_uom_node(p_node, uom, property_kind, min_value, max_value, facet_type, facet, title)

        self._create_xml_add_as_part(add_as_part, p_uuid, p_node, add_relationships, support_root, property_kind_uuid,
                                     related_time_series_node, sl_root, discrete, string_lookup_uuid, const_value,
                                     ext_uuid)

        return p_node

    def create_property_set_xml(self,
                                title,
                                ps_uuid = None,
                                originator = None,
                                add_as_part = True,
                                add_relationships = True):
        """Creates an xml node for a property set to represent this collection of properties.

        arguments:
           title (string): to be used as citation title
           ps_uuid (string, optional): if present, used as the uuid for the property set, otherwise a new uuid is generated
           originator (string, optional): if present, used as the citation creator (otherwise login name is used)
           add_as_part (boolean, default True): if True, the property set is added to the model as a part
           add_relationships (boolean, default True): if True, the relationships to the member properties are added

        note:
           xml for individual properties should exist before calling this method
        """

        assert self.model is not None, 'cannot create xml for property set as model is not set'
        assert self.number_of_parts() > 0, 'cannot create xml for property set as no parts in collection'

        ps_node = self.model.new_obj_node('PropertySet')
        if ps_uuid is None:
            ps_uuid = bu.uuid_from_string(ps_node.attrib['uuid'])
        else:
            ps_node.attrib['uuid'] = str(ps_uuid)

        self.model.create_citation(root = ps_node, title = title, originator = originator)

        tsk_node = rqet.SubElement(ps_node, ns['resqml2'] + 'TimeSetKind')
        tsk_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeSetKind')
        tsk_node.text = self.time_set_kind()

        hspk_node = rqet.SubElement(ps_node, ns['resqml2'] + 'HasSinglePropertyKind')
        hspk_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
        hspk_node.text = str(self.has_single_property_kind()).lower()

        hmr_node = rqet.SubElement(ps_node, ns['resqml2'] + 'HasMultipleRealizations')
        hmr_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
        hmr_node.text = str(self.has_multiple_realizations()).lower()

        if self.parent_set_root is not None:
            parent_set_ref_node = self.model.create_ref_node('ParentSet',
                                                             self.model.title_for_root(self.parent_set_root),
                                                             self.parent_set_root.attrib['uuid'],
                                                             content_type = 'obj_PropertySet',
                                                             root = ps_node)

        prop_node_list = []
        for part in self.parts():
            part_root = self.model.root_for_part(part)
            self.model.create_ref_node('Properties',
                                       self.model.title_for_root(part_root),
                                       part_root.attrib['uuid'],
                                       content_type = self.model.type_of_part(part),
                                       root = ps_node)
            if add_as_part and add_relationships:
                prop_node_list.append(part_root)

        if add_as_part:
            self.model.add_part('obj_PropertySet', ps_uuid, ps_node)
            if add_relationships:
                # todo: add relationship with time series if time set kind is not 'not a time set'?
                if self.parent_set_root is not None:
                    self.model.create_reciprocal_relationship(ps_node, 'destinationObject', parent_set_ref_node,
                                                              'sourceObject')
                for prop_node in prop_node_list:
                    self.model.create_reciprocal_relationship(ps_node, 'destinationObject', prop_node, 'sourceObject')

        return ps_node

    def basic_static_property_parts(self,
                                    realization = None,
                                    share_perm_parts = False,
                                    perm_k_mode = None,
                                    perm_k_ratio = 1.0):
        """Returns five parts: net to gross ratio, porosity, permeability rock I, J & K; each returned part may be None.

        arguments:
           realization: (int, optional): if present, only properties with the given realization are considered; if None,
              all properties in the collection are considered
           share_perm_parts (boolean, default False): if True, the permeability I part will also be returned for J and/or K
              if no other properties are found for those directions; if False, None will be returned for such parts
           perm_k_mode (string, optional): if present, indicates what action to take when no K direction permeability is
              found; valid values are:
              'none': same as None, perm K part return value will be None
              'shared': if share_perm_parts is True, then perm I value will be used for perm K, else same as None
              'ratio': multiply IJ permeability by the perm_k_ratio argument
              'ntg': multiply IJ permeability by ntg and by perm_k_ratio
              'ntg squared': multiply IJ permeability by square of ntg and by perm_k_ratio
           perm_k_ratio (float, default 1.0): a Kv:Kh ratio, typically in the range zero to one, applied if generating
              a K permeability array (perm_k_mode is 'ratio', 'ntg' or 'ntg squared' and no existing K permeability found);
              ignored otherwise

        returns:
           tuple of 5 strings being part names for: net to gross ratio, porosity, perm i, perm j, perm k (respectively);
           any of the returned elements may be None if no appropriate property was identified

        note:
           if generating a K permeability array, the data is appended to the hdf5 file and the xml is created, however
           the epc re-write must be carried out by the calling code

        :meta common:
        """

        perm_i_part = perm_j_part = perm_k_part = None

        ntg_part = self._find_single_part('net to gross ratio', realization)
        poro_part = self._find_single_part('porosity', realization)

        perms = selective_version_of_collection(self, realization = realization, property_kind = 'permeability rock')
        if perms is None or perms.number_of_parts() == 0:
            log.error('no rock permeabilities present')
        else:
            perm_i_part, perm_j_part, perm_k_part = self._get_single_perm_ijk_parts(perms, share_perm_parts,
                                                                                    perm_k_mode, perm_k_ratio, ntg_part)

        return ntg_part, poro_part, perm_i_part, perm_j_part, perm_k_part

    def basic_static_property_parts_dict(self,
                                         realization = None,
                                         share_perm_parts = False,
                                         perm_k_mode = None,
                                         perm_k_ratio = 1.0):
        """Same as basic_static_property_parts() method but returning a dictionary with 5 items.

        note:
           returned dictionary contains following keys: 'NTG', 'PORO', 'PERMI', 'PERMJ', 'PERMK'
        """

        five_parts = self.basic_static_property_parts(realization = realization,
                                                      share_perm_parts = share_perm_parts,
                                                      perm_k_mode = perm_k_mode,
                                                      perm_k_ratio = perm_k_ratio)
        return {
            'NTG': five_parts[0],
            'PORO': five_parts[1],
            'PERMI': five_parts[2],
            'PERMJ': five_parts[3],
            'PERMK': five_parts[4]
        }

    def basic_static_property_uuids(self,
                                    realization = None,
                                    share_perm_parts = False,
                                    perm_k_mode = None,
                                    perm_k_ratio = 1.0):
        """Returns five uuids: net to gross ratio, porosity, permeability rock I, J & K; each returned uuid may be None.

        note:
           see basic_static_property_parts() method for argument documentation

        :meta common:
        """

        five_parts = self.basic_static_property_parts(realization = realization,
                                                      share_perm_parts = share_perm_parts,
                                                      perm_k_mode = perm_k_mode,
                                                      perm_k_ratio = perm_k_ratio)
        uuid_list = []
        for part in five_parts:
            if part is None:
                uuid_list.append(None)
            else:
                uuid_list.append(rqet.uuid_in_part_name(part))
        return tuple(uuid_list)

    def basic_static_property_uuids_dict(self,
                                         realization = None,
                                         share_perm_parts = False,
                                         perm_k_mode = None,
                                         perm_k_ratio = 1.0):
        """Same as basic_static_property_uuids() method but returning a dictionary with 5 items.

        note:
           returned dictionary contains following keys: 'NTG', 'PORO', 'PERMI', 'PERMJ', 'PERMK'
        """

        five_uuids = self.basic_static_property_uuids(realization = realization,
                                                      share_perm_parts = share_perm_parts,
                                                      perm_k_mode = perm_k_mode,
                                                      perm_k_ratio = perm_k_ratio)
        return {
            'NTG': five_uuids[0],
            'PORO': five_uuids[1],
            'PERMI': five_uuids[2],
            'PERMJ': five_uuids[3],
            'PERMK': five_uuids[4]
        }

    def _part_direction(self, part):
        facet_t = self.facet_type_for_part(part)
        if facet_t is None or facet_t != 'direction':
            return None
        return self.facet_for_part(part)

    def _check_shape_list(self, indexable_element, direction, property_array, points, count):
        shape_list = self.supporting_shape(indexable_element = indexable_element, direction = direction)
        if shape_list is not None:
            if count > 1:
                shape_list.append(count)
            if points:
                shape_list.append(3)
            if property_array is not None:
                assert tuple(shape_list) == property_array.shape, \
                    f'property array shape {property_array.shape} is not the expected {tuple(shape_list)}'

    def _get_property_kind_uuid(self, property_kind_uuid, property_kind, uom, discrete):
        if property_kind_uuid is None:
            pk_parts_list = self.model.parts_list_of_type('PropertyKind')
            for part in pk_parts_list:
                if self.model.citation_title_for_part(part) == property_kind:
                    property_kind_uuid = self.model.uuid_for_part(part)
                    break
            if property_kind_uuid is None:
                # create local property kind object and fetch uuid
                lpk = PropertyKind(self.model,
                                   title = property_kind,
                                   example_uom = uom,
                                   parent_property_kind = 'discrete' if discrete else 'continuous')
                lpk.create_xml()
                property_kind_uuid = lpk.uuid
        return property_kind_uuid

    def _create_xml_property_kind(self, p_node, find_local_property_kinds, property_kind, uom, discrete,
                                  property_kind_uuid):
        p_kind_node = rqet.SubElement(p_node, ns['resqml2'] + 'PropertyKind')
        p_kind_node.text = rqet.null_xml_text
        if find_local_property_kinds and property_kind not in supported_property_kind_list:
            property_kind_uuid = self._get_property_kind_uuid(property_kind_uuid, property_kind, uom, discrete)

        if property_kind_uuid is None:
            p_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StandardPropertyKind')  # todo: local prop kind ref
            kind_node = rqet.SubElement(p_kind_node, ns['resqml2'] + 'Kind')
            kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlPropertyKind')
            kind_node.text = property_kind
        else:
            p_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'LocalPropertyKind')  # todo: local prop kind ref
            self.model.create_ref_node('LocalPropertyKind',
                                       property_kind,
                                       property_kind_uuid,
                                       content_type = 'obj_PropertyKind',
                                       root = p_kind_node)

    def _create_xml_patch_node(self, p_node, points, const_value, indexable_element, direction, p_uuid, ext_uuid):
        # create patch node
        const_count = None
        if const_value is not None:
            s_shape = self.supporting_shape(indexable_element = indexable_element, direction = direction)
            assert s_shape is not None
            const_count = np.product(np.array(s_shape, dtype = int))
        _ = self.model.create_patch(p_uuid,
                                    ext_uuid,
                                    root = p_node,
                                    hdf5_type = self.hdf5_type,
                                    xsd_type = self.xsd_type,
                                    null_value = self.null_value,
                                    const_value = const_value,
                                    const_count = const_count,
                                    points = points)

    def _get_property_type_details(self, discrete, string_lookup_uuid, points):
        if discrete:
            if string_lookup_uuid is None:
                self.d_or_c_text = 'Discrete'

            else:
                self.d_or_c_text = 'Categorical'
            self.xsd_type = 'integer'
            self.hdf5_type = 'IntegerHdf5Array'
        elif points:
            self.d_or_c_text = 'Points'
            self.xsd_type = 'double'
            self.hdf5_type = 'Point3dHdf5Array'
            self.null_value = None
        else:
            self.d_or_c_text = 'Continuous'
            self.xsd_type = 'double'
            self.hdf5_type = 'DoubleHdf5Array'
            self.null_value = None

    def _get_property_array_min_max_value(self, property_array, const_value, discrete, min_value, max_value):
        if const_value is not None:
            return _get_property_array_min_max_const(const_value, self.null_value, min_value, max_value, discrete)
        elif property_array is not None:
            return _get_property_array_min_max_array(property_array, min_value, max_value, discrete)

    def _create_xml_property_min_max(self, p_node, min_value, max_value):
        if min_value is not None:
            min_node = rqet.SubElement(p_node, ns['resqml2'] + 'MinimumValue')
            min_node.set(ns['xsi'] + 'type', ns['xsd'] + self.xsd_type)
            min_node.text = str(min_value)
        if max_value is not None:
            max_node = rqet.SubElement(p_node, ns['resqml2'] + 'MaximumValue')
            max_node.set(ns['xsi'] + 'type', ns['xsd'] + self.xsd_type)
            max_node.text = str(max_value)

    def _create_xml_lookup_node(self, p_node, string_lookup_uuid):
        sl_root = None
        if string_lookup_uuid is not None:
            sl_root = self.model.root_for_uuid(string_lookup_uuid)
            assert sl_root is not None, 'string table lookup is missing whilst importing categorical property'
            assert rqet.node_type(sl_root) == 'obj_StringTableLookup', 'referenced uuid is not for string table lookup'
            self.model.create_ref_node('Lookup',
                                       self.model.title_for_root(sl_root),
                                       string_lookup_uuid,
                                       content_type = 'obj_StringTableLookup',
                                       root = p_node)
        return sl_root

    def _create_xml_uom_node(self, p_node, uom, property_kind, min_value, max_value, facet_type, facet, title):
        if not uom:
            uom = guess_uom(property_kind, min_value, max_value, self.support, facet_type = facet_type, facet = facet)
            if not uom:
                uom = 'Euc'  # todo: put RESQML base uom for quantity class here, instead of Euc
                log.warning(f'uom set to Euc for property {title} of kind {property_kind}')
        self.model.uom_node(p_node, uom)

    def _create_xml_add_relationships(self, p_node, support_root, property_kind_uuid, related_time_series_node, sl_root,
                                      discrete, string_lookup_uuid, const_value, ext_uuid):
        if support_root is not None:
            self.model.create_reciprocal_relationship(p_node, 'destinationObject', support_root, 'sourceObject')
        if property_kind_uuid is not None:
            pk_node = self.model.root_for_uuid(property_kind_uuid)
            if pk_node is not None:
                self.model.create_reciprocal_relationship(p_node, 'destinationObject', pk_node, 'sourceObject')
        if related_time_series_node is not None:
            self.model.create_reciprocal_relationship(p_node, 'destinationObject', related_time_series_node,
                                                      'sourceObject')
        if discrete and string_lookup_uuid is not None:
            self.model.create_reciprocal_relationship(p_node, 'destinationObject', sl_root, 'sourceObject')

        if const_value is None:
            ext_node = self.model.root_for_part(
                rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False))
            self.model.create_reciprocal_relationship(p_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

    def _create_xml_realization_node(self, realization, p_node):
        if realization is not None and realization >= 0:
            ri_node = rqet.SubElement(p_node, ns['resqml2'] + 'RealizationIndex')
            ri_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
            ri_node.text = str(realization)

    def _create_xml_time_series_node(self, time_series_uuid, time_index, p_node, support_uuid, support_type,
                                     support_root):
        if time_series_uuid is None or time_index is None:
            related_time_series_node = None
        else:
            related_time_series_node = self.model.root(uuid = time_series_uuid)
            time_series = rts.any_time_series(self.model, uuid = time_series_uuid)
            time_series.create_time_index(time_index, root = p_node)

        support_title = '' if support_root is None else rqet.citation_title_for_node(support_root)
        self.model.create_supporting_representation(support_uuid = support_uuid,
                                                    root = p_node,
                                                    title = support_title,
                                                    content_type = support_type)
        return related_time_series_node

    def _create_xml_get_p_node(self, p_uuid):
        p_node = self.model.new_obj_node(self.d_or_c_text + 'Property')
        if p_uuid is None:
            p_uuid = bu.uuid_from_string(p_node.attrib['uuid'])
        else:
            p_node.attrib['uuid'] = str(p_uuid)
        return p_node, p_uuid

    def _create_xml_add_as_part(self, add_as_part, p_uuid, p_node, add_relationships, support_root, property_kind_uuid,
                                related_time_series_node, sl_root, discrete, string_lookup_uuid, const_value, ext_uuid):
        if add_as_part:
            self.model.add_part('obj_' + self.d_or_c_text + 'Property', p_uuid, p_node)
            if add_relationships:
                self._create_xml_add_relationships(p_node, support_root, property_kind_uuid, related_time_series_node,
                                                   sl_root, discrete, string_lookup_uuid, const_value, ext_uuid)

    def _set_support_and_model_from_collection(self, other, support_uuid, grid):
        if support_uuid is None and grid is not None:
            support_uuid = grid.uuid
        if support_uuid is not None and self.support_uuid is not None:
            assert bu.matching_uuids(support_uuid, self.support_uuid)

        assert other is not None
        if self.model is None:
            self.model = other.model
        else:
            assert self.model is other.model

        assert self.support_uuid is None or other.support_uuid is None or bu.matching_uuids(
            self.support_uuid, other.support_uuid)
        if self.support_uuid is None and self.number_of_parts() == 0:
            self.set_support(support_uuid = other.support_uuid, support = other.support)

    def _add_selected_part_from_other_dict(self, part, other, realization, support_uuid, uuid, continuous, categorical,
                                           count, points, indexable, property_kind, facet_type, facet, citation_title,
                                           citation_title_match_starts_with, time_series_uuid, time_index,
                                           string_lookup_uuid, ignore_clashes):

        if _check_not_none_and_not_equals(realization, other.realization_for_part, part):
            return
        if _check_not_none_and_not_uuid_match(support_uuid, other.support_uuid_for_part, part):
            return
        if _check_not_none_and_not_uuid_match(uuid, other.uuid_for_part, part):
            return
        if _check_not_none_and_not_equals(continuous, other.continuous_for_part, part):
            return
        if _check_categorical_and_lookup(categorical, other, part):
            return
        if _check_not_none_and_not_equals(count, other.count_for_part, part):
            return
        if _check_not_none_and_not_equals(points, other.points_for_part, part):
            return
        if _check_not_none_and_not_equals(indexable, other.indexable_for_part, part):
            return
        if property_kind is not None and not same_property_kind(other.property_kind_for_part(part), property_kind):
            return
        if _check_not_none_and_not_equals(facet_type, other.facet_type_for_part, part):
            return
        if _check_not_none_and_not_equals(facet, other.facet_for_part, part):
            return
        if _check_citation_title(citation_title, citation_title_match_starts_with, other, part):
            return
        if _check_not_none_and_not_uuid_match(time_series_uuid, other.time_series_uuid_for_part, part):
            return
        if _check_not_none_and_not_equals(time_index, other.time_index_for_part, part):
            return
        if _check_not_none_and_not_uuid_match(string_lookup_uuid, other.string_lookup_uuid_for_part, part):
            return
        if part in self.dict.keys():
            if ignore_clashes:
                return
            assert (False)
        self.dict[part] = other.dict[part]

    def _find_single_part(self, kind, realization):
        try:
            part = self.singleton(realization = realization, property_kind = kind)
        except Exception:
            log.error(f'problem with {kind} (more than one array present?)')
            part = None
        return part

    def _get_single_perm_ijk_parts(self, perms, share_perm_parts, perm_k_mode, perm_k_ratio, ntg_part):
        if perms.number_of_parts() == 1:
            return _get_single_perm_ijk_parts_one(perms, share_perm_parts)
        else:
            perm_i_part = _get_single_perm_ijk_for_direction(perms, 'I')
            perm_j_part = _get_single_perm_ijk_for_direction(perms, 'J')
            if perm_j_part is None and share_perm_parts:
                perm_j_part = perm_i_part
            elif perm_i_part is None and share_perm_parts:
                perm_i_part = perm_j_part
            perm_k_part = _get_single_perm_ijk_for_direction(perms, 'K')
            if perm_k_part is None:
                assert perm_k_mode in [None, 'none', 'shared', 'ratio', 'ntg', 'ntg squared']
                # note: could switch ratio mode to shared if perm_k_ratio is 1.0
                if perm_k_mode is None or perm_k_mode == 'none':
                    pass
                elif perm_k_mode == 'shared':
                    if share_perm_parts:
                        perm_k_part = perm_i_part
                elif perm_i_part is not None:
                    log.info('generating K permeability data using mode ' + str(perm_k_mode))
                    if perm_j_part is not None and perm_j_part != perm_i_part:
                        # generate root mean square of I & J permeabilities to use as horizontal perm
                        kh = np.sqrt(
                            perms.cached_part_array_ref(perm_i_part) * perms.cached_part_array_ref(perm_j_part))
                    else:  # use I permeability as horizontal perm
                        kh = perms.cached_part_array_ref(perm_i_part)
                    kv = kh * perm_k_ratio
                    if ntg_part is not None:
                        if perm_k_mode == 'ntg':
                            kv *= self.cached_part_array_ref(ntg_part)
                        elif perm_k_mode == 'ntg squared':
                            ntg = self.cached_part_array_ref(ntg_part)
                            kv *= ntg * ntg
                    kv_collection = PropertyCollection()
                    kv_collection.set_support(support_uuid = self.support_uuid, model = self.model)
                    kv_collection.add_cached_array_to_imported_list(
                        kv,
                        'derived from horizontal perm with mode ' + str(perm_k_mode),
                        'KK',
                        discrete = False,
                        uom = 'mD',
                        time_index = None,
                        null_value = None,
                        property_kind = 'permeability rock',
                        facet_type = 'direction',
                        facet = 'K',
                        realization = perms.realization_for_part(perm_i_part),
                        indexable_element = perms.indexable_for_part(perm_i_part),
                        count = 1,
                        points = False)
                    self.model.h5_release()
                    kv_collection.write_hdf5_for_imported_list()
                    kv_collection.create_xml_for_imported_list_and_add_parts_to_model()
                    self.inherit_parts_from_other_collection(kv_collection)
                    perm_k_part = kv_collection.singleton()
        return perm_i_part, perm_j_part, perm_k_part

    def _add_part_to_dict_get_realization(self, realization, xml_node):
        if realization is not None and self.realization is not None:
            assert (realization == self.realization)
        if realization is None:
            realization = self.realization
        realization_node = rqet.find_tag(xml_node,
                                         'RealizationIndex')  # optional; if present use to populate realization
        if realization_node is not None:
            realization = int(realization_node.text)
        return realization

    def _add_part_to_dict_get_type_details(self, part, continuous, xml_node):
        sl_ref_node = None
        type = self.model.type_of_part(part)
        #      log.debug('adding part ' + part + ' of type ' + type)
        assert type in [
            'obj_ContinuousProperty', 'obj_DiscreteProperty', 'obj_CategoricalProperty', 'obj_PointsProperty'
        ]
        if continuous is None:
            continuous = (type in ['obj_ContinuousProperty', 'obj_PointsProperty'])
        else:
            assert continuous == (type in ['obj_ContinuousProperty', 'obj_PointsProperty'])
        points = (type == 'obj_PointsProperty')
        string_lookup_uuid = None
        if type == 'obj_CategoricalProperty':
            sl_ref_node = rqet.find_tag(xml_node, 'Lookup')
            string_lookup_uuid = bu.uuid_from_string(rqet.find_tag_text(sl_ref_node, 'UUID'))

        return type, continuous, points, string_lookup_uuid, sl_ref_node

    def _add_part_to_dict_get_support_uuid(self, part):
        support_uuid = self.model.supporting_representation_for_part(part)
        if support_uuid is None:
            support_uuid = self.support_uuid
        elif self.support_uuid is None:
            self.set_support(support_uuid)
        elif not bu.matching_uuids(support_uuid, self.support.uuid):  # multi-support collection
            self.set_support(None)
        if isinstance(support_uuid, str):
            support_uuid = bu.uuid_from_string(support_uuid)
        return support_uuid

    def _add_part_to_dict_get_uom(self, part, continuous, xml_node, trust_uom, property_kind, minimum, maximum, facet, facet_type):
        uom = None
        if continuous:
            uom_node = rqet.find_tag(xml_node, 'UOM')
            if uom_node is not None and (trust_uom or uom_node.text not in ['', 'Euc']):
                uom = uom_node.text
            else:
                uom = guess_uom(property_kind, minimum, maximum, self.support, facet_type = facet_type, facet = facet)
        return uom

    def _normalized_part_array_get_minmax(self,trust_min_max, part, p_array, masked):
        min_value = max_value = None
        if trust_min_max:
            min_value = self.minimum_value_for_part(part)
            max_value = self.maximum_value_for_part(part)
        if min_value is None or max_value is None:
            min_value = np.nanmin(p_array)
            if masked and min_value is ma.masked:
                min_value = None
            max_value = np.nanmax(p_array)
            if masked and max_value is ma.masked:
                max_value = None
            self.override_min_max(part, min_value, max_value)  # NB: this does not modify xml
        return min_value, max_value

    def _cached_part_array_ref_get_array(self, part, dtype, model, cached_array_name):
        const_value = self.constant_value_for_part(part)
        if const_value is None:
            self._cached_part_array_ref_const_none(part, dtype, model, cached_array_name)
        else:
            self._cached_part_array_ref_const_notnone(part, const_value, cached_array_name)
        if not hasattr(self, cached_array_name):
            return None

    def _cached_part_array_ref_const_none(self, part, dtype, model, cached_array_name):
        part_node = self.node_for_part(part)
        if part_node is None:
            return None
        if self.points_for_part(part):
            first_values_node, tag, dtype = _cached_part_array_ref_get_node_points(part_node, dtype)
        else:
            first_values_node, tag, dtype = _cached_part_array_ref_get_node_values(part_node, dtype)

        h5_key_pair = model.h5_uuid_and_path_for_node(first_values_node, tag=tag)
        if h5_key_pair is None:
            return None
        model.h5_array_element(h5_key_pair,
                               index=None,
                               cache_array=True,
                               object=self,
                               array_attribute=cached_array_name,
                               dtype=dtype)

    def _cached_part_array_ref_const_notnone(self, part, const_value, cached_array_name):
        assert not self.points_for_part(part), 'constant arrays not supported for points properties'
        assert self.support is not None
        shape = self.supporting_shape(indexable_element=self.indexable_for_part(part),
                                      direction=self._part_direction(part))
        assert shape is not None
        a = np.full(shape, const_value, dtype=float if self.continuous_for_part(part) else int)
        setattr(self, cached_array_name, a)


def _get_indexable_element(indexable_element, support_type):
    if indexable_element is None:
        if support_type in [
                'obj_IjkGridRepresentation', 'obj_BlockedWellboreRepresentation', 'obj_Grid2dRepresentation',
                'obj_UnstructuredGridRepresentation'
        ]:
            indexable_element = 'cells'
        elif support_type == 'obj_WellboreFrameRepresentation':
            indexable_element = 'nodes'  # note: could be 'intervals'
        elif support_type == 'obj_GridConnectionSetRepresentation':
            indexable_element = 'faces'
        else:
            raise Exception('indexable element unknown for unsupported supporting representation object')
    return indexable_element


def _create_xml_facet_node(facet_type, facet, p_node):
    if facet_type is not None and facet is not None:
        facet_node = rqet.SubElement(p_node, ns['resqml2'] + 'Facet')
        facet_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'PropertyKindFacet')
        facet_node.text = rqet.null_xml_text
        facet_type_node = rqet.SubElement(facet_node, ns['resqml2'] + 'Facet')
        facet_type_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Facet')
        facet_type_node.text = facet_type
        facet_value_node = rqet.SubElement(facet_node, ns['resqml2'] + 'Value')
        facet_value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
        facet_value_node.text = facet


def _get_property_array_min_max_const(const_value, null_value, min_value, max_value, discrete):
    if (discrete and const_value != null_value) or (not discrete and not np.isnan(const_value)):
        if min_value is None:
            min_value = const_value
        if max_value is None:
            max_value = const_value
    return min_value, max_value


def _get_property_array_min_max_array(property_array, min_value, max_value, discrete):
    if discrete:
        if min_value is None:
            try:
                min_value = int(property_array.min())
            except Exception:
                min_value = None
                log.warning('no xml minimum value set for discrete property')
        if max_value is None:
            try:
                max_value = int(property_array.max())
            except Exception:
                max_value = None
                log.warning('no xml maximum value set for discrete property')
    else:
        if min_value is None or max_value is None:
            all_nan = np.all(np.isnan(property_array))
        if min_value is None and not all_nan:
            min_value = np.nanmin(property_array)
            if np.isnan(min_value) or min_value is ma.masked:
                min_value = None
        if max_value is None and not all_nan:
            max_value = np.nanmax(property_array)
            if np.isnan(max_value) or max_value is ma.masked:
                max_value = None

    return min_value, max_value


def _check_not_none_and_not_equals(attrib, method, part):
    return (attrib is not None and method(part) != attrib)


def _check_not_none_and_not_uuid_match(uuid, method, part):
    return (uuid is not None and not bu.matching_uuids(uuid, method(part)))


def _check_citation_title(citation_title, citation_title_match_starts_with, other, part):
    if citation_title is not None:
        if citation_title_match_starts_with:
            if not other.citation_title_for_part(part).startswith():
                return True
        else:
            if other.citation_title_for_part(part) != citation_title:
                return True
    return False


def _check_categorical_and_lookup(categorical, other, part):
    if categorical is not None:
        if categorical:
            if other.continuous_for_part(part) or (other.string_lookup_uuid_for_part(part) is None):
                return True
        else:
            if not (other.continuous_for_part(part) or (other.string_lookup_uuid_for_part(part) is None)):
                return True
    return False


def _get_single_perm_ijk_parts_one(perms, share_perm_parts):
    perm_i_part = perms.singleton()
    if share_perm_parts:
        perm_j_part = perm_k_part = perm_i_part
    elif perms.facet_type_for_part(perm_i_part) == 'direction':
        direction = perms.facet_for_part(perm_i_part)
        if direction == 'J':
            perm_j_part = perm_i_part
            perm_i_part = None
        elif direction == 'IJ':
            perm_j_part = perm_i_part
        elif direction == 'K':
            perm_k_part = perm_i_part
            perm_i_part = None
        elif direction == 'IJK':
            perm_j_part = perm_k_part = perm_i_part
    return perm_i_part, perm_j_part, perm_k_part


def _get_single_perm_ijk_for_direction(perms, direction):
    facet_options, title_options = _get_facet_title_options_for_direction(direction)

    try:
        part = None
        for facet_op in facet_options:
            if not part:
                part = perms.singleton(facet_type = 'direction', facet = facet_op)
        if not part:
            for title in title_options:
                if not part:
                    part = perms.singleton(citation_title = title)
        if not part:
            log.error(f'unable to discern which rock permeability to use for {direction} direction')
    except Exception:
        log.error(f'problem with permeability data (more than one {direction} direction array present?)')
        part = None
    return part


def _get_facet_title_options_for_direction(direction):
    if direction == 'I':
        facet_options = ['I', 'IJ', 'IJK']
        title_options = ['KI', 'PERMI', 'KX', 'PERMX']
    elif direction == 'J':
        facet_options = ['J', 'IJ', 'IJK']
        title_options = ['KJ', 'PERMJ', 'KY', 'PERMY']
    else:
        facet_options = ['K', 'IJK']
        title_options = ['KK', 'PERMK', 'KZ', 'PERMZ']
    return facet_options, title_options

def _add_part_to_dict_get_count_and_indexable(xml_node):
    count_node = rqet.find_tag(xml_node, 'Count')
    assert count_node is not None
    count = int(count_node.text)

    indexable_node = rqet.find_tag(xml_node, 'IndexableElement')
    assert indexable_node is not None
    indexable = indexable_node.text

    return count, indexable

def _add_part_to_dict_get_property_kind(xml_node, citation_title):
    (p_kind_from_keyword, facet_type, facet) = property_kind_and_facet_from_keyword(citation_title)
    prop_kind_node = rqet.find_tag(xml_node, 'PropertyKind')
    assert (prop_kind_node is not None)
    kind_node = rqet.find_tag(prop_kind_node, 'Kind')
    property_kind_uuid = None  # only used for bespoke (local) property kinds
    if kind_node is not None:
        property_kind = kind_node.text  # could check for consistency with that derived from citation title
        lpk_node = None
    else:
        lpk_node = rqet.find_tag(prop_kind_node, 'LocalPropertyKind')
        if lpk_node is not None:
            property_kind = rqet.find_tag_text(lpk_node, 'Title')
            property_kind_uuid = rqet.find_tag_text(lpk_node, 'UUID')
    assert property_kind is not None and len(property_kind) > 0
    if (p_kind_from_keyword and p_kind_from_keyword != property_kind and
            (p_kind_from_keyword not in ['cell length', 'length', 'thickness'] or
             property_kind not in ['cell length', 'length', 'thickness'])):
        log.warning(
            f'property kind {property_kind} not the expected {p_kind_from_keyword} for keyword {citation_title}')
    return property_kind, property_kind_uuid, lpk_node


def _add_part_to_dict_get_facet(xml_node):
    facet_type = None
    facet = None
    facet_node = rqet.find_tag(xml_node, 'Facet')  # todo: handle more than one facet for a property
    if facet_node is not None:
        facet_type = rqet.find_tag(facet_node, 'Facet').text
        facet = rqet.find_tag(facet_node, 'Value').text
        if facet_type is not None and facet_type == '':
            facet_type = None
        if facet is not None and facet == '':
            facet = None
    return facet_type, facet


def _add_part_to_dict_get_timeseries(xml_node):
    time_series_uuid = None
    time_index = None
    time_node = rqet.find_tag(xml_node, 'TimeIndex')
    if time_node is not None:
        time_index = int(rqet.find_tag(time_node, 'Index').text)
        time_series_uuid = bu.uuid_from_string(rqet.find_tag(rqet.find_tag(time_node, 'TimeSeries'), 'UUID').text)

    return time_series_uuid, time_index


def _add_part_to_dict_get_minmax(xml_node):
    minimum = None
    min_node = rqet.find_tag(xml_node, 'MinimumValue')
    if min_node is not None:
        minimum = min_node.text  # NB: left as text
    maximum = None
    max_node = rqet.find_tag(xml_node, 'MaximumValue')
    if max_node is not None:
        maximum = max_node.text  # NB: left as text

    return minimum, maximum


def _add_part_to_dict_get_null_constvalue_points(xml_node, continuous, points):
    null_value = None
    if not continuous:
        null_value = rqet.find_nested_tags_int(xml_node, ['PatchOfValues', 'Values', 'NullValue'])
    const_value = None
    if points:
        values_node = rqet.find_nested_tags(xml_node, ['PatchOfPoints', 'Points'])
    else:
        values_node = rqet.find_nested_tags(xml_node, ['PatchOfValues', 'Values'])
    values_type = rqet.node_type(values_node)
    assert values_type is not None
    if values_type.endswith('ConstantArray'):
        if continuous:
            const_value = rqet.find_tag_float(values_node, 'Value')
        elif values_type.startswith('Bool'):
            const_value = rqet.find_tag_bool(values_node, 'Value')
        else:
            const_value = rqet.find_tag_int(values_node, 'Value')

    return null_value, const_value


def _normalized_part_array_apply_discrete_cycle(discrete_cycle, p_array, min_value, max_value):
    if 'int' in str(
            p_array.dtype) and discrete_cycle is not None:  # could use continuous flag in metadata instead of dtype
        p_array = p_array % discrete_cycle
        min_value = 0
        max_value = discrete_cycle - 1
    elif str(p_array.dtype).startswith('bool'):
        min_value = int(min_value)
        max_value = int(max_value)
    return min_value, max_value, p_array


def _normalized_part_array_nan_if_masked(min_value, max_value, masked):
    min_value = float(min_value)  # will return np.ma.masked if all values are masked out
    if masked and min_value is ma.masked:
        min_value = np.nan
    max_value = float(max_value)
    if masked and max_value is ma.masked:
        max_value = np.nan
    return min_value, max_value

def _normalized_part_array_use_logarithm(min_value, n_prop, masked):
    if min_value <= 0.0:
        n_prop[:] = np.where(n_prop < 0.0001, 0.0001, n_prop)
    n_prop = np.log10(n_prop)
    min_value = np.nanmin(n_prop)
    max_value = np.nanmax(n_prop)
    if masked:
        if min_value is ma.masked:
            min_value = np.nan
        if max_value is ma.masked:
            max_value = np.nan
    return min_value, max_value


def _normalized_part_array_fix_zero_at(min_value, max_value, n_prop, fix_zero_at):
    if fix_zero_at <= 0.0:
        if min_value < 0.0:
            n_prop[:] = np.where(n_prop < 0.0, 0.0, n_prop)
        min_value = 0.0
    elif fix_zero_at >= 1.0:
        if max_value > 0.0:
            n_prop[:] = np.where(n_prop > 0.0, 0.0, n_prop)
        max_value = 0.0
    else:
        upper_scaling = max_value / (1.0 - fix_zero_at)
        lower_scaling = -min_value / fix_zero_at
        if upper_scaling >= lower_scaling:
            min_value = -upper_scaling * fix_zero_at
            n_prop[:] = np.where(n_prop < min_value, min_value, n_prop)
        else:
            max_value = lower_scaling * (1.0 - fix_zero_at)
            n_prop[:] = np.where(n_prop > max_value, max_value, n_prop)
    return min_value, max_value, n_prop


def _supporting_shape_grid(support, indexable_element, direction):
    if indexable_element is None or indexable_element == 'cells':
        shape_list = [support.nk, support.nj, support.ni]
    elif indexable_element == 'columns':
        shape_list = [support.nj, support.ni]
    elif indexable_element == 'layers':
        shape_list = [support.nk]
    elif indexable_element == 'faces':
        assert direction is not None and direction.upper() in 'IJK'
        axis = 'KJI'.index(direction.upper())
        shape_list = [support.nk, support.nj, support.ni]
        shape_list[axis] += 1  # note: properties for grid faces include outer faces
    elif indexable_element == 'column edges':
        shape_list = [(support.nj * (support.ni + 1)) + ((support.nj + 1) * support.ni)
                      ]  # I edges first; include outer edges
    elif indexable_element == 'edges per column':
        shape_list = [support.nj, support.ni, 4]  # assume I-, J+, I+, J- ordering
    elif indexable_element == 'faces per cell':
        shape_list = [support.nk, support.nj, support.ni, 6]  # assume K-, K+, J-, I+, J+, I- ordering
        # TODO: resolve ordering of edges and make consistent with maps code (edges per column) and fault module (gcs faces)
    elif indexable_element == 'nodes per cell':
        shape_list = [support.nk, support.nj, support.ni, 2, 2,
                      2]  # kp, jp, ip within each cell; todo: check RESQML shaping
    elif indexable_element == 'nodes':
        assert not support.k_gaps, 'indexable element of nodes not currently supported for grids with K gaps'
        if support.has_split_coordinate_lines:
            pillar_count = (support.nj + 1) * (support.ni + 1) + support.split_pillars_count
            shape_list = [support.nk + 1, pillar_count]
        else:
            shape_list = [support.nk + 1, support.nj + 1, support.ni + 1]
    return shape_list


def _supporting_shape_wellboreframe(support, indexable_element):
    if indexable_element is None or indexable_element == 'nodes':
        shape_list = [support.node_count]
    elif indexable_element == 'intervals':
        shape_list = [support.node_count - 1]
    return shape_list


def _supporting_shape_blockedwell(support, indexable_element):
    if indexable_element is None or indexable_element == 'intervals':
        shape_list = [support.node_count - 1]  # all intervals, including unblocked
    elif indexable_element == 'nodes':
        shape_list = [support.node_count]
    elif indexable_element == 'cells':
        shape_list = [support.cell_count]  # ie. blocked intervals only
    return shape_list


def _supporting_shape_mesh(support, indexable_element):
    if indexable_element is None or indexable_element == 'cells' or indexable_element == 'columns':
        shape_list = [support.nj - 1, support.ni - 1]
    elif indexable_element == 'nodes':
        shape_list = [support.nj, support.ni]
    return shape_list


def _supporting_shape_gridconnectionset(support, indexable_element):
    if indexable_element is None or indexable_element == 'faces':
        shape_list = [support.count]
    return shape_list


def _supporting_shape_other(support, indexable_element):
    if indexable_element is None or indexable_element == 'cells':
        shape_list = [support.cell_count]
    elif indexable_element == 'faces per cell':
        support.cache_all_geometry_arrays()
        shape_list = [len(support.faces_per_cell)]
    return shape_list, support


def _cached_part_array_ref_get_node_points(part_node, dtype):
    patch_list = rqet.list_of_tag(part_node, 'PatchOfPoints')
    assert len(patch_list) == 1  # todo: handle more than one patch of points
    first_values_node = rqet.find_tag(patch_list[0], 'Points')
    if first_values_node is None:
        return None  # could treat as fatal error
    if dtype is None:
        dtype = 'float'
    else:
        assert dtype in ['float', float, np.float32, np.float64]
    tag = 'Coordinates'
    return first_values_node, tag, dtype


def _cached_part_array_ref_get_node_values(part_node, dtype):
    patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
    assert len(patch_list) == 1  # todo: handle more than one patch of values
    first_values_node = rqet.find_tag(patch_list[0], 'Values')
    if first_values_node is None:
        return None  # could treat as fatal error
    if dtype is None:
        array_type = rqet.node_type(first_values_node)
        assert array_type is not None
        if array_type == 'DoubleHdf5Array':
            dtype = 'float'
        elif array_type == 'IntegerHdf5Array':
            dtype = 'int'
        elif array_type == 'BooleanHdf5Array':
            dtype = 'bool'
        else:
            raise ValueError('array type not catered for: ' + str(array_type))
    tag = 'Values'
    return first_values_node, tag, dtype