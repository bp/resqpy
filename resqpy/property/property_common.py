"""Module containing common methods for properties"""

# Nexus is a trademark of Halliburton

import logging

log = logging.getLogger(__name__)

import warnings
import numpy as np

import resqpy.property as rqp
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.weights_and_measures as bwam

# the following resqml property kinds and facet types are 'known about' by this module in relation to nexus
# other property kinds should be handled okay but without any special treatment
# see property_kind_and_facet_from_keyword() for simulator keyword to property kind and facet mapping

supported_property_kind_list = [
    'continuous', 'discrete', 'categorical', 'code', 'index', 'depth', 'rock volume', 'pore volume', 'volume',
    'thickness', 'length', 'cell length', 'area', 'net to gross ratio', 'porosity', 'permeability thickness',
    'permeability length', 'permeability rock', 'rock permeability', 'fluid volume', 'transmissibility', 'pressure',
    'saturation', 'solution gas-oil ratio', 'vapor oil-gas ratio', 'property multiplier', 'thermodynamic temperature'
]

supported_local_property_kind_list = [
    'active', 'transmissibility multiplier', 'fault transmissibility', 'mat transmissibility'
]

supported_facet_type_list = ['direction', 'netgross', 'what']

# current implementation limits a property to having at most one facet; resqml standard allows for many
# if a property kind does not appear in the following dictionary, facet_type and facet are expected to be None
# mapping from property kind to (facet_type, list of possible facet values), only applicable when indexable element is 'cells'
# use of the following in code is DEPRECATED; it remains here to give a hint of facet types and facets in use

expected_facet_type_dict = {
    'depth': ('what', ['cell centre', 'cell top']),  # made up
    'rock volume': ('netgross', ['net', 'gross']),  # resqml standard
    'thickness': ('netgross', ['net', 'gross']),
    'length': ('direction', ['X', 'Y', 'Z']),  # facet values made up
    'cell length': ('direction', ['I', 'J', 'K']),  # facet values made up
    'permeability rock': ('direction', ['I', 'J', 'K']),  # todo: allow IJ and IJK
    'rock permeability': ('direction', ['I', 'J', 'K']),  # todo: allow IJ and IJK
    'transmissibility': ('direction', ['I', 'J', 'K']),
    'transmissibility multiplier': (
        'direction',  # local property kind
        ['I', 'J', 'K']),
    'fault transmissibility': ('direction', ['I', 'J']),  # local property kind; K faces also permitted
    'mat transmissibility': ('direction', ['K']),  # local property kind; I & J faces also permitted
    'saturation': (
        'what',
        [
            'water',
            'oil',
            'gas',  # made up but probably good
            'water minimum',
            'gas minimum',
            'oil minimum',  # made up for end-points
            'water residual',
            'gas residual',
            'oil residual',
            'water residual to oil',
            'gas residual to oil'
        ]),
    'fluid volume': (
        'what',
        [
            'water',
            'oil',
            'gas',  # made up but probably good
            'water (mobile)',
            'oil (mobile)',
            'gas (mobile)'
        ]),  # made up
    'property multiplier': (
        'what',
        [
            'rock volume',  # made up; todo: add rock permeability?
            'pore volume',
            'transmissibility'
        ]),
    'code': ('what', ['inactive', 'active']),  # NB: user defined keywords also accepted
    'index': ('what', ['uid', 'pixel map'])
}


def same_property_kind(pk_a, pk_b):
    """Returns True if the two property kinds are the same, or pseudonyms."""

    if pk_a is None or pk_b is None:
        return False
    if pk_a == pk_b:
        return True
    if pk_a in ['permeability rock', 'rock permeability'] and pk_b in ['permeability rock', 'rock permeability']:
        return True
    return False


def property_kind_and_facet_from_keyword(keyword):
    """If keyword is recognised, returns equivalent resqml PropertyKind and Facet info.

    argument:
       keyword (string): Nexus grid property keyword

    returns:
       (property_kind, facet_type, facet) as defined in resqml standard; some or all may be None

    note:
       this function may now return the local property kind 'transmissibility multiplier'; calling code must ensure that
       the local property kind object is created if not already present
    """
    # note: this code doesn't cater for a property having more than one facet, eg. direction and phase
    # note: 'what' facet_type for made-up uses might be better changed to the generic 'qualifier' facet type

    property_kind = None
    facet_type = None
    facet = None
    lk = keyword.lower()

    if lk in ['bv', 'brv', 'pv', 'pvr', 'porv', 'netv', 'nrv']:
        property_kind, facet_type, facet = _pkf_from_keyword_rock_volume(lk)
    elif lk in ['ntg', 'netgrs']:
        property_kind = 'net to gross ratio'  # net-to-gross
    elif lk in ['por', 'poro', 'porosity']:
        property_kind = 'porosity'  # porosity
    elif lk == 'kh':
        property_kind = 'permeability thickness'  # K.H (not horizontal permeability)
    elif lk in ['p', 'pressure']:
        property_kind = 'pressure'  # pressure; todo: phase pressures
    elif lk == 'rs':
        property_kind = 'solution gas-oil ratio'
    elif lk == 'rv':
        property_kind = 'vapor oil-gas ratio'
    elif lk in ['temp', 'temperature']:
        property_kind = 'thermodynamic temperature'
    elif lk in ['sw', 'so', 'sg', 'satw', 'sato', 'satg', 'soil']:  # saturations
        property_kind, facet_type, facet = _pkf_from_keyword_saturation(lk)
    elif lk in ['swl', 'swr', 'sgl', 'sgr', 'swro', 'sgro', 'sor', 'swu', 'sgu']:  # nexus saturation end points
        property_kind, facet_type, facet = _pkf_from_keyword_saturation_end(lk)
    elif lk in ['dzc', 'dzn', 'dz', 'dznet']:
        property_kind, facet_type, facet = _pkf_from_keyword_thickness(lk)
    elif lk in ['mdep', 'depth', 'tops', 'mids']:
        property_kind, facet_type, facet = _pkf_from_keyword_depth(lk)
    elif lk == 'uid':
        property_kind = 'index'
        facet_type = 'what'
        facet = 'uid'
    elif lk in ['wip', 'oip', 'gip', 'mobw', 'mobo', 'mobg', 'ocip']:  # todo: check these, especially ocip
        property_kind, facet_type, facet = _pkf_from_keyword_fluid_volume(lk)
    elif lk in ['tmx', 'tmy', 'tmz', 'tmflx', 'tmfly', 'tmflz', 'multx', 'multy', 'multz']:
        property_kind, facet_type, facet = _pkf_from_keyword_transmissibility_multiplier(lk)
    elif lk in ['multbv', 'multpv']:
        property_kind, facet_type, facet = _pkf_from_keyword_property_multiplier(lk)
    elif lk in ['dad', 'kid', 'unpack', 'deadcell', 'inactive']:
        property_kind, facet_type, facet = _pkf_from_keyword_inactive(lk)
    elif len(lk) >= 2 and lk[0] == 'd' and lk[1] in 'xyz':
        property_kind, facet_type, facet = _pkf_from_keyword_length(lk)
    elif lk[:4] == 'perm' or (len(lk) == 2 and lk[0] == 'k'):  # permeability
        property_kind = 'permeability rock'
        (facet_type, facet) = _facet_info_for_dir_ch(lk[-1])
    elif lk[:5] == 'trans' or (len(lk) == 2 and lk[0] == 't'):  # transmissibility (for unit viscosity)
        if 'mult' in lk:
            property_kind = 'transmissibility multiplier'
        else:
            property_kind = 'transmissibility'
        if lk != 'transmissibility':
            (facet_type, facet) = _facet_info_for_dir_ch(lk[-1])
    # elif lk == 'sal':    # todo: salinity; local property kind needed; various units possible in Nexus
    elif lk == 'livecell' or lk.startswith('act'):
        property_kind = 'active'  # local property kind, see RESQML (2.0.1) usage guide, section 11.17
    elif lk[0] == 'i' or lk.startswith('reg') or lk.startswith('creg'):
        property_kind = 'region initialization'  # local property kind, see RESQML (2.0.1) usage guide, section 11.18
        if lk[0] == 'i':
            facet_type = 'what'
            facet = lk[1:].lower()
    return property_kind, facet_type, facet


def _pkf_from_keyword_length(lk):
    if lk in ['dxc', 'dyc', 'dx', 'dy']:
        property_kind = 'cell length'
        (facet_type, facet) = _facet_info_for_dir_ch(lk[1])
    else:
        property_kind = 'length'
        facet_type = 'direction'
        facet = lk[1].upper()  # keep as 'X', 'Y' or 'Z'
    return property_kind, facet_type, facet


def _pkf_from_keyword_rock_volume(lk):
    if lk in ['bv', 'brv']:  # bulk rock volume
        return 'rock volume', 'netgross', 'gross'  # todo: check this is the facet in xsd
    elif lk in ['pv', 'pvr', 'porv']:
        return 'pore volume', None, None
    elif lk in ['netv', 'nrv']:  # net volume
        return 'rock volume', 'netgross', 'net'
    else:
        return None, None, None  # should never come to this


def _pkf_from_keyword_fluid_volume(lk):
    property_kind = 'fluid volume'
    facet_type = 'what'  # todo: check use of 'what' for phase
    facet = ''
    if lk in ['wip', 'mobw']:
        facet = 'water'  # todo: add another facet indicating mobile volume
    elif lk in ['oip', 'mobo']:
        facet = 'oil'
    elif lk in ['gip', 'mobg']:
        facet = 'gas'
    elif lk == 'ocip':
        facet = 'oil condensate'  # todo: this seems unlikely: check
    if lk[:3] == 'mob':
        facet += ' (mobile)'
    return property_kind, facet_type, facet


def _facet_info_for_dir_ch(dir_ch):
    facet_type = None
    facet = None
    if dir_ch in ['i', 'j', 'k', 'x', 'y', 'z']:
        facet_type = 'direction'
        if dir_ch in ['i', 'x']:
            facet = 'I'
        elif dir_ch in ['j', 'y']:
            facet = 'J'
        else:
            facet = 'K'
        # NB: resqml also allows for combinations, eg. IJ, IJK
    return facet_type, facet


def _pkf_from_keyword_property_multiplier(lk):
    property_kind = 'property multiplier'
    facet_type = 'what'  # here 'what' facet indicates property affected
    if lk == 'multbv':
        facet = 'rock volume'  # NB: making this up as I go along
    elif lk == 'multpv':
        facet = 'pore volume'
    return property_kind, facet_type, facet


def _pkf_from_keyword_inactive(lk):
    property_kind = 'code'
    facet_type = 'what'
    # todo: kid can only be used as an inactive cell indication for the root grid
    if lk in ['kid', 'deadcell', 'inactive']:
        facet = 'inactive'  # standize on 'inactive' to indicate use as mask
    else:
        facet = lk  # note: use deadcell or unpack for inactive, if nothing better?
    return property_kind, facet_type, facet


def _pkf_from_keyword_saturation_end(lk):
    property_kind = 'saturation'
    facet_type = 'what'  # note: use of 'what' for phase is a guess
    facet = ''
    if lk[1] == 'w':
        facet = 'water'
    elif lk[1] == 'g':
        facet = 'gas'
    elif lk[1] == 'o':
        facet = 'oil'
    if lk[-1] == 'l':
        facet += ' minimum'
    elif lk[-1] == 'u':
        facet += ' maximum'
    elif lk[2:] == 'ro':
        facet += ' residual to oil'
    elif lk[-1] == 'r':
        facet += ' residual'
    else:
        assert False, 'problem deciphering saturation end point keyword: ' + lk
    return property_kind, facet_type, facet


def _pkf_from_keyword_saturation(lk):
    property_kind = 'saturation'
    facet_type = 'what'  # todo: check use of 'what' for phase
    if lk in ['sw', 'satw', 'swat']:
        facet = 'water'
    elif lk in ['so', 'sato', 'soil']:
        facet = 'oil'
    elif lk in ['sg', 'satg', 'sgas']:
        facet = 'gas'
    return property_kind, facet_type, facet


def _pkf_from_keyword_thickness(lk):
    property_kind = 'thickness'  # or should these keywords use cell length in K direction?
    facet_type = 'netgross'
    if lk.startswith('dzn'):
        facet = 'net'
    else:
        facet = 'gross'
    return property_kind, facet_type, facet


def _pkf_from_keyword_depth(lk):
    property_kind = 'depth'  # depth (nexus) and tops mean top depth
    facet_type = 'what'  # this might need to be something different
    if lk in ['mdep', 'mids']:
        facet = 'cell centre'
    else:
        facet = 'cell top'
    return property_kind, facet_type, facet


def _pkf_from_keyword_transmissibility_multiplier(lk):
    property_kind = 'transmissibility multiplier'  # NB: resqpy local property kind
    facet_type = 'direction'
    _, facet = _facet_info_for_dir_ch(lk[-1])
    return property_kind, facet_type, facet


def infer_property_kind(name, unit):
    """Guess a valid property kind."""

    # Currently unit is ignored

    valid_kinds = bwam.valid_property_kinds()

    if name in valid_kinds:
        kind = name
    else:
        # TODO: use an appropriate default
        kind = 'Unknown'

    # TODO: determine facet_type and facet somehow
    facet_type = None
    facet = None

    return kind, facet_type, facet


def _crs_m_or_ft(crs_node):  # NB. models not-so-rarely use metres for xy and feet for z
    if crs_node is None:
        return None
    xy_units = rqet.find_tag(crs_node, 'ProjectedUom').text.lower()
    z_units = rqet.find_tag(crs_node, 'VerticalUom').text.lower()
    if xy_units == 'm' and z_units == 'm':
        return 'm'
    if xy_units == 'ft' and z_units == 'ft':
        return 'ft'
    if xy_units == 'cm' and z_units == 'cm':
        return 'cm'
    return None


def guess_uom(property_kind, minimum, maximum, support, facet_type = None, facet = None):
    """Returns a guess at the units of measure for the given kind of property.

    arguments:
       property_kind (string): a valid resqml property kind, lowercase
       minimum: the minimum value in the data for which the units are being guessed
       maximum: the maximum value in the data for which the units are being guessed
       support: the grid.Grid or well.WellboreFrame object which the property data relates to
       facet_type (string, optional): a valid resqml facet type, lowercase, one of:
               'direction', 'what', 'netgross', 'qualifier', 'conditions', 'statistics'
       facet: (string, present if facet_type is present): the value relating to the facet_type,
                eg. 'I' for direction, or 'oil' for 'what'

    returns:
       a valid resqml unit of measure (uom) for the property_kind, or None

    notes:
       this function is tailored towards Nexus unit systems;
       the resqml standard allows a property to have any number of facets; however,
       this module currently only supports zero or one facet per property
    """
    crs_node, from_crs = _guess_uom_get_crs_info(support)

    if property_kind in ['rock volume', 'pore volume', 'volume', 'fluid volume']:
        return _guess_uom_volume(property_kind, from_crs, facet_type, facet)
    if property_kind == 'depth':
        return _guess_uom_depth(crs_node)
    if property_kind == 'cell length':  # todo: pass in direction facet to pick up xy_units or z_units
        return from_crs
    if property_kind in ['net to gross ratio', 'porosity', 'saturation']:
        return _guess_uom_ntg_por_sat(maximum, from_crs)
    if property_kind == 'permeability rock' or property_kind == 'rock permeability':
        return 'mD'
    if property_kind in ['permeability thickness', 'permeability length']:
        return _guess_uom_permeability_thickness(property_kind, crs_node, from_crs)
    if property_kind.endswith('transmissibility'):
        return _guess_uom_transmissibility(from_crs)
    if property_kind == 'pressure':
        return _guess_uom_pressure(from_crs, maximum)
    if property_kind in ['solution gas-oil ratio', 'vapor oil-gas ratio']:
        return _guess_uom_gor_ogr(property_kind, from_crs)
    if property_kind.endswith('multiplier'):
        return 'Euc'
    # todo: 'degC' or 'degF' for thermodynamic temperature
    return None


def _guess_uom_get_crs_info(support):
    if support is None or not hasattr(support, 'extract_crs_uuid'):
        crs_node = None
    else:
        crs_node = support.model.root_for_uuid(support.extract_crs_uuid())
    return crs_node, _crs_m_or_ft(crs_node)


def _guess_uom_depth(crs_node):
    if crs_node is None:
        return None
    return rqet.find_tag(crs_node, 'VerticalUom').text.lower()


def _guess_uom_volume(property_kind, from_crs, facet_type, facet):
    if property_kind == 'fluid volume':
        if from_crs == 'm':
            return 'm3'
        if from_crs == 'ft':
            if facet_type == 'what' and facet == 'gas':
                return '1000 ft3'  # todo: check units output by nexus for GIP
            else:
                return 'bbl'  # todo: check whether nexus uses 10^3 or 10^6 units
        if from_crs == 'cm':
            return 'cm3'
        return None
    if from_crs is None:
        return None
    if from_crs == 'ft' and property_kind == 'pore volume':
        return 'bbl'  # seems to be Nexus 'ENGLISH' uom for pv out
    return from_crs + '3'  # ie. m3 or ft3 or cm3


def _guess_uom_permeability_thickness(property_kind, crs_node, from_crs):
    z_units = rqet.find_tag(crs_node, 'VerticalUom').text.lower()
    if z_units in ['m', 'cm']:  # note: mD.cm is not a valid RESQML uom
        return 'mD.m'
    if z_units == 'ft':
        return 'mD.ft'
    return None


def _guess_uom_ntg_por_sat(maximum, from_crs):
    if maximum is not None and str(maximum) != 'unknown':
        max_real = float(maximum)
        if max_real > 1.0 and max_real <= 100.0:
            return '%'
        if max_real < 0.0 or max_real > 1.0:
            return None
    if from_crs == 'm':
        return 'm3/m3'
    if from_crs == 'ft':
        return 'ft3/ft3'
    if from_crs == 'cm':
        return 'cm3/cm3'
    return 'Euc'


def _guess_uom_transmissibility(from_crs):
    # note: RESQML QuantityClass only includes a unit-viscosity VolumePerTimePerPressureUom
    if from_crs == 'm':
        return 'm3.cP/(kPa.d)'  # NB: might actually be m3/(psi.d) or m3/(bar.d)
    if from_crs == 'ft':
        return 'bbl.cP/(psi.d)'  # gamble on barrels per day per psi; could be ft3/(psi.d)
    if from_crs == 'cm':
        return 'cm3.cP/(psi.d)'  # complete guess, though transmissibility probably not used in lab scale models
    return None


def _guess_uom_pressure(from_crs, maximum):
    if from_crs == 'm':
        return 'kPa'  # NB: might actually be psi or bar
    if from_crs in ['ft', 'cm']:  # note: Nexus uses psi for lab pressure units
        return 'psi'
    if maximum is not None:
        max_real = float(maximum)
        if max_real == 0.0:
            return None
        if max_real > 10000.0:
            return 'kPa'
        if max_real < 500.0:
            return 'bar'
        if max_real < 5000.0:
            return 'psi'
    return None


def _guess_uom_gor_ogr(property_kind, from_crs):
    if from_crs == 'm':
        return 'm3/m3'  # NB: might actually be psi or bar
    if from_crs == 'ft':
        if 'solution' in property_kind:
            return '1000 ft3/bbl'
        else:
            return '0.001 bbl/ft3'
    if from_crs == 'cm':
        return 'cm3/cm3'
    return None


def selective_version_of_collection(collection,
                                    realization = None,
                                    support_uuid = None,
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
                                    title = None,
                                    title_mode = None,
                                    related_uuid = None,
                                    const_value = None):
    """Returns a new PropertyCollection with those parts which match all arguments that are not None.

    arguments:
       collection (PropertyCollection): an existing collection from which a subset will be returned as a new object
       realization (int, optional): realization number to filter on
       support_uuid (UUID or str, optional): UUID of supporting representation, to filter on
       uuid (UUID or str, optional): a property uuid to select a singleton property from the collection
       continuous (bool, optional): if True, continuous properties are selected; if False, discrete and categorical
       points (bool, optional): if True, points properties are selected; if False, they are excluded
       count (int, optional): a count value to filter on
       indexable (str, optional): indexable elements flavour to filter on
       property_kind (str, optional): property kind to filter on (commonly used)
       facet_type (str, optional): a facet_type to filter on (must be present for property to be selected)
       facet (str, optional): a facet value to filter on (if used, facet_type should be specified)
       citation_title (str, optional): citation to title to filter on; see also title_mode argument
       time_series_uuid (UUID or str, optional): UUID of a TimeSeries to filter on
       time_index (int, optional): a time series time index to filter on
       uom (str, optional): unit of measure to filter on
       string_lookup_uuid (UUID or str, optional): UUID of a string lookup table to filter on
       categorical (bool, optional): if True, only categorical properties are selected; if False they are excluded
       title (str, optional): synonymous with citation_title argument
       title_mode (str, optional): if present, one of 'is', 'starts', 'ends', 'contains', 'is not',
           'does not start', 'does not end', 'does not contain'; None is the same as 'is'; will default to 'is'
           if not specified and title or citation_title argument is present
       related_uuid (UUID or str, optional): only properties with direct relationship to this uuid are selected
       const_value (float or int, optional): only properties flagged as constant, with given value, are selected

    returns:
       a new PropertyCollection containing those properties which match the filter parameters that are not None

    notes:
       the existing collection might often be the 'main' collection holding all the properties
       for a supporting representation (eg. grid, blocked well or wellbore frame);
       for each of the filtering arguments: if None, then all members of collection pass this filter;
       if not None then only those members with the given value pass this filter;
       special values: '*' any non-None value passes; 'none' only None passes;
       citation_title (or its synonym title) uses string filtering in association with title_mode argument;
       finally, the filters for all the attributes must be passed for a given member
       to be included in the returned collection; title is a synonym for the citation_title argument;
       related_uuid will pass if any relationship exists ('hard' or 'soft');
       the categorical boolean argument can be used to select only categorical (or non-categorical) properties,
       even though this is not explicitly held as a field in the internal dictionary
    """

    assert collection is not None
    view = rqp.PropertyCollection()
    if support_uuid is not None:
        view.set_support(support_uuid = support_uuid, model = collection.model)
    if realization is not None:
        view.set_realization(realization)
    if citation_title is None:
        citation_title = title
    view.inherit_parts_selectively_from_other_collection(collection,
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
                                                         citation_title_match_mode = title_mode,
                                                         time_series_uuid = time_series_uuid,
                                                         time_index = time_index,
                                                         uom = uom,
                                                         string_lookup_uuid = string_lookup_uuid,
                                                         categorical = categorical,
                                                         related_uuid = related_uuid,
                                                         const_value = const_value)
    return view


def property_over_time_series_from_collection(collection, example_part):
    """Returns a new PropertyCollection with parts like the example part, over all indices in its time series.

    arguments:
       collection: an existing PropertyCollection from which a subset will be returned as a new object;
                   the existing collection might often be the 'main' collection holding all the properties
                   for a grid
       example_part (string): the part name of an example member of collection (which has an associated time_series)

    returns:
       a new PropertyCollection containing those memners of collection which have the same property kind
       (and facet, if any) as the example part and which have the same associated time series
    """

    assert collection is not None and example_part is not None
    assert collection.part_in_collection(example_part)
    view = rqp.PropertyCollection()
    if collection.support_uuid is not None:
        view.set_support(support_uuid = collection.support_uuid, model = collection.model)
    if collection.realization is not None:
        view.set_realization(collection.realization)
    view.inherit_similar_parts_for_time_series_from_other_collection(collection, example_part)
    return view


def property_collection_for_keyword(collection, keyword):
    """Returns a new PropertyCollection with parts that match the property kind and facet deduced for the keyword.

    arguments:
       collection: an existing PropertyCollection from which a subset will be returned as a new object;
                   the existing collection might often be the 'main' collection holding all the properties
                   for a supporting representation (grid or wellbore frame)
       keyword (string): a simulator keyword for which the property kind (and facet, if any) can be deduced

    returns:
       a new PropertyCollection containing those memners of collection which have the property kind
       (and facet, if any) as that deduced for the keyword

    note:
       this function is particularly relevant to grid property collections for simulation models;
       the handling of simulator keywords in this module is based on the main grid property keywords
       for Nexus; if the resqml dataset was generated from simulator data using this module then
       the result of this function should be reliable; resqml data sets from other sources might use facets
       if a different way, leading to an omission in the results of this function
    """

    assert collection is not None and keyword
    (property_kind, facet_type, facet) = property_kind_and_facet_from_keyword(keyword)
    if property_kind is None:
        log.warning('failed to deduce property kind for keyword: ' + keyword)
        return None
    return selective_version_of_collection(collection,
                                           property_kind = property_kind,
                                           facet_type = facet_type,
                                           facet = facet)


def reformat_column_edges_to_resqml_format(array):
    """Converts an array of shape (nj,ni,2,2) to shape (nj,ni,4) in RESQML edge ordering."""
    newarray = np.empty((array.shape[0], array.shape[1], 4), dtype = array.dtype)
    newarray[:, :, 0] = array[:, :, 1, 0]
    newarray[:, :, 1] = array[:, :, 0, 1]
    newarray[:, :, 2] = array[:, :, 1, 1]
    newarray[:, :, 3] = array[:, :, 0, 0]
    return newarray


def reformat_column_edges_from_resqml_format(array):
    """Converts an array of shape (nj,ni,4) in RESQML edge ordering to shape (nj,ni,2,2)"""
    newarray = np.empty((array.shape[0], array.shape[1], 2, 2), dtype = array.dtype)
    newarray[:, :, 0, 0] = array[:, :, 3]
    newarray[:, :, 0, 1] = array[:, :, 1]
    newarray[:, :, 1, 0] = array[:, :, 0]
    newarray[:, :, 1, 1] = array[:, :, 2]
    return newarray


# 'private' functions returning attribute name for cached version of property array
# I find the leading underscore so ugly, I can't bring myself to use it for 'private' functions, even though many people do


def _cache_name_for_uuid(uuid):
    """Returns the attribute name used for the cached copy of the property array for the given uuid.

    :meta private:
    """

    return 'c_' + bu.string_from_uuid(uuid)


def _cache_name(part):
    """Returns the attribute name used for the cached copy of the property array for the given part.

    :meta private:
    """

    if part is None:
        return None
    uuid = rqet.uuid_in_part_name(part)
    if uuid is None:
        return None
    return _cache_name_for_uuid(uuid)


def dtype_flavour(continuous, use_32_bit):
    """Returns the numpy elemental data type depending on the two boolean flags.

    :meta private:
    """

    if continuous:
        if use_32_bit:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        if use_32_bit:
            dtype = np.int32
        else:
            dtype = np.int64
    return dtype


def return_cell_indices(i, cell_indices):
    """Returns the i'th entry in the cell_indices array, or NaN if i has null value of -1."""

    if i == -1:
        return np.nan
    else:
        return cell_indices[i]


def write_hdf5_and_create_xml_for_active_property(model,
                                                  active_property_array,
                                                  support_uuid,
                                                  title = 'ACTIVE',
                                                  realization = None,
                                                  time_series_uuid = None,
                                                  time_index = None):
    """Writes hdf5 data and creates xml for an active cell property; returns uuid."""

    active = rqp.Property.from_array(parent_model = model,
                                     cached_array = active_property_array,
                                     source_info = None,
                                     keyword = title,
                                     support_uuid = support_uuid,
                                     property_kind = 'active',
                                     local_property_kind_uuid = None,
                                     indexable_element = 'cells',
                                     discrete = True,
                                     time_series_uuid = time_series_uuid,
                                     time_index = time_index,
                                     realization = realization,
                                     find_local_property_kind = True)
    return active.uuid


def check_and_warn_property_kind(pk, activity):
    """Check property kind and warn if one of three main abstract kinds."""
    if pk in ['continuous', 'discrete', 'categorical']:
        warnings.warn(
            f"abstract property kind '{pk}', whilst {activity}, will be or have been replaced by local property kind")
        # raise ValueError


def property_parts(model,
                   obj_type,
                   parts_list = None,
                   property_kind = None,
                   facet_type = None,
                   facet = None,
                   related_uuid = None):
    """Returns list of property parts from model matching filters."""

    def facet_match(root, facet_type, facet):
        if not facet_type and not facet:
            return True
        assert facet_type and facet is not None
        facets = rqet.list_of_tag(root, 'Facet')
        if facet_type == 'none' and facets:
            return False
        for f_node in facets:
            if rqet.find_tag_text(f_node, 'Facet') == facet_type:
                if facet == 'none':
                    return False
                if facet == '*':
                    return True
                return rqet.find_tag_text(f_node, 'Value') == str(facet)
        if facet == 'none':
            return True
        return False

    if not obj_type.endswith('Property'):
        obj_type += 'Property'
    assert obj_type in ['ContinuousProperty', 'DiscreteProperty', 'CategoricalProperty', 'PointsProperty']
    parts = model.parts(parts_list = parts_list, obj_type = obj_type, related_uuid = related_uuid)
    if property_kind:
        pk_parts = []
        for part in parts:
            root = model.root_for_part(part)
            node = rqet.find_nested_tags(root, ['PropertyKind', 'Kind'])
            if node is None:
                # following relies on title in reference xml matching that of the local proparty kind
                node = rqet.find_nested_tags(root, ['PropertyKind', 'LocalPropertyKind', 'Title'])
                assert node is not None
            if node.text != property_kind and (property_kind not in ['rock permeability', 'permeability rock'] or
                                               node.text not in ['rock permeability', 'permeability rock']):
                continue
            if facet_match(root, facet_type, facet):
                pk_parts.append(part)
        parts = pk_parts
    elif facet_type and facet is not None:
        f_parts = []
        for part in parts:
            root = model.root_for_part(part)
            if facet_match(root, facet_type, facet):
                pk_parts.append(part)
    return parts


def property_part(model,
                  obj_type,
                  parts_list = None,
                  property_kind = None,
                  facet_type = None,
                  facet = None,
                  related_uuid = None):
    """Returns individual property part from model matching filters."""
    parts = property_parts(model,
                           obj_type,
                           parts_list = parts_list,
                           property_kind = property_kind,
                           facet_type = facet_type,
                           facet = facet,
                           related_uuid = related_uuid)
    if parts is None or len(parts) == 0:
        return None
    assert len(parts) == 1, 'more than one property part matches criteria'
    return parts[0]
