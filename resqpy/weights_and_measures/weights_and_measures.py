"""Units of measure."""

import json
import warnings
from functools import lru_cache
from pathlib import Path

from resqpy.olio.exceptions import IncompatibleUnitsError, InvalidUnitError

# physical constants – deprecated
feet_to_metres = 0.3048
metres_to_feet = 1.0 / feet_to_metres
ft3_to_m3 = 0.028316846592
m3_to_ft3 = 1.0 / ft3_to_m3
bbl_to_m3 = 0.158987294928
m3_to_bbl = 1.0 / bbl_to_m3
psi_to_kPa = 44.482216152605 / 6.4516
kPa_to_psi = 1.0 / psi_to_kPa
d_to_s = float(24 * 60 * 60)
s_to_d = 1.0 / d_to_s

# Mapping from uom to set of common case-insensitive aliases
# Nb. No need to write out fractional combinations such as "bbl/day"
# note: some of the aliases are ambiguous, e.g. 'gm' could mean gigametre or gramme
UOM_ALIASES = {
    # Mass
    'g': {'gm', 'gram', 'gramme', 'grams', 'grammes'},
    'lbm': {'lb', 'lbs'},  # assumes pounds mass rather than pounds force

    # Length
    'm': {'m', 'metre', 'metres', 'meter', 'meters'},
    'ft': {'ft', 'foot', 'feet'},
    'cm': {'centimetre', 'centimetres', 'centimeter', 'centimeters'},

    # Time
    'ms': {'ms', 'msec', 'millisecs', 'millisecond', 'milliseconds'},
    's': {'s', 'sec', 'secs', 'second', 'seconds'},
    'min': {'min', 'mins', 'minute', 'minutes'},
    'h': {'h', 'hr', 'hour', 'hours'},
    'd': {'day', 'days'},
    'wk': {'wk', 'week', 'weeks'},
    'a': {'a', 'yr', 'year', 'years'},

    # Ratio
    '%': {'%', 'pu', 'p.u.', 'percent'},
    'm3/m3': {'m3/m3', 'v/v'},
    'g/cm3': {'g/cm3', 'g/cc'},

    # Volume
    'bbl': {'bbl', 'stb', 'rb'},
    '1000 bbl': {'1000 bbl', 'mstb', 'mbbl', 'mrb'},
    '1E6 bbl': {'1E6 bbl', 'mmstb', 'mmbbl'},
    '1E6 ft3': {'1E6 ft3', 'mmscf'},
    '1000 ft3': {'1000 ft3', 'mscf'},
    'm3': {'m3', 'sm3', 'stm3', 'rm3'},
    '1000 m3': {'kstm3', 'krm3', 'msm3', 'mrm3'},
    'ft3': {'ft3', 'scf', 'cf', 'rcf', 'cu.ft.'},
    'cm3': {'cc', 'scc', 'stcc'},
    'L': {'krcc', 'kstcc', 'kscc', 'litre', 'litres', 'liter', 'liters'},

    # Pressure & Reciprocal Pressure
    'psi': {'psi', 'psia'},

    # Thermodynamic Temperature
    'degC': {'c', 'degrees c', 'degreesc'},
    'degF': {'f', 'degrees f', 'degreesf'},

    # Energy
    'Btu[IT]': {'btu'},  # assumes BTU to refer to ISO standard, rather than older Btu[UK]
    'kJ': {'kj'},

    # Other
    'gAPI': {'gapi'},
    'S': {'mho'},
    'mS': {'mmho'},
    'mol': {'mole', 'moles'},
    'Euc': {'count', 'fraction', 'none'},
}
# Mapping from alias to valid uom
UOM_ALIAS_MAP = {alias.casefold(): uom for uom, aliases in UOM_ALIASES.items() for alias in aliases}

# Set of uoms that can be safely matched case-insensitive
CASE_INSENSITIVE_UOMS = {'m', 'ft', 'm3', 'ft3', 'm3/m3', 'ft3/ft3', 'bbl', 'bar', 'psi', 'm3/d', 'bbl/d'}


@lru_cache(None)
def rq_uom(units, quantity = None):
    """Returns RESQML uom string equivalent to units.

    arguments:
       units (str): unit to coerce
       quantity (str, optional): if given, raise an exception if the uom is not supported
          for this quantity

    returns:
       str: unit of measure

    raises:
       InvalidUnitError: if units cannot be coerced into RESQML units for the given quantity
    """
    if not units:
        raise InvalidUnitError("Must provide non-empty unit")

    uom = _try_parse_unit(units.strip())

    if uom is None:
        raise InvalidUnitError(f"Cannot coerce {units} into a valid RESQML unit of measure.")

    if quantity is not None:
        supported_uoms = valid_uoms(quantity = quantity)
        if uom not in supported_uoms:
            raise InvalidUnitError(f"Unit {uom} is not supported for quantity {quantity}.\n"
                                   f"Supported units:\n{supported_uoms}")
    return uom


def convert(x, unit_from, unit_to, quantity = None, inplace = False):
    """Convert value between two compatible units.

    arguments:
       x (numeric or np.array): value(s) to convert
       unit_from (str): resqml uom
       unit_to (str): resqml uom
       quantity (str, optional): If provided, raise an exception if units are not supported
          by this quantity
       inplace (bool): if True, convert arrays in-place. Else, return new value

    returns:
       Converted value(s)

    raises:
       InvalidUnitError: if units cannot be coerced into RESQML units
       IncompatibleUnitsError: if units do not have compatible base units
    """

    # conversion data assume the formula "y=(A + Bx)/(C + Dx)" where "y" represents a value in the base unit.
    # Backwards formula: x=(A-Cy)/(Dy-B)
    # All current units have D==0

    uom1 = rq_uom(unit_from, quantity = quantity)
    uom2 = rq_uom(unit_to, quantity = quantity)

    if uom1 == uom2:
        return x

    base1, dim1, (A1, B1, C1, D1) = get_conversion_factors(uom1)
    base2, dim2, (A2, B2, C2, D2) = get_conversion_factors(uom2)

    if base1 != base2:
        if dim1 != dim2:
            raise IncompatibleUnitsError(f"Cannot convert from '{unit_from}' to '{unit_to}':"
                                         f"\n - '{uom1}' has base unit '{base1} and dimension '{dim1}'."
                                         f"\n - '{uom2}' has base unit '{base2} and dimension '{dim2}'.")
        else:
            warnings.warn(f"Assuming base units {base1} and {base2} are equivalent as they have the same dimensions:"
                          f"\n - '{uom1}' has base unit '{base1} and dimension '{dim1}'."
                          f"\n - '{uom2}' has base unit '{base2} and dimension '{dim2}'.")

    if not inplace:
        y = (A1 + (B1 * x)) / (C1 + (D1 * x))
        return (A2 - (C2 * y)) / ((D2 * y) - B2)

    else:
        if any(f != 0 for f in [A1, A2, D1, D2]):
            raise NotImplementedError("In-place conversion not yet implemented for non-trivial conversions")

        factor = (B1 * C2) / (C1 * B2)
        x *= factor
        return x


@lru_cache(None)
def valid_uoms(quantity = None, return_attributes = False):
    """Return set of valid RESQML units of measure.

    arguments:
       quantity (str): If given, filter to uoms supported by this quanitity.
       return_attributes (bool): If True, return a dict of all uoms and their
          attributes, such as the full name and dimension. Else, simply return
          the set of valid uoms.

    returns
       set or dict
    """
    uoms = _properties_data()['units']
    if quantity:
        all_quantities = _properties_data()['quantities']
        supported_members = all_quantities[quantity]['members']
        uoms = {k: v for k, v in uoms.items() if k in supported_members}
    if return_attributes:
        return uoms
    else:
        return set(uoms.keys())


@lru_cache(None)
def valid_quantities(return_attributes = False):
    """Return set of valid RESQML quantities.

    arguments:
       return_attributes (bool): If True, return a dict of all quantities and their
          attributes, such as the supported units of measure. Else, simply return
          the set of valid properties.

    returns
       set or dict
    """
    quantities = _properties_data()['quantities']
    if return_attributes:
        return quantities
    else:
        return set(quantities.keys())


def valid_property_kinds():
    """Return set of valid property kinds."""

    return set(_properties_data()['property_kinds'].keys())


def rq_uom_list(units_list):
    """Returns a list of RESQML uom equivalents for units in list."""

    return [rq_uom(u) for u in units_list]


def rq_length_unit(units):
    """Returns length units string as expected by resqml."""

    return rq_uom(units, quantity = 'length')


def rq_time_unit(units):
    """Returns time units string as expected by resqml."""

    return rq_uom(units, quantity = 'time')


def convert_times(a, from_units, to_units, invert = False):
    """Converts values in numpy array (or a scalar) from one time unit to another, in situ if array.

    note:
       To see supported units, use: `valid_uoms(quantity='time')`
    """

    if invert:
        from_units, to_units = to_units, from_units

    return convert(a, from_units, to_units, quantity = 'time', inplace = True)


def convert_lengths(a, from_units, to_units):
    """Converts values in numpy array (or a scalar) from one length unit to another, in situ if array.

    arguments:
       a (numpy float array, or float): array of length values to undergo unit conversion in situ, or a scalar
       from_units (string): the units of the data before conversion
       to_units (string): the required units

    returns:
       a after unit conversion

    note:
       To see supported units, use: `valid_uoms(quantity='length')`
    """

    return convert(a, from_units, to_units, quantity = 'length', inplace = True)


def convert_pressures(a, from_units, to_units):
    """Converts values in numpy array (or a scalar) from one pressure unit to another, in situ if array.

    arguments:
       a (numpy float array, or float): array of pressure values to undergo unit conversion in situ, or a scalar
       from_units (string): the units of the data before conversion
       to_units (string): the required units

    returns:
       a after unit conversion

    note:
       To see supported units, use: `valid_uoms(quantity='pressure')`
    """
    return convert(a, from_units, to_units, quantity = 'pressure', inplace = True)


def convert_volumes(a, from_units, to_units):
    """Converts values in numpy array (or a scalar) from one volume unit to another, in situ if array.

    arguments:
       a (numpy float array, or float): array of volume values to undergo unit conversion in situ, or a scalar
       from_units (string): units of the data before conversion; see note for accepted units
       to_units (string): the required units; see note for accepted units

    returns:
       a after unit conversion

    note:
       To see supported units, use: `valid_uoms(quantity='volume')`
    """
    return convert(a, from_units, to_units, quantity = 'volume', inplace = True)


def convert_flow_rates(a, from_units, to_units):
    """Converts values in numpy array (or a scalar) from one volume flow rate unit to another, in situ if array.

    arguments:
       a (numpy float array, or float): array of volume flow rate values to undergo unit conversion in situ, or a scalar
       from_units (string): units of the data before conversion, eg. 'm3/d'; see notes for acceptable units
       to_units (string): required units of the data after conversion, eg. 'ft3/d'; see notes for acceptable units

    returns:
       a after unit conversion

    note:
       To see supported units, use: `valid_uoms(quantity='volume per time')`
    """
    return convert(a, from_units, to_units, quantity = 'volume per time', inplace = True)


@lru_cache(None)
def get_conversion_factors(uom):
    """Return base unit and conversion factors (A, B, C, D) for a given uom.

    The formula "y=(A + Bx)/(C + Dx)" where "y" represents a value in the base unit.

    Returns:
       3-tuple of (base_unit, dimension, factors). Factors is a 4-tuple of conversion factors

    Raises:
       ValueError if either uom is not a valid resqml uom
    """
    if uom not in valid_uoms():
        raise ValueError(f"{uom} is not a valid uom")
    uoms_data = _properties_data()["units"][uom]

    dimension = uoms_data["dimension"]
    try:
        a, b, c, d = uoms_data["A"], uoms_data["B"], uoms_data["C"], uoms_data["D"]
        base_unit = uoms_data["baseUnit"]
    except KeyError:  # Base units do not have factors defined
        a, b, c, d = 0, 1, 1, 0
        base_unit = uom
    return base_unit, dimension, (a, b, c, d)


# Private functions


@lru_cache(None)
def _properties_data():
    """Return a data structure that represents resqml unit system.

    The dict is loaded directly from a JSON file which is bundled with resqpy.
    The unit system is represented as a dict with the following keys:

    - dimensions
    - quantities
    - units
    - prefixes
    - property_kinds

    Returns:
       dict: resqml unit system
    """
    json_path = Path(__file__).parent.parent / 'olio/data/properties.json'
    with open(json_path) as f:
        data = json.load(f)
    return data


def _try_parse_unit(units):
    """Try to match unit against known uoms and aliases, else return None."""

    uom_list = valid_uoms()
    ul = units.casefold()

    uom = None

    if units in uom_list:
        uom = units
    elif ul in CASE_INSENSITIVE_UOMS:
        uom = ul
    elif ul in UOM_ALIAS_MAP:
        uom = UOM_ALIAS_MAP[ul]
    elif ul in uom_list:
        uom = ul  # dangerous! for example, 'D' means D'Arcy and 'd' means day
    elif units.startswith('(') and units.endswith(')') and '(' not in units[1:]:  # simplistic
        uom = _try_parse_unit(units[1:-1])
    elif '/' in units:  # May be a fraction: match each part against known aliases
        parts = units.split('/', 1)
        newpart0 = _try_parse_unit(parts[0])
        newpart1 = _try_parse_unit(parts[1])
        if newpart0 and newpart1:
            ratio = f"{newpart0}/{newpart1}"
            if ratio in uom_list:
                uom = ratio

    return uom
