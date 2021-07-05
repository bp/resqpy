"""Units of measure module."""

from pathlib import Path
import json
import warnings
from functools import lru_cache
from resqpy.olio.exceptions import InvalidUnitError, IncompatibleUnitsError


version = '17th June 2021'

# physical constants
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

# Mapping from uom to set of common (non-resqml) case-insensitive aliases
UOM_ALIASES = {
   # Lengths
   'm': {'m', 'metre', 'metres', 'meter', 'meters'},
   'ft': {'ft', 'foot', 'feet'},

   # Times
   'd': {'day', 'days'},

   # Ratios
   '%': {'%', 'pu', 'p.u.'},
   'm3/m3': {'m3/m3', 'v/v'},
   'g/cm3': {'g/cm3', 'g/cc'},

   # Volumes
   'bbl': {'bbl', 'stb'},
   '1000 bbl': {'1000 bbl', 'mstb', 'mbbl'},
   '1E6 bbl': {'1E6 bbl', 'mmstb', 'mmbbl'},
   '1E6 ft3': {'1E6 ft3', 'mmscf'},
   '1000 ft3': {'1000 ft3', 'mscf'},
   'm3': {'m3', 'sm3'},
   'ft3': {'ft3', 'scf'},

   # Rates
   'bbl/d': {'bbl/d', 'stb/d', 'bbl/day', 'stb/day'},
   '1000 bbl/d': {'1000 bbl/d', 'mstb/day', 'mbbl/day', 'mstb/d', 'mbbl/d'},
   '1E6 bbl/d': {'1E6 bbl/d', 'mmstb/day', 'mmbbl/day', 'mmstb/d', 'mmbbl/d'},
   '1000 ft3/d': {'1000 ft3/d', 'mscf/day', 'mscf/d'},
   'ft3/d': {'ft3/d', 'scf/day', 'scf/d'},
   '1E6 ft3/d': {'1E6 ft3/d', 'mmscf/day', 'mmscf/d'},
   'm3/d': {'m3/d', 'm3/day', 'sm3/d', 'sm3/day'},
   '1000 m3/d': {'1000 m3/d', '1000 m3/day'},
   '1E6 m3/d': {'1E6 m3/d', '1E6 m3/day'},

   # Other
   'ft3/bbl': {'ft3/bbl', 'scf/bbl', 'ft3/stb', 'scf/stb'},
   '1000 ft3/bbl': {'1000 ft3/bbl', 'mscf/bbl', 'mscf/stb'},
   'gAPI': {'gapi'},
   'psi': {'psi', 'psia'},
   'Euc': {'count'},
}
# Mapping from alias to valid uom
UOM_ALIAS_MAP = {alias.casefold(): uom for uom, aliases in UOM_ALIASES.items() for alias in aliases}

# Set of uoms that can be safely matched case-insensitive
CASE_INSENSITIVE_UOMS = {'m', 'ft', 'm3', 'ft3', 'm3/m3', 'ft3/ft3', 'bbl', 'bar', 'psi', 'm3/d', 'bbl/d'}


@lru_cache(None)
def rq_uom(units):
   """Returns RESQML uom string equivalent to units
   
   Args:
      units (str): unit to coerce

   Returns:
      str: unit of measure

   Raises:
      InvalidUnitError: if units cannot be coerced into RESQML units
   """
   if not units:
      raise InvalidUnitError("Must provide non-empty unit")
   
   # Valid uoms
   uom_list = valid_uoms()
   if units in uom_list: return units

   # Common alises
   ul = units.casefold()
   if ul in CASE_INSENSITIVE_UOMS: return ul
   if ul in UOM_ALIAS_MAP: return UOM_ALIAS_MAP[ul]
   if ul in uom_list: return ul  # dangerous! for example, 'D' means D'Arcy and 'd' means day

   raise InvalidUnitError(f"Cannot coerce {units} into a valid RESQML unit of measure.")


def convert(x, unit_from, unit_to):
   """Convert value between two compatible units
   
   Args:
      x (numeric or np.array): value(s) to convert
      unit_from (str): resqml uom
      unit_to (str): resqml uom

   Returns:
      Converted value(s)

   Raises:
      InvalidUnitError: if units cannot be coerced into RESQML units
      IncompatibleUnitsError: if units do not have compatible base units
   """
   # TODO: robust handling of errors. At present, bad units are treated as "EUC", and will fail silently.

   # conversion data assume the formula "y=(A + Bx)/(C + Dx)" where "y" represents a value in the base unit.
   # Backwards formula: x=(A-Cy)/(Dy-B)
   # All current units have D==0

   uom1, uom2 = rq_uom(unit_from), rq_uom(unit_to)
   if uom1 == uom2:
      return x
   
   base1, dim1, (A1, B1, C1, D1) = get_conversion_factors(uom1)
   base2, dim2, (A2, B2, C2, D2) = get_conversion_factors(uom2)

   if base1 != base2:
      if dim1 != dim2:
         raise IncompatibleUnitsError(
            f"Cannot convert from '{unit_from}' to '{unit_to}':"
            f"\n - '{uom1}' has base unit '{base1} and dimension '{dim1}'."
            f"\n - '{uom2}' has base unit '{base2} and dimension '{dim2}'."
         )
      else:
         warnings.warn(
            f"Assuming base units {base1} and {base2} are equivalent as they have the same dimensions:"
            f"\n - '{uom1}' has base unit '{base1} and dimension '{dim1}'."
            f"\n - '{uom2}' has base unit '{base2} and dimension '{dim2}'."
         )

   y = (A1 + (B1*x)) / (C1 + (D1*x))
   return (A2 - (C2*y)) / ((D2*y) - B2)


@lru_cache(None)
def valid_uoms():
   """Return set of valid uoms"""

   return set(_properties_data()['units'].keys())


@lru_cache(None)
def valid_property_kinds():
   """Return set of valid property kinds"""
   
   return set(_properties_data()['property_kinds'].keys())


def rq_uom_list(units_list):
   """Returns a list of RESQML uom equivalents for units in list."""

   return [rq_uom(u) for u in units_list]


def rq_length_unit(units):
   """Returns length units string as expected by resqml."""

   # NB: other length units are supported by resqml
   if units.lower() in ['m', 'metre', 'metres']: return 'm'
   if units.lower() in ['ft', 'foot', 'feet', 'ft[us]']: return 'ft'  # NB. treating different foot sizes as identical
   raise ValueError(f'unrecognised length units {units}')


def rq_time_unit(units):
   """Returns time units string as expected by resqml."""

   #  NB: other time units are supported by resqml
   if units.lower() in ['d', 'day', 'days']: return 'd'  # note: 'D' is actually RESQML uom for D'Arcy
   if units.lower() in ['s', 'sec', 'secs', 'second', 'seconds']: return 's'
   if units.lower() in ['ms', 'msec', 'millisecs', 'millisecond', 'milliseconds']: return 'ms'
   if units.lower() in ['min', 'mins', 'minute', 'minutes']: return 'min'
   if units.lower() in ['h', 'hr', 'hour', 'hours']: return 'h'
   if units.lower() in ['wk', 'week', 'weeks']: return 'wk'
   if units.lower() in ['a', 'yr', 'year', 'years']: return 'a'
   assert(False)  # unrecognised time units


def convert_times(a, from_units, to_units, invert = False):
   """Converts values in numpy array (or a scalar) from one time unit to another, in situ if array."""

   # TODO: check RESQML standard definition of length of day, week, year
   valid_units = ('d', 's', 'h', 'ms', 'min')
   from_units = rq_time_unit(from_units)
   to_units = rq_time_unit(to_units)
   if from_units == to_units: return a
   assert from_units in valid_units and to_units in valid_units
   factor = 1.0
   if from_units == 's': factor = s_to_d
   elif from_units == 'h': factor = 1.0 / 24.0
   elif from_units == 'ms': factor = 0.001 * s_to_d
   elif from_units == 'min': factor = 60.0 * s_to_d
   if to_units == 's': factor *= d_to_s
   elif to_units == 'h': factor *= 24.0
   elif to_units == 'ms': factor *= 1000.0 * d_to_s
   elif to_units == 'min': factor *= d_to_s / 60.0
   if invert: factor = 1.0 / factor
   a *= factor
   return a


def convert_lengths(a, from_units, to_units):
   """Converts values in numpy array (or a scalar) from one length unit to another, in situ if array.

      arguments:
         a (numpy float array, or float): array of length values to undergo unit conversion in situ, or a scalar
         from_units (string): 'm', 'metres', 'ft' or 'feet' being the units of the data before conversion
         to_units (string): 'm', 'metres', 'ft' or 'feet' being the required units

      returns:
         a after unit conversion
   """

   from_units = rq_length_unit(from_units)
   to_units = rq_length_unit(to_units)
   if from_units == to_units: return a
   if from_units == 'ft' and to_units == 'm': a *= feet_to_metres
   elif from_units == 'm' and to_units == 'ft': a *= metres_to_feet
   else: raise ValueError('unsupported length unit conversion')
   return a


def convert_pressures(a, from_units, to_units):
   """Converts values in numpy array (or a scalar) from one pressure unit to another, in situ if array.

      arguments:
         a (numpy float array, or float): array of pressure values to undergo unit conversion in situ, or a scalar
         from_units (string): 'kPa', 'Pa', 'bar' or 'psi' being the units of the data before conversion
         to_units (string): 'kPa', 'Pa', 'bar' or 'psi' being the required units

      returns:
         a after unit conversion
   """

   from_units = rq_uom(from_units)
   to_units = rq_uom(to_units)
   assert from_units in ['kPa', 'Pa', 'bar', 'psi'] and to_units in ['kPa', 'Pa', 'bar', 'psi']
   if from_units == to_units: return a
   if from_units in ['kPa', 'Pa', 'bar'] and to_units == 'psi': factor = kPa_to_psi
   elif from_units == 'psi' and to_units in ['kPa', 'Pa', 'bar']: factor = psi_to_kPa
   else: factor = 1.0
   if from_units == 'Pa': factor *= 0.001
   elif from_units == 'bar': factor *= 100.0
   if to_units == 'Pa': factor *= 1000.0
   elif to_units == 'bar': factor *= 0.01
   a *= factor
   return a


def convert_volumes(a, from_units, to_units):
   """Converts values in numpy array (or a scalar) from one volume unit to another, in situ if array.

      arguments:
         a (numpy float array, or float): array of volume values to undergo unit conversion in situ, or a scalar
         from_units (string): units of the data before conversion; see note for accepted units
         to_units (string): the required units; see note for accepted units

      returns:
         a after unit conversion

      note:
         currently accepted units are:
         'm3', 'ft3', 'bbl', '1000 m3', '1000 ft3', '1000 bbl', '1E6 m3', '1E6 ft3', '1E6 bbl'
   """

   valid_units = ('m3', 'ft3', 'bbl', '1000 m3', '1000 ft3', '1000 bbl', '1E6 m3', '1E6 ft3', '1E6 bbl')
   from_units = rq_uom(from_units)
   to_units = rq_uom(to_units)
   factor = 1.0
   assert from_units in valid_units and to_units in valid_units
   if from_units == to_units: return a
   if from_units.startswith('1000 ') and to_units.startswith('1000 '):
      from_units = from_units[5:]
      to_units = to_units[5:]
   elif from_units.startswith('1000 '):
      factor = 1000.0
      from_units = from_units[5:]
   elif to_units.startswith('1000 '):
      factor = 0.001
      to_units = to_units[5:]
   if from_units.startswith('1E6 ') and to_units.startswith('1E6 '):
      from_units = from_units[4:]
      to_units = to_units[4:]
   elif from_units.startswith('1E6 '):
      factor *= 1000000.0
      from_units = from_units[4:]
   elif to_units.startswith('1E6 '):
      factor *= 0.000001
      to_units = to_units[4:]
   if from_units != to_units:
      if from_units == 'm3':
         if to_units == 'ft3': factor *= m3_to_ft3
         else: factor *= m3_to_bbl
      elif from_units == 'ft3':
         if to_units == 'm3': factor *= ft3_to_m3
         else: factor *= ft3_to_m3 * m3_to_bbl
      elif from_units == 'bbl':
         if to_units == 'm3': factor *= bbl_to_m3
         else: factor *= bbl_to_m3 * m3_to_ft3
      else:
         raise ValueError(f'unacceptable volume units {from_units}')
   a *= factor
   return a


def convert_flow_rates(a, from_units, to_units):
   """Converts values in numpy array (or a scalar) from one volume flow rate unit to another, in situ if array.

      arguments:
         a (numpy float array, or float): array of volume flow rate values to undergo unit conversion in situ, or a scalar
         from_units (string): units of the data before conversion, eg. 'm3/d'; see notes for acceptable units
         to_units (string): required units of the data after conversion, eg. 'ft3/d'; see notes for acceptable units

      returns:
         a after unit conversion

      note:
         units should be in the form volume/time where valid volume units are:
         'm3', 'ft3', 'bbl', '1000 m3', '1000 ft3', '1000 bbl', '1E6 m3', '1E6 ft3', '1E6 bbl'
         and valid time units are:
         'd', 's', 'h', 'ms', 'min'
   """

   valid_volume_units = ('m3', 'ft3', 'bbl', '1000 m3', '1000 ft3', '1000 bbl', '1E6 m3', '1E6 ft3', '1E6 bbl')
   valid_time_units = ('d', 'h', 's')

   from_unit_pair = from_units.split('/')
   to_unit_pair = to_units.split('/')
   assert len(from_unit_pair) == len(to_unit_pair) == 2

   a = convert_volumes(a, from_unit_pair[0], to_unit_pair[0])
   a = convert_times(a, from_unit_pair[1], to_unit_pair[1], invert = True)
   return a


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
   """ Return a data structure that represents resqml unit system.

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
   json_path = Path(__file__).parent / 'data' / 'properties.json'
   with open(json_path) as f:
      data = json.load(f)
   return data



