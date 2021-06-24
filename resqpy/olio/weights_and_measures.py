"""Units of measure module."""

from pathlib import Path
import json
from functools import lru_cache
from pint import UnitRegistry, set_application_registry

# Bifrost weights and measures module
# Temporary version based on pagoda code
# todo: Replace with RESQML uom based version at a later date

version = '17th June 2021'

# Shared instance of pint unit registry
ureg = UnitRegistry(
   filename=str(Path(__file__).parent / 'data/unit_registry.txt'),
   preprocessors=(
      # Preprocessors: see https://github.com/hgrecco/pint/issues/185
      lambda s: s.replace('%', ' percent '),
      lambda s: s.replace('p.u.', ' poreunits '),
   ),

)
set_application_registry(ureg)
Q_ = ureg.Quantity

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



def rq_uom(units):
   """Returns RESQML uom string equivalent to units, or 'Euc' if not determined."""

   if not isinstance(units, str): return 'Euc'
   if units == '' or units == 'Euc': return 'Euc'
   ul = units.lower()
   if ul in ['m', 'ft', 'm3', 'ft3', 'm3/m3', 'ft3/ft3', 'bbl', 'bar', 'psi', 'm3/d', 'bbl/d']: return ul
   if ul in ['m', 'metre', 'metres', 'meter']: return 'm'
   if ul in ['ft', 'foot', 'feet', 'ft[us]']: return 'ft'  # NB. treating different foot sizes as identical
   if units == 'd' or ul in ['days', 'day']: return 'd'
   if units in ['kPa', 'Pa', 'mD']: return units
   if ul in ['psi', 'psia']: return 'psi'
   if ul in ['1000 bbl', 'mstb', 'mbbl']: return '1000 bbl'
   if ul in ['bbl', 'stb']: return 'bbl'
   if ul in ['1E6 bbl', 'mmstb', 'mmbbl']: return '1E6 bbl'
   if ul in ['1E6 ft3', 'mmscf']: return '1E6 ft3'
   if ul in ['1000 ft3', 'mscf']: return '1000 ft3'
   if ul in ['ft3', 'scf']: return 'ft3'
   if ul in ['bbl/d', 'stb/d', 'bbl/day', 'stb/day']: return 'bbl/d'
   if ul in ['1000 bbl/d', 'mstb/day', 'mbbl/day', 'mstb/d', 'mbbl/d']: return '1000 bbl/d'
   if ul in ['1E6 bbl/d', 'mmstb/day', 'mmbbl/day', 'mmstb/d', 'mmbbl/d']: return '1E6 bbl/d'
   if ul in ['1000 ft3/d', 'mscf/day', 'mscf/d']: return '1000 ft3/d'
   if ul in ['ft3/d', 'scf/day', 'scf/d']: return 'ft3/d'
   if ul in ['1E6 ft3/d', 'mmscf/day', 'mmscf/d']: return '1E6 ft3/d'
   if ul in ['ft3/bbl', 'scf/bbl', 'ft3/stb', 'scf/stb']: return 'ft3/bbl'
   if ul in ['1000 ft3/bbl', 'mscf/bbl', 'mscf/stb']: return '1000 ft3/bbl'
   if ul in ['m3', 'sm3']: return 'm3'
   if ul in ['m3/d', 'm3/day', 'sm3/d', 'sm3/day']: return 'm3/d'
   if ul == '1000 m3': return '1000 m3'
   if ul in ['1000 m3/d', '1000 m3/day']: return '1000 m3/d'
   if ul == '1E6 m3': return '1E6 m3'
   if ul in ['1E6 m3/d', '1E6 m3/day']: return '1E6 m3/d'
   if units in ['mD.m', 'mD.ft']: return units
   if ul == 'count': return 'Euc'
   uom_list = properties_data()['uoms']
   if units in uom_list: return units
   if ul in uom_list: return ul  # dangerous! for example, 'D' means D'Arcy and 'd' means day
   return 'Euc'


def rq_uom_list(units_list):
   """Returns a list of RESQML uom equivalents for units in list."""

   rq_list = []
   for u in units_list: rq_list.append(rq_uom(u))
   return rq_list


def p_length_unit(units):
   """Returns length units string as expected by pagoda weights and measures module."""

   # NB: other length units are supported by resqml
   if units.lower() in ['m', 'metre', 'metres']: return 'metres'
   if units.lower() in ['ft', 'foot', 'feet', 'ft[us]']: return 'feet'
   assert(False)  # unrecognised length units


def rq_length_unit(units):
   """Returns length units string as expected by resqml."""

   # NB: other length units are supported by resqml
   if units.lower() in ['m', 'metre', 'metres']: return 'm'
   if units.lower() in ['ft', 'foot', 'feet', 'ft[us]']: return 'ft'  # NB. treating different foot sizes as identical
   raise ValueError(f'unrecognised length units {units}')


def p_time_unit(units):
   """Returns human readable version of time units string."""

   #  NB: other time units are supported by resqml
   if units.lower() in ['d', 'day', 'days']: return 'days'
   if units.lower() in ['s', 'sec', 'secs', 'second', 'seconds']: return 'seconds'
   if units.lower() in ['ms', 'msec', 'millisecs', 'millisecond', 'milliseconds']: return 'milliseconds'
   if units.lower() in ['min', 'mins', 'minute', 'minutes']: return 'minutes'
   if units.lower() in ['h', 'hr', 'hour', 'hours']: return 'hours'
   if units.lower() in ['wk', 'week', 'weeks']: return 'weeks'
   if units.lower() in ['a', 'yr', 'year', 'years']: return 'years'
   assert(False)  # unrecognised time units


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



@lru_cache(maxsize=None)
def properties_data():
   """ Return valid resqml uoms and property kinds.

   Returns a dict with keys:
   - "uoms" : list of valid units of measure
   - "property_kinds" : dict mapping valid property kinds to their description
   """
   json_path = Path(__file__).parent / 'data' / 'properties.json'
   with open(json_path) as f:
      data = json.load(f)
   return data
