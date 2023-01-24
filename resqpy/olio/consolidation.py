"""Support for consolidation of datasets based on equivalence between parts."""

import logging

log = logging.getLogger(__name__)

import resqpy.crs as rqc
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.property as rqp
import resqpy.time_series as rqt

# the following list contains those RESQML classes with equivalence methods in their resqpy class
# it is lightly ordered with earlier classes having no dependence on classes later in the list

consolidatable_list = [
    'OrganizationFeature', 'GeobodyFeature', 'BoundaryFeature', 'FrontierFeature', 'GeologicUnitFeature',
    'FluidBoundaryFeature', 'RockFluidUnitFeature', 'TectonicBoundaryFeature', 'GeneticBoundaryFeature',
    'WellboreFeature', 'FaultInterpretation', 'EarthModelInterpretation', 'HorizonInterpretation',
    'GeobodyBoundaryInterpretation', 'GeobodyInterpretation', 'WellboreInterpretation', 'LocalDepth3dCrs',
    'LocalTime3dCrs', 'TimeSeries', 'StringTableLookup', 'PropertyKind'
]

# todo: add to this list as other classes gain an is_equivalent() method


class Consolidation:
    """Class supporting equivalence mapping of high level RESQML parts between models."""

    def __init__(self, resident_model):
        """Initialise a new Consolidation object prior to merging parts from another model.

        arguments:
           resident_model (model.Model): the model into which potentially equivalent parts will be merged

        returns:
           the new Consolidation object
        """

        self.model = resident_model
        log.debug(f'new consolidation for {self.model.epc_file} with {len(self.model.uuids())} uuids')
        self.map = {}  # dictionary mapping immigrant uuid int to primary uuid int
        self.stale = True

    def equivalent_uuid_for_part(self, part, immigrant_model = None, ignore_identical_part = False):
        """Returns uuid of an equivalent part in resident model, or None if no equivalent found."""
        uuid_int = self.equivalent_uuid_int_for_part(part,
                                                     immigrant_model = immigrant_model,
                                                     ignore_identical_part = ignore_identical_part)
        return bu.uuid_from_int(uuid_int)

    def equivalent_uuid_int_for_part(self, part, immigrant_model = None, ignore_identical_part = False):
        """Returns uuid.int of an equivalent part in resident model, or None if no equivalent found."""

        # log.debug('Looking for equivalent uuid for: ' + str(part))
        if not part:
            return None
        if immigrant_model is None:
            immigrant_model = self.model
        immigrant_uuid_int = rqet.uuid_in_part_name(part).int
        # log.debug('   immigrant uuid: ' + str(immigrant_uuid))
        if immigrant_uuid_int in self.map:
            # log.debug('   known to be equivalent to: ' + str(self.map[immigrant_uuid_int]))
            return self.map[immigrant_uuid_int]
        obj_type = immigrant_model.type_of_part(part, strip_obj = True)
        if obj_type is None or obj_type not in consolidatable_list:
            return None
        # log.debug('   object type is consolidatable')
        resident_uuids = self.model.uuids(obj_type = obj_type)
        if resident_uuids is None or len(resident_uuids) == 0:
            # log.debug('   no resident parts found of type: ' + str(obj_type))
            return None
        # log.debug(f'   {len(resident_uuids)} resident parts of same class')
        if not ignore_identical_part:
            for resident_uuid in resident_uuids:
                if resident_uuid.int == immigrant_uuid_int:
                    # log.debug('   uuid already resident: ' + str(resident_uuid))
                    return resident_uuid.int

        # log.debug('   preparing immigrant object')
        immigrant_uuid = bu.uuid_from_int(immigrant_uuid_int)
        if obj_type.endswith('Interpretation') or obj_type.endswith('Feature'):
            immigrant_obj = rqo.__dict__[obj_type](immigrant_model, uuid = immigrant_uuid)
        elif obj_type.endswith('Crs'):
            immigrant_obj = rqc.Crs(immigrant_model, uuid = immigrant_uuid)
        elif obj_type == 'TimeSeries':
            immigrant_obj = rqt.TimeSeries(immigrant_model, uuid = immigrant_uuid)
        elif obj_type == 'StringTableLookup':
            immigrant_obj = rqp.StringLookup(immigrant_model, uuid = immigrant_uuid)
        elif obj_type == 'PropertyKind':
            immigrant_obj = rqp.PropertyKind(immigrant_model, uuid = immigrant_uuid)
        else:
            raise Exception('code failure')
        assert immigrant_obj is not None
        for resident_uuid in resident_uuids:
            resident_uuid_int = resident_uuid.int
            # log.debug('   considering resident: ' + str(resident_uuid))
            if ignore_identical_part and bu.matching_uuids(resident_uuid, immigrant_uuid):
                continue
            if obj_type.endswith('Interpretation') or obj_type.endswith('Feature'):
                resident_obj = rqo.__dict__[obj_type](self.model, uuid = resident_uuid)
            elif obj_type.endswith('Crs'):
                resident_obj = rqc.Crs(self.model, uuid = resident_uuid)
            elif obj_type == 'TimeSeries':
                resident_obj = rqt.TimeSeries(self.model, uuid = resident_uuid)
            elif obj_type == 'StringTableLookup':
                resident_obj = rqp.StringLookup(self.model, uuid = resident_uuid)
            elif obj_type == 'PropertyKind':
                resident_obj = rqp.PropertyKind(self.model, uuid = resident_uuid)
            else:
                raise Exception('code failure')
            assert resident_obj is not None
            # log.debug('   comparing with: ' + str(resident_obj.uuid))
            if immigrant_obj == resident_obj:  # note: == operator overloaded with equivalence method for these classes
                while resident_uuid_int in self.map:
                    # log.debug('   following equivalence for: ' + str(resident_uuid))
                    resident_uuid_int = self.map[resident_uuid_int]
                self.map[immigrant_uuid_int] = resident_uuid_int
                # log.debug('   new equivalence found with: ' + str(resident_uuid))
                return resident_uuid_int
        return None

    def force_uuid_equivalence(self, immigrant_uuid, resident_uuid):
        """Forces immigrant object to be treated as equivalent to (same as) resident object, identified by uuids."""

        assert immigrant_uuid is not None and resident_uuid is not None
        if isinstance(immigrant_uuid, str):
            immigrant_uuid = bu.uuid_from_string(immigrant_uuid)
        if isinstance(resident_uuid, str):
            resident_uuid = bu.uuid_from_string(resident_uuid)
        if bu.matching_uuids(immigrant_uuid, resident_uuid):
            return
        assert immigrant_uuid not in self.map.values()

        self.force_uuid_int_equivalence(immigrant_uuid.int, resident_uuid.int)

    def force_uuid_int_equivalence(self, immigrant_uuid_int, resident_uuid_int):
        """Forces immigrant object to be treated as equivalent to (same as) resident object, identified by uuid ints."""

        assert immigrant_uuid_int is not None and resident_uuid_int is not None
        if immigrant_uuid_int == resident_uuid_int:
            return
        assert immigrant_uuid_int not in self.map.values()

        self.map[immigrant_uuid_int] = resident_uuid_int

    def force_part_equivalence(self, immigrant_part, resident_part):
        """Forces immigrant part to be treated as equivalent to resident part."""

        assert immigrant_part is not None and resident_part is not None
        if immigrant_part == resident_part:
            return
        self.force_uuid_int_equivalence(
            rqet.uuid_in_part_name(immigrant_part).int,
            rqet.uuid_in_part_name(resident_part).int)

    def check_map_integrity(self):
        """Raises assertion failure if map contains any potentially circular references."""

        for immigrant in self.map.keys():
            assert immigrant not in self.map.values()
        for resident in self.map.values():
            assert resident not in self.map.keys()


def _ordering(obj_type):
    if obj_type in consolidatable_list:
        return consolidatable_list.index(obj_type)
    seq = len(consolidatable_list)
    if obj_type.endswith('Interpretation'):
        seq += 1
    elif obj_type.endswith('Representation'):
        seq += 2
    elif obj_type.endswith('Property'):
        seq += 3
    return seq


def sort_parts_list(model, parts_list):
    """Returns a copy of the parts list sorted into the preferred order for consolidating."""

    pair_list = [(_ordering(model.type_of_part(part, strip_obj = True)), part) for part in parts_list]
    pair_list.sort()
    return [part for _, part in pair_list]


def sort_uuids_list(model, uuids_list):
    """Returns a copy of the uuids list (or uuid ints list) sorted into the preferred order for consolidating."""

    pair_list = [
        (_ordering(model.type_of_part(model.part_for_uuid(uuid), strip_obj = True)), uuid) for uuid in uuids_list
    ]
    pair_list.sort()
    return [uuid for _, uuid in pair_list]
