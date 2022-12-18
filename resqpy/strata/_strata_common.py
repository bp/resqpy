"""Common functions and valid xml constant values for stratigraphy related RESQML classes."""

# note: two compositions have a spurious trailing space in the RESQML xsd; resqpy hides this from calling code
valid_compositions = [
    'intrusive clay ', 'intrusive clay', 'organic', 'intrusive mud ', 'intrusive mud', 'evaporite salt',
    'evaporite non salt', 'sedimentary siliclastic', 'carbonate', 'magmatic intrusive granitoid',
    'magmatic intrusive pyroclastic', 'magmatic extrusive lava flow', 'other chemichal rock', 'sedimentary turbidite'
]

valid_implacements = ['autochtonous', 'allochtonous']

valid_domains = ('depth', 'time', 'mixed')

valid_deposition_modes = [
    'proportional between top and bottom', 'parallel to bottom', 'parallel to top', 'parallel to another boundary'
]

valid_ordering_criteria = ['age', 'apparent depth', 'measured depth']  # stratigraphic column must be ordered by age

valid_contact_relationships = [
    'frontier feature to frontier feature', 'genetic boundary to frontier feature',
    'genetic boundary to genetic boundary', 'genetic boundary to tectonic boundary',
    'stratigraphic unit to frontier feature', 'stratigraphic unit to stratigraphic unit',
    'tectonic boundary to frontier feature', 'tectonic boundary to genetic boundary',
    'tectonic boundary to tectonic boundary'
]

valid_contact_verbs = ['splits', 'interrupts', 'contains', 'erodes', 'stops at', 'crosses', 'includes']

valid_contact_sides = ['footwall', 'hanging wall', 'north', 'south', 'east', 'west', 'younger', 'older', 'both']

valid_contact_modes = ['baselap', 'erosion', 'extended', 'proportional']


def _index_attr(obj):
    """Returns the index attribute of any object – typically used as a sort key function."""
    return obj.index
