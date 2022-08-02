"""uuid.py: Thin wrapper around python uuid (universally unique identifier) module."""

# NB: at present the code does not enforce multiprocessor safe generation of unique identifiers
# it calls uuid.uuid1() to generate new uuids, ie. using version 1 of the iso standard options

version = '25th August 2021'

# import logging
# log = logging.getLogger(__name__)

import uuid

test_mode = False
test_latest_int = 0

max_version_string_length = 10


def switch_on_test_mode(seed = 0):
    """Causes subsequent calls to new_uid() to produce integer sequence starting from successor to seed.

    arguments:
       seed (integer, default 0): The predecessor to the first uuid returned by subsequent calls to
          new_uuid()

    returns:
       None

    notes:
       call switch_off_test_mode() to reactivate normal behaviour;
       uuids generated whilst in test mode do not adhere to the iso standard;
       test mode is intended to allow replicatable behaviour for testing purposes
    """

    global test_mode
    global test_latest_int
    test_latest_int = seed
    test_mode = True


def switch_off_test_mode():
    """Subsequent calls to new_uid() will produce standard uuid values (default behaviour).

    note:
       this function will have no effect unless switch_on_test_mode() has previously been called
    """

    global test_mode
    test_mode = False


def new_uuid():
    """Returns a new uuid based on the time (to 100ns) & MAC address option of the iso standard.

    returns:
       uuid.UUID object

    notes:
       at present, the multi-processor safe functionality is not deployed, so multiple processors
       sharing the same MAC address could generate the same uuid simultaneously;
       an integer sequence is generated when in test mode
    """

    global test_latest_int
    if test_mode:
        test_latest_int += 1
        return uuid.UUID(bytes = test_latest_int.to_bytes(16, byte_order = 'big'))
    else:
        return uuid.uuid1()  # time to 100ns & MAC address


def string_from_uuid(uuid_obj):
    """Returns standard hexadecimal string for uuid; same as str(uuid_obj).

    arguments:
       uuid_obj (uuid.UUID object): the uuid which is required in hexadecimal string format

    returns:
       string (40 characters: 36 lowercase hexadecimals and 4 hyphens)
    """

    return str(uuid_obj)


def uuid_from_string(uuid_str):
    """Returns a uuid object for the given uuid string; hyphens are ignored.

    arguments:
       uuid_str (string): the hexadecimal representation of the 128 bit uuid integer;
          hyphens are ignored

    returns:
       uuid.UUID object

    notes:
       if a uuid.UUID object is passed by accident, it is returned;
       if the string starts with an underscore, the underscore is skipped (to cater for a fesapi quirk);
       any tail beyond the uuid string is ignored
    """

    if uuid_str is None:
        return None
    if isinstance(uuid_str, uuid.UUID):
        return uuid_str  # resilience to accidentally passing a uuid object
    try:
        if uuid_str[0] == '_':  # tolerate one of the fesapi quirks
            if len(uuid_str) < 37:
                return None
            return uuid.UUID(uuid_str[1:37])
        else:
            if len(uuid_str) < 36:
                return None
            return uuid.UUID(uuid_str[:36])
    except Exception:
        # could log or raise an error or warning?
        return None


def uuid_as_bytes(uuid_obj):
    """Returns the uuid as a 16 byte bytes sequence; same as uuid_obj.bytes.

    arguments:
       uuid_obj (uuid.UUID object): the uuid for which a bytes representation is required

    returns:
       bytes (16 bytes long)
    """

    if uuid_obj is None:
        return None
    if isinstance(uuid_obj, str):
        uuid_obj = uuid_from_string(uuid_obj)  # resilience to accidental string arg
    assert isinstance(uuid_obj, uuid.UUID)
    return uuid_obj.bytes


def uuid_as_int(uuid_obj):
    """Returns the uuid as a 128 bit int; same as uuid_obj.int.

    arguments:
       uuid_obj (uuid.UUID object): the uuid for which a bytes representation is required

    returns:
       bytes (16 bytes long)
    """

    if uuid_obj is None:
        return None
    if isinstance(uuid_obj, str):
        uuid_obj = uuid_from_string(uuid_obj)  # resilience to accidental string arg
    if not isinstance(uuid_obj, uuid.UUID):
        raise ValueError(f'non uuid object where uuid expected: {uuid_obj}; type: {type(uuid_obj)}')
    return uuid_obj.int


def matching_uuids(uuid_a, uuid_b):
    """Returns True if the 2 uuid objects are for the same id; False otherwise.

    arguments:
       uuid_a, uuid_b (uuid.UUID objects): the two uuids to be compared

    returns:
       boolean: True if the two uuids are the same; False otherwise

    note:
       this function is resilient to uuids being passed in hexadecimal string format
    """

    if isinstance(uuid_a, str):
        uuid_a = uuid_from_string(uuid_a)  # resilience to accidental string arg
    if isinstance(uuid_b, str):
        uuid_b = uuid_from_string(uuid_b)
    if uuid_a is None or uuid_b is None:
        return False
    return uuid_a.int == uuid_b.int


def version_string(uuid_obj):
    """Returns an integer string rendering of the time element of the uuid.

    arguments:
       uuid_obj (uuid.UUID): the uuid for which a string representation of the time component is required

    returns:
       string (of digits)

    notes:
       this function has nothing to do with the uuid.version attribute, it is used to populate the version
       field of a resqml citation block;
       the time component of the uuid is the number of 100ns periods that have elapsed since October 1582
       (when the Gregorian calendar was adopted), as a 60 bit integer
    """

    if isinstance(uuid_obj, str):
        uuid_obj = uuid_from_string(uuid_obj)  # resilience to accidental string arg
    v_str = str(uuid_obj.time)
    if len(v_str) > max_version_string_length:
        v_str = v_str[:max_version_string_length]
    return v_str


def is_uuid(uuid_obj):
    """Returns boolean indicating whether uuid_obj seems to be a uuid."""

    if isinstance(uuid_obj, uuid.UUID):
        return True
    if not uuid_obj or not isinstance(uuid_obj, str):
        return False
    if uuid_obj[0] == '_':
        return len(uuid_obj) == 37
    return len(uuid_obj) == 36
    # could also check for hyphens in correct places, and hexadecimals only
