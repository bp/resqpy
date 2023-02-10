# test functions for olio.uuid module

import resqpy.olio.uuid as bu


def test_new_uuid():
    u1 = bu.new_uuid()
    u2 = bu.new_uuid()
    assert bu.is_uuid(u1) and bu.is_uuid(u2)
    assert not bu.matching_uuids(u1, u2)


def test_test_mode():
    # note: test failure between on and off calls is likely to result in later tests failing
    u_a = bu.new_uuid()
    bu.switch_on_test_mode(seed = 11)
    u12 = bu.new_uuid()
    u13 = bu.new_uuid()
    bu.switch_off_test_mode()
    u_b = bu.new_uuid()
    assert u_a.int != 11
    assert u12.int == 12
    assert u13.int == 13
    assert u_b.int != 14


def test_string_to_from_uuid():
    u = bu.new_uuid()
    s = bu.string_from_uuid(u)
    u2 = bu.uuid_from_string(s)
    assert isinstance(s, str)
    assert len(s) == 36
    assert all([s[i] == '-' for i in (8, 13, 18, 23)])
    assert all([ch in '0123456789abcdef-' for ch in s])
    assert bu.is_uuid(s)
    assert bu.matching_uuids(s, u)
    assert bu.matching_uuids(u, s)
    assert bu.is_uuid(u2)
    assert u2.int == u.int
    assert bu.matching_uuids(u2, u)
    assert u2 == u


def test_int_to_from_uuid():
    u = bu.new_uuid()
    i = bu.uuid_as_int(u)
    u2 = bu.uuid_from_int(i)
    assert isinstance(i, int)
    assert i > 0
    assert u2 == u
    assert i == u.int


def test_uuid_as_bytes():
    ub = bu.uuid_as_bytes(bu.new_uuid())
    assert isinstance(ub, bytes)
    assert len(ub) == 16


def test_uuid_in_list():
    u = bu.new_uuid()
    u1 = bu.new_uuid()
    u2 = bu.new_uuid()
    u3 = bu.new_uuid()
    not_list = [u1, u2, u3]
    u_list = [u1, u2, u, u3]
    s_list = (str(u1), str(u3), str(u))
    us = str(u)
    ui = u.int
    assert not bu.uuid_in_list(u, not_list)
    assert not bu.uuid_in_list(us, not_list)
    for uuid_list in [u_list, s_list]:
        for uuid in (u, us, ui):
            assert bu.uuid_in_list(uuid, uuid_list)
