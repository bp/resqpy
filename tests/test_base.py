import resqpy.olio.uuid as bu
from resqpy.olio.base import BaseResqpy


class DummyObj(BaseResqpy):
    resqml_type = 'DummyResqmlInterpretation'


def test_base_creation(tmp_model):

    # Setup new object
    title = 'Wondermuffin'
    dummy = DummyObj(model=tmp_model, title=title)

    # UUID should exist, and part (name) should be determinable,
    # but root should not exist yet
    assert dummy.uuid is not None
    assert dummy.part is not None
    assert dummy.root is None

    # After creating XML, root should exist
    dummy.create_xml(add_as_part=True)
    assert dummy.root is not None


def test_base_save_and_load(tmp_model):

    # Create and save a DummyObj
    title = 'feefifofum'
    originator = 'Scruffian'
    metadata = {"balderdash": "pumpernickel"}
    dummy1 = DummyObj(model=tmp_model, title=title, originator=originator)
    dummy1.extra_metadata = metadata

    # Save to XML
    dummy1.create_xml(add_as_part=True)

    # Load a new object
    dummy2 = DummyObj(model=tmp_model, uuid=dummy1.uuid)

    # Properties should match
    assert bu.matching_uuids(dummy2.uuid, dummy1.uuid)
    assert dummy2.root is not None
    assert dummy2.title == title
    assert dummy2.originator == originator
    assert dummy2.extra_metadata == metadata


def test_base_comparison(tmp_model):

    # Setup new object
    dummy1 = DummyObj(model=tmp_model)
    dummy2 = DummyObj(model=tmp_model, uuid=dummy1.uuid)  # Same UUID
    dummy3 = DummyObj(model=tmp_model, title='hello')     # Different UUID

    # Comparison should work by UUID
    assert dummy1 == dummy2
    assert dummy1 != dummy3

    
def test_base_repr(tmp_model):

    dummy = DummyObj(model=tmp_model)

    # Check repr method produces a string
    rep = repr(dummy)
    assert isinstance(rep, str)
    assert len(rep) > 0

    # Check HTML can be generated
    html = dummy._repr_html_()
    assert len(html) > 0


def test_base_reuse_duplicate(tmp_model):

    # Create object and save
    dummy_1 = DummyObj(model=tmp_model)
    dummy_1.create_xml(add_as_part=True)

    # Should be present in model
    uuids = tmp_model.uuids(obj_type=DummyObj.resqml_type)
    assert len(uuids) == 1
    assert dummy_1.try_reuse() is False

    # Create duplicate object
    dummy_2 = DummyObj(model=tmp_model)
    dummy_2.uuid = dummy_1.uuid

    # Two should exist in model
    uuids = tmp_model.uuids(obj_type=DummyObj.resqml_type)
    assert len(uuids) == 2
    assert dummy_2.try_reuse() is True
