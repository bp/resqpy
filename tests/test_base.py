from resqpy.olio.base import BaseResqpy

class DummyObj(BaseResqpy):
    _content_type = 'DummyResqmlInterpretation'


def test_base_creation(example_model):

    # Setup new object
    title = 'Wondermuffin'
    model, crs = example_model
    dummy = DummyObj(model=model, title=title)

    # UUID should exist, but not root or part
    assert dummy.uuid is not None
    assert dummy.root is None
    assert dummy.part is None

    # After creating XML, root and part should exist
    dummy.create_xml(add_as_part=True)
    assert dummy.root is not None
    assert dummy.part is not None


def test_base_save_and_load(example_model):

    # Create and save a DummyObj
    title = 'feefifofum'
    originator = 'Scruffian'
    model, crs = example_model
    dummy1 = DummyObj(model=model, title=title, originator=originator)
    dummy1.create_xml()

    # Load a new object
    dummy2 = DummyObj(model=model, uuid=dummy1.uuid)

    # Properties should match
    assert dummy2.uuid == dummy1.uuid
    assert dummy2.root is not None
    assert dummy2.title == title
    assert dummy2.originator == originator


def test_base_comparison(example_model):

    # Setup new object
    model, crs = example_model
    dummy1 = DummyObj(model=model)
    dummy2 = DummyObj(model=model, uuid=dummy1.uuid)  # Same UUID
    dummy3 = DummyObj(model=model, title='hello')     # Different UUID

    # Comparison should work by UUID
    assert dummy1 == dummy2
    assert dummy1 != dummy3

    
def test_base_repr(example_model):

    model, crs = example_model
    dummy = DummyObj(model=model)

    # Check repr method produces a string
    rep = repr(dummy)
    assert isinstance(rep, str)
    assert len(rep) > 0

    # Check HTML can be generated
    html = dummy._repr_html_()
    assert len(html) > 0
