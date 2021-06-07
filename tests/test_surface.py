import resqpy.surface
import resqpy.organize


def test_surface(example_model):

    # Set up a Surface
    title = 'Mountbatten'
    model, crs = example_model
    surf = resqpy.surface.Surface(
        parent_model=model, extract_from_xml=False, title=title
    )
    surf.create_xml()

    # Add a interpretation
    assert surf.represented_interpretation_root is None
    surf.create_interpretation_and_feature(kind='fault')
    assert surf.represented_interpretation_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    fault_interp = resqpy.organize.FaultInterpretation(
        model, root_node=surf.represented_interpretation_root
    )
    fault_feature = resqpy.organize.TectonicBoundaryFeature(
        model, root_node=fault_interp.feature_root
    )

    # Check title matches expected title
    assert fault_feature.feature_name == title
