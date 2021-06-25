import resqpy.surface
import resqpy.organize


def test_surface(tmp_model):

    # Set up a Surface
    title = 'Mountbatten'
    model = tmp_model
    surf = resqpy.surface.Surface(
        parent_model=model, title=title
    )
    surf.create_xml()

    # Add a interpretation
    assert surf.represented_interpretation_root is None
    surf.create_interpretation_and_feature(kind='fault')
    assert surf.represented_interpretation_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    fault_interp = resqpy.organize.FaultInterpretation(
        model, uuid=surf.represented_interpretation_uuid
    )
    fault_feature = resqpy.organize.TectonicBoundaryFeature(
        model, uuid=fault_interp.tectonic_boundary_feature.uuid
    )

    # Check title matches expected title
    assert fault_feature.feature_name == title
