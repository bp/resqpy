import pytest

import resqpy.organize as rqo
from resqpy.model import Model


# Test saving and loading from disk
@pytest.mark.parametrize(
   "cls, data",
   [
      (
         rqo.OrganizationFeature,
         dict(feature_name = 'hello', organization_kind = 'stratigraphic'),
      ),
      (
         rqo.GeobodyFeature,
         dict(feature_name = 'hi'),
      ),
      (
         rqo.BoundaryFeature,
         dict(feature_name = 'foobar'),
      ),
      (
         rqo.FrontierFeature,
         dict(feature_name = 'foobar'),
      ),
      (
         rqo.GeologicUnitFeature,
         dict(feature_name = 'foobar'),
      ),
      (
         rqo.FluidBoundaryFeature,
         dict(feature_name = 'foobar', kind = 'gas oil contact'),
      ),
      (
         rqo.TectonicBoundaryFeature,
         dict(feature_name = 'foobar', kind = 'fracture'),
      ),
      (
         rqo.GeneticBoundaryFeature,
         dict(feature_name = 'foobar', kind = 'geobody boundary'),  # Age as well?
      ),
      (
         rqo.WellboreFeature,
         dict(feature_name = 'foobar'),
      ),
   ])
def test_organize_classes(tmp_model, cls, data):

   # Load example model from a fixture
   model = tmp_model
   epc = model.epc_file

   # Create the feature
   obj = cls(parent_model = model, **data)
   uuid = obj.uuid

   # Save to disk
   obj.create_xml()
   model.store_epc()
   model.h5_release()

   # Reload from disk
   del model, obj
   model2 = Model(epc_file = epc)
   obj2 = cls(parent_model = model2, uuid = uuid)

   # Check all attributes were loaded correctly
   for key, expected_value in data.items():
      assert getattr(obj2, key) == expected_value, f"Error for {key}"


def test_RockFluidUnitFeature(tmp_model):

   # Create the features
   top = rqo.BoundaryFeature(tmp_model, feature_name = 'the top')
   base = rqo.BoundaryFeature(tmp_model, feature_name = 'the base')
   rfuf_1 = rqo.RockFluidUnitFeature(parent_model = tmp_model,
                                     feature_name = 'foobar',
                                     phase = 'seal',
                                     top_boundary_feature = top,
                                     base_boundary_feature = base)
   uuid = rfuf_1.uuid

   # Save to disk
   top.create_xml()
   base.create_xml()
   rfuf_1.create_xml()
   tmp_model.store_epc()

   # Reload from disk
   rfuf_2 = rqo.RockFluidUnitFeature(parent_model = tmp_model, uuid = uuid)

   # Check properties the same
   assert rfuf_2.feature_name == 'foobar'
   assert rfuf_2.phase == 'seal'
   assert rfuf_2.top_boundary_feature.feature_name == 'the top'
   assert rfuf_2.base_boundary_feature.feature_name == 'the base'


def test_FaultInterp(tmp_model):

   title = "not my fault"
   tect_boundary = rqo.TectonicBoundaryFeature(tmp_model, kind = 'fault')
   fault_interp = rqo.FaultInterpretation(
      tmp_model,
      tectonic_boundary_feature = tect_boundary,
      title = title,
      domain = "depth",
      is_normal = True,
      maximum_throw = 3,
   )

   tect_boundary.create_xml()
   fault_interp.create_xml()

   fault_interp_2 = rqo.FaultInterpretation(tmp_model, uuid = fault_interp.uuid)
   assert fault_interp_2.title == title
   assert fault_interp_2.maximum_throw == 3


def test_EarthModel(tmp_model):

   title = 'gaia'
   org_feat = rqo.OrganizationFeature(tmp_model, feature_name = 'marie kondo', organization_kind = "earth model")
   em1 = rqo.EarthModelInterpretation(tmp_model, title = title, organization_feature = org_feat)

   org_feat.create_xml()
   em1.create_xml()

   em2 = rqo.EarthModelInterpretation(tmp_model, uuid = em1.uuid)
   assert em2.title == title


def test_Horizon(tmp_model):

   gen = rqo.GeneticBoundaryFeature(tmp_model, kind = 'horizon')
   hor = rqo.HorizonInterpretation(tmp_model,
                                   genetic_boundary_feature = gen,
                                   sequence_stratigraphy_surface = 'maximum flooding')

   gen.create_xml()
   hor.create_xml()

   hor2 = rqo.HorizonInterpretation(tmp_model, uuid = hor.uuid)
   assert hor2.sequence_stratigraphy_surface == hor.sequence_stratigraphy_surface


def test_GeobodyBoundary(tmp_model):

   gen = rqo.GeneticBoundaryFeature(tmp_model, kind = 'geobody boundary')
   gb = rqo.GeobodyBoundaryInterpretation(tmp_model, genetic_boundary_feature = gen)

   gen.create_xml()
   gb.create_xml()
   gb2 = rqo.GeobodyBoundaryInterpretation(tmp_model, uuid = gb.uuid)
   assert gb == gb2


def test_wellbore_interp_title(tmp_model):
   # Create a feature and interp objects
   feature_name = 'well A'
   well_feature = rqo.WellboreFeature(tmp_model, feature_name = feature_name)
   well_feature.create_xml()
   well_interp_1 = rqo.WellboreInterpretation(tmp_model, wellbore_feature = well_feature, is_drilled = True)
   well_interp_1.create_xml()

   # Create a duplicate object, loading from XML
   well_interp_2 = rqo.WellboreInterpretation(tmp_model, uuid = well_interp_1.uuid)

   # Check feature name is present
   assert well_interp_1.title == feature_name
   assert well_interp_2.title == feature_name
