"""_add_surfaces.py: Module to import a list of surfaces into a RESQML model, as triangulatedsets or mesh objects."""

version = '15th November 2021'

# RMS and GOCAD are trademarks of Emerson

import logging

log = logging.getLogger(__name__)

import os

import resqpy.model as rq
# import resqpy.olio.grid_functions as gf
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.surface as rqs


def add_surfaces(
    epc_file,  # existing resqml model
    crs_uuid = None,  # optional crs uuid, defaults to crs associated with model (usually main grid crs)
    ext_uuid = None,  # if None, uuid for hdf5 file holding main grid geometry will be used
    surface_file_format = 'zmap',  # zmap, rms (roxar) or GOCAD-Tsurf only formats currently supported
    rq_class = 'surface',  # 'surface' or 'mesh': the class of object to be created
    surface_role = 'map',  # 'map' or 'pick'
    quad_triangles = False,  # if True, 4 triangles per quadrangle will be used for mesh formats, otherwise 2
    surface_file_list = None,  # list of full file names (paths), each holding one surface
    make_horizon_interpretations_and_features = True):  # if True, feature and interpretation objects are created
    """Process a list of surface files, adding each surface as a new part in the resqml model.

    Arguments:
        epc_file (str): file name and path to an existing resqml model
        crs_uuid (uuid.UUID, default None): uuid for a coordinate reference system. Defaults to crs associated with model (usually the main grid crs)
        ext_uuid (uuid.UUID, default None): uuid for h5 file. If None, uuid for hdf5 file holding main grid geometry will be used
        surface_file_format (str, default 'zmap'): 'zmap', 'rms', 'roxar' or 'GOCAD-Tsurf'. The format of the input file
        rq_class (str, default 'surface'): 'surface' or 'mesh'. The class of object ot be
        surface_role (str, default 'map'): 'map' or 'pick'
        quad_triangles (bool, default False): if True, 4 triangles per quadrangle will be used for mesh formats, otherwise 2
        surface_file_list (list, default None): list of full file names (paths), each holding one surface
        make_horizon_interpretations_and_features (bool, default True): if True, feature and interpretation objects are created

    Returns:
        resqml model object with added surfaces
    """

    assert surface_file_list, 'surface file list is empty or missing'
    assert surface_file_format in ['zmap', 'rms', 'roxar',
                                   'GOCAD-Tsurf'], 'unsupported surface file format: ' + str(surface_file_format)

    rq_class = _get_rq_class(rq_class)

    model, crs_uuid, h5_mode, ext_uuid, hdf5_file = _get_model_details(epc_file, crs_uuid, ext_uuid)

    for surf_file in surface_file_list:
        model = _add_single_surface(model, surf_file, surface_file_format, surface_role, quad_triangles, crs_uuid,
                                    rq_class, hdf5_file, h5_mode, make_horizon_interpretations_and_features, ext_uuid)

    # mark model as modified
    model.set_modified()

    # store new version of model
    log.info('storing model with additional parts in epc file: ' + epc_file)
    model.store_epc(epc_file)

    return model


def _add_single_surface(model, surf_file, surface_file_format, surface_role, quad_triangles, crs_uuid, rq_class,
                        hdf5_file, h5_mode, make_horizon_interpretations_and_features, ext_uuid):
    _, short_name = os.path.split(surf_file)
    dot = short_name.rfind('.')
    if dot > 0:
        short_name = short_name[:dot]

    log.info('surface ' + short_name + ' processing file: ' + surf_file + ' using format: ' + surface_file_format)
    if rq_class == 'surface':
        if surface_file_format == 'GOCAD-Tsurf':
            surface = rqs.Surface(model,
                                  tsurf_file = surf_file,
                                  surface_role = surface_role,
                                  quad_triangles = quad_triangles)
        else:
            surface = rqs.Surface(model,
                                  mesh_file = surf_file,
                                  mesh_format = surface_file_format,
                                  surface_role = surface_role,
                                  quad_triangles = quad_triangles)
    elif rq_class == 'mesh':
        if surface_file_format == 'GOCAD-Tsurf':
            log.info(f"Cannot convert a GOCAD-Tsurf to mesh, only to TriangulatedSurface - skipping file {surf_file}")
            return model
        else:
            surface = rqs.Mesh(model,
                               mesh_file = surf_file,
                               mesh_format = surface_file_format,
                               mesh_flavour = 'reg&z',
                               surface_role = surface_role,
                               crs_uuid = crs_uuid)
    else:
        log.critical('this is impossible')
    # NB. surface may be either a Surface object or a Mesh object

    log.debug('appending to hdf5 file for surface file: ' + surf_file)
    surface.write_hdf5(hdf5_file, mode = h5_mode)

    if make_horizon_interpretations_and_features:
        feature = rqo.GeneticBoundaryFeature(model, kind = 'horizon', feature_name = short_name)
        feature.create_xml()
        interp = rqo.HorizonInterpretation(model, genetic_boundary_feature = feature, domain = 'depth')
        interp_root = interp.create_xml()
        surface.set_represented_interpretation_root(interp_root)

    surface.create_xml(ext_uuid,
                       add_as_part = True,
                       add_relationships = True,
                       title = short_name + ' sourced from ' + surf_file,
                       originator = None)

    return model


def _get_rq_class(rq_class):
    if 'TriangulatedSet' in rq_class:
        rq_class = 'surface'
    elif 'Grid2d' in rq_class:
        rq_class = 'mesh'
    assert rq_class in ['surface', 'mesh']
    return rq_class


def _get_model_details(epc_file, crs_uuid, ext_uuid):
    log.info('accessing existing resqml model from: ' + epc_file)
    model = rq.Model(epc_file = epc_file)
    assert model, 'failed to read existing resqml model from file: ' + epc_file

    if crs_uuid is None:
        assert model.crs_uuid is not None, 'no crs uuid given and no default in model'
        crs_uuid = model.crs_uuid

    if ext_uuid is None:
        ext_uuid = model.h5_uuid()
    if ext_uuid is None:  # no pre-existing hdf5 part or references in model
        hdf5_file = epc_file[:-4] + '.h5'
        ext_node = model.create_hdf5_ext(file_name = hdf5_file)
        ext_uuid = rqet.uuid_for_part_root(ext_node)
        h5_mode = 'w'
    else:
        hdf5_file = model.h5_file_name(uuid = ext_uuid)
        h5_mode = 'a'

    assert ext_uuid is not None, 'failed to establish hdf5 uuid'

    # append to hdf5 file using arrays from Surface object's patch(es)
    log.info('will append to hdf5 file: ' + hdf5_file)

    return model, crs_uuid, h5_mode, ext_uuid, hdf5_file
