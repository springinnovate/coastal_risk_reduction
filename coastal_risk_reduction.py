# TODO: get the smallest example set up to run

"""Global CV Analysis."""
import glob
import argparse
import bisect
import collections
import configparser
import logging
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import types
from numbers import Number

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from shapely.geometry import MultiPoint, GeometryCollection
from shapely.ops import voronoi_diagram
import shapely.ops
import numpy
import retrying
import shapely.errors
import shapely.geometry
import shapely.strtree
import shapely.wkt

gdal.SetCacheMax(2**20)
ogr.UseExceptions()

GLOBAL_INI_PATH = 'global_config.ini'
_VALUE_COVER_NODATA = 0

# These are globally defined keys expected in global_config.ini
HABITAT_MAP_KEY = 'habitat_map'
BENEFICIARIES_MAP_KEY = 'beneficiaries'
SHORE_POINT_SAMPLE_DISTANCE_KEY = 'shore_point_sample_distance'
LULC_CODE_TO_HAB_MAP_KEY = 'lulc_code_to_hab_map'

MAX_WORKERS = 24

TARGET_NODATA = -1

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

STOP_SENTINEL = 'STOP'


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, daemon=None):
        threading.Thread.__init__(
            self, group=group, target=target, name=name, args=args,
            kwargs=kwargs, daemon=daemon)
        self._return = None

    def run(self):
        try:
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)
        except Exception:
            LOGGER.exception('exception when running thread')
            raise

    def join(self, *args):
        try:
            threading.Thread.join(self, *args)
            return self._return
        except Exception:
            LOGGER.exception('exception when joining thread')
            raise


def convolve_2d(
            signal_path, kernel_radius, temp_workspace_dir,
            target_path):
    base_id = os.path.basename(os.path.splitext(signal_path)[0])
    kernel_filepath = os.path.join(
        temp_workspace_dir, f'{base_id}_kernel.tif')
    LOGGER.debug(f'*********** kernel_radius {kernel_radius}')
    create_averaging_kernel_raster(
        kernel_radius, kernel_filepath, normalize=True)
    geoprocessing.convolve_2d(
        (signal_path, 1), (kernel_filepath, 1),
        target_path, mask_nodata=False)
    os.remove(kernel_filepath)


def build_strtree(
        vector_path, fields_to_copy=None):
    """Build an rtree that generates geom and preped geometry.

    Parameters:
        vector_path (str): path to vector of geometry to build into
            r tree.
        fields_to_copy (list): if not None, only copy fields that contain
            these substrings

    Returns:
        strtree.STRtree object that will return indexes to object list
            based on spatial queries.
        feature_object_list, indexed by same index of STR query indexing
            into object list of objects
            with a .prep field that is prepared geomtry for fast testing,
            a .geom field that is the base gdal geometry, and a field_val_map
            field that contains the 'fieldname'->value pairs from the original
            vector. The main object will also have a `field_name_type_list`
            field which contains original fieldname/field type pairs

    """
    try:
        start_time = time.time()
        geometry_prep_list = []
        vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_name_type_list = []
        if fields_to_copy is not None:
            for index in range(layer_defn.GetFieldCount()):
                field_name = layer_defn.GetFieldDefn(index).GetName()
                if any([x in field_name for x in fields_to_copy]):
                    field_type = layer_defn.GetFieldDefn(index).GetType()
                    field_name_type_list.append((field_name, field_type))

        LOGGER.debug(f'loop through features for rtree {vector_path}')
        object_list = []
        n_features = layer.GetFeatureCount()
        for index, feature in enumerate(layer):
            if index % 1000 == 0:
                LOGGER.debug(
                    f'{index/n_features*100:.1f}% complete '
                    f'{index+1}/{n_features}: {vector_path}')
            feature_geom = feature.GetGeometryRef()
            feature_geom_shapely = shapely.wkb.loads(
                bytes(feature_geom.ExportToWkb()))
            feature_geom = None
            feature_object = types.SimpleNamespace()
            feature_object.prep = shapely.prepared.prep(feature_geom_shapely)
            feature_object.geom = feature_geom_shapely
            feature_object.id = index
            if fields_to_copy is not None:
                feature_object.field_val_map = {
                    field_name: feature.GetField(field_name)
                    for field_name, _ in field_name_type_list
                }
            geometry_prep_list.append(feature_geom_shapely)
            object_list.append(feature_object)
        LOGGER.debug(f'constructing the tree for {vector_path}')
        str_tree = shapely.strtree.STRtree(geometry_prep_list)
        LOGGER.debug(
            f'constrcuted tree for {vector_path} in '
            f'{time.time()-start_time:.2f}s')
        if fields_to_copy is not None:
            str_tree.field_name_type_list = field_name_type_list
        return str_tree, object_list
    except Exception:
        LOGGER.exception(
            f'something bad happened on building STR tree for {vector_path}')
        raise
    finally:
        LOGGER.info(f'all done for STRtree {vector_path}')


def cv_grid_worker(
        bb_work_queue,
        cv_point_complete_queue,
        local_data_path_map,
        grid_workspace_dir,
        ):
    """Worker process to calculate CV for a grid.

    Parameters:
        bb_work_queue (multiprocessing.Queue): contains
            [minx, miny, maxx, maxy] bounding box values to be processed or
            `STOP_SENTINEL` values to indicate the worker should be terminated.
        cv_point_complete_queue (multiprocessing.Queue): this queue is used to
            pass completed CV point vectors for further processing. It will be
            terminated by `STOP_SENTINEL`.
        local_data_path_map (dict): maps necessary habitat and biophysical
            layers to actual files and data.
        grid_workspace_dir (str): path to workspace to use to handle grid
        shore_point_sample_distance (float): straight line distance between
            shore sample points.
    Returns:
        None.

    """
    try:
        LOGGER.info('build geomorphology, landmass, and wwiii lookup')

        geomorphology_strtree_thread = ThreadWithReturnValue(
            target=build_strtree,
            args=(local_data_path_map['geomorphology_vector_path'], ['Rgeo']))
        landmass_strtree_thread = ThreadWithReturnValue(
            target=build_strtree,
            args=(local_data_path_map['landmass_vector_path'],))
        wwiii_strtree_thread = ThreadWithReturnValue(
            target=build_strtree,
            args=(local_data_path_map['wwiii_vector_path'],
                  ['REI_PCT', 'REI_V', 'V10PCT_', 'REI_PCT', 'WavP_',
                   'WavPPCT']))

        geomorphology_strtree_thread.start()
        landmass_strtree_thread.start()
        wwiii_strtree_thread.start()

        LOGGER.debug('waiting for geomorphology_strtree to finish')
        geomorphology_strtree, geomorphology_object_list = \
            geomorphology_strtree_thread.join()
        LOGGER.debug('waiting for landmass_strtree to finish')
        landmass_strtree, landmass_object_list = landmass_strtree_thread.join()
        LOGGER.debug('waiting for wwiii_strtree to finish')
        wwiii_strtree, wwiii_object_list = wwiii_strtree_thread.join()
        LOGGER.debug('done with all three spatial indexes being built')

        geomorphology_proj_wkt = geoprocessing.get_vector_info(
            local_data_path_map['geomorphology_vector_path'])['projection_wkt']
        gegeomorphology_proj = osr.SpatialReference()
        gegeomorphology_proj.ImportFromWkt(geomorphology_proj_wkt)

        shore_point_sample_distance = float(local_data_path_map[
            SHORE_POINT_SAMPLE_DISTANCE_KEY])
        target_pixel_size = [
            shore_point_sample_distance / 4, -shore_point_sample_distance / 4]

        LOGGER.debug(f'******************** {local_data_path_map[HABITAT_MAP_KEY]}')
        max_sample_distance = max([
            v[1] for v in eval(local_data_path_map[HABITAT_MAP_KEY]).values()])

        max_sample_distance = max([max_sample_distance] + [
            0 if not isinstance(v, tuple) else v[2]
            for v in eval(local_data_path_map[LULC_CODE_TO_HAB_MAP_KEY]).values()
            ])

        risk_distance_lucode_map = _parse_lulc_code_to_hab(eval(
            local_data_path_map['lulc_code_to_hab_map']))
        LOGGER.debug(f'risk_distance_lucode_map: {risk_distance_lucode_map}')

        risk_dist_raster_map = _parse_habitat_map(eval(
            local_data_path_map['habitat_map']))
        LOGGER.debug(f'risk_dist_raster_map: {risk_dist_raster_map}')

        while True:
            try:
                LOGGER.info('waiting for work payload')
                payload = bb_work_queue.get()
                LOGGER.info(f'processing payload {payload}')
                if payload == STOP_SENTINEL:
                    LOGGER.debug('stopping')
                    # put it back so others can stop
                    bb_work_queue.put(STOP_SENTINEL)
                    break
                else:
                    LOGGER.debug('running')
                # otherwise payload is the bounding box
                index, (lng_min, lat_min, lng_max, lat_max) = payload
                bounding_box_list = [lng_min, lat_min, lng_max, lat_max]
                buffered_bounding_box_list = [
                    lng_min-0.1, lat_min-0.1, lng_max+0.1, lat_max+0.1]
                # create workspace
                workspace_dir = os.path.join(
                    grid_workspace_dir, '%d_%s_%s_%s_%s' % (
                        index, lng_min, lat_min, lng_max, lat_max))
                os.makedirs(workspace_dir, exist_ok=True)

                utm_srs = calculate_utm_srs(
                    (lng_min+lng_max)/2, (lat_min+lat_max)/2)
                wgs84_srs = osr.SpatialReference()
                wgs84_srs.ImportFromWkt(osr.SRS_WKT_WGS84_LAT_LONG)

                local_bounding_box = geoprocessing.transform_bounding_box(
                    buffered_bounding_box_list, osr.SRS_WKT_WGS84_LAT_LONG,
                    utm_srs.ExportToWkt(), edge_samples=11)

                # calculate (hab_id, risk, dist) -> raster mask tuples given
                # lulc map and a lookup of lulc to risk tuples
                habitat_raster_path_map = clip_and_mask_habitat(
                    risk_distance_lucode_map,
                    local_data_path_map['lulc_raster_path'],
                    risk_dist_raster_map, local_bounding_box,
                    utm_srs.ExportToWkt(), target_pixel_size, workspace_dir)

                LOGGER.debug(habitat_raster_path_map)

                local_geomorphology_vector_path = os.path.join(
                    workspace_dir, 'geomorphology.gpkg')
                LOGGER.debug(f'git geomorphology for {payload}')
                clip_geometry(
                    bounding_box_list, wgs84_srs, utm_srs,
                    ogr.wkbMultiLineString, geomorphology_strtree,
                    geomorphology_object_list,
                    local_geomorphology_vector_path)

                shore_point_vector_path = os.path.join(
                    workspace_dir, 'shore_points.gpkg')
                LOGGER.debug(f'sample geomorphology to points for {payload}')
                sample_line_to_points(
                    local_geomorphology_vector_path, shore_point_vector_path,
                    shore_point_sample_distance)

                local_landmass_vector_path = os.path.join(
                    workspace_dir, 'landmass.gpkg')
                LOGGER.debug(f'clip landmass for {payload}')
                clip_geometry(
                    buffered_bounding_box_list, wgs84_srs, utm_srs,
                    ogr.wkbPolygon, landmass_strtree,
                    landmass_object_list,
                    local_landmass_vector_path)

                landmass_boundary_vector_path = os.path.join(
                    workspace_dir, 'landmass_boundary.gpkg')
                LOGGER.debug(f'landmass to lines for {payload}')
                vector_to_lines(
                    local_landmass_vector_path, landmass_boundary_vector_path)

                local_dem_path = os.path.join(
                    workspace_dir, 'dem.tif')
                LOGGER.debug(f'clip and reproject dem {payload}')
                clip_and_reproject_raster(
                    local_data_path_map['dem_raster_path'], local_dem_path,
                    utm_srs.ExportToWkt(), bounding_box_list,
                    float(local_data_path_map['relief_sample_distance']),
                    'bilinear', True, target_pixel_size)

                local_slr_path = os.path.join(
                    workspace_dir, 'slr.tif')
                LOGGER.debug(f'clip and reproject slr {payload}')
                clip_and_reproject_raster(
                    local_data_path_map['slr_raster_path'], local_slr_path,
                    utm_srs.ExportToWkt(), bounding_box_list, 0, 'bilinear',
                    True, target_pixel_size)

                # Rrelief
                LOGGER.info(f'calculate relief on {payload}')
                calculate_relief(
                    shore_point_vector_path,
                    float(local_data_path_map['relief_sample_distance']),
                    local_dem_path, 'relief')
                LOGGER.info(f'calculate rhab on {payload}')
                # Rhab
                calculate_rhab(
                    shore_point_vector_path, habitat_raster_path_map, 'Rhab',
                    target_pixel_size)

                # Rslr
                LOGGER.debug(f'calculate slr {payload}')
                calculate_slr(shore_point_vector_path, local_slr_path, 'slr')

                # wind and wave power
                LOGGER.debug(f'calculate wind and wave on {payload}')
                calculate_wind_and_wave(
                    shore_point_vector_path,
                    shore_point_sample_distance,
                    local_landmass_vector_path,
                    landmass_boundary_vector_path,
                    local_dem_path, wwiii_strtree,
                    wwiii_object_list,
                    int(local_data_path_map['n_fetch_rays']),
                    float(local_data_path_map['max_fetch_distance']),
                    'rei', 'ew')

                # Rsurge
                LOGGER.debug(f'calculate rsurge on {payload}')
                calculate_surge(
                    shore_point_vector_path, local_dem_path,
                    float(local_data_path_map['max_fetch_distance']), 'surge')

                # Rgeomorphology
                LOGGER.debug(f'calculate geomorphology on {payload}')
                calculate_geomorphology(
                    shore_point_vector_path, local_geomorphology_vector_path,
                    float(local_data_path_map['max_fetch_distance']),
                    'Rgeomorphology')

                LOGGER.info('completed %s', shore_point_vector_path)
                cv_point_complete_queue.put(shore_point_vector_path)

            except Exception:
                LOGGER.warning(
                    f'error on {payload}, removing workspace, skipping')
                retrying_rmtree(workspace_dir)
    except Exception:
        LOGGER.exception('something horrible happened on')
        raise


def make_shore_kernel(kernel_path):
    """Make a 3x3 raster with a 9 in the middle and 1s on the outside."""
    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_path.encode('utf-8'), 3, 3, 1,
        gdal.GDT_Byte)

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    kernel_raster.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(127)
    kernel_band.WriteArray(numpy.array([[1, 1, 1], [1, 9, 1], [1, 1, 1]]))


def calculate_geomorphology(
        shore_point_vector_path, geomorphology_vector_path,
        max_fetch_distance,
        geomorphology_fieldname):
    """Sample the geomorphology vector path for the closest line to each point.

    Parameters:
        shore_point_vector_path (str):  path to a point shapefile to
            for relief point analysis.
        geomorphology_vector_path (str): path to a vector of lines that
            contains the integer field 'SEDTYPE'.
        max_fetch_distance (float): maximum distance to search for
            geomporphology risk.
        geomorphology_fieldname (str): fieldname to add to
            `shore_point_vector_path`.

    Returns:
        None.

    """
    shore_point_vector = gdal.OpenEx(
        shore_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    shore_point_layer = shore_point_vector.GetLayer()
    shore_point_layer.CreateField(
        ogr.FieldDefn(geomorphology_fieldname, ogr.OFTReal))
    geomorphology_strtree, geomporphology_object_list = build_strtree(
        geomorphology_vector_path, ['Rgeo'])

    shore_point_layer.StartTransaction()
    for shore_point_feature in shore_point_layer:
        shore_point_geom = shapely.wkb.loads(
            bytes(shore_point_feature.GetGeometryRef().ExportToWkb()))
        min_dist = max_fetch_distance
        geo_risk = 5
        for line_index in geomorphology_strtree.query(
                shore_point_geom.buffer(500)):
            line = geomporphology_object_list[line_index]
            cur_dist = line.geom.distance(shore_point_geom)
            if cur_dist < min_dist:
                min_dist = cur_dist
                geo_risk = line.field_val_map['Rgeo']
        shore_point_feature.SetField(geomorphology_fieldname, geo_risk)
        shore_point_layer.SetFeature(shore_point_feature)
    shore_point_layer.CommitTransaction()
    shore_point_layer = None
    shore_point_vector = None


def calculate_surge(
        shore_point_vector_path, bathymetry_raster_path,
        max_fetch_distance, surge_fieldname):
    """Calculate surge potential as distance to continental shelf (-150m).

    Parameters:
        base_shore_point_vector_path (string):  path to a point shapefile to
            for relief point analysis.
        global_dem_path (string): path to a DEM raster projected in wgs84.
        surge_fieldname (str): fieldname to add to `shore_point_vector_path`
        max_fetch_distance (float): maximum distance to send a ray to determine
            surge risk
        target_surge_point_vector_path (string): path to output vector.
            after completion will a value for closest distance to continental
            shelf called 'surge'.

    Returns:
        None.

    """
    shore_point_vector = gdal.OpenEx(
        shore_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    shore_point_layer = shore_point_vector.GetLayer()
    shore_point_layer.CreateField(ogr.FieldDefn(surge_fieldname, ogr.OFTReal))

    shelf_nodata = 2

    bathymetry_nodata = geoprocessing.get_raster_info(
        bathymetry_raster_path)['nodata'][0]

    def mask_shelf(depth_array):
        valid_mask = ~numpy.isclose(depth_array, bathymetry_nodata)
        result_array = numpy.empty(
            depth_array.shape, dtype=numpy.int16)
        result_array[:] = shelf_nodata
        result_array[valid_mask] = 0
        result_array[depth_array < -150] = 1
        return result_array

    workspace_dir = os.path.dirname(shore_point_vector_path)

    shelf_mask_path = os.path.join(workspace_dir, 'shelf_mask.tif')
    geoprocessing.raster_calculator(
        [(bathymetry_raster_path, 1)], mask_shelf,
        shelf_mask_path, gdal.GDT_Byte, shelf_nodata)

    # convolve to find edges
    # grid shoreline from raster
    shelf_kernel_path = os.path.join(workspace_dir, 'shelf_kernel.tif')
    shelf_convoultion_raster_path = os.path.join(
        workspace_dir, 'shelf_convolution.tif')
    make_shore_kernel(shelf_kernel_path)
    geoprocessing.convolve_2d(
        (shelf_mask_path, 1), (shelf_kernel_path, 1),
        shelf_convoultion_raster_path, target_datatype=gdal.GDT_Byte,
        target_nodata=255)

    nodata = geoprocessing.get_raster_info(
        shelf_convoultion_raster_path)['nodata'][0]

    def _shelf_mask_op(shelf_convolution):
        """Mask values on land that border the continental shelf."""
        result = numpy.empty(shelf_convolution.shape, dtype=numpy.uint8)
        result[:] = nodata
        valid_mask = shelf_convolution != nodata
        # If a pixel is on land, it gets at least a 9, but if it's all on
        # land it gets an 17 (8 neighboring pixels), so we search between 9
        # and 17 to determine a shore pixel
        result[valid_mask] = numpy.where(
            (shelf_convolution[valid_mask] >= 9) &
            (shelf_convolution[valid_mask] < 17), 1, nodata)
        return result

    shelf_edge_raster_path = os.path.join(workspace_dir, 'shelf_edge.tif')
    geoprocessing.raster_calculator(
        [(shelf_convoultion_raster_path, 1)], _shelf_mask_op,
        shelf_edge_raster_path, gdal.GDT_Byte, nodata)

    shore_geotransform = geoprocessing.get_raster_info(
        shelf_edge_raster_path)['geotransform']

    point_list = []
    for offset_info, data_block in geoprocessing.iterblocks(
            (shelf_edge_raster_path, 1)):
        row_indexes, col_indexes = numpy.mgrid[
            offset_info['yoff']:offset_info['yoff']+offset_info['win_ysize'],
            offset_info['xoff']:offset_info['xoff']+offset_info['win_xsize']]
        valid_mask = data_block == 1
        x_coordinates = (
            shore_geotransform[0] +
            shore_geotransform[1] * (col_indexes[valid_mask] + 0.5) +
            shore_geotransform[2] * (row_indexes[valid_mask] + 0.5))
        y_coordinates = (
            shore_geotransform[3] +
            shore_geotransform[4] * (col_indexes[valid_mask] + 0.5) +
            shore_geotransform[5] * (row_indexes[valid_mask] + 0.5))

        for x_coord, y_coord in zip(x_coordinates, y_coordinates):
            point_list.append(shapely.geometry.Point(x_coord, y_coord))

    shelf_strtree = shapely.strtree.STRtree(point_list)

    shore_point_layer.StartTransaction()
    for point_feature in shore_point_layer:
        point_geometry = point_feature.GetGeometryRef()
        point_shapely = shapely.wkb.loads(bytes(point_geometry.ExportToWkb()))
        try:
            nearest_point = point_list[shelf_strtree.query_nearest(
                point_shapely, all_matches=False)[0]]
            distance = nearest_point.distance(point_shapely)
            point_feature.SetField(surge_fieldname, float(distance))
        except IndexError:
            # so far away it's essentially not an issue
            point_feature.SetField(surge_fieldname, max_fetch_distance)
        shore_point_layer.SetFeature(point_feature)

    shore_point_layer.CommitTransaction()
    shore_point_layer.SyncToDisk()
    shore_point_layer = None
    shore_point_vector = None


def calculate_wind_and_wave(
        shore_point_vector_path,
        shore_point_sample_distance, landmass_vector_path,
        landmass_boundary_vector_path, bathymetry_raster_path,
        wwiii_strtree, wwiii_object_list, n_fetch_rays, max_fetch_distance,
        wind_fieldname, wave_fieldname):
    """Calculate wind exposure for given points.

    Parameters:
        shore_point_vector_path (str): path to a point vector, this value will
            be modified to hold the total wind exposure at this point.
        shore_point_sample_distance (float): straight line distance between
            shore sample points.
        landmass_vector_path (str): path to a vector indicating landmass that
            will block wind exposure.
        landmass_boundary_vector_path (str): path to a string vector containing
            the perimeter of `landmass_vector_path`.
        bathymetry_raster_path (str): path to a raster indicating bathymetry
            values. (negative is deeper).
        wwiii_strtree (str): path to an r_tree that can find the nearest point
            index.
        wwiii_object_list (list): a list containing objects that can be
            indexed with nearest point index with in lat/lng whose object has
            values 'REI_PCT', 'REI_V', 'WavP_[DIR]', 'WavPPCT', 'V10PCT_[DIR]'.
        n_fetch_rays (int): number of equally spaced rays to cast out from
            a point to determine fetch value
        max_fetch_distance (float): maximum distance to send a ray to determine
            wind/wave risk
        wind_fieldname (str): fieldname to add to `shore_point_vector_path` for
            wind power.
        wave_fieldname (str): fieldname to add to `shore_point_vector_path` for
            wave power.

    Returns:
        None

    """
    gpkg_driver = ogr.GetDriverByName('gpkg')
    temp_workspace_dir = tempfile.mkdtemp(
        dir=os.path.dirname(shore_point_vector_path),
        prefix='calculate_rwind_')
    temp_fetch_rays_vector = gpkg_driver.CreateDataSource(
        os.path.join(temp_workspace_dir, 'fetch_rays.gpkg'))
    layer_name = 'fetch_rays'
    shore_point_projection_wkt = geoprocessing.get_vector_info(
        shore_point_vector_path)['projection_wkt']
    shore_point_srs = osr.SpatialReference()
    shore_point_srs.ImportFromWkt(shore_point_projection_wkt)
    shore_point_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    temp_fetch_rays_layer = (
        temp_fetch_rays_vector.CreateLayer(
            str(layer_name), shore_point_srs, ogr.wkbLineString))
    temp_fetch_rays_layer.CreateField(ogr.FieldDefn(
        'fetch_dist', ogr.OFTReal))
    temp_fetch_rays_layer.CreateField(ogr.FieldDefn(
        'direction', ogr.OFTReal))
    temp_fetch_rays_defn = temp_fetch_rays_layer.GetLayerDefn()

    # These WWIII fields are the only ones needed for wind & wave equations
    # Copy them to a new vector which also gets more fields added with
    # computed values.
    target_shore_point_vector = gdal.OpenEx(
        shore_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    target_shore_point_layer = target_shore_point_vector.GetLayer()
    target_shore_point_layer.CreateField(
        ogr.FieldDefn(wind_fieldname, ogr.OFTReal))
    target_shore_point_layer.CreateField(
        ogr.FieldDefn(wave_fieldname, ogr.OFTReal))
    for ray_index in range(n_fetch_rays):
        compass_degree = int(ray_index * 360 / n_fetch_rays)
        target_shore_point_layer.CreateField(
            ogr.FieldDefn('fdist_%d' % compass_degree, ogr.OFTReal))
        target_shore_point_layer.CreateField(
            ogr.FieldDefn('fdepth_%d' % compass_degree, ogr.OFTReal))

    # Iterate over every shore point
    LOGGER.info("Casting rays and extracting bathymetry values")
    bathy_raster = gdal.OpenEx(
        bathymetry_raster_path, gdal.OF_RASTER | gdal.GA_ReadOnly)
    bathy_band = bathy_raster.GetRasterBand(1)
    bathy_raster_info = geoprocessing.get_raster_info(bathymetry_raster_path)
    bathy_gt = bathy_raster_info['geotransform']
    bathy_inv_gt = gdal.InvGeoTransform(bathy_gt)

    landmass_vector = gdal.OpenEx(landmass_vector_path, gdal.OF_VECTOR)
    landmass_layer = landmass_vector.GetLayer()
    landmass_geom_list = [
        shapely.wkb.loads(bytes(f.GetGeometryRef().ExportToWkb()))
        for f in landmass_layer]
    if landmass_geom_list:
        try:
            landmass_union_geom = shapely.ops.unary_union(landmass_geom_list)
        except shapely.errors.GEOSException:
            LOGGER.warning('error on invalid geom, skipping')
            landmass_union_geom = shapely.Polygon()
    else:
        landmass_union_geom = shapely.Polygon()

    landmass_layer = None
    landmass_vector = None
    landmass_union_geom_prep = shapely.prepared.prep(landmass_union_geom)

    landmass_boundary_vector = gdal.OpenEx(
        landmass_boundary_vector_path, gdal.OF_VECTOR)
    landmass_boundary_layer = landmass_boundary_vector.GetLayer()
    landmass_boundary_geom_list = [
        shapely.wkb.loads(bytes(f.GetGeometryRef().ExportToWkb()))
        for f in landmass_boundary_layer]
    if landmass_boundary_geom_list:
        landmass_boundary_union_geom = shapely.ops.unary_union(
            landmass_boundary_geom_list)
    else:
        landmass_boundary_union_geom = shapely.Polygon()
    landmass_boundary_layer = None
    landmass_boundary_vector = None
    landmass_boundary_union_geom_prep = shapely.prepared.prep(
        landmass_boundary_union_geom)
    landmass_boundary_strtree, landmass_boundary_object_list = build_strtree(
        landmass_boundary_vector_path)

    LOGGER.debug('Starting local ray cast sampling')

    target_shore_point_layer.StartTransaction()
    temp_fetch_rays_layer.StartTransaction()

    # make a transfomer for local points to lat/lng for wwiii_strtree
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    base_to_target_transform = osr.CoordinateTransformation(
        shore_point_srs, wgs84_srs)

    ray_test_time = 0
    ray_iterate_time = 0
    ray_geometry_construction_time = 0
    compute_wave_height_time = 0
    local_interp_time = 0
    bathy_read_time = 0
    point_creation_time = 0

    for shore_point_feature in target_shore_point_layer:
        shore_point_geom = shore_point_feature.GetGeometryRef().Clone()
        _ = shore_point_geom.Transform(base_to_target_transform)
        shore_point_shapely = shapely.wkb.loads(
            bytes(shore_point_geom.ExportToWkb()))
        wwiii_index = next(iter(wwiii_strtree.query_nearest(
            shore_point_shapely, all_matches=False)))
        wwiii_point = wwiii_object_list[wwiii_index].field_val_map

        rei_value = 0.0
        height_list = []
        period_list = []
        e_local = 0.0
        e_ocean = 0.0

        shore_point_geometry = shore_point_feature.GetGeometryRef()
        shapely_point = shapely.wkb.loads(
            bytes(shore_point_geometry.ExportToWkb()))
        if landmass_union_geom_prep.contains(shapely_point):
            new_point = shapely.ops.nearest_points(
                landmass_boundary_union_geom, shapely_point)[0]
            shore_point_geometry = ogr.CreateGeometryFromWkb(new_point.wkb)

        # Iterate over every ray direction
        for sample_index in range(n_fetch_rays):
            compass_degree = int(sample_index * 360 / n_fetch_rays)
            compass_theta = float(sample_index) / n_fetch_rays * 360

            # wwiii_point should be closest point to shore point
            rei_pct = wwiii_point[
                'REI_PCT%d' % int(compass_theta)]
            rei_v = wwiii_point[
                'REI_V%d' % int(compass_theta)]
            cartesian_theta = -(compass_theta - 90)

            # Determine the direction the ray will point
            delta_x = math.cos(cartesian_theta * math.pi / 180)
            delta_y = math.sin(cartesian_theta * math.pi / 180)

            # Start a ray offset from the shore point
            # so that rays start outside of the landmass.
            # Shore points are interpolated onto the coastline,
            # but floating point error results in points being just
            # barely inside/outside the landmass.
            offset = 10

            point_a_x = (
                shore_point_geometry.GetX() + delta_x * offset)
            point_a_y = (
                shore_point_geometry.GetY() + delta_y * offset)

            origin_point = shapely.geometry.Point(point_a_x, point_a_y)
            if landmass_union_geom_prep.intersects(origin_point):
                # the origin is inside the landmass, skip
                continue

            point_b_x = point_a_x + delta_x * (
                max_fetch_distance)
            point_b_y = point_a_y + delta_y * (
                max_fetch_distance)

            # build ray geometry so we can intersect it later
            ray_shapely = shapely.geometry.LineString(
                [[point_a_x, point_a_y], [point_b_x, point_b_y]])

            # keep a shapely version of the ray so we can do fast intersection
            # with it and the entire landmass
            ray_point_origin_shapely = shapely.geometry.Point(
                point_a_x, point_a_y)

            if not landmass_boundary_union_geom_prep.intersects(
                    ray_point_origin_shapely):
                # the origin is in ocean, so we'll get a ray length > 0.0
                start_time = time.time()
                tested_indexes = set()
                while True:
                    intersection = False
                    for landmass_index in landmass_boundary_strtree.query(
                            shapely.box(*ray_shapely.bounds)):
                        landmass_line = landmass_boundary_object_list[
                            landmass_index]
                        if landmass_line.id in tested_indexes:
                            continue
                        tested_indexes.add(landmass_line.id)
                        if ray_shapely.intersects(landmass_line.geom):
                            intersection_point = ray_shapely.intersection(
                                landmass_line.geom)
                            # offset the dist with smallest_feature_size
                            # update the endpoint of the ray
                            intersection = True
                            try:
                                ray_shapely = shapely.geometry.LineString(
                                    [[point_a_x, point_a_y],
                                     [intersection_point.x, intersection_point.y]])
                            except AttributeError:
                                LOGGER.warning(
                                    f'for some reason result was not a point: '
                                    f'{intersection_point}')
                                continue
                            break
                    if not intersection:
                        break
                ray_test_time += time.time() - start_time
                start_time = time.time()
                bathy_values = []
                local_time = time.time()
                (start_x, start_y), (end_x, end_y) = ray_shapely.coords

                delta_x = end_x-start_x
                delta_y = end_y-start_y
                point_array = (
                    numpy.arange(
                        0, ray_shapely.length,
                        shore_point_sample_distance/4).reshape(-1, 1) /
                    ray_shapely.length * numpy.array([delta_x, delta_y]) +
                    numpy.array([start_x, start_y]))

                local_interp_time += time.time()-local_time

                local_time = time.time()
                point_array = (
                    point_array*[bathy_inv_gt[1], bathy_inv_gt[5]] +
                    [bathy_inv_gt[0], bathy_inv_gt[3]])
                point_creation_time += time.time()-local_time

                local_time = time.time()
                for pixel_x, pixel_y in point_array:
                    if (pixel_x < 0 or pixel_y < 0 or
                            pixel_x >= bathy_band.XSize or
                            pixel_y >= bathy_band.YSize):
                        break
                    bathy_values.append(
                        bathy_band.ReadAsArray(
                            int(pixel_x), int(pixel_y), 1, 1)[0][0])
                bathy_read_time += time.time()-local_time

                if bathy_values:
                    avg_bathy_value = numpy.mean(bathy_values)
                else:
                    avg_bathy_value = 0.0
                ray_iterate_time += time.time()-start_time

                start_time = time.time()
                # when we get here, we have the final ray geometry
                ray_feature = ogr.Feature(temp_fetch_rays_defn)
                ray_feature.SetField('fetch_dist', ray_shapely.length)
                ray_feature.SetField('direction', compass_degree)
                ray_geometry = ogr.CreateGeometryFromWkb(ray_shapely.wkb)
                ray_feature.SetGeometry(ray_geometry)
                temp_fetch_rays_layer.CreateFeature(ray_feature)
                ray_length = ray_shapely.length
                rei_value += ray_length * rei_pct * rei_v
                ray_feature = None
                ray_geometry = None

                shore_point_feature.SetField(
                    'fdist_%d' % compass_degree, ray_length)
                shore_point_feature.SetField(
                    'fdepth_%d' % compass_degree, float(avg_bathy_value))

                ray_geometry_construction_time += time.time()-start_time

                velocity = wwiii_point['V10PCT_%d' % compass_degree]
                occurrence = wwiii_point['REI_PCT%d' % compass_degree]

                start_time = time.time()
                height = compute_wave_height(
                    velocity, ray_shapely.length, avg_bathy_value)
                height_list.append(height)
                period = compute_wave_period(
                    velocity, ray_shapely.length, avg_bathy_value)
                period_list.append(period)
                power = 0.5 * float(height)**2 * float(period)  # UG Eq. 8
                e_local += power * occurrence  # UG Eq. 9

                if intersection:
                    e_ocean += (
                        wwiii_point['WavP_%d' % compass_degree] *
                        wwiii_point['WavPPCT%d' % compass_degree])

                ray_feature = None
                rei_value += ray_length * rei_pct * rei_v
                compute_wave_height_time += time.time() - start_time
        shore_point_feature.SetField(wind_fieldname, rei_value)
        shore_point_feature.SetField(wave_fieldname, max(e_ocean, e_local))
        target_shore_point_layer.SetFeature(shore_point_feature)
        shore_point_geometry = None
    target_shore_point_layer.CommitTransaction()
    target_shore_point_layer.SyncToDisk()
    target_shore_point_layer = None
    target_shore_point_vector = None
    temp_fetch_rays_layer.CommitTransaction()
    temp_fetch_rays_layer.SyncToDisk()
    temp_fetch_rays_layer = None
    temp_fetch_rays_vector = None
    bathy_raster = None
    bathy_band = None
    try:
        shutil.rmtree(temp_workspace_dir)
    except Exception:
        LOGGER.exception('unable to remove %s', temp_workspace_dir)
    finally:
        LOGGER.info(
            f'\nray_test_time: {ray_test_time:.2f}s\n'
            f'ray_iterate_time: {ray_iterate_time:.2f}s\n'
            f'\tlocal_interp_time: {local_interp_time:.2f}s\n'
            f'\tbathy_read_time: {bathy_read_time:.2f}s\n'
            f'\tpoint_creation_time: {point_creation_time:.2f}s\n'
            f'ray_geometry_construction_time: {ray_geometry_construction_time:.2f}s\n'
            f'compute_wave_height_time: {compute_wave_height_time:.2f}s\n')


def calculate_slr(shore_point_vector_path, slr_raster_path, target_fieldname):
    """Sample sea level rise raster and store values in shore points.

    Parameters:
        shore_point_vector_path (str): path to a vector of points in a local
            projected coordinate system. This vector will be modified by this
            function to include a new field called `target_fieldname`
            containing the weighted Rhab risk for the given point.
        slr_raster_path (str): path to a sea level rise raster indicating
            sea level rise amout in m.
        target_fieldname (str): fieldname to add to `shore_point_vector_path`
            that will contain the value of sea level rise for that point.

    Returns:
        None.

    """
    try:
        shore_point_vector = gdal.OpenEx(
            shore_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
        shore_point_layer = shore_point_vector.GetLayer()
        slr_field = ogr.FieldDefn(target_fieldname, ogr.OFTReal)
        slr_field.SetPrecision(5)
        slr_field.SetWidth(24)
        shore_point_layer.CreateField(slr_field)
        slr_info = geoprocessing.get_raster_info(slr_raster_path)
        inv_gt = gdal.InvGeoTransform(slr_info['geotransform'])

        slr_raster = gdal.OpenEx(slr_raster_path, gdal.OF_RASTER)
        slr_band = slr_raster.GetRasterBand(1)

        shore_point_layer.ResetReading()
        shore_point_layer.StartTransaction()
        for point_feature in shore_point_layer:
            point_geometry = point_feature.GetGeometryRef()
            point_x, point_y = point_geometry.GetX(), point_geometry.GetY()
            point_geometry = None

            pixel_x, pixel_y = [
                int(x) for x in
                gdal.ApplyGeoTransform(inv_gt, point_x, point_y)]
            if pixel_x < 0:
                pixel_x = 0
            if pixel_y < 0:
                pixel_y = 0
            if pixel_x >= slr_band.XSize:
                pixel_x = slr_band.XSize-1
            if pixel_y >= slr_band.YSize:
                pixel_y = slr_band.YSize-1
            try:
                pixel_value = slr_band.ReadAsArray(
                    xoff=pixel_x, yoff=pixel_y, win_xsize=1,
                    win_ysize=1)[0, 0]
            except Exception:
                LOGGER.exception(
                    'slr_band size %d %d', slr_band.XSize,
                    slr_band.YSize)
                raise
            point_feature.SetField(target_fieldname, float(pixel_value))
            shore_point_layer.SetFeature(point_feature)
        shore_point_layer.CommitTransaction()

        shore_point_layer.SyncToDisk()
        shore_point_layer = None
        shore_point_vector = None
        slr_raster = None
        slr_band = None

    except Exception:
        LOGGER.exception('error in slr calc')
        raise


def calculate_rhab(
        shore_point_vector_path, habitat_raster_path_map, target_fieldname,
        target_pixel_size):
    """Add Rhab risk to the shore point vector path.

    Parameters:
        shore_point_vector_path (str): path to a vector of points in a local
            projected coordinate system. This vector will be modified by this
            function to include a new field called `target_fieldname`
            containing the weighted Rhab risk for the given point.
        habitat_raster_path_map (dict): a dictionary mapping (risk, dist)
            tuples to raster mask paths.
        target_fieldname (str): fieldname to add to `shore_point_vector_path`
            that will contain the value of Rhab calculated for that point.
        target_pixel_size (list): x/y size of clipped habitat in projected
            units

    Returns:
        None.

    """
    shore_point_vector = gdal.OpenEx(
        shore_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
    shore_point_layer = shore_point_vector.GetLayer()

    shore_point_layer.StartTransaction()
    shore_point_defn = shore_point_layer.GetLayerDefn()

    hab_id_list = []

    for payload in habitat_raster_path_map:
        if isinstance(payload, tuple):
            hab_id_list.append(payload[0])
        else:
            continue
    hab_knockout_pattern = 'Rhab_no_{hab_id}'
    hab_knockout_id_list = [
        hab_knockout_pattern.format(hab_id=hab_id) for hab_id in hab_id_list]
    for hab_id in hab_id_list + hab_knockout_id_list + ['Rhab']:
        if shore_point_defn.GetFieldIndex(hab_id) == -1:
            hab_r_field = ogr.FieldDefn(str(hab_id), ogr.OFTReal)
            hab_r_field.SetPrecision(5)
            hab_r_field.SetWidth(24)
            shore_point_layer.CreateField(hab_r_field)
    shore_point_layer.CommitTransaction()

    shore_point_info = geoprocessing.get_vector_info(shore_point_vector_path)

    tmp_working_dir = tempfile.mkdtemp(
        prefix='calculate_rhab_',
        dir=os.path.dirname(shore_point_vector_path))

    hab_id_feature_risk_map = collections.defaultdict(
        lambda: collections.defaultdict(float))
    hab_to_risk_map = {}
    for payload, hab_raster_path_list in (
                habitat_raster_path_map.items()):
        LOGGER.debug(f'{payload} {hab_raster_path_list}')
        if isinstance(payload, tuple):
            (hab_id, risk_val, eff_dist) = payload
        else:
            raise ValueError(f'unknown input "{payload}"')

        local_hab_raster_path = os.path.join(
            tmp_working_dir, f'{hab_id}.tif')
        local_clip_stack = []
        for hab_raster_path in hab_raster_path_list:
            basename = os.path.basename(os.path.splitext(hab_raster_path)[0])
            local_clip_raster_path = os.path.join(
                tmp_working_dir, f'{basename}.tif')
            clip_and_reproject_raster(
                hab_raster_path, local_clip_raster_path,
                shore_point_info['projection_wkt'],
                shore_point_info['bounding_box'], eff_dist, 'near', False,
                target_pixel_size)
            local_clip_stack.append(local_clip_raster_path)
        merge_mask_list(local_clip_stack, local_hab_raster_path)

        kernel_filepath = '%s_kernel%s' % os.path.splitext(
            local_hab_raster_path)
        kernel_radius = [abs(eff_dist / x) for x in target_pixel_size]
        create_averaging_kernel_raster(
            kernel_radius, kernel_filepath, normalize=True)
        hab_coverage_raster_path = (
            '%s_hab_coverage%s' % os.path.splitext(local_hab_raster_path))
        geoprocessing.convolve_2d(
            (local_hab_raster_path, 1), (kernel_filepath, 1),
            hab_coverage_raster_path, mask_nodata=False)
        gt = geoprocessing.get_raster_info(
            hab_coverage_raster_path)['geotransform']
        inv_gt = gdal.InvGeoTransform(gt)

        hab_effective_raster = gdal.OpenEx(
            hab_coverage_raster_path, gdal.OF_RASTER)
        hab_effective_band = hab_effective_raster.GetRasterBand(1)
        shore_point_layer.ResetReading()
        LOGGER.info(f'setting the {hab_id} field in {shore_point_vector_path}')
        for shore_feature in shore_point_layer:
            shore_geom = shore_feature.GetGeometryRef()
            pixel_x, pixel_y = [
                int(x) for x in
                gdal.ApplyGeoTransform(
                    inv_gt, shore_geom.GetX(), shore_geom.GetY())]
            if pixel_x < 0:
                pixel_x = 0
            if pixel_y < 0:
                pixel_y = 0
            if pixel_x >= hab_effective_band.XSize:
                pixel_x = hab_effective_band.XSize-1
            if pixel_y >= hab_effective_band.YSize:
                pixel_y = hab_effective_band.YSize-1
            try:
                pixel_val = hab_effective_band.ReadAsArray(
                    xoff=pixel_x, yoff=pixel_y, win_xsize=1,
                    win_ysize=1)[0, 0]
            except Exception:
                LOGGER.exception('error on pixel fetch for hab')
            hab_to_risk_map[hab_id] = risk_val
            hab_id_feature_risk_map[shore_feature.GetFID()][hab_id] += pixel_val

    min_risk_value = numpy.min(list(hab_to_risk_map.values()))

    # set all the features now that the hab values have been collected
    max_risk_threshold = 1-(1/2)**0.5
    shore_point_layer.StartTransaction()
    for shore_feature in shore_point_layer:
        fid = shore_feature.GetFID()
        hab_risk_dict = hab_id_feature_risk_map[fid]
        shore_point_feature_risk_map = {}
        for hab_id, hab_risk_val in hab_to_risk_map.items():
            pixel_val = hab_risk_dict[hab_id]
            local_risk_val = 5-(5-hab_risk_val)*numpy.minimum(1, numpy.sin(
                numpy.minimum(
                    numpy.pi/2, pixel_val*numpy.pi/2/max_risk_threshold)))
            shore_point_feature_risk_map[hab_id] = local_risk_val
            shore_feature.SetField(hab_id, local_risk_val)
        # TODO: combine all the hab risk into an overall Rhab
        for hab_id in hab_id_list:
            hab_knockout_id = hab_knockout_pattern.format(hab_id=hab_id)
            remaining_hab_ids = list(set(hab_id_list)-set([hab_id]))
            hab_risk_list = [
                shore_point_feature_risk_map[h] for h in remaining_hab_ids]
            r_hab = numpy.maximum(
                min_risk_value, 5-4*numpy.linalg.norm(
                    1-((numpy.array(hab_risk_list)-1)/4), axis=0))
            shore_feature.SetField(hab_knockout_id, r_hab)
        # Do once more for all habitat
        hab_risk_list = list(shore_point_feature_risk_map.values())
        r_hab = numpy.maximum(
            min_risk_value, 5-4*numpy.linalg.norm(
                1-((numpy.array(hab_risk_list)-1)/4), axis=0))
        shore_feature.SetField('Rhab', r_hab)
        shore_point_layer.SetFeature(shore_feature)

    shore_point_layer.CommitTransaction()

    shore_point_layer = None
    shore_point_vector = None
    hab_effective_raster = None
    hab_effective_band = None
    retrying_rmtree(tmp_working_dir)


@retrying.retry(
    wait_exponential_multiplier=100, wait_exponential_max=2000,
    stop_max_attempt_number=5)
def retrying_rmtree(dir_path):
    """Remove `dir_path` but try a few times."""
    try:
        shutil.rmtree(dir_path)
    except Exception:
        LOGGER.exception('unable to remove %s' % dir_path)
        raise


def calculate_utm_srs(lng, lat):
    """Calculate UTM SRS from the lng/lat point given.

    Parameters:
        lng (float): longitude point.
        lat (float): latitude point.

    Returns:
        osr.SpatialReference in the UTM zone that contains the point (lng, lat)

    """
    utm_code = (math.floor((lng+180)/6) % 60) + 1
    lat_code = 6 if lat > 0 else 7
    epsg_code = int('32%d%02d' % (lat_code, utm_code))
    utm_srs = osr.SpatialReference()
    utm_srs.ImportFromEPSG(epsg_code)
    return utm_srs


def clip_geometry(
        bounding_box_coords, base_srs, target_srs, ogr_geometry_type,
        global_geom_strtree, strtree_object_list, target_vector_path):
    """Clip geometry in `global_geom_strtree` to bounding box.

    Parameters:
        bounding_box_coords (list): a list of bounding box coordinates in
            the same coordinate system as the geometry in
            `global_geom_strtree`.
        target_srs (osr.SpatialReference): target spatial reference for
            creating the target vector.
        ogr_geometry_type (ogr.wkb[TYPE]): geometry type to create for the
            target vector.
        global_geom_strtree (shapely.strtree.STRtree): an rtree loaded with
            geometry to query via bounding box.
        strtree_object_list (list): list of objects that are indexed by result
            of strtree query. Each object will contain
            parameters `field_val_map` and `prep` that have values to copy to
            `target_fieldname` and used to quickly query geometry. Main object
            will have `field_name_type_list` field used to describe the
            original field name/types.
        target_vector_path (str): path to vector to create that will contain
            locally projected geometry clipped to the given bounding box.

    Returns:
        None.

    """
    gpkg_driver = ogr.GetDriverByName("GPKG")
    vector = gpkg_driver.CreateDataSource(
        target_vector_path)
    layer = vector.CreateLayer(
        os.path.splitext(os.path.basename(target_vector_path))[0],
        target_srs, ogr_geometry_type)
    field_names_defined = hasattr(global_geom_strtree, 'field_name_type_list')
    if field_names_defined:
        field_name_type_list = list(global_geom_strtree.field_name_type_list)
        for field_name, field_type in field_name_type_list:
            layer.CreateField(ogr.FieldDefn(field_name, field_type))
    layer_defn = layer.GetLayerDefn()
    base_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    base_to_target_transform = osr.CoordinateTransformation(
        base_srs, target_srs)

    bounding_box = shapely.geometry.box(*bounding_box_coords)
    index_intersections = global_geom_strtree.query(bounding_box)
    possible_geom_list = [
        strtree_object_list[index].geom for index in index_intersections]
    if not possible_geom_list:
        layer = None
        vector = None
        raise ValueError('no data intersects this box')
    for index, geom in zip(index_intersections, possible_geom_list):
        clipped_shapely_geom = bounding_box.intersection(geom)
        clipped_geom = ogr.CreateGeometryFromWkb(clipped_shapely_geom.wkb)
        error_code = clipped_geom.Transform(base_to_target_transform)
        if error_code:
            LOGGER.warning(f'error code {error_code} encountered on {geom}')
            continue
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(clipped_geom.Clone())
        if field_names_defined:
            for field_name, _ in field_name_type_list:
                feature.SetField(
                    field_name,
                    strtree_object_list[index].field_val_map[field_name])
        layer.CreateFeature(feature)


def voroni_polygons_from_points(
        point_vector_path, shore_point_sample_distance, max_sample_distance,
        voroni_poly_vector_path):
    """Convert points to voroni polygons.

    Parameters:
        point_vector_path (str): path to point based vector
        voroni_poly_vector_path (str): path to vector to create on output

    Returns:
        None.
    """
    driver = ogr.GetDriverByName('GPKG')
    point_vector = ogr.Open(point_vector_path, 0)
    point_layer = point_vector.GetLayer()

    srs = point_layer.GetSpatialRef()

    # convert to shapely multipoint
    points_list = []
    for feature in point_layer:
        geom = feature.GetGeometryRef()
        points_list.append((geom.GetX(), geom.GetY()))
    points = MultiPoint(points_list)
    buffered_convex_hull = points.convex_hull.buffer(max_sample_distance)
    regions = voronoi_diagram(points, envelope=buffered_convex_hull.envelope)
    regions = GeometryCollection([
        geom.intersection(buffered_convex_hull)
        for geom in regions.geoms if geom.intersects(buffered_convex_hull)])

    # Check and delete if the file already exists
    if os.path.exists(voroni_poly_vector_path):
        os.remove(voroni_poly_vector_path)

    # Create the output driver and file
    shore_point_aoi_vector = driver.CreateDataSource(voroni_poly_vector_path)
    shore_point_aoi_layer = shore_point_aoi_vector.CreateLayer(
        'shore_point_aoi', geom_type=ogr.wkbPolygon, srs=srs)
    # Copy fields from point_layer to shore_point_aoi_layer
    point_layer_defn = point_layer.GetLayerDefn()
    for i in range(point_layer_defn.GetFieldCount()):
        field_defn = point_layer_defn.GetFieldDefn(i)
        shore_point_aoi_layer.CreateField(field_defn)

    # Loop through shapely polygons and write to GPKG
    point_layer.ResetReading()
    for base_point_feature, poly in zip(point_layer, regions.geoms):
        aoi_feature = ogr.Feature(shore_point_aoi_layer.GetLayerDefn())
        for i in range(base_point_feature.GetFieldCount()):
            field_name = base_point_feature.GetFieldDefnRef(i).GetName()
            field_value = base_point_feature.GetField(i)
            aoi_feature.SetField(field_name, field_value)

        geom = ogr.CreateGeometryFromWkb(poly.wkb)
        aoi_feature.SetGeometry(geom)
        shore_point_aoi_layer.CreateFeature(aoi_feature)
        aoi_feature = None

    shore_point_aoi_layer = None
    shore_point_aoi_vector = None
    point_vector = None
    point_vector = None


def sample_line_to_points(
        line_vector_path, target_point_path, point_step_size):
    """Sample lines in line vector to points along the path.

    Parameters:
        line_vector_path (str): path to line based vector.
        target_point_path (str): created by this function. A GPKG that is in
            the same projection as `line_vector` where points lie on those line
            segments spaced no less than `point_step_size` apart.
        point_step_size (float): step size in projected units of `line_vector`
            for points to drop along the line segments.

    Returns:
        None.

    """
    line_vector = gdal.OpenEx(line_vector_path)
    line_layer = line_vector.GetLayer()
    layer_name = os.path.splitext(os.path.basename(line_vector_path))[0]

    gpkg_driver = ogr.GetDriverByName('GPKG')
    if os.path.exists(target_point_path):
        os.remove(target_point_path)
    point_vector = gpkg_driver.CreateDataSource(target_point_path)
    point_layer = point_vector.CreateLayer(
        layer_name, line_layer.GetSpatialRef(), ogr.wkbPoint,
        ['OVERWRITE=YES'])
    point_defn = point_layer.GetLayerDefn()
    for feature in line_layer:
        try:
            current_distance = 0.0
            line_geom = feature.GetGeometryRef()
            line = shapely.wkb.loads(bytes(line_geom.ExportToWkb()))
            if isinstance(line, shapely.geometry.collection.GeometryCollection):
                line_list = []
                for geom in list(line):
                    if isinstance(geom, (
                            shapely.geometry.linestring.LineString,
                            shapely.geometry.multilinestring.MultiLineString)):
                        line_list.append(geom)
                print('building: %s', line_list)
                line = shapely.geometry.MultiLineString(line_list)
            while current_distance < line.length:
                try:
                    new_point = line.interpolate(current_distance)
                    current_distance += point_step_size
                    new_point_feature = ogr.Feature(point_defn)
                    new_point_geom = ogr.CreateGeometryFromWkb(new_point.wkb)
                    new_point_feature.SetGeometry(new_point_geom)
                    point_layer.CreateFeature(new_point_feature)
                except Exception:
                    LOGGER.exception('error on %s', line_geom)
                    raise
        except TypeError:
            LOGGER.warning(
                f'error when processing {feature} from {line_vector_path}, skipping')
            continue

    point_layer = None
    point_vector = None
    line_layer = None
    line_vector = None


def calculate_relief(
        shore_point_vector_path, relief_sample_distance, dem_path,
        target_fieldname):
    """Calculate DEM relief as average coastal land area within 5km.

    Parameters:
        shore_point_vector_path (string):  path to a point shapefile to
            for relief point analysis.
        relief_sample_distance (float): distance to send a ray out to
            determine relief value
        dem_path (string): path to a DEM raster projected in local coordinates.
        target_fieldname (string): this field name will be added to
            `shore_point_vector_path` and filled with Relief values.

    Returns:
        None.

    """
    try:
        shore_point_vector = gdal.OpenEx(
            shore_point_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
        shore_point_layer = shore_point_vector.GetLayer()
        relief_field = ogr.FieldDefn(target_fieldname, ogr.OFTReal)
        relief_field.SetPrecision(5)
        relief_field.SetWidth(24)
        shore_point_layer.CreateField(relief_field)
        dem_info = geoprocessing.get_raster_info(dem_path)

        tmp_working_dir = tempfile.mkdtemp(
            prefix='calculate_relief_',
            dir=os.path.dirname(shore_point_vector_path))

        dem_nodata = dem_info['nodata'][0]

        def zero_negative_values(depth_array):
            valid_mask = depth_array != dem_nodata
            result_array = numpy.empty_like(depth_array)
            result_array[:] = dem_nodata
            result_array[valid_mask] = 0
            result_array[depth_array > 0] = depth_array[depth_array > 0]
            return result_array

        positive_dem_path = os.path.join(
            tmp_working_dir, 'positive_dem.tif')

        geoprocessing.raster_calculator(
            [(dem_path, 1)], zero_negative_values,
            positive_dem_path, gdal.GDT_Int16, dem_nodata)

        # convolve over a 5km radius
        dem_pixel_size = dem_info['pixel_size']
        kernel_radius = (
            abs(relief_sample_distance // dem_pixel_size[0]),
            abs(relief_sample_distance // dem_pixel_size[1]))

        kernel_filepath = os.path.join(
            tmp_working_dir, 'averaging_kernel.tif')
        create_averaging_kernel_raster(
            kernel_radius, kernel_filepath, normalize=True)

        relief_path = os.path.join(tmp_working_dir, 'relief.tif')
        geoprocessing.convolve_2d(
            (positive_dem_path, 1), (kernel_filepath, 1), relief_path)
        relief_raster = gdal.Open(relief_path)
        relief_band = relief_raster.GetRasterBand(1)

        inv_gt = gdal.InvGeoTransform(dem_info['geotransform'])

        shore_point_layer.ResetReading()
        shore_point_layer.StartTransaction()
        for point_feature in shore_point_layer:
            point_geometry = point_feature.GetGeometryRef()
            point_x, point_y = point_geometry.GetX(), point_geometry.GetY()
            point_geometry = None

            pixel_x, pixel_y = [
                int(x) for x in
                gdal.ApplyGeoTransform(inv_gt, point_x, point_y)]

            if pixel_x < 0:
                pixel_x = 0
            if pixel_y < 0:
                pixel_y = 0
            if pixel_x >= relief_band.XSize:
                pixel_x = relief_band.XSize-1
            if pixel_y >= relief_band.YSize:
                pixel_y = relief_band.YSize-1

            try:
                pixel_value = relief_band.ReadAsArray(
                    xoff=pixel_x, yoff=pixel_y, win_xsize=1,
                    win_ysize=1)[0, 0]
            except Exception:
                LOGGER.exception(
                    'relief_band size %d %d', relief_band.XSize,
                    relief_band.YSize)
                raise
            # Make relief "negative" so when we histogram it for risk a
            # "higher" value will show a lower risk.
            point_feature.SetField(target_fieldname, -float(pixel_value))
            shore_point_layer.SetFeature(point_feature)

        shore_point_layer.CommitTransaction()
        shore_point_layer.SyncToDisk()
        shore_point_layer = None
        shore_point_vector = None
        relief_raster = None
        relief_band = None

        try:
            retrying_rmtree(tmp_working_dir)
        except OSError:
            LOGGER.warning('unable to rm %s' % tmp_working_dir)

    except Exception:
        LOGGER.exception('error in relief calc')
        raise


def clip_and_reproject_raster(
        base_raster_path, target_raster_path, target_srs_wkt,
        target_bounding_box, edge_buffer, resample_method,
        reproject_bounding_box, target_pixel_size):
    """Clip and reproject base to target raster.

    Parameters:
        base_raster_path (str): path to the raster to clip from.
        target_raster_path (str): path to target raster that is a clip from
            base projected in `target_srs_wkt` coordinate system.
        target_srs_wkt (str): spatial reference of target coordinate system in
            wkt.
        target_bounding_box (list): List of float describing target bounding
            box in base coordinate system as [minx, miny, maxx, maxy].
        edge_buffer (float): amount to extend sides of bounding box in target
            coordinate system units.
        resample_method (str): one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        reproject_bounding_box (bool): If true, project `target_bounding_box`
            from base coordinate system to `target_srs_wkt`.
        target_pixel_size (float): desired target pixel size in projected
            coordinates.

    Returns:
        None.

    """
    base_raster_info = geoprocessing.get_raster_info(base_raster_path)
    bb_centroid = (
        (target_bounding_box[0]+target_bounding_box[2])/2,
        (target_bounding_box[1]+target_bounding_box[3])/2)

    if reproject_bounding_box:
        local_bounding_box = geoprocessing.transform_bounding_box(
            target_bounding_box, base_raster_info['projection_wkt'],
            target_srs_wkt, edge_samples=11)
    else:
        local_bounding_box = target_bounding_box
        base_srs = osr.SpatialReference()
        try:
            base_srs.ImportFromWkt(base_raster_info['projection_wkt'])
        except Exception as e:
            LOGGER.exception(
                f'error on {base_raster_path}\n{base_raster_info}')
            raise e
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(target_srs_wkt)
        base_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_to_base_transform = osr.CoordinateTransformation(
            target_srs, base_srs)
        point = ogr.CreateGeometryFromWkt("POINT (%f %f)" % bb_centroid)
        point.Transform(target_to_base_transform)
        bb_centroid = (point.GetX(), point.GetY())

    if edge_buffer is not None:
        buffered_bounding_box = [
            local_bounding_box[0]-edge_buffer,
            local_bounding_box[1]-edge_buffer,
            local_bounding_box[2]+edge_buffer,
            local_bounding_box[3]+edge_buffer,
        ]
    else:
        buffered_bounding_box = local_bounding_box

    # target_pixel_size = estimate_projected_pixel_size(
    #     base_raster_path, bb_centroid, target_srs_wkt)
    geoprocessing.warp_raster(
        base_raster_path, target_pixel_size, target_raster_path,
        resample_method, target_bb=buffered_bounding_box,
        target_projection_wkt=target_srs_wkt,
        working_dir=os.path.dirname(target_raster_path))


def clip_raster(
        base_raster_path, target_raster_path,
        target_bounding_box, edge_buffer):
    """Clip and reproject base to target raster.

    Parameters:
        base_raster_path (str): path to the raster to clip from.
        target_raster_path (str): path to target raster that is a clip from
            base projected in `target_srs_wkt` coordinate system.
        target_srs_wkt (str): spatial reference of target coordinate system in
            wkt.
        target_bounding_box (list): List of float describing target bounding
            box in base coordinate system as [minx, miny, maxx, maxy].
        edge_buffer (float): amount to extend sides of bounding box in target
            coordinate system units.
        resample_method (str): one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        target_bounding_box (bool): If True, assumes bounding box is in
            base coordinate system and will transform it to target.

    Returns:
        None.

    """
    buffered_bounding_box = [
        target_bounding_box[0]-edge_buffer,
        target_bounding_box[1]-edge_buffer,
        target_bounding_box[2]+edge_buffer,
        target_bounding_box[3]+edge_buffer,
    ]

    base_raster = gdal.OpenEx(base_raster_path)
    gdal.Translate(
        target_raster_path, base_raster,
        format='GTiff',
        creationOptions=[
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'],
        outputBounds=buffered_bounding_box,
        callback=geoprocessing._make_logger_callback(
            "Translate %.1f%% complete"))
    base_raster = None


def estimate_projected_pixel_size(
        base_raster_path, sample_point, target_srs_wkt):
    """Estimate the pixel size of raster if projected in `target_srs_wkt`.

    Parameters:
        base_raster_path (str): path to a raster in some coordinate system.
        sample_point (list): [x, y] coordinate in base coordinate system of
            point to estimate projected pixel size around.
        target_srs_wkt (str): desired target coordinate system in wkt for
            estimate pixel size.

    Returns:
        None.

    """
    base_raster_info = geoprocessing.get_raster_info(base_raster_path)
    base_pixel_size = base_raster_info['pixel_size']
    raster_center_pixel_bb = [
        sample_point[0] - abs(base_pixel_size[0]/2),
        sample_point[1] - abs(base_pixel_size[1]/2),
        sample_point[0] + abs(base_pixel_size[0]/2),
        sample_point[1] + abs(base_pixel_size[1]/2),
    ]
    pixel_bb = geoprocessing.transform_bounding_box(
        raster_center_pixel_bb, base_raster_info['projection_wkt'],
        target_srs_wkt)
    # x goes to the right, y goes down
    estimated_pixel_size = [
        pixel_bb[2]-pixel_bb[0],
        pixel_bb[1]-pixel_bb[3]]
    return estimated_pixel_size


def create_averaging_kernel_raster(
        radius_in_pixels, kernel_filepath, normalize=True):
    """Create a flat raster kernel with a 2d radius given.

    Parameters:
        radius_in_pixels (tuple): the (x/y) distance of the averaging kernel.
        kernel_filepath (string): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.

    Returns:
        None

    """
    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath, int(2*radius_in_pixels[0]),
        int(2*radius_in_pixels[1]), 1, gdal.GDT_Float32,
        options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=ALL_CPUS'))

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([1, 0.1, 0, 1, 0, -0.1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    kernel_raster.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    n_cols = kernel_raster.RasterXSize
    n_rows = kernel_raster.RasterYSize
    iv, jv = numpy.meshgrid(range(n_rows), range(n_cols), indexing='ij')

    cx = n_cols / 2.0
    cy = n_rows / 2.0

    kernel_array = radius_in_pixels[0] - ((cx-jv)**2 + (cy-iv)**2)**0.5
    kernel_array[kernel_array < 0] = 0.0

    # normalize
    if normalize:
        kernel_array /= numpy.sum(kernel_array)
    kernel_band.WriteArray(kernel_array)
    kernel_band = None
    kernel_raster = None


def vector_to_lines(base_vector_path, target_line_vector_path):
    """Convert polygon vector to list of lines.

    Parameters:
        base_vector_path (str): path to polygon vector.
        target_line_vector_path (str): created by this file all polygons are
            converted to their line boundary equivalents.

    Returns:
        None.

    """
    # explode landmass into lines for easy intersection
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()

    gpkg_driver = ogr.GetDriverByName('GPKG')
    line_vector = gpkg_driver.CreateDataSource(
        target_line_vector_path)
    line_layer = line_vector.CreateLayer(
        target_line_vector_path, base_layer.GetSpatialRef(), ogr.wkbLineString)
    line_vector_defn = line_layer.GetLayerDefn()

    line_layer.StartTransaction()
    for base_feature in base_layer:
        base_shapely = shapely.wkb.loads(
            bytes(base_feature.GetGeometryRef().ExportToWkb()))
        for line in geometry_to_lines(base_shapely):
            segment_feature = ogr.Feature(line_vector_defn)
            segement_geometry = ogr.Geometry(ogr.wkbLineString)
            segement_geometry.AddPoint(*line.coords[0])
            segement_geometry.AddPoint(*line.coords[1])
            segment_feature.SetGeometry(segement_geometry)
            line_layer.CreateFeature(segment_feature)
    line_layer.CommitTransaction()
    line_layer = None
    line_vector = None
    base_vector = None
    base_layer = None


def geometry_to_lines(geometry):
    """Convert a geometry object to a list of lines."""
    if geometry.geom_type == 'Polygon':
        return polygon_to_lines(geometry)
    elif geometry.geom_type == 'MultiPolygon':
        line_list = []
        for geom in geometry.geoms:
            line_list.extend(geometry_to_lines(geom))
        return line_list
    else:
        return []


def polygon_to_lines(geometry):
    """Return a list of shapely lines given higher order shapely geometry."""
    line_list = []
    if len(geometry.exterior.coords) == 0:
        return line_list
    last_point = geometry.exterior.coords[0]
    for point in geometry.exterior.coords[1::]:
        if point == last_point:
            continue
        line_list.append(shapely.geometry.LineString([last_point, point]))
        last_point = point
    line_list.append(shapely.geometry.LineString([
        last_point, geometry.exterior.coords[0]]))
    for interior in geometry.interiors:
        last_point = interior.coords[0]
        for point in interior.coords[1::]:
            if point == last_point:
                continue
            line_list.append(shapely.geometry.LineString([last_point, point]))
            last_point = point
        line_list.append(shapely.geometry.LineString([
            last_point, interior.coords[0]]))
    return line_list


def compute_wave_height(Un, Fn, dn):
    """Compute Wave Height by User Guide eq 10.

    This equation may not be suitable for wind speed values < 1 m/s
    The WWIII database tends to include some 0s, otherwise values > 2.

    Parameters:
        Un (float): wind velocity in meters per second.
        Fn (float): fetch ray length in meters.
        dn (float): water depth in negative meters.

    Returns:
        Float: Wave height in meters

    """
    if Un < 1.0:
        LOGGER.warning(
            'Found wind velocity of %.2f, '
            'using 1.0m/s in wave height calculation instead' % Un)
        Un = 1.0
    g = 9.81
    dn = abs(dn)
    ds = g*dn/Un**2
    Fs = g*Fn/Un**2
    A = numpy.tanh(0.343*ds**1.14)
    B = numpy.tanh(4.41e-4*Fs**0.79/A)
    H_n = (0.24*Un**2/g)*(A*B)**0.572
    return H_n


def compute_wave_period(Un, Fn, dn):
    """Compute Wave Period by User Guide eq 10.

    This equation may not be suitable for wind speed values < 1 m/s
    The WWIII database tends to include some 0s, otherwise values > 2.

    Parameters:
        Un (float): wind velocity in meters per second.
        Fn (float): fetch ray length in meters.
        dn (float): water depth in negative meters.

    Returns:
        Float: Wave period in seconds

    """
    # This equation may not be suitable for wind speed values < 1 m/s
    # The WWIII database tends to include some 0s, otherwise values > 2
    if Un < 1.0:
        LOGGER.warning(
            'Found wind velocity of %.2f, '
            'using 1.0m/s in wave height calculation instead' % Un)
        Un = 1.0
    g = 9.81
    dn = abs(dn)
    ds = g*dn/Un**2
    Fs = g*Fn/Un**2
    A = numpy.tanh(0.1*ds**2.01)
    B = numpy.tanh(2.77e-7*Fs**1.45/A)
    T_n = 7.69*Un/g*(A*B)**0.187
    return T_n


def merge_mask_list(mask_path_list, target_mask_path):
    """Merge all nodata/1 masks in mask_path list to target."""

    def merge_masks_op(*mask_list):
        mask_stack = numpy.dstack(mask_list)
        # apply along axis 2
        result = numpy.any(mask_stack, axis=2)
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in mask_path_list], merge_masks_op,
        target_mask_path, gdal.GDT_Byte, 0)


def unmask_raster(base_raster_path, unmask_raster_path, target_raster_path):
    """Set base to 0 where unmask is 1."""

    def knockout_masks_op(base, unmask):
        result = base.copy()
        result[unmask == 1] = 0
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (unmask_raster_path, 1)], knockout_masks_op,
        target_raster_path, gdal.GDT_Byte, 0)


def merge_masks_op(mask_a, mask_b, nodata_a, nodata_b, target_nodata):
    """Logically 'or' two masks together."""
    result = numpy.empty(mask_a.shape, dtype=numpy.int16)
    valid_mask = (~numpy.isclose(mask_a, nodata_a) |
                  ~numpy.isclose(mask_b, nodata_b))
    result[:] = target_nodata
    result[valid_mask] = 1
    return result


def merge_cv_points_and_add_risk(
        cv_vector_queue, expected_payload_count, habitat_fieldname_list,
        target_cv_vector_path):
    """Merge vectors in `cv_vector_queue` into single vector.

    Parameters:
        cv_vector_queue (multiprocessing.Processing): a queue containing
            paths to CV workspace point vectors. Terminated with
            `STOP_SENTINEL`.
        target_cv_vector_path (str): path to a point vector created by this
            function.

    Returns:
        None.

    """
    try:
        if os.path.exists(target_cv_vector_path):
            os.remove(target_cv_vector_path)
        payload_count = 0
        start_time = time.time()
        result = None
        while True:
            cv_vector_path = cv_vector_queue.get()
            payload_count += 1
            if cv_vector_path == STOP_SENTINEL:
                break
            add_cv_vector_risk(
                habitat_fieldname_list, cv_vector_path)
            if not os.path.exists(target_cv_vector_path):
                shutil.copy(cv_vector_path, target_cv_vector_path)
                cmd = f'ogr2ogr -f "GPKG" -t_srs EPSG:4326 {target_cv_vector_path} {cv_vector_path}'
            else:
                cmd = f'ogr2ogr -f "GPKG" -t_srs EPSG:4326 -update -append {target_cv_vector_path} {cv_vector_path}'
            if result is not None:
                result.wait()
            result = subprocess.Popen(cmd, shell=True)
            LOGGER.info(result)
            expected_runtime = (expected_payload_count-payload_count) * (
                time.time()-start_time)/payload_count
            LOGGER.info(
                f'merge cv points {payload_count/expected_payload_count*100:.2f}% complete, '
                f'expect {expected_runtime:.2f}s to complete')
        result.wait()
    except Exception:
        LOGGER.exception('error in merge_cv_points_and_add_risk')
    finally:
        LOGGER.info('merge cv points complete')


def add_cv_vector_risk(habitat_fieldname_list, cv_risk_vector_path):
    """Use existing biophysical fields in `cv_risk_vector_path to calc total R.

    Args:
        habitat_fieldname_list (list): list of habitat ids used

        cv_risk_vector_path (str): path to point vector that has at least
            the following fields in it:

            * surge
            * ew
            * rei
            * slr
            * relief

            Will add the following fields:
                * Rwave
                * Rwind
                * Rsurge
                * Rrelief
                * Rslr
    Return:
        None
    """
    try:
        LOGGER.debug(f'adding risk for {cv_risk_vector_path}')
        cv_risk_vector = gdal.OpenEx(
            cv_risk_vector_path, gdal.OF_VECTOR | gdal.GA_Update)
        cv_risk_layer = cv_risk_vector.GetLayer()
        cv_risk_layer_defn = cv_risk_layer.GetLayerDefn()

        cv_risk_layer.StartTransaction()
        for base_field, risk_field in [
                ('surge', 'Rsurge'), ('ew', 'Rwave'), ('rei', 'Rwind'),
                ('slr', 'Rslr'), ('relief', 'Rrelief')]:
            LOGGER.info(f'set risk field for {base_field} {risk_field}')
            if cv_risk_layer_defn.GetFieldIndex(risk_field) == -1:
                cv_risk_layer.CreateField(ogr.FieldDefn(
                    risk_field, ogr.OFTReal))
            n_features = cv_risk_layer.GetFeatureCount()
            if n_features == 0:
                continue
            base_array = numpy.empty(shape=(n_features,))
            for index, feature in enumerate(cv_risk_layer):
                base_array[index] = feature.GetField(base_field)
            nan_mask = numpy.isnan(base_array)
            max_val = numpy.max(base_array[~nan_mask])
            base_array[nan_mask] = max_val
            n_bins = 100
            try:
                hist, bin_edges = numpy.histogram(base_array, bins=n_bins)
            except IndexError:
                LOGGER.warning(f'possible nodata error on histogram for {base_field}/{risk_field} on {cv_risk_vector_path}: {base_array}')
                bin_edges = [0]*5

            cv_risk_layer.ResetReading()
            n_features = cv_risk_layer.GetFeatureCount()
            for index, feature in enumerate(cv_risk_layer):
                if index % 1000 == 0:
                    LOGGER.info(
                        f'{index/n_features*100:.2f}% complete for {risk_field} '
                        f'{index+1}/{n_features}')
                base_val = feature.GetField(base_field)
                risk = 1+4*bisect.bisect_left(bin_edges, base_val)/n_bins
                feature.SetField(risk_field, risk)
                cv_risk_layer.SetFeature(feature)
        cv_risk_layer.ResetReading()

        for risk_id in ['Rt', 'Rt_nohab', 'Rt_hab_service_index'] + [f'Rt_hab_{hab_id}_service_index' for hab_id in habitat_fieldname_list]:
            if cv_risk_layer_defn.GetFieldIndex(risk_id) == -1:
                cv_risk_layer.CreateField(
                    ogr.FieldDefn(risk_id, ogr.OFTReal))

        for hab_id in habitat_fieldname_list:
            rt_hab_service_field_id = f'Rt_hab_service_index_{hab_id}'
            if cv_risk_layer_defn.GetFieldIndex(risk_id) == -1:
                cv_risk_layer.CreateField(
                    ogr.FieldDefn(rt_hab_service_field_id, ogr.OFTReal))

        cv_risk_layer.CommitTransaction()
        cv_risk_layer.ResetReading()
        cv_risk_layer.StartTransaction()
        LOGGER.info('calcualte Rts for all points')
        for index, feature in enumerate(cv_risk_layer):
            if index % 1000 == 0:
                LOGGER.info(
                    f'{index/n_features*100:.2f}% complete for {risk_field} '
                    f'{index+1}/{n_features}')

            rt = 1.0
            for risk_field in [
                    'Rgeomorphology', 'Rhab', 'Rsurge', 'Rwave', 'Rwind',
                    'Rslr', 'Rrelief']:
                rt *= feature.GetField(risk_field)
            rt = rt**(1/7)
            feature.SetField('Rt', rt)

            # calc the hab service
            biophysical_exposure_index = 1
            for risk_field in [
                    'Rgeomorphology', 'Rsurge', 'Rwave', 'Rwind',
                    'Rslr', 'Rrelief']:
                biophysical_exposure_index *= feature.GetField(risk_field)
            rt_nohab = (5*biophysical_exposure_index)**(1/7)
            rt_hab_service_index = rt_nohab-rt
            feature.SetField(
                'Rt_nohab', rt_nohab)
            feature.SetField(
                'Rt_hab_service_index', rt_hab_service_index)
            for hab_id in habitat_fieldname_list:
                knockout_hab_risk = feature.GetField(f'Rhab_no_{hab_id}')
                hab_knockout_exposure_index = (
                    knockout_hab_risk*biophysical_exposure_index)**(1/7)
                feature.SetField(
                    f'Rt_hab_{hab_id}_service_index',
                    hab_knockout_exposure_index-rt)

            cv_risk_layer.SetFeature(feature)
        cv_risk_layer.CommitTransaction()
        cv_risk_layer = None
        cv_risk_vector = None
    except Exception:
        LOGGER.exception(
            f'something bad happened when calculating risk on '
            f'{cv_risk_vector_path}')
    finally:
        LOGGER.info(f'all done adding risk to {cv_risk_vector_path}')


def mask_by_height_op(pop_array, dem_array, mask_height, pop_nodata):
    """Set pop to 0 if > height."""
    result = numpy.zeros(shape=pop_array.shape)
    valid_mask = (
        (dem_array < mask_height) & ~numpy.isclose(pop_array, pop_nodata))
    result[valid_mask] = pop_array[valid_mask]
    return result


def calculate_habitat_value(
        shore_sample_point_vector_path, scenario_config,
        beneficiary_id, bene_effective_dist, beneficiary_path,
        results_dir):
    """Calculate habitat value for a given beneficiary.

    Will create rasters in the `results_dir` directory named from the
    `habitat_vector_path_map + beneficiary_id` keys containing relative
    importance of that habitat in protecting that beneficiary. The higher the
    value of a pixel the more important that pixel of habitat is for protection
    of that beneficiary.

    Parameters:
        shore_sample_point_vector_path (str): path to CV analysis vector
            containing at least the fields `Rt` for all
            habitat types under consideration.
        scenario_config (configparser): scenario config with fields for
            habitat_map, lulc_raster_path, lulc_code_to_hab_map.
        beneficiary_id (str): unique ID to be used in filenaming for
            beneficiary
        bene_effective_dist (float): distance in meters of beneficiary effect on
            shore
        beneficiary_path (str): raster to beneficiary raster.
        results_dir (str): path to directory containing habitat back projection
            results

    Returns:
        None.
    """
    temp_workspace_dir = os.path.join(results_dir, 'hvc')
    for dir_path in [results_dir, temp_workspace_dir]:
        os.makedirs(dir_path, exist_ok=True)

    risk_distance_lucode_map = _parse_lulc_code_to_hab(eval(
        scenario_config['lulc_code_to_hab_map']))
    risk_dist_raster_map = _parse_habitat_map(eval(
        scenario_config['habitat_map']))
    task_graph = taskgraph.TaskGraph(
        temp_workspace_dir, len(risk_dist_raster_map))
    shore_point_info = geoprocessing.get_vector_info(
        shore_sample_point_vector_path)

    target_pixel_size = (
        eval(scenario_config['hab_val_pixel_size_wgs84']),
        -eval(scenario_config['hab_val_pixel_size_wgs84']))

    # make the habitat masks
    hab_raster_task = task_graph.add_task(
        func=clip_and_mask_habitat,
        args=(
            risk_distance_lucode_map,
            scenario_config['lulc_raster_path'],
            risk_dist_raster_map, shore_point_info['bounding_box'],
            osr.SRS_WKT_WGS84_LAT_LONG, target_pixel_size, temp_workspace_dir),
        store_result=True,
        task_name='clip and mask habitat layer')

    # clip beneficiary
    local_beneficiary_path = os.path.join(
        temp_workspace_dir, f'clipped_{beneficiary_id}.tif')
    beneficiary_clip_task = task_graph.add_task(
        func=clip_and_reproject_raster,
        args=(beneficiary_path, local_beneficiary_path,
              shore_point_info['projection_wkt'],
              shore_point_info['bounding_box'],
              0, 'near', False, target_pixel_size),
        target_path_list=[local_beneficiary_path],
        task_name=f'clip {local_beneficiary_path}')

    # project beneficiaries onto coverage
    beneficiary_coverage_raster_path = os.path.join(
        temp_workspace_dir, f'{beneficiary_id}_coverage.tif')
    beneficiary_effect = bene_effective_dist / 111320  # approximate
    kernel_radius = [abs(beneficiary_effect / x) for x in target_pixel_size]
    LOGGER.debug(f'************ kernel radisu {kernel_radius}')
    beneficiary_coverage_task = task_graph.add_task(
        func=convolve_2d,
        args=(
            local_beneficiary_path, kernel_radius, temp_workspace_dir,
            beneficiary_coverage_raster_path),
        target_path_list=[beneficiary_coverage_raster_path],
        dependent_task_list=[beneficiary_clip_task],
        task_name=(
            f'calculate beneficiary coverage for '
            f'{beneficiary_coverage_raster_path}'))

    habitat_raster_path_map = hab_raster_task.get()
    merged_habitat_raster_path_map = {}
    for (hab_id, _, hab_protective_radius), hab_raster_path_list in \
            habitat_raster_path_map.items():
        local_hab_raster_path = os.path.join(
            temp_workspace_dir, f'{hab_id}.tif')
        merge_mask_list(hab_raster_path_list, local_hab_raster_path)
        merged_habitat_raster_path_map[(hab_id, hab_protective_radius)] = \
            local_hab_raster_path

    beneficiary_coverage_task.join()
    beneficary_coverage_raster = gdal.OpenEx(
        beneficiary_coverage_raster_path, gdal.OF_RASTER)
    beneficary_coverage_band = beneficary_coverage_raster.GetRasterBand(1)

    gt = beneficary_coverage_raster.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    x_size = beneficary_coverage_band.XSize
    y_size = beneficary_coverage_band.YSize

    shore_point_vector = gdal.OpenEx(
        shore_sample_point_vector_path, gdal.OF_VECTOR)
    shore_point_layer = shore_point_vector.GetLayer()
    #TODO: get the hab list and hab mask in this function so I can access directly
    LOGGER.debug(f'*********** {merged_habitat_raster_path_map}')
    for (hab_id, hab_protective_radius), hab_mask_path in merged_habitat_raster_path_map.items():
        # TODO: make a new field for the {beneficiary_id}_{hab_id}_index
        local_beneficiary_hab_index_raster_path = os.path.join(
            temp_workspace_dir,
            f'pre_masked_{hab_id}_{beneficiary_id}_value_index.tif')
        geoprocessing.new_raster_from_base(
            beneficiary_coverage_raster_path,
            local_beneficiary_hab_index_raster_path, gdal.GDT_Float32, [-1])
        local_beneficiary_hab_index_raster = gdal.OpenEx(
            local_beneficiary_hab_index_raster_path,
            gdal.OF_RASTER | gdal.GA_Update)
        local_beneficiary_hab_index_band = \
            local_beneficiary_hab_index_raster.GetRasterBand(1)
        for shore_feature in shore_point_layer:
            # TODO: calcualte bene_hab_index, write to raster and to point

            point_geometry = shore_feature.GetGeometryRef()
            point_x, point_y = point_geometry.GetX(), point_geometry.GetY()
            point_geometry = None

            pixel_x, pixel_y = [
                int(x) for x in
                gdal.ApplyGeoTransform(inv_gt, point_x, point_y)]
            if pixel_x < 0:
                pixel_x = 0
            if pixel_y < 0:
                pixel_y = 0
            if pixel_x >= x_size:
                pixel_x = x_size-1
            if pixel_y >= y_size:
                pixel_y = y_size-1
            try:
                pixel_value = beneficary_coverage_band.ReadAsArray(
                    xoff=pixel_x, yoff=pixel_y, win_xsize=1,
                    win_ysize=1)[0, 0]
            except Exception:
                LOGGER.exception(
                    'beneficary_coverage_band size %d %d', beneficary_coverage_band.XSize,
                    beneficary_coverage_band.YSize)
                raise
            service_id = f'Rt_hab_{hab_id}_service_index'
            try:
                service_index_value = shore_feature.GetField(service_id)
            except KeyError:
                LOGGER.exception(f'error on {service_id}')
                raise
            target_value = pixel_value * service_index_value
            local_beneficiary_hab_index_band.WriteArray(
                numpy.array([[target_value]]),
                xoff=pixel_x, yoff=pixel_y)

        local_beneficiary_hab_index_band = None
        local_beneficiary_hab_index_raster = None

        pre_mask_value_raster_path = os.path.join(
            temp_workspace_dir, f'pre_masked_{hab_id}_{beneficiary_id}_value_index_coverage.tif')
        kernel_radius = [
            abs(hab_protective_radius / x / 111320) for x in target_pixel_size]
        beneficiary_coverage_task = task_graph.add_task(
            func=convolve_2d,
            args=(
                local_beneficiary_hab_index_raster_path, kernel_radius,
                temp_workspace_dir, pre_mask_value_raster_path),
            target_path_list=[pre_mask_value_raster_path],
            task_name=(
                f'calculate beneficiary hab coverage for '
                f'{pre_mask_value_raster_path}'))

        # TODO: mask it by the original habitat mask, the result is the bene hab value
        value_raster_path = os.path.join(
            results_dir, f'{hab_id}_{beneficiary_id}_value_index.tif')
        task_graph.add_task(
            func=mask_raster,
            args=(pre_mask_value_raster_path, hab_mask_path,
                  value_raster_path),
            dependent_task_list=[beneficiary_coverage_task],
            target_path_list=[value_raster_path],
            task_name=f'mask final {value_raster_path}')

    task_graph.join()
    task_graph.close()


def mask_raster(value_raster_path, mask_raster_path, target_raster_path):
    target_nodata = -1

    def mask_op(base_array, mask_array):
        result = base_array.copy()
        result[mask_array != 1] = target_nodata
        return result

    geoprocessing.raster_calculator(
        [(value_raster_path, 1), (mask_raster_path, 1)], mask_op,
        target_raster_path, gdal.GDT_Float32, target_nodata)


def _process_hab_value(
        shore_sample_point_vector_path, key, temp_workspace_dir,
        hab_mask_raster_path_list, target_pixel_size, value_raster_path):

    LOGGER.debug(f'********* processing hab on {shore_sample_point_vector_path}')
    (hab_id, risk, protective_distance_m) = key
    protective_distance_d = protective_distance_m / 111320  # approximate
    habitat_service_id = f'Rt_hab_{hab_id}_service_index'

    # TODO: locally clip and project everything
    shore_point_info = geoprocessing.get_vector_info(
        shore_sample_point_vector_path)
    local_clip_stack = []

    for hab_raster_path in hab_mask_raster_path_list:
        basename = os.path.basename(os.path.splitext(hab_raster_path)[0])
        local_clip_raster_path = os.path.join(
            temp_workspace_dir, f'{basename}.tif')
        clip_and_reproject_raster(
            hab_raster_path, local_clip_raster_path,
            shore_point_info['projection_wkt'],
            shore_point_info['bounding_box'], protective_distance_d,
            'near', False, target_pixel_size)
        local_clip_stack.append(local_clip_raster_path)
    local_hab_raster_path = os.path.join(
        temp_workspace_dir, f'{hab_id}_merged_mask.tif')
    merge_mask_list(local_clip_stack, local_hab_raster_path)

    # TODO: make the raster to rasterize on
    hab_service_source_raster_path = os.path.join(
        temp_workspace_dir, f'{hab_id}_value_cover.tif')
    geoprocessing.new_raster_from_base(
        local_hab_raster_path, hab_service_source_raster_path,
        gdal.GDT_Float32, [_VALUE_COVER_NODATA],
        raster_driver_creation_tuple=(
            'GTIFF', (
                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                'BLOCKXSIZE=256', 'BLOCKYSIZE=256',)))

    # TODO: rasterize shore_sample_point_vector_path's habitat_service_id
    geoprocessing.rasterize(
        shore_sample_point_vector_path, hab_service_source_raster_path,
        option_list=[f'ATTRIBUTE={habitat_service_id}'])
    kernel_filepath = os.path.join(temp_workspace_dir, f'{key}_kernel.tif')
    kernel_radius = [abs(protective_distance_d / x) for x in target_pixel_size]
    create_averaging_kernel_raster(
        kernel_radius, kernel_filepath, normalize=True)
    hab_coverage_raster_path = os.path.join(
        temp_workspace_dir, f'{key}_hab_coverage.tif')

    # convolve it
    geoprocessing.convolve_2d(
        (hab_service_source_raster_path, 1), (kernel_filepath, 1),
        hab_coverage_raster_path, mask_nodata=False)

    # mask the convolution to the corresponding habitat mask
    target_nodata = -1
    def mask_op(base_array, mask_array):
        result = base_array.copy()
        result[mask_array != 1] = target_nodata
        return result

    geoprocessing.raster_calculator(
        [(hab_coverage_raster_path, 1), (local_hab_raster_path, 1)], mask_op,
        value_raster_path, gdal.GDT_Float32, target_nodata)


def intersect_raster_op(array_a, array_b, array_knockout, nodata_a, nodata_b):
    """Only return values from a where a and b are defined."""
    result = numpy.empty_like(array_a)
    result[:] = nodata_a
    valid_mask = (
        ~numpy.isclose(array_a, nodata_a) &
        ~numpy.isclose(array_b, nodata_b) &
        (array_knockout != 1))
    result[valid_mask] = array_a[valid_mask]
    return result


def align_raster_list(raster_path_list, target_directory):
    """Aligns all the raster paths.

    Rasters are aligned using the pixel size of the first raster and use
    the intersection and near interpolation methods.

    Parameters:
        raster_path_list (list): list of str paths to rasters.
        target_directory (str): path to a directory to hold the aligned
            rasters.

    Returns:
        list of raster paths that are aligned with intersection and near
            interpolation algorithm.

    """
    if not hasattr(align_raster_list, 'task_graph_map'):
        align_raster_list.task_graph_map = {}
    if target_directory not in align_raster_list.task_graph_map:
        align_raster_list.task_graph_map[target_directory] = (
            taskgraph.TaskGraph(target_directory, -1))
    task_graph = align_raster_list.task_graph_map[target_directory]
    aligned_path_list = [
        os.path.join(target_directory, f'aligned_{os.path.basename(path)}')
        for path in raster_path_list]
    target_pixel_size = geoprocessing.get_raster_info(
        raster_path_list[0])['pixel_size']
    LOGGER.debug(
        f'about to align: {raster_path_list} in directory {target_directory}')
    task_graph.add_task(
        func=geoprocessing.align_and_resize_raster_stack,
        args=(
            raster_path_list, aligned_path_list,
            ['near'] * len(raster_path_list), target_pixel_size,
            'intersection'),
        target_path_list=aligned_path_list,
        task_name=(
            f'align raster list for {raster_path_list} to '
            f'{aligned_path_list}'))
    return aligned_path_list


def clip_and_mask_habitat(
        risk_dist_lucode_map, lulc_raster_path, risk_dist_raster_map,
        target_bb, target_projection_wkt, target_pixel_size, workspace_dir):
    """Extract out lulc rasters given lulc code and risk/distance mappings.

    This function is used when a worker is given a local bounding box to
    to process for CV.

    Args:
        risk_dist_lucode_map (dict): dictionary that maps (hab_id, risk, dist)
            tuples to a list of landcover codes that match
            ``lulc_raster_path``.
        lulc_raster_path (str): path to landcover raster with codes that are
            in the lists of ``risk_dist_lucode_map``.
        risk_dist_raster_map (dict): mapping of (hab_id, risk, dist) tuples to
            raster binary masks indicating where those risks are
        target_bb (list): this is the desired target bounding box in the
            same coordinate system as ``target_projection_wkt``.
        target_projection_wkt (str): desired target projection system
        target_pixel_size (tuple): desired target pixel size
        workspace_dir (str): path to a directory that can be used to create
            intermediate files for calculation

    Returns:
        dict mapping (risk, dist) tuples to clipped local paths of rasters
            which are 1 where the original lulc matches the input lulc.
    """
    # First process the LULC raster into len(risk_dist_lucode_map) risk
    # masks
    local_lulc_path = os.path.join(
        workspace_dir, f'''{os.path.basename(os.path.splitext(
            lulc_raster_path)[0])}_{target_bb}.tif''')
    geoprocessing.warp_raster(
        lulc_raster_path, target_pixel_size, local_lulc_path,
        'mode', target_bb=target_bb,
        target_projection_wkt=target_projection_wkt,
        working_dir=workspace_dir)

    reclassify_threads = []
    habitat_raster_risk_map = collections.defaultdict(list)
    LOGGER.debug(f'risk dist lucode tuple: {risk_dist_lucode_map}')
    for hab_id_risk_distance_tuple, risk_lucodes in \
            risk_dist_lucode_map.items():
        reclass_map = {
            val: 1 for val in risk_lucodes
        }
        lulc_risk_distance_mask_path = os.path.join(
            workspace_dir, f'lulc_{hab_id_risk_distance_tuple}_mask.tif')

        reclassify_thread = threading.Thread(
            target=geoprocessing.reclassify_raster,
            args=(
                (local_lulc_path, 1), reclass_map,
                lulc_risk_distance_mask_path, gdal.GDT_Byte, 0),
            daemon=True)
        reclassify_thread.start()
        reclassify_threads.append(reclassify_thread)
        habitat_raster_risk_map[hab_id_risk_distance_tuple].append(
            lulc_risk_distance_mask_path)

    LOGGER.debug(risk_dist_raster_map)
    for hab_id_risk_distance_tuple, mask_path_list in risk_dist_raster_map.items():
        for mask_path in mask_path_list:
            local_mask_path = os.path.join(
                workspace_dir, os.path.basename(mask_path))
            geoprocessing.warp_raster(
                mask_path, target_pixel_size, local_mask_path,
                'mode', target_bb=target_bb,
                target_projection_wkt=target_projection_wkt,
                working_dir=workspace_dir)
            habitat_raster_risk_map[hab_id_risk_distance_tuple].append(
                local_mask_path)

    for reclassify_thread in reclassify_threads:
        reclassify_thread.join()

    return habitat_raster_risk_map


def _clip_vector(
        base_vector_path, target_vector_path, target_bb):
    """Clip base vector to target with the given bounding box."""
    start_time = time.time()
    clip_options = gdal.VectorTranslateOptions(
        format='GPKG', spatFilter=target_bb)
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    gdal.VectorTranslate(target_vector_path, base_vector, options=clip_options)
    LOGGER.debug(f'{time.time()-start_time:.2f}s took this long')


def calculate_degree_cell_cv(
        local_data_path_map, target_cv_vector_path, local_workspace_dir):
    """Process all global degree grids to calculate local hab risk.

    Paramters:
        local_data_path_map (dict): maps keys from GLOBAL_DATA_URL to the
            local filepaths.
        shore_point_sample_distance (float): straight line distance between
            two sample points on the shore points.
        target_cv_vector_path (str): path to desiered point CV result.
        local_workspace_dir (str): directory to write local grid workspaces

    Returns:
        None
    """
    LOGGER.info('just checking that lulcode map is correctly defined')
    _ = _parse_lulc_code_to_hab(eval(
        local_data_path_map['lulc_code_to_hab_map']))

    shore_grid_vector = gdal.OpenEx(
        local_data_path_map['shore_grid_vector_path'], gdal.OF_VECTOR)
    shore_grid_layer = shore_grid_vector.GetLayer()

    bb_work_queue = multiprocessing.Queue()
    cv_point_complete_queue = multiprocessing.Queue()

    cv_grid_worker_list = []
    grid_workspace_dir = os.path.join(local_workspace_dir, 'grid_workspaces')

    shore_grid_vector = gdal.OpenEx(
        local_data_path_map['shore_grid_vector_path'], gdal.OF_VECTOR)
    shore_grid_layer = shore_grid_vector.GetLayer()

    lulc_bb_box = None
    if 'lulc_raster_path' in local_data_path_map:
        lulc_raster_info = geoprocessing.get_raster_info(
            local_data_path_map['lulc_raster_path'])
        lulc_wgs84_bb = geoprocessing.transform_bounding_box(
            lulc_raster_info['bounding_box'],
            lulc_raster_info['projection_wkt'],
            osr.SRS_WKT_WGS84_LAT_LONG)
        lulc_bb_box = shapely.geometry.box(*lulc_wgs84_bb)
        # clip all the inputs to the lulc bounding box
        clipped_dir = os.path.join(local_workspace_dir, 'clipped')
        os.makedirs(clipped_dir, exist_ok=True)
        worker_list = []
        for vector_id in [
                'wwiii_vector_path', 'geomorphology_vector_path',
                'landmass_vector_path', 'buffer_vector_path',
                'shore_grid_vector_path']:
            vector_path = local_data_path_map[vector_id]
            clipped_vector_path = os.path.join(
                clipped_dir, os.path.basename(vector_path))
            # buffer out the vector bounding box in case there are vector
            # data that might need to be in the clip
            vector_wgs84_bb = [
                lulc_wgs84_bb[0]-1,
                lulc_wgs84_bb[1]-1,
                lulc_wgs84_bb[2]+1,
                lulc_wgs84_bb[3]+1,
            ]
            clip_thread = threading.Thread(
                daemon=True,
                target=_clip_vector,
                args=(vector_path, clipped_vector_path, vector_wgs84_bb))
            LOGGER.debug(f'about to start {vector_id}')
            clip_thread.start()
            worker_list.append((vector_id, clip_thread))
            local_data_path_map[vector_id] = clipped_vector_path

        for raster_id in ['slr_raster_path', 'dem_raster_path']:
            raster_path = local_data_path_map[raster_id]
            raster_info = geoprocessing.get_raster_info(raster_path)
            clipped_raster_path = os.path.join(
                clipped_dir, os.path.basename(raster_path))
            clip_thread = threading.Thread(
                daemon=True,
                target=geoprocessing.warp_raster,
                args=(
                    raster_path, raster_info['pixel_size'],
                    clipped_raster_path, 'near'),
                kwargs={
                    'target_bb': lulc_wgs84_bb, 'working_dir': clipped_dir})
            LOGGER.debug(f'about to start {raster_id}')
            clip_thread.start()
            local_data_path_map[raster_id] = clipped_raster_path
            worker_list.append((raster_id, clip_thread))
        for worker_id, worker_thread in worker_list:
            LOGGER.debug(f'waiting for {worker_id} to finish')
            worker_thread.join()

    n_boxes = 0
    for index, shore_grid_feature in enumerate(shore_grid_layer):
        if 'shore_grid_fid' in local_data_path_map:
            if shore_grid_feature.GetFID() != int(local_data_path_map['shore_grid_fid']):
                continue
        shore_grid_geom = shore_grid_feature.GetGeometryRef()
        boundary_box = shapely.wkb.loads(
            bytes(shore_grid_geom.ExportToWkb()))
        if (lulc_bb_box is not None and
                not boundary_box.intersects(lulc_bb_box)):
            continue
        bb_work_queue.put((index, boundary_box.bounds))
        n_boxes += 1

    for worker_id in range(min(MAX_WORKERS, n_boxes+1, int(multiprocessing.cpu_count()))):
        cv_grid_worker_thread = multiprocessing.Process(
            target=cv_grid_worker,
            args=(
                bb_work_queue,
                cv_point_complete_queue,
                local_data_path_map,
                grid_workspace_dir,
                ))
        cv_grid_worker_thread.start()
        cv_grid_worker_list.append(cv_grid_worker_thread)
        LOGGER.debug('starting worker %d', worker_id)

    shore_grid_vector = None
    shore_grid_layer = None

    bb_work_queue.put(STOP_SENTINEL)

    habitat_fieldname_list = list(eval(
        local_data_path_map['habitat_map']).keys())
    risk_dist_raster_map = _parse_lulc_code_to_hab(eval(
        local_data_path_map['lulc_code_to_hab_map']))

    for key in risk_dist_raster_map:
        LOGGER.debug(key)
        habitat_fieldname_list.append(key[0])

    merge_cv_points_and_add_risk_thread = threading.Thread(
        target=merge_cv_points_and_add_risk,
        args=(cv_point_complete_queue, n_boxes,
              list(set(habitat_fieldname_list)),
              target_cv_vector_path))
    merge_cv_points_and_add_risk_thread.start()

    for cv_grid_worker_thread in cv_grid_worker_list:
        cv_grid_worker_thread.join()

    # when workers are complete signal merger complete
    cv_point_complete_queue.put(STOP_SENTINEL)
    merge_cv_points_and_add_risk_thread.join()

    LOGGER.debug('calculate cv vector risk')
    add_cv_vector_risk(
        list(set(habitat_fieldname_list)), target_cv_vector_path)


def _process_scenario_ini(scenario_config_path):
    """Verify that the ini file has correct structure.

    Args:
        ini_config (configparser obj): config parsed object with
            'expected_keys' and '`scenario_id`' as fields.

    Returns
        (configparser object, scenario id)

    Raises errors if config file not formatted correctly
    """
    global_config = configparser.ConfigParser(allow_no_value=True)
    global_config.read(GLOBAL_INI_PATH)
    global_config.read(scenario_config_path)
    scenario_id = os.path.basename(os.path.splitext(scenario_config_path)[0])
    if scenario_id not in global_config:
        raise ValueError(
            f'expected a section called [{scenario_id}] in configuration file'
            f'but was not found')
    scenario_config = global_config[scenario_id]
    missing_keys = []
    for key in global_config['expected_keys']:
        if key not in scenario_config:
            missing_keys.append(key)
    if missing_keys:
        raise ValueError(
            f'expected the following keys in "{scenario_config_path}" '
            f'but not found: "{", ".join(missing_keys)}"')
    LOGGER.debug(scenario_config)
    for key in scenario_config:
        if key.endswith('_path'):
            possible_path = scenario_config[key]
            if not os.path.exists(possible_path):
                raise ValueError(
                    f'expected a file from "{key}" at "{possible_path}" '
                    f'but file not found')

    for _, _, hab_path_list in eval(scenario_config[HABITAT_MAP_KEY]).values():
        if not isinstance(hab_path_list, list):
            hab_path_list = [hab_path_list]
        for hab_path in hab_path_list:
            if not os.path.exists(hab_path):
                raise ValueError(
                    f'expected a habitat raster at "{hab_path}" but one '
                    f'not found')

    for _, beneficiary_path in eval(
            scenario_config[BENEFICIARIES_MAP_KEY]).values():
        if not os.path.exists(beneficiary_path):
            raise ValueError(
                f'expected a beneficiary raster at "{beneficiary_path}" but '
                f'nothing was found')
    return scenario_config, scenario_id


def _parse_habitat_map(habitat_raster_path_map):
    risk_dist_raster_map = collections.defaultdict(list)
    for hab_id, risk_dist_path_tuple in habitat_raster_path_map.items():
        hab_list = risk_dist_path_tuple[2]
        if isinstance(hab_list, str):
            hab_list = [hab_list]
        risk_dist_raster_map[
            (hab_id, risk_dist_path_tuple[0], risk_dist_path_tuple[1])] = (
                hab_list)

    return risk_dist_raster_map


def _parse_lulc_code_to_hab(lulc_code_to_hab_map):
    # map a set of (hab_id, risk, dist) tuples to lists of landcover codes that
    # match them
    risk_distance_lucode_map = collections.defaultdict(list)
    for lucode, id_risk_dist_tuple in lulc_code_to_hab_map.items():
        if not (isinstance(id_risk_dist_tuple, tuple) and
                len(id_risk_dist_tuple) == 3 and
                isinstance(id_risk_dist_tuple[0], str) and
                isinstance(id_risk_dist_tuple[1], Number) and
                isinstance(id_risk_dist_tuple[2], Number)):
            raise ValueError(
                f'from the set {lulc_code_to_hab_map} expected only '
                f'(hab_id, risk number, dist number) tuples '
                f'but got this value instead "{id_risk_dist_tuple}", '
                f'could it be that it is quoted as a string on accident?')
        risk_distance_lucode_map[id_risk_dist_tuple].append(lucode)
    return risk_distance_lucode_map


def main():
    LOGGER.debug('starting')
    parser = argparse.ArgumentParser(description='Global CV analysis')
    parser.add_argument(
        'scenario_config_path', nargs='+',
        help='Pattern to .INI file(s) that describes scenario(s) to run.')
    args = parser.parse_args()

    scenario_config_path_list = [
        path for pattern in args.scenario_config_path
        for path in glob.glob(pattern)]
    LOGGER.info(f'''parsing and validating {
        len(scenario_config_path_list)} configuration files''')
    config_scenario_list = []
    for scenario_config_path in scenario_config_path_list:
        scenario_config, scenario_id = _process_scenario_ini(
            scenario_config_path)
        config_scenario_list.append((scenario_config, scenario_id))

    for scenario_config, scenario_id in config_scenario_list:
        workspace_dir = scenario_config['workspace_dir']
        os.makedirs(workspace_dir, exist_ok=True)
        task_graph = taskgraph.TaskGraph(workspace_dir, -1)
        local_workspace_dir = os.path.join(workspace_dir, scenario_id)
        local_habitat_value_dir = os.path.join(
            workspace_dir, scenario_id, 'value_rasters')
        for dir_path in [local_workspace_dir, local_habitat_value_dir]:
            os.makedirs(dir_path, exist_ok=True)
        target_cv_vector_path = os.path.join(
            local_workspace_dir, '%s.gpkg' % scenario_id)

        calculate_cv_vector_task = task_graph.add_task(
            func=calculate_degree_cell_cv,
            args=(
                scenario_config,
                target_cv_vector_path, local_workspace_dir),
            ignore_path_list=[target_cv_vector_path],
            target_path_list=[target_cv_vector_path],
            task_name=f'calculate CV for {scenario_id}')
        calculate_cv_vector_task.join()

        LOGGER.info('************starting hab value calc on ' + scenario_config[BENEFICIARIES_MAP_KEY])
        for beneficiary_id, (effective_dist, beneficiary_path) in \
                eval(scenario_config[BENEFICIARIES_MAP_KEY]).items():
            calculate_habitat_value(
                target_cv_vector_path, scenario_config, beneficiary_id,
                effective_dist, beneficiary_path, local_habitat_value_dir)

    task_graph.join()
    task_graph.close()
    LOGGER.info('completed successfully')


if __name__ == '__main__':
    main()
