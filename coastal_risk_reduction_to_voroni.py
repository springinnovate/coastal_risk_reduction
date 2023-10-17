"""Convert CV points to shore wedges."""
import argparse
import logging
import os
import sys

from osgeo import gdal
from osgeo import ogr
from shapely.geometry import MultiPoint, GeometryCollection, Point
from shapely.ops import voronoi_diagram
import shapely.ops
import shapely.errors
import shapely.geometry
import shapely.strtree

gdal.SetCacheMax(2**20)
ogr.UseExceptions()

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


def voroni_polygons_from_points(
        point_vector_path, max_sample_distance,
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
    LOGGER.info('building the point list')
    for feature in point_layer:
        geom = feature.GetGeometryRef()
        points_list.append((geom.GetX(), geom.GetY()))
    points = MultiPoint(points_list)
    LOGGER.info('creating the buffer points')
    buffer_list = []
    for feature in point_layer:
        geom = feature.GetGeometryRef()
        buffer_list.append(
            Point(geom.GetX(), geom.GetY()).buffer(max_sample_distance))
    LOGGER.info('creating the buffer union')
    points_buffer_area = shapely.unary_union(buffer_list)

    # save the buffer
    shore_buffer_vector = driver.CreateDataSource('shorebuffer.gpkg')
    shore_buffer_layer = shore_buffer_vector.CreateLayer(
        'shore buffer', geom_type=ogr.wkbPolygon, srs=srs)
    for poly in points_buffer_area.geoms:
        base_feature = ogr.Feature(shore_buffer_layer.GetLayerDefn())
        geom = ogr.CreateGeometryFromWkb(poly.wkb)
        base_feature.SetGeometry(geom)
        shore_buffer_layer.CreateFeature(base_feature)
        base_feature = None

    LOGGER.info('generating voronoi diagram')
    regions = voronoi_diagram(points, envelope=points_buffer_area.envelope)
    LOGGER.info('clipping voronoi diagram to aoi')
    cleaned_regions = []
    for index, geom in enumerate(regions.geoms):
        if index % 100 == 0:
            LOGGER.debug(f'{index/len(regions.geoms)*100:.2f}% complete')
        cleaned_regions.append(geom.intersection(points_buffer_area))
    regions = GeometryCollection(cleaned_regions)

    LOGGER.info('saving polygons to disk')
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

    aoi_feature = None
    shore_point_aoi_layer = None
    shore_point_aoi_vector = None
    point_vector = None

def main():
    LOGGER.debug('starting')
    parser = argparse.ArgumentParser(description='Global CV analysis')
    parser.add_argument(
        'shore_point_vector_path')
    parser.add_argument(
        'max_sample_distance', type=float)
    parser.add_argument(
        'shore_wedge_vector_path')
    args = parser.parse_args()

    voroni_polygons_from_points(
        args.shore_point_vector_path,
        args.max_sample_distance,
        args.shore_wedge_vector_path)

if __name__ == '__main__':
    main()
