import os
from ecoshard import geoprocessing
from osgeo import osr
from osgeo import ogr
import shapely.geometry


def _clip_vector(
        base_vector_path, target_vector_path, target_bb):

    base_vector = ogr.Open(base_vector_path)
    base_layer = base_vector.GetLayer()
    base_layer_dfn = base_layer.GetLayerDefn()
    basename = os.path.splitext(os.path.basename(base_vector_path))[0]
    driver = ogr.GetDriverByName('GPKG')
    vector = driver.CreateDataSource(target_vector_path)
    target_layer = vector.CreateLayer(
        basename, base_layer.GetSpatialRef(), base_layer.GetGeomType())

    field_names = []
    for fld_index in range(base_layer_dfn.GetFieldCount()):
        original_field = base_layer_dfn.GetFieldDefn(fld_index)
        field_name = original_field.GetName()
        target_field = ogr.FieldDefn(
            field_name, original_field.GetType())
        target_layer.CreateField(target_field)
        field_names.append(field_name)

    filter_box = shapely.geometry.box(*target_bb)
    base_layer.SetSpatialFilter(ogr.CreateGeometryFromWkt(filter_box.wkt))

    target_layer.StartTransaction()
    for base_feature in base_layer:
        geom = base_feature.GetGeometryRef()

        # Copy original_datasource's feature and set as new shapes feature
        target_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_feature.SetGeometry(geom)

        # For all the fields in the feature set the field values from the
        # source field
        for field_name in field_names:
            target_feature.SetField(
                field_name, base_feature.GetField(field_name))

        target_layer.CreateFeature(target_feature)
        target_feature = None
        base_feature = None
    target_layer.CommitTransaction()


if __name__ == '__main__':
    dir_path = './data/clipped/'
    os.makedirs(dir_path, exist_ok=True)
    bb = [118.532, 11.111, 125.505, 21.728]

    vector_path_list = [
        './data/basedata/wave_watch_iii.gpkg',
        './data/basedata/geomorphology_md5_e65eff55840e7a80cfcb11fdad2d02d7.gpkg',
        './data/basedata/ipbes-cv_global_polygon_simplified_geometries_md5_653118dde775057e24de52542b01eaee.gpkg',
        './data/basedata/buffered_global_shore_5km_md5_a68e1049c1c03673add014cd29b7b368.gpkg',
        './data/basedata/shore_grid_md5_07aea173cf373474c096f1d5e3463c2f.gpkg',
        ]

    for vector_path in vector_path_list:
        print(f'clipping {vector_path}')
        clipped_vector_path = os.path.join(
            dir_path, f'{os.path.basename(vector_path)}')
        vector_info = geoprocessing.get_vector_info(vector_path)
        target_bb = geoprocessing.transform_bounding_box(
            bb, osr.SRS_WKT_WGS84_LAT_LONG, vector_info['projection_wkt'])

        _clip_vector(vector_path, clipped_vector_path, target_bb)

    raster_path_list = [
        './data/basedata/MSL_Map_MERGED_Global_AVISO_NoGIA_Adjust_md5_3072845759841d0b2523d00fe9518fee.tif',
        './data/basedata/global_dem_3s_md5_22d0c3809af491fa09d03002bdf09748/global_dem_3s/srtm.vrt',
        './data/basedata/ipbes-cv_reef_wgs84_compressed_md5_96d95cc4f2c5348394eccff9e8b84e6b.tif',
        './data/basedata/ipbes-cv_mangrove_md5_0ec85cb51dab3c9ec3215783268111cc.tif',
        './data/basedata/ipbes-cv_seagrass_md5_a9cc6d922d2e74a14f74b4107c94a0d6.tif',
        './data/basedata/ipbes-cv_saltmarsh_md5_203d8600fd4b6df91f53f66f2a011bcd.tif',
        ]

    for raster_path in raster_path_list:
        print(f'clipping {raster_path}')
        clipped_raster = os.path.join(dir_path, os.path.basename(raster_path))
        raster_info = geoprocessing.get_raster_info(raster_path)
        target_bb = geoprocessing.transform_bounding_box(
            bb, osr.SRS_WKT_WGS84_LAT_LONG, raster_info['projection_wkt'])
        geoprocessing.warp_raster(
            raster_path, raster_info['pixel_size'], clipped_raster,
            'near', target_bb=target_bb,
            working_dir=dir_path)
