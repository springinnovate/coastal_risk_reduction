import logging
import sys
import os
import tempfile
from ecoshard import geoprocessing
from ecoshard import taskgraph
import shutil
from osgeo import gdal

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


WORK_LIST = [
    ('PH', r"D:\repositories\wwf-sipa\data\pop\phl_ppp_2020.tif", r"D:\repositories\wwf-sipa\data\infrastructure_polygons\PH_All_Roads_Merged.gpkg"),
    ('IDN', r"D:\repositories\wwf-sipa\data\pop\idn_ppp_2020.tif", r"D:\repositories\wwf-sipa\data\infrastructure_polygons\IDN_All_Roads_Merged.gpkg"),
    ]
TARGET_DIR = r"D:\repositories\wwf-sipa\data\infrastructure_rasters"


def rasterize(base_raster_path, vector_path, target_raster_path):
    geoprocessing.new_raster_from_base(
        base_raster_path, target_raster_path, gdal.GDT_Byte, [0])
    geoprocessing.rasterize(
        vector_path, target_raster_path, burn_values=[1])


def main():
    task_graph = taskgraph.TaskGraph(TARGET_DIR, 2, 15.0)
    temp_dir_list = []
    for prefix, base_path, vector_path in WORK_LIST:
        working_dir = tempfile.mkdtemp(prefix=f'{prefix}_', dir=TARGET_DIR)
        temp_dir_list.append(working_dir)
        target_path = os.path.join(
            TARGET_DIR,
            f'{os.path.splitext(os.path.basename(vector_path))[0]}.tif')
        _ = task_graph.add_task(
            func=rasterize,
            args=(
                base_path, vector_path, target_path),
            target_path_list=[target_path],
            task_name=f'rasterize {target_path}')
    task_graph.join()
    task_graph.close()
    for temp_dir in temp_dir_list:
        shutil.rmtree(temp_dir)
    LOGGER.info(f'ALL DONE! results in {TARGET_DIR}')


if __name__ == '__main__':
    main()
