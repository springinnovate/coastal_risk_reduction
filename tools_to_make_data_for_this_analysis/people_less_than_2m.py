import logging
import sys
import os
import tempfile
from ecoshard import geoprocessing
from ecoshard import taskgraph
import shutil

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


POPULATION_RASTER_PATH_LIST = [
    r"D:\repositories\wwf-sipa\data\pop\phl_ppp_2020.tif",
    r"D:\repositories\wwf-sipa\data\pop\idn_ppp_2020.tif"]
DEM_PATH = r"D:\repositories\wwf-sipa\data\aster_dem\aster_dem.vrt"
TARGET_DIR = r"D:\repositories\wwf-sipa\data\pop"


def mask_to_2m(pop_path, dem_path, target_path):
    raster_info = geoprocessing.get_raster_info(pop_path)['nodata'][0]
    nodata = raster_info['nodata'][0]
    if nodata is None:
        nodata = -9999
    def less_than_2m(pop, dem):
        result = pop.copy()
        result[dem <= 2] = nodata
        return result

    geoprocessing.raster_calculator(
        [(pop_path, 1), dem_path, 1], less_than_2m, target_path,
        raster_info['datatype'], nodata)


def main():
    task_graph = taskgraph.TaskGraph(TARGET_DIR, 2, 15.0)
    temp_dir_list = []
    for pop_path in POPULATION_RASTER_PATH_LIST:
        working_dir = tempfile.mkdtemp(prefix='2mpop', dir=TARGET_DIR)
        temp_dir_list.append(working_dir)
        base_path_list = [pop_path, DEM_PATH]
        align_path_list = [
            os.path.join(
                working_dir,
                f'less_than_2m_{os.path.basename(path)}')
            for path in base_path_list]
        target_pixel_size = geoprocessing.get_raster_info(
            pop_path)['pixel_size']
        align_task = task_graph.add_task(
            func=geoprocessing.align_and_resize_raster_stack,
            args=(
                base_path_list, align_path_list, ['near']*2,
                target_pixel_size, 'intersection'),
            target_path_list=align_path_list,
            task_name=f'align {pop_path}')
        target_path = os.path.join(
            TARGET_DIR, f'less_than_2m_{os.path.basename(pop_path)}')
        task_graph.add_task(
            func=mask_to_2m,
            args=(
                align_path_list[0], align_path_list[1], target_path),
            target_path_list=[target_path],
            dependent_task_list=[align_task],
            task_name=f'mask to 2m {target_path}')
    task_graph.join()
    task_graph.close()
    for temp_dir in temp_dir_list:
        shutil.rmtree(temp_dir)
    LOGGER.info(f'ALL DONE! results in {TARGET_DIR}')


if __name__ == '__main__':
    main()
