import os, \
    shutil, \
    requests, \
    sys

from glob import glob
import re
import natsort

from bs4 import BeautifulSoup

import pandas as pd
import numpy
import matplotlib.pyplot as plt

from osgeo import gdal
from osgeo import gdal_array

import earthpy.spatial as es
import earthpy.plot as ep

from sklearn import cluster


def filterImages(path, row, clouds, startDate, endDate, details, max_scenes):
    """
    This function filters Landsat 8 data available on the s3 server.
    :param path: WRS-2 path; integer from 1 to 233
    :param row: WRS-2 row; integer from 1 to 248
    :param clouds: acceptable proportion of cloud cover within the image
    :param startDate: date before which no images will be listed; the start date; format: YYYY-MM-dd
    :param endDate: date after which no images will be listed; the end date; format: YYYY-MM-dd
    :param details: boolean; either print or skip printing the fully qualified details of each image.
    :param max_scenes: the total number of scenes to find that meet all of the above criteria
    :return:
    """
    s3_scenes = pd.read_csv(
        'http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz',
        compression='gzip'
    )

    scenes_to_class = []

    # filter scenes and drop duplicate productIds
    scenes = s3_scenes[(s3_scenes.path == path) & (s3_scenes.row == row) &
                       (s3_scenes.cloudCover <= clouds) &
                       (s3_scenes.productId.str.contains('_T1') &
                       (s3_scenes.acquisitionDate >= startDate) &
                       (s3_scenes.acquisitionDate <= endDate))].drop_duplicates()

    # Sort the least cloudiest images to the top
    if len(scenes):
        scene = scenes.sort_values('cloudCover')
    else:
        print("All images found had more than the acceptable amount of could cover - try again.")
        sys.exit()

    scenes_to_class.append(scene.head(max_scenes))

    if len(scenes) == 0:
        print("No images found matching the criteria.")
    elif len(scenes) < max_scenes:
        print("Only {} scenes were found.".format(scenes_to_class))
    else:
        print("Found {} requested images. \n".format(max_scenes))

    if details:
        row_counter = 1

        for row in scene.head(max_scenes).iterrows():
            print('################################################################################# \n'
                  'Details for ', str(scene.productId.iloc[0]), ' No. ', str(row_counter), ' of ', str(max_scenes))
            row_counter += 1
            print(row)

    return scene.head(max_scenes)


def downloadImages(pathname):
    """
    This function
    :param pathname:
    :return:
    """
    # Get user input to define the image set
    user_path = input("""Choose a valid Landsat 8 WRS-2 PATH value (1 - 233): \n""")
    path = int(user_path)

    user_row = input("""Choose a valid Landsat 8 WRS-2 ROW value (1 - 248): \n""")
    row = int(user_row)

    user_cloud = input("""Choose acceptable proportion of cloud cover: \n""")
    clouds = int(user_cloud)

    user_startDate = input("""Select the earliest acquisition date needed for your study:
    (formatted as YYYY-MM-dd) \n""")
    startDate = str(user_startDate)

    user_endDate = input("""Select the latest acquisition date needed for your study:
    (formatted as YYYY-MM-dd) \n""")
    endDate = str(user_endDate)

    user_max = input("""Select the maximum number of scenes to find: \n""")
    print("Keep in mind that scenes can take a while to download, so it is best to limit the maximum "
          "number of scenes to ~ 5. ")
    max_scenes = int(user_max)

    user_details = input("""Would you like image details? (y/n) \n""")
    if str(user_details) == 'y':
        details = True
    else:
        details = False

    filter = filterImages(path, row, clouds, startDate, endDate, details, max_scenes)

    print("\nDownloading images will take some time...")

    for i, row in filter.iterrows():

        row_counter = 1
        print('\n', " Getting images: {}".format(row_counter), row.productId, '\n')
        row_counter += 1

        response = requests.get(row.download_url)

        if response.status_code == 200:

            # Add the html to the soup
            html = BeautifulSoup(response.content, 'html.parser')

            # Create directory to store images
            save_path = os.path.join(pathname, row.productId) # maybe add nickname
            # Overwrite old directory
            os.makedirs(save_path, exist_ok=True)

            # iterate over <li> tags in the html that represent separate bands
            for li in html.find_all('li'):

                file = li.find_next('a').get('href')

                print(' Downloading: {}'.format(file))

                response = requests.get(row.download_url.replace('index.html', file), stream=True)

                with open(os.path.join(save_path, file), 'wb') as output:
                    shutil.copyfileobj(response.raw, output)
                del response


def compositeImages(root_dir):
    # walk over directory to find scene packages
    regex = re.compile("^L.08")

    out_path_list = []

    directory_list = list()
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in dirs:
            if regex.match(name):
                directory_list.append(os.path.join(root, name))

    image_path_list = []

    for folder in directory_list:
        image_path_list.append(natsort.natsorted(glob(
            os.path.join(folder, "*B*.TIF")
        )))

        composite_dir = folder + "/composite_folder"

        os.makedirs(composite_dir, exist_ok=True)

        out_raster_name = os.path.join(composite_dir, "composite_bands.tif")

        out_path_list.append(out_raster_name)

        for image in image_path_list:
            if str(image) not in out_raster_name:
                es.stack(image, out_raster_name)

    return out_path_list


def classify(path, clusters, band1, band2, band3):

    composite_data = gdal.Open(path, gdal.GA_ReadOnly)

    image = numpy.zeros((composite_data.RasterYSize,
                         composite_data.RasterXSize,
                         composite_data.RasterCount),
                     gdal_array.GDALTypeCodeToNumericTypeCode(
                         composite_data.GetRasterBand(1).DataType
                     ))

    band_list = []

    band_list.append(int(band1))
    band_list.append(int(band2))
    band_list.append(int(band3))

    for b in band_list:
        image[:, :, b] = composite_data.GetRasterBand(int(b)).ReadAsArray()

    new_shape = (image.shape[0] * image.shape[1], image.shape[2])

    x = image[:, :, :13].reshape(new_shape)

    k_means = cluster.MiniBatchKMeans(n_clusters=clusters)
    k_means.fit(x)

    x_cluster = k_means.labels_
    x_cluster = x_cluster.reshape(image[:, :, 0].shape)

    plot_bool = input("Would you like to plot the classified images now? [y]/n \n")
    if plot_bool == "y":
        plt.figure(figsize=(20, 20))
        plt.imshow(x_cluster, cmap="hsv")

        plt.show()

    out_raster_path = input("Provide a path to store the output data: \n"
                            "This will take some time... \n")

    band = composite_data.GetRasterBand(2)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    driver = gdal.GetDriverByName("GTiff")

    new_file_count = 0

    while os.path.exists(os.path.join(out_raster_path, "/k_means%s.tif" % new_file_count)):
        new_file_count += 1

    outDataRaster = driver.Create(os.path.join(out_raster_path, "/k_means{}.tif".format(new_file_count)),
                                  rows,
                                  cols, 1, gdal.GDT_Byte)

    outDataRaster.SetGeoTransform(composite_data.GetGeoTransform())
    outDataRaster.SetProjection(composite_data.GetProjection())

    outDataRaster.GetRasterBand(1).WriteArray(x_cluster)

    outDataRaster.FlushCache()
    del outDataRaster



