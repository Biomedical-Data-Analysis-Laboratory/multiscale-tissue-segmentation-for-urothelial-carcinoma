# region IMPORTS
# -------------------  My files  ---------------------
import my_constants
# -------------------  GPU Stuff  ---------------------
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.constraints import max_norm
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import InputSpec
from keras.engine import Layer
from keras.models import *
from keras.layers import *
from keras.callbacks import CSVLogger
from keras import backend as K
import tensorflow as tf
# -------------------  Other  ---------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import skimage
from PIL import ImageFilter
import matplotlib.patches as mpatches
import random
import operator
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from skimage.color import rgb2gray
from skimage import img_as_bool
import inspect
import platform
from shutil import copyfile
import re
import csv
import scipy
import scipy.misc
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import itertools
import datetime
import logging
import pandas as pd

my_logger = logging.getLogger('rune_logger')
import time
import math
import pickle
import glob
import os
from xml.etree import ElementTree

# -------------------  PyVips  ---------------------
vipshome = 'C:/Users/2918257/Downloads/Vips/vips-dev-w64-all-8.7.3/vips-dev-8.7/bin'
os.environ['PATH'] = vipshome + ':' + os.environ['PATH']
# os.environ['VIPS_WARNING'] = ''
import pyvips


# endregion


# region INIT / HELP FUNCTIONS
def init_file(SAVED_DATA_FOLDER, LOG_FOLDER, FILE_NAME, START_NEW_MODEL, CONTINUE_FROM_MODEL,
              METADATA_FOLDER, MODELS_CORRELATION_FOLDER,
              TISSUE_PREDICTED_CLASSES_DICTS_PATH, USE_MULTIPROCESSING):
    # Function that runs in the beginning of the program
    if START_NEW_MODEL in [True, 'True', 'true']:
        # Start a new model
        my_print('Starting new project')

        # Make a new folder to save current run inside
        current_run_path = '{}{}'.format(SAVED_DATA_FOLDER, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S/'))
        os.makedirs(current_run_path, exist_ok=True)

    elif START_NEW_MODEL in [False, 'False', 'false']:
        # Check to see that previous projects exists
        if os.listdir(SAVED_DATA_FOLDER).__len__() == 0:
            my_print('No previous projects found. Please start a new project by setting START_NEW_MODEL to True. Stopping program.')
            exit()
        else:
            my_print('Continue previous project')

        if CONTINUE_FROM_MODEL == 'last':
            project_list = os.listdir(SAVED_DATA_FOLDER)
            project_list.sort()
            current_run_path = '{}{}'.format(SAVED_DATA_FOLDER, project_list[-1] + '/')
        elif CONTINUE_FROM_MODEL in os.listdir(SAVED_DATA_FOLDER):
            current_run_path = '{}{}/'.format(SAVED_DATA_FOLDER, CONTINUE_FROM_MODEL)
        else:
            my_print('Project specified in CONTINUE_FROM_MODEL not found. Stopping program.')
            exit()
    else:
        my_print('Wrong format of START_NEW_MODEL. Please choose True or False. Stopping program.')
        exit()

    # Check if SAVED_DATA_FOLDER exist. If not, create one
    os.makedirs(SAVED_DATA_FOLDER, exist_ok=True)

    # Create metadata folder
    os.makedirs(current_run_path + METADATA_FOLDER, exist_ok=True)

    # Check if MODELS_CORRELATION_FOLDER exist. If not, create one
    os.makedirs(MODELS_CORRELATION_FOLDER, exist_ok=True)

    # Check if TISSUE_PREDICTED_CLASSES_DICTS_PATH exist. If not, create one
    os.makedirs(TISSUE_PREDICTED_CLASSES_DICTS_PATH, exist_ok=True)

    # Make a new folder for logs
    current_log_path = '{0}{1}'.format(current_run_path, LOG_FOLDER)
    os.makedirs(current_log_path, exist_ok=True)

    # Test start time
    global start_time
    start_time = time.time()
    start_time_formatted = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    start_time_logger = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H-%M-%S')

    # Create a logger (logger is created in top of file (import section))
    my_logger_path = '{0}{1}-{2}.log'.format(current_log_path, start_time_logger, FILE_NAME)
    my_logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(my_logger_path)

    # create formatter
    # fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')

    # add formatter to fh
    # fh.setFormatter(fh_formatter)

    # add FileHandler to logger
    my_logger.addHandler(fh)

    if USE_MULTIPROCESSING is False:
        my_print('Warning: MULTIPROCESSING IS DISABLED!')

    # Print test start
    my_print('Program started at {}'.format(start_time_formatted))

    return current_run_path


def my_print(msg, visible=True, error=False):
    # Function that both prints a message to console and to a log file
    if not error:
        my_logger.info(msg)
        if visible:
            print(msg)
    else:
        msg = 'ERROR: {}'.format(msg)
        my_logger.error(msg)
        print(msg)


def create_keras_logger(summary_path, model_name):
    # Create a log to save epoches/accuracy/loss
    log_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    log_name = '{}{}keras_log_{}.csv'.format(summary_path, model_name, log_timestamp)
    csv_logger = CSVLogger(log_name, append=True, separator=';')

    return csv_logger


def csv_2_dict_label_function(DATASET_ROOT, DATASET_MAIN_FOLDER_NAME, current_run_path, METADATA_PATH,
                              HNUMBER_WITH_CUSTOM_LABELS_DICT_PICKLE_FILE, label_name_to_index_dict_PICKLE_FILE):
    # Search for all CSV files in dataset folder (should only be one file, containing list of labels)
    csv_label_file = [i for i in glob.glob(DATASET_ROOT + DATASET_MAIN_FOLDER_NAME + '*.csv')]

    # Check that one file was found
    if len(csv_label_file) > 1:
        my_print('Found more than one CSV Label file in dataset folder. Stopping program.')
        exit()

    # Declare variables
    FILENAME_WITH_CUSTOM_LABELS_DICT = dict()
    label_name_to_index_dict = dict()
    label_name_to_index_dict['WHO73'] = dict()
    label_name_to_index_dict['WHO04'] = dict()
    label_name_to_index_dict['Stage'] = dict()
    label_name_to_index_dict['Recurrence'] = dict()
    label_name_to_index_dict['Progression'] = dict()
    label_name_to_index_dict['WHOcombined'] = dict()

    errors = 0
    n_rows = 0
    who73 = 0
    who04 = 0
    stage = 0
    recurrence = 0
    progression = 0
    WHOcombined = 0

    # Read from CSV file
    with open(csv_label_file[0]) as csvfile:

        # Jump over first line (header info)
        next(csvfile, None)

        # Create a reader
        readCSV = csv.reader(csvfile, delimiter=';')

        # Go through each row of the file
        for row in readCSV:

            n_rows += 1
            H_number = 'H' + row[0]

            # WHO73
            if row[1] == 'Grade 1':
                who73 = 0
                label_name_to_index_dict['WHO73']['Grade_1'] = who73
            elif row[1] == 'Grade 2':
                who73 = 1
                label_name_to_index_dict['WHO73']['Grade_2'] = who73
            elif row[1] == 'Grade 3':
                who73 = 2
                label_name_to_index_dict['WHO73']['Grade_3'] = who73
            else:
                errors += 1

            # WHO04
            if row[2] == 'High grade':
                who04 = 0
                label_name_to_index_dict['WHO04']['High_grade'] = who04
            elif row[2] == 'Low grade':
                who04 = 1
                label_name_to_index_dict['WHO04']['Low_grade'] = who04
            else:
                errors += 1

            # Stage
            if row[3] == 'TA':
                stage = 0
                label_name_to_index_dict['Stage']['TA'] = stage
            elif row[3] == 'T1':
                stage = 1
                label_name_to_index_dict['Stage']['T1'] = stage
            else:
                errors += 1

            # Recurrence
            if row[4] in ['NO', 'No', 'no']:
                recurrence = 0
                label_name_to_index_dict['Recurrence']['No'] = recurrence
            elif row[4] in ['YES', 'Yes', 'yes']:
                recurrence = 1
                label_name_to_index_dict['Recurrence']['Yes'] = recurrence
            else:
                errors += 1

            # Progression
            if row[5] in ['NO', 'No', 'no']:
                progression = 0
                label_name_to_index_dict['Progression']['No'] = progression
            elif row[5] in ['YES', 'Yes', 'yes']:
                progression = 1
                label_name_to_index_dict['Progression']['Yes'] = progression
            else:
                errors += 1

            # WHO-Combined
            if row[6] in [0, '0']:
                WHOcombined = 0
                label_name_to_index_dict['WHOcombined']['No'] = WHOcombined
            elif row[6] in [1, '1']:
                WHOcombined = 1
                label_name_to_index_dict['WHOcombined']['Yes'] = WHOcombined
            elif row[6] in [2, '2']:
                WHOcombined = 2
                label_name_to_index_dict['WHOcombined']['Yes'] = WHOcombined
            elif row[6] in [3, '3']:
                WHOcombined = 3
                label_name_to_index_dict['WHOcombined']['Yes'] = WHOcombined
            else:
                errors += 1

            # Insert values into temporary dict
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number] = {}
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number]['WHO73'] = who73
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number]['WHO04'] = who04
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number]['Recurrence'] = recurrence
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number]['Progression'] = progression
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number]['Stage'] = stage
            FILENAME_WITH_CUSTOM_LABELS_DICT[H_number]['WHOcombined'] = WHOcombined

    if errors > 0:
        my_print('Found {} errors in {} rows'.format(errors, n_rows))
    else:
        my_print('Saved {} rows from csv file'.format(n_rows))

    # Save dict with pickle
    pickle_save(FILENAME_WITH_CUSTOM_LABELS_DICT, current_run_path + METADATA_PATH + HNUMBER_WITH_CUSTOM_LABELS_DICT_PICKLE_FILE)
    pickle_save(label_name_to_index_dict, current_run_path + METADATA_PATH + label_name_to_index_dict_PICKLE_FILE)

    return label_name_to_index_dict


def remove_white_background_v3(input_img, PADDING, folder_path):
    # Reset variables
    remove_rows_top = 0
    remove_rows_bottom = 0
    remove_cols_left = 0
    remove_cols_right = 0
    x_list = []
    y_list = []
    white_background_vector = [250, 251, 252, 253, 254, 255]
    csv_override_filename = folder_path + 'override.csv'

    if os.path.isfile(csv_override_filename):
        # Some images need special care. We can override the values of x_inside and y_inside here using the CSV file in the folder
        # csv file in the folder should have the name 'override.csv', and contain values "x,y", e.g. "1000,2500".
        # Read from CSV file
        with open(csv_override_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for line in reader:
                override_xy = line

        if input_img.height < 10000:
            x_inside = int(override_xy[0])
            y_inside = int(override_xy[1])
        elif input_img.height < 50000:
            x_inside = int(override_xy[0]) // my_constants.Scale_between_25x_100x
            y_inside = int(override_xy[1]) // my_constants.Scale_between_25x_100x
        elif input_img.height > 100000:
            x_inside = int(override_xy[0]) // my_constants.Scale_between_25x_400x
            y_inside = int(override_xy[1]) // my_constants.Scale_between_25x_400x
    else:
        # If there is not a CSV file with coordinates of x_inside and y_inside, this section will find them automatically.
        # Make a grid of all X lines and find minimum values (which indicate a WSI in the large white space) (there may be more than one WSI)
        step_x = int(input_img.width // 100)
        step_y = int(input_img.height // 100)

        for x_pos in range(step_x, input_img.width, step_x):
            tmp = input_img.extract_area(x_pos, 0, 1, input_img.height)
            x_list.append((x_pos, tmp.min()))

        # Go through x_list and find all transitions between "white background" and "WSI image".
        threshold = 250
        dict_of_transitions_x = dict()
        over_under_threshold = 'under'

        for index, value in enumerate(x_list):
            if over_under_threshold == 'under':
                if value[1] < threshold:
                    dict_of_transitions_x[len(dict_of_transitions_x)] = index
                    over_under_threshold = 'over'
            elif over_under_threshold == 'over':
                if value[1] > threshold:
                    dict_of_transitions_x[len(dict_of_transitions_x)] = index
                    over_under_threshold = 'under'

        x_inside = x_list[dict_of_transitions_x[0]][0] + ((x_list[dict_of_transitions_x[1]][0] - x_list[dict_of_transitions_x[0]][0]) // 2)

        # Extract position of firste WSI in the image
        # for index, item in enumerate(x_list):
        #     if item[1] < 250:
        #         x_first = item[0]
        #         for index2 in range(index, len(x_list)):
        #             if x_list[index2][1] > 250:
        #                 x_last = x_list[index2][0]
        #                 break
        #         break
        # x_inside = x_first + ((x_last - x_first) // 2)

        # Initial crop (if there are more than one WSI in the image, this crops out the first one)
        if len(dict_of_transitions_x) > 2:
            init_crop_x = x_list[dict_of_transitions_x[1]][0] + ((x_list[dict_of_transitions_x[2]][0] - x_list[dict_of_transitions_x[1]][0]) // 2)
            input_img = input_img.extract_area(0, 0, init_crop_x, input_img.height)

        # Make a grid of all Y lines and find minimum values (which indicate a WSI in the large white space)
        for y_pos in range(step_y, input_img.height, step_y):
            tmp = input_img.extract_area(0, y_pos, input_img.width, 1)
            y_list.append((y_pos, tmp.min()))

        # for index, item in enumerate(y_list):
        #     if item[1] < 250:
        #         y_first = item[0]
        #         for index2 in range(index, len(y_list)):
        #             if y_list[index2][1] > 250:
        #                 y_last = y_list[index2][0]
        #                 break
        #         break
        # y_inside = y_first + ((y_last - y_first) // 2)

        dict_of_transitions_y = dict()
        over_under_threshold = 'under'

        for index, value in enumerate(y_list):
            if over_under_threshold == 'under':
                if value[1] < threshold:
                    dict_of_transitions_y[len(dict_of_transitions_y)] = index
                    over_under_threshold = 'over'
            elif over_under_threshold == 'over':
                if value[1] > threshold:
                    dict_of_transitions_y[len(dict_of_transitions_y)] = index
                    over_under_threshold = 'under'

        y_inside = y_list[dict_of_transitions_y[0]][0] + ((y_list[dict_of_transitions_y[1]][0] - y_list[dict_of_transitions_y[0]][0]) // 2)

    ##### REMOVE HORIZONTAL WHITE LINES (TOP AND DOWN)
    if input_img(x_inside, 0)[1] in white_background_vector:
        first = 0
        last = y_inside
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_rows_top = midpoint - 1
    ##### REMOVE HORIZONTAL WHITE LINES (BOTTOM AND UP)
    if input_img(x_inside, (input_img.height - 1))[1] in white_background_vector:
        # first = (current_image.height // 2) - 5000
        first = y_inside
        last = input_img.height

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # if current_image(((current_image.width // current_divide_constant)-(current_image.width//4)), midpoint)[1] == 255:
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                last = midpoint - 1
            else:
                first = midpoint + 1

        remove_rows_bottom = midpoint
    ##### REMOVE VERTICAL WHITE LINES (VENSTRE MOT HoYRE)
    if input_img(0, y_inside)[1] == 255:
        first = 0
        last = x_inside

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(midpoint, y_inside)[1] == 255:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_cols_left = midpoint - 1
    ##### REMOVE VERTICAL WHITE LINES (HOYRE MOT VENSTRE)
    if input_img(input_img.width - 1, y_inside)[1] == 255:
        first = x_inside
        last = input_img.width
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(midpoint, y_inside)[1] == 255:
                last = midpoint - 1
            else:
                first = midpoint + 1
        remove_cols_right = midpoint + 1

    # print('remove_cols_left ', remove_cols_left)
    # print('remove_cols_right ', remove_cols_right)

    # Calculate new width/height of image and crop.
    if remove_rows_bottom != 0:
        # Calculate new width/height
        new_width = (input_img.width - remove_cols_left - (input_img.width - remove_cols_right))
        new_height = (input_img.height - remove_rows_top - (input_img.height - remove_rows_bottom))

        # Include a border around image (to extract 25x tiles later)
        remove_cols_left = remove_cols_left - PADDING
        remove_rows_top = remove_rows_top - PADDING
        new_width = new_width + 2 * PADDING
        new_height = new_height + 2 * PADDING

        return remove_cols_left, remove_rows_top, new_width, new_height


def create_wsi_binary_mask(wsi_path, TILE_SIZE, override_predict_region, x_pos_min, y_pos_min, x_pos_max, y_pos_max,
                           current_scale_to_use, region_width, region_height, current_wsi_dataset_folder):
    my_print('Creating binary background mask.')

    # Load image
    full_image_25x = pyvips.Image.new_from_file(wsi_path, level=2).flatten().rot(1)

    # Remove white background
    scn_offset_x, scn_offset_y, scn_width, scn_height = remove_white_background_v3(input_img=full_image_25x, PADDING=0, folder_path=current_wsi_dataset_folder)

    if override_predict_region:

        # 400x region config
        if (current_scale_to_use == '400x') or (current_scale_to_use[1] == '400x') or (current_scale_to_use == ['25x', '100x', '400x']):
            region_width_400x = region_width
            region_height_400x = region_height

            region_400x_centre_x = (x_pos_min + region_width_400x / 2)
            region_400x_centre_y = (y_pos_min + region_height_400x / 2)

            region_25x_centre_x = region_400x_centre_x * my_constants.Scale_between_25x_400x
            region_25x_centre_y = region_400x_centre_y * my_constants.Scale_between_25x_400x

            region_width_25x = region_width_400x * my_constants.Scale_between_25x_400x
            region_height_25x = region_height_400x * my_constants.Scale_between_25x_400x

            region_25x_start_x = region_25x_centre_x - region_width_25x / 2
            region_25x_start_y = region_25x_centre_y - region_height_25x / 2

            current_image = full_image_25x.extract_area(scn_offset_x + region_25x_start_x,
                                                        scn_offset_y + region_25x_start_y,
                                                        region_width_25x,
                                                        region_height_25x)

            my_print('Background mask coordinates: x: {}. y: {}'.format(x_pos_min * my_constants.Scale_between_25x_400x, y_pos_min * my_constants.Scale_between_25x_400x))
        # 100x region config
        elif (current_scale_to_use == '100x') or (current_scale_to_use[1] == '100x'):
            region_width_100x = region_width
            region_height_100x = region_height

            region_100x_centre_x = (x_pos_min + region_width_100x / 2)
            region_100x_centre_y = (y_pos_min + region_height_100x / 2)

            region_25x_centre_x = region_100x_centre_x * my_constants.Scale_between_25x_100x
            region_25x_centre_y = region_100x_centre_y * my_constants.Scale_between_25x_100x

            region_width_25x = region_width_100x * my_constants.Scale_between_25x_100x
            region_height_25x = region_height_100x * my_constants.Scale_between_25x_100x

            region_25x_start_x = region_25x_centre_x - region_width_25x / 2
            region_25x_start_y = region_25x_centre_y - region_height_25x / 2

            current_image = full_image_25x.extract_area(scn_offset_x + region_25x_start_x,
                                                        scn_offset_y + region_25x_start_y,
                                                        region_width_25x,
                                                        region_height_25x)
            my_print('Background mask coordinates: x: {}. y: {}'.format(x_pos_min * my_constants.Scale_between_100x_400x, y_pos_min * my_constants.Scale_between_100x_400x))
        # 25x region config
        elif current_scale_to_use == '25x':
            region_width_25x = region_width
            region_height_25x = region_height

            # region_100x_centre_x = (x_pos_min + region_width_25x / 2)
            # region_100x_centre_y = (y_pos_min + region_height_25x / 2)
            #
            # region_25x_centre_x = region_100x_centre_x * my_constants.Scale_between_25x_100x
            # region_25x_centre_y = region_100x_centre_y * my_constants.Scale_between_25x_100x
            #
            # region_width_25x = region_width_100x * my_constants.Scale_between_25x_100x
            # region_height_25x = region_height_100x * my_constants.Scale_between_25x_100x
            #
            # region_25x_start_x = region_25x_centre_x - region_width_25x / 2
            # region_25x_start_y = region_25x_centre_y - region_height_25x / 2

            current_image = full_image_25x.extract_area(scn_offset_x + x_pos_min,
                                                        scn_offset_y + y_pos_min,
                                                        region_width_25x,
                                                        region_height_25x)
            my_print('Background mask coordinates: x:{}, y:{}'.format(x_pos_min, y_pos_min))

        # if current_scale_to_use == '25x':
        #     current_image = current_image.extract_area(x_pos_min,
        #                                                y_pos_min,
        #                                                ((x_pos_max + TILE_SIZE) - (x_pos_min)),
        #                                                ((y_pos_max + TILE_SIZE) - (y_pos_min)))
        # elif (current_scale_to_use[0] == '25x' and current_scale_to_use[1] == '100x' and current_scale_to_use[2] == None) or current_scale_to_use == '100x':
        #     current_image = current_image.extract_area(x_pos_min * my_constants.Scale_between_25x_100x,
        #                                                y_pos_min * my_constants.Scale_between_25x_100x,
        #                                                ((x_pos_max+TILE_SIZE) * my_constants.Scale_between_25x_100x - x_pos_min * my_constants.Scale_between_25x_100x),
        #                                                ((y_pos_max+TILE_SIZE) * my_constants.Scale_between_25x_100x - y_pos_min * my_constants.Scale_between_25x_100x))
        # else:
        #     current_image = current_image.extract_area(x_pos_min * my_constants.Scale_between_25x_400x,
        #                                                y_pos_min * my_constants.Scale_between_25x_400x,
        #                                                ((x_pos_max+TILE_SIZE) * my_constants.Scale_between_25x_400x - (x_pos_min * my_constants.Scale_between_25x_400x)),
        #                                                ((y_pos_max+TILE_SIZE) * my_constants.Scale_between_25x_400x - (y_pos_min * my_constants.Scale_between_25x_400x)))
    else:
        # Remove white background around the image
        current_image = full_image_25x.extract_area(scn_offset_x, scn_offset_y, scn_width, scn_height)

    # TESTING
    # current_image_25x = current_image_25x.extract_area(1880, 540, 32, 32) # tissue
    # current_image_25x = current_image_25x.extract_area(1707, 577, 25, 25) # vanskelig bakgrunn
    # current_image_25x = current_image_25x.extract_area(1660, 680, 64, 64)  # bakgrunn
    # overview_filename = '{}/{}#overview.jpeg'.format(wsi_folder, wsi_name)
    # current_image_25x.jpegsave(overview_filename, Q=100)

    # Write tile to memory and convert to numpy array
    tile_numpy = np.ndarray(buffer=current_image.write_to_memory(),
                            dtype=my_constants.format_to_dtype[current_image.format],
                            shape=[current_image.height, current_image.width, current_image.bands])

    # Create a blank mask
    current_tile_mask = np.zeros(shape=(current_image.height, current_image.width))

    # Go through each pixel of tile, and set class in mask
    for row_pixel in range(0, current_image.height):
        for column_pixel in range(0, current_image.width):
            # Background
            if tile_numpy[row_pixel][column_pixel][1] > 195:
                current_tile_mask[row_pixel, column_pixel] = 0
            # Everything else (current class)
            else:
                current_tile_mask[row_pixel, column_pixel] = 1

    return current_tile_mask


def pickle_load(path):
    with open(path, 'rb') as handle:
        output = pickle.load(handle)
    return output


def pickle_save(variable_to_save, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def summary_csv_file_create_new(SUMMARY_CSV_FILE_PATH):
    # Create a new summary.csv file
    if not os.path.isfile(SUMMARY_CSV_FILE_PATH):
        try:
            with open(SUMMARY_CSV_FILE_PATH, 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(
                    ['Project ID', 'Description', 'Version', 'Mode', 'Model name', 'Label',
                     'Base model', 'Freeze', 'blocks_to_unfreeze_vgg16_vgg19', 'delayed_unfreeze_start_epoche_vgg16_vgg19', 'Base model pooling',
                     'Training samples', 'Validation samples', 'Test samples', 'Augment classes', 'Augment multiplier', 'Tile size',
                     'Layer config', 'Learning rate', 'Batch size', 'Dropout',
                     'N_neurons1', 'N_neurons2', 'N_neurons3',
                     'F1-Score', 'Std Dev',
                     'Best train loss', 'Best train acc', 'Best val loss', 'Best val acc',
                     'Best val loss epoch', 'Best val acc epoch', 'Trained epoches', 'Total epochs',
                     'Latent size', 'Compression', 'Time(H:M:S)',
                     'Optimizer', 'ReduceLROnPlateau',
                     'Trainable params(Start)', 'Non-trainable params(Start)', 'Trainable params(End)', 'Non-trainable params(End)',
                     'Python', 'Keras', 'TensorFlow', 'Date'])
        except Exception as e:
            my_print('Error writing to file', error=True)
            my_print(e, error=True)


def summary_csv_file_update(SUMMARY_CSV_FILE_PATH, PROJECT_ID, DESCRIPTION, FILE_NAME, MODE, model_name, label,
                            base_model, freeze_base_model, blocks_to_unfreeze_vgg16_vgg19, delayed_unfreeze_start_epoche_vgg16_vgg19, base_model_pooling,
                            training_samples, validation_samples, test_samples, augment_classes, augment_multiplier, tile_size,
                            layer_config, learning_rate, batch_size, dropout,
                            n_neurons1, n_neurons2, n_neurons3,
                            F1_score, F1_std,
                            best_train_loss, best_train_acc, best_val_loss, best_val_acc,
                            best_val_loss_epoch, best_val_acc_epoch, trained_epoches, total_epochs,
                            latent_size, compression, model_time,
                            optimizer, ReduceLRstatus,
                            n_trainable_parameters_start, n_non_trainable_parameters_start, n_trainable_parameters_end, n_non_trainable_parameters_end,
                            python_version, keras_version, tf_version):
    # Update existing summary.csv file
    try:
        with open(SUMMARY_CSV_FILE_PATH, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [PROJECT_ID, DESCRIPTION, FILE_NAME, MODE, model_name, label,
                 base_model, freeze_base_model, blocks_to_unfreeze_vgg16_vgg19, delayed_unfreeze_start_epoche_vgg16_vgg19, base_model_pooling,
                 training_samples, validation_samples, test_samples, augment_classes, augment_multiplier, tile_size,
                 layer_config, str(learning_rate).replace('.', ','), batch_size, str(dropout).replace('.', ','),
                 n_neurons1, n_neurons2, n_neurons3,
                 F1_score, F1_std,
                 str(best_train_loss).replace('.', ','), str(best_train_acc).replace('.', ','),
                 str(best_val_loss).replace('.', ','), str(best_val_acc).replace('.', ','),
                 best_val_loss_epoch, best_val_acc_epoch, trained_epoches, total_epochs,
                 latent_size, str(compression).replace('.', ','), model_time,
                 optimizer, ReduceLRstatus,
                 n_trainable_parameters_start, n_non_trainable_parameters_start, n_trainable_parameters_end, n_non_trainable_parameters_end,
                 '\'' + python_version, '\'' + keras_version, tf_version,
                 datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')])
    except Exception as e:
        my_print('Error writing to file', error=True)
        my_print(e, error=True)


def augment_tiles(aug_argument, tile):
    # Augment tiles by rotation and flipping
    if aug_argument == '1':
        return tile
    elif aug_argument == '2':
        # rot90
        tile = tile.rot(1)
    elif aug_argument == '3':
        # rot180
        tile = tile.rot(2)
    elif aug_argument == '4':
        # rot270
        tile = tile.rot(3)
    elif aug_argument == '5':
        # rot90_flipHoriz
        tile = tile.rot(1)
        tile = tile.flip(0)
    elif aug_argument == '6':
        # rot270_flipHoriz
        tile = tile.rot(3)
        tile = tile.flip(0)
    elif aug_argument == '7':
        # flipVert
        tile = tile.flip(1)
    elif aug_argument == '8':
        # rot180_flipVert
        tile = tile.rot(2)
        tile = tile.flip(1)

    return tile


# endregion


# region DATASET
class mode_2b_mono_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, tile_dicts, batch_size, n_classes, shuffle, TILE_SIZE, which_scale_to_use, name_of_classes):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.tile_dicts = tile_dicts
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.TILE_SIZE = TILE_SIZE
        # self.which_scale_to_use = which_scale_to_use
        self.name_of_classes = name_of_classes
        self.on_epoch_end()

        if which_scale_to_use == '400x':
            self.img_one_level = 0
            self.coordinate_scale = 'coordinates_400x'
        elif which_scale_to_use == '100x':
            self.img_one_level = 1
            self.coordinate_scale = 'coordinates_100x'
        elif which_scale_to_use == '25x':
            self.img_one_level = 2
            self.coordinate_scale = 'coordinates_25x'

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.tile_dicts) / self.batch_size)) + 1
        return int(np.ceil(len(self.tile_dicts) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        tile_dicts_temp = [self.tile_dicts[k] for k in indexes]

        # Generate data
        X_400x, y = self.__data_generation(tile_dicts_temp)

        return X_400x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.tile_dicts))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tile_dicts_temp):
        """Generates data containing batch_size samples."""
        X_img = []
        y = np.empty((len(tile_dicts_temp)), dtype=int)

        # Generate data
        for i, tile_dict in enumerate(tile_dicts_temp):
            # Read image, flatten and rotate 90 degree so that the coordinated line up correctly
            full_image = pyvips.Image.new_from_file(tile_dict['path'], level=self.img_one_level).flatten().rot(1)

            # Extract tile
            tile = full_image.extract_area(tile_dict[self.coordinate_scale][0], tile_dict[self.coordinate_scale][1], self.TILE_SIZE, self.TILE_SIZE)

            # Augmentation
            tile = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile)

            # Preprocess. Normalize from 0-255 to 0-1.
            # tile = dataset_preprocessing(preprocess_mode=1, input_tile=tile)

            # Write tile to memory and convert to numpy array
            tile_numpy = np.ndarray(buffer=tile.write_to_memory(),
                                    dtype=my_constants.format_to_dtype[tile.format],
                                    shape=[tile.height, tile.width, tile.bands])

            # Preprocess. Subtract mean
            # tile_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_numpy)

            # Preprocess. 'RGB'->'BGR'.
            # tile_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_numpy)

            X_img.append(tile_numpy)
            y[i] = self.name_of_classes.index(tile_dict['label'])
        X_tile_array = np.array(X_img)
        return X_tile_array, keras.utils.to_categorical(y, num_classes=self.n_classes)


class mode_2b_di_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, tile_dicts, batch_size, n_classes, shuffle, TILE_SIZE, which_scale_to_use, name_of_classes):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.tile_dicts = tile_dicts
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.which_scale_to_use = which_scale_to_use
        self.name_of_classes = name_of_classes
        self.on_epoch_end()

        if which_scale_to_use[0] == '400x':
            self.img_one_level = 0
            self.coordinate_one_scale = 'coordinates_400x'
        elif which_scale_to_use[0] == '100x':
            self.img_one_level = 1
            self.coordinate_one_scale = 'coordinates_100x'
        elif which_scale_to_use[0] == '25x':
            self.img_one_level = 2
            self.coordinate_one_scale = 'coordinates_25x'

        if which_scale_to_use[1] == '400x':
            self.img_two_level = 0
            self.coordinate_two_scale = 'coordinates_400x'
        elif which_scale_to_use[1] == '100x':
            self.img_two_level = 1
            self.coordinate_two_scale = 'coordinates_100x'
        elif which_scale_to_use[1] == '25x':
            self.img_two_level = 2
            self.coordinate_two_scale = 'coordinates_25x'

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.tile_dicts) / self.batch_size)) + 1
        return int(np.ceil(len(self.tile_dicts) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        tile_dicts_temp = [self.tile_dicts[k] for k in indexes]

        # Generate data
        [X_one, X_two], y = self.__data_generation(tile_dicts_temp)

        return [X_one, X_two], y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.tile_dicts))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tile_dicts_temp):
        """Generates data containing batch_size samples."""
        X_img_one = []
        X_img_two = []
        y = np.empty((len(tile_dicts_temp)), dtype=int)

        # Generate data
        for i, tile_dict in enumerate(tile_dicts_temp):
            # Read image, flatten and rotate image 90 degree so that the coordinated line up correctly
            full_image_one = pyvips.Image.new_from_file(tile_dict['path'], level=self.img_one_level).flatten().rot(1)
            full_image_two = pyvips.Image.new_from_file(tile_dict['path'], level=self.img_two_level).flatten().rot(1)

            # Extract tile
            tile_one = full_image_one.extract_area(tile_dict[self.coordinate_one_scale][0], tile_dict[self.coordinate_one_scale][1], self.tile_size, self.tile_size)
            tile_two = full_image_two.extract_area(tile_dict[self.coordinate_two_scale][0], tile_dict[self.coordinate_two_scale][1], self.tile_size, self.tile_size)

            # Augmentation
            tile_one = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_one)
            tile_two = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_two)

            # Preprocess. Normalize from 0-255 to 0-1.
            # tile_one = dataset_preprocessing(preprocess_mode=1, input_tile=tile_one)
            # tile_two = dataset_preprocessing(preprocess_mode=1, input_tile=tile_two)

            # Write tile to memory and convert to numpy array
            tile_one_numpy = np.ndarray(buffer=tile_one.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_one.format],
                                        shape=[tile_one.height, tile_one.width, tile_one.bands])
            tile_two_numpy = np.ndarray(buffer=tile_two.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_two.format],
                                        shape=[tile_two.height, tile_two.width, tile_two.bands])

            # Preprocess. Subtract mean
            # tile_one_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_one_numpy)
            # tile_two_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_two_numpy)

            # Preprocess. 'RGB'->'BGR'.
            # tile_one_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_one_numpy)
            # tile_two_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_two_numpy)

            X_img_one.append(tile_one_numpy)
            X_img_two.append(tile_two_numpy)
            y[i] = self.name_of_classes.index(tile_dict['label'])

        X_tile_one_array = np.array(X_img_one)
        X_tile_two_array = np.array(X_img_two)
        return [X_tile_one_array, X_tile_two_array], keras.utils.to_categorical(y, num_classes=self.n_classes)


class mode_2b_tri_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, tile_dicts, batch_size, n_classes, shuffle, TILE_SIZE, which_scale_to_use, name_of_classes):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.tile_dicts = tile_dicts
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.which_scale_to_use = which_scale_to_use
        self.name_of_classes = name_of_classes
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.tile_dicts) / self.batch_size)) + 1
        return int(np.ceil(len(self.tile_dicts) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        tile_dicts_temp = [self.tile_dicts[k] for k in indexes]

        # Generate data
        [X_one, X_two, X_three], y = self.__data_generation(tile_dicts_temp)

        return [X_one, X_two, X_three], y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.tile_dicts))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tile_dicts_temp):
        """Generates data containing batch_size samples."""
        X_img_400x = []
        X_img_100x = []
        X_img_25x = []
        y = np.empty((len(tile_dicts_temp)), dtype=int)

        # Generate data
        for i, tile_dict in enumerate(tile_dicts_temp):
            # Read image, flatten and rotate image 90 degree so that the coordinated line up correctly
            full_image_400x = pyvips.Image.new_from_file(tile_dict['path'], level=0).flatten().rot(1)
            full_image_100x = pyvips.Image.new_from_file(tile_dict['path'], level=1).flatten().rot(1)
            full_image_25x = pyvips.Image.new_from_file(tile_dict['path'], level=2).flatten().rot(1)

            # Extract tile
            tile_400x = full_image_400x.extract_area(tile_dict['coordinates_400x'][0], tile_dict['coordinates_400x'][1], self.tile_size, self.tile_size)
            tile_100x = full_image_100x.extract_area(tile_dict['coordinates_100x'][0], tile_dict['coordinates_100x'][1], self.tile_size, self.tile_size)
            tile_25x = full_image_25x.extract_area(tile_dict['coordinates_25x'][0], tile_dict['coordinates_25x'][1], self.tile_size, self.tile_size)

            # Augmentation
            tile_400x = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_400x)
            tile_100x = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_100x)
            tile_25x = augment_tiles(aug_argument=tile_dict['augmentarg'], tile=tile_25x)

            # Preprocess. Normalize from 0-255 to 0-1.
            # tile_400x = dataset_preprocessing(preprocess_mode=1, input_tile=tile_400x)
            # tile_100x = dataset_preprocessing(preprocess_mode=1, input_tile=tile_100x)
            # tile_25x = dataset_preprocessing(preprocess_mode=1, input_tile=tile_25x)

            # Write tile to memory and convert to numpy array
            tile_400x_numpy = np.ndarray(buffer=tile_400x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_400x.format],
                                         shape=[tile_400x.height, tile_400x.width, tile_400x.bands])

            tile_100x_numpy = np.ndarray(buffer=tile_100x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_100x.format],
                                         shape=[tile_100x.height, tile_100x.width, tile_100x.bands])

            tile_25x_numpy = np.ndarray(buffer=tile_25x.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_25x.format],
                                        shape=[tile_25x.height, tile_25x.width, tile_25x.bands])

            # Preprocess. Subtract mean
            # tile_400x_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_400x_numpy)
            # tile_100x_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_100x_numpy)
            # tile_25x_numpy = dataset_preprocessing(preprocess_mode=3, input_tile=tile_25x_numpy)

            # Preprocess. 'RGB'->'BGR'.
            # tile_400x_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_400x_numpy)
            # tile_100x_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_100x_numpy)
            # tile_25x_numpy = dataset_preprocessing(preprocess_mode=2, input_tile=tile_25x_numpy)

            X_img_400x.append(tile_400x_numpy)
            X_img_100x.append(tile_100x_numpy)
            X_img_25x.append(tile_25x_numpy)
            y[i] = self.name_of_classes.index(tile_dict['label'])

        X_tile_400x_array = np.array(X_img_400x)
        X_tile_100x_array = np.array(X_img_100x)
        X_tile_25x_array = np.array(X_img_25x)
        return [X_tile_400x_array, X_tile_100x_array, X_tile_25x_array], keras.utils.to_categorical(y, num_classes=self.n_classes)


class mode_7c_mono_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, all_xy_pos_in_scn_img, batch_size, n_classes, shuffle, TILE_SIZE, scn_path, scn_offset_x, scn_offset_y, which_scale_to_use):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.all_xy_pos_in_scn_img = all_xy_pos_in_scn_img
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.scn_path = scn_path
        self.scn_offset_x = scn_offset_x
        self.scn_offset_y = scn_offset_y
        self.which_scale_to_use = which_scale_to_use
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.all_xy_pos_in_scn_img) / self.batch_size)) + 1
        return int(np.ceil(len(self.all_xy_pos_in_scn_img) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        all_xy_pos_in_scn_img_temp = [self.all_xy_pos_in_scn_img[k] for k in indexes]

        # Generate data
        X_400x = self.__data_generation(all_xy_pos_in_scn_img_temp)

        return X_400x

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.all_xy_pos_in_scn_img))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_xy_pos_in_scn_img_temp):
        """Generates data containing batch_size samples."""
        X_img_400x = []

        # Read image
        if self.which_scale_to_use == '400x':
            full_image = pyvips.Image.new_from_file(self.scn_path, level=0).flatten().rot(1)
        elif self.which_scale_to_use == '100x':
            full_image = pyvips.Image.new_from_file(self.scn_path, level=1).flatten().rot(1)
        elif self.which_scale_to_use == '25x':
            full_image = pyvips.Image.new_from_file(self.scn_path, level=2).flatten().rot(1)

        # Generate data
        for i, current_xy_pos in enumerate(all_xy_pos_in_scn_img_temp):
            # Extract tile
            tile_400x = full_image.extract_area(self.scn_offset_x + current_xy_pos[0], self.scn_offset_y + current_xy_pos[1], self.tile_size, self.tile_size)

            # Write tile to memory and convert to numpy array
            X_img_400x.append(np.ndarray(buffer=tile_400x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_400x.format],
                                         shape=[tile_400x.height, tile_400x.width, tile_400x.bands]))

        X_400x = np.array(X_img_400x)
        return X_400x


class mode_7c_di_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, all_xy_pos_in_scn_img, batch_size, n_classes, shuffle, TILE_SIZE, scn_path, scn_offset_x, scn_offset_y, which_scale_to_use):
        # Initialization
        self.all_xy_pos_in_scn_img = all_xy_pos_in_scn_img
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.scn_path = scn_path
        self.scn_offset_x = scn_offset_x
        self.scn_offset_y = scn_offset_y
        self.which_scale_to_use = which_scale_to_use
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.all_xy_pos_in_scn_img) / self.batch_size)) + 1
        return int(np.ceil(len(self.all_xy_pos_in_scn_img) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        all_xy_pos_in_scn_img_temp = [self.all_xy_pos_in_scn_img[k] for k in indexes]

        # Generate data
        [X_one, X_two] = self.__data_generation(all_xy_pos_in_scn_img_temp)

        return [X_one, X_two]

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.all_xy_pos_in_scn_img))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_xy_pos_in_scn_img_temp):
        """Generates data containing batch_size samples."""
        X_img_one = []
        X_img_two = []

        # Read image one
        if self.which_scale_to_use[0] == '400x':
            full_image_400x = pyvips.Image.new_from_file(self.scn_path, level=0).flatten().rot(1)
        elif self.which_scale_to_use[0] == '100x':
            full_image_100x = pyvips.Image.new_from_file(self.scn_path, level=1).flatten().rot(1)
        elif self.which_scale_to_use[0] == '25x':
            full_image_25x = pyvips.Image.new_from_file(self.scn_path, level=2).flatten().rot(1)

        # Read image two
        if self.which_scale_to_use[1] == '400x':
            full_image_400x = pyvips.Image.new_from_file(self.scn_path, level=0).flatten().rot(1)
        elif self.which_scale_to_use[1] == '100x':
            full_image_100x = pyvips.Image.new_from_file(self.scn_path, level=1).flatten().rot(1)
        elif self.which_scale_to_use[1] == '25x':
            full_image_25x = pyvips.Image.new_from_file(self.scn_path, level=2).flatten().rot(1)

        if self.which_scale_to_use[0] == '25x' and self.which_scale_to_use[1] == '100x':
            # Generate data for 25x/100x
            # for i, current_xy_center_pos_400x in enumerate(all_xy_pos_in_scn_img_temp):
            for i, current_xy_center_pos_100x in enumerate(all_xy_pos_in_scn_img_temp):
                # Transform the coordinate from the 400x image to the 100x image
                # image_x100_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_400x[0]) * my_constants.Scale_between_100x_400x) - (self.tile_size / 2))
                # image_x100_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_400x[1]) * my_constants.Scale_between_100x_400x) - (self.tile_size / 2))
                image_x100_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_100x[0])) - (self.tile_size / 2))
                image_x100_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_100x[1])) - (self.tile_size / 2))

                # Transform the coordinate from the 100x image to the 25x image
                # image_x25_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_400x[0]) * my_constants.Scale_between_25x_400x) - (self.tile_size / 2))
                # image_x25_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_400x[1]) * my_constants.Scale_between_25x_400x) - (self.tile_size / 2))
                image_x25_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_100x[0]) * my_constants.Scale_between_25x_100x) - (self.tile_size / 2))
                image_x25_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_100x[1]) * my_constants.Scale_between_25x_100x) - (self.tile_size / 2))

                # Extract first tile
                tile_one = full_image_25x.extract_area(image_x25_tile_x_pos, image_x25_tile_y_pos, self.tile_size, self.tile_size)

                # Extract second tile
                tile_two = full_image_100x.extract_area(image_x100_tile_x_pos, image_x100_tile_y_pos, self.tile_size, self.tile_size)
                # tile_two = full_image_100x.extract_area(int((self.scn_offset_x + current_xy_center_pos_100x[0]) - (self.tile_size / 2)),
                #                                         int((self.scn_offset_y + current_xy_center_pos_100x[1]) - (self.tile_size / 2)),
                #                                         self.tile_size, self.tile_size)

                # Write tile to memory and convert to numpy array
                X_img_one.append(np.ndarray(buffer=tile_one.write_to_memory(),
                                            dtype=my_constants.format_to_dtype[tile_one.format],
                                            shape=[tile_one.height, tile_one.width, tile_one.bands]))

                X_img_two.append(np.ndarray(buffer=tile_two.write_to_memory(),
                                            dtype=my_constants.format_to_dtype[tile_two.format],
                                            shape=[tile_two.height, tile_two.width, tile_two.bands]))
        else:
            # Generate data for 25x/400x and 100x/400x
            for i, current_xy_center_pos_400x in enumerate(all_xy_pos_in_scn_img_temp):

                # Transform the coordinate from the 400x image to the 100x image
                image_x100_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_400x[0]) * my_constants.Scale_between_100x_400x) - (self.tile_size / 2))
                image_x100_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_400x[1]) * my_constants.Scale_between_100x_400x) - (self.tile_size / 2))

                # Transform the coordinate from the 40x image to the 25x image
                image_x25_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_400x[0]) * my_constants.Scale_between_25x_400x) - (self.tile_size / 2))
                image_x25_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_400x[1]) * my_constants.Scale_between_25x_400x) - (self.tile_size / 2))

                # Extract first tile
                if self.which_scale_to_use[0] == '400x':
                    tile_one = full_image_400x.extract_area((current_xy_center_pos_400x[0] - self.tile_size / 2),
                                                            (current_xy_center_pos_400x[1] - self.tile_size / 2),
                                                            self.tile_size, self.tile_size)
                elif self.which_scale_to_use[0] == '100x':
                    tile_one = full_image_100x.extract_area(image_x100_tile_x_pos, image_x100_tile_y_pos, self.tile_size, self.tile_size)
                elif self.which_scale_to_use[0] == '25x':
                    tile_one = full_image_25x.extract_area(image_x25_tile_x_pos, image_x25_tile_y_pos, self.tile_size, self.tile_size)

                # Extract second tile
                if self.which_scale_to_use[1] == '400x':
                    tile_two = full_image_400x.extract_area((current_xy_center_pos_400x[0] - self.tile_size / 2),
                                                            (current_xy_center_pos_400x[1] - self.tile_size / 2),
                                                            self.tile_size, self.tile_size)
                elif self.which_scale_to_use[1] == '100x':
                    tile_two = full_image_100x.extract_area(image_x100_tile_x_pos, image_x100_tile_y_pos, self.tile_size, self.tile_size)
                elif self.which_scale_to_use[1] == '25x':
                    tile_two = full_image_25x.extract_area(image_x25_tile_x_pos, image_x25_tile_y_pos, self.tile_size, self.tile_size)

                # Write tile to memory and convert to numpy array
                X_img_one.append(np.ndarray(buffer=tile_one.write_to_memory(),
                                            dtype=my_constants.format_to_dtype[tile_one.format],
                                            shape=[tile_one.height, tile_one.width, tile_one.bands]))

                X_img_two.append(np.ndarray(buffer=tile_two.write_to_memory(),
                                            dtype=my_constants.format_to_dtype[tile_two.format],
                                            shape=[tile_two.height, tile_two.width, tile_two.bands]))

        X_one = np.array(X_img_one)
        X_two = np.array(X_img_two)
        return [X_one, X_two]


class mode_7c_tri_coordinates_generator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, all_xy_pos_in_scn_img, batch_size, n_classes, shuffle, TILE_SIZE, scn_path, scn_offset_x, scn_offset_y,
                 which_scale_to_use):
        """Initialization.

        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.all_xy_pos_in_scn_img = all_xy_pos_in_scn_img
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tile_size = TILE_SIZE
        self.scn_path = scn_path
        self.scn_offset_x = scn_offset_x
        self.scn_offset_y = scn_offset_y
        self.which_scale_to_use = which_scale_to_use
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # Added one to allow last batch smaller
        # return int(np.floor(len(self.all_xy_pos_in_scn_img) / self.batch_size)) + 1
        return int(np.ceil(len(self.all_xy_pos_in_scn_img) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch

        # Modified to allow smaller last batch
        if ((index + 1) * self.batch_size) > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        all_xy_pos_in_scn_img_temp = [self.all_xy_pos_in_scn_img[k] for k in indexes]

        # Generate data
        [X_400x, X_100x, X_25x] = self.__data_generation(all_xy_pos_in_scn_img_temp)

        return [X_400x, X_100x, X_25x]

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.all_xy_pos_in_scn_img))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_xy_pos_in_scn_img_temp):
        """Generates data containing batch_size samples."""
        X_img_400x = []
        X_img_100x = []
        X_img_25x = []

        # Read image
        full_image_400x = pyvips.Image.new_from_file(self.scn_path, level=0).flatten().rot(1)
        full_image_100x = pyvips.Image.new_from_file(self.scn_path, level=1).flatten().rot(1)
        full_image_25x = pyvips.Image.new_from_file(self.scn_path, level=2).flatten().rot(1)

        # Generate data
        for i, current_xy_center_pos_400x in enumerate(all_xy_pos_in_scn_img_temp):
            # Transform the coordinate from the 400x image to the 100x image
            image_x100_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_400x[0]) * my_constants.Scale_between_100x_400x) - (self.tile_size / 2))
            image_x100_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_400x[1]) * my_constants.Scale_between_100x_400x) - (self.tile_size / 2))

            # Transform the coordinate from the 40x image to the 25x image
            image_x25_tile_x_pos = int(((self.scn_offset_x + current_xy_center_pos_400x[0]) * my_constants.Scale_between_25x_400x) - (self.tile_size / 2))
            image_x25_tile_y_pos = int(((self.scn_offset_y + current_xy_center_pos_400x[1]) * my_constants.Scale_between_25x_400x) - (self.tile_size / 2))

            # Extract first tile
            tile_400x = full_image_400x.extract_area((current_xy_center_pos_400x[0] - self.tile_size / 2),
                                                     (current_xy_center_pos_400x[1] - self.tile_size / 2),
                                                     self.tile_size, self.tile_size)
            tile_100x = full_image_100x.extract_area(image_x100_tile_x_pos, image_x100_tile_y_pos, self.tile_size, self.tile_size)
            tile_25x = full_image_25x.extract_area(image_x25_tile_x_pos, image_x25_tile_y_pos, self.tile_size, self.tile_size)

            # Write tile to memory and convert to numpy array
            X_img_400x.append(np.ndarray(buffer=tile_400x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_400x.format],
                                         shape=[tile_400x.height, tile_400x.width, tile_400x.bands]))

            X_img_100x.append(np.ndarray(buffer=tile_100x.write_to_memory(),
                                         dtype=my_constants.format_to_dtype[tile_100x.format],
                                         shape=[tile_100x.height, tile_100x.width, tile_100x.bands]))

            X_img_25x.append(np.ndarray(buffer=tile_25x.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_25x.format],
                                        shape=[tile_25x.height, tile_25x.width, tile_25x.bands]))

        X_400x = np.array(X_img_400x)
        X_100x = np.array(X_img_100x)
        X_25x = np.array(X_img_25x)
        return [X_400x, X_100x, X_25x]


# endregion


# region DEEP LEARNING / NEURAL NETWORKS
def list_of_CVTL_models(LAYER_CONFIG, BASE_MODEL, OPTIMIZER, LEARNING_RATE, N_NEURONS_FIRST_LAYER,
                        N_NEURONS_SECOND_LAYER, N_NEURONS_THIRD_LAYER, DROPOUT, freeze_base_model,
                        BASE_MODEL_POOLING, which_model_mode_to_use, which_scale_to_use_mono, which_scale_to_use_di,
                        augmentation_multiplier, WHAT_LABELS_TO_USE, middle_layer_config,
                        n_neurons_mid_first_layer, n_neurons_mid_second_layer):
    temp_model_dict = dict()
    TL_MODEL_PARAMETERS = []
    TL_MODELS_AND_LOSS_ARRAY = dict()

    # Create list of all classifier models
    n = 0
    for layer in LAYER_CONFIG:
        for mid_layer in middle_layer_config:
            for base_model in BASE_MODEL:
                for freeze in freeze_base_model:
                    for base_model_pooling in BASE_MODEL_POOLING:
                        for optimizer in OPTIMIZER:
                            for lr in LEARNING_RATE:
                                for n_mid_neurons1 in n_neurons_mid_first_layer:
                                    for n_mid_neurons2 in n_neurons_mid_second_layer:
                                        for n_neurons1 in N_NEURONS_FIRST_LAYER:
                                            for n_neurons2 in N_NEURONS_SECOND_LAYER:
                                                for n_neurons3 in N_NEURONS_THIRD_LAYER:
                                                    for dropout in DROPOUT:
                                                        for aug_mult in augmentation_multiplier:
                                                            for label in WHAT_LABELS_TO_USE:
                                                                if 'mono' in which_model_mode_to_use:
                                                                    for scale_in_mono in which_scale_to_use_mono:
                                                                        temp_model_dict.update(dict(ID=n, layer_config=layer, base_model=base_model, optimizer=optimizer, learning_rate=lr,
                                                                                                    n_neurons1=n_neurons1, n_neurons2=n_neurons2, n_neurons3=n_neurons3, dropout=dropout,
                                                                                                    freeze_base_model=freeze, base_model_pooling=base_model_pooling,
                                                                                                    which_scale_to_use=scale_in_mono, trained_epoches=0, early_stopping=0,
                                                                                                    augment_multiplier=aug_mult, model_mode='mono', model_trained_flag=0, fold_trained_flag=0,
                                                                                                    what_labels_to_use=label, mid_layer_config=mid_layer, n_mid_neurons1=n_mid_neurons1,
                                                                                                    n_mid_neurons2=n_mid_neurons2))
                                                                        TL_MODEL_PARAMETERS.append(temp_model_dict.copy())
                                                                        n = n + 1

                                                                if 'di' in which_model_mode_to_use:
                                                                    for scale_in_di in which_scale_to_use_di:
                                                                        temp_model_dict.update(dict(ID=n, layer_config=layer, base_model=base_model, optimizer=optimizer, learning_rate=lr,
                                                                                                    n_neurons1=n_neurons1, n_neurons2=n_neurons2, n_neurons3=n_neurons3, dropout=dropout,
                                                                                                    freeze_base_model=freeze, base_model_pooling=base_model_pooling,
                                                                                                    which_scale_to_use=scale_in_di, trained_epoches=0, early_stopping=0,
                                                                                                    augment_multiplier=aug_mult, model_mode='di', model_trained_flag=0, fold_trained_flag=0,
                                                                                                    what_labels_to_use=label, mid_layer_config=mid_layer, n_mid_neurons1=n_mid_neurons1,
                                                                                                    n_mid_neurons2=n_mid_neurons2))
                                                                        TL_MODEL_PARAMETERS.append(temp_model_dict.copy())
                                                                        n = n + 1

                                                                if 'tri' in which_model_mode_to_use:
                                                                    temp_model_dict.update(dict(ID=n, layer_config=layer, base_model=base_model, optimizer=optimizer, learning_rate=lr,
                                                                                                n_neurons1=n_neurons1, n_neurons2=n_neurons2, n_neurons3=n_neurons3, dropout=dropout,
                                                                                                freeze_base_model=freeze, base_model_pooling=base_model_pooling,
                                                                                                which_scale_to_use=['25x', '100x', '400x'], trained_epoches=0, early_stopping=0,
                                                                                                augment_multiplier=aug_mult, model_mode='tri', model_trained_flag=0, fold_trained_flag=0,
                                                                                                what_labels_to_use=label, mid_layer_config=mid_layer, n_mid_neurons1=n_mid_neurons1,
                                                                                                n_mid_neurons2=n_mid_neurons2))
                                                                    TL_MODEL_PARAMETERS.append(temp_model_dict.copy())
                                                                    n = n + 1

    return TL_MODEL_PARAMETERS, TL_MODELS_AND_LOSS_ARRAY


def get_mono_scale_model(img_width, img_height, n_channels, N_CLASSES, base_model, layer_config, n_neurons1,
                         n_neurons2, n_neurons3, freeze_base_model, base_model_pooling, dropout):
    # Model input
    image_input = Input(shape=(img_height, img_width, n_channels), name='input2')

    # Load base model
    if base_model == 'VGG16':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:
            TL_base_model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.get_layer('block5_pool').output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'VGG19':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:
            TL_base_model = VGG19(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.get_layer('block5_pool').output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'Xception':
        # Check that input size is valid for this base model.
        if img_width >= 71 and img_height >= 71:
            TL_base_model = Xception(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 71. Stopping program.', error=True)
            exit()
    elif base_model == 'Resnet50':
        # Check that input size is valid for this base model.
        if img_width >= 197 and img_height >= 197:
            TL_base_model = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 197. Stopping program.', error=True)
            exit()
    elif base_model == 'DenseNet':
        # Check that input size is valid for this base model.
        if img_width >= 32 and img_height >= 32:
            TL_base_model = DenseNet121(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 197. Stopping program.', error=True)
            exit()
    elif base_model == 'MobileNet':
        # Check that input size is valid for this base model.
        if img_width >= 32 and img_height >= 32:
            TL_base_model = MobileNet(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 197. Stopping program.', error=True)
            exit()
    elif base_model == 'MobileNetV2':
        # Check that input size is valid for this base model.
        if img_width >= 32 and img_height >= 32:
            TL_base_model = MobileNetV2(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 197. Stopping program.', error=True)
            exit()
    elif base_model == 'NASNetLarge':
        # Check that input size is valid for this base model.
        if img_width >= 32 and img_height >= 32:
            TL_base_model = NASNetLarge(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 197. Stopping program.', error=True)
            exit()
    elif base_model == 'NASNetMobile':
        # Check that input size is valid for this base model.
        if img_width >= 32 and img_height >= 32:
            TL_base_model = NASNetMobile(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 197. Stopping program.', error=True)
            exit()
    elif base_model == 'InceptionV3':
        # Check that input size is valid for this base model.
        if img_width >= 139 and img_height >= 139:
            TL_base_model = InceptionV3(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 139. Stopping program.', error=True)
            exit()
    elif base_model == 'InceptionResNetV2':
        # Check that input size is valid for this base model.
        if img_width >= 139 and img_height >= 139:
            TL_base_model = InceptionResNetV2(input_tensor=image_input, include_top=False, weights='imagenet', pooling=None)
            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                flatten_layer = GlobalAveragePooling2D()(last_layer)
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                flatten_layer = GlobalMaxPooling2D()(last_layer)
            elif base_model_pooling in ['NONE', 'None', 'none']:
                flatten_layer = Flatten(name='flatten')(last_layer)
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 139. Stopping program.', error=True)
            exit()
    else:
        my_print('Error in transfer learning base_model. Please choose another base model. Stopping program.', error=True)
        exit()

    # Get size of "latent vector"
    latent_vector_size = 1
    if base_model_pooling in ['NONE', 'None', 'none']:
        for n in range(1, 4):
            latent_vector_size *= last_layer.get_shape().as_list()[n]
    elif base_model_pooling in ['AVG', 'Avg', 'avg', 'MAX', 'Max', 'max']:
        latent_vector_size = flatten_layer.get_shape().as_list()[1]

    # Freeze all convolutional layers
    if freeze_base_model is True or freeze_base_model == 'Hybrid':
        for layer in TL_base_model.layers:
            layer.trainable = False

    # Define new classifier architecture
    if layer_config == 'config0':
        my_dense = flatten_layer
    elif layer_config == 'config1':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
    elif layer_config == 'config2':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
    elif layer_config == 'config3':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    elif layer_config == 'config1_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config2_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config3_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
        # my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    else:
        my_print('Error in layer_config. Please choose another layer.', error=True)
        exit()

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(N_CLASSES, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(image_input, my_output)

    return TL_classifier_model, latent_vector_size


def get_di_scale_model(img_width, img_height, n_channels, N_CLASSES, base_model, layer_config, n_neurons1,
                       n_neurons2, n_neurons3, freeze_base_model, base_model_pooling, dropout):
    # Model input
    image_input_400x = Input(shape=(img_width, img_height, n_channels), name='input_400x')
    image_input_100x = Input(shape=(img_width, img_height, n_channels), name='input_100x')

    if base_model == 'VGG16':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            # last_layer_10x = base_model_10x.get_layer('block5_pool').output
            # last_layer_40x = base_model_40x.get_layer('block5_pool').output

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'VGG19':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            # last_layer_10x = base_model_10x.get_layer('block5_pool').output
            # last_layer_40x = base_model_40x.get_layer('block5_pool').output

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'Xception':
        # Check that input size is valid for this base model.
        if img_width >= 71 and img_height >= 71:

            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            # last_layer = TL_base_model.output
            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:
                base_model_400x = Xception(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = Xception(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')
                # Remove the existing classifier from the model, get the last convolutional/pooling layer.
                last_layer_400x = base_model_400x.output
                last_layer_100x = base_model_100x.output
            elif base_model_pooling in ['MAX', 'Max', 'max']:
                base_model_400x = Xception(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = Xception(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')
                # Remove the existing classifier from the model, get the last convolutional/pooling layer.
                last_layer_400x = base_model_400x.output
                last_layer_100x = base_model_100x.output
            elif base_model_pooling in ['NONE', 'None', 'none']:
                base_model_400x = Xception(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = Xception(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)
                # Remove the existing classifier from the model, get the last convolutional/pooling layer.
                last_layer_400x = base_model_400x.output
                last_layer_100x = base_model_100x.output
            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 71. Stopping program.', error=True)
            exit()
    else:
        my_print('Error in transfer learning base_model. Please choose another base model. Stopping program.', error=True)
        exit()

    # Get size of "latent vector"
    if base_model_pooling in ['AVG', 'Avg', 'avg', 'MAX', 'Max', 'max']:
        latent_vector_size = last_layer_400x.get_shape().as_list()[1]

    # Rename all layers in first model
    for layer in base_model_100x.layers:
        layer.name = layer.name + str("_10x")

    # Freeze all convolutional layers
    if freeze_base_model is True or freeze_base_model == 'Hybrid':
        for layer in base_model_400x.layers:
            layer.trainable = False

        for layer in base_model_100x.layers:
            layer.trainable = False

    # Concatenate models
    flatten_layer = keras.layers.concatenate([last_layer_400x, last_layer_100x], axis=-1)

    # Define new classifier architecture
    if layer_config == 'config0':
        my_dense = flatten_layer
    elif layer_config == 'config1':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
    elif layer_config == 'config2':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
    elif layer_config == 'config3':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    elif layer_config == 'config1_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config2_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config3_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    else:
        my_print('Error in layer_config. Please choose another layer.', error=True)
        exit()

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(N_CLASSES, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(inputs=[image_input_400x, image_input_100x], outputs=my_output)

    return TL_classifier_model, latent_vector_size


def get_tri_scale_model(img_width, img_height, n_channels, N_CLASSES, base_model, layer_config, n_neurons1,
                        n_neurons2, n_neurons3, freeze_base_model, base_model_pooling, dropout):
    # Model input
    image_input_400x = Input(shape=(img_width, img_height, n_channels), name='input_400x')
    image_input_100x = Input(shape=(img_width, img_height, n_channels), name='input_100x')
    image_input_25x = Input(shape=(img_width, img_height, n_channels), name='input_25x')

    if base_model == 'VGG16':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            # last_layer_10x = base_model_10x.get_layer('block5_pool').output
            # last_layer_40x = base_model_40x.get_layer('block5_pool').output

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')
                base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')
                base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)
                base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)
                last_layer_25x = Flatten(name='flatten_25x')(last_layer_25x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    elif base_model == 'VGG19':
        # Check that input size is valid for this base model.
        if img_width >= 48 and img_height >= 48:

            # Remove the existing classifier from the model, get the last convolutional/pooling layer.
            # last_layer_10x = base_model_10x.get_layer('block5_pool').output
            # last_layer_40x = base_model_40x.get_layer('block5_pool').output

            # FLATTEN LAYER
            if base_model_pooling in ['AVG', 'Avg', 'avg']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')
                base_model_25x = VGG19(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='avg')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['MAX', 'Max', 'max']:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='max')
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='max')
                base_model_25x = VGG19(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='max')

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output
            elif base_model_pooling in ['NONE', 'None', 'none', None]:

                base_model_400x = VGG19(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling=None)
                base_model_100x = VGG19(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling=None)
                base_model_25x = VGG19(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling=None)

                last_layer_400x = base_model_400x.layers[-1].output
                last_layer_100x = base_model_100x.layers[-1].output
                last_layer_25x = base_model_25x.layers[-1].output

                # Get size of "latent vector"
                latent_vector_size = 1
                for n in range(1, 4):
                    latent_vector_size *= last_layer_400x.get_shape().as_list()[n]

                last_layer_400x = Flatten(name='flatten_400x')(last_layer_400x)
                last_layer_100x = Flatten(name='flatten_100x')(last_layer_100x)
                last_layer_25x = Flatten(name='flatten_25x')(last_layer_25x)

            else:
                my_print('Error in base_model_pooling_MSTL. Stopping program.', error=True)
                exit()
        else:
            my_print('Error in input size for this transfer learning base model. Minimum size is 48. Stopping program.', error=True)
            exit()
    else:
        my_print('Error in transfer learning base_model. Please choose another base model. Stopping program.', error=True)
        exit()

    # Get size of "latent vector"
    if base_model_pooling in ['AVG', 'Avg', 'avg', 'MAX', 'Max', 'max']:
        latent_vector_size = last_layer_400x.get_shape().as_list()[1]

    # Rename all layers in first model
    for layer in base_model_100x.layers:
        layer.name = layer.name + str("_100x")

    # Rename all layers in second model
    for layer in base_model_25x.layers:
        layer.name = layer.name + str("_25x")

    # Freeze all convolutional layers
    if freeze_base_model is True or freeze_base_model == 'Hybrid':
        for layer in base_model_400x.layers:
            layer.trainable = False

        for layer in base_model_100x.layers:
            layer.trainable = False

        for layer in base_model_25x.layers:
            layer.trainable = False

    # Add fully-connected layers after each VGG16 and before concatenation
    # last_layer_40x = Dense(512, activation='relu', name='pre_dense_40x')(last_layer_40x)
    # last_layer_10x = Dense(512, activation='relu', name='pre_dense_10')(last_layer_10x)
    # last_layer_2x = Dense(512, activation='relu', name='pre_dense_2x')(last_layer_2x)

    # Concatenate models
    flatten_layer = keras.layers.concatenate([last_layer_400x, last_layer_100x, last_layer_25x], axis=-1)

    # Define new classifier architecture
    if layer_config == 'config0':
        my_dense = flatten_layer
    elif layer_config == 'config1':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
    elif layer_config == 'config2':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
    elif layer_config == 'config3':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    elif layer_config == 'config1_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config2_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
    elif layer_config == 'config3_drop':
        my_dense = Dense(n_neurons1, activation='relu', name='my_dense1')(flatten_layer)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons2, activation='relu', name='my_dense2')(my_dense)
        my_dense = Dropout(rate=dropout, noise_shape=None, seed=None)(my_dense)
        my_dense = Dense(n_neurons3, activation='relu', name='my_dense3')(my_dense)
    else:
        my_print('Error in layer_config. Please choose another layer.', error=True)
        exit()

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(N_CLASSES, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(inputs=[image_input_400x, image_input_100x, image_input_25x], outputs=my_output)

    return TL_classifier_model, latent_vector_size


# endregion


# region PLOTTING / SAVING
def my_plot_misclassifications(current_epoch, image_400x, y_true, y_pred, N_CLASSES_TRAINING,
                               SUMMARY_PATH, model_name, FIGURE_PATH, n_channels, name_of_classes_array, prediction_type):
    # Plot input image_400x together with my_pred and save to plot_filepath

    # Calculate number of plots
    if len(y_true) > 10:
        n_figures = math.ceil(len(y_true) / 10)
    else:
        n_figures = 1

    # Create path and check that folder exist
    plot_filepath = '{}{}{}/'.format(SUMMARY_PATH, model_name, FIGURE_PATH)
    os.makedirs(plot_filepath, exist_ok=True)

    # Make a vector with the same number as classes
    classes = [i for i in range(N_CLASSES_TRAINING)]

    # Plot each figure
    for current_figure in range(n_figures):

        # Calculate number of sub-plots in current figure
        if current_figure == n_figures - 1:
            n_plots = len(y_true) - 10 * (n_figures - 1)
        else:
            n_plots = 10

        # Make a new figure
        fig, a = plt.subplots(2, n_plots, figsize=(10, 3), squeeze=False)

        # Loop through each sub-plot
        for current_plot in range(n_plots):

            # Update current image_400x index
            current_img = 10 * current_figure + current_plot

            if current_plot == 0:
                a[0][current_plot].set_title('Input images - {} (epoch {})'.format(prediction_type, current_epoch), loc='left', fontsize=12)
                # a[1][current_plot].set_title('Prediction', loc='left', fontsize=12)
                a[1][current_plot].set_yticks(classes)  # Set number of y-ticks
                # a[1][i].set_yticklabels(['Class 0:', 'Class 1:'])  # Set label of y-ticks
                a[1][current_plot].set_yticklabels(name_of_classes_array)  # Set label of y-ticks
                a[1][current_plot].set_xticks([0, 1])  # specify x-ticks
                a[1][current_plot].set_xticklabels(['0%', '100%'])  # Set label of y-ticks
            else:
                a[1][current_plot].set_xticks([0, 1])  # Set number of y-ticks
                a[1][current_plot].set_xticklabels([''])  # Set label of y-ticks
                a[1][current_plot].set_yticks(classes)  # Set number of y-ticks
                a[1][current_plot].set_yticklabels([''])  # Set label of y-ticks

            a[0][current_plot].set_yticks([])  # Remove y-ticks
            a[0][current_plot].set_xticks([])  # Remove x-ticks

            a[0][current_plot].spines['top'].set_linewidth(0)  # remove boarder around image_400x
            a[0][current_plot].spines['right'].set_linewidth(0)  # remove boarder around image_400x
            a[0][current_plot].spines['bottom'].set_linewidth(0)  # remove boarder around image_400x
            a[0][current_plot].spines['left'].set_linewidth(0)  # remove boarder around image_400x

            # Plot image_400x and bar chart
            # Keras generated images is in a float64 format, which appears to be fine for Keras, but not for Matplotlib.
            # Casting the NumPy array to a uint8 for correct colors
            if n_channels == 1:
                a[0][current_plot].imshow(np.squeeze(image_400x[current_img].astype(np.uint8)), cmap='gray')  # plot image_400x
            elif n_channels == 3:
                a[0][current_plot].imshow(image_400x[current_img].astype(np.uint8))  # plot image_400x

            # Create a vector for the true label, used to draw on the bar chart.
            y_true_draw_vector = [0] * N_CLASSES_TRAINING
            y_true_draw_vector[y_true[current_img]] = 1

            # Draw the vectors on the bar chart.
            a[1][current_plot].barh(y=classes, width=y_true_draw_vector, align='center', color='green', edgecolor='None')  # plot horizontal bar graph
            a[1][current_plot].barh(y=classes, width=np.round(y_pred[current_img], 1), height=0.5, color='red', align='center', edgecolor='None')  # plot horizontal bar graph

            # Plot axis label
            if current_plot == 0:
                a[0][current_plot].set_xlabel('True class: ' + str(y_true[current_img]), fontsize=8)  # set the xlabel
            else:
                a[0][current_plot].set_xlabel(y_true[current_img], fontsize=8)  # set the xlabel

            # a[1][i].set_yticks(classes)  # Set number of y-ticks
            a[1][current_plot].tick_params(axis='both', which='major', labelsize=6)
            # a[1][i].set_xlabel('score', fontsize=8)             # Set xlabel

            a[1][current_plot].spines['top'].set_linewidth(0.2)  # specify boarder width around image_400x
            a[1][current_plot].spines['right'].set_linewidth(0.2)  # specify boarder width around image_400x
            a[1][current_plot].spines['bottom'].set_linewidth(0.2)  # specify boarder width around image_400x
            a[1][current_plot].spines['left'].set_linewidth(0.2)  # specify boarder width around image_400x

            for tic in a[1][current_plot].xaxis.get_major_ticks():
                tic.tick1line.set_visible = False
            for tic in a[1][current_plot].yaxis.get_major_ticks():
                tic.tick1line.set_visible = False

            # Go to next plot
            current_plot += 1

        # plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
        fig.savefig(plot_filepath + 'Epoch_' + str(current_epoch) + '_Figure_' + str(current_figure) + '.png', dpi=200)
        # fig.savefig(plot_filepath + 'Figure_' + str(image_index) + '.png', dpi=200)
        plt.close(fig)
        plt.cla()


def my_plot_di_scale_misclassifications(current_epoch, image_400x, image_100x, y_true, y_pred, N_CLASSES_TRAINING,
                                        SUMMARY_PATH, model_name, FIGURE_PATH, n_channels, name_of_classes_array, prediction_type):
    # Plot input image_400x together with my_pred and save to plot_filepath

    # Calculate number of plots
    if len(y_true) > 10:
        n_figures = math.ceil(len(y_true) / 10)
    else:
        n_figures = 1

    # Create path and check that folder exist
    # plot_filepath = '{}{}{}/{}/'.format(current_model_path, SUMMARY_PATH, FIGURE_PATH, folder)
    plot_filepath = '{}{}{}/'.format(SUMMARY_PATH, model_name, FIGURE_PATH)
    os.makedirs(plot_filepath, exist_ok=True)

    # Make a vector with the same number as classes
    classes = [i for i in range(N_CLASSES_TRAINING)]

    # Plot each figure
    for current_figure in range(n_figures):

        # Calculate number of sub-plots in current figure
        if current_figure == n_figures - 1:
            n_plots = len(y_true) - 10 * (n_figures - 1)
        else:
            n_plots = 10

        # Make a new figure
        fig, a = plt.subplots(3, n_plots, figsize=(10, 4), squeeze=False)

        # Loop through each sub-plot
        for current_plot in range(n_plots):

            # Update current image_400x index
            current_img = 10 * current_figure + current_plot

            if current_plot == 0:
                a[0][current_plot].set_title('Input images - {} (epoch {})'.format(prediction_type, current_epoch), loc='left', fontsize=12)
                a[1][current_plot].set_title('Input images', loc='left', fontsize=12)
                # a[2][current_plot].set_title('Prediction', loc='left', fontsize=12)
                a[2][current_plot].set_yticks(classes)  # Set number of y-ticks
                # a[2][i].set_yticklabels(['Class 0:', 'Class 1:'])  # Set label of y-ticks
                a[2][current_plot].set_yticklabels(name_of_classes_array)  # Set label of y-ticks
                a[2][current_plot].set_xticks([0, 1])  # specify x-ticks
                a[2][current_plot].set_xticklabels(['0%', '100%'])  # Set label of y-ticks
            else:
                a[2][current_plot].set_xticks([0, 1])  # Set number of y-ticks
                a[2][current_plot].set_xticklabels([''])  # Set label of y-ticks
                a[2][current_plot].set_yticks(classes)  # Set number of y-ticks
                a[2][current_plot].set_yticklabels([''])  # Set label of y-ticks

            a[0][current_plot].set_yticks([])  # Remove y-ticks
            a[0][current_plot].set_xticks([])  # Remove x-ticks
            a[1][current_plot].set_yticks([])  # Remove y-ticks
            a[1][current_plot].set_xticks([])  # Remove x-ticks

            a[0][current_plot].spines['top'].set_linewidth(0)  # remove boarder around image_400x
            a[0][current_plot].spines['right'].set_linewidth(0)  # remove boarder around image_400x
            a[0][current_plot].spines['bottom'].set_linewidth(0)  # remove boarder around image_400x
            a[0][current_plot].spines['left'].set_linewidth(0)  # remove boarder around image_400x

            a[1][current_plot].spines['top'].set_linewidth(0)  # remove boarder around image_400x
            a[1][current_plot].spines['right'].set_linewidth(0)  # remove boarder around image_400x
            a[1][current_plot].spines['bottom'].set_linewidth(0)  # remove boarder around image_400x
            a[1][current_plot].spines['left'].set_linewidth(0)  # remove boarder around image_400x

            # Plot image_400x and bar chart
            # Keras generated images is in a float64 format, which appears to be fine for Keras, but not for Matplotlib.
            # Casting the NumPy array to a uint8 for correct colors
            if n_channels == 1:
                a[0][current_plot].imshow(np.squeeze(image_400x[current_img].astype(np.uint8)), cmap='gray')  # plot image_400x
                a[1][current_plot].imshow(np.squeeze(image_100x[current_img].astype(np.uint8)), cmap='gray')  # plot image_400x
            elif n_channels == 3:
                a[0][current_plot].imshow(image_400x[current_img].astype(np.uint8))  # plot image_400x
                a[1][current_plot].imshow(image_100x[current_img].astype(np.uint8))  # plot image_400x

            # Create a vector for the true label, used to draw on the bar chart.
            y_true_draw_vector = [0] * N_CLASSES_TRAINING
            y_true_draw_vector[y_true[current_img]] = 1

            # Draw the vectors on the bar chart.
            a[2][current_plot].barh(classes, y_true_draw_vector, align='center', color='green', edgecolor='None')  # plot horizontal bar graph
            a[2][current_plot].barh(classes, np.round(y_pred[current_img], 1), height=0.5, color='red', align='center', edgecolor='None')  # plot horizontal bar graph

            # Plot axis label
            if current_plot == 0:
                a[1][current_plot].set_xlabel('True class: ' + str(y_true[current_img]), fontsize=8)  # set the xlabel
            else:
                a[1][current_plot].set_xlabel(y_true[current_img], fontsize=8)  # set the xlabel

            # a[2][i].set_yticks(classes)  # Set number of y-ticks
            a[2][current_plot].tick_params(axis='both', which='major', labelsize=6)
            # a[2][i].set_xlabel('score', fontsize=8)             # Set xlabel

            a[2][current_plot].spines['top'].set_linewidth(0.2)  # specify boarder width around image_400x
            a[2][current_plot].spines['right'].set_linewidth(0.2)  # specify boarder width around image_400x
            a[2][current_plot].spines['bottom'].set_linewidth(0.2)  # specify boarder width around image_400x
            a[2][current_plot].spines['left'].set_linewidth(0.2)  # specify boarder width around image_400x

            for tic in a[2][current_plot].xaxis.get_major_ticks():
                tic.tick1line.set_visible = False
            for tic in a[2][current_plot].yaxis.get_major_ticks():
                tic.tick1line.set_visible = False

            # Go to next plot
            current_plot += 1

        # plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
        # fig.savefig(plot_filepath + plot_timestamp + '.png', dpi=200)
        # fig.savefig(plot_filepath + 'Figure_' + str(current_figure) + '.png', dpi=200)
        fig.savefig(plot_filepath + 'Epoch_' + str(current_epoch) + '_Figure_' + str(current_figure) + '.png', dpi=200)
        # fig.savefig(plot_filepath + 'Figure_' + str(image_index) + '.png', dpi=200)
        plt.close(fig)
        plt.cla()


def my_plot_tri_scale_misclassifications(current_epoch, image_400x, image_100x, image_25x, y_true, y_pred, N_CLASSES_TRAINING,
                                         SUMMARY_PATH, model_name, FIGURE_PATH, n_channels, name_of_classes_array, prediction_type):
    # Plot input image together with my_pred and save to plot_filepath

    # Calculate number of plots
    if len(y_true) > 10:
        n_figures = math.ceil(len(y_true) / 10)
    else:
        n_figures = 1

    # Create path and check that folder exist
    # plot_filepath = '{}{}{}/{}/'.format(current_model_path, SUMMARY_PATH, FIGURE_PATH, folder)
    plot_filepath = '{}{}{}/'.format(SUMMARY_PATH, model_name, FIGURE_PATH)
    os.makedirs(plot_filepath, exist_ok=True)

    # Make a vector with the same number as classes
    classes = [i for i in range(N_CLASSES_TRAINING)]

    # Plot each figure
    for current_figure in range(n_figures):

        # Calculate number of sub-plots in current figure
        if current_figure == n_figures - 1:
            n_plots = len(y_true) - 10 * (n_figures - 1)
        else:
            n_plots = 10

        # Make a new figure
        fig, a = plt.subplots(4, n_plots, figsize=(10, 5), squeeze=False)

        # Loop through each sub-plot
        for current_plot in range(n_plots):

            # Update current image index
            current_img = 10 * current_figure + current_plot

            if current_plot == 0:
                a[0][current_plot].set_title('Input images 400x - {} (epoch {})'.format(prediction_type, current_epoch), loc='left', fontsize=12)
                a[1][current_plot].set_title('Input images 100x', loc='left', fontsize=12)
                a[2][current_plot].set_title('Input images 25x', loc='left', fontsize=12)
                # a[3][current_plot].set_title('Prediction', loc='left', fontsize=12)
                a[3][current_plot].set_yticks(classes)  # Set number of y-ticks
                # a[3][i].set_yticklabels(['Class 0:', 'Class 1:'])  # Set label of y-ticks
                a[3][current_plot].set_yticklabels(name_of_classes_array)  # Set label of y-ticks
                a[3][current_plot].set_xticks([0, 1])  # specify x-ticks
                a[3][current_plot].set_xticklabels(['0%', '100%'])  # Set label of y-ticks
            else:
                a[3][current_plot].set_xticks([0, 1])  # Set number of y-ticks
                a[3][current_plot].set_xticklabels([''])  # Set label of y-ticks
                a[3][current_plot].set_yticks(classes)  # Set number of y-ticks
                a[3][current_plot].set_yticklabels([''])  # Set label of y-ticks

            a[0][current_plot].set_yticks([])  # Remove y-ticks
            a[0][current_plot].set_xticks([])  # Remove x-ticks
            a[1][current_plot].set_yticks([])  # Remove y-ticks
            a[1][current_plot].set_xticks([])  # Remove x-ticks
            a[2][current_plot].set_yticks([])  # Remove y-ticks
            a[2][current_plot].set_xticks([])  # Remove x-ticks

            a[0][current_plot].spines['top'].set_linewidth(0)  # remove boarder around image
            a[0][current_plot].spines['right'].set_linewidth(0)  # remove boarder around image
            a[0][current_plot].spines['bottom'].set_linewidth(0)  # remove boarder around image
            a[0][current_plot].spines['left'].set_linewidth(0)  # remove boarder around image

            a[1][current_plot].spines['top'].set_linewidth(0)  # remove boarder around image
            a[1][current_plot].spines['right'].set_linewidth(0)  # remove boarder around image
            a[1][current_plot].spines['bottom'].set_linewidth(0)  # remove boarder around image
            a[1][current_plot].spines['left'].set_linewidth(0)  # remove boarder around image

            a[2][current_plot].spines['top'].set_linewidth(0)  # remove boarder around image
            a[2][current_plot].spines['right'].set_linewidth(0)  # remove boarder around image
            a[2][current_plot].spines['bottom'].set_linewidth(0)  # remove boarder around image
            a[2][current_plot].spines['left'].set_linewidth(0)  # remove boarder around image

            # Plot image and bar chart
            # Keras generated images is in a float64 format, which appears to be fine for Keras, but not for Matplotlib.
            # Casting the NumPy array to a uint8 for correct colors
            if n_channels == 1:
                a[0][current_plot].imshow(np.squeeze(image_400x[current_img].astype(np.uint8)), cmap='gray')  # plot image
                a[1][current_plot].imshow(np.squeeze(image_100x[current_img].astype(np.uint8)), cmap='gray')  # plot image
                a[2][current_plot].imshow(np.squeeze(image_25x[current_img].astype(np.uint8)), cmap='gray')  # plot image
            elif n_channels == 3:
                a[0][current_plot].imshow(image_400x[current_img].astype(np.uint8))  # plot image
                a[1][current_plot].imshow(image_100x[current_img].astype(np.uint8))  # plot image
                a[2][current_plot].imshow(image_25x[current_img].astype(np.uint8))  # plot image

            # Create a vector for the true label, used to draw on the bar chart.
            y_true_draw_vector = [0] * N_CLASSES_TRAINING
            y_true_draw_vector[y_true[current_img]] = 1

            # Draw the vectors on the bar chart.
            a[3][current_plot].barh(classes, y_true_draw_vector, align='center', color='green', edgecolor='None')  # plot horizontal bar graph
            a[3][current_plot].barh(classes, np.round(y_pred[current_img], 1), height=0.5, color='red', align='center', edgecolor='None')  # plot horizontal bar graph

            # Plot axis label
            if current_plot == 0:
                a[2][current_plot].set_xlabel('True class: ' + str(y_true[current_img]), fontsize=8)  # set the xlabel
            else:
                a[2][current_plot].set_xlabel(y_true[current_img], fontsize=8)  # set the xlabel

            # a[3][i].set_yticks(classes)  # Set number of y-ticks
            a[3][current_plot].tick_params(axis='both', which='major', labelsize=6)
            # a[3][i].set_xlabel('score', fontsize=8)             # Set xlabel

            a[3][current_plot].spines['top'].set_linewidth(0.2)  # specify boarder width around image
            a[3][current_plot].spines['right'].set_linewidth(0.2)  # specify boarder width around image
            a[3][current_plot].spines['bottom'].set_linewidth(0.2)  # specify boarder width around image
            a[3][current_plot].spines['left'].set_linewidth(0.2)  # specify boarder width around image

            for tic in a[3][current_plot].xaxis.get_major_ticks():
                tic.tick1line.set_visible = False
            for tic in a[3][current_plot].yaxis.get_major_ticks():
                tic.tick1line.set_visible = False

            # Go to next plot
            current_plot += 1

        # plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
        # fig.savefig(plot_filepath + plot_timestamp + '.png', dpi=200)
        # fig.savefig(plot_filepath + 'Figure_' + str(current_figure) + '.png', dpi=200)
        # fig.savefig(plot_filepath + 'Figure_' + str(image_index) + '.png', dpi=200)
        fig.savefig(plot_filepath + 'Epoch_' + str(current_epoch) + '_Figure_' + str(current_figure) + '.png', dpi=200)
        plt.close(fig)
        plt.cla()


def plot_confusion_matrix(cm, epoch, classes, SUMMARY_PATH, folder_name, title='Confusion matrix', cmap=plt.cm.Blues):
    # This function prints and plots the confusion matrix.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    plot_filepath = '{}{}/'.format(SUMMARY_PATH, folder_name)
    os.makedirs(plot_filepath, exist_ok=True)
    plt.savefig(plot_filepath + 'Confusion_matrix_epoch_' + str(epoch) + '.png', dpi=200)
    plt.close()
    plt.cla()


def save_history_plot(history, path, mode, model_no):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    title = '{} - model {} - accuracy'.format(mode, model_no)
    plt.title(title)
    plt.grid()
    plt.ylabel('accuracy')
    # plt.ylim(0.6, 1.05)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    plt.savefig(path + 'accuracy_plot_ ' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    title = '{} - model {} - loss'.format(mode, model_no)
    plt.title(title)
    plt.grid()
    plt.ylabel('loss')
    # plt.ylim(-0.5, 1)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(path + 'loss_plot_' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()


def save_history_plot_only_training(history, path, mode, model_no):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    title = '{} - model {} - accuracy'.format(mode, model_no)
    plt.title(title)
    plt.grid()
    plt.ylabel('accuracy')
    # plt.ylim(0.6, 1.05)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plt.legend(['train'], loc='lower right')
    plot_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    plt.savefig(path + 'accuracy_plot_ ' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()

    # summarize history for loss
    plt.plot(history.history['loss'])
    title = '{} - model {} - loss'.format(mode, model_no)
    plt.title(title)
    plt.ylabel('loss')
    # plt.ylim(-0.5, 1)  # set the ylim to ymin, ymax
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig(path + 'loss_plot_' + plot_timestamp + '.png', dpi=200)
    plt.close()
    plt.cla()


def make_heatmap(current_run_path, METADATA_FOLDER, N_CLASSES, NAME_OF_CLASSES, MODE_TISSUE_PREDICTION_FOLDER,
                 HEAT_MAPS_FOLDER, HEATMAP_SIGMA, HEATMAP_THRESHOLD, PREDICT_WINDOW_SIZE, PROBABILITY_IMAGES_PICKLE_FILE,
                 COLORMAP_IMAGES_PICKLE_FILE, current_wsi_pickle_path, heatmap_save_folder):
    my_print('Making heatmaps')

    # Check if we have previously generated list of classifier models
    if os.path.isfile(current_wsi_pickle_path + PROBABILITY_IMAGES_PICKLE_FILE):
        # File exist, load parameters.
        all_probability_images = pickle_load(current_wsi_pickle_path + PROBABILITY_IMAGES_PICKLE_FILE)
    else:
        my_print('No probability pickle file to make heatmaps from. Stopping program.', error=True)
        exit()

    # Check that data exist in the probability images
    if not len(all_probability_images) > 0:
        my_print('Probability pickle file is empty. Stopping program.', error=True)
        exit()

    # name_of_all_classes = NAME_OF_CLASSES.copy()
    # name_of_all_classes.append('undefined')

    # Loop through all probability images
    # for wsi_prob_map in all_probability_images:

    # filename_folder = current_run_path + MODE_TISSUE_PREDICTION_FOLDER + wsi_prob_map + '/' + HEAT_MAPS_FOLDER
    os.makedirs(heatmap_save_folder, exist_ok=True)

    # Set own color map
    # from matplotlib.colors import LinearSegmentedColormap
    # temp = cm.get_cmap('jet')
    # my_cmap.set_under('white')
    # rgba = my_cmap(0.01)
    # my_cmap = matplotlib.cm.get_cmap('jet')
    # my_cmap._init()
    # my_cmap._lut[:, -1] = np.linspace(0, 0.8, 255 + 4)

    #####################################################

    # colormaps: 'jet', 'seismic', 'hot', 'inferno', 'magma', 'plasma', 'viridis'.
    # See full list here: https://matplotlib.org/api/pyplot_summary.html

    # create a colormap that consists of
    # - 1/5 : custom colormap, ranging from white to the first color of the colormap
    # - 4/5 : existing colormap

    # set upper part: 4 * 256/4 entries
    upper = matplotlib.cm.jet(np.arange(256))

    # set lower part: 1 * 256/4 entries
    # - initialize all entries to 1 to make sure that the alpha channel (4th column) is 1
    lower = np.ones((int(256 / 4), 4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

    # combine parts of colormap
    my_cmap = np.vstack((lower, upper))

    # convert to matplotlib colormap
    my_cmap = matplotlib.colors.ListedColormap(my_cmap, name='myColorMap', N=my_cmap.shape[0])

    ##################################################

    # Loop through all classes for each probability map
    for current_class in range(N_CLASSES):
        # Filter the image
        if HEATMAP_SIGMA > 0:
            current_heatmap = scipy.ndimage.filters.gaussian_filter(all_probability_images[current_class], sigma=HEATMAP_SIGMA)

        # Apply threshold to one of the lists. Set all values below threshold to zero
        if HEATMAP_THRESHOLD > 0:
            for row in current_heatmap:
                row[row < HEATMAP_THRESHOLD] = 0

        # current_heatmap = scipy.ndimage.morphology.grey_erosion(input=current_heatmap,
        #                                                         size=(5, 5),
        #                                                         footprint=None,
        #                                                         structure=None,
        #                                                         output=None,
        #                                                         mode='reflect',
        #                                                         cval=0.0,
        #                                                         origin=0)

        title = 'Size:{}, Sigma:{}, Thres:{}, Class:{}'.format(PREDICT_WINDOW_SIZE, HEATMAP_SIGMA, HEATMAP_THRESHOLD, NAME_OF_CLASSES[current_class])
        filename = heatmap_save_folder + 'sigma_' + str(HEATMAP_SIGMA) + '_thres_' + str(HEATMAP_THRESHOLD) + '_' + \
                   NAME_OF_CLASSES[current_class] + '_heatmap_' + str(random.getrandbits(6)) + '.png'

        # Create a new figure
        fig, (ax, cax) = plt.subplots(ncols=2, figsize=(6, 4), gridspec_kw={"width_ratios": [1, 0.05]})
        fig.subplots_adjust(wspace=0.1)

        # Create heat map
        im = ax.imshow(current_heatmap, interpolation='none', cmap=my_cmap, vmin=0, vmax=1)

        # plt.contourf(filtered_img, n_levels, vmin=0, vmax=1, cmap=my_cmap, origin='upper')
        # plt.pcolormesh(filtered_img, cmap=my_cmap, vmin=0, vmax=1)

        # Add contour lines
        # plt.contour(all_probability_images[wsi_prob_map][current_class], 2, colors='K')

        # Remove x,y-ticks
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along left side are off
            top=False,  # ticks along the top edge are off
            labelleft=False,  # labels along the left edge are off
            labelbottom=False)  # labels along the bottom edge are off

        # Set the title
        ax.set_title(title)

        # plt.ylim(plt.ylim()[::-1])

        # Add a colorbar
        # plt.colorbar(boundaries=np.linspace(0,1,5))
        # plt.colorbar(ticks=np.linspace(-.1, 2.0, 15, endpoint=True))
        # plt.colorbar(cs, ticks=[0, 0.25, 0.5, 0.75, 1])

        # from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
        # ip = InsetPosition(ax3, [1.05, 0, 0.05, 1])
        # cax.set_axes_locator(ip)

        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(c, cax=cbar_ax)
        fig.colorbar(im, cax=cax)
        # cbar.ax.set_yticklabels(['0%', '50%', '100%'])  # vertically oriented colorbar

        # fig.tight_layout()

        # Save heat map with title/colorbar
        plt.savefig(filename, dpi=200)
        plt.close()
        plt.cla()


def make_colormap(current_run_path, METADATA_FOLDER, N_CLASSES_ALL, NAME_OF_CLASSES_ALL, threshold, HEATMAP_SIGMA, MODE_TISSUE_PREDICTION_FOLDER,
                  color_mode, background_mask_resized, PROBABILITY_IMAGES_PICKLE_FILE, COLORMAP_IMAGES_PICKLE_FILE,
                  current_wsi_pickle_path, colormap_save_folder):
    my_print('Making colormap')

    # Check if we have previously generated all_colormap_images
    if os.path.isfile(current_wsi_pickle_path + COLORMAP_IMAGES_PICKLE_FILE):
        # File exist, load parameters.
        all_colormap_images = pickle_load(current_wsi_pickle_path + COLORMAP_IMAGES_PICKLE_FILE)
    else:
        my_print('No all_colormap_images pickle file to make colormaps from. Stopping program.', error=True)
        exit()

    # Check that data exist in the probability images
    if not (len(all_colormap_images) > 0):
        my_print('Probability pickle file is empty. Stopping program.', error=True)
        exit()

    # Get seg mask width/height
    seg_mask_width = all_colormap_images[0].shape[1]
    seg_mask_heights = all_colormap_images[0].shape[0]

    if color_mode == 'tissue_binary':
        # NAME_OF_CLASSES_ALL = ['Other', 'Urothelium']

        temp_list = []
        temp_list.append(all_colormap_images[1])
        temp_list.append(all_colormap_images[0])
        temp_list.append(all_colormap_images[2])

        all_colormap_images = dict()
        all_colormap_images = temp_list

        for cur_class in range(N_CLASSES_ALL):
            curr_prob_img = all_colormap_images[cur_class]
            for row in curr_prob_img:
                row[row < threshold] = 0
                row[row >= threshold] = cur_class

        colors = [
            (0, 0, 0),  # Black     - Other
            (0.96078, 0.51, 0.18823),  # Orange    - Urothelium
            (0.5, 0.5, 0.5)  # Grey      - Undefined
        ]

        # Create an empty mask of same size as image
        seg_img = np.zeros(shape=(seg_mask_heights, seg_mask_width, 3))

        # Color the image
        segc_0 = (all_colormap_images[0] == 0)
        segc_1 = (all_colormap_images[1] == 1)
        segc_2 = (all_colormap_images[2] == 2)

        seg_img[:, :, 0] += (segc_0 * (colors[0][0]))
        seg_img[:, :, 1] += (segc_0 * (colors[0][1]))
        seg_img[:, :, 2] += (segc_0 * (colors[0][2]))

        seg_img[:, :, 0] += (segc_1 * (colors[1][0]))
        seg_img[:, :, 1] += (segc_1 * (colors[1][1]))
        seg_img[:, :, 2] += (segc_1 * (colors[1][2]))

        seg_img[:, :, 0] += (segc_2 * (colors[2][0]))
        seg_img[:, :, 1] += (segc_2 * (colors[2][1]))
        seg_img[:, :, 2] += (segc_2 * (colors[2][2]))
    elif color_mode == 'tissue_multiclass':

        for cur_class in range(N_CLASSES_ALL):
            curr_prob_img = all_colormap_images[cur_class]
            for row in curr_prob_img:
                row[row < threshold] = 0
                row[row >= threshold] = cur_class

        colors = [
            (0, 0, 0),  # Black     - Background
            (0.5, 0, 0),  # Maroon    - Blood
            (0, 0.51, 0.7843),  # Blue      - Damaged
            (0.23529, 0.70588, 0.294117),  # Green     - Muscle
            (0.9412, 0.196, 0.9019),  # Magenta   - Stroma
            (0.96078, 0.51, 0.18823),  # Orange    - Urothelium
            (0.5, 0.5, 0.5)  # Grey      - Undefined
        ]

        # RGB Values
        # colors = [
        #     (0, 0, 0),  # Black     - Background
        #     (128, 0, 0),  # Maroon    - Blood
        #     (0, 130, 200),  # Blue      - Damaged
        #     (60, 180, 75),  # Green     - Muscle
        #     (240, 50, 230),  # Magenta   - Stroma
        #     (244, 130, 47),  # Orange    - Urothelium
        #     (128, 128, 128)  # Grey      - Undefined
        # ]

        # Create an empty mask of same size as image
        seg_img = np.zeros(shape=(seg_mask_heights, seg_mask_width, 3))

        for c in range(N_CLASSES_ALL):
            segc = (all_colormap_images[c] == c)
            seg_img[:, :, 0] += (segc * (colors[c][0]))
            seg_img[:, :, 1] += (segc * (colors[c][1]))
            seg_img[:, :, 2] += (segc * (colors[c][2]))
    elif color_mode == 'diagnostic_two_classes':

        # We need to rearrange the order of the classes to get the colors correct on the segmentation images
        new_order = [2, 0, 1, 3]
        NAME_OF_CLASSES_ALL = [NAME_OF_CLASSES_ALL[i] for i in new_order]

        temp_list = []
        temp_list.append(all_colormap_images[2])  # backgorund
        temp_list.append(all_colormap_images[0])  # no recurrence
        temp_list.append(all_colormap_images[1])  # recurrence
        temp_list.append(all_colormap_images[3])  # undefined
        all_colormap_images = dict()
        all_colormap_images = temp_list

        for cur_class in range(N_CLASSES_ALL):
            curr_prob_img = all_colormap_images[cur_class]
            for row in curr_prob_img:
                row[row < threshold] = 0
                row[row >= threshold] = cur_class

        colors = [
            (0, 0, 0),  # Black     - Background
            (0, 0.51, 0.7843),  # Blue      - Damaged
            (0.5, 0, 0),  # Maroon    - Blood
            (0.5, 0.5, 0.5)  # Grey      - Undefined
        ]

        # Create an empty mask of same size as image
        seg_img = np.zeros(shape=(seg_mask_heights, seg_mask_width, 3))

        # Color the image
        segc_0 = (all_colormap_images[0] == 0)  # background
        segc_1 = (all_colormap_images[1] == 1)  # no recurrence
        segc_2 = (all_colormap_images[2] == 2)  # recurrence
        segc_3 = (all_colormap_images[3] == 3)  # undefined

        seg_img[:, :, 0] += (segc_0 * (colors[0][0]))
        seg_img[:, :, 1] += (segc_0 * (colors[0][1]))
        seg_img[:, :, 2] += (segc_0 * (colors[0][2]))

        seg_img[:, :, 0] += (segc_1 * (colors[1][0]))
        seg_img[:, :, 1] += (segc_1 * (colors[1][1]))
        seg_img[:, :, 2] += (segc_1 * (colors[1][2]))

        seg_img[:, :, 0] += (segc_2 * (colors[2][0]))
        seg_img[:, :, 1] += (segc_2 * (colors[2][1]))
        seg_img[:, :, 2] += (segc_2 * (colors[2][2]))

        seg_img[:, :, 0] += (segc_3 * (colors[3][0]))
        seg_img[:, :, 1] += (segc_3 * (colors[3][1]))
        seg_img[:, :, 2] += (segc_3 * (colors[3][2]))
    elif color_mode == 'diagnostic_three_classes':

        # We need to rearrange the order of the classes to get the colors correct on the segmentation images
        new_order = [3, 0, 1, 2, 4]
        NAME_OF_CLASSES_ALL = [NAME_OF_CLASSES_ALL[i] for i in new_order]

        temp_list = []
        temp_list.append(all_colormap_images[3])  # backgorund
        temp_list.append(all_colormap_images[0])  # grade1
        temp_list.append(all_colormap_images[1])  # grade2
        temp_list.append(all_colormap_images[2])  # grade3
        temp_list.append(all_colormap_images[4])  # undefined
        all_colormap_images = dict()
        all_colormap_images = temp_list

        for cur_class in range(N_CLASSES_ALL):
            curr_prob_img = all_colormap_images[cur_class]
            for row in curr_prob_img:
                row[row < threshold] = 0
                row[row >= threshold] = cur_class

        colors = [
            (0, 0, 0),  # Black     - Background
            (0.5, 0, 0),  # Maroon    - Blood
            (0, 0.51, 0.7843),  # Blue      - Damaged
            (0.23529, 0.70588, 0.294117),  # Green     - Muscle
            (0.5, 0.5, 0.5)  # Grey      - Undefined
        ]

        # Create an empty mask of same size as image
        seg_img = np.zeros(shape=(seg_mask_heights, seg_mask_width, 3))

        # Color the image
        segc_0 = (all_colormap_images[0] == 0)  # background
        segc_1 = (all_colormap_images[1] == 1)  # grade1
        segc_2 = (all_colormap_images[2] == 2)  # grade2
        segc_3 = (all_colormap_images[3] == 3)  # grade3
        segc_4 = (all_colormap_images[4] == 4)  # undefined

        seg_img[:, :, 0] += (segc_0 * (colors[0][0]))
        seg_img[:, :, 1] += (segc_0 * (colors[0][1]))
        seg_img[:, :, 2] += (segc_0 * (colors[0][2]))

        seg_img[:, :, 0] += (segc_1 * (colors[1][0]))
        seg_img[:, :, 1] += (segc_1 * (colors[1][1]))
        seg_img[:, :, 2] += (segc_1 * (colors[1][2]))

        seg_img[:, :, 0] += (segc_2 * (colors[2][0]))
        seg_img[:, :, 1] += (segc_2 * (colors[2][1]))
        seg_img[:, :, 2] += (segc_2 * (colors[2][2]))

        seg_img[:, :, 0] += (segc_3 * (colors[3][0]))
        seg_img[:, :, 1] += (segc_3 * (colors[3][1]))
        seg_img[:, :, 2] += (segc_3 * (colors[3][2]))

        seg_img[:, :, 0] += (segc_4 * (colors[4][0]))
        seg_img[:, :, 1] += (segc_4 * (colors[4][1]))
        seg_img[:, :, 2] += (segc_4 * (colors[4][2]))
    else:
        my_print('Error in color_mode. Stopping program.', error=True)
        exit()

    # Backup color
    # (1, 0.98, 0.7843),  # Beige
    # (0.9412, 0.196, 0.9019),  # Magenta   - Stroma
    # (0.23529, 0.70588, 0.294117),  # Green     - Muscle
    # (0.96078, 0.51, 0.18823),  # Orange    - Urothelium

    # Create legend
    legend_patch = []
    for n in range(N_CLASSES_ALL):
        legend_patch.append(mpatches.Patch(color=colors[n], label=NAME_OF_CLASSES_ALL[n]))

    # Convert to Vips and save image
    # height, width, bands = seg_img.shape
    # linear = seg_img.reshape(width * height * bands)
    # vi = pyvips.Image.new_from_memory(linear.data, width, height, bands, my_constants.dtype_to_format[str(seg_img.dtype)])
    # filename = colormap_save_folder + 'Pyvips.png'
    # vi.write_to_file(filename)

    # Create segmentation image
    filename = colormap_save_folder + 'Colormap_image_thres_' + str(threshold) + '.png'
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.imshow(seg_img)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # filename_folder = current_run_path + MODE_TISSUE_PREDICTION_FOLDER + wsi_prob_map + '/'
    # filename = colormap_save_folder + 'Colormap_image_thres_' + str(threshold) + '_' + str(random.getrandbits(6)) + '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    ax2.legend(handles=legend_patch, loc='lower left')
    filename = colormap_save_folder + 'Colormap_image_legend_thres_' + str(threshold) + '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

# endregion
