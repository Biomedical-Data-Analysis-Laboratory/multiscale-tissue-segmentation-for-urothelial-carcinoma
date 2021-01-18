# region IMPORTS
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_bool
from shutil import copy
from PIL import Image
import numpy as np
import my_functions
import my_constants
import operator
import datetime
import pyvips
import pickle
import time
import csv
import os
# endregion


def tissue_prediction(current_run_path, METADATA_FOLDER, MODEL_WEIGHT_FOLDER, N_CHANNELS, IMG_WIDTH, IMG_HEIGHT, N_WORKERS,
                      WHAT_MODEL_TO_LOAD, WHAT_MODEL_EPOCH_TO_LOAD, TISSUE_TO_BE_PREDICTED_WSI_PATH, TILE_SIZE_TISSUE,
                      wsi_index, total_no_of_wsi, PROBABILITY_FILENAME, PROBABILITY_FOLDER, UNDEFINED_CLASS_THRESHOLD,
                      PROBABILITY_IMAGES_PICKLE_FILE, USE_MULTIPROCESSING, MAX_QUEUE_SIZE, batch_size, SMALL_DATASET_DEBUG_MODE,
                      RUN_NEW_PREDICTION, ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE, TRAINING_DATA_MDTTTL_PICKLE_FILE,
                      ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE, PREDICT_WINDOW_SIZE, SAVE_SCN_OVERVIEW, SUMMARY_PREDICTIONS_CSV_FILE,
                      MODE_TISSUE_PREDICTION_FOLDER, MODEL_MAKE_HEAT_MAPS, MODEL_MAKE_COLOR_MAP, HEATMAP_THRESHOLD,
                      HEATMAP_SIGMA, HEAT_MAPS_FOLDER, override_predict_region, ENABLE_BINARY_MODE, MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE,
                      TISSUE_PREDICTED_CLASSES_DICTS_PATH, USE_XY_POSITION_FROM_BACKGROUND_MASK, SAVE_PROBABILITY_MAPS,
                      OVERWRITE_BACKGROUND_MASK, ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE, MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE,
                      COLORMAP_IMAGES_PICKLE_FILE, PICKLE_FILES_FOLDER, wsi_filename_no_extension,
                      wsi_filename_w_extension, wsi_dataset_folder, wsi_dataset_file_path, PROJECT_ID):
    # region FILE INIT
    # Start timer
    current_start_time = time.time()

    # Variable initialization
    total_summary_array = []

    # Define current_mode_summary_path
    current_mode_summary_path = current_run_path + MODE_TISSUE_PREDICTION_FOLDER
    os.makedirs(current_mode_summary_path, exist_ok=True)

    # Create current_wsi_summary_path
    current_wsi_summary_path = current_mode_summary_path + wsi_filename_no_extension + '/'
    os.makedirs(current_wsi_summary_path, exist_ok=True)

    # Create current_wsi_summary_path
    current_wsi_pickle_path = current_wsi_summary_path + PICKLE_FILES_FOLDER
    os.makedirs(current_wsi_pickle_path, exist_ok=True)

    # Create current_wsi_summary_path
    current_wsi_probability_images_path = current_wsi_summary_path + PROBABILITY_FOLDER
    os.makedirs(current_wsi_probability_images_path, exist_ok=True)

    # Define coordinates folder path
    os.makedirs(TISSUE_PREDICTED_CLASSES_DICTS_PATH, exist_ok=True)

    # Define name of classes and number of classes
    if ENABLE_BINARY_MODE is True:
        name_and_index_of_classes = my_constants.get_tissue_name_and_index_of_classes_binary_mode()
        index_of_background = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'other'][0]
    elif ENABLE_BINARY_MODE is False:
        name_and_index_of_classes = my_constants.get_tissue_name_and_index_of_classes()
        index_of_background = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'background'][0]

    N_CLASSES_ALL = len(name_and_index_of_classes)
    N_CLASSES_TRAINING = sum(1 for _, tile in name_and_index_of_classes.items() if tile['used_in_training'] == 1)
    index_of_undefined = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'undefined'][0]
    NAME_OF_CLASSES_ALL = []
    NAME_OF_CLASSES_ALL_DISPLAYNAME = []
    NAME_OF_CLASSES_TRAINING = []
    NAME_OF_CLASSES_TRAINING_DISPLAYNAME = []
    for index, value in name_and_index_of_classes.items():
        NAME_OF_CLASSES_ALL.append(value['name'])
        NAME_OF_CLASSES_ALL_DISPLAYNAME.append(value['display_name'])
        if value['used_in_training'] == 1:
            NAME_OF_CLASSES_TRAINING.append(value['name'])
            NAME_OF_CLASSES_TRAINING_DISPLAYNAME.append(value['display_name'])

    # Make an array for top row in summary CSV file containing IMAGE_NAME and name of each class
    csv_array = ['IMAGE_NAME']
    csv_array.extend(NAME_OF_CLASSES_ALL)
    csv_array.append('TIME')
    csv_array.append('PREDICT_WINDOW_SIZE')
    csv_array.append('PROJECT_ID')
    csv_array.append('TIME_COMPLETED')

    # Create a new csv file
    if not os.path.isfile(current_mode_summary_path + SUMMARY_PREDICTIONS_CSV_FILE):
        try:
            with open(current_mode_summary_path + SUMMARY_PREDICTIONS_CSV_FILE, 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(csv_array)
        except Exception as e:
            my_functions.my_print('Error writing to file', error=True)
            my_functions.my_print(e, error=True)

    # Save overview image
    fileName = '{}{}/{}#overview.jpeg'.format(current_mode_summary_path, wsi_filename_no_extension, wsi_filename_no_extension)
    if SAVE_SCN_OVERVIEW is True:
        image_25x = pyvips.Image.new_from_file(wsi_dataset_file_path, level=2).flatten().rot(1)
        remove_cols_left, remove_rows_top, new_width, new_height = my_functions.remove_white_background_v3(input_img=image_25x, PADDING=0, folder_path=wsi_dataset_folder)
        overview_img = image_25x.extract_area(remove_cols_left, remove_rows_top, new_width, new_height)
        overview_img.jpegsave(fileName, Q=100)
    # endregion

    # region FIND CORRECT MODE
    # Check if pickle files exists
    MDTTTL_exist = os.path.isfile(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE)
    MDTTTL2_exist = os.path.isfile(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE)
    CVTTTL_exist = os.path.isfile(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE)
    All_exist = MDTTTL_exist + MDTTTL2_exist + CVTTTL_exist

    # Check that only one model have been trained
    assert All_exist == 1, 'Either no models trained or more than one model trained. stopping program'

    # If ONLY MONO/DI/TRI model file exist
    if MDTTTL_exist:
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE)
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE)
    elif MDTTTL2_exist:
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE)
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE)
    # If ONLY Cross-validation model file exist
    elif CVTTTL_exist:
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE)
    # No models found, stopping program.
    else:
        my_functions.my_print('No models found, need a model to do ROI extraction. stopping program', error=True)
        exit()
    # endregion

    # region FIND CORRECT MODEL NO
    if WHAT_MODEL_TO_LOAD in ['Best', 'best']:
        # Load best model. Sort all transfer learning models by lowest validation loss, and choose best model.
        MODELS_AND_LOSS_ARRAY_SORTED = sorted(MODELS_AND_LOSS_ARRAY.items(), key=operator.itemgetter(1), reverse=False)
        MODEL_TO_USE = MODELS_AND_LOSS_ARRAY_SORTED[0][0]
    elif isinstance(WHAT_MODEL_TO_LOAD, int):
        # Load specific model
        MODEL_TO_USE = WHAT_MODEL_TO_LOAD
    else:
        my_functions.my_print('Error in WHAT_MODEL_TO_LOAD. stopping program', error=True)
        exit()
    # endregion

    # Loop through models until the model we want
    for current_model in ALL_MODEL_PARAMETERS:

        # Check if current_model is the model we want to use
        if current_model['ID'] == MODEL_TO_USE:

            # region LOAD MODEL AND DATASET
            if MDTTTL_exist or MDTTTL2_exist or CVTTTL_exist:
                # Read hyperparameters of current model
                current_model_no = current_model['ID']
                current_base_model = current_model['base_model']
                current_layer_config = current_model['layer_config']
                current_optimizer = current_model['optimizer']
                current_learning_rate = current_model['learning_rate']
                current_n_neurons1 = current_model['n_neurons1']
                current_n_neurons2 = current_model['n_neurons2']
                current_n_neurons3 = current_model['n_neurons3']
                current_dropout = current_model['dropout']
                current_freeze_base_model = current_model['freeze_base_model']
                current_base_model_pooling = current_model['base_model_pooling']
                current_scale_to_use = current_model['which_scale_to_use']
                current_model_mode = current_model['model_mode']

                # 400x region config
                if (current_scale_to_use == '400x') or (current_scale_to_use[1] == '400x') or (current_scale_to_use == ['25x', '100x', '400x']):
                    full_image = pyvips.Image.new_from_file(wsi_dataset_file_path, level=0).flatten().rot(1)
                    full_image_100 = pyvips.Image.new_from_file(wsi_dataset_file_path, level=1).flatten().rot(1)
                    full_image_25 = pyvips.Image.new_from_file(wsi_dataset_file_path, level=2).flatten().rot(1)

                    # Process SCN image
                    scn_offset_400x_x, scn_offset_400x_y, scn_width, scn_height = my_functions.remove_white_background_v3(input_img=full_image, PADDING=0, folder_path=wsi_dataset_folder)
                    scn_offset_100x_x, scn_offset_100x_y, _, _ = my_functions.remove_white_background_v3(input_img=full_image_100, PADDING=0, folder_path=wsi_dataset_folder)
                    scn_offset_25x_x, scn_offset_25x_y, _, _ = my_functions.remove_white_background_v3(input_img=full_image_25, PADDING=0, folder_path=wsi_dataset_folder)

                    scn_offset_x = scn_offset_400x_x
                    scn_offset_y = scn_offset_400x_y
                # 100x region config
                elif (current_scale_to_use == '100x') or (current_scale_to_use[1] == '100x'):

                    full_image_400 = pyvips.Image.new_from_file(wsi_dataset_file_path, level=0).flatten().rot(1)
                    full_image = pyvips.Image.new_from_file(wsi_dataset_file_path, level=1).flatten().rot(1)
                    full_image_25 = pyvips.Image.new_from_file(wsi_dataset_file_path, level=2).flatten().rot(1)

                    # Process SCN image
                    scn_offset_400x_x, scn_offset_400x_y, _, _ = my_functions.remove_white_background_v3(input_img=full_image_400, PADDING=0, folder_path=wsi_dataset_folder)
                    scn_offset_100x_x, scn_offset_100x_y, scn_width, scn_height = my_functions.remove_white_background_v3(input_img=full_image, PADDING=0, folder_path=wsi_dataset_folder)
                    scn_offset_25x_x, scn_offset_25x_y, _, _ = my_functions.remove_white_background_v3(input_img=full_image_25, PADDING=0, folder_path=wsi_dataset_folder)

                    scn_offset_x = scn_offset_100x_x
                    scn_offset_y = scn_offset_100x_y
                # 25x region config
                elif current_scale_to_use == '25x':
                    full_image_400 = pyvips.Image.new_from_file(wsi_dataset_file_path, level=0).flatten().rot(1)
                    full_image_100 = pyvips.Image.new_from_file(wsi_dataset_file_path, level=1).flatten().rot(1)
                    full_image = pyvips.Image.new_from_file(wsi_dataset_file_path, level=2).flatten().rot(1)

                    # Process SCN image
                    scn_offset_400x_x, scn_offset_400x_y, _, _ = my_functions.remove_white_background_v3(input_img=full_image_400, PADDING=0, folder_path=wsi_dataset_folder)
                    scn_offset_100x_x, scn_offset_100x_y, _, _ = my_functions.remove_white_background_v3(input_img=full_image_100, PADDING=0, folder_path=wsi_dataset_folder)
                    scn_offset_25x_x, scn_offset_25x_y, scn_width, scn_height = my_functions.remove_white_background_v3(input_img=full_image, PADDING=0, folder_path=wsi_dataset_folder)

                    scn_offset_x = scn_offset_25x_x
                    scn_offset_y = scn_offset_25x_y

                # Define new variables
                all_x_pos = []
                all_y_pos = []
                all_xy_pos_in_scn_img = []
                all_xy_pos_center_in_scn_img = []
                all_xy_pos_in_scn_set_as_background_from_mask = []

                # Process region
                if override_predict_region:

                    # 400x coordinates (7170)
                    # x_pos_min = 63500
                    # x_pos_max = 68876
                    # y_pos_min = 30000
                    # y_pos_max = 34992

                    # 400x coordinates (11638)
                    # x_pos_min = 106150
                    # x_pos_max = 111200
                    # y_pos_min = 36750
                    # y_pos_max = 42000

                    # 400x coordinates (23454)
                    x_pos_min = 72200
                    x_pos_max = 104500
                    y_pos_min = 800
                    y_pos_max = 22800

                    # 400x region config
                    if (current_scale_to_use == '400x') or (current_scale_to_use[1] == '400x') or (current_scale_to_use == ['25x', '100x', '400x']):
                        # We don't need to do anything as the coordinates are already at 400x
                        # Calculate region width/height
                        region_width = x_pos_max - x_pos_min
                        region_height = y_pos_max - y_pos_min

                        my_functions.my_print('Region coordinates: x: {}. y: {}'.format(x_pos_min, y_pos_min))
                    # 100x region config
                    elif (current_scale_to_use == '100x') or (current_scale_to_use[1] == '100x'):
                        # We need to transform the region coordinates from 400x to 100x. To do this we need to find coordinates of centre of region first.
                        # Calculate region width/height
                        region_width = x_pos_max - x_pos_min
                        region_height = y_pos_max - y_pos_min

                        # Find centre, and transform from 400x to 100x
                        region_centre_100x_x = int((x_pos_min + region_width / 2) * my_constants.Scale_between_100x_400x)
                        region_centre_100x_y = int((y_pos_min + region_height / 2) * my_constants.Scale_between_100x_400x)

                        # Calculate region width/height
                        region_width = int(region_width * my_constants.Scale_between_100x_400x)
                        region_height = int(region_height * my_constants.Scale_between_100x_400x)

                        # Go from centre of region, to top-left corner
                        x_pos_min = int(region_centre_100x_x - region_width / 2)
                        y_pos_min = int(region_centre_100x_y - region_height / 2)

                        # Calculate max position
                        x_pos_max = x_pos_min + region_width
                        y_pos_max = y_pos_min + region_height

                        my_functions.my_print('Region coordinates(100x): x: {}. y: {}'.format(x_pos_min, y_pos_min))
                    # 25x region config
                    elif current_scale_to_use == '25x':
                        # We need to transform the region coordinates from 400x to 25x. To do this we need to find coordinates of centre of region first.
                        # Calculate region width/height
                        region_width = x_pos_max - x_pos_min
                        region_height = y_pos_max - y_pos_min

                        # Find centre, and transform from 400x to 25x
                        region_centre_25x_x = int((x_pos_min + region_width / 2) * my_constants.Scale_between_25x_400x)
                        region_centre_25x_y = int((y_pos_min + region_height / 2) * my_constants.Scale_between_25x_400x)

                        # Calculate region width/height
                        region_width = int(region_width * my_constants.Scale_between_25x_400x)
                        region_height = int(region_height * my_constants.Scale_between_25x_400x)

                        # Go from centre of region, to top-left corner
                        x_pos_min = int(region_centre_25x_x - (region_width / 2))
                        y_pos_min = int(region_centre_25x_y - (region_height / 2))

                        # Calculate max position
                        x_pos_max = x_pos_min + region_width
                        y_pos_max = y_pos_min + region_height

                        my_functions.my_print('Region coordinates(25x): x: {}. y: {}'.format(x_pos_min, y_pos_min))

                    # Remove parts of the boarder on the right-hand side, and bottom part, to make sure that an n tiles fits
                    # scn_width = ((x_pos_max-x_pos_min) // TILE_SIZE) * TILE_SIZE
                    # scn_height = ((y_pos_max-y_pos_min) // TILE_SIZE) * TILE_SIZE
                    scn_width = x_pos_max - x_pos_min
                    scn_height = y_pos_max - y_pos_min

                    # Remove parts of the boarder on the right-hand side, and bottom part, to make sure that an n tiles fits
                    # region_width = (region_width // TILE_SIZE) * TILE_SIZE
                    # region_height = (region_height // TILE_SIZE) * TILE_SIZE

                    # Extract region
                    region_img = full_image.extract_area(scn_offset_x + x_pos_min,
                                                         scn_offset_y + y_pos_min,
                                                         scn_width,
                                                         scn_height)

                    # Save region
                    fileName = current_mode_summary_path + 'region.jpeg'
                    region_img.jpegsave(fileName, Q=100)

                    my_functions.my_print('Region size: {}, {}'.format(region_width, region_height))
                # Process entire WSI
                else:
                    x_pos_min = 0
                    x_pos_max = int(scn_width)
                    y_pos_min = 0
                    y_pos_max = int(scn_height)
                    my_functions.my_print('WSI size: {}, {}'.format(scn_width, scn_height))

                    # Calculate region width/height
                    region_width = x_pos_max - x_pos_min
                    region_height = y_pos_max - y_pos_min

                # Create an array of all possible x-coordinates
                for x_pos in range(x_pos_min, x_pos_max - TILE_SIZE_TISSUE, PREDICT_WINDOW_SIZE):
                    all_x_pos.append(x_pos)
                # Create an array of all possible y-coordinates
                for y_pos in range(y_pos_min, y_pos_max - TILE_SIZE_TISSUE, PREDICT_WINDOW_SIZE):
                    all_y_pos.append(y_pos)
                my_functions.my_print('Number of tiles in x- and y-direction: {}, {}'.format(len(all_x_pos), len(all_y_pos)))

                # Calculate probability map size
                prob_width = len(all_x_pos)
                prob_height = len(all_y_pos)

                # region BACKGROUND MASK
                background_mask_filename = wsi_dataset_folder.replace(wsi_filename_w_extension, "") + '/background_mask_25x.obj'
                if not os.path.exists(background_mask_filename) or OVERWRITE_BACKGROUND_MASK is True:
                    # Create a numpy background mask
                    # Shape is (height, width) e.g. like (3712, 4864)
                    # The mask only contain binary values. Example output from np.unique() = (array([0., 1.]), array([14409914,  3645254]))
                    background_mask = my_functions.create_wsi_binary_mask(wsi_path=wsi_dataset_file_path,
                                                                          TILE_SIZE=TILE_SIZE_TISSUE,
                                                                          override_predict_region=override_predict_region,
                                                                          x_pos_min=x_pos_min,
                                                                          y_pos_min=y_pos_min,
                                                                          x_pos_max=x_pos_max,
                                                                          y_pos_max=y_pos_max,
                                                                          current_scale_to_use=current_scale_to_use,
                                                                          region_width=region_width,
                                                                          region_height=region_height,
                                                                          current_wsi_dataset_folder=wsi_dataset_folder)

                    # Save mask before resize
                    fig1 = plt.figure(figsize=(10, 8))
                    ax1 = fig1.add_subplot(1, 1, 1)
                    ax1.imshow(background_mask)
                    ax1.set_title("Background mask before resize/padding")
                    filename_folder = current_run_path + MODE_TISSUE_PREDICTION_FOLDER + wsi_filename_no_extension + '/'
                    filename = filename_folder + 'Background_mask_before_resize_and_padding.png'
                    plt.savefig(filename)

                    # Save background mask
                    my_functions.pickle_save(background_mask, background_mask_filename)
                else:
                    # Restore existing background map
                    background_mask = my_functions.pickle_load(background_mask_filename)

                # Resize background mask
                # Output of np.unique after resize = (array([0.00000000e+00, 6.11126705e-12, 1.05783861e-11, ..., 1.00000000e+00, 1.00000000e+00, 1.00000000e+00]), array([94695,     3,     3, ...,     3,    48,    60]))
                # Output of np.unique after img_as_bool = (array([False,  True]), array([170163,  46485]))
                my_functions.my_print('Resizing background mask')
                background_mask_resized = img_as_bool(resize(background_mask, (prob_height, prob_width, 3)))
                # background_mask_resized_diagnostic = img_as_bool(resize(background_mask, (prob_height//2, prob_width//2, 3)))

                # Save mask after resize
                fig2 = plt.figure(figsize=(10, 8))
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                ax2 = fig2.add_subplot(1, 1, 1)
                ax2.imshow(rgb2gray(background_mask_resized), cmap='gray')
                ax2.set_title("Background mask after resize")
                filename_folder = current_run_path + MODE_TISSUE_PREDICTION_FOLDER + wsi_filename_no_extension + '/'
                filename = filename_folder + 'Background_mask_after_resize.png'
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)

                # Save mask after resize
                # fig2 = plt.figure(figsize=(10, 8))
                # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                # ax2 = fig2.add_subplot(1, 1, 1)
                # ax2.imshow(rgb2gray(background_mask_resized_diagnostic), cmap='gray')
                # ax2.set_title("Background mask after resize")
                # filename_folder = current_run_path + MODE_TISSUE_PREDICTION_FOLDER + current_wsi_folder + '/'
                # filename = filename_folder + 'Background_mask_after_resize_diagnostic.png'
                # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

                # Save background mask
                # mask_resized_filename = current_wsi_dataset_path.replace(current_wsi_filename_w_extension, "") + '/background_mask_25x_resized_{}_{}.obj'.format(TILE_SIZE_TISSUE, PREDICT_WINDOW_SIZE)
                # my_functions.pickle_save(background_mask_resized, mask_resized_filename)
                # endregion

                my_functions.my_print('Background mask before resize/padding width:{}, height:{}'.format(background_mask.shape[1], background_mask.shape[0]))
                my_functions.my_print('Background mask after resize (before padding) width:{}, height:{}'.format(background_mask_resized.shape[1], background_mask_resized.shape[0]))

                if USE_XY_POSITION_FROM_BACKGROUND_MASK is True:
                    # Create a new list with all xy-positions in current SCN image
                    for y_index, y_pos in enumerate(all_y_pos):
                        for x_index, x_pos in enumerate(all_x_pos):

                            # if background_mask_resized[int(y_pos_temp), int(x_pos_temp)][0] == 1:
                            if background_mask_resized[y_index, x_index][0] == 1:
                                # Not background. Add to list
                                # Create a new list with all xy-positions in current SCN image
                                all_xy_pos_in_scn_img.append((x_pos, y_pos))

                                # Create a new list with all xy-positions in current SCN image (center of tile)
                                all_xy_pos_center_in_scn_img.append((x_pos + (TILE_SIZE_TISSUE // 2), y_pos + (TILE_SIZE_TISSUE // 2)))
                            else:
                                all_xy_pos_in_scn_set_as_background_from_mask.append((x_pos, y_pos))

                elif USE_XY_POSITION_FROM_BACKGROUND_MASK is False:
                    # Create a new list with all xy-positions in current SCN image
                    for y_pos in all_y_pos:
                        for x_pos in all_x_pos:
                            all_xy_pos_in_scn_img.append((x_pos, y_pos))
                            # Create a new list with all xy-positions in current SCN image (center of tile)
                            all_xy_pos_center_in_scn_img.append((x_pos + (TILE_SIZE_TISSUE // 2), y_pos + (TILE_SIZE_TISSUE // 2)))

                if SMALL_DATASET_DEBUG_MODE is True:
                    all_xy_pos_in_scn_img = all_xy_pos_in_scn_img[0:64]
                    all_xy_pos_center_in_scn_img = all_xy_pos_center_in_scn_img[0:64]

                predict_dataset_size = len(all_xy_pos_in_scn_img)
                my_functions.my_print('Loaded {:,} tiles from WSI: {}.'.format(predict_dataset_size, wsi_dataset_folder))

                # Define name of the model
                if current_model_mode == 'mono':
                    current_model_to_load = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use + '/'

                    current_deep_learning_model, _ = my_functions.get_mono_scale_model(img_width=IMG_WIDTH,
                                                                                       img_height=IMG_HEIGHT,
                                                                                       n_channels=N_CHANNELS,
                                                                                       N_CLASSES=N_CLASSES_TRAINING,
                                                                                       base_model=current_base_model,
                                                                                       layer_config=current_layer_config,
                                                                                       n_neurons1=current_n_neurons1,
                                                                                       n_neurons2=current_n_neurons2,
                                                                                       n_neurons3=current_n_neurons3,
                                                                                       freeze_base_model=current_freeze_base_model,
                                                                                       base_model_pooling=current_base_model_pooling,
                                                                                       dropout=current_dropout)

                    # Create a data generator
                    predict_generator = my_functions.mode_7c_mono_coordinates_generator(all_xy_pos_in_scn_img=all_xy_pos_in_scn_img,
                                                                                        batch_size=batch_size,
                                                                                        n_classes=N_CLASSES_TRAINING,
                                                                                        shuffle=False,
                                                                                        TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                        scn_path=wsi_dataset_file_path,
                                                                                        scn_offset_x=scn_offset_x,
                                                                                        scn_offset_y=scn_offset_y,
                                                                                        which_scale_to_use=current_scale_to_use)
                elif str(current_model_mode) == 'di':
                    current_model_to_load = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use[0] + '_' + current_scale_to_use[1] + '/'

                    current_deep_learning_model, _ = my_functions.get_di_scale_model(img_width=IMG_WIDTH,
                                                                                     img_height=IMG_HEIGHT,
                                                                                     n_channels=N_CHANNELS,
                                                                                     N_CLASSES=N_CLASSES_TRAINING,
                                                                                     base_model=current_base_model,
                                                                                     layer_config=current_layer_config,
                                                                                     n_neurons1=current_n_neurons1,
                                                                                     n_neurons2=current_n_neurons2,
                                                                                     n_neurons3=current_n_neurons3,
                                                                                     freeze_base_model=current_freeze_base_model,
                                                                                     base_model_pooling=current_base_model_pooling,
                                                                                     dropout=current_dropout)

                    # Create a data generator
                    predict_generator = my_functions.mode_7c_di_coordinates_generator(all_xy_pos_in_scn_img=all_xy_pos_center_in_scn_img,
                                                                                      batch_size=batch_size,
                                                                                      n_classes=N_CLASSES_TRAINING,
                                                                                      shuffle=False,
                                                                                      TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                      scn_path=wsi_dataset_file_path,
                                                                                      scn_offset_x=scn_offset_x,
                                                                                      scn_offset_y=scn_offset_y,
                                                                                      which_scale_to_use=current_scale_to_use)
                elif current_model_mode == 'tri':
                    current_model_to_load = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_25x_100x_400x/'

                    current_deep_learning_model, _ = my_functions.get_tri_scale_model(img_width=IMG_WIDTH,
                                                                                      img_height=IMG_HEIGHT,
                                                                                      n_channels=N_CHANNELS,
                                                                                      N_CLASSES=N_CLASSES_TRAINING,
                                                                                      base_model=current_base_model,
                                                                                      layer_config=current_layer_config,
                                                                                      n_neurons1=current_n_neurons1,
                                                                                      n_neurons2=current_n_neurons2,
                                                                                      n_neurons3=current_n_neurons3,
                                                                                      freeze_base_model=current_freeze_base_model,
                                                                                      base_model_pooling=current_base_model_pooling,
                                                                                      dropout=current_dropout)

                    # Create a data generator
                    predict_generator = my_functions.mode_7c_tri_coordinates_generator(all_xy_pos_in_scn_img=all_xy_pos_center_in_scn_img,
                                                                                       batch_size=batch_size,
                                                                                       n_classes=N_CLASSES_TRAINING,
                                                                                       shuffle=False,
                                                                                       TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                       scn_path=wsi_dataset_file_path,
                                                                                       scn_offset_x=scn_offset_x,
                                                                                       scn_offset_y=scn_offset_y,
                                                                                       which_scale_to_use=current_scale_to_use)

                # Define path of pickle training data
                model_training_data_pickle_path = current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_MDTTTL_PICKLE_FILE
            # endregion

            # region LOAD WEIGHTS
            # Define path where weights are saved
            if CVTTTL_exist:
                weight_save_path_to_load = current_run_path + MODEL_WEIGHT_FOLDER + current_model_to_load + '/Fold_1/'
            else:
                weight_save_path_to_load = current_run_path + MODEL_WEIGHT_FOLDER + current_model_to_load

            # Check if there exist some weights we can load.
            if len(os.listdir(weight_save_path_to_load)) >= 1:
                # Load weights into model
                if WHAT_MODEL_EPOCH_TO_LOAD in ['Last', 'last']:
                    # Load weights from last epoch
                    all_weights = os.listdir(weight_save_path_to_load)
                    all_weights = sorted(all_weights, key=lambda a: int(a.split("_")[1].split(".")[0]))
                    weight_to_load = all_weights[-1]
                elif WHAT_MODEL_EPOCH_TO_LOAD in ['Best', 'best']:
                    # Load weights from best epoch
                    # Restore data
                    pickle_reader = open(model_training_data_pickle_path, 'rb')
                    (batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                     current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                     epoch_time_start_data) = pickle.load(pickle_reader)
                    pickle_reader.close()

                    # Save best epoch number
                    weight_to_load = 'Epoch_' + str(current_best_val_acc_epoch_data) + '.h5'
                elif isinstance(WHAT_MODEL_EPOCH_TO_LOAD, int):
                    # Load weights from a specific epoch
                    weight_to_load = 'Epoch_' + str(WHAT_MODEL_EPOCH_TO_LOAD) + '.h5'
                else:
                    my_functions.my_print('Error in WHAT_MODEL_EPOCH_TO_LOAD. stopping program', error=True)
                    exit()

                weight_filename = weight_save_path_to_load + weight_to_load
                my_functions.my_print('\tLoading weights: {}'.format(weight_filename))
                current_deep_learning_model.load_weights(weight_filename)
            else:
                my_functions.my_print('Found no weights to load in folder. stopping program', error=True)
                exit()
            # endregion

            # region OPTIMIZER, LOSS, COMPILING AND PLOTTING OF MODEL
            my_functions.my_print('\t')
            my_functions.my_print('Starting tissue prediction - {}, Freeze:{}, Layer_{}, optimizer:{}, learning rate:{}, '
                                  'batch size:{}, n_neurons1:{}, n_neurons2:{}, n_neurons3:{}, dropout:{}'.format(
                current_model_to_load, current_freeze_base_model, current_layer_config, current_optimizer,
                current_learning_rate, batch_size, current_n_neurons1, current_n_neurons2,
                current_n_neurons3, current_dropout))

            # Define optimizer
            if current_optimizer == 'adam':
                my_optimist = Adam(lr=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            elif current_optimizer == 'adamax':
                my_optimist = Adamax(lr=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            elif current_optimizer == 'nadam':
                my_optimist = Nadam(lr=current_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            elif current_optimizer == 'adadelta':
                my_optimist = Adadelta(lr=current_learning_rate, rho=0.95, epsilon=None, decay=0.0)
            elif current_optimizer == 'adagrad':
                my_optimist = Adagrad(lr=current_learning_rate, epsilon=None, decay=0.0)
            elif current_optimizer == 'SGD':
                my_optimist = SGD(lr=current_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                my_optimist = None
                my_functions.my_print('Error on choosing optimizer. stopping program', error=True)
                exit()

            # Compile the model
            # For for info on loss function, see https://github.com/keras-team/keras/blob/master/keras/losses.py
            current_deep_learning_model.compile(optimizer=my_optimist,
                                                loss='categorical_crossentropy',
                                                metrics=['accuracy'])
            # endregion

            # region RUNNING OR LOADING PREDICTIONS
            all_predictions_filename = '/All_Predictions_window{}_undefined-class-threshold{}.obj'.format(PREDICT_WINDOW_SIZE, UNDEFINED_CLASS_THRESHOLD)
            # Use model to predict WSI
            if RUN_NEW_PREDICTION:
                # Run prediction on all tiles in image
                all_predictions = current_deep_learning_model.predict_generator(generator=predict_generator,
                                                                                max_queue_size=MAX_QUEUE_SIZE,
                                                                                workers=N_WORKERS,
                                                                                use_multiprocessing=USE_MULTIPROCESSING,
                                                                                verbose=1)

                my_functions.my_print('Prediction complete.')

                # Backup all data
                my_functions.pickle_save(all_predictions, current_wsi_pickle_path + all_predictions_filename)

            elif not RUN_NEW_PREDICTION:
                # RESTORE PREDICTIONS
                if os.path.exists(current_wsi_pickle_path + all_predictions_filename):
                    all_predictions = my_functions.pickle_load(current_wsi_pickle_path + all_predictions_filename)
                else:
                    my_functions.my_print('Missing Predictions_window{}_undefined-class-threshold{}.obj file. Is RUN_NEW_PREDICTION set to FALSE? Stopping program', error=True)
                    exit()
            # endregion

    # region USE PREDICTIONS TO MAKE PROBABILITY MAPS
    # Make a new array of same size as number of classes. Loop trough and zero out each element
    count_prediction_all_classes = [0] * N_CLASSES_ALL

    # Calculate probability map size
    my_functions.my_print('Probability map width:{}, height:{}'.format(prob_width, prob_height))

    # Make a empty list to store the probability array in
    probability_image_all_classes_list = []
    colormap_image_all_classes_list = []
    for _ in range(N_CLASSES_ALL):
        probability_image_all_classes_list.append(np.zeros(shape=(prob_height, prob_width), dtype=float))
        colormap_image_all_classes_list.append(np.zeros(shape=(prob_height, prob_width), dtype=float))

    # Loop through all coordinates of current image
    for xy_pos_index, current_xy_pos in enumerate(all_xy_pos_in_scn_img):

        # Update probability arrays for current position
        for i in range(N_CLASSES_TRAINING):
            probability_image_all_classes_list[i][
                (current_xy_pos[1] - y_pos_min) // PREDICT_WINDOW_SIZE,
                (current_xy_pos[0] - x_pos_min) // PREDICT_WINDOW_SIZE
            ] = round(all_predictions[xy_pos_index][i], 3)

        # Check if the largest prediction is below the threshold. If yes, it is defined as undefined class. Add prediction to the array.
        curr_max_pred = max(all_predictions[xy_pos_index])
        if curr_max_pred < UNDEFINED_CLASS_THRESHOLD:
            probability_image_all_classes_list[index_of_undefined][
                (current_xy_pos[1] - y_pos_min) // PREDICT_WINDOW_SIZE,
                (current_xy_pos[0] - x_pos_min) // PREDICT_WINDOW_SIZE
            ] = round(1 - max(all_predictions[xy_pos_index]), 3)

            colormap_image_all_classes_list[index_of_undefined][
                (current_xy_pos[1] - y_pos_min) // PREDICT_WINDOW_SIZE,
                (current_xy_pos[0] - x_pos_min) // PREDICT_WINDOW_SIZE
            ] = 1

            # count_prediction_undefined_class += 1
            count_prediction_all_classes[index_of_undefined] += 1
        else:
            for i in range(N_CLASSES_TRAINING):
                if curr_max_pred == all_predictions[xy_pos_index][i]:
                    colormap_image_all_classes_list[i][
                        (current_xy_pos[1] - y_pos_min) // PREDICT_WINDOW_SIZE,
                        (current_xy_pos[0] - x_pos_min) // PREDICT_WINDOW_SIZE
                    ] = 1

            # If current image is same class as copy_to_folder_class, we need to copy the image to this folder
            pred_class_index = int(np.argmax(all_predictions[xy_pos_index], axis=0))

            # Update counter for current classified tile
            count_prediction_all_classes[pred_class_index] += 1

    # Save all background
    if USE_XY_POSITION_FROM_BACKGROUND_MASK is True:
        for xy_pos_index, current_xy_pos in enumerate(all_xy_pos_in_scn_set_as_background_from_mask):
            # Update probability arrays for background position
            probability_image_all_classes_list[index_of_background][
                (current_xy_pos[1] - y_pos_min) // PREDICT_WINDOW_SIZE,
                (current_xy_pos[0] - x_pos_min) // PREDICT_WINDOW_SIZE
            ] = 1

            # Update colormap for background positions
            colormap_image_all_classes_list[index_of_background][
                (current_xy_pos[1] - y_pos_min) // PREDICT_WINDOW_SIZE,
                (current_xy_pos[0] - x_pos_min) // PREDICT_WINDOW_SIZE
            ] = 1

            # Update counter for current classified tile
            count_prediction_all_classes[index_of_background] += 1
    # endregion

    # region SAVE PROBABILITY MAPS AND SUMMARY

    # Save class probability image
    if SAVE_PROBABILITY_MAPS:
        for i in range(N_CLASSES_ALL):
            img = Image.fromarray(probability_image_all_classes_list[i])
            # Resize image
            # if (scn_width // TILE_SIZE) >= 240 or (scn_height // TILE_SIZE) >= 240:
            #     img = img.resize((240, 240))
            temp_filename = current_wsi_probability_images_path + PROBABILITY_FILENAME + NAME_OF_CLASSES_ALL[i] + '.tiff'
            img.save(temp_filename)

    my_functions.pickle_save(probability_image_all_classes_list, current_wsi_pickle_path + PROBABILITY_IMAGES_PICKLE_FILE)
    my_functions.pickle_save(colormap_image_all_classes_list, current_wsi_pickle_path + COLORMAP_IMAGES_PICKLE_FILE)

    # Copy the overview image into the summary path folder, if it exist
    if os.path.isfile(TISSUE_TO_BE_PREDICTED_WSI_PATH + wsi_filename_no_extension + '/' + wsi_filename_no_extension + '#overview.jpeg'):
        src = TISSUE_TO_BE_PREDICTED_WSI_PATH + wsi_filename_no_extension + '/' + wsi_filename_no_extension + '#overview.jpeg'
        dst = current_mode_summary_path + wsi_filename_no_extension + '/'
        copy(src, dst)
    # endregion

    # region CREATE HEATMAPS AND COLORMAP
    if MODEL_MAKE_HEAT_MAPS:
        heatmap_save_folder = current_wsi_summary_path + HEAT_MAPS_FOLDER
        my_functions.make_heatmap(current_run_path=current_run_path,
                                  METADATA_FOLDER=METADATA_FOLDER,
                                  N_CLASSES=N_CLASSES_ALL,
                                  NAME_OF_CLASSES=NAME_OF_CLASSES_ALL,
                                  MODE_TISSUE_PREDICTION_FOLDER=MODE_TISSUE_PREDICTION_FOLDER,
                                  HEAT_MAPS_FOLDER=HEAT_MAPS_FOLDER,
                                  HEATMAP_SIGMA=HEATMAP_SIGMA,
                                  HEATMAP_THRESHOLD=HEATMAP_THRESHOLD,
                                  PREDICT_WINDOW_SIZE=PREDICT_WINDOW_SIZE,
                                  PROBABILITY_IMAGES_PICKLE_FILE=PROBABILITY_IMAGES_PICKLE_FILE,
                                  COLORMAP_IMAGES_PICKLE_FILE=COLORMAP_IMAGES_PICKLE_FILE,
                                  current_wsi_pickle_path=current_wsi_pickle_path,
                                  heatmap_save_folder=heatmap_save_folder)

    if MODEL_MAKE_COLOR_MAP:
        colormap_save_folder = current_wsi_summary_path
        if ENABLE_BINARY_MODE is True:
            color_mode = 'tissue_binary'
        else:
            color_mode = 'tissue_multiclass'

        my_functions.make_colormap(current_run_path=current_run_path,
                                   METADATA_FOLDER=METADATA_FOLDER,
                                   N_CLASSES_ALL=N_CLASSES_ALL,
                                   NAME_OF_CLASSES_ALL=NAME_OF_CLASSES_ALL_DISPLAYNAME,
                                   threshold=HEATMAP_THRESHOLD,
                                   HEATMAP_SIGMA=HEATMAP_SIGMA,
                                   MODE_TISSUE_PREDICTION_FOLDER=MODE_TISSUE_PREDICTION_FOLDER,
                                   color_mode=color_mode,
                                   background_mask_resized=background_mask_resized,
                                   PROBABILITY_IMAGES_PICKLE_FILE=PROBABILITY_IMAGES_PICKLE_FILE,
                                   COLORMAP_IMAGES_PICKLE_FILE=COLORMAP_IMAGES_PICKLE_FILE,
                                   current_wsi_pickle_path=current_wsi_pickle_path,
                                   colormap_save_folder=colormap_save_folder)
    # endregion

    # region FINISH

    # Calculate elapse time for current run
    elapse_time = time.time() - current_start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    model_time = '%02d:%02d:%02d' % (h, m, s)

    # Create a new array of same size as the CSV array
    summary_arr = [0] * len(csv_array)

    # The first entry in the array is the name of the current image. This was the last to be appended to the folders_used array.
    summary_arr[0] = wsi_filename_no_extension

    # Insert the number of classified tiles for each class
    for i in range(N_CLASSES_ALL):
        summary_arr[i + 1] = count_prediction_all_classes[i]

    # noinspection PyTypeChecker
    summary_arr[-4] = model_time
    summary_arr[-3] = PREDICT_WINDOW_SIZE
    summary_arr[-2] = PROJECT_ID
    # noinspection PyTypeChecker
    summary_arr[-1] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    # Append results to new list
    total_summary_array.append(summary_arr)

    # Write result to summary file
    try:
        with open(current_mode_summary_path + SUMMARY_PREDICTIONS_CSV_FILE, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(summary_arr)
    except Exception as e:
        my_functions.my_print('Error writing to file', error=True)
        my_functions.my_print(e, error=True)

    # Print report of predictions
    my_functions.my_print('')
    my_functions.my_print(csv_array)
    for item in total_summary_array:
        my_functions.my_print(item)
    my_functions.my_print('')

    # Print out results
    my_functions.my_print('Finished classifying image {}/{}: {}'.format(wsi_index + 1, total_no_of_wsi, summary_arr))
    my_functions.my_print('')
    my_functions.my_print('')
    # endregion
