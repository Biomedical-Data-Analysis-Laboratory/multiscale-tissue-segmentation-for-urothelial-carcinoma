from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
#from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.callbacks import Callback
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as k
import my_functions
import my_constants
import numpy as np
import datetime
import pickle
import random
import keras
import glob
import time
import csv
import sys
import os


def transfer_learning_model(OPTIMIZER, batch_size, current_run_path, METADATA_FOLDER, ACTIVATE_TENSORBOARD, MODEL_WEIGHT_FOLDER,
                            N_CHANNELS, IMG_WIDTH, IMG_HEIGHT, PROJECT_ID, DESCRIPTION, FILE_NAME, N_WORKERS, EPOCHES,
                            CLASSIFICATION_REPORT_FOLDER, layer_config, freeze_base_model, learning_rate, n_neurons_first_layer, n_neurons_second_layer,
                            n_neurons_third_layer, dropout, base_model, MODE_FOLDER, base_model_pooling,
                            TRAINING_DATA_PICKLE_FILE, ACTIVATE_ReduceLROnPlateau, TENSORBOARD_DATA_FOLDER, REDUCE_LR_PATIENCE, USE_MULTIPROCESSING,
                            MAX_QUEUE_SIZE, SMALL_DATASET_DEBUG_MODE, TILE_SIZE_TISSUE, which_model_mode_to_use,
                            SMALL_DATASET_DEBUG_N_TILES, TISSUE_GROUNDTRUTH_TRAINING_PATH, SCN_PATH, AUGMENTATION_MULTIPLIER,
                            CLASSES_TO_AUGMENT, EARLY_STOPPING_LOSS_PATIENCE, ALL_MODEL_PARAMETERS_PICKLE_FILE,
                            which_scale_to_use_mono, which_scale_to_use_di, MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE,
                            TRAIN_VALIDATION_SPLIT, ALL_CLASSIFIED_IMAGE_FOLDER, MISCLASSIFIED_IMAGE_FOLDER, SAVE_CONF_MAT_AND_REPORT_EVERY_N,
                            SAVE_MISCLASSIFIED_FIGURES_EVERY_N, SAVE_CONF_MAT_AND_REPORT_ON_END, SAVE_MISCLASSIFIED_FIGURES_ON_END,
                            SAVE_ALL_CLASSIFIED_FIGURES_ON_END, tf_version, SUMMARY_CSV_FILE_PATH, ENABLE_BINARY_MODE):
    # region FILE INIT
    if CLASSES_TO_AUGMENT[0] is None:
        AUGMENTATION_MULTIPLIER = [1]

    # Define summary path
    current_mode_summary_path = current_run_path + MODE_FOLDER
    os.makedirs(current_mode_summary_path, exist_ok=True)

    # Define name of classes and number of classes
    if ENABLE_BINARY_MODE is True:
        name_and_index_of_classes = my_constants.get_tissue_name_and_index_of_classes_binary_mode()
        # index_of_background = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'other'][0]
    elif ENABLE_BINARY_MODE is False:
        name_and_index_of_classes = my_constants.get_tissue_name_and_index_of_classes()
        # index_of_background = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'background'][0]

    # N_CLASSES_ALL = len(name_and_index_of_classes)
    N_CLASSES_TRAINING = sum(1 for _, tile in name_and_index_of_classes.items() if tile['used_in_training'] == 1)
    # index_of_undefined = [index for index, tile in name_and_index_of_classes.items() if tile['name'] == 'undefined'][0]
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

    # Make a vector with name of each class
    name_of_classes_array = []
    for index, class_name in enumerate(NAME_OF_CLASSES_TRAINING_DISPLAYNAME):
        name_of_classes_array.append('Class ' + str(index) + ': ' + class_name)

    # Create list of labels for confusion matrix
    cm_label = list(range(N_CLASSES_TRAINING))

    # Load or generate list of classifier models
    if os.path.isfile(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_PICKLE_FILE):
        # File exist, load parameters.
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_PICKLE_FILE)
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE)
    else:
        # File does not exist, generate new lists.
        ALL_MODEL_PARAMETERS, MODELS_AND_LOSS_ARRAY = my_functions.list_of_CVTL_models(LAYER_CONFIG=layer_config,
                                                                                       BASE_MODEL=base_model,
                                                                                       OPTIMIZER=OPTIMIZER,
                                                                                       LEARNING_RATE=learning_rate,
                                                                                       N_NEURONS_FIRST_LAYER=n_neurons_first_layer,
                                                                                       N_NEURONS_SECOND_LAYER=n_neurons_second_layer,
                                                                                       N_NEURONS_THIRD_LAYER=n_neurons_third_layer,
                                                                                       DROPOUT=dropout,
                                                                                       freeze_base_model=freeze_base_model,
                                                                                       BASE_MODEL_POOLING=base_model_pooling,
                                                                                       which_model_mode_to_use=which_model_mode_to_use,
                                                                                       which_scale_to_use_mono=which_scale_to_use_mono,
                                                                                       which_scale_to_use_di=which_scale_to_use_di,
                                                                                       augmentation_multiplier=AUGMENTATION_MULTIPLIER,
                                                                                       WHAT_LABELS_TO_USE=[0],
                                                                                       middle_layer_config=[0],
                                                                                       n_neurons_mid_first_layer=[0],
                                                                                       n_neurons_mid_second_layer=[0])

        # Save to file
        my_functions.pickle_save(ALL_MODEL_PARAMETERS, current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_PICKLE_FILE)
        my_functions.pickle_save(MODELS_AND_LOSS_ARRAY, current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE)

    # endregion

    # Loop through all models to train
    for current_model_dict in ALL_MODEL_PARAMETERS:

        # Check if model already have been trained. If value is zero, we have not trained the model, or not finished the training. Start training.
        if (current_model_dict['trained_epoches'] < EPOCHES) and (current_model_dict['early_stopping'] is 0):

            # region MODEL INIT
            # Start timer
            model_time_start_data = time.time()

            # Read hyperparameters of current model
            current_model_no = current_model_dict['ID']
            current_base_model = current_model_dict['base_model']
            current_layer_config = current_model_dict['layer_config']
            current_optimizer = current_model_dict['optimizer']
            current_learning_rate = current_model_dict['learning_rate']
            current_n_neurons1 = current_model_dict['n_neurons1']
            current_n_neurons2 = current_model_dict['n_neurons2']
            current_n_neurons3 = current_model_dict['n_neurons3']
            current_dropout = current_model_dict['dropout']
            current_freeze_base_model = current_model_dict['freeze_base_model']
            current_base_model_pooling = current_model_dict['base_model_pooling']
            current_scale_to_use = current_model_dict['which_scale_to_use']
            current_model_mode = current_model_dict['model_mode']
            current_augment_multiplier = current_model_dict['augment_multiplier']

            my_functions.my_print('')
            my_functions.my_print('Model {} of {}'.format(current_model_dict['ID'], len(ALL_MODEL_PARAMETERS) - 1))
            my_functions.my_print('\tModel_mode:{}, Scale:{}'.format(current_model_mode, current_scale_to_use))
            my_functions.my_print('\tBase_model:{}, freeze:{}, Model_pooling:{}, Layer_config:{}, optimizer:{}'.format(current_base_model, current_freeze_base_model,
                                                                                                                       current_base_model_pooling, current_layer_config,
                                                                                                                       current_optimizer))
            my_functions.my_print('\tlearning rate:{}, batch size:{}, n_neurons1:{}, n_neurons2:{}, n_neurons3:{}, dropout:{}'.format(current_learning_rate, batch_size,
                                                                                                                                      current_n_neurons1, current_n_neurons2,
                                                                                                                                      current_n_neurons3, current_dropout))

            # Define name of the model
            if current_model_mode == 'mono':
                current_model_name = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use + '/'
            elif str(current_model_mode) == 'di':
                current_model_name = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use[0] + '_' + current_scale_to_use[1] + '/'
            elif current_model_mode == 'tri':
                current_model_name = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_25x_100x_400x/'

            # Create folder to save models weight
            weight_main_save_path = current_run_path + MODEL_WEIGHT_FOLDER + current_model_name
            os.makedirs(weight_main_save_path, exist_ok=True)

            # Weight save path
            weight_save_path = current_run_path + MODEL_WEIGHT_FOLDER + current_model_name
            os.makedirs(weight_save_path, exist_ok=True)

            # Define classification report path. But do not make folder until we need it
            classification_report_path = current_mode_summary_path + current_model_name + CLASSIFICATION_REPORT_FOLDER

            # Check if we are using learning rate decay on plateau (for summary csv file)
            if ACTIVATE_ReduceLROnPlateau:
                ReduceLRstatus = str(REDUCE_LR_PATIENCE)
            else:
                ReduceLRstatus = 'False'

            # Make folder for new model.
            path = current_mode_summary_path + current_model_name
            os.makedirs(path, exist_ok=True)

            # Create a tensorboard folder
            if ACTIVATE_TENSORBOARD is True:
                tensorboard_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
                tensorboard_path = current_run_path + TENSORBOARD_DATA_FOLDER + str(tensorboard_timestamp) + '/'
                os.makedirs(tensorboard_path, exist_ok=True)

                # Make a callback
                my_tensorboard_callback = TensorBoard(log_dir=tensorboard_path,
                                                      batch_size=batch_size,
                                                      write_graph=False,
                                                      write_images=True)

            # Create a log to save epoches/accuracy/loss
            csv_logger = my_functions.create_keras_logger(current_mode_summary_path, current_model_name)
            # endregion

            # region DATASET

            # List path of all files in a list
            patient_list = os.listdir(TISSUE_GROUNDTRUTH_TRAINING_PATH)
            patient_list.sort()

            # Create a new dict to rule them all
            new_main_train_dict = dict()
            dict_tile_counter_train = dict()
            dict_patient_counter_train = dict()
            training_dict = dict()
            validation_dict = dict()
            curr_class_counter_helper = []

            # Initialize counter
            for class_name in NAME_OF_CLASSES_TRAINING_DISPLAYNAME:
                dict_tile_counter_train[class_name] = 0
                dict_patient_counter_train[class_name] = 0

            # Get training data. Go through each patient and extract tiles
            for current_patient in patient_list:
                # Restore coordinate data
                list_of_all_tiles_for_curr_patient_dict = my_functions.pickle_load(TISSUE_GROUNDTRUTH_TRAINING_PATH + current_patient)

                if SMALL_DATASET_DEBUG_MODE is True:
                    # Make training set smaller
                    list_of_all_tiles_for_curr_patient_dict_temp = list(list_of_all_tiles_for_curr_patient_dict.items())
                    random.shuffle(list_of_all_tiles_for_curr_patient_dict_temp)
                    list_of_all_tiles_for_curr_patient_dict_temp = list_of_all_tiles_for_curr_patient_dict_temp[:SMALL_DATASET_DEBUG_N_TILES]
                    list_of_all_tiles_for_curr_patient_dict.clear()
                    for index, item in enumerate(list_of_all_tiles_for_curr_patient_dict_temp):
                        list_of_all_tiles_for_curr_patient_dict[index] = item[1]

                # Add data to main dict
                curr_class_counter_helper.clear()

                for _, value in list_of_all_tiles_for_curr_patient_dict.items():
                    # Example of value:
                    # {'path': 'H10700-02 A_2013-07-01 10_45_23.scn',
                    # 'label': 'Urothelium',
                    # 'coordinates_100x': (44649, 14422),
                    # 'coordinates_400x': (178793, 57893),
                    # 'coordinates_25x': (11113, 3555)}

                    tempvalue = value.copy()

                    if ENABLE_BINARY_MODE:
                        # BINARY MODE
                        # Combine all classes except urothelium
                        if value['label'] == 'Urothelium':
                            # Current value IS urothelium
                            # Check if class belongs to one of the classes to augment. If yes, add n copies to dict.
                            if value['label'] in CLASSES_TO_AUGMENT:
                                for n in range(current_augment_multiplier):
                                    tempvalue = value.copy()
                                    tempvalue['augmentarg'] = n
                                    tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                                    new_main_train_dict[len(new_main_train_dict)] = tempvalue
                                    dict_tile_counter_train[value['label']] += 1
                            else:
                                # No augmentation, add to dict.
                                tempvalue['augmentarg'] = 1
                                tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                                new_main_train_dict[len(new_main_train_dict)] = tempvalue
                                dict_tile_counter_train[value['label']] += 1
                        else:
                            # Current value is NOT urothelium
                            # Check if class belongs to one of the classes to augment. If yes, add n copies to dict.
                            if value['label'] in CLASSES_TO_AUGMENT:
                                for n in range(current_augment_multiplier):
                                    tempvalue = value.copy()
                                    tempvalue['augmentarg'] = n
                                    tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                                    tempvalue['label'] = 'Other'
                                    new_main_train_dict[len(new_main_train_dict)] = tempvalue
                                    dict_tile_counter_train[tempvalue['label']] += 1
                            else:
                                # No augmentation, add to dict.
                                tempvalue['augmentarg'] = 1
                                tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                                tempvalue['label'] = 'Other'
                                new_main_train_dict[len(new_main_train_dict)] = tempvalue
                                dict_tile_counter_train[tempvalue['label']] += 1

                        # First time a new class is checked, update patient class counter
                        if not tempvalue['label'] in curr_class_counter_helper:
                            dict_patient_counter_train[tempvalue['label']] += 1
                            curr_class_counter_helper.append(tempvalue['label'])
                    else:
                        # MULTICLASS MODE
                        # First time a new class is checked, update patient class counter
                        if not value['label'] in curr_class_counter_helper:
                            dict_patient_counter_train[value['label']] += 1
                            curr_class_counter_helper.append(tempvalue['label'])

                        # Check if class belongs to one of the classes to augment. If yes, add n copies to dict.
                        if value['label'] in CLASSES_TO_AUGMENT:
                            for n in range(current_augment_multiplier):
                                tempvalue = value.copy()
                                tempvalue['augmentarg'] = n
                                tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                                new_main_train_dict[len(new_main_train_dict)] = tempvalue
                                dict_tile_counter_train[value['label']] += 1
                        else:
                            # No augmentation, add to dict.
                            tempvalue['augmentarg'] = 1
                            tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                            new_main_train_dict[len(new_main_train_dict)] = tempvalue
                            dict_tile_counter_train[value['label']] += 1

            # Calculate size of dataset
            main_dataset_size = len(new_main_train_dict)

            # Print output
            my_functions.my_print('Main dataset:', visible=True)
            for class_name, n_value in dict_tile_counter_train.items():
                if len(class_name) >= 7:
                    my_functions.my_print('\t{} \t- {} tiles \t- {} patients'.format(class_name, n_value, dict_patient_counter_train[class_name]), visible=True)
                else:
                    my_functions.my_print('\t{} \t\t- {} tiles \t- {} patients'.format(class_name, n_value, dict_patient_counter_train[class_name]), visible=True)
            my_functions.my_print('\tMain dataset size: {:,} tiles'.format(main_dataset_size), visible=True)

            # Get validation data
            # Split dataset into training and validation dataset
            no_of_tiles_for_validation_dataset = int(main_dataset_size * TRAIN_VALIDATION_SPLIT)

            list_of_random_keys_for_validation_dataset = random.sample(list(new_main_train_dict), no_of_tiles_for_validation_dataset)

            # Create the validation dataset dict
            new_validation_dataset = dict([(key, new_main_train_dict[key]) for key in list_of_random_keys_for_validation_dataset])

            # Remove all validation tiles from the training dict
            for key in list_of_random_keys_for_validation_dataset:
                if key in new_main_train_dict:
                    del new_main_train_dict[key]

            # We need to rename all keys so they are in order (7,42,18... -> 1,2,3...)
            for _, value in new_main_train_dict.items():
                training_dict[len(training_dict)] = value.copy()

            # We need to rename all keys so they are in order (7,42,18... -> 1,2,3...)
            for _, value in new_validation_dataset.items():
                validation_dict[len(validation_dict)] = value.copy()

            # Calculate size of dataset
            train_dataset_size = len(training_dict)
            val_dataset_size = len(validation_dict)

            # Calculate size of dataset
            my_functions.my_print('Training dataset size: {:,} tiles'.format(train_dataset_size))
            my_functions.my_print('Validation dataset size: {:,} tiles'.format(val_dataset_size))

            # Create models and data generator for dataset
            # A dict containing path, label and coordinates for each tile. The coordinated should have added the offset value.
            if current_model_mode == 'mono':
                current_deep_learning_model, current_latent_size = my_functions.get_mono_scale_model(img_width=IMG_WIDTH,
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

                train_generator = my_functions.mode_2b_mono_coordinates_generator(tile_dicts=training_dict,
                                                                                  batch_size=batch_size,
                                                                                  n_classes=N_CLASSES_TRAINING,
                                                                                  shuffle=True,
                                                                                  TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                  which_scale_to_use=current_scale_to_use,
                                                                                  name_of_classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME)

                val_generator = my_functions.mode_2b_mono_coordinates_generator(tile_dicts=validation_dict,
                                                                                batch_size=batch_size,
                                                                                n_classes=N_CLASSES_TRAINING,
                                                                                shuffle=True,
                                                                                TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                which_scale_to_use=current_scale_to_use,
                                                                                name_of_classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME)
            elif current_model_mode == 'di':
                current_deep_learning_model, current_latent_size = my_functions.get_di_scale_model(img_width=IMG_WIDTH,
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

                train_generator = my_functions.mode_2b_di_coordinates_generator(tile_dicts=training_dict,
                                                                                batch_size=batch_size,
                                                                                n_classes=N_CLASSES_TRAINING,
                                                                                shuffle=True,
                                                                                TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                which_scale_to_use=current_scale_to_use,
                                                                                name_of_classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME)

                val_generator = my_functions.mode_2b_di_coordinates_generator(tile_dicts=validation_dict,
                                                                              batch_size=batch_size,
                                                                              n_classes=N_CLASSES_TRAINING,
                                                                              shuffle=True,
                                                                              TILE_SIZE=TILE_SIZE_TISSUE,
                                                                              which_scale_to_use=current_scale_to_use,
                                                                              name_of_classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME)
            elif current_model_mode == 'tri':
                current_deep_learning_model, current_latent_size = my_functions.get_tri_scale_model(img_width=IMG_WIDTH,
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

                train_generator = my_functions.mode_2b_tri_coordinates_generator(tile_dicts=training_dict,
                                                                                 batch_size=batch_size,
                                                                                 n_classes=N_CLASSES_TRAINING,
                                                                                 shuffle=True,
                                                                                 TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                 which_scale_to_use=current_scale_to_use,
                                                                                 name_of_classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME)

                val_generator = my_functions.mode_2b_tri_coordinates_generator(tile_dicts=validation_dict,
                                                                               batch_size=batch_size,
                                                                               n_classes=N_CLASSES_TRAINING,
                                                                               shuffle=True,
                                                                               TILE_SIZE=TILE_SIZE_TISSUE,
                                                                               which_scale_to_use=current_scale_to_use,
                                                                               name_of_classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME)

            # Check if dataset is smaller than batch size. If yes, my_functions.my_print a warning.
            if train_dataset_size < batch_size:
                my_functions.my_print('WARNING. Dataset smaller than batch size. Batch size is {}'.format(batch_size))

            # Calculate compression ratio
            compression = abs(round((1 - (current_latent_size / (IMG_HEIGHT * IMG_WIDTH * N_CHANNELS))) * 100, 1))
            # endregion

            # region LOAD WEIGHTS
            # Check if there exist some weights we can load. Else, start a new model
            if len(os.listdir(weight_save_path)) >= 1:
                all_weights = os.listdir(weight_save_path)
                all_weights = sorted(all_weights, key=lambda a: int(a.split("_")[1].split(".")[0]))
                last_weight = all_weights[-1]
                weight_filename = weight_save_path + last_weight

                current_deep_learning_model.load_weights(weight_filename)
                # noinspection PyTypeChecker
                start_epoch = int(last_weight.split('_')[1].split('.')[0])

                # Restore training data
                pickle_reader = open(current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_PICKLE_FILE, 'rb')
                (batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                 current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                 epoch_time_start_data, current_best_train_acc_epoch, current_best_train_loss_epoch) = pickle.load(pickle_reader)
                pickle_reader.close()

                # Fix numbering
                epochs_data = epochs_data + 1
            elif len(os.listdir(weight_save_path)) is 0:
                # Start new model
                start_epoch = 0
                batch_data = 0
                epochs_data = 1
                current_best_train_loss_data = 50000
                current_best_train_acc_data = 0
                current_best_train_acc_epoch = 0
                current_best_train_loss_epoch = 0
                current_best_val_loss_data = 50000
                current_best_val_acc_data = 0
                current_best_val_acc_epoch_data = 0
                current_best_val_loss_epoch_data = 0
                epoch_time_start_data = time.time()

                # Backup all data
                pickle_writer = open(current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_PICKLE_FILE, 'wb')
                pickle.dump(
                    (batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                     current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                     epoch_time_start_data, current_best_train_acc_epoch, current_best_train_loss_epoch), pickle_writer)
                pickle_writer.close()
            else:
                my_functions.my_print('Error in length of weight_save_path. stopping program', error=True)
                exit()
            # endregion

            # region MODEL INIT
            # Define optimizer
            # A typical choice of momentum is between 0.5 to 0.9.
            # Nesterov momentum is a different version of the momentum method which has stronger theoretical converge guarantees for convex functions. In practice, it works slightly better than standard momentum.
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

            # Save model structure to file
            with open(current_mode_summary_path + current_model_name + '/arcitecture_report.txt', 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                current_deep_learning_model.summary(print_fn=lambda x: fh.write(x + '\n'))

            # Plot model
            # plot_model(current_deep_learning_model, to_file=summary_path + current_model_name + '/model.png', show_shapes=True, show_layer_names=True)

            # Get the number of parameters in the model
            n_trainable_parameters = int(np.sum([k.count_params(p) for p in set(current_deep_learning_model.trainable_weights)]))
            n_non_trainable_parameters = int(np.sum([k.count_params(p) for p in set(current_deep_learning_model.non_trainable_weights)]))
            # endregion

            # region CALLBACK
            my_reduce_LR_callback = ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=REDUCE_LR_PATIENCE,
                                                      verbose=1,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0)

            early_stop_val_loss_callback = EarlyStopping(monitor='val_loss',  # quantity to be monitored
                                                         min_delta=0.000001,  # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
                                                         patience=EARLY_STOPPING_LOSS_PATIENCE,  # number of epochs with no improvement after which training will be stopped.
                                                         verbose=1,
                                                         mode='auto',
                                                         baseline=None,
                                                         restore_best_weights=True)

            # Define what to do every N epochs
            class MyCallbackFunction(Callback):
                def __init__(self, model, batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                             current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data,
                             current_best_val_loss_epoch_data, epoch_time_start_data, current_best_train_acc_epoch,
                             current_best_train_loss_epoch):
                    super().__init__()
                    self.model = model
                    self.batch = batch_data
                    self.epochs = epochs_data
                    self.current_best_train_loss = current_best_train_loss_data
                    self.current_best_train_acc = current_best_train_acc_data
                    self.current_best_train_loss_epoch = current_best_train_loss_epoch
                    self.current_best_train_acc_epoch = current_best_train_acc_epoch
                    self.current_best_val_loss = current_best_val_loss_data
                    self.current_best_val_acc = current_best_val_acc_data
                    self.current_best_val_acc_epoch = current_best_val_acc_epoch_data
                    self.current_best_val_loss_epoch = current_best_val_loss_epoch_data
                    self.epoch_time_start = epoch_time_start_data

                def on_epoch_begin(self, epoch, logs=None):
                    pass

                def on_epoch_end(self, batch, logs=None):

                    # Print learning rate
                    # my_functions.my_print(K.eval(self.model.optimizer.lr))

                    # Save weights
                    weight_save_filename = weight_save_path + 'Epoch_' + str(self.epochs) + '.h5'
                    self.model.save_weights(weight_save_filename)

                    # If new best model, save the accuracy of the model
                    if logs.get('loss') < self.current_best_train_loss:
                        self.current_best_train_loss = logs.get('loss')
                        self.current_best_train_loss_epoch = self.epochs
                    if logs.get('acc') > self.current_best_train_acc:
                        self.current_best_train_acc = logs.get('acc')
                        self.current_best_train_acc_epoch = self.epochs
                    if logs.get('val_loss') < self.current_best_val_loss:
                        self.current_best_val_loss = logs.get('val_loss')
                        self.current_best_val_loss_epoch = self.epochs
                    if logs.get('val_acc') > self.current_best_val_acc:
                        self.current_best_val_acc = logs.get('val_acc')
                        self.current_best_val_acc_epoch = self.epochs

                    # Backup all data
                    pickle_writer = open(current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_PICKLE_FILE, 'wb')
                    pickle.dump(
                        (self.batch, self.epochs, self.current_best_train_loss, self.current_best_train_acc,
                         self.current_best_val_loss, self.current_best_val_acc, self.current_best_val_acc_epoch, self.current_best_val_loss_epoch,
                         self.epoch_time_start, self.current_best_train_acc_epoch, self.current_best_train_loss_epoch), pickle_writer)
                    pickle_writer.close()

                    # Delete previous model (to save HDD space)
                    for previous_epochs in range(self.epochs - 1):
                        if not (previous_epochs == self.current_best_val_acc_epoch) and not (previous_epochs == self.current_best_val_loss_epoch):
                            delete_filename = '/Epoch_{}.*'.format(previous_epochs)
                            for files in glob.glob(weight_save_path + delete_filename):
                                os.remove(files)

                    # Update number of epoches trained for current model
                    current_model_dict['trained_epoches'] = self.epochs

                    # Save to file
                    my_functions.pickle_save(ALL_MODEL_PARAMETERS, current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_PICKLE_FILE)

                    # Check if we need to do more
                    if self.epochs % SAVE_CONF_MAT_AND_REPORT_EVERY_N == 0:

                        # Go through all batches of test images
                        my_functions.my_print("Evaluating validation dataset for confusion matrix, classification report and misclassified images..")

                        # Define some lists. If already defined, they will be reset for each epoch.
                        misclassified_imgs_400x = []
                        misclassified_imgs_100x = []
                        misclassified_imgs_25x = []
                        misclassified_prediction = []
                        misclassified_true_class = []
                        y_true_class_total = []
                        y_pred_class_total = []
                        y_pred_probabilities_total = []

                        # Calculate number of steps
                        if float(val_dataset_size / batch_size).is_integer():
                            n_steps_validation = int(np.floor(val_dataset_size / batch_size))
                        else:
                            n_steps_validation = int(np.floor(val_dataset_size / batch_size)) + 1

                        # Go through all batches of validation images
                        for batch_index in range(n_steps_validation):

                            # If Mono-scale
                            if current_model_mode == 'mono':
                                # Load one batch of images and labels
                                y_images_400x, y_true_one_hot_encoded = val_generator.__getitem__(batch_index)

                                # Use model to predict images
                                y_pred_probabilities = current_deep_learning_model.predict(x=y_images_400x,
                                                                                           batch_size=None,
                                                                                           verbose=1,
                                                                                           steps=None)

                            # If Di-scale
                            elif current_model_mode == 'di':
                                # Load one batch of images and labels
                                [y_images_400x, y_images_100x], y_true_one_hot_encoded = val_generator.__getitem__(batch_index)

                                # Use model to predict images
                                y_pred_probabilities = current_deep_learning_model.predict(x=[y_images_400x, y_images_100x],
                                                                                           batch_size=None,
                                                                                           verbose=0,
                                                                                           steps=None)
                            # If Tri-scale
                            elif current_model_mode == 'tri':
                                # Load one batch of images and labels
                                [y_images_400x, y_images_100x, y_images_25x], y_true_one_hot_encoded = val_generator.__getitem__(batch_index)

                                # Use model to predict images
                                y_pred_probabilities = current_deep_learning_model.predict(x=[y_images_400x, y_images_100x, y_images_25x],
                                                                                           batch_size=None,
                                                                                           verbose=0,
                                                                                           steps=None)

                            # Convert variables
                            y_true_class_batch = np.argmax(y_true_one_hot_encoded, axis=1)
                            y_true_class_total.extend(y_true_class_batch)
                            y_pred_classified_class_batch = np.argmax(y_pred_probabilities, axis=1)
                            y_pred_class_total.extend(y_pred_classified_class_batch)
                            y_pred_probabilities_total.extend(y_pred_probabilities)

                            # Find indices of incorrect predictions
                            incorrects = np.nonzero(y_pred_classified_class_batch != y_true_class_batch)

                            # Save all incorrect images, labels and true class to new lists
                            if len(incorrects[0]) >= 1:
                                for incorrect_indices in incorrects[0]:
                                    # for incorrect_indices in range(current_batch_size):
                                    misclassified_imgs_400x.append(y_images_400x[incorrect_indices])
                                    misclassified_prediction.append(y_pred_probabilities[incorrect_indices])
                                    misclassified_true_class.append(y_true_class_batch[incorrect_indices])
                                    if current_model_mode == 'di':
                                        misclassified_imgs_100x.append(y_images_100x[incorrect_indices])
                                    elif current_model_mode == 'tri':
                                        misclassified_imgs_100x.append(y_images_100x[incorrect_indices])
                                        misclassified_imgs_25x.append(y_images_25x[incorrect_indices])

                        # Print message
                        my_functions.my_print("Saving confusion matrix, classification report and misclassified images..")

                        # Save images with misclassification. Limited to 1000 images to avoid flooding.
                        if SAVE_MISCLASSIFIED_FIGURES_EVERY_N is True and len(misclassified_imgs_400x) >= 1:
                            if current_model_mode == 'mono':
                                my_functions.my_plot_misclassifications(current_epoch=self.epochs,
                                                                        image_400x=misclassified_imgs_400x[:500],
                                                                        y_true=misclassified_true_class[:500],
                                                                        y_pred=misclassified_prediction[:500],
                                                                        N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                        SUMMARY_PATH=current_mode_summary_path,
                                                                        model_name=current_model_name,
                                                                        FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                        n_channels=N_CHANNELS,
                                                                        name_of_classes_array=name_of_classes_array,
                                                                        prediction_type='Tissue')
                            elif current_model_mode == 'di':
                                my_functions.my_plot_di_scale_misclassifications(current_epoch=self.epochs,
                                                                                 image_400x=misclassified_imgs_400x[:500],
                                                                                 image_100x=misclassified_imgs_100x[:500],
                                                                                 y_true=misclassified_true_class[:500],
                                                                                 y_pred=misclassified_prediction[:500],
                                                                                 N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                                 SUMMARY_PATH=current_mode_summary_path,
                                                                                 model_name=current_model_name,
                                                                                 FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                                 n_channels=N_CHANNELS,
                                                                                 name_of_classes_array=name_of_classes_array,
                                                                                 prediction_type='Tissue')
                            elif current_model_mode == 'tri':
                                my_functions.my_plot_tri_scale_misclassifications(current_epoch=self.epochs,
                                                                                  image_400x=misclassified_imgs_400x,
                                                                                  image_100x=misclassified_imgs_100x,
                                                                                  image_25x=misclassified_imgs_25x,
                                                                                  y_true=misclassified_true_class,
                                                                                  y_pred=misclassified_prediction,
                                                                                  N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                                  SUMMARY_PATH=current_mode_summary_path,
                                                                                  model_name=current_model_name,
                                                                                  FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                                  n_channels=N_CHANNELS,
                                                                                  name_of_classes_array=name_of_classes_array,
                                                                                  prediction_type='Tissue')

                        # region COMPUTE CONFUSION MATRIX AND CLASSIFICATION REPORT
                        # Confusion Matrix
                        cm = confusion_matrix(y_true=y_true_class_total,
                                              y_pred=y_pred_class_total,
                                              labels=cm_label,
                                              sample_weight=None)

                        # Define a title
                        cm_title = 'Validation set - Epoch {}'.format(self.epochs)

                        # Save confusion matrix
                        my_functions.plot_confusion_matrix(cm=cm,
                                                           epoch=self.epochs,
                                                           classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                           SUMMARY_PATH=current_mode_summary_path + current_model_name,
                                                           folder_name='Confusion_matrix',
                                                           title=cm_title)

                        # Compute classification report
                        cr = classification_report(y_true=y_true_class_total,
                                                   y_pred=y_pred_class_total,
                                                   target_names=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                   digits=8)

                        # Print result to console
                        my_functions.my_print(cr)

                        # Parse the classification report, so we can save it to a CSV file
                        tmp = list()
                        for row in cr.split("\n"):
                            parsed_row = [x for x in row.split(" ") if len(x) > 0]
                            if len(parsed_row) > 0:
                                tmp.append(parsed_row)

                        # Add an empty item to line up header in CSV file
                        tmp[0].insert(0, '')

                        # Make a classification report folder
                        os.makedirs(classification_report_path, exist_ok=True)

                        # Save classification report to CSV
                        with open(classification_report_path + 'Validation_classification_report_' + str(self.epochs) + '.csv', 'w') as newFile:
                            newFileWriter = csv.writer(newFile, delimiter=';', lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
                            for rows in range(len(tmp)):
                                newFileWriter.writerow(tmp[rows])
                        # endregion

                    # Raise the epoch number
                    self.epochs += 1

                def on_train_begin(self, logs=None):
                    pass

                def on_train_end(self, logs=None):

                    # Subtract one to get the correct number of epoches.
                    self.epochs -= 1

                    # Check if training was stopped by early stopping callback
                    if early_stop_val_loss_callback.stopped_epoch > 0:
                        # Update early stopping for current model
                        my_stopped_epoche = early_stop_val_loss_callback.stopped_epoch
                        current_model_dict['early_stopping'] = my_stopped_epoche + 1

                    # Update AE_MODELS_ARRAY_SORT_BY_ACC
                    MODELS_AND_LOSS_ARRAY.update({current_model_dict['ID']: self.current_best_val_loss})

                    # Calculate model time
                    model_time_elapse = time.time() - model_time_start_data
                    m, s = divmod(model_time_elapse, 60)
                    h, m = divmod(m, 60)
                    model_time = '%02d:%02d:%02d' % (h, m, s)

                    my_functions.my_print('------')
                    my_functions.my_print(model_time)
                    my_functions.my_print('------')

                    # Restore training data
                    pickle_reader = open(current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_PICKLE_FILE, 'rb')
                    (batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                     current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                     epoch_time_start_data, current_best_train_acc_epoch, current_best_train_loss_epoch) = pickle.load(pickle_reader)
                    pickle_reader.close()

                    # Write result to summary.csv file
                    my_functions.summary_csv_file_update(SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                         PROJECT_ID=str(int(PROJECT_ID) + int(current_model_no)),
                                                         DESCRIPTION=DESCRIPTION,
                                                         FILE_NAME=FILE_NAME,
                                                         MODE='2d',
                                                         model_name=current_model_name,
                                                         label='Tissue',
                                                         base_model=current_base_model,
                                                         freeze_base_model=current_freeze_base_model,
                                                         blocks_to_unfreeze_vgg16_vgg19='N/A',
                                                         delayed_unfreeze_start_epoche_vgg16_vgg19='N/A',
                                                         base_model_pooling=current_base_model_pooling,
                                                         training_samples=train_dataset_size,
                                                         validation_samples=val_dataset_size,
                                                         test_samples='N/A',
                                                         layer_config=current_layer_config,
                                                         augment_classes=CLASSES_TO_AUGMENT,
                                                         augment_multiplier=current_augment_multiplier,
                                                         learning_rate=current_learning_rate,
                                                         batch_size=batch_size,
                                                         n_neurons1=current_n_neurons1,
                                                         n_neurons2=current_n_neurons2,
                                                         n_neurons3=current_n_neurons3,
                                                         dropout=current_dropout,
                                                         F1_score='N/A',
                                                         F1_std='N/A',
                                                         best_train_loss=current_best_train_loss_data,
                                                         best_train_acc=current_best_train_acc_data,
                                                         best_val_loss=current_best_val_loss_data,
                                                         best_val_acc=current_best_val_acc_data,
                                                         best_val_loss_epoch=current_best_val_loss_epoch_data,
                                                         best_val_acc_epoch=current_best_val_acc_epoch_data,
                                                         trained_epoches=epochs_data,
                                                         total_epochs=EPOCHES,
                                                         latent_size=current_latent_size,
                                                         compression=compression,
                                                         model_time=model_time,
                                                         optimizer=current_optimizer,
                                                         ReduceLRstatus=ReduceLRstatus,
                                                         n_trainable_parameters_start=n_trainable_parameters,
                                                         n_non_trainable_parameters_start=n_non_trainable_parameters,
                                                         n_trainable_parameters_end='N/A',
                                                         n_non_trainable_parameters_end='N/A',
                                                         python_version=sys.version.split(" ")[0],
                                                         keras_version=keras.__version__,
                                                         tf_version=tf_version,
                                                         tile_size=TILE_SIZE_TISSUE)

                    # endregion

                    # Update trained flag for current model
                    current_model_dict['model_trained_flag'] = 1

                    # Save to file
                    my_functions.pickle_save(MODELS_AND_LOSS_ARRAY, current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE)
                    my_functions.pickle_save(ALL_MODEL_PARAMETERS, current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_PICKLE_FILE)

                    # Check if we need to do more
                    if (SAVE_CONF_MAT_AND_REPORT_ON_END is True) or (SAVE_MISCLASSIFIED_FIGURES_ON_END is True):

                        # Go through all batches of test images
                        my_functions.my_print("Evaluating validation dataset for confusion matrix, classification report and misclassified images..")

                        # Define some lists. If already defined, they will be reset for each epoch.
                        misclassified_imgs_400x = []
                        misclassified_imgs_100x = []
                        misclassified_imgs_25x = []
                        misclassified_prediction = []
                        misclassified_true_class = []
                        y_true_class_total = []
                        y_pred_class_total = []
                        y_pred_probabilities_total = []
                        all_classified_imgs_400x = []
                        all_classified_imgs_100x = []
                        all_classified_imgs_25x = []

                        # Calculate number of steps
                        if float(val_dataset_size / batch_size).is_integer():
                            n_steps_validation = int(np.floor(val_dataset_size / batch_size))
                        else:
                            n_steps_validation = int(np.floor(val_dataset_size / batch_size)) + 1

                        # Go through all batches of validation images
                        for batch_index in range(n_steps_validation):

                            # If Mono-scale
                            if current_model_mode == 'mono':
                                # Load one batch of images and labels
                                y_images_400x, y_true_one_hot_encoded = val_generator.__getitem__(batch_index)

                                # Save classified images
                                if SAVE_ALL_CLASSIFIED_FIGURES_ON_END is True:
                                    all_classified_imgs_400x.extend(y_images_400x)

                                # Use model to predict images
                                y_pred_probabilities = current_deep_learning_model.predict(x=y_images_400x,
                                                                                           batch_size=None,
                                                                                           verbose=0,
                                                                                           steps=None)
                            # If Di-scale
                            elif current_model_mode == 'di':
                                # Load one batch of images and labels
                                [y_images_400x, y_images_100x], y_true_one_hot_encoded = val_generator.__getitem__(batch_index)

                                # Save classified images
                                if SAVE_ALL_CLASSIFIED_FIGURES_ON_END is True:
                                    all_classified_imgs_400x.extend(y_images_400x)
                                    all_classified_imgs_100x.extend(y_images_100x)

                                # Use model to predict images
                                y_pred_probabilities = current_deep_learning_model.predict(x=[y_images_400x, y_images_100x],
                                                                                           batch_size=None,
                                                                                           verbose=0,
                                                                                           steps=None)
                            # If Tri-scale
                            elif current_model_mode == 'tri':
                                # Load one batch of images and labels
                                [y_images_400x, y_images_100x, y_images_25x], y_true_one_hot_encoded = val_generator.__getitem__(batch_index)

                                # Save classified images
                                if SAVE_ALL_CLASSIFIED_FIGURES_ON_END is True:
                                    all_classified_imgs_400x.extend(y_images_400x)
                                    all_classified_imgs_100x.extend(y_images_100x)
                                    all_classified_imgs_25x.extend(y_images_25x)

                                # Use model to predict images
                                y_pred_probabilities = current_deep_learning_model.predict(x=[y_images_400x, y_images_100x, y_images_25x],
                                                                                           batch_size=None,
                                                                                           verbose=0,
                                                                                           steps=None)

                            # Convert variables
                            y_true_class_batch = np.argmax(y_true_one_hot_encoded, axis=1)
                            y_true_class_total.extend(y_true_class_batch)
                            y_pred_classified_class_batch = np.argmax(y_pred_probabilities, axis=1)
                            y_pred_class_total.extend(y_pred_classified_class_batch)
                            y_pred_probabilities_total.extend(y_pred_probabilities)

                            # Find indices of incorrect predictions
                            incorrects = np.nonzero(y_pred_classified_class_batch != y_true_class_batch)

                            # Save all incorrect images, labels and true class to new lists
                            if len(incorrects[0]) >= 1:
                                for incorrect_indices in incorrects[0]:
                                    # for incorrect_indices in range(current_batch_size):
                                    misclassified_imgs_400x.append(y_images_400x[incorrect_indices])
                                    misclassified_prediction.append(y_pred_probabilities[incorrect_indices])
                                    misclassified_true_class.append(y_true_class_batch[incorrect_indices])
                                    if current_model_mode == 'di':
                                        misclassified_imgs_100x.append(y_images_100x[incorrect_indices])
                                    elif current_model_mode == 'tri':
                                        misclassified_imgs_100x.append(y_images_100x[incorrect_indices])
                                        misclassified_imgs_25x.append(y_images_25x[incorrect_indices])

                        # Print message
                        my_functions.my_print("Saving confusion matrix, classification report and misclassified images..")

                        # Save images with misclassification. Limited to 1000 images to avoid flooding.
                        if SAVE_MISCLASSIFIED_FIGURES_ON_END is True and len(misclassified_imgs_400x) >= 1:
                            if current_model_mode == 'mono':
                                my_functions.my_plot_misclassifications(current_epoch=self.epochs,
                                                                        image_400x=misclassified_imgs_400x[:500],
                                                                        y_true=misclassified_true_class[:500],
                                                                        y_pred=misclassified_prediction[:500],
                                                                        N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                        SUMMARY_PATH=current_mode_summary_path,
                                                                        model_name=current_model_name,
                                                                        FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                        n_channels=N_CHANNELS,
                                                                        name_of_classes_array=name_of_classes_array,
                                                                        prediction_type='Tissue')
                            elif current_model_mode == 'di':
                                my_functions.my_plot_di_scale_misclassifications(current_epoch=self.epochs,
                                                                                 image_400x=misclassified_imgs_400x[:500],
                                                                                 image_100x=misclassified_imgs_100x[:500],
                                                                                 y_true=misclassified_true_class[:500],
                                                                                 y_pred=misclassified_prediction[:500],
                                                                                 N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                                 SUMMARY_PATH=current_mode_summary_path,
                                                                                 model_name=current_model_name,
                                                                                 FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                                 n_channels=N_CHANNELS,
                                                                                 name_of_classes_array=name_of_classes_array,
                                                                                 prediction_type='Tissue')
                            elif current_model_mode == 'tri':
                                my_functions.my_plot_tri_scale_misclassifications(current_epoch=self.epochs,
                                                                                  image_400x=misclassified_imgs_400x,
                                                                                  image_100x=misclassified_imgs_100x,
                                                                                  image_25x=misclassified_imgs_25x,
                                                                                  y_true=misclassified_true_class,
                                                                                  y_pred=misclassified_prediction,
                                                                                  N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                                  SUMMARY_PATH=current_mode_summary_path,
                                                                                  model_name=current_model_name,
                                                                                  FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                                  n_channels=N_CHANNELS,
                                                                                  name_of_classes_array=name_of_classes_array,
                                                                                  prediction_type='Tissue')

                        # region COMPUTE CONFUSION MATRIX AND CLASSIFICATION REPORT
                        # Confusion Matrix
                        cm = confusion_matrix(y_true=y_true_class_total,
                                              y_pred=y_pred_class_total,
                                              labels=cm_label,
                                              sample_weight=None)

                        # Define a title
                        cm_title = 'Validation set - Epoch {}'.format(self.epochs)

                        # Save confusion matrix
                        my_functions.plot_confusion_matrix(cm=cm,
                                                           epoch=self.epochs,
                                                           classes=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                           SUMMARY_PATH=current_mode_summary_path + current_model_name,
                                                           folder_name='Confusion_matrix',
                                                           title=cm_title)

                        # Compute classification report
                        cr = classification_report(y_true=y_true_class_total,
                                                   y_pred=y_pred_class_total,
                                                   target_names=NAME_OF_CLASSES_TRAINING_DISPLAYNAME,
                                                   digits=8)

                        # Print result to console
                        my_functions.my_print(cr)

                        # Parse the classification report, so we can save it to a CSV file
                        tmp = list()
                        for row in cr.split("\n"):
                            parsed_row = [x for x in row.split(" ") if len(x) > 0]
                            if len(parsed_row) > 0:
                                tmp.append(parsed_row)

                        # Add an empty item to line up header in CSV file
                        tmp[0].insert(0, '')

                        # Make a classification report folder
                        os.makedirs(classification_report_path, exist_ok=True)

                        # Save classification report to CSV
                        with open(classification_report_path + 'Validation_classification_report_' + str(self.epochs) + '.csv', 'w') as newFile:
                            newFileWriter = csv.writer(newFile, delimiter=';', lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
                            for rows in range(len(tmp)):
                                newFileWriter.writerow(tmp[rows])
                        # endregion

                        # Save all classified images
                        if SAVE_ALL_CLASSIFIED_FIGURES_ON_END is True:
                            my_functions.my_print('Saving ALL classified images..')
                            if current_model_mode == 'mono':
                                my_functions.my_plot_misclassifications(current_epoch=self.epochs,
                                                                        image_400x=all_classified_imgs_400x,
                                                                        y_true=y_true_class_total,
                                                                        y_pred=y_pred_probabilities_total,
                                                                        N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                        SUMMARY_PATH=current_mode_summary_path,
                                                                        model_name=current_model_name,
                                                                        FIGURE_PATH=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                                        n_channels=N_CHANNELS,
                                                                        name_of_classes_array=name_of_classes_array,
                                                                        prediction_type='Tissue')
                            elif current_model_mode == 'di':
                                my_functions.my_plot_di_scale_misclassifications(current_epoch=self.epochs,
                                                                                 image_400x=all_classified_imgs_400x,
                                                                                 image_100x=all_classified_imgs_100x,
                                                                                 y_true=y_true_class_total,
                                                                                 y_pred=y_pred_probabilities_total,
                                                                                 N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                                 SUMMARY_PATH=current_mode_summary_path,
                                                                                 model_name=current_model_name,
                                                                                 FIGURE_PATH=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                                                 n_channels=N_CHANNELS,
                                                                                 name_of_classes_array=name_of_classes_array,
                                                                                 prediction_type='Tissue')
                            elif current_model_mode == 'tri':
                                my_functions.my_plot_tri_scale_misclassifications(current_epoch=self.epochs,
                                                                                  image_400x=all_classified_imgs_400x,
                                                                                  image_100x=all_classified_imgs_100x,
                                                                                  image_25x=all_classified_imgs_25x,
                                                                                  y_true=y_true_class_total,
                                                                                  y_pred=y_pred_probabilities_total,
                                                                                  N_CLASSES_TRAINING=N_CLASSES_TRAINING,
                                                                                  SUMMARY_PATH=current_mode_summary_path,
                                                                                  model_name=current_model_name,
                                                                                  FIGURE_PATH=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                                                  n_channels=N_CHANNELS,
                                                                                  name_of_classes_array=name_of_classes_array,
                                                                                  prediction_type='Tissue')

                def on_batch_begin(self, batch, logs=None):
                    pass

                def on_batch_end(self, batch, logs=None):
                    pass

            # Define the callback function array
            main_callback = MyCallbackFunction(model=current_deep_learning_model,
                                               batch_data=batch_data,
                                               epochs_data=epochs_data,
                                               current_best_train_loss_data=current_best_train_loss_data,
                                               current_best_train_acc_data=current_best_train_acc_data,
                                               current_best_val_loss_data=current_best_val_loss_data,
                                               current_best_val_acc_data=current_best_val_acc_data,
                                               current_best_val_acc_epoch_data=current_best_val_acc_epoch_data,
                                               current_best_val_loss_epoch_data=current_best_val_loss_epoch_data,
                                               epoch_time_start_data=epoch_time_start_data,
                                               current_best_train_loss_epoch=current_best_train_loss_epoch,
                                               current_best_train_acc_epoch=current_best_train_acc_epoch)

            callback_array = [csv_logger, main_callback, early_stop_val_loss_callback]

            if ACTIVATE_TENSORBOARD is True:
                callback_array.append(my_tensorboard_callback)

            if ACTIVATE_ReduceLROnPlateau is True:
                callback_array.append(my_reduce_LR_callback)
            # endregion

            # region RUN MODEL
            history_TL_obj = current_deep_learning_model.fit_generator(generator=train_generator,
                                                                       steps_per_epoch=None,
                                                                       epochs=EPOCHES,
                                                                       verbose=1,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                                                                       callbacks=callback_array,
                                                                       validation_data=val_generator,
                                                                       validation_steps=None,
                                                                       class_weight=None,
                                                                       max_queue_size=MAX_QUEUE_SIZE,
                                                                       workers=N_WORKERS,
                                                                       use_multiprocessing=USE_MULTIPROCESSING,
                                                                       shuffle=True,
                                                                       initial_epoch=start_epoch)

            # Save history
            my_functions.save_history_plot(history=history_TL_obj,
                                           path=current_mode_summary_path + current_model_name,
                                           mode='Tissue',
                                           model_no=str(current_model_no))
            # endregion
