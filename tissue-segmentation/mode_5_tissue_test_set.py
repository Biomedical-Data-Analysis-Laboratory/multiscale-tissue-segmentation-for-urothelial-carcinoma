from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import SGD
import my_functions
import numpy as np
import operator
import pickle
import random
import keras
import time
import csv
import sys
import os


def tissue_test_set(current_run_path, METADATA_FOLDER, MODEL_WEIGHT_FOLDER, N_CHANNELS, IMG_WIDTH, IMG_HEIGHT,
                    WHAT_MODEL_TO_LOAD, WHAT_MODEL_EPOCH_TO_LOAD,
                    CLASSIFICATION_REPORT_FOLDER, MODE_FOLDER, MISCLASSIFIED_IMAGE_FOLDER, SAVE_MISCLASSIFIED_FIGURES_TTS,
                    PROJECT_ID, DESCRIPTION, FILE_NAME, USE_MULTIPROCESSING,
                    batch_size, MAX_QUEUE_SIZE, N_WORKERS,
                    SAVE_ALL_CLASSIFIED_FIGURES_TTS, ALL_CLASSIFIED_IMAGE_FOLDER, SAVE_CONF_MAT_AND_REPORT_TTL,
                    ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE, ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE,
                    MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE, ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE,
                    MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE, TISSUE_GROUNDTRUTH_TEST_PATH, SMALL_DATASET_DEBUG_MODE,
                    SMALL_DATASET_DEBUG_N_TILES, SCN_PATH, TILE_SIZE_TISSUE, TRAINING_DATA_MDTTTL2_PICKLE_FILE,
                    TRAINING_DATA_MDTTTL_PICKLE_FILE, SUMMARY_CSV_FILE_PATH, tf_version):
    # Start timer
    current_start_time = time.time()

    TISSUE_NO_OF_CLASSES = 6
    N_CLASSES = TISSUE_NO_OF_CLASSES

    TISSUE_NAME_OF_CLASSES = ['Background', 'Blood', 'Damaged', 'Muscle', 'Stroma', 'Urothelium']
    NAME_OF_CLASSES = TISSUE_NAME_OF_CLASSES

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
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE)
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE)
    elif MDTTTL2_exist:
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE)
        MODELS_AND_LOSS_ARRAY = my_functions.pickle_load(current_run_path + METADATA_FOLDER + MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE)
    # If ONLY Cross-validation model file exist
    elif CVTTTL_exist:
        ALL_MODEL_PARAMETERS = my_functions.pickle_load(current_run_path + METADATA_FOLDER + ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE)
    # No models found, stopping program.
    else:
        my_functions.my_print('No models found, need a model to do ROI extraction. stopping program', error=True)
        exit()
    # endregion

    # Make a new array of same size as number of classes. Loop trough and zero out each element
    # count_prediction = NAME_OF_CLASSES[:]
    # for n in range(len(count_prediction)):
    # count_prediction[n] = 0

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

            # List path of all files in a list
            patient_list = os.listdir(TISSUE_GROUNDTRUTH_TEST_PATH)
            patient_list.sort()

            test_dataset_dict = dict()
            dict_tile_counter_train = dict()
            curr_class_counter_helper = []
            dict_patient_counter_train = dict()

            # Initialize counter
            for class_name in NAME_OF_CLASSES:
                dict_tile_counter_train[class_name] = 0
                dict_patient_counter_train[class_name] = 0

            # Get test data. Go through each patient and extract tiles
            for current_patient in patient_list:
                # Restore coordinate data
                list_of_all_tiles_for_curr_patient_dict = my_functions.pickle_load(TISSUE_GROUNDTRUTH_TEST_PATH + current_patient)

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

                    tempvalue = value.copy()

                    # First time a new class is checked, update patient class counter
                    if not value['label'] in curr_class_counter_helper:
                        dict_patient_counter_train[value['label']] += 1
                        curr_class_counter_helper.append(tempvalue['label'])

                    # Add tile to dict
                    tempvalue['augmentarg'] = 1
                    tempvalue['path'] = SCN_PATH + value['path'].split(".")[0] + '/' + value['path']
                    test_dataset_dict[len(test_dataset_dict)] = tempvalue
                    # Update counter
                    dict_tile_counter_train[value['label']] += 1

            # Calculate size of dataset
            test_dataset_size = len(test_dataset_dict)

            # Print output
            my_functions.my_print('Test dataset:', visible=True)
            for class_name, n_value in dict_tile_counter_train.items():
                if len(class_name) >= 7:
                    my_functions.my_print('\t{} \t- {} tiles \t- {} patients'.format(class_name, n_value, dict_patient_counter_train[class_name]), visible=True)
                else:
                    my_functions.my_print('\t{} \t\t- {} tiles \t- {} patients'.format(class_name, n_value, dict_patient_counter_train[class_name]), visible=True)
            my_functions.my_print('\tDataset size: {:,} tiles'.format(test_dataset_size), visible=True)

            # Define name of the model
            if current_model_mode == 'mono':
                current_model_name = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use + '/'

                current_deep_learning_model, current_latent_size = my_functions.get_mono_scale_model(img_width=IMG_WIDTH,
                                                                                                     img_height=IMG_HEIGHT,
                                                                                                     n_channels=N_CHANNELS,
                                                                                                     N_CLASSES=N_CLASSES,
                                                                                                     base_model=current_base_model,
                                                                                                     layer_config=current_layer_config,
                                                                                                     n_neurons1=current_n_neurons1,
                                                                                                     n_neurons2=current_n_neurons2,
                                                                                                     n_neurons3=current_n_neurons3,
                                                                                                     freeze_base_model=current_freeze_base_model,
                                                                                                     base_model_pooling=current_base_model_pooling,
                                                                                                     dropout=current_dropout)

                # Create a data generator
                test_generator = my_functions.mode_2b_mono_coordinates_generator(tile_dicts=test_dataset_dict,
                                                                                 batch_size=batch_size,
                                                                                 n_classes=N_CLASSES,
                                                                                 shuffle=True,
                                                                                 TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                 which_scale_to_use=current_scale_to_use,
                                                                                 name_of_classes=NAME_OF_CLASSES)
            elif str(current_model_mode) == 'di':
                current_model_name = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_' + current_scale_to_use[0] + '_' + current_scale_to_use[1] + '/'

                current_deep_learning_model, current_latent_size = my_functions.get_di_scale_model(img_width=IMG_WIDTH,
                                                                                                   img_height=IMG_HEIGHT,
                                                                                                   n_channels=N_CHANNELS,
                                                                                                   N_CLASSES=N_CLASSES,
                                                                                                   base_model=current_base_model,
                                                                                                   layer_config=current_layer_config,
                                                                                                   n_neurons1=current_n_neurons1,
                                                                                                   n_neurons2=current_n_neurons2,
                                                                                                   n_neurons3=current_n_neurons3,
                                                                                                   freeze_base_model=current_freeze_base_model,
                                                                                                   base_model_pooling=current_base_model_pooling,
                                                                                                   dropout=current_dropout)

                # Create a data generator
                test_generator = my_functions.mode_2b_di_coordinates_generator(tile_dicts=test_dataset_dict,
                                                                               batch_size=batch_size,
                                                                               n_classes=N_CLASSES,
                                                                               shuffle=True,
                                                                               TILE_SIZE=TILE_SIZE_TISSUE,
                                                                               which_scale_to_use=current_scale_to_use,
                                                                               name_of_classes=NAME_OF_CLASSES)
            elif current_model_mode == 'tri':
                current_model_name = 'Model_' + str(current_model_no) + '_' + current_model_mode + '_25x_100x_400x/'

                current_deep_learning_model, current_latent_size = my_functions.get_tri_scale_model(img_width=IMG_WIDTH,
                                                                                                    img_height=IMG_HEIGHT,
                                                                                                    n_channels=N_CHANNELS,
                                                                                                    N_CLASSES=N_CLASSES,
                                                                                                    base_model=current_base_model,
                                                                                                    layer_config=current_layer_config,
                                                                                                    n_neurons1=current_n_neurons1,
                                                                                                    n_neurons2=current_n_neurons2,
                                                                                                    n_neurons3=current_n_neurons3,
                                                                                                    freeze_base_model=current_freeze_base_model,
                                                                                                    base_model_pooling=current_base_model_pooling,
                                                                                                    dropout=current_dropout)

                # Create a data generator
                test_generator = my_functions.mode_2b_tri_coordinates_generator(tile_dicts=test_dataset_dict,
                                                                                batch_size=batch_size,
                                                                                n_classes=N_CLASSES,
                                                                                shuffle=True,
                                                                                TILE_SIZE=TILE_SIZE_TISSUE,
                                                                                which_scale_to_use=current_scale_to_use,
                                                                                name_of_classes=NAME_OF_CLASSES)

            # Define path of pickle training data
            if MDTTTL_exist:
                model_training_data_pickle_path = current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_MDTTTL_PICKLE_FILE
            elif MDTTTL2_exist or CVTTTL_exist:
                model_training_data_pickle_path = current_run_path + METADATA_FOLDER + 'Model_' + str(current_model_no) + TRAINING_DATA_MDTTTL2_PICKLE_FILE

            # Define summary path
            summary_path = current_run_path + MODE_FOLDER
            os.makedirs(summary_path, exist_ok=True)

            # Make folder for new model.
            path = summary_path + current_model_name
            os.makedirs(path, exist_ok=True)

            # Make a classification report folder
            path = summary_path + current_model_name + CLASSIFICATION_REPORT_FOLDER
            os.makedirs(path, exist_ok=True)

            # Calculate compression ratio
            compression = abs(round((1 - (current_latent_size / (IMG_HEIGHT * IMG_WIDTH * N_CHANNELS))) * 100, 1))

            my_functions.my_print('\t')
            my_functions.my_print('Testing model on tissue test set - {}, Freeze:{}, Layer_{}, optimizer:{}, learning rate:{}, '
                                  'batch size:{}, n_neurons1:{}, n_neurons2:{}, n_neurons3:{}, dropout:{}'.format(
                current_model_name, current_freeze_base_model, current_layer_config, current_optimizer,
                current_learning_rate, batch_size, current_n_neurons1, current_n_neurons2,
                current_n_neurons3, current_dropout))

            # region LOAD WEIGHTS
            # Define path where weights are saved
            if CVTTTL_exist:
                weight_save_path_to_load = current_run_path + MODEL_WEIGHT_FOLDER + current_model_name + '/Fold_1/'
            else:
                weight_save_path_to_load = current_run_path + MODEL_WEIGHT_FOLDER + current_model_name

            # Check if there exist some weights we can load.
            if len(os.listdir(weight_save_path_to_load)) >= 1:
                # Load weights into model
                if WHAT_MODEL_EPOCH_TO_LOAD in ['Last', 'last']:
                    # Load weights from last epoch
                    all_weights = os.listdir(weight_save_path_to_load)
                    all_weights = sorted(all_weights, key=lambda a: int(a.split("_")[1].split(".")[0]))
                    weight_to_load = all_weights[-1]
                    # noinspection PyTypeChecker
                    loaded_epoch_for_plot = weight_to_load.split(".")[0].split("_")[-1]
                elif WHAT_MODEL_EPOCH_TO_LOAD in ['Best', 'best']:
                    # Load weights from best epoch
                    # Restore data
                    pickle_reader = open(model_training_data_pickle_path, 'rb')

                    if MDTTTL_exist:
                        (batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                         current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                         epoch_time_start_data) = pickle.load(pickle_reader)
                        pickle_reader.close()
                    elif MDTTTL2_exist or CVTTTL_exist:
                        (batch_data, epochs_data, current_best_train_loss_data, current_best_train_acc_data,
                         current_best_val_loss_data, current_best_val_acc_data, current_best_val_acc_epoch_data, current_best_val_loss_epoch_data,
                         epoch_time_start_data, current_best_train_acc_epoch, current_best_train_loss_epoch) = pickle.load(pickle_reader)
                        pickle_reader.close()

                    # Save best epoch number
                    weight_to_load = 'Epoch_' + str(current_best_val_acc_epoch_data) + '.h5'
                    loaded_epoch_for_plot = current_best_val_acc_epoch_data
                elif isinstance(WHAT_MODEL_EPOCH_TO_LOAD, int):
                    # Load weights from a specific epoch
                    weight_to_load = 'Epoch_' + str(WHAT_MODEL_EPOCH_TO_LOAD) + '.h5'
                    loaded_epoch_for_plot = WHAT_MODEL_EPOCH_TO_LOAD
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

            my_functions.my_print("Starting evaluating test set")
            test_history = current_deep_learning_model.evaluate_generator(generator=test_generator,
                                                                          steps=None,
                                                                          max_queue_size=MAX_QUEUE_SIZE,
                                                                          workers=N_WORKERS,
                                                                          use_multiprocessing=USE_MULTIPROCESSING,
                                                                          verbose=1)

            current_test_loss = test_history[0]
            current_test_acc = test_history[1]
            my_functions.my_print('Test Loss: {}'.format(current_test_loss))
            my_functions.my_print('Test Accuracy: {}'.format(current_test_acc))

            # Check if we need to do more
            if SAVE_CONF_MAT_AND_REPORT_TTL is True or SAVE_ALL_CLASSIFIED_FIGURES_TTS is True or SAVE_MISCLASSIFIED_FIGURES_TTS is True:

                # Create list of labels for confusion matrix
                cm_label = list(range(N_CLASSES))

                # Make a vector with name of each class
                name_of_classes_array = []
                for index, class_name in enumerate(NAME_OF_CLASSES):
                    name_of_classes_array.append('Class ' + str(index) + ': ' + class_name)

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

                # Go through all batches of test images
                my_functions.my_print("Creating confusion matrix, report and misclassified images..")

                # Calculate number of steps
                if float(test_dataset_size / batch_size).is_integer():
                    n_steps_test = int(np.floor(test_dataset_size / batch_size))
                else:
                    n_steps_test = int(np.floor(test_dataset_size / batch_size)) + 1

                for batch_index in range(n_steps_test):

                    # If Mono-scale / Encoder-Classifier
                    if current_model_mode == 'mono':
                        # Load one batch of images and labels
                        y_images_400x, y_true_one_hot_encoded = test_generator.__getitem__(batch_index)

                        # Save classified images
                        if SAVE_ALL_CLASSIFIED_FIGURES_TTS is True:
                            all_classified_imgs_400x.extend(y_images_400x)

                        # Use model to predict images
                        y_pred_probabilities = current_deep_learning_model.predict(x=y_images_400x,
                                                                                   batch_size=None,
                                                                                   verbose=0,
                                                                                   steps=None)
                    # If Di-scale
                    elif current_model_mode == 'di':
                        # Load one batch of images and labels
                        [y_images_400x, y_images_100x], y_true_one_hot_encoded = test_generator.__getitem__(batch_index)

                        # Save classified images
                        if SAVE_ALL_CLASSIFIED_FIGURES_TTS is True:
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
                        [y_images_400x, y_images_100x, y_images_25x], y_true_one_hot_encoded = test_generator.__getitem__(batch_index)

                        # Save classified images
                        if SAVE_ALL_CLASSIFIED_FIGURES_TTS is True:
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

                # Save images with misclassification. Limited to 1000 images to avoid flooding.
                if SAVE_MISCLASSIFIED_FIGURES_TTS is True and len(misclassified_imgs_400x) >= 1:
                    if current_model_mode == 'mono':
                        my_functions.my_plot_misclassifications(current_epoch=loaded_epoch_for_plot,
                                                                image_400x=misclassified_imgs_400x[:500],
                                                                y_true=misclassified_true_class[:500],
                                                                y_pred=misclassified_prediction[:500],
                                                                N_CLASSES_TRAINING=N_CLASSES,
                                                                SUMMARY_PATH=summary_path,
                                                                model_name=current_model_name,
                                                                FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                n_channels=N_CHANNELS,
                                                                name_of_classes_array=name_of_classes_array,
                                                                prediction_type='Tissue')
                    elif current_model_mode == 'di':
                        my_functions.my_plot_di_scale_misclassifications(current_epoch=loaded_epoch_for_plot,
                                                                         image_400x=misclassified_imgs_400x[:500],
                                                                         image_100x=misclassified_imgs_100x[:500],
                                                                         y_true=misclassified_true_class[:500],
                                                                         y_pred=misclassified_prediction[:500],
                                                                         N_CLASSES_TRAINING=N_CLASSES,
                                                                         SUMMARY_PATH=summary_path,
                                                                         model_name=current_model_name,
                                                                         FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                         n_channels=N_CHANNELS,
                                                                         name_of_classes_array=name_of_classes_array,
                                                                         prediction_type='Tissue')
                    elif current_model_mode == 'tri':
                        my_functions.my_plot_tri_scale_misclassifications(current_epoch=loaded_epoch_for_plot,
                                                                          image_400x=misclassified_imgs_400x,
                                                                          image_100x=misclassified_imgs_100x,
                                                                          image_25x=misclassified_imgs_25x,
                                                                          y_true=misclassified_true_class,
                                                                          y_pred=misclassified_prediction,
                                                                          N_CLASSES_TRAINING=N_CLASSES,
                                                                          SUMMARY_PATH=summary_path,
                                                                          model_name=current_model_name,
                                                                          FIGURE_PATH=MISCLASSIFIED_IMAGE_FOLDER,
                                                                          n_channels=N_CHANNELS,
                                                                          name_of_classes_array=name_of_classes_array,
                                                                          prediction_type='Tissue')

                # Save all classified images
                if SAVE_ALL_CLASSIFIED_FIGURES_TTS is True:
                    if current_model_mode == 'mono':
                        my_functions.my_plot_misclassifications(current_epoch=loaded_epoch_for_plot,
                                                                image_400x=all_classified_imgs_400x,
                                                                y_true=y_true_class_total,
                                                                y_pred=y_pred_probabilities_total,
                                                                N_CLASSES_TRAINING=N_CLASSES,
                                                                SUMMARY_PATH=summary_path,
                                                                model_name=current_model_name,
                                                                FIGURE_PATH=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                                n_channels=N_CHANNELS,
                                                                name_of_classes_array=name_of_classes_array,
                                                                prediction_type='Tissue')
                    elif current_model_mode == 'di':
                        my_functions.my_plot_di_scale_misclassifications(current_epoch=loaded_epoch_for_plot,
                                                                         image_400x=all_classified_imgs_400x,
                                                                         image_100x=all_classified_imgs_100x,
                                                                         y_true=y_true_class_total,
                                                                         y_pred=y_pred_probabilities_total,
                                                                         N_CLASSES_TRAINING=N_CLASSES,
                                                                         SUMMARY_PATH=summary_path,
                                                                         model_name=current_model_name,
                                                                         FIGURE_PATH=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                                         n_channels=N_CHANNELS,
                                                                         name_of_classes_array=name_of_classes_array,
                                                                         prediction_type='Tissue')
                    elif current_model_mode == 'tri':
                        my_functions.my_plot_tri_scale_misclassifications(current_epoch=loaded_epoch_for_plot,
                                                                          image_400x=all_classified_imgs_400x,
                                                                          image_100x=all_classified_imgs_100x,
                                                                          image_25x=all_classified_imgs_25x,
                                                                          y_true=y_true_class_total,
                                                                          y_pred=y_pred_probabilities_total,
                                                                          N_CLASSES_TRAINING=N_CLASSES,
                                                                          SUMMARY_PATH=summary_path,
                                                                          model_name=current_model_name,
                                                                          FIGURE_PATH=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                                          n_channels=N_CHANNELS,
                                                                          name_of_classes_array=name_of_classes_array,
                                                                          prediction_type='Tissue')

                # Confusion Matrix
                cm = confusion_matrix(y_true=y_true_class_total,
                                      y_pred=y_pred_class_total,
                                      labels=cm_label,
                                      sample_weight=None)

                # Define a title
                cm_title = 'Test set - Epoch {}'.format(loaded_epoch_for_plot)

                # Save confusion matrix
                my_functions.plot_confusion_matrix(cm=cm,
                                                   epoch=loaded_epoch_for_plot,
                                                   classes=NAME_OF_CLASSES,
                                                   SUMMARY_PATH=summary_path + current_model_name,
                                                   folder_name='Confusion_matrix_test_set',
                                                   title=cm_title)

                # Compute classification report
                cr = classification_report(y_true=y_true_class_total,
                                           y_pred=y_pred_class_total,
                                           target_names=NAME_OF_CLASSES,
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

                # Save classification report to CSV
                with open(summary_path + current_model_name + CLASSIFICATION_REPORT_FOLDER + 'Test_classification_report_' + str(loaded_epoch_for_plot) + '.csv', 'w') as newFile:
                    newFileWriter = csv.writer(newFile, delimiter=';', lineterminator='\r', quoting=csv.QUOTE_MINIMAL)
                    for rows in range(len(tmp)):
                        newFileWriter.writerow(tmp[rows])

            # Calculate elapse time for current run
            elapse_time = time.time() - current_start_time
            m, s = divmod(elapse_time, 60)
            h, m = divmod(m, 60)
            model_time = '%02d:%02d:%02d' % (h, m, s)

            # Print out results
            my_functions.my_print('Finished testing model on test set. Time: {}'.format(model_time))

            # Write result to summary.csv file
            my_functions.summary_csv_file_update(SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                 PROJECT_ID=str(int(PROJECT_ID) + int(current_model_no)),
                                                 DESCRIPTION=DESCRIPTION,
                                                 FILE_NAME=FILE_NAME,
                                                 MODE='5',
                                                 model_name=current_model_name,
                                                 label='Tissue',
                                                 base_model=current_base_model,
                                                 freeze_base_model=current_freeze_base_model,
                                                 blocks_to_unfreeze_vgg16_vgg19='N/A',
                                                 delayed_unfreeze_start_epoche_vgg16_vgg19='N/A',
                                                 base_model_pooling=current_base_model_pooling,
                                                 training_samples='N/A',
                                                 validation_samples='N/A',
                                                 test_samples=test_dataset_size,
                                                 layer_config=current_layer_config,
                                                 augment_classes='N/A',
                                                 augment_multiplier='N/A',
                                                 learning_rate=current_learning_rate,
                                                 batch_size=batch_size,
                                                 n_neurons1=current_n_neurons1,
                                                 n_neurons2=current_n_neurons2,
                                                 n_neurons3=current_n_neurons3,
                                                 dropout=current_dropout,
                                                 F1_score='N/A',
                                                 F1_std='N/A',
                                                 best_train_loss='N/A',
                                                 best_train_acc='N/A',
                                                 best_val_loss=current_test_loss,
                                                 best_val_acc=current_test_acc,
                                                 best_val_loss_epoch='N/A',
                                                 best_val_acc_epoch='N/A',
                                                 trained_epoches='N/A',
                                                 total_epochs='N/A',
                                                 latent_size=current_latent_size,
                                                 compression=compression,
                                                 model_time=model_time,
                                                 optimizer=current_optimizer,
                                                 ReduceLRstatus='N/A',
                                                 n_trainable_parameters_start='N/A',
                                                 n_non_trainable_parameters_start='N/A',
                                                 n_trainable_parameters_end='N/A',
                                                 n_non_trainable_parameters_end='N/A',
                                                 python_version=sys.version.split(" ")[0],
                                                 keras_version=keras.__version__,
                                                 tf_version=tf_version,
                                                 tile_size=TILE_SIZE_TISSUE)
