import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DESCRIPTION = 'Debugging'
PROJECT_ID = '0'

# region IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = True
import mode_2b_tissue_mono_di_tri
import mode_2c_tissue_cross_validation
import mode_2d_tissue_mono_di_tri
import mode_5_tissue_test_set
import mode_7c_tissue_prediction_coordinates
import my_functions
import platform
import sys


# endregion
if __name__ == '__main__':

    # region MODE 2b - TISSUE TRANSFER LEARNING - MONO/DI/TRI - ONLY TRAINING SET - MDTTTL
    MODE_TRANSFER_LEARNING_MDTTTL = True
    """ HYPERPARAMETERS """
    EPOCHES_MDTTTL = 1
    base_model_MDTTTL = ['VGG16']  # Base model. Supported models: VGG16, VGG19, Xception, Resnet50, InceptionV3, InceptionResNetV2, DenseNet, NASNetLarge, NASNetMobile, MobileNetV2, MobileNet
    base_model_pooling_MDTTTL = ['avg']  # Optional pooling mode for feature extraction. Must be one of: 'None', 'avg' or 'max'
    classifier_layer_config_MDTTTL = ['config2_drop']  # Classifier architecture. Available: 'config1', 'config2', 'config3', 'config4'.
    learning_rate_MDTTTL = [0.00015]  # Learning rate. Float.
    n_neurons_first_layer_MDTTTL = [4096]  # Number of neurons in first layer. Integer.
    n_neurons_second_layer_MDTTTL = [4096]  # Number of neurons in second layer. Integer.
    n_neurons_third_layer_MDTTTL = [0]  # Number of neurons in third layer. Integer.
    dropout_MDTTTL = [0.1]  # Dropout rate. float between 0-0.99. Setting to 0 or 1 will ignore the dropout layer.
    freeze_base_model_MDTTTL = [True]  # Freeze encoder during training. Boolean.
    REDUCE_LR_PATIENCE_MDTTTL = 200  # If loss does not decrease for this many epoches, lower the learning rate
    EARLY_STOPPING_LOSS_PATIENCE_MDTTTL = 200
    which_model_mode_to_use_MDTTTL = ['mono']  # Available: mono, di and tri.
    which_scale_to_use_mono_MDTTTL = ['400x']  # Which magnification scale to load images from. Can be [400x, 100x, 25x]
    which_scale_to_use_di_MDTTTL = [['100x', '400x']]  # Which magnification scale to load images from. [['100x', '400x'], ['25x', '400x'], ['25x', '100x']]
    """ DATASET """
    SMALL_DATASET_DEBUG_MODE_MDTTTL = True  # Load a small fraction of the dataset for testing only
    SMALL_DATASET_DEBUG_N_TILES_MDTTTL = 128  # How many tiles to extract for each class in each patient. Integer.
    AUGMENTATION_MULTIPLIER_MDTTTL = [2]  # How many times to augment (rotate/flip) each tile in the classes belonging to CLASSES_TO_AUGMENT_CVTTL
    CLASSES_TO_AUGMENT_MDTTTL = ['Stroma', 'Muscle']  # List of classes to augment. To disable augmentation, put None in the list.
    """ OTHERS """
    ENABLE_BINARY_MODE_MDTTL = False  # Set to True to train a binary model. False will train a multiclass model.
    # endregion

    # region MODE 2c - TISSUE TRANSFER LEARNING - MONO/DI/TRI - CROSS VALIDATION - CVTTL
    MODE_TRANSFER_LEARNING_CVTTL = False
    """ HYPERPARAMETERS """
    EPOCHES_CVTTL = 2
    base_model_CVTTL = ['VGG16']  # Base model. Supported models: VGG16, VGG19, Xception, Resnet50, InceptionV3, InceptionResNetV2, DenseNet, NASNetLarge, NASNetMobile, MobileNetV2, MobileNet
    base_model_pooling_CVTTL = ['avg']  # Optional pooling mode for feature extraction. Must be one of: 'None', 'avg' or 'max'
    classifier_layer_config_CVTTL = ['config2_drop']  # Classifier architecture. Available: 'config1', 'config2', 'config3', 'config4'.
    learning_rate_CVTTL = [0.00015]  # Learning rate. Float.
    n_neurons_first_layer_CVTTL = [4096]  # Number of neurons in first layer. Integer.
    n_neurons_second_layer_CVTTL = [4096]  # Number of neurons in second layer. Integer.
    n_neurons_third_layer_CVTTL = [0]  # Number of neurons in third layer. Integer.
    dropout_CVTTL = [0.3]  # Dropout rate. float between 0-0.99. Setting to 0 or 1 will ignore the dropout layer.
    freeze_base_model_CVTTL = [True]  # Freeze encoder during training. Boolean.
    REDUCE_LR_PATIENCE_CVTTL = 500  # If validation loss does not decrease for this many epoches, lower the learning rate
    EARLY_STOPPING_LOSS_PATIENCE_CVTTL = 10
    which_model_mode_to_use_CVTTL = ['mono']  # Available: mono, di and tri.
    which_scale_to_use_mono_CVTTL = ['25x']  # Which magnification scale to load images from. Can be [400x, 100x, 25x]
    which_scale_to_use_di_CVTTL = [['25x', '100x']]  # Which magnification scale to load images from. [['100x', '400x'], ['25x', '400x'], ['25x', '100x']]
    """ SETTINGS """
    TRAINING_OR_TESTING_MODE_CVTTL = 'train'  # train=training mode. 'test'=test mode
    SAVE_MISCLASSIFIED_FIGURES_CVTTL = False  # Plot an image of images and models predictions for misclassified samples. Boolean.
    N_FOLDS_CVTTL = 3  # How many folds to use in cross validation. Integer.
    """ DATASET """
    SMALL_DATASET_DEBUG_MODE_CVTTL = True  # Load a small fraction of the dataset for testing only
    SMALL_DATASET_DEBUG_N_TILES_CVTTL = 128  # How many tiles to extract for each class in each patient. Integer.
    AUGMENTATION_MULTIPLIER_CVTTL = [2]  # How many times to augment (rotate/flip) each tile in the classes belonging to CLASSES_TO_AUGMENT_CVTTL
    CLASSES_TO_AUGMENT_CVTTL = ['Stroma', 'Muscle']  # List of classes to augment. To disable augmentation, put None in the list.
    """ OTHERS """
    ENABLE_BINARY_MODE_CVTTL = False  # Set to True to train a binary model. False will train a multiclass model.
    # endregion

    # region MODE 2d - TISSUE TRANSFER LEARNING - MONO/DI/TRI - TRAIN/VAL - MDTTTL2
    MODE_TRANSFER_LEARNING_MDTTTL2 = False
    """ HYPERPARAMETERS """
    which_model_mode_to_use_MDTTTL2 = ['mono']  # Available: mono, di and tri.
    which_scale_to_use_mono_MDTTTL2 = ['100x']  # Which magnification scale to load images from. Can be [400x, 100x, 25x]
    which_scale_to_use_di_MDTTTL2 = [['25x', '100x']]  # Which magnification scale to load images from. [['100x', '400x'], ['25x', '400x'], ['25x', '100x']]
    base_model_MDTTTL2 = ['VGG16']  # Base model. Supported models: VGG16, VGG19, Xception, Resnet50, InceptionV3, InceptionResNetV2, DenseNet, NASNetLarge, NASNetMobile, MobileNetV2, MobileNet
    base_model_pooling_MDTTTL2 = ['avg']  # Optional pooling mode for feature extraction. Must be one of: 'None', 'avg' or 'max'
    classifier_layer_config_MDTTTL2 = ['config2_drop']  # Classifier architecture. Available: 'config1', 'config2', 'config3', 'config4'.
    EPOCHES_MDTTTL2 = 2  # How many epoces to train. Integer.
    learning_rate_MDTTTL2 = [0.00015]  # Learning rate. Float.
    n_neurons_first_layer_MDTTTL2 = [4096]  # Number of neurons in first layer. Integer.
    n_neurons_second_layer_MDTTTL2 = [4096]  # Number of neurons in second layer. Integer.
    n_neurons_third_layer_MDTTTL2 = [0]  # Number of neurons in third layer. Integer.
    dropout_MDTTTL2 = [0.2]  # Dropout rate. float between 0-0.99. Setting to 0 or 1 will ignore the dropout layer.
    freeze_base_model_MDTTTL2 = [True]  # Freeze encoder during training. Boolean.
    REDUCE_LR_PATIENCE_MDTTTL2 = 500  # If loss does not decrease for this many epoches, lower the learning rate
    EARLY_STOPPING_LOSS_PATIENCE_MDTTTL2 = 500
    """ SETTINGS """
    SAVE_CONF_MAT_AND_REPORT_EVERY_N_MDTTL2 = 200  # How often to save confusion matrix and classification report. Set a number larger than number of epoches to disable function. Integear.
    SAVE_MISCLASSIFIED_FIGURES_EVERY_N_MDTTTL2 = False  # When saving conf.matrix and class.report, also save all misclassified images? Boolean.
    SAVE_CONF_MAT_AND_REPORT_ON_END_MDTTL2 = False  # Save conf.matrix and classification report after training is done? Boolean
    SAVE_MISCLASSIFIED_FIGURES_ON_END_MDTTTL2 = False  # Also save misclassified imaged after training? Boolean
    SAVE_ALL_CLASSIFIED_FIGURES_ON_END_MDTTL2 = False  # Also save ALL classified images after training? Boolean
    """ DATASET """
    SMALL_DATASET_DEBUG_MODE_MDTTTL2 = True  # Load a small fraction of the dataset for testing only
    SMALL_DATASET_DEBUG_N_TILES_MDTTTL2 = 128  # How many tiles to extract for each class in each patient. Integer.
    AUGMENTATION_MULTIPLIER_MDTTTL2 = [4]  # How many times to augment (rotate/flip) each tile in the classes belonging to CLASSES_TO_AUGMENT_CVTTL
    CLASSES_TO_AUGMENT_MDTTTL2 = ['Stroma', 'Muscle']  # List of classes to augment. To disable augmentation, put None in the list.
    TRAIN_VALIDATION_SPLIT_MDTTL2 = 0.15  # How large percentage of available data is split into validation data (e.g. 0.1 = 10% of data is validation data). Float between 0 and 1.
    """ OTHERS """
    ENABLE_BINARY_MODE_MDTTL2 = False  # Set to True to train a binary model. False will train a multiclass model.
    # endregion

    # region MODE 5 - TISSUE TEST SET - TTS
    MODEL_TISSUE_TEST_SET = False
    WHAT_MODEL_TO_LOAD_TTS = 'Best'  # If several models have been trained, which one to use. 'Best'=best model. or an integer to specify model number, e.g. 3 would load model 3.
    WHAT_MODEL_EPOCH_TO_LOAD_TTS = 'last'  # What epoch to load weights from. 'last'=last epoch, 'Best'=best epoch, or an integer to specify epoch, e.g. 120 would load weights from epoch 120.
    SAVE_MISCLASSIFIED_FIGURES_TTS = False  # Plot an image of images and models predictions for misclassified samples. Boolean.
    SAVE_ALL_CLASSIFIED_FIGURES_TTS = False  # SLOW - Plot an image of images and models predictions for ALL classified samples. Boolean.
    SAVE_CONF_MAT_AND_REPORT_TTS = True  # Takes extra time. Save confusion matrix and classification report
    SMALL_DATASET_DEBUG_MODE_TEST = True  # Load a small fraction of the dataset for debugging
    SMALL_DATASET_DEBUG_N_TILES_TEST = 128  # How many tiles to extract for each class in each patient. Integer.
    # endregion

    # region MODE 7c - TISSUE PREDICTION (USE ONE MODEL ON DATA) - SCN Images - TPSCN
    MODEL_TISSUE_PREDICTION_TPSCN = False
    """ MODELS """
    WHAT_MODEL_TO_LOAD_TPSCN = 'Best'  # If several models have been trained, which one to use. 'Best'=best model. or an integer to specify model number, e.g. 3 would load model 3.
    WHAT_MODEL_EPOCH_TO_LOAD_TPSCN = 'last'  # What epoch to load weights from. 'last'=last epoch, 'Best'=best epoch, or an integer to specify epoch, e.g. 120 would load weights from epoch 120.
    """ SETTINGS """
    SMALL_DATASET_DEBUG_MODE_TPSCN = True  # True=Load a small fraction of the dataset for testing only. False=Process entire dataset
    SAVE_PROBABILITY_MAPS_TPSCN = False  # True=Save probability images
    SAVE_SCN_OVERVIEW_TPSCN = False  # True=Save a overview image of the SCN images
    override_predict_region_TPSCN = False  # False=Process entire WSI. True=Process a region of the WSI, go to file to input region coordinates.
    USE_XY_POSITION_FROM_BACKGROUND_MASK_TPSCN = True  # True=Only predict tiles which is not background from binary mask. False=Predicts ALL tiles in WSI.
    OVERWRITE_BACKGROUND_MASK_TPSCN = True  # FALSE=Load existing background mask if exist. TRUE=compute background mask and overwrite existing background mask if it exist.
    """ PREDICTIONS """
    RUN_NEW_PREDICTION_TPSCN = True  # True: Use model to predict all tiles and save result as pickle. False: Load existing pickle file.
    PREDICT_WINDOW_SIZE_TPSCN = 128  # When doing prediction, how many pixels should the predict window be. Only use powers of 2 (32, 64, 128 etc).
    """ HEAT MAPS AND COLORMAP """
    MODEL_MAKE_HEAT_MAPS_TPSCN = True
    MODEL_MAKE_COLOR_MAP_TPSCN = True
    HEATMAP_SIGMA_TPSCN = 0.5  # The probability maps are filtered before heat maps are made. Sigma sets the standard deviation for the Gaussian kernel.
    HEATMAP_THRESHOLD_TPSCN = 0.2  # Threshold value to use when making heat maps
    UNDEFINED_CLASS_THRESHOLD_TPSCN = 0.6  # Threshold for defining a prediction as undefined class
    """ OTHERS """
    ENABLE_BINARY_MODE_TPSCN = False  # Set to True for binary classification (urothelium vs non-urothelium)
    # endregion

    """ SETTINGS """
    # region SETTINGS AND PARAMETERS
    OPTIMIZER = ['SGD']  # Available: 'adam', 'adamax', 'nadam', 'adadelta', 'adagrad', 'SGD'.
    batch_size = 128  # Batch size, how many samples to include in each batch during training.
    ACTIVATE_ReduceLROnPlateau = True  # Reduce learning rate on plateau.
    USE_MULTIPROCESSING = True  # Use keras multiprocessing function. Gives warning, recommended to be False.
    ACTIVATE_TENSORBOARD = False  # For logging to tensorboard during training
    CONTINUE_FROM_MODEL = 'last'  # If START_NEW_MODEL=False, this variable determines which model to continue working on. 'last'=continue last model, else specify model name eg. '2017-03-13_17-50-50'.
    FILE_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Name of current file
    N_WORKERS = 10
    MAX_QUEUE_SIZE = 32
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    N_CHANNELS = 3
    TILE_SIZE_TISSUE = 128
    START_NEW_MODEL = sys.argv[1]  # True=Start all new model from MODEL_MODE=0. False=Continue from previous
    # noinspection PyUnresolvedReferences
    tf_version = tf.__version__
    # endregion

    # region FOLDERS AND PATHS

    """ FOLDERS """
    LOG_FOLDER = 'logs/'
    SAVED_DATA_FOLDER = 'Saved_data/'
    MODEL_WEIGHT_FOLDER = 'Model_weights/'
    METADATA_FOLDER = 'metadata/'
    MISCLASSIFIED_IMAGE_FOLDER = 'Misclassified_images/'
    ALL_CLASSIFIED_IMAGE_FOLDER = 'All_classified_images/'
    CLASSIFICATION_REPORT_FOLDER = 'Classification_reports/'
    PROBABILITY_FOLDER = 'Probability_maps/'
    TENSORBOARD_DATA_FOLDER = 'Tensorboard/'
    MODELS_CORRELATION_FOLDER = 'Correlation_folder/'
    PICKLE_FILES_FOLDER = 'Pickle_files/'
    HEAT_MAPS_FOLDER = 'Heat_maps/'

    """ MODE FOLDERS """
    MODE_MDTTTL_FOLDER = 'Mode 02b - Tissue Mono-di-tri-scale/'
    MODE_CVTTL_FOLDER = 'Mode 02c - Cross Validation Tissue Mono-scale/'
    MODE_MDTTTL2_FOLDER = 'Mode 02d - Tissue Mono-di-tri-scale/'
    MODE_TISSUE_TEST_SET_FOLDER = 'Mode 05 - Tissue test set/'
    MODE_TISSUE_PREDICTION_FOLDER_TPSCN = 'Mode 07c - Tissue prediction on SCN/'

    """ FILES """
    SUMMARY_TRAINING_CSV_FILE = 'Summary_training.csv'  # A CSV file that includes summary of all model modes. Can be opened in Excel.
    SUMMARY_PREDICTIONS_CSV_FILE = 'Summary_predictions.csv'  # A CSV file that includes summary of all model modes. Can be opened in Excel.
    PROBABILITY_FILENAME = 'probability_image_class_'

    """ PICKLE FILES """
    ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE = 'ALL_MODELS_PARAMETER_CVTTL_PICKLE.obj'
    TRAINING_DATA_CVTTL_PICKLE_FILE = '_TRAINING_DATA_CVTTL_PICKLE.obj'
    CLASSIFICATION_REPORT_CVTTL_PICKLE_FILE = 'CLASSIFICATION_REPORT_CVTTL_PICKLE_FILE.obj'
    CV_F1_list_PICKLE_FILE = 'cv_f1_list.obj'

    ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE = 'ALL_MODELS_PARAMETER_MDTTTL_PICKLE.obj'
    MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE = 'MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE.obj'
    TRAINING_DATA_MDTTTL_PICKLE_FILE = '_TRAINING_DATA_MDTTTL_PICKLE.obj'

    ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE = 'ALL_MODELS_PARAMETER_MDTTTL2_PICKLE.obj'
    MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE = 'MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE.obj'
    TRAINING_DATA_MDTTTL2_PICKLE_FILE = '_TRAINING_DATA_MDTTTL2_PICKLE.obj'

    label_name_to_index_dict_PICKLE_FILE = 'label_name_to_index_dict_PICKLE.obj'
    HNUMBER_WITH_CUSTOM_LABELS_DICT_PICKLE_FILE = 'HNUMBER_WITH_CUSTOM_LABELS_DICT_PICKLE.obj'
    PROBABILITY_IMAGES_PICKLE_FILE = 'PROBABILITY_IMAGES_PICKLE.obj'
    COLORMAP_IMAGES_PICKLE_FILE = 'COLORMAP_IMAGES_PICKLE.obj'

    """ DATASET COORDINATE DICTS """
    COORDINATE_DICTS_MAIN_FOLDER = 'Coordinate_dicts_files/'
    TISSUE_GROUNDTRUTH_TRAINING_DICTS_PATH = COORDINATE_DICTS_MAIN_FOLDER + 'tissue_groundtruth_training/'
    TISSUE_GROUNDTRUTH_TEST_DICTS_PATH = COORDINATE_DICTS_MAIN_FOLDER + 'tissue_groundtruth_test/'
    TISSUE_PREDICTED_CLASSES_DICTS_PATH = COORDINATE_DICTS_MAIN_FOLDER + 'tissue_predicted_classes/'

    # Set DATASET_ROOT_PATH based on if the code is run on windows or unix
    if platform.system() != 'Windows':
        DATASET_ROOT_PATH = '../../Dataset/'
    else:
        DATASET_ROOT_PATH = '../00-Dataset/'

    # Specify main dataset folder
    DATASET_MAIN_FOLDER = 'WSI2/'

    """ DATASET FOLDERS """
    SCN_FOLDER = '1_scn_images/'
    TISSUE_TO_BE_PREDICTED_WSI_FOLDER = '2_tissue_to_be_predicted_wsi/'  # Folder that contains one folder per WSI, which contain SCN, overview image, metedata (no tiles)
    TISSUE_TO_BE_PREDICTED_WSI_PATH = DATASET_ROOT_PATH + DATASET_MAIN_FOLDER + TISSUE_TO_BE_PREDICTED_WSI_FOLDER
    SCN_PATH = DATASET_ROOT_PATH + DATASET_MAIN_FOLDER + SCN_FOLDER
    # endregion

    # region FILE_INITIALIZATION
    current_run_path = my_functions.init_file(
        SAVED_DATA_FOLDER=SAVED_DATA_FOLDER,
        LOG_FOLDER=LOG_FOLDER,
        FILE_NAME=FILE_NAME,
        START_NEW_MODEL=START_NEW_MODEL,
        CONTINUE_FROM_MODEL=CONTINUE_FROM_MODEL,
        METADATA_FOLDER=METADATA_FOLDER,
        MODELS_CORRELATION_FOLDER=MODELS_CORRELATION_FOLDER,
        TISSUE_PREDICTED_CLASSES_DICTS_PATH=TISSUE_PREDICTED_CLASSES_DICTS_PATH,
        USE_MULTIPROCESSING=USE_MULTIPROCESSING)

    # Create a new summary.csv file
    SUMMARY_CSV_FILE_PATH = current_run_path + SUMMARY_TRAINING_CSV_FILE
    my_functions.summary_csv_file_create_new(SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH)

    # endregion

    """ MODELS """

    # region MODE 2b - TISSUE TRANSFER LEARNING - MONO/DI/TRI - ONLY TRAINING SET - MDTTTL
    if MODE_TRANSFER_LEARNING_MDTTTL in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 2b - Tissue transfer learning - mono/di/tri - MDTTTL')

        # Load transfer learning function
        mode_2b_tissue_mono_di_tri.transfer_learning_model(OPTIMIZER=OPTIMIZER,
                                                           batch_size=batch_size,
                                                           current_run_path=current_run_path,
                                                           METADATA_FOLDER=METADATA_FOLDER,
                                                           ACTIVATE_TENSORBOARD=ACTIVATE_TENSORBOARD,
                                                           MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                                           N_CHANNELS=N_CHANNELS,
                                                           IMG_WIDTH=IMG_WIDTH,
                                                           IMG_HEIGHT=IMG_HEIGHT,
                                                           PROJECT_ID=PROJECT_ID,
                                                           DESCRIPTION=DESCRIPTION,
                                                           FILE_NAME=FILE_NAME,
                                                           N_WORKERS=N_WORKERS,
                                                           EPOCHES=EPOCHES_MDTTTL,
                                                           layer_config=classifier_layer_config_MDTTTL,
                                                           freeze_base_model=freeze_base_model_MDTTTL,
                                                           learning_rate=learning_rate_MDTTTL,
                                                           n_neurons_first_layer=n_neurons_first_layer_MDTTTL,
                                                           n_neurons_second_layer=n_neurons_second_layer_MDTTTL,
                                                           n_neurons_third_layer=n_neurons_third_layer_MDTTTL,
                                                           dropout=dropout_MDTTTL,
                                                           base_model=base_model_MDTTTL,
                                                           MODE_FOLDER=MODE_MDTTTL_FOLDER,
                                                           base_model_pooling=base_model_pooling_MDTTTL,
                                                           TRAINING_DATA_PICKLE_FILE=TRAINING_DATA_MDTTTL_PICKLE_FILE,
                                                           ACTIVATE_ReduceLROnPlateau=ACTIVATE_ReduceLROnPlateau,
                                                           TENSORBOARD_DATA_FOLDER=TENSORBOARD_DATA_FOLDER,
                                                           REDUCE_LR_PATIENCE=REDUCE_LR_PATIENCE_MDTTTL,
                                                           USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                                           MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                                           SMALL_DATASET_DEBUG_MODE=SMALL_DATASET_DEBUG_MODE_MDTTTL,
                                                           TILE_SIZE_TISSUE=TILE_SIZE_TISSUE,
                                                           which_scale_to_use_di=which_scale_to_use_di_MDTTTL,
                                                           SMALL_DATASET_DEBUG_N_TILES=SMALL_DATASET_DEBUG_N_TILES_MDTTTL,
                                                           TISSUE_GROUNDTRUTH_TRAINING_PATH=TISSUE_GROUNDTRUTH_TRAINING_DICTS_PATH,
                                                           SCN_PATH=SCN_PATH,
                                                           AUGMENTATION_MULTIPLIER=AUGMENTATION_MULTIPLIER_MDTTTL,
                                                           CLASSES_TO_AUGMENT=CLASSES_TO_AUGMENT_MDTTTL,
                                                           EARLY_STOPPING_LOSS_PATIENCE=EARLY_STOPPING_LOSS_PATIENCE_MDTTTL,
                                                           ALL_MODEL_PARAMETERS_PICKLE_FILE=ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE,
                                                           which_model_mode_to_use=which_model_mode_to_use_MDTTTL,
                                                           which_scale_to_use_mono=which_scale_to_use_mono_MDTTTL,
                                                           MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE,
                                                           tf_version=tf_version,
                                                           SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                           ENABLE_BINARY_MODE=ENABLE_BINARY_MODE_MDTTL)

    # endregion

    # region MODE 2c - CROSS VALIDATION - TISSUE TRANSFER LEARNING
    if MODE_TRANSFER_LEARNING_CVTTL in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 2c - Cross Validation - Tissue transfer learning')

        # Load transfer learning function
        mode_2c_tissue_cross_validation.transfer_learning_model(OPTIMIZER=OPTIMIZER,
                                                                batch_size=batch_size,
                                                                current_run_path=current_run_path,
                                                                METADATA_FOLDER=METADATA_FOLDER,
                                                                ACTIVATE_TENSORBOARD=ACTIVATE_TENSORBOARD,
                                                                MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                                                N_CHANNELS=N_CHANNELS,
                                                                IMG_WIDTH=IMG_WIDTH,
                                                                IMG_HEIGHT=IMG_HEIGHT,
                                                                PROJECT_ID=PROJECT_ID,
                                                                DESCRIPTION=DESCRIPTION,
                                                                FILE_NAME=FILE_NAME,
                                                                N_WORKERS=N_WORKERS,
                                                                EPOCHES=EPOCHES_CVTTL,
                                                                CLASSIFICATION_REPORT_FOLDER=CLASSIFICATION_REPORT_FOLDER,
                                                                layer_config=classifier_layer_config_CVTTL,
                                                                freeze_base_model=freeze_base_model_CVTTL,
                                                                learning_rate=learning_rate_CVTTL,
                                                                n_neurons_first_layer=n_neurons_first_layer_CVTTL,
                                                                n_neurons_second_layer=n_neurons_second_layer_CVTTL,
                                                                n_neurons_third_layer=n_neurons_third_layer_CVTTL,
                                                                dropout=dropout_CVTTL,
                                                                SAVE_MISCLASSIFIED_FIGURES=SAVE_MISCLASSIFIED_FIGURES_CVTTL,
                                                                MISCLASSIFIED_IMAGE_FOLDER=MISCLASSIFIED_IMAGE_FOLDER,
                                                                base_model=base_model_CVTTL,
                                                                MODE_FOLDER=MODE_CVTTL_FOLDER,
                                                                base_model_pooling=base_model_pooling_CVTTL,
                                                                TRAINING_DATA_PICKLE_FILE=TRAINING_DATA_CVTTL_PICKLE_FILE,
                                                                ACTIVATE_ReduceLROnPlateau=ACTIVATE_ReduceLROnPlateau,
                                                                TENSORBOARD_DATA_FOLDER=TENSORBOARD_DATA_FOLDER,
                                                                REDUCE_LR_PATIENCE=REDUCE_LR_PATIENCE_CVTTL,
                                                                USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                                                MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                                                SMALL_DATASET_DEBUG_MODE=SMALL_DATASET_DEBUG_MODE_CVTTL,
                                                                TILE_SIZE_TISSUE=TILE_SIZE_TISSUE,
                                                                which_model_mode_to_use=which_model_mode_to_use_CVTTL,
                                                                CLASSIFICATION_REPORT_PICKLE_FILE=CLASSIFICATION_REPORT_CVTTL_PICKLE_FILE,
                                                                which_scale_to_use_mono=which_scale_to_use_mono_CVTTL,
                                                                which_scale_to_use_di=which_scale_to_use_di_CVTTL,
                                                                N_FOLDS=N_FOLDS_CVTTL,
                                                                SMALL_DATASET_DEBUG_N_TILES=SMALL_DATASET_DEBUG_N_TILES_CVTTL,
                                                                TISSUE_GROUNDTRUTH_TRAINING_PATH=TISSUE_GROUNDTRUTH_TRAINING_DICTS_PATH,
                                                                SCN_PATH=SCN_PATH,
                                                                AUGMENTATION_MULTIPLIER=AUGMENTATION_MULTIPLIER_CVTTL,
                                                                CLASSES_TO_AUGMENT=CLASSES_TO_AUGMENT_CVTTL,
                                                                CV_F1_list_PICKLE_FILE=CV_F1_list_PICKLE_FILE,
                                                                EARLY_STOPPING_LOSS_PATIENCE=EARLY_STOPPING_LOSS_PATIENCE_CVTTL,
                                                                ALL_MODEL_PARAMETERS_PICKLE_FILE=ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE,
                                                                TRAINING_OR_TESTING_MODE=TRAINING_OR_TESTING_MODE_CVTTL,
                                                                MODELS_CORRELATION_FOLDER=MODELS_CORRELATION_FOLDER,
                                                                tf_version=tf_version,
                                                                SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                                ENABLE_BINARY_MODE=ENABLE_BINARY_MODE_CVTTL)

    # endregion

    # region MODE 2d - TISSUE TRANSFER LEARNING - MONO/DI/TRI - TRAIN/VAL/TEST SET - MDTTTL2
    if MODE_TRANSFER_LEARNING_MDTTTL2 in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 2d - Tissue transfer learning - mono/di/tri - MDTTTL2')

        # Load transfer learning function
        mode_2d_tissue_mono_di_tri.transfer_learning_model(OPTIMIZER=OPTIMIZER,
                                                           batch_size=batch_size,
                                                           current_run_path=current_run_path,
                                                           METADATA_FOLDER=METADATA_FOLDER,
                                                           ACTIVATE_TENSORBOARD=ACTIVATE_TENSORBOARD,
                                                           MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                                           N_CHANNELS=N_CHANNELS,
                                                           IMG_WIDTH=IMG_WIDTH,
                                                           IMG_HEIGHT=IMG_HEIGHT,
                                                           PROJECT_ID=PROJECT_ID,
                                                           DESCRIPTION=DESCRIPTION,
                                                           FILE_NAME=FILE_NAME,
                                                           N_WORKERS=N_WORKERS,
                                                           EPOCHES=EPOCHES_MDTTTL2,
                                                           CLASSIFICATION_REPORT_FOLDER=CLASSIFICATION_REPORT_FOLDER,
                                                           layer_config=classifier_layer_config_MDTTTL2,
                                                           freeze_base_model=freeze_base_model_MDTTTL2,
                                                           learning_rate=learning_rate_MDTTTL2,
                                                           n_neurons_first_layer=n_neurons_first_layer_MDTTTL2,
                                                           n_neurons_second_layer=n_neurons_second_layer_MDTTTL2,
                                                           n_neurons_third_layer=n_neurons_third_layer_MDTTTL2,
                                                           dropout=dropout_MDTTTL2,
                                                           base_model=base_model_MDTTTL2,
                                                           MODE_FOLDER=MODE_MDTTTL2_FOLDER,
                                                           base_model_pooling=base_model_pooling_MDTTTL2,
                                                           TRAINING_DATA_PICKLE_FILE=TRAINING_DATA_MDTTTL2_PICKLE_FILE,
                                                           ACTIVATE_ReduceLROnPlateau=ACTIVATE_ReduceLROnPlateau,
                                                           TENSORBOARD_DATA_FOLDER=TENSORBOARD_DATA_FOLDER,
                                                           REDUCE_LR_PATIENCE=REDUCE_LR_PATIENCE_MDTTTL2,
                                                           USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                                           MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                                           SMALL_DATASET_DEBUG_MODE=SMALL_DATASET_DEBUG_MODE_MDTTTL2,
                                                           TILE_SIZE_TISSUE=TILE_SIZE_TISSUE,
                                                           which_scale_to_use_di=which_scale_to_use_di_MDTTTL2,
                                                           SMALL_DATASET_DEBUG_N_TILES=SMALL_DATASET_DEBUG_N_TILES_MDTTTL2,
                                                           TISSUE_GROUNDTRUTH_TRAINING_PATH=TISSUE_GROUNDTRUTH_TRAINING_DICTS_PATH,
                                                           SCN_PATH=SCN_PATH,
                                                           AUGMENTATION_MULTIPLIER=AUGMENTATION_MULTIPLIER_MDTTTL2,
                                                           CLASSES_TO_AUGMENT=CLASSES_TO_AUGMENT_MDTTTL2,
                                                           EARLY_STOPPING_LOSS_PATIENCE=EARLY_STOPPING_LOSS_PATIENCE_MDTTTL2,
                                                           ALL_MODEL_PARAMETERS_PICKLE_FILE=ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE,
                                                           which_model_mode_to_use=which_model_mode_to_use_MDTTTL2,
                                                           which_scale_to_use_mono=which_scale_to_use_mono_MDTTTL2,
                                                           MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE,
                                                           TRAIN_VALIDATION_SPLIT=TRAIN_VALIDATION_SPLIT_MDTTL2,
                                                           ALL_CLASSIFIED_IMAGE_FOLDER=ALL_CLASSIFIED_IMAGE_FOLDER,
                                                           MISCLASSIFIED_IMAGE_FOLDER=MISCLASSIFIED_IMAGE_FOLDER,
                                                           SAVE_CONF_MAT_AND_REPORT_EVERY_N=SAVE_CONF_MAT_AND_REPORT_EVERY_N_MDTTL2,
                                                           SAVE_MISCLASSIFIED_FIGURES_EVERY_N=SAVE_MISCLASSIFIED_FIGURES_EVERY_N_MDTTTL2,
                                                           SAVE_CONF_MAT_AND_REPORT_ON_END=SAVE_CONF_MAT_AND_REPORT_ON_END_MDTTL2,
                                                           SAVE_MISCLASSIFIED_FIGURES_ON_END=SAVE_MISCLASSIFIED_FIGURES_ON_END_MDTTTL2,
                                                           SAVE_ALL_CLASSIFIED_FIGURES_ON_END=SAVE_ALL_CLASSIFIED_FIGURES_ON_END_MDTTL2,
                                                           tf_version=tf_version,
                                                           SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                                           ENABLE_BINARY_MODE=ENABLE_BINARY_MODE_MDTTL2)

    # endregion

    # region MODE 5 - TISSUE TEST SET
    if MODEL_TISSUE_TEST_SET in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 5 - Tissue test set')

        # Run prediction model
        mode_5_tissue_test_set.tissue_test_set(current_run_path=current_run_path,
                                               METADATA_FOLDER=METADATA_FOLDER,
                                               MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                               N_CHANNELS=N_CHANNELS,
                                               IMG_WIDTH=IMG_WIDTH,
                                               IMG_HEIGHT=IMG_HEIGHT,
                                               WHAT_MODEL_TO_LOAD=WHAT_MODEL_TO_LOAD_TTS,
                                               WHAT_MODEL_EPOCH_TO_LOAD=WHAT_MODEL_EPOCH_TO_LOAD_TTS,
                                               CLASSIFICATION_REPORT_FOLDER=CLASSIFICATION_REPORT_FOLDER,
                                               MODE_FOLDER=MODE_TISSUE_TEST_SET_FOLDER,
                                               MISCLASSIFIED_IMAGE_FOLDER=MISCLASSIFIED_IMAGE_FOLDER,
                                               SAVE_MISCLASSIFIED_FIGURES_TTS=SAVE_MISCLASSIFIED_FIGURES_TTS,
                                               PROJECT_ID=PROJECT_ID,
                                               DESCRIPTION=DESCRIPTION,
                                               FILE_NAME=FILE_NAME,
                                               USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                               batch_size=batch_size,
                                               MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                               N_WORKERS=N_WORKERS,
                                               SAVE_ALL_CLASSIFIED_FIGURES_TTS=SAVE_ALL_CLASSIFIED_FIGURES_TTS,
                                               ALL_CLASSIFIED_IMAGE_FOLDER=ALL_CLASSIFIED_IMAGE_FOLDER,
                                               SAVE_CONF_MAT_AND_REPORT_TTL=SAVE_CONF_MAT_AND_REPORT_TTS,
                                               ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE,
                                               ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE,
                                               MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE,
                                               ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE=ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE,
                                               MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE,
                                               TISSUE_GROUNDTRUTH_TEST_PATH=TISSUE_GROUNDTRUTH_TEST_DICTS_PATH,
                                               SMALL_DATASET_DEBUG_MODE=SMALL_DATASET_DEBUG_MODE_TEST,
                                               SMALL_DATASET_DEBUG_N_TILES=SMALL_DATASET_DEBUG_N_TILES_TEST,
                                               SCN_PATH=SCN_PATH,
                                               TILE_SIZE_TISSUE=TILE_SIZE_TISSUE,
                                               TRAINING_DATA_MDTTTL2_PICKLE_FILE=TRAINING_DATA_MDTTTL2_PICKLE_FILE,
                                               TRAINING_DATA_MDTTTL_PICKLE_FILE=TRAINING_DATA_MDTTTL_PICKLE_FILE,
                                               SUMMARY_CSV_FILE_PATH=SUMMARY_CSV_FILE_PATH,
                                               tf_version=tf_version)
    # endregion

    # region MODE 7c - TISSUE PREDICTION (ONE MODEL) - SCN Images - (Finished)
    if MODEL_TISSUE_PREDICTION_TPSCN in [True, 'True', 'true']:
        my_functions.my_print('')
        my_functions.my_print('Mode 7c - Tissue prediction (SCN)')

        # Check if labels-CSV file exist, if not, call function to create it
        if os.path.isfile(current_run_path + METADATA_FOLDER + HNUMBER_WITH_CUSTOM_LABELS_DICT_PICKLE_FILE) is False:
            my_functions.csv_2_dict_label_function(DATASET_ROOT_PATH, DATASET_MAIN_FOLDER, current_run_path,
                                                   METADATA_FOLDER, HNUMBER_WITH_CUSTOM_LABELS_DICT_PICKLE_FILE,
                                                   label_name_to_index_dict_PICKLE_FILE)

        # Calculate number of WSI to predict
        total_no_of_wsi = len(os.listdir(TISSUE_TO_BE_PREDICTED_WSI_PATH))

        # Loop through all WSI. Process one WSI at a time
        for current_wsi_index, wsi_filename_no_extension in enumerate(os.listdir(TISSUE_TO_BE_PREDICTED_WSI_PATH)):
            wsi_filename_w_extension = wsi_filename_no_extension + '.scn'
            wsi_dataset_folder = TISSUE_TO_BE_PREDICTED_WSI_PATH + wsi_filename_no_extension + '/'
            wsi_dataset_file_path = wsi_dataset_folder + wsi_filename_w_extension

            my_functions.my_print('Starting WSI {} of {} - {}'.format(str(current_wsi_index + 1), total_no_of_wsi, wsi_filename_no_extension))

            # Run prediction model
            mode_7c_tissue_prediction_coordinates.tissue_prediction(current_run_path=current_run_path,
                                                                    METADATA_FOLDER=METADATA_FOLDER,
                                                                    MODEL_WEIGHT_FOLDER=MODEL_WEIGHT_FOLDER,
                                                                    N_CHANNELS=N_CHANNELS,
                                                                    IMG_WIDTH=IMG_WIDTH,
                                                                    IMG_HEIGHT=IMG_HEIGHT,
                                                                    N_WORKERS=N_WORKERS,
                                                                    WHAT_MODEL_TO_LOAD=WHAT_MODEL_TO_LOAD_TPSCN,
                                                                    WHAT_MODEL_EPOCH_TO_LOAD=WHAT_MODEL_EPOCH_TO_LOAD_TPSCN,
                                                                    TISSUE_TO_BE_PREDICTED_WSI_PATH=TISSUE_TO_BE_PREDICTED_WSI_PATH,
                                                                    TILE_SIZE_TISSUE=TILE_SIZE_TISSUE,
                                                                    wsi_index=current_wsi_index,
                                                                    total_no_of_wsi=total_no_of_wsi,
                                                                    PROBABILITY_FILENAME=PROBABILITY_FILENAME,
                                                                    PROBABILITY_FOLDER=PROBABILITY_FOLDER,
                                                                    UNDEFINED_CLASS_THRESHOLD=UNDEFINED_CLASS_THRESHOLD_TPSCN,
                                                                    PROBABILITY_IMAGES_PICKLE_FILE=PROBABILITY_IMAGES_PICKLE_FILE,
                                                                    USE_MULTIPROCESSING=USE_MULTIPROCESSING,
                                                                    MAX_QUEUE_SIZE=MAX_QUEUE_SIZE,
                                                                    batch_size=batch_size,
                                                                    SMALL_DATASET_DEBUG_MODE=SMALL_DATASET_DEBUG_MODE_TPSCN,
                                                                    RUN_NEW_PREDICTION=RUN_NEW_PREDICTION_TPSCN,
                                                                    ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_MDTTTL_PICKLE_FILE,
                                                                    TRAINING_DATA_MDTTTL_PICKLE_FILE=TRAINING_DATA_MDTTTL_PICKLE_FILE,
                                                                    ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE=ALL_MODEL_PARAMETERS_CVTTL_PICKLE_FILE,
                                                                    PREDICT_WINDOW_SIZE=PREDICT_WINDOW_SIZE_TPSCN,
                                                                    SAVE_SCN_OVERVIEW=SAVE_SCN_OVERVIEW_TPSCN,
                                                                    SUMMARY_PREDICTIONS_CSV_FILE=SUMMARY_PREDICTIONS_CSV_FILE,
                                                                    MODE_TISSUE_PREDICTION_FOLDER=MODE_TISSUE_PREDICTION_FOLDER_TPSCN,
                                                                    MODEL_MAKE_HEAT_MAPS=MODEL_MAKE_HEAT_MAPS_TPSCN,
                                                                    MODEL_MAKE_COLOR_MAP=MODEL_MAKE_COLOR_MAP_TPSCN,
                                                                    HEATMAP_THRESHOLD=HEATMAP_THRESHOLD_TPSCN,
                                                                    HEATMAP_SIGMA=HEATMAP_SIGMA_TPSCN,
                                                                    HEAT_MAPS_FOLDER=HEAT_MAPS_FOLDER,
                                                                    override_predict_region=override_predict_region_TPSCN,
                                                                    ENABLE_BINARY_MODE=ENABLE_BINARY_MODE_TPSCN,
                                                                    MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_MDTTTL_PICKLE_FILE,
                                                                    TISSUE_PREDICTED_CLASSES_DICTS_PATH=TISSUE_PREDICTED_CLASSES_DICTS_PATH,
                                                                    USE_XY_POSITION_FROM_BACKGROUND_MASK=USE_XY_POSITION_FROM_BACKGROUND_MASK_TPSCN,
                                                                    SAVE_PROBABILITY_MAPS=SAVE_PROBABILITY_MAPS_TPSCN,
                                                                    OVERWRITE_BACKGROUND_MASK=OVERWRITE_BACKGROUND_MASK_TPSCN,
                                                                    ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE=ALL_MODEL_PARAMETERS_MDTTTL2_PICKLE_FILE,
                                                                    MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE=MODELS_AND_LOSS_ARRAY_MDTTTL2_PICKLE_FILE,
                                                                    COLORMAP_IMAGES_PICKLE_FILE=COLORMAP_IMAGES_PICKLE_FILE,
                                                                    PICKLE_FILES_FOLDER=PICKLE_FILES_FOLDER,
                                                                    wsi_filename_no_extension=wsi_filename_no_extension,
                                                                    wsi_filename_w_extension=wsi_filename_w_extension,
                                                                    wsi_dataset_folder=wsi_dataset_folder,
                                                                    wsi_dataset_file_path=wsi_dataset_file_path,
                                                                    PROJECT_ID=PROJECT_ID)

    # endregion
