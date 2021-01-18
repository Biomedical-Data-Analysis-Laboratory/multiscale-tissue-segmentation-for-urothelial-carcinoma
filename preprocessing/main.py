import preprocess_region
import MyFunctions
import sys
import os

""" Settings """
REMOVE_WHITE_BACKGROUND = True  # If SCN file has white background, this function removes it.
ENABLE_LOGGING = True  # Enable logging of information to file while running the program.
TILE_SIZE = 128  # Final size of each tile image.
OVERLAPPING_TILES = False  # True=tiles are halfway-overlapped. False=No overlap
SAVE_OVERVIEW_IMG = True  # Save an overview image after removing white background
SAVE_DELETE_IMG = False  # Enable this to also save the deleted tile images.
DI_SCALE_PREPROCESSING = False  # Enable this to preprocess WSI for use with Di-Scale model (saves 400x and 100x tiles)
TRI_SCALE_PREPROCESSING = True  # Enable this to preprocess WSI for use with Tri-Scale model (saves 400x, 100x and 25x tiles)
PADDING_AROUND_IMG_SIZE = 200  # Include padding around image when removing white background

""" Preprocess regions settings """
KEEP_INSIDE_OF_REGION = True  # If a XML file is found, should the masked area be kept or not? True=Everything around is masked out. False=The region is masked.
SAVE_REGION_IMG = True  # True=Save an image of the full image after binary mask have been applied. False=Do not save image.
REGION_MARGIN_SIZE = 200  # A small margin around regions
SAVE_BACKGROUND_AS_CLASS = False  # When extracting background class, set this to TRUE. Remember to enable SAVE_DELETE_IMG.
SAVE_FORMAT_REGION = 'coordinates'  # How to save tiles. 'jpeg'=save tiles as JPEG images. 'coordinates'=save tiles coordinates in a dict
REDUCE_TO_ONE_REGION_ONLY = True  # For Debugging. Save time by only preprocessing one region in each WSI. False=disable function.
ONLY_EXTRACT_N_TILES = 'no'  # For Debugging. Save time by only extracting N tiles. Specify an int to how many tiles. Set to random string to disable. (Setting to False will not work)
FIND_OPTIMAL_START_COORDINATES = False  # TAKES A LONG TIME. Will search through each region many times to find the best staring coordinates.

""" PATHS """
INPUT_ROOT_PATH = 'input/'  # Path where input images are located
SAVE_ROOT_PATH = 'combine_dicts/'
MASK_ROOT_PATH = 'Masks/'
ALL_DICTS_TO_BE_COMBINED_PATH = 'combine_dicts/'
LOG_PATH = 'Logs/'  # Path where to save logs
FILE_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Name of this file
SUMMARY_IMAGE_FILE = 'Summary/summary_image.csv'  # A CSV file that includes summary of each model. Can be opened in Excel.
SUMMARY_REGION_FILE = 'Summary/summary_region.csv'  # A CSV file that includes summary of each model. Can be opened in Excel.

""" Start logging """
if ENABLE_LOGGING:
    MyFunctions.start_logging(LOG_PATH, FILE_NAME)

""" Program init """
MyFunctions.program_init(SUMMARY_IMAGE_FILE)

""" PREPROCESS REGION """
# Update indexes
class_index = 0

if REDUCE_TO_ONE_REGION_ONLY:
    print('')
    print('WARNING! REDUCE_TO_ONE_REGION_ONLY set to True. Only PreProcessing one region per WSI!')
    print('')

if isinstance(ONLY_EXTRACT_N_TILES, int):
    print('')
    print('WARNING! ONLY_EXTRACT_N_TILES active. Only extracting {} tiles per region!'.format(ONLY_EXTRACT_N_TILES))
    print('')

# Search through the input directory for all class folders
for current_class in os.listdir(INPUT_ROOT_PATH):
    # Update indexes
    class_index += 1
    wsi_index = 0

    # Search through for all WSI in each class folder
    for current_wsi in os.listdir(INPUT_ROOT_PATH + current_class):
        # Update indexes
        wsi_index += 1
        print('Preprocessing class {}/{}, WSI {}/{}.'.format(class_index, len(os.listdir(INPUT_ROOT_PATH)), wsi_index, len(os.listdir(INPUT_ROOT_PATH + current_class))))

        # Get all files in current folder
        current_folder_files = [f for f in os.listdir(INPUT_ROOT_PATH + current_class + '/' + current_wsi) if os.path.isfile(os.path.join(INPUT_ROOT_PATH + current_class + '/' + current_wsi, f))]

        # Process one file at the time
        for current_filename in current_folder_files:

            # Remove extension from filename
            current_filename_no_extension = os.path.splitext(current_filename)[0]

            if current_filename_no_extension == current_wsi:
                # Check that the current file is an image file, and not an XML file.
                if os.path.splitext(current_filename)[1] in {'.scn', '.tif', '.jpeg', '.jpg', '.bmp', '.png'}:
                    MyFunctions.myPrint('Loaded {}. Starting preprocessing'.format(current_filename))

                    # Update current paths. Also check if directory for saving images exist, if not, create one.
                    current_image_path, current_saved_tiles_path_400x, current_saved_tiles_path_100x, current_saved_tiles_path_25x, \
                        current_black_mask_tiles_path, current_deleted_tiles_path, current_metadata_path = MyFunctions.image_init(
                            filename_no_extension=current_filename_no_extension,
                            input_root_path=INPUT_ROOT_PATH + current_class + '/',
                            CLASS_NAME=current_class,
                            DI_SCALE_PREPROCESSING=DI_SCALE_PREPROCESSING,
                            TRI_SCALE_PREPROCESSING=TRI_SCALE_PREPROCESSING,
                            SAVE_DELETE_IMG=SAVE_DELETE_IMG,
                            SAVE_ROOT_PATH=SAVE_ROOT_PATH,
                            SAVE_FORMAT_IMAGE=SAVE_FORMAT_REGION)

                    # Define path for XML file
                    current_xml_path = '{}{}/{}/{}.xml'.format(INPUT_ROOT_PATH, current_class, current_filename_no_extension, current_filename_no_extension)

                    # Check that the XML file exist.
                    if os.path.exists(current_xml_path):
                        MyFunctions.myPrint('Found XML file for {}.'.format(current_filename))
                    else:
                        MyFunctions.myPrint('No XML file found for {}.'.format(current_filename))

                    # Preprocess current image
                    preprocess_region.preprocess_region(
                        remove_white_background=REMOVE_WHITE_BACKGROUND,
                        save_delete_img=SAVE_DELETE_IMG,
                        xml_path=current_xml_path,
                        tile_size=TILE_SIZE,
                        filename=current_filename,
                        saved_tiles_path_400x=current_saved_tiles_path_400x,
                        black_mask_tiles_path=current_black_mask_tiles_path,
                        deleted_tiles_path=current_deleted_tiles_path,
                        KEEP_INSIDE_OF_REGION=KEEP_INSIDE_OF_REGION,
                        SAVE_REGION_IMG=SAVE_REGION_IMG,
                        SUMMARY_FILE=SUMMARY_REGION_FILE,
                        metadata_path=current_metadata_path,
                        INPUT_ROOT_PATH=INPUT_ROOT_PATH,
                        CLASS_NAME=current_class,
                        OVERLAPPING_TILES=OVERLAPPING_TILES,
                        image_path=current_image_path,
                        saved_tiles_path_100x=current_saved_tiles_path_100x,
                        saved_tiles_path_25x=current_saved_tiles_path_25x,
                        PADDING_AROUND_IMG_SIZE=PADDING_AROUND_IMG_SIZE,
                        DI_SCALE_PREPROCESSING=DI_SCALE_PREPROCESSING,
                        TRI_SCALE_PREPROCESSING=TRI_SCALE_PREPROCESSING,
                        REGION_MARGIN_SIZE=REGION_MARGIN_SIZE,
                        SAVE_BACKGROUND_AS_CLASS=SAVE_BACKGROUND_AS_CLASS,
                        SAVE_FORMAT=SAVE_FORMAT_REGION,
                        REDUCE_TO_ONE_REGION_ONLY=REDUCE_TO_ONE_REGION_ONLY,
                        current_class=current_class,
                        ONLY_EXTRACT_N_TILES=ONLY_EXTRACT_N_TILES,
                        filename_no_extension=current_filename_no_extension,
                        SAVE_ROOT_PATH=SAVE_ROOT_PATH,
                        SAVE_OVERVIEW_IMG=SAVE_OVERVIEW_IMG,
                        FIND_OPTIMAL_START_COORDINATES=FIND_OPTIMAL_START_COORDINATES)

""" END LOG """
if ENABLE_LOGGING:
    MyFunctions.end_logging()
