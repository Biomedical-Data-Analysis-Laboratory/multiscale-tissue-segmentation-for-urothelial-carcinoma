from xml.etree import ElementTree
import logging
import datetime
import time
import csv
import os
import glob


# Function runs when program starts
def program_init(SUMMARY_IMAGE_FILE):
    # Check if summary directory exist, if not, create one.
    if not os.path.exists(SUMMARY_IMAGE_FILE.split('/')[0]):
        os.makedirs(SUMMARY_IMAGE_FILE.split('/')[0])


# Function that runs whenever a new image is loaded
def image_init(filename_no_extension, input_root_path, CLASS_NAME, DI_SCALE_PREPROCESSING, TRI_SCALE_PREPROCESSING,
               SAVE_DELETE_IMG, SAVE_ROOT_PATH, SAVE_FORMAT_IMAGE):
    # Define paths
    current_image_path = '{}{}/'.format(input_root_path, filename_no_extension)
    # current_unlabelled_tiles_path = '{}{}/'.format(input_root_path, unlabelled_tiles_path)
    # current_labelled_tiles_path = '{}{}/'.format(input_root_path, labelled_tiles_path)
    # current_working_path = '{}{}/'.format(input_root_path, os.path.splitext(filename_no_extension)[0])
    # current_save_path_400x = '{}/400x/{}/{}'.format(SAVE_ROOT_PATH, filename_no_extension, CLASS_NAME)
    current_save_path_400x = '{}/400x/{}'.format(SAVE_ROOT_PATH, filename_no_extension)
    # current_save_path = '../Dataset/WSI_tri-scale/2_labelled_tiles/{}'.format(CLASS_NAME)
    # current_save_path_100x = '../Dataset/WSI_tri-scale/2_labelled_tiles_100x/{}'.format(CLASS_NAME)
    # current_save_path_25x = '../Dataset/WSI_tri-scale/2_labelled_tiles_25x/{}'.format(CLASS_NAME)
    # current_save_path_100x = '{}/{}/saved tiles (100x)'.format(SAVE_ROOT_PATH, filename_no_extension)
    current_save_path_100x = '{}/100x/{}/{}'.format(SAVE_ROOT_PATH, filename_no_extension, CLASS_NAME)
    # current_save_path_25x = '{}/{}/saved tiles (25x)'.format(SAVE_ROOT_PATH, filename_no_extension)
    current_save_path_25x = '{}/25x/{}/{}'.format(SAVE_ROOT_PATH, filename_no_extension, CLASS_NAME)
    current_mask_path = '{}/{}/deleted black mask tiles'.format(SAVE_ROOT_PATH, filename_no_extension)
    current_delete_path = '{}/{}/deleted background tiles'.format(SAVE_ROOT_PATH, filename_no_extension)
    current_metadata_path = "{}/{}#metadata.xml".format(current_image_path, filename_no_extension)

    # Create folders
    if SAVE_FORMAT_IMAGE is 'jpeg':
        if not os.path.exists(current_save_path_400x):
            os.makedirs(current_save_path_400x)

        if (DI_SCALE_PREPROCESSING and not os.path.exists(current_save_path_100x)) or (TRI_SCALE_PREPROCESSING and not os.path.exists(current_save_path_100x)):
            os.makedirs(current_save_path_100x)

        if TRI_SCALE_PREPROCESSING and not os.path.exists(current_save_path_25x):
            os.makedirs(current_save_path_25x)

        if SAVE_DELETE_IMG and not os.path.exists(current_mask_path):
            os.makedirs(current_mask_path)

        if SAVE_DELETE_IMG and not os.path.exists(current_delete_path):
            os.makedirs(current_delete_path)
    elif SAVE_FORMAT_IMAGE is 'coordinates':
        if not os.path.exists(SAVE_ROOT_PATH):
            os.makedirs(SAVE_ROOT_PATH)

    # Get (x_inside, y_inside) values from CSV file (if it exist)
    x_inside_400x, y_inside_400x, x_inside_100x, y_inside_100x, x_inside_25x, y_inside_25x = csv_2_dict_function(current_image_path)

    if not os.path.isfile(current_metadata_path):
        # Create XML file to store information about the image
        metadata_root = ElementTree.Element("metadata_root")
        metadata_doc = ElementTree.SubElement(metadata_root, "metadata_doc")

        # Build the XML structure
        ElementTree.SubElement(metadata_doc, "size", height="1", width="2")
        ElementTree.SubElement(metadata_doc, "x_inside", a400x=x_inside_400x, a100x=x_inside_100x, a25x=x_inside_25x)
        ElementTree.SubElement(metadata_doc, "y_inside", a400x=y_inside_400x, a100x=y_inside_100x, a25x=y_inside_25x)
        current_metadata_xmlfile = ElementTree.ElementTree(metadata_root)
        current_metadata_xmlfile.write(current_metadata_path)

    return current_image_path, current_save_path_400x, current_save_path_100x, current_save_path_25x, current_mask_path, current_delete_path, current_metadata_path


def readXML(path, mask_width, padding, REDUCE_TO_ONE_REGION_ONLY):
    # This function read the coordinates from the XML file created by the ImageScope SCN viewer
    # program, and returns a array list. The function is made generic and can read an
    # arbitrary number of polygons from the XML file.
    # Rune Wetteland - 09.02.2017

    # Temporary variables used in the function
    # temp1_x = []
    # temp1_y = []
    temp2_x = []
    # temp2_y = []
    dict_list = {}
    xml_liste = []

    # Parse the root of the XML file structure
    root = ElementTree.parse(path).getroot()

    # Iterate through the XML file to get the data
    for Region in root.iter('Region'):

        # Lag en ny key i dictionary for hver region
        dict_list[Region.get('Id')] = {}

        # Reset values. Viktig, for hvis ikke overskrives de samme dataene.
        temp_x = []
        temp_y = []

        for current_vertex in Region.find('Vertices'):

            if int(float(current_vertex.get('X'))) <= mask_width:
                # Sett inn i mask1
                temp_x.append(int(float(current_vertex.get('X')) + padding))
                temp_y.append(int(float(current_vertex.get('Y')) + padding))
            elif int(float(current_vertex.get('X'))) > mask_width:
                # Sett inn i mask2
                temp2_x.append(int(current_vertex.get('X')) + padding - mask_width)
                # temp2_y.append(int(current_vertex.get('Y')))

            # Legg verdiene inn i dictionary
            dict_list[Region.get('Id')]['X'] = temp_x
            dict_list[Region.get('Id')]['Y'] = temp_y

        if REDUCE_TO_ONE_REGION_ONLY:
            break

    # Konverter dictionary om til en liste. Slaar sammen X og Y koordinater til tuples.
    for list_items in dict_list.keys():
        xml_liste.append(list(zip(dict_list[list_items]['X'], dict_list[list_items]['Y'])))

    # Returner listen
    return xml_liste


def start_logging(log_path, file_name):
    # Check if directory for logs exist, if not, create one.
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Test start time
    global start_time
    start_time = time.time()
    start_time_formatted = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    start_time_logger = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H-%M-%S')

    # Create a logger
    logg_navn = '{0}{1}-{2}.log'.format(log_path, start_time_logger, file_name)
    logging.basicConfig(filename=logg_navn, level=logging.INFO)

    # Test start
    print("\n")
    print('Program started at {}'.format(start_time_formatted))
    logging.info('Program started at {}'.format(start_time_formatted))

def end_logging():
    end_time = time.time()
    end_time_formatted = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    myPrint('\t')
    myPrint("Program finished at: {}".format(end_time_formatted))
    elapse_time = end_time - start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    myPrint('Total time(H:M:S): %02d:%02d:%02d' % (h, m, s))


def myPrint(msg, error=False):
    if not error:
        # logging.debug(msg)
        logging.info(msg)
        print(msg)
    else:
        logging.error(msg)
        print(msg)


def remove_white_background(input_img, PADDING_AROUND_IMG_SIZE, OVERRIDE_X_INSIDE, OVERRIDE_Y_INSIDE):
    myPrint('Starting removing white background')

    # Reset variables
    remove_rows_top = 0
    remove_rows_bottom = 0
    remove_cols_left = 0
    remove_cols_right = 0

    white_background_vector = [250, 251, 252, 253, 254, 255]

    # Search for a point within the SCN image which is not white.
    a = 10
    jump_x = [i * (input_img.width // a + 1) for i in list(range(1, a))]
    jump_y = [i * (input_img.height // a + 1) for i in list(range(1, a))]

    for curr_jmp_y in jump_y:
        for curr_jmp_x in jump_x:
            if not input_img(curr_jmp_x - 1, curr_jmp_y)[1] in white_background_vector:
                break
        if not input_img(curr_jmp_x - 1, curr_jmp_y)[1] in white_background_vector:
            break

    if OVERRIDE_X_INSIDE == 0:
        x_inside = curr_jmp_x
    else:
        x_inside = OVERRIDE_X_INSIDE

    if OVERRIDE_Y_INSIDE == 0:
        y_inside = curr_jmp_y
    else:
        y_inside = OVERRIDE_Y_INSIDE

    ##### REMOVE HORIZONTAL WHITE LINES (TOP AND DOWN)
    if input_img(x_inside, 0)[1] in white_background_vector:
        first = 0
        last = y_inside
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # print('first {}, midpoint {}, last {}'.format(first, midpoint, last))
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                # print('if statement True')
                first = midpoint + 1
            else:
                # print('if statement False')
                last = midpoint - 1
        # print('midpoint', midpoint)
        remove_rows_top = midpoint - 1
    ##### END HORIZONTAL WHITE LINES (TOP AND DOWN)
    ##### REMOVE HORIZONTAL WHITE LINES (BOTTOM AND UP)
    if input_img(x_inside, (input_img.height - 1))[1] in white_background_vector:
        # first = (current_image.height // 2) - 5000
        first = y_inside
        last = input_img.height

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # print('first {}, midpoint {}, last {}'.format(first, midpoint, last))
            # if current_image(((current_image.width // current_divide_constant)-(current_image.width//4)), midpoint)[1] == 255:
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                # print('if statement True')
                last = midpoint - 1
            else:
                # print('if statement False')
                first = midpoint + 1

        remove_rows_bottom = midpoint
    ##### END HORIZONTAL WHITE LINES (BOTTOM AND UP)
    ##### REMOVE VERTICAL WHITE LINES (VENSTRE MOT HoYRE)
    if input_img(0, y_inside)[1] == 255:
        first = 0
        last = x_inside

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # print('first {}, midpoint {}, last {}'.format(first, midpoint, last))
            if input_img(midpoint, y_inside)[1] == 255:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_cols_left = midpoint - 1

    ##### END VERTICAL WHITE LINES (VENSTRE MOT HOYRE)
    ##### REMOVE VERTICAL WHITE LINES (HOYRE MOT VENSTRE)
    if input_img(input_img.width - 1, y_inside)[1] == 255:
        first = x_inside
        last = input_img.width
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # print('first {}, midpoint {}, last {}'.format(first, midpoint, last))
            if input_img(midpoint, y_inside)[1] == 255:
                last = midpoint - 1
            else:
                first = midpoint + 1
        remove_cols_right = midpoint + 1
    ##### END VERTICAL WHITE LINES (HOYRE MOT VENSTRE)

    # Calculate new width/height of image and crop.
    if remove_rows_bottom != 0:
        # Calculate new width/height
        # new_width = ((input_img.width - remove_cols_left - (input_img.width - remove_cols_right)) // tile_size) * tile_size
        new_width = (input_img.width - remove_cols_left - (input_img.width - remove_cols_right))
        # new_height = ((input_img.height - remove_rows_top - (input_img.height - remove_rows_bottom)) // tile_size) * tile_size
        new_height = (input_img.height - remove_rows_top - (input_img.height - remove_rows_bottom))

        # Make sure that new width/height is an even number, if not remove one extra line.
        # if new_width % 2 != 0:
        # new_width -= 1

        # if new_height % 2 != 0:
        # new_height -= 1

        # Include a border around image (to extract 25x tiles later)
        remove_cols_left = remove_cols_left - PADDING_AROUND_IMG_SIZE
        remove_rows_top = remove_rows_top - PADDING_AROUND_IMG_SIZE
        new_width = new_width + 2 * PADDING_AROUND_IMG_SIZE
        new_height = new_height + 2 * PADDING_AROUND_IMG_SIZE

        #####################

        # print(remove_cols_left)
        # print(remove_rows_top)
        # exit()
        #####################

        # Remove white background around the image
        input_img = input_img.extract_area(remove_cols_left, remove_rows_top, new_width, new_height)

        # Rotate image 90 degree (necessary because aperio imagescope does this, and the region coordinates need to match)
        # input_img = input_img.rot(1)

        myPrint('Finished removing white background. New height:{}, width:{}'.format(new_height, new_width))
        return input_img, remove_cols_left, remove_rows_top
    else:
        myPrint('No white background found around image. No cropping done.')
        return input_img


def csv_2_dict_function(current_image_path):
    x_inside_400x = y_inside_400x = x_inside_100x = y_inside_100x = x_inside_25x = y_inside_25x = 0

    # Search for all CSV files in dataset folder (should only be one file, containing list of labels)
    csv_label_file = [i for i in glob.glob(current_image_path + '*.csv')]

    # Check that one file was found
    if len(csv_label_file) == 1:
        # Read from CSV file
        with open(csv_label_file[0]) as csvfile:

            # Jump over first line (header info)
            next(csvfile, None)

            # Create a reader
            readCSV = csv.reader(csvfile, delimiter=';')

            # Go through each row of the file
            for row in readCSV:
                # Read values from file
                if row[0] == '400x':
                    x_inside_400x = row[1]
                    y_inside_400x = row[2]
                elif row[0] == '100x':
                    x_inside_100x = row[1]
                    y_inside_100x = row[2]
                elif row[0] == '25x':
                    x_inside_25x = row[1]
                    y_inside_25x = row[2]
                else:
                    print('No')

    return str(x_inside_400x), str(y_inside_400x), str(x_inside_100x), str(y_inside_100x), str(x_inside_25x), str(y_inside_25x)


def save_tile_coordinate(tile_dict, label, x_pos_400x, y_pos_400x, x_pos_100x, y_pos_100x, x_pos_25x, y_pos_25x, wsi_filename):
    id_number = len(tile_dict.keys())
    tile_dict[id_number] = dict()

    # Save coordinates in array (so deep learning model can extract tile directly)
    tile_dict[id_number]['path'] = wsi_filename
    tile_dict[id_number]['coordinates_400x'] = (int(x_pos_400x), int(y_pos_400x))
    tile_dict[id_number]['coordinates_100x'] = (int(x_pos_100x), int(y_pos_100x))
    tile_dict[id_number]['coordinates_25x'] = (int(x_pos_25x), int(y_pos_25x))
    tile_dict[id_number]['label'] = label


def save_tile_jpeg(tile, x_pos, y_pos, cropped_xml, magnification_scale, jpeg_save_path, wsi_filename, crop_x_min, REGION_MARGIN_SIZE):
    filename = '{}/{}#{}#{}#{}.jpg'.format(jpeg_save_path, os.path.splitext(wsi_filename)[0], int(x_pos), int(y_pos), magnification_scale)
    tile.jpegsave(filename, Q=100)
