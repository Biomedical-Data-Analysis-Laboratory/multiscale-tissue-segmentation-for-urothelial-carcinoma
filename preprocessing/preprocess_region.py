import os
vipshome = 'C:/Users/2918257/Downloads/Vips/vips-dev-w64-all-8.7.3/vips-dev-8.7/bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import MyFunctions
import pyvips
Vips = pyvips
from xml.etree import ElementTree
import pickle
import time
import os
import math
import csv


def preprocess_region(remove_white_background, save_delete_img, xml_path, tile_size, filename, saved_tiles_path_400x, black_mask_tiles_path,
                      deleted_tiles_path, KEEP_INSIDE_OF_REGION, SAVE_REGION_IMG, SUMMARY_FILE, metadata_path, INPUT_ROOT_PATH,
                      CLASS_NAME, OVERLAPPING_TILES, image_path, saved_tiles_path_100x, saved_tiles_path_25x, PADDING_AROUND_IMG_SIZE,
                      DI_SCALE_PREPROCESSING, TRI_SCALE_PREPROCESSING, REGION_MARGIN_SIZE, SAVE_BACKGROUND_AS_CLASS, SAVE_FORMAT,
                      REDUCE_TO_ONE_REGION_ONLY, current_class, ONLY_EXTRACT_N_TILES, filename_no_extension, SAVE_ROOT_PATH,
                      SAVE_OVERVIEW_IMG, FIND_OPTIMAL_START_COORDINATES):
    # This function pre-process the images in the input folder.
    # First the function removes any white area/border around the images.
    # Then the function applies a binary mask (if XML file exist) to mask out any unwanted parts of the image.
    # Then the function splits the large image up into small tile-images and saves them. The function
    # checks each tile of it should be saved or discarded.

    # region PREPROCESSING INIT
    # Start main timer
    main_timer_start = time.time()

    # Variable to keep track if we have cropped the coordinates in XML files
    cropped_xml = False

    # Define scale between images
    X_scale_400x_100x = 0.25
    Y_scale_400x_100x = 0.25
    X_scale_400x_25x = 0.0625
    Y_scale_400x_25x = 0.0625

    # Variables to count total number of tiles in each case
    count_tiles_saved = 0
    count_background_tiles_deleted = 0
    count_mask_tiles_deleted = 0
    current_list_of_all_tiles_dict = dict()

    # Parameters
    current_image_path = '{}/{}'.format(image_path, filename)

    # Parse the root of the XML file structure
    metadata_root = ElementTree.parse(metadata_path)

    # Check if summary file exist, if not, create one.
    if not os.path.isfile(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, 'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['FILENAME', 'BLACK_MASK_TILES_DELETED', 'BACKGROUND_TILES_DELETED', 'TILES_SAVED',
                                     'CLASS_NAME', 'TIME(H:M:S)', 'REGION_MARGIN_SIZE', 'TILE_SIZE', 'OVERLAPPING_TILES',
                                     'SAVE_DELETE_IMG', 'KEEP_INSIDE_OF_REGION', 'DI_SCALE_PREPROCESS', 'TRI_SCALE_PREPROCESS'])
        except Exception as e:
            MyFunctions.myPrint('Error writing to file', error=True)
            MyFunctions.myPrint(e, error=True)

    # endregion

    # region LOAD 400x IMAGE
    current_image_400x = Vips.Image.new_from_file(current_image_path, level=0)

    # Rotate image 90 degree (necessary because aperio imagescope does this, and the region coordinates need to match)
    current_image_400x = current_image_400x.rot(1)

    MyFunctions.myPrint('Current 400x image: {}, height:{} width:{}'.format(current_image_path, current_image_400x.height, current_image_400x.width))
    # Flatten image to remove alpha channel (only if SCN images)
    if filename[-3:] == 'scn':
        current_image_400x = current_image_400x.flatten()

    # Read the attributes from the metadata XML file
    x_inside_400x = int(metadata_root.find('metadata_doc/x_inside').attrib['a400x'])
    y_inside_400x = int(metadata_root.find('metadata_doc/y_inside').attrib['a400x'])

    # Remove white background
    if remove_white_background:
        current_image_400x, x_offset_400x, y_offset_400x = MyFunctions.remove_white_background(input_img=current_image_400x,
                                                                                               PADDING_AROUND_IMG_SIZE=PADDING_AROUND_IMG_SIZE,
                                                                                               OVERRIDE_X_INSIDE=x_inside_400x,
                                                                                               OVERRIDE_Y_INSIDE=y_inside_400x)

    # Find original image width
    original_img_width = current_image_400x.width - 2 * PADDING_AROUND_IMG_SIZE

    # Find the correct attribute to update
    current_attribute = metadata_root.find('metadata_doc/size')
    current_attribute.set('height', str(current_image_400x.height - 2 * PADDING_AROUND_IMG_SIZE))
    current_attribute.set('width', str(current_image_400x.width - 2 * PADDING_AROUND_IMG_SIZE))

    # Save the XML document
    metadata_root.write(metadata_path)
    # endregion

    # region LOAD 100x IMAGE
    if DI_SCALE_PREPROCESSING is True or TRI_SCALE_PREPROCESSING is True:
        current_image_100x = Vips.Image.new_from_file(current_image_path, level=1)

        # Rotate image 90 degree (necessary because aperio imagescope does this, and the region coordinates need to match)
        current_image_100x = current_image_100x.rot(1)

        MyFunctions.myPrint('Current 100x image: {}, height:{} width:{}'.format(current_image_path, current_image_100x.height, current_image_100x.width))
        # Flatten image to remove alpha channel (only if SCN images)
        if filename[-3:] == 'scn':
            current_image_100x = current_image_100x.flatten()

        # Read the attributes from the metadata XML file
        x_inside_100x = int(metadata_root.find('metadata_doc/x_inside').attrib['a100x'])
        y_inside_100x = int(metadata_root.find('metadata_doc/y_inside').attrib['a100x'])

        # Remove white background
        if remove_white_background:
            current_image_100x, x_offset_100x, y_offset_100x = MyFunctions.remove_white_background(input_img=current_image_100x,
                                                                                                   PADDING_AROUND_IMG_SIZE=PADDING_AROUND_IMG_SIZE * X_scale_400x_100x,
                                                                                                   OVERRIDE_X_INSIDE=x_inside_100x,
                                                                                                   OVERRIDE_Y_INSIDE=y_inside_100x)
    # endregion

    # region LOAD 25x IMAGE
    if TRI_SCALE_PREPROCESSING is True:
        current_image_25x = Vips.Image.new_from_file(current_image_path, level=2)

        # Rotate image 90 degree (necessary because aperio imagescope does this, and the region coordinates need to match)
        current_image_25x = current_image_25x.rot(1)

        MyFunctions.myPrint('Current 25x image: {}, height:{} width:{}'.format(current_image_path, current_image_25x.height, current_image_25x.width))
        # Flatten image to remove alpha channel (only if SCN images)
        if filename[-3:] == 'scn':
            current_image_25x = current_image_25x.flatten()

        # Read the attributes from the metadata XML file
        x_inside_25x = int(metadata_root.find('metadata_doc/x_inside').attrib['a25x'])
        y_inside_25x = int(metadata_root.find('metadata_doc/y_inside').attrib['a25x'])

        # Remove white background
        if remove_white_background:
            current_image_25x, x_offset_25x, y_offset_25x = MyFunctions.remove_white_background(input_img=current_image_25x,
                                                                                                PADDING_AROUND_IMG_SIZE=PADDING_AROUND_IMG_SIZE * X_scale_400x_25x,
                                                                                                OVERRIDE_X_INSIDE=x_inside_25x,
                                                                                                OVERRIDE_Y_INSIDE=y_inside_25x)

    # endregion

    # region OVERVIEW IMAGE
    if SAVE_OVERVIEW_IMG:
        if TRI_SCALE_PREPROCESSING is False:
            current_image_25x = Vips.Image.new_from_file(current_image_path, level=2)
            # Flatten image to remove alpha channel (only if SCN images)
            if filename[-3:] == 'scn':
                current_image_25x = current_image_25x.flatten()

            # Rotate image 90 degree (necessary because aperio imagescope does this, and the region coordinates need to match)
            current_image_25x = current_image_25x.rot(1)

            # Read the attributes from the metadata XML file
            x_inside_25x = int(metadata_root.find('metadata_doc/x_inside').attrib['a25x'])
            y_inside_25x = int(metadata_root.find('metadata_doc/y_inside').attrib['a25x'])

            # Remove white background
            if remove_white_background:
                current_image_25x, _, _ = MyFunctions.remove_white_background(input_img=current_image_25x,
                                                                              PADDING_AROUND_IMG_SIZE=PADDING_AROUND_IMG_SIZE,
                                                                              OVERRIDE_X_INSIDE=x_inside_25x,
                                                                              OVERRIDE_Y_INSIDE=y_inside_25x)
        fileName = '{}/{}#overview.jpeg'.format(image_path, filename[:-4])
        current_image_25x.jpegsave(fileName, Q=100)
    # endregion

    # region PREPROCESSING
    MyFunctions.myPrint('Starting applying binary mask on 400x image')

    # Read XML file, make a list with all X- and Y-coordinates
    xml_list = MyFunctions.readXML(path=xml_path,
                                   mask_width=original_img_width,
                                   padding=PADDING_AROUND_IMG_SIZE,
                                   REDUCE_TO_ONE_REGION_ONLY=REDUCE_TO_ONE_REGION_ONLY)

    # Process each region in the XML file
    for current_region_index, current_region in enumerate(xml_list):
        MyFunctions.myPrint('Processing region {} of {}'.format(current_region_index + 1, len(xml_list)))

        # region DRAW POLYGON AND FLOOD OUTSIDE REGION
        # Define some variable which are too large. We will then read the XML file and adjust these variables.
        crop_x_max = 0
        crop_x_min = 500000
        crop_y_max = 0
        crop_y_min = 500000

        # Variables to count number of tiles in each case
        region_count_tiles_saved = 0
        region_count_background_tiles_deleted = 0
        region_count_mask_tiles_deleted = 0

        # Flag to determine if drawing first element
        first_element = True

        # Find box inside image that fits around current region
        for coordinate_index in range(len(xml_list[current_region_index])):

            # Check Largest x-value
            if xml_list[current_region_index][coordinate_index][0] > crop_x_max:
                crop_x_max = xml_list[current_region_index][coordinate_index][0]

            # Check Smallest x-value
            if xml_list[current_region_index][coordinate_index][0] < crop_x_min:
                crop_x_min = xml_list[current_region_index][coordinate_index][0]

            # Check Largest y-value
            if xml_list[current_region_index][coordinate_index][1] > crop_y_max:
                crop_y_max = xml_list[current_region_index][coordinate_index][1]

            # Check Smallest y-value
            if xml_list[current_region_index][coordinate_index][1] < crop_y_min:
                crop_y_min = xml_list[current_region_index][coordinate_index][1]

        # Add some margin around
        crop_x_max = crop_x_max + REGION_MARGIN_SIZE
        crop_x_min = crop_x_min - REGION_MARGIN_SIZE
        crop_y_max = crop_y_max + REGION_MARGIN_SIZE
        crop_y_min = crop_y_min - REGION_MARGIN_SIZE

        # Adjust height/width to tile size
        region_width = math.ceil((crop_x_max - crop_x_min) / tile_size) * tile_size
        region_height = math.ceil((crop_y_max - crop_y_min) / tile_size) * tile_size

        # Crop image
        current_region_img_400x = current_image_400x.extract_area(crop_x_min, crop_y_min, region_width, region_height)

        MyFunctions.myPrint('\tCropped 400x image height:{} width:{}'.format(current_region_img_400x.height, current_region_img_400x.width))

        # We need to also "crop" the regions by subtracting the maximum x- and y-values form each coordinate
        cropped_xml = True
        for coordinate_index in range(len(xml_list[current_region_index])):
            xml_list[current_region_index][coordinate_index] = ((xml_list[current_region_index][coordinate_index][0] - crop_x_min),
                                                                (xml_list[current_region_index][coordinate_index][1] - crop_y_min))

        # Create a new binary mask filled with 1's
        MyFunctions.myPrint('\tCreating new black mask')
        vips_mask = Vips.Image.black(current_region_img_400x.width, current_region_img_400x.height)
        MyFunctions.myPrint('\tFlooding mask with 1s')
        vips_mask = vips_mask.draw_flood(1, 50, 50)

        # Draw lines between all coordinates in current region
        for coordinates in current_region:

            if first_element != True:
                # Draw a line from (last_x, last_y) to (x,y)
                vips_mask = vips_mask.draw_line(0.0, last_x, last_y, coordinates[0], coordinates[1])
            else:
                # If first time, set initial values and toggle flag
                first_element = False
                x_max = coordinates[0]
                x_min = coordinates[0]
                y_max = coordinates[1]
                y_min = coordinates[1]
                start_x = coordinates[0]
                start_y = coordinates[1]

            # Update last coordinates
            last_x = coordinates[0]
            last_y = coordinates[1]

            # Update max/min values
            if last_x > x_max:
                x_max = coordinates[0]
            elif last_x < x_min:
                x_min = coordinates[0]

            if last_y > y_max:
                y_max = coordinates[1]
            elif last_y < y_min:
                y_min = coordinates[1]

        # Reset first_element flag
        # first_element = True

        # Calculate flood x-y coordinates for current polygon
        mass_center_x = ((x_min + x_max) // 2)
        mass_center_y = ((y_min + y_max) // 2)

        if mass_center_x > start_x:
            # Calculate the angle
            flood_angle = (math.atan((mass_center_y - start_y) / (mass_center_x - start_x)))
        else:
            # Calculate the angle and add Pi to the value
            flood_angle = math.pi + (math.atan((mass_center_y - start_y) / (mass_center_x - start_x)))

        # Calculate the step-size to go away from start point and into the flood area.
        flood_length = current_region_img_400x.width // 100

        # Calculate the point that layes inside the polygon. The flood-function will start
        # flooding from this point and out towards the polygon border.
        flood_x = int(start_x + round(flood_length * math.cos(flood_angle), 0))
        flood_y = int(start_y + round(flood_length * math.sin(flood_angle), 0))

        # Flood the inside of the current polygon with 1s
        if KEEP_INSIDE_OF_REGION == True:
            # Start flooding with 0s from coordinate x=1, y=1.
            vips_mask = vips_mask.draw_flood(0, 1, 1)
        else:
            # Start flooding with 0s from coordinate x=flood_x, y=flood_y.
            vips_mask = vips_mask.draw_flood(0, flood_x, flood_y)

        # Draw the line that shows where to flood. Good for debugging when it floods wrong.
        # vips_mask = vips_mask.draw_line(200.0, start_x, start_y, flood_x, flood_y)

        # Apply binary mask on current image by multiplying the two together.
        # The two input images are cast up to the smallest common format. This means:
        # Before multiplying the image is uchar (0-255)
        # After multiplying the image is ushort (0-65535)
        # MyFunctions.myPrint('\tMultiplying current image with binary mask and casting image type to UCHAR')
        current_region_img_400x = current_region_img_400x.multiply(vips_mask)

        # We need to cast the image format back down to uchar.
        current_region_img_400x = current_region_img_400x.cast(0)

        if SAVE_REGION_IMG:
            current_image_after_mask_400x = current_region_img_400x
            # Resize image
            MyFunctions.myPrint('\tPreparing to save region img after binary mask is applied')
            if current_region_img_400x.width > 10000:
                current_image_after_mask_400x = current_image_after_mask_400x.resize(0.05)

            # Saving to saved folder
            fileName_400x = '{}{}/{}/{}#400x#binarymask_region{}.jpeg'.format(INPUT_ROOT_PATH, current_class, filename[:-4], os.path.splitext(filename)[0], filename[:-4], current_region_index + 1)
            current_image_after_mask_400x.jpegsave(fileName_400x)

        MyFunctions.myPrint('\tFinish applying binary mask on image. Starting splitting current region up in tiles.')
        # endregion

        # region INIT BEFORE EXTRACTING TILES
        # Calculate how many tiles there are room for in width/height. Define start (x,y) coordinates

        # Calculate stop and step size
        if OVERLAPPING_TILES:
            n_search_width = ((current_region_img_400x.width // tile_size) * tile_size) - 2 * tile_size
            n_search_height = ((current_region_img_400x.height // tile_size) * tile_size) - 2 * tile_size
            row_step = tile_size // 2
            column_step = tile_size // 2
        else:
            n_search_width = (current_region_img_400x.width // tile_size) * tile_size - tile_size
            n_search_height = (current_region_img_400x.height // tile_size) * tile_size - tile_size
            row_step = tile_size
            column_step = tile_size

        # Initial start coordinates
        start_x_coordinate = crop_x_min
        start_y_coordinate = crop_y_min
        best_start_x_coordinate = 0
        best_start_y_coordinate = 0

        if FIND_OPTIMAL_START_COORDINATES:

            xy_start_coordinate_high_score = 0

            temp_counter = 0

            if CLASS_NAME in ['Stroma', 'Muscle']:
                search_step_size = 8
            else:
                search_step_size = 32

            # Finding optimal start coordinates
            for current_start_y_coordinate in range(0, tile_size, search_step_size):
                for current_start_x_coordinate in range(0, tile_size, search_step_size):
                    # Reset counter
                    current_start_xy_save_counter = 0
                    temp_counter += 1
                    for current_row in range(current_start_y_coordinate, n_search_height, tile_size):
                        for current_column in range(current_start_x_coordinate, n_search_width, tile_size):

                            # Extract tile area of the image
                            current_tile_400x_with_mask = current_region_img_400x.extract_area(current_column, current_row, tile_size, tile_size)

                            # Find 10% threshold limit
                            current_tile_thresh = current_tile_400x_with_mask.percent(10)

                            # Check if tile contains tissue
                            if (20 <= current_tile_thresh <= 185):
                                # Update save counter
                                current_start_xy_save_counter += 1

                    # Check if current start coordinates are a new best
                    if current_start_xy_save_counter > xy_start_coordinate_high_score:
                        # Found new optimal set with start coordinates, save coordinates.
                        print('\tSearching {}/{} - Found {} tiles - new optimal set'.format(temp_counter, int((tile_size / search_step_size) ** 2), current_start_xy_save_counter))
                        best_start_x_coordinate = current_start_x_coordinate
                        best_start_y_coordinate = current_start_y_coordinate
                        xy_start_coordinate_high_score = current_start_xy_save_counter
                    else:
                        print('\tSearching {}/{} - Found {} tiles'.format(temp_counter, int((tile_size / search_step_size) ** 2), current_start_xy_save_counter))

        # endregion

        # region EXTRACT TILES
        # We will sort through the images two times. First time using 'search_size' and second
        # time using 'tile-size'. This is to sort out the background more efficiently.
        for current_row in range(best_start_y_coordinate, n_search_height, row_step):

            # Calculate current y-coordinate
            current_y_pos = start_y_coordinate + current_row

            MyFunctions.myPrint('\tExtracting tiles row {}/{}'.format(current_row, n_search_height))

            for current_column in range(best_start_x_coordinate, n_search_width, column_step):

                # Calculate current x-coordinate
                current_x_pos = start_x_coordinate + current_column

                # Extract tile area of the image
                current_tile_400x_with_mask = current_region_img_400x.extract_area(current_column, current_row, tile_size, tile_size)

                # Find 10% threshold limit
                current_tile_thresh = current_tile_400x_with_mask.percent(10)

                # If most of the pixel values are in the upper range of the image, it means that
                # most of the image consist of gray background. Delete image. Else, save image.
                # Check if tile consist of mostly binary mask. Save to mask_folder (if save_delete_img is True)
                if current_tile_thresh < 20:
                    # TILE IS BLACK MASK
                    region_count_mask_tiles_deleted += 1
                    # if save_delete_img:
                    # fileName_400x = '{}/{}#{}#{}#400x.jpg'.format(black_mask_tiles_path, os.path.splitext(filename)[0], current_y_pos, current_x_pos)
                    # current_tile_400x_with_mask.jpegsave(fileName_400x, Q=60)
                elif (current_tile_thresh < 185) and SAVE_BACKGROUND_AS_CLASS is False:
                    # TILE CONTAINS TISSUE, SAVE IT

                    # Calculate coordinate of the middle-pixel value of the tile on 400x image
                    image_400x_tile_center_x = current_x_pos + (tile_size / 2) + REGION_MARGIN_SIZE
                    image_400x_tile_center_y = current_y_pos + (tile_size / 2) + REGION_MARGIN_SIZE

                    # Extract the tile from the 400x image (transform from middle-pixel, to top-left pixel coordinate)
                    if SAVE_FORMAT in ['JPEG', 'Jpeg', 'jpeg']:
                        current_tile_400x = current_image_400x.extract_area(image_400x_tile_center_x - (tile_size / 2),
                                                                            image_400x_tile_center_y - (tile_size / 2),
                                                                            tile_size, tile_size)

                        MyFunctions.save_tile_jpeg(tile=current_tile_400x,
                                                   x_pos=x_offset_400x + image_400x_tile_center_x - (tile_size / 2),
                                                   y_pos=y_offset_400x + image_400x_tile_center_y - (tile_size / 2),
                                                   cropped_xml=cropped_xml,
                                                   magnification_scale='400x',
                                                   jpeg_save_path=saved_tiles_path_400x,
                                                   wsi_filename=filename,
                                                   crop_x_min=crop_x_min,
                                                   REGION_MARGIN_SIZE=REGION_MARGIN_SIZE)

                    # Transform the coordinate from the 400x image to the 100x image
                    if DI_SCALE_PREPROCESSING is True or TRI_SCALE_PREPROCESSING is True:
                        image_100x_tile_center_x = (image_400x_tile_center_x * X_scale_400x_100x)
                        image_100x_tile_center_y = (image_400x_tile_center_y * Y_scale_400x_100x)

                        # Extract the tile from the 100x image (transform from middle-pixel, to top-left pixel coordinate)
                        if SAVE_FORMAT in ['JPEG', 'Jpeg', 'jpeg']:
                            current_tile_100x = current_image_100x.extract_area(image_100x_tile_center_x - (tile_size / 2),
                                                                                image_100x_tile_center_y - (tile_size / 2),
                                                                                tile_size, tile_size)

                            MyFunctions.save_tile_jpeg(tile=current_tile_100x,
                                                       x_pos=x_offset_100x + image_100x_tile_center_x - (tile_size / 2),
                                                       y_pos=y_offset_100x + image_100x_tile_center_y - (tile_size / 2),
                                                       cropped_xml=cropped_xml,
                                                       magnification_scale='100x',
                                                       jpeg_save_path=saved_tiles_path_100x,
                                                       wsi_filename=filename,
                                                       crop_x_min=crop_x_min,
                                                       REGION_MARGIN_SIZE=REGION_MARGIN_SIZE)

                    # Transform the coordinate from the 400x image to the 25x image
                    if TRI_SCALE_PREPROCESSING is True:
                        image_25x_tile_center_x = (image_400x_tile_center_x * X_scale_400x_25x)
                        image_25x_tile_center_y = (image_400x_tile_center_y * Y_scale_400x_25x)

                        # Extract the tile from the 25x image (transform from middle-pixel, to top-left pixel coordinate)
                        if SAVE_FORMAT in ['JPEG', 'Jpeg', 'jpeg']:
                            current_tile_25x = current_image_25x.extract_area(image_25x_tile_center_x - (tile_size / 2),
                                                                              image_25x_tile_center_y - (tile_size / 2),
                                                                              tile_size, tile_size)

                            MyFunctions.save_tile_jpeg(tile=current_tile_25x,
                                                       x_pos=x_offset_25x + image_25x_tile_center_x - (tile_size / 2),
                                                       y_pos=y_offset_25x + image_25x_tile_center_y - (tile_size / 2),
                                                       cropped_xml=cropped_xml,
                                                       magnification_scale='25x',
                                                       jpeg_save_path=saved_tiles_path_25x,
                                                       wsi_filename=filename,
                                                       crop_x_min=crop_x_min,
                                                       REGION_MARGIN_SIZE=REGION_MARGIN_SIZE)

                    # Save tiles
                    if SAVE_FORMAT in ['COORDINATES', 'Coordinates', 'coordinates']:
                        MyFunctions.save_tile_coordinate(tile_dict=current_list_of_all_tiles_dict,
                                                         label=CLASS_NAME,
                                                         x_pos_400x=x_offset_400x + image_400x_tile_center_x - (tile_size / 2),
                                                         y_pos_400x=y_offset_400x + image_400x_tile_center_y - (tile_size / 2),
                                                         x_pos_100x=x_offset_100x + image_100x_tile_center_x - (tile_size / 2),
                                                         y_pos_100x=y_offset_100x + image_100x_tile_center_y - (tile_size / 2),
                                                         x_pos_25x=x_offset_25x + image_25x_tile_center_x - (tile_size / 2),
                                                         y_pos_25x=y_offset_25x + image_25x_tile_center_y - (tile_size / 2),
                                                         wsi_filename=filename)

                    # Update save counter
                    region_count_tiles_saved += 1

                    # Break out if early stopping is enabled
                    if isinstance(ONLY_EXTRACT_N_TILES, int):
                        if region_count_tiles_saved >= ONLY_EXTRACT_N_TILES:
                            break
                else:
                    # TILE IS BACKGROUND
                    if save_delete_img:

                        # Check if should save tile in deleted_tiles_path or saved_tiles_path_400x
                        if SAVE_BACKGROUND_AS_CLASS is False:
                            fileName_400x = '{}/{}#{}#{}#400x.jpg'.format(deleted_tiles_path, os.path.splitext(filename)[0], current_y_pos, current_x_pos)
                            current_tile_400x_with_mask.jpegsave(fileName_400x, Q=60)
                            region_count_background_tiles_deleted += 1
                        elif SAVE_BACKGROUND_AS_CLASS is True:
                            # Calculate coordinate of the middle-pixel value of the tile on 400x image
                            image_400x_tile_center_x = current_x_pos + (tile_size / 2) + REGION_MARGIN_SIZE
                            image_400x_tile_center_y = current_y_pos + (tile_size / 2) + REGION_MARGIN_SIZE

                            # Extract the tile from the 25x image (transform from middle-pixel, to top-left pixel coordinate)
                            if SAVE_FORMAT in ['JPEG', 'Jpeg', 'jpeg']:
                                # Extract the tile from the 400x image (transform from middle-pixel, to top-left pixel coordinate)
                                current_tile_400x = current_image_400x.extract_area(image_400x_tile_center_x - (tile_size / 2),
                                                                                    image_400x_tile_center_y - (tile_size / 2),
                                                                                    tile_size, tile_size)

                                MyFunctions.save_tile_jpeg(tile=current_tile_400x,
                                                           x_pos=x_offset_400x + image_400x_tile_center_x - (tile_size / 2),
                                                           y_pos=y_offset_400x + image_400x_tile_center_y - (tile_size / 2),
                                                           cropped_xml=cropped_xml,
                                                           magnification_scale='400x',
                                                           jpeg_save_path=saved_tiles_path_400x,
                                                           wsi_filename=filename,
                                                           crop_x_min=crop_x_min,
                                                           REGION_MARGIN_SIZE=REGION_MARGIN_SIZE)

                            if DI_SCALE_PREPROCESSING is True or TRI_SCALE_PREPROCESSING is True:
                                # Transform the coordinate from the 400x image to the 100x image
                                image_100x_tile_center_x = (image_400x_tile_center_x * X_scale_400x_100x)
                                image_100x_tile_center_y = (image_400x_tile_center_y * Y_scale_400x_100x)

                                if SAVE_FORMAT in ['JPEG', 'Jpeg', 'jpeg']:
                                    # Extract the tile from the 100x image (transform from middle-pixel, to top-left pixel coordinate)
                                    current_tile_100x = current_image_100x.extract_area(image_100x_tile_center_x - (tile_size / 2),
                                                                                        image_100x_tile_center_y - (tile_size / 2),
                                                                                        tile_size,
                                                                                        tile_size)

                                    MyFunctions.save_tile_jpeg(tile=current_tile_100x,
                                                               x_pos=x_offset_100x + image_100x_tile_center_x - (tile_size / 2),
                                                               y_pos=y_offset_100x + image_100x_tile_center_y - (tile_size / 2),
                                                               cropped_xml=cropped_xml,
                                                               magnification_scale='100x',
                                                               jpeg_save_path=saved_tiles_path_100x,
                                                               wsi_filename=filename,
                                                               crop_x_min=crop_x_min,
                                                               REGION_MARGIN_SIZE=REGION_MARGIN_SIZE)

                            if TRI_SCALE_PREPROCESSING is True:
                                # Transform the coordinate from the 400x image to the 25x image
                                image_25x_tile_center_x = (image_400x_tile_center_x * X_scale_400x_25x)
                                image_25x_tile_center_y = (image_400x_tile_center_y * Y_scale_400x_25x)

                                # Extract the tile from the 25x image (transform from middle-pixel, to top-left pixel coordinate)
                                if SAVE_FORMAT in ['JPEG', 'Jpeg', 'jpeg']:
                                    current_tile_25x = current_image_25x.extract_area(image_25x_tile_center_x - (tile_size / 2),
                                                                                      image_25x_tile_center_y - (tile_size / 2),
                                                                                      tile_size, tile_size)

                                    MyFunctions.save_tile_jpeg(tile=current_tile_25x,
                                                               x_pos=x_offset_25x + image_25x_tile_center_x - (tile_size / 2),
                                                               y_pos=y_offset_25x + image_25x_tile_center_y - (tile_size / 2),
                                                               cropped_xml=cropped_xml,
                                                               magnification_scale='25x',
                                                               jpeg_save_path=saved_tiles_path_25x,
                                                               wsi_filename=filename,
                                                               crop_x_min=crop_x_min,
                                                               REGION_MARGIN_SIZE=REGION_MARGIN_SIZE)

                            # Save tiles
                            if SAVE_FORMAT in ['COORDINATES', 'Coordinates', 'coordinates']:
                                MyFunctions.save_tile_coordinate(tile_dict=current_list_of_all_tiles_dict,
                                                                 label=CLASS_NAME,
                                                                 x_pos_400x=x_offset_400x + image_400x_tile_center_x - (tile_size / 2),
                                                                 y_pos_400x=y_offset_400x + image_400x_tile_center_y - (tile_size / 2),
                                                                 x_pos_100x=x_offset_100x + image_100x_tile_center_x - (tile_size / 2),
                                                                 y_pos_100x=y_offset_100x + image_100x_tile_center_y - (tile_size / 2),
                                                                 x_pos_25x=x_offset_25x + image_25x_tile_center_x - (tile_size / 2),
                                                                 y_pos_25x=y_offset_25x + image_25x_tile_center_y - (tile_size / 2),
                                                                 wsi_filename=filename)

                        # Break out if early stopping is enabled
                        if isinstance(ONLY_EXTRACT_N_TILES, int):
                            if region_count_background_tiles_deleted >= ONLY_EXTRACT_N_TILES:
                                break

                    # Update counter
                    region_count_background_tiles_deleted += 1

            # Break out if early stopping is enabled
            if isinstance(ONLY_EXTRACT_N_TILES, int):
                if (region_count_tiles_saved >= ONLY_EXTRACT_N_TILES) or (region_count_background_tiles_deleted >= ONLY_EXTRACT_N_TILES):
                    print('Early stopping, breaking out of loop.')
                    break

        # Variables to count total number of tiles in each case
        count_tiles_saved += region_count_tiles_saved
        count_background_tiles_deleted += region_count_background_tiles_deleted
        count_mask_tiles_deleted += region_count_mask_tiles_deleted

        MyFunctions.myPrint('Tiles saved: {}'.format(region_count_tiles_saved))
        MyFunctions.myPrint('Tiles deleted: {}'.format(region_count_background_tiles_deleted))
        MyFunctions.myPrint('Masked tile deleted: {}'.format(region_count_mask_tiles_deleted))
        # endregion

    # endregion

    # region FINAL CALCULATIONS
    # Calculate total time for preprocessing
    total_time = time.time() - main_timer_start
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)

    total_runtime = '%02d:%02d:%02d' % (h, m, s)

    # Save coordinates array to pickle
    pickle_filename = '{}{}_pickle.obj'.format(SAVE_ROOT_PATH, filename_no_extension)
    if not os.path.isfile(pickle_filename):
        # Save as new pickle file
        pickle_writer = open(pickle_filename, 'wb')
        pickle.dump(current_list_of_all_tiles_dict, pickle_writer)
        pickle_writer.close()
    else:
        # Update existing dict
        pickle_reader = open(pickle_filename, 'rb')
        list_of_all_tiles_dict = pickle.load(pickle_reader)
        pickle_reader.close()

        # Find last index value
        last_dict_index = len(list_of_all_tiles_dict)

        # Add data to main dict
        for index, value in current_list_of_all_tiles_dict.items():
            list_of_all_tiles_dict[last_dict_index + index] = value

        # Save as new pickle file
        pickle_writer = open(pickle_filename, 'wb')
        pickle.dump((list_of_all_tiles_dict), pickle_writer)
        pickle_writer.close()

    # Write result to summary.csv file
    try:
        with open(SUMMARY_FILE, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([filename, count_mask_tiles_deleted, count_background_tiles_deleted, count_tiles_saved, CLASS_NAME, total_runtime,
                                 REGION_MARGIN_SIZE, tile_size, OVERLAPPING_TILES, save_delete_img, KEEP_INSIDE_OF_REGION,
                                 DI_SCALE_PREPROCESSING, TRI_SCALE_PREPROCESSING])
    except Exception as e:
        MyFunctions.myPrint('Error writing to file', error=True)
        MyFunctions.myPrint(e, error=True)
    # endregion
