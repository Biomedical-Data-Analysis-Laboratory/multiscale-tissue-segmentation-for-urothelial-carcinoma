import MyFunctions
from xml.etree import ElementTree
import pickle
import os

""" Some utilities which may be useful """


def offset_calibration_routine(wsi_calibration_point_x, wsi_calibration_point_y, offset_400x_x, offset_400x_y, offset_100x_x, offset_100x_y, offset_25x_x, offset_25x_y,
                               X_scale_400x_100x, Y_scale_400x_100x, X_scale_400x_25x, Y_scale_400x_25x, current_image_400x, current_image_100x, current_image_25x,
                               metadata_path, padding):
    MyFunctions.myPrint('Starting Offset Calibration')

    # Size of extracted window, and also line draw length
    extract_width = 300
    extract_height = 300

    # Start X,Y coordinates of extracted window
    extraxt_x_400x = int(wsi_calibration_point_x - (extract_width // 2) + offset_400x_x + padding)
    extraxt_y_400x = int(wsi_calibration_point_y - (extract_height // 2) + offset_400x_y + padding)
    extraxt_x_100x = int((wsi_calibration_point_x * X_scale_400x_100x) - (extract_width // 2) + offset_100x_x + padding)
    extraxt_y_100x = int((wsi_calibration_point_y * Y_scale_400x_100x) - (extract_height // 2) + offset_100x_y + padding)
    extraxt_x_25x = int(wsi_calibration_point_x * X_scale_400x_25x - (extract_width // 2) + offset_25x_x + padding)
    extraxt_y_25x = int(wsi_calibration_point_y * Y_scale_400x_25x - (extract_height // 2) + offset_25x_y + padding)

    # Extract tiles
    MyFunctions.myPrint("Extracting tile from ({},{})".format((extraxt_x_400x), (extraxt_y_400x)))
    current_image_400x_tile = current_image_400x.extract_area(extraxt_x_400x, extraxt_y_400x, extract_width, extract_width)

    MyFunctions.myPrint("Extracting tile from ({},{})".format(extraxt_x_100x, extraxt_y_100x))
    current_image_100x_tile = current_image_100x.extract_area(extraxt_x_100x, extraxt_y_100x, extract_width, extract_height)

    MyFunctions.myPrint("Extracting tile from ({},{})".format(extraxt_x_25x, extraxt_y_25x))
    current_image_25x_tile = current_image_25x.extract_area(extraxt_x_25x, extraxt_y_25x, extract_width, extract_height)

    # Coordinate of draw line cross
    start_horizontal_x_40 = 0
    start_horizontal_y_40 = extract_height // 2
    end_horizontal_x_40 = extract_width
    end_horizontal_y_40 = extract_height // 2

    start_vertical_x_40 = extract_width // 2
    start_vertical_y_40 = 0
    end_vertical_x_40 = extract_width // 2
    end_vertical_y_40 = extract_height

    start_horizontal_x_10 = 0
    start_horizontal_y_10 = extract_height // 2
    end_horizontal_x_10 = extract_width
    end_horizontal_y_10 = extract_height // 2

    start_vertical_x_10 = extract_width // 2
    start_vertical_y_10 = 0
    end_vertical_x_10 = extract_width // 2
    end_vertical_y_10 = extract_height

    start_horizontal_x_2 = 0
    start_horizontal_y_2 = extract_height // 2
    end_horizontal_x_2 = extract_width
    end_horizontal_y_2 = extract_height // 2

    start_vertical_x_2 = extract_width // 2
    start_vertical_y_2 = 0
    end_vertical_x_2 = extract_width // 2
    end_vertical_y_2 = extract_height

    # Draw horizontal line in 400x WSI cross
    MyFunctions.myPrint("Drawing line from ({},{}) to ({},{})".format(start_horizontal_x_40, start_horizontal_y_40, end_horizontal_x_40, end_horizontal_y_40))
    current_image_400x_tile = current_image_400x_tile.draw_line(0.0, start_horizontal_x_40, start_horizontal_y_40, end_horizontal_x_40, end_horizontal_y_40)

    # Draw vertical line in 400x WSI cross
    MyFunctions.myPrint("Drawing line from ({},{}) to ({},{})".format(start_vertical_x_40, start_vertical_y_40, end_vertical_x_40, end_vertical_y_40))
    current_image_400x_tile = current_image_400x_tile.draw_line(0.0, start_vertical_x_40, start_vertical_y_40, end_vertical_x_40, end_vertical_y_40)

    # Draw horizontal line in 100x WSI cross
    MyFunctions.myPrint("Drawing line from ({},{}) to ({},{})".format(start_horizontal_x_10, start_horizontal_y_10, end_horizontal_x_10, end_horizontal_y_10))
    current_image_100x_tile = current_image_100x_tile.draw_line(0.0, start_horizontal_x_10, start_horizontal_y_10, end_horizontal_x_10, end_horizontal_y_10)

    # Draw vertical line in 100x WSI cross
    MyFunctions.myPrint("Drawing line from ({},{}) to ({},{})".format(start_vertical_x_10, start_vertical_y_10, end_vertical_x_10, end_vertical_y_10))
    current_image_100x_tile = current_image_100x_tile.draw_line(0.0, start_vertical_x_10, start_vertical_y_10, end_vertical_x_10, end_vertical_y_10)

    # Draw horizontal line in 25x WSI cross
    MyFunctions.myPrint("Drawing line from ({},{}) to ({},{})".format(start_horizontal_x_2, start_horizontal_y_2, end_horizontal_x_2, end_horizontal_y_2))
    current_image_25x_tile = current_image_25x_tile.draw_line(0.0, start_horizontal_x_2, start_horizontal_y_2, end_horizontal_x_2, end_horizontal_y_2)

    # Draw vertical line in 25x WSI cross
    MyFunctions.myPrint("Drawing line from ({},{}) to ({},{})".format(start_vertical_x_2, start_vertical_y_2, end_vertical_x_2, end_vertical_y_2))
    current_image_25x_tile = current_image_25x_tile.draw_line(0.0, start_vertical_x_2, start_vertical_y_2, end_vertical_x_2, end_vertical_y_2)

    fileName_400x = 'Calibrate_400x#x-offset_{}#y-offset_{}.jpeg'.format(offset_400x_x, offset_400x_y)
    fileName_100x = 'Calibrate_100x#x-offset_{}#y-offset_{}.jpeg'.format(offset_100x_x, offset_100x_y)
    fileName_25x = 'Calibrate_25x#x-offset_{}#y-offset_{}.jpeg'.format(offset_25x_x, offset_25x_y)
    current_image_400x_tile.jpegsave(fileName_400x, Q=100)
    current_image_100x_tile.jpegsave(fileName_100x, Q=100)
    current_image_25x_tile.jpegsave(fileName_25x, Q=100)

    MyFunctions.myPrint("Center point in WSI is ({},{})".format(wsi_calibration_point_x, wsi_calibration_point_y))
    MyFunctions.myPrint("Center point in 400x is ({},{})".format(start_horizontal_x_40 + (extract_width / 2), start_horizontal_y_40 + (extract_width / 2)))
    MyFunctions.myPrint("Center point in 100x is ({},{})".format(start_horizontal_x_10 + (extract_width / 2), start_horizontal_y_10 + (extract_width / 2)))
    MyFunctions.myPrint("Center point in 25x is ({},{})".format(start_horizontal_x_2 + (extract_width / 2), start_horizontal_y_2 + (extract_width / 2)))

    # Save image height/width to metadata file. Parse the root of the XML file structure
    metadata_root = ElementTree.parse(metadata_path)

    # Find the correct attribute to update
    current_attribute = metadata_root.find('metadata_doc/offset')
    current_attribute.set('x_400x', str(offset_400x_x))
    current_attribute.set('y_400x', str(offset_400x_y))
    current_attribute.set('x_100x', str(offset_100x_x))
    current_attribute.set('y_100x', str(offset_100x_y))

    # Save the XML document
    metadata_root.write(metadata_path)


def COMBINE_DICT_FILES(ALL_DICTS_TO_BE_COMBINED_PATH, READ_ONLY=True):
    #  Create a new dict to rule them all
    new_main_dict = dict()

    # Find all existing dicts inside folder
    all_dicts_list = os.listdir(ALL_DICTS_TO_BE_COMBINED_PATH)

    # Only print content of each dict to console, not saving anything.
    if READ_ONLY:
        dict_counter = dict()
        # Loop through each dict, restore them and add them to the new main dict
        for my_dict in all_dicts_list:
            dict_counter.clear()
            # Restore dict data
            pickle_reader = open(ALL_DICTS_TO_BE_COMBINED_PATH + my_dict, 'rb')
            list_of_all_tiles_dict = pickle.load(pickle_reader)
            pickle_reader.close()

            for index, value in list_of_all_tiles_dict.items():
                # print(value)
                if value['label'] in dict_counter:
                    # Key in dict
                    dict_counter[value['label']] += 1
                else:
                    # key not in dict
                    dict_counter[value['label']] = 1
            print('')
            print(my_dict)
            for class_name, n_value in dict_counter.items():
                print('\t{} - {} tiles'.format(class_name, n_value))
        exit()

    # Loop through each dict, restore them and add them to the new main dict
    for my_dict in all_dicts_list:
        # Restore dict data
        pickle_reader = open(ALL_DICTS_TO_BE_COMBINED_PATH + my_dict, 'rb')
        list_of_all_tiles_dict = pickle.load(pickle_reader)
        pickle_reader.close()

        # Find last index value
        last_dict_index = len(new_main_dict)

        # Add data to main dict
        for index, value in list_of_all_tiles_dict.items():
            # if not value['label'] == 'Stroma':
            new_main_dict[last_dict_index + index] = value

    # Save as new pickle file
    pickle_writer = open('combine_dicts/new_main_dict_pickle.obj', 'wb')
    pickle.dump(new_main_dict, pickle_writer)
    pickle_writer.close()
