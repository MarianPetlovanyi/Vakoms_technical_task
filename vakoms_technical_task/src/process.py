import pandas as pd
import os
import xml.etree.ElementTree as et 


def parse_horizontal_xml(path_to_xml):
    '''
    Parse information from a horizontal bounding box XML file.

    Parameters:
        path_to_xml (str): The path to the XML file to be parsed.

    Returns:
        metadata (list): A list containing image metadata including filename, database, width, height, depth, and segmented.
        object_data (list): A list of tuples containing object information, such as name, xmin, ymin, xmax, and ymax.

    Example:
        To parse an XML file and retrieve image metadata and object data:
        metadata, object_data = parse_horizontal_xml('path_to_xml_file.xml')
    '''
    xtree = et.parse(path_to_xml)
    xroot = xtree.getroot()
    
    filename = []
    database = []
    width = []
    height = []
    depth = []
    segmented = []

    object_data = []
    
    filename.append(xroot.find("filename").text)
    database.append(xroot.find(".//source/database").text)
    width.append(int(xroot.find(".//size/width").text))
    height.append(int(xroot.find(".//size/height").text))
    depth.append(int(xroot.find(".//size/depth").text))
    segmented.append(int(xroot.find("segmented").text))
    
    for obj in xroot.findall(".//object"):
        object_name = obj.find("name").text
        xmin = int(obj.find(".//bndbox/xmin").text)
        ymin = int(obj.find(".//bndbox/ymin").text)
        xmax = int(obj.find(".//bndbox/xmax").text)
        ymax = int(obj.find(".//bndbox/ymax").text)
        object_data.append((object_name, xmin, ymin, xmax, ymax))
    
    return [filename, database, width, height, depth, segmented], object_data

def parse_oriented_xml(path_to_xml):
    '''
    Parse information from an oriented bounding box XML file.

    Parameters:
        path_to_xml (str): The path to the XML file to be parsed.

    Returns:
        metadata (list): A list containing image metadata including filename, database, width, height, depth, and segmented.
        object_data (list): A list of tuples containing object information, such as name and oriented bounding box coordinates.

    Example:
        To parse an XML file and retrieve image metadata and object data:
        metadata, object_data = parse_oriented_xml('path_to_xml_file.xml')
    '''
    
    xtree = et.parse(path_to_xml)
    xroot = xtree.getroot()
    
    filename = []
    database = []
    width = []
    height = []
    depth = []
    segmented = []

    object_data = []
    
    filename.append(xroot.find("filename").text)
    database.append(xroot.find(".//source/database").text)
    width.append(int(xroot.find(".//size/width").text))
    height.append(int(xroot.find(".//size/height").text))
    depth.append(int(xroot.find(".//size/depth").text))
    segmented.append(int(xroot.find("segmented").text))


    
    for obj in xroot.findall(".//object"):
        object_name = obj.find("name").text
        x_left_top = float(obj.find(".//robndbox/x_left_top").text)
        x_right_top = float(obj.find(".//robndbox/x_right_top").text)
        y_left_top = float(obj.find(".//robndbox/y_left_top").text)
        y_right_top = float(obj.find(".//robndbox/y_right_top").text)
        x_left_bottom = float(obj.find(".//robndbox/x_left_bottom").text)
        x_right_bottom = float(obj.find(".//robndbox/x_right_bottom").text)
        y_left_bottom = float(obj.find(".//robndbox/y_left_bottom").text)
        y_right_bottom = float(obj.find(".//robndbox/y_right_bottom").text)
        object_data.append((object_name, x_left_top, y_left_top, x_right_top, y_right_top,
                             x_right_bottom,  y_right_bottom, x_left_bottom, y_left_bottom,))
    
    return [filename, database, width, height, depth, segmented], object_data



def process_data():
    '''
    Process datasets annotation into csv files.
    '''
    
    annotations_h_path = "../data/raw/military-aircraft-recognition-dataset\Annotations\Horizontal Bounding Boxes"
    annotations_o_path = "../data/raw/military-aircraft-recognition-dataset\Annotations\Oriented Bounding Boxes"

    xml_files_h = [os.path.join(annotations_h_path, file) for file in os.listdir(annotations_h_path) if file.endswith('.xml')]
    xml_files_o = [os.path.join(annotations_o_path, file) for file in os.listdir(annotations_o_path) if file.endswith('.xml')]

    data = []
    for file in xml_files_h:
        image_data, objects = parse_horizontal_xml(file)
        path = os.path.join(annotations_h_path, os.path.splitext(image_data[0][0])[0]+'.jpg')
        for object in objects:
            data_sample = {'name': path, 'class':object[0],
                        'xmin': object[1], 'ymin': object[2], 'xmax': object[3], 'ymax': object[4]}
            data.append(data_sample)
    df_h = pd.DataFrame(data)


    data = []
    for file in xml_files_o:
        image_data, objects = parse_oriented_xml(file)
        path = os.path.join(annotations_o_path, os.path.splitext(image_data[0][0])[0]+'.jpg')
        for object in objects:
            data_sample = {'name': path, 'class':object[0],
                        'x_left_top': object[1], 'y_left_top': object[2], 'x_right_top': object[3], 'y_right_top': object[4],
                        'x_left_bottom': object[5], 'y_left_bottom': object[6], 'x_right_bottom': object[7], 'y_right_bottom': object[8]}
            data.append(data_sample)
    df_o = pd.DataFrame(data)

    df_h.to_csv("../data/processed/horizontal.csv")
    df_o.to_csv("../data/processed/oriented.csv")


if __name__ == "__main__":
    process_data()
