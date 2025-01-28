import os
import json

"""
.. module:: streamlit_img_label
   :synopsis: annotation.
.. moduleauthor:: Tianning Li <ltianningli@gmail.com>
"""

def read_json(json_path):
    """Reads the JSON annotation file and extracts bounding boxes.

    Args:
        json_path (str): The full path to the JSON file.

    Returns:
        list: The bounding boxes of the image, or an empty list if no JSON is found.
    """
    if not os.path.isfile(json_path):
        print(f"JSON file not found: {json_path}")
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get("rects", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return []


def output_json(img_file, img, rects):
    """output_json
    Output the json image annotation file

    Args:
        img_file(str): the image file.
        img(PIL.Image): the image object.
        rects(list): the bounding boxes of the image.
    """
    file_name = os.path.splitext(img_file)[0]
    output_data = {
        "image_width": img.width,
        "image_height": img.height,
        "rects": rects
    }
    with open(f"{file_name}.json", "w") as f:
        json.dump(output_data, f, indent=4)
