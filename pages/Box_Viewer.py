import streamlit as st
import os
import json
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
import numpy as np
from PIL import Image

# --- Configuration ---
IMG_DIR = "/Users/stuartgano/Downloads/Streamlit/streamlit-img-label/temp_pdf_pages"  # Update with your image directory
COLLECTIONS_DIR = os.environ.get(
    "COLLECTIONS_DIR",
    "/Users/stuartgano/Downloads/Streamlit/streamlit-img-label/pages/streamlit_img_label/collections",
)
DEFAULT_LABEL = "Unlabeled"
ALL_LABELS = [
    DEFAULT_LABEL,
    "Client Initials",
    "Account Name",
    "Client Signature",
    "Advisor Signature",
    "Account Number",
    "Fee to be Charged",
    "Additional Notes",
    "Client Profile Owner 1",
    "Client Profile Owner 2",
    "Investment Experience Owner 1",
    "Investment Experience Owner 2",
    "Account Profile Page 7",
    "Additional Details Page 7",
    "Page 5 Checkboxes",
    "Page 5 Email 1",
    "Page 5 Email 2",
    "Page 5 Email 3",
    "Page 5 Email 4",
    "Page 5 Client Initials",
]

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(COLLECTIONS_DIR, exist_ok=True)

# --- Hardcoded Template Data (Corrected for Pixel Coordinates) ---
TEMPLATE_DATA = {
    "page_1.png": {
        "image_path": os.path.join(IMG_DIR, "page_1.png"),
        "rects": [
            {
                "left": 950,
                "top": 70,
                "width": 200,
                "height": 50,
                "label": "Account Number"
            },
            {
                "left": 500,
                "top": 300,
                "width": 700,
                "height": 70,
                "label": "Account Name"
            },
            {
                "left": 50,
                "top": 1000,
                "width": 1200,
                "height": 150,
                "label": "Client Initials"
            }
        ]
    },
    "page_2.png": {
        "image_path": os.path.join(IMG_DIR, "page_2.png"),
        "rects": [
            {
                "left": 90,
                "top": 200,
                "width": 1100,
                "height": 125,
                "label": "Client Initials"
            }
        ]
    },
    "page_5.png": {
        "image_path": os.path.join(IMG_DIR, "page_5.png"),
        "rects": [
            {
                "left": 100,
                "top": 1250,
                "width": 1000,
                "height": 300,
                "label": "Page 5 Client Initials"
            },
            {
                "left": 100,
                "top": 600,
                "width": 500,
                "height": 310,
                "label": "Page 5 Checkboxes"
            },
            {
                "left": 100,
                "top": 950,
                "width": 500,
                "height": 100,
                "label": "Page 5 Email 1"
            },
            {
                "left": 700,
                "top": 950,
                "width": 500,
                "height": 100,
                "label": "Page 5 Email 2"
            },
            {
                "left": 100,
                "top": 1000,
                "width": 500,
                "height": 100,
                "label": "Page 5 Email 3"
            },
            {
                "left": 700,
                "top": 1000,
                "width": 500,
                "height": 100,
                "label": "Page 5 Email 4"
            }
        ]
    },
    "page_7.png": {
        "image_path": os.path.join(IMG_DIR, "page_7.png"),
        "rects": [
            {
                "left": 75,
                "top": 300,
                "width": 500,
                "height": 250,
                "label": "Client Profile Owner 1"
            },
            {
                "left": 600,
                "top": 300,
                "width": 500,
                "height": 250,
                "label": "Client Profile Owner 2"
            },
            {
                "left": 75,
                "top": 550,
                "width": 550,
                "height": 340,
                "label": "Investment Experience Owner 1"
            },
            {
                "left": 600,
                "top": 550,
                "width": 550,
                "height": 340,
                "label": "Investment Experience Owner 2"
            },
            {
                "left": 95,
                "top": 900,
                "width": 1100,
                "height": 370,
                "label": "Account Profile Page 7"
            },
            {
                "left": 95,
                "top": 1200,
                "width": 1200,
                "height": 350,
                "label": "Additional Details Page 7"
            }
        ]
    },
    "page_10.png": {
        "image_path": os.path.join(IMG_DIR, "page_10.png"),
        "rects": [
            {
                "left": 200,
                "top": 1200,
                "width": 1000,
                "height": 300,
                "label": "Client Initials"
            },
            {
                "left": 100,
                "top": 350,
                "width": 450,
                "height": 165,
                "label": "Fee to be Charged"
            },
            {
                "left": 100,
                "top": 550,
                "width": 975,
                "height": 200,
                "label": "Additional Notes"
            }
        ]
    },
    "page_12.png": {
        "image_path": os.path.join(IMG_DIR, "page_12.png"),
        "rects": [
            {
                "left": 100,
                "top": 300,
                "width": 1000,
                "height": 145,
                "label": "Advisor Signature"
            },
            {
                "left": 100,
                "top": 550,
                "width": 1000,
                "height": 145,
                "label": "Client Signature"
            },
            {
                "left": 100,
                "top": 800,
                "width": 1000,
                "height": 145,
                "label": "Client Signature"
            },
            {
                "left": 100,
                "top": 1000,
                "width": 1000,
                "height": 145,
                "label": "Client Signature"
            },
            {
                "left": 100,
                "top": 1200,
                "width": 1000,
                "height": 145,
                "label": "Client Signature"
            }
        ]
    }
}


def go_to_image():
    st.session_state["image_index"] = st.session_state["files"].index(
        st.session_state["file"]
    )


def previous_image():
    st.session_state["image_index"] = max(0, st.session_state["image_index"] - 1)


def next_image():
    st.session_state["image_index"] = min(
        len(st.session_state["files"]) - 1, st.session_state["image_index"] + 1
    )


def next_annotate_file():
    all_files = set(st.session_state["files"])
    annotate_files = set(st.session_state["annotation_files"])
    need_annotate_files = all_files - annotate_files

    if need_annotate_files:
        next_annotate_index = min(
            i
            for i, file in enumerate(st.session_state["files"])
            if file in need_annotate_files
        )
        st.session_state["image_index"] = next_annotate_index
    else:
        st.write("All files are annotated!")

# --- Streamlit App ---
def run():
    idm = ImageDirManager(IMG_DIR)

    # Initialize session state
    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
        st.session_state["annotation_data"] = {}

    # --- Refresh Logic ---
    st.session_state["files"] = idm.get_all_files()
    st.session_state["annotation_files"] = idm.get_exist_annotation_files()

    # --- Main Content: Annotate Images ---
    if st.session_state["image_index"] < len(st.session_state["files"]):
        img_file_name = idm.get_image(st.session_state["image_index"])
        img_path = os.path.join(IMG_DIR, img_file_name)
        im = ImageManager(img_path)
        resized_img = im.resizing_img()
        
        # Load image using PIL to get dimensions
        pil_img = Image.open(img_path)
        img_width, img_height = pil_img.size

        # Check for pre-existing data in TEMPLATE_DATA
        if img_file_name in TEMPLATE_DATA:
            existing_rects = TEMPLATE_DATA[img_file_name]["rects"]
        else:
            existing_rects = []
            
        # Convert existing rects to the format expected by st_img_label, including scaling
        formatted_existing_rects = [
            {
                "left": int(rect["left"] * resized_img.width / img_width),
                "top": int(rect["top"] * resized_img.height / img_height),
                "width": int(rect["width"] * resized_img.width / img_width),
                "height": int(rect["height"] * resized_img.height / img_height),
                "label": rect["label"],
            }
            for rect in existing_rects
        ]

        # Display the image using st_img_label with pre-populated boxes
        rects = st_img_label(resized_img, box_color="red", rects=formatted_existing_rects)

        def annotate(im, rects, current_labels):
            original_width, original_height = im.get_img().size
            resized_width, resized_height = im.resized_img().size
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height
            
            scaled_rects = []
            for rect in rects:
                scaled_rect = {
                    "left": int(rect["left"] * scale_x),
                    "top": int(rect["top"] * scale_y),
                    "width": int(rect["width"] * scale_x),
                    "height": int(rect["height"] * scale_y),
                    "label": rect.get("label", DEFAULT_LABEL) # Ensure label is set
                }
                scaled_rects.append(scaled_rect)

            # Save annotations to annotation_data
            st.session_state["annotation_data"][img_file_name] = {
                "image_path": img_path,
                "rects": scaled_rects  
            }

            # Save to annotation file
            annotation_file_path = os.path.join(IMG_DIR, img_file_name + ".json")
            with open(annotation_file_path, "w") as f:
                json.dump(st.session_state["annotation_data"][img_file_name], f, indent=4)

        if rects:
            st.button(label="Save", on_click=lambda im=im, rects=rects: annotate(im, rects, ALL_LABELS))

        # Sidebar: show status
        n_files = len(st.session_state["files"])
        n_annotate_files = len(st.session_state["annotation_files"])
        st.sidebar.write("Total files:", n_files)
        st.sidebar.write("Total annotate files:", n_annotate_files)
        st.sidebar.write("Remaining files:", n_files - n_annotate_files)

 
    

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button(label="Previous image", on_click=previous_image)
        with col2:
            st.button(label="Next image", on_click=next_image)
        st.sidebar.button(label="Next need annotate", on_click=next_annotate_file)

    else:
        st.warning("Invalid image index. Please refresh the file list.")
        st.session_state["image_index"] = 0

if __name__ == "__main__":
    run()