import streamlit as st
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import logging
import re
import cv2  
import numpy as np
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import logging
import sqlite3
from datetime import datetime



# --- Configuration ---
COLLECTIONS_DIR = "/Users/stuartgano/Downloads/Streamlit/streamlit-img-label/pages/streamlit_img_label/collections"
OUTPUT_DIR = "/Users/stuartgano/Downloads/Streamlit/streamlit-img-label/temp_pdf_pages"
IMG_DIR = "/Users/stuartgano/Downloads/Streamlit/streamlit-img-label/temp_pdf_pages"
LEFT_THRESHOLD = 425 # Define left_threshold as a global constant


# --- Logging Setup ---
logging.basicConfig(
    filename="pdf_extraction_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                "top": 100,
                "width": 800,
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
                "left": 100,
                "top": 1050,
                "width": 500,
                "height": 80,
                "label": "Page 5 Email 2"
            },
            {
                "left": 100,
                "top": 1120,
                "width": 500,
                "height": 80,
                "label": "Page 5 Email 3"
            },
            {
                "left": 100,
                "top": 1200,
                "width": 500,
                "height": 80,
                "label": "Page 5 Email 4"
            }
        ]
    },
    "page_7.png": {
        "image_path": os.path.join(IMG_DIR, "page_7.png"),
        "rects": [
            {
                "left": 75,
                "top": 335,
                "width": 750,
                "height": 50,  # Adjusted height
                "label": "Client Profile Owner 1: Annual Income"
            },
            {
                "left": 75,
                "top": 385, # Adjusted top
                "width": 750,
                "height": 50, # Adjusted height
                "label": "Client Profile Owner 1: Net Worth"
            },
            {
                "left": 75,
                "top": 435, # Adjusted top
                "width": 750,
                "height": 50, # Adjusted height
                "label": "Client Profile Owner 1: liquid Net Worth"
            },
            {
                "left": 75,
                "top": 485, # Adjusted top
                "width": 750,
                "height": 50, # Adjusted height
                "label": "Client Profile Owner 1 Tax Bracket"
            },
            {
                "left": 475,
                "top": 335,
                "width": 750,
                "height": 50,
                "label": "Client Profile Owner 2: Annual Income"
            },
            {
                "left": 475,
                "top": 385,
                "width": 750,
                "height": 50,
                "label": "Client Profile Owner 2: Net Worth"
            },
            {
                "left": 475,
                "top": 435,
                "width": 750,
                "height": 50,
                "label": "Client Profile Owner 2: liquid Net Worth"
            },
            {
                "left": 475,
                "top": 485,
                "width": 750,
                "height": 50,
                "label": "Client Profile Owner 2 Tax Bracket"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: Equities"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: Bonds"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: MFs/UITs"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: ETFs"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: Annuities"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: Margin Trading"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: Options/Futures"
            },
            {
                "left": 75,
                "top": 635,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 1: Alternatives"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: Equities"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: Bonds"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: MFs/UITs"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: ETFs"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: Annuities"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: Margin Trading"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: Options/Futures"
            },
            {
                "left": 425,
                "top": 600,
                "width": 450,
                "height": 440,
                "label": "Investment Experience Owner 2: Alternatives"
            },
            {
                "left": 95,
                "top": 775,
                "width": 880,
                "height": 570,
                "label": "Account Profile Page 7"
            },
            {
                "left": 95,
                "top": 1200,
                "width": 800,
                "height": 250,
                "label": "Additional Details Page 7"
            }
        ]
    },
    "page_10.png": {
        "image_path": os.path.join(IMG_DIR, "page_10.png"),
        "rects": [
            {
                "left": 50,
                "top": 1200,
                "width": 800,
                "height": 265,
                "label": "Client Initials"
            },
            {
                "left": 50,
                "top": 400,
                "width": 450,
                "height": 165,
                "label": "Fee to be Charged"
            },
            {
                "left": 65,
                "top": 500,
                "width": 800,
                "height": 300,
                "label": "Additional Notes"
            }
        ]
    },
    "page_12.png": {
        "image_path": os.path.join(IMG_DIR, "page_12.png"),
        "rects": [
            {
                "left": 535,
                "top": 75,
                "width": 1000,
                "height": 145,
                "label": "Advisor Signature"
            },
            {
                "left": 535,
                "top": 220,
                "width": 1000,
                "height": 110,
                "label": "Client Signature 1"
            },
            {
                "left": 535,
                "top": 330,
                "width": 1000,
                "height": 105,
                "label": "Client Signature 2"
            },
            {
                "left": 535,
                "top": 435,
                "width": 1000,
                "height": 110,
                "label": "Client Signature 3"
            },
            {
                "left": 535,
                "top": 545,
                "width": 1000,
                "height": 110,
                "label": "Client Signature 4"
            }
        ]
    }
}

def write_to_db(df):
    conn = sqlite3.connect('pdf_data.db')
    df['timestamp'] = datetime.now()
    df.to_sql('extracted_data', conn, if_exists='append', index=False)
    return len(df)

def preprocess_image(image, grayscale=True):
    """Preprocess the image for better OCR results."""
    if grayscale:
        # Convert to grayscale
        gray_img = image.convert("L")
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced_img = enhancer.enhance(2.0)
        # Apply a filter to reduce noise
        filtered_img = enhanced_img.filter(ImageFilter.MedianFilter())
        return filtered_img
    else:
        # Enhance contrast (without grayscale)
        enhancer = ImageEnhance.Contrast(image)
        enhanced_img = enhancer.enhance(1.5)  # Adjusted enhancement factor
        # Apply a filter to reduce noise
        filtered_img = enhanced_img.filter(ImageFilter.MedianFilter())
        return filtered_img


def extract_underlined_text(img, x, y, width, height):
    """Extracts underlined text from a specified region in an image."""
    cropped_img = img.crop((x, y, x + width, y + height))
    open_cv_image = np.array(cropped_img.convert('L'))  # Convert to grayscale
    edges = cv2.Canny(open_cv_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Perform OCR on the cropped image
    text = pytesseract.image_to_string(cropped_img).strip()
    words = text.split()

    # Filter words based on proximity to detected lines
    underlined_words = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y1 - y2) < 5:  # Horizontal line
                    baseline_y = y1
                    for word in words:
                        # Assume word is underlined if baseline is close to line
                        # This is a heuristic; adjust as needed
                        if baseline_y in range(y1 - 5, y1 + 5):
                            underlined_words.append(word)

    return " ".join(underlined_words)

def is_checkbox_checked(img, x, y, width, height):
    """
    Determines if a checkbox is checked based on the recognized text.

    Args:
        img (PIL.Image): The image containing the checkbox.
        x (int): The x-coordinate of the top-left corner of the checkbox.
        y (int): The y-coordinate of the top-left corner of the checkbox.
        width (int): The width of the checkbox.
        height (int): The height of the checkbox.

    Returns:
        bool: True if the checkbox is checked, False otherwise.
    """
    cropped_img = img.crop((x, y, x + width, y + height))
    text = pytesseract.image_to_string(cropped_img).strip().lower()
    checked_indicators = ['x', '✓', '✔', 'yes', 'checked']
    return any(indicator in text for indicator in checked_indicators)

def accountName_extract_text_between_een_and_parenthesis(text):
    """Extracts text between 'een' and ')'."""
    start_index = text.find('een') + len('een')
    end_index = text.find(')')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index:end_index].strip()
    else:
        return "Pattern Not Found"

def has_docusign_signature_v1(image):
    """Checks if an image has a DocuSign signature using text detection."""
    try:
        text = pytesseract.image_to_string(image)
        if "DocuSign" in text:
            return True
        return False
    except Exception as e:
        logging.error(f"Error in v1: {e}")
        return False
def extract_email(text):
    """
    Extracts email addresses from text using a regular expression.
    
    Args:
        text: The input text.
        
    Returns:
        A list of extracted email addresses.
    """
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    return emails

def extract_account_profile_info(text):
    account_profile = {}

    # Define a dictionary mapping the labels to the corresponding fields
    field_mapping = {
        "Account Purpose": "account_purpose",
        "Risk Tolerance": "risk_tolerance",
        "Time Horizon": "time_horizon",
    }

    # Extract the values for each field
    for field, label in field_mapping.items():
        match = re.search(f"{label}: ([^\n]+)", text)
        if match:
            account_profile[field] = match.group(1).strip()

    return account_profile

def has_docusign_signature_v2(image):
    """Checks if an image has a DocuSign signature using contour detection."""
    try:
        open_cv_image = np.array(image)
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edged_image = cv2.Canny(blurred_image, 50, 150)
        contours, _ = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours to detect signature-like patterns
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Example threshold
                # Implement logic to check if the contour matches signature characteristics
                return True
        return False
    except Exception as e:
        logging.error(f"Error in v2: {e}")
        return False



def client_initials_majority_vote_signature_detection(image):
    """Determines if a DocuSign signature is present based on majority vote."""
    votes = [
        has_docusign_signature_v1(image),
        has_docusign_signature_v2(image),
    ]
    return votes.count(True) >= 1


def detect_checkboxes_contours(image, min_area=25, max_area=400, aspect_ratio_range=(0.7, 1.3), solidity_threshold=0.4):
    """
    Detects checkboxes using contour analysis and returns True if any are checked (filled), 
    False otherwise.

    Args:
        image: The input image (grayscale).
        min_area: Minimum area of a contour to be considered a checkbox.
        max_area: Maximum area of a contour to be considered a checkbox.
        aspect_ratio_range: Allowed aspect ratio range (width/height) for checkboxes.
        solidity_threshold: Minimum solidity (contour area / convex hull area) for checkboxes.

    Returns:
        True if at least one checkbox is detected as filled, False otherwise.
    """
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert image to grayscale if it isn't already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Preprocessing
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 3. Filter Contours based on Properties
        area = cv2.contourArea(cnt)

        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                if solidity > solidity_threshold:
                    # 4. Determine Checkbox State (Filled/Empty)
                    roi = image[y:y+h, x:x+w]
                    average_intensity = np.mean(roi)

                    #  Classify as filled or empty based on the average intensity.
                    if average_intensity < 128:  # Adjust threshold if needed
                        return True  # Checkbox is filled, return True immediately

    return False  # No filled checkboxes found
def extract_text_from_regions(image, regions):
    """
    Extracts text from specified regions in an image using Tesseract OCR.
    
    Args:
        image (PIL.Image): The image to process
        regions (list): List of region dictionaries containing coordinates and labels
    
    Returns:
        list: Extracted data for each region (list of dictionaries with 'label' and 'text')
    """
    extracted_data = []

    # Create logs directory if it doesn't exist
    log_dir = "pattern_match_logs"
    os.makedirs(log_dir, exist_ok=True)

    for region in regions:
        left = region["left"]
        top = region["top"]
        width = region["width"]
        height = region["height"]
        label = region["label"]

        
        # Crop the image to the bounding box using PIL Image object
        cropped_img = image.crop((left, top, left + width, top + height))
        preprocessed_img = preprocess_image(cropped_img)

        try:
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(preprocessed_img, config=custom_config).strip()

            logging.debug(f"Raw OCR Text for {label}: {text}")

            # --- Specific label processing ---
            if "Email" in label:
                emails = extract_email(text)
                text = ", ".join(emails) if emails else "No valid email found"
                extracted_data.append({"label": label, "text": text})


            elif label == "Account Number":
                cleaned_text = re.sub(r'\D', '', text)
                match = re.search(r'\d{6}', cleaned_text)
                if match:
                    text = "GCW" + match.group(0)
                else:
                    # Pass cropped_img instead of image_path
                    log_pattern_failure(label, text, cleaned_text, cropped_img)
                    text = "Pattern Not Found"
                extracted_data.append({"label": label, "text": text})


            elif label == "Account Name":
                text = accountName_extract_text_between_een_and_parenthesis(text)
                if text == "Pattern Not Found":
                    log_pattern_failure()
                extracted_data.append({"label": label, "text": text})

            elif label in ["Client Initials", "Page 5 Client Initials"]:
                text = "Yes" if client_initials_majority_vote_signature_detection(image=preprocessed_img) else "No"
                extracted_data.append({"label": label, "text": text})

            elif "Checkbox" in label:
                checkbox_detected = detect_checkboxes_contours(preprocessed_img)
                text = "Yes" if checkbox_detected else "No"
                print(f"Checkboxes detected as checked in {label}: {text}")
                extracted_data.append({"label": label, "text": text})

            elif label == "Client Profile Owner 1: Annual Income":
                text = text.strip()
                extracted_data.append({"label": label, "text": text})

            elif label == "Client Profile Owner 1: Net Worth":
                match = re.search(r"Net Worth:\s*([\d,]+ - [\d,]+)", text)
                if match:
                    text = match.group(1)
                else:
                    log_pattern_failure()
                    text = "Pattern Not Found"
                extracted_data.append({"label": label, "text": text})

            elif label == "Client Profile Owner 1: Liquid Net Worth":
                match = re.search(r"Net Worth:\s*([\d,]+ - [\d,]+)", text)
                if match:
                    text = match.group(1)
                else:
                    log_pattern_failure()
                    text = "Pattern Not Found"
                extracted_data.append({"label": label, "text": text})
            
            elif label == "Client Profile Owner 2: Annual Income":
                text = text.strip()
                extracted_data.append({"label": label, "text": text})

            elif label == "Client Profile Owner 2: Liquid Net Worth":
                # 1. Remove Redundant Label:
                text = text.replace("Liquid Net Worth", "").strip()

                # 2. Extract Potential Ranges using Regex:
                potential_ranges = re.findall(r"[\d,]+ - [\d,]+", text)

                # 3. Filter and Validate Ranges:
                if potential_ranges:
                    # a. Find the Longest Range (most likely to be correct):
                    longest_range = max(potential_ranges, key=len)

                    # b. Basic Validation (check if it's a reasonable range):
                    parts = longest_range.split(" - ")
                    if len(parts) == 2:
                        try:
                            start = int(parts[0].replace(",", ""))
                            end = int(parts[1].replace(",", ""))
                            if start < end:  # Ensure start is less than end
                                text = longest_range
                            else:
                                text = "Range Invalid"
                        except ValueError:
                            text = "Range Invalid"
                    else:
                        text = "Range Invalid"
                else:
                    text = "Pattern Not Found"
                    log_pattern_failure(label, text, "", cropped_img)
                extracted_data.append({"label": label, "text": text})

            elif label == "Client Profile Owner 2: Net Worth":
                match = re.search(r"Net Worth:\s*([\d,]+ - [\d,]+)", text)
                if match:
                    text = match.group(1)
                else:
                    log_pattern_failure()
                    text = "Pattern Not Found"
                extracted_data.append({"label": label, "text": text})

            elif "Investment Experience" in label:
                lines = text.split("\n")

                # Determine if Owner 1 or Owner 2 based on left value
                if left < LEFT_THRESHOLD:
                    owner_label = "Owner 1"
                else:
                    owner_label = "Owner 2"

                owner_data = {}  # Initialize dictionary to store results for this owner

                for product_type in ["Equities", "Bonds", "MFs/UITs", "ETFs", "Annuities", "Margin Trading", "Options/Futures", "Alternatives"]:
                    match = re.search(fr".*(None|Limited|Moderate|Extensive)\s*{product_type}.*", text, re.IGNORECASE)
                    if match:
                        extracted_value = match.group(1)
                        best_match = max(["None", "Limited", "Moderate", "Extensive"], key=lambda x: fuzz.ratio(extracted_value, x))
                        logging.debug(f"Fuzzy Match for {product_type}: Extracted='{extracted_value}', Best Match='{best_match}', Score={fuzz.ratio(extracted_value, best_match)}")
                        owner_data[product_type] = best_match
                    else:
                        logging.debug(f"No Match for {product_type}: Raw Text='{text}', Label='{label}'")
                        owner_data[product_type] = "Not Found"

                # Store results in a dictionary with full labels as keys
                full_label = f"Investment Experience {owner_label}: {label.split(': ')[-1]}"
                text = str(owner_data) # Convert dictionary to string for display
                extracted_data.append({"label": label, "text": text}) # Use the full label


            elif "Account Profile Information" in label:
                # Extract account profile info from the PIL Image object
                account_profile = extract_account_profile_info(text)
                extracted_data.append({
                    "label": "Account Profile Information",
                    "text": str(account_profile)  # Convert dict to string for storage
                })

            elif "Additional Details" in label:
                # Remove instruction text and process the PIL Image
                text = text.replace("Please provide any additional details below.", "").strip()
                text = "Empty" if not text else text
                extracted_data.append({
                    "label": "Additional Details",
                    "text": text
                })
            elif label == "Fee to be Charged":
                # Extract and format fee information
                text = text.strip()
                # Add any specific fee text processing logic here
                extracted_data.append({
                    "label": label,
                    "text": text
                })

            elif label == "Additional Notes":
                # Process additional notes while preserving meaningful formatting
                text = text.strip()
                # Remove redundant whitespace while keeping line breaks
                text = ' '.join(text.split())
                
                # Handle empty notes
                if not text:
                    text = "No additional notes provided"
                    
                extracted_data.append({
                    "label": label,
                    "text": text
                })    

            elif label == "Advisor Signature":
                # Check for signature using the preprocessed PIL Image
                signature_present = client_initials_majority_vote_signature_detection(preprocessed_img)
                text = "Yes" if signature_present else "No"
                extracted_data.append({
                    "label": label,
                    "text": text
                })

            elif "Client Signature" in label:
                # Check for signature using the preprocessed PIL Image
                signature_present = client_initials_majority_vote_signature_detection(preprocessed_img)
                text = "Yes" if signature_present else "No"
                extracted_data.append({
                    "label": label,
                    "text": text
                })        

        except Exception as e:
            logging.error(f"Error during OCR: {e}, Label='{label}'")
            text = "OCR Error"

    return extracted_data  


def extract_data_from_boxes(pdf_directory):
    all_extracted_data = []
    account_info = {}

    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Processing: {pdf_path}")

            pages = convert_from_path(pdf_path)

            # First pass - extract account info from page 1
            if "page_1.png" in TEMPLATE_DATA:
                page_1 = pages[0]
                pil_image = Image.frombytes("RGB", page_1.size, page_1.tobytes())
                page_1_data = extract_text_from_regions(pil_image, TEMPLATE_DATA["page_1.png"]["rects"])
                
                # Add account info as separate rows
                for data in page_1_data:
                    if data["label"] in ["Account Name", "Account Number"]:
                        # Store in account_info dictionary
                        account_info[pdf_file] = account_info.get(pdf_file, {})
                        account_info[pdf_file][data["label"]] = data["text"]
                        
                        # Add as separate row to extracted_data
                        account_record = {
                            "label": data["label"],
                            "text": data["text"],
                            "filename": pdf_file,
                            "page": 1
                        }
                        all_extracted_data.append(account_record)

            # Process remaining pages
            for page_index, page in enumerate(pages):
                page_key = f"page_{page_index + 1}.png"
                
                if page_key in TEMPLATE_DATA:
                    pil_image = Image.frombytes("RGB", page.size, page.tobytes())
                    extracted_data = extract_text_from_regions(pil_image, TEMPLATE_DATA[page_key]["rects"])

                    for data in extracted_data:
                        data["filename"] = pdf_file
                        data["page"] = page_index + 1
                        if pdf_file in account_info:
                            data["Account Name"] = account_info[pdf_file].get("Account Name", "")
                            data["Account Number"] = account_info[pdf_file].get("Account Number", "")

                    all_extracted_data.extend(extracted_data)

    df = pd.DataFrame(all_extracted_data)
    return df



# --- Streamlit App ---
st.title("PDF Data Extraction")

# Add a textbox for the PDF directory path
pdf_directory = st.text_input("Enter the directory containing PDF files:")

if st.button("Extract Data from PDF Directory"):
    if pdf_directory:
        df = extract_data_from_boxes(pdf_directory)  # Call the function for directory processing
        st.write("Extracted Data:")
        st.dataframe(df)
    else:
        st.warning("Please enter a valid directory path.")

if st.button("Extract Data from Template"):
    df = extract_data_from_boxes(TEMPLATE_DATA)  # Call the function for template processing
    st.write("Extracted Data:")
    st.dataframe(df)

if "rerun" not in st.session_state:
    st.session_state["rerun"] = False

if st.button("Extract and Store Data"):
    # Your existing extraction code
    df = extract_data_from_boxes(pdf_directory)
    
    # Write to database
    records_added = write_to_db(df)
    st.success(f"Successfully added {records_added} records to database!")
    
    # Display the extracted data
    st.write("Extracted Data:")
    st.dataframe(df)
