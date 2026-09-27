import os
import torch
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageFile
import gradio as gr
import matplotlib.pyplot as plt # Not strictly needed for Gradio output, but good for local debugging if plots were used

# Ensure ImageFile loading is robust (from previous notebook state)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Initialize EasyOCR reader once globally for efficiency ---
# 'en' for English language. You can add more languages if needed, e.g., ['en', 'hi'] for Hindi
# The 'gpu' check uses torch.cuda.is_available() which is already imported.
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
print(f"EasyOCR initialized. Using GPU: {torch.cuda.is_available()}")

def recognize_license_plate_from_image(image_pil):
    """
    Processes an input image to recognize license plates using EasyOCR.
    This function performs OCR on the entire image and draws EasyOCR's
    detected bounding boxes and text.

    Args:
        image_pil (PIL.Image): The input image from Gradio.

    Returns:
        PIL.Image: Image with EasyOCR's detected text and bounding boxes drawn.
        str: Concatenated recognized license plate text.
        PIL.Image: Cropped license plate image (first detected), or a placeholder.
    """
    if image_pil is None:
        # Return placeholder images and text if no image is uploaded
        blank_output_image = Image.new('RGB', (400, 300), color = 'gray')
        blank_cropped_plate = Image.new('RGB', (100, 50), color = 'red')
        return blank_output_image, "Please upload an image.", blank_cropped_plate

    # Convert PIL Image to OpenCV format (BGR for OpenCV operations, RGB for cropping with PIL)
    img_cv_original_rgb = np.array(image_pil) # Keep as RGB for later PIL conversion
    img_cv_bgr = cv2.cvtColor(img_cv_original_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for EasyOCR and cv2 drawing

    # Create a copy to draw on
    img_display = img_cv_bgr.copy()

    # Perform OCR
    ocr_results = reader.readtext(img_cv_bgr)

    recognized_texts = []
    cropped_plates_pil = []

    if ocr_results:
        for (bbox, text, prob) in ocr_results:
            recognized_texts.append(text)

            # Draw bounding box on the display image
            top_left_pt = (int(bbox[0][0]), int(bbox[0][1]))
            bottom_right_pt = (int(bbox[2][0]), int(bbox[2][1]))
            cv2.rectangle(img_display, top_left_pt, bottom_right_pt, (0, 255, 0), 2) # Green rectangle
            cv2.putText(img_display, text, (top_left_pt[0], top_left_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Crop the detected region using EasyOCR's own bbox
            x_min, y_min = top_left_pt
            x_max, y_max = bottom_right_pt

            # Ensure coordinates are within image bounds
            height, width, _ = img_cv_original_rgb.shape
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Use the original RGB numpy array for cropping to avoid color issues
            cropped_plate_cv_rgb = img_cv_original_rgb[y_min:y_max, x_min:x_max]

            if cropped_plate_cv_rgb.size > 0:
                cropped_plates_pil.append(Image.fromarray(cropped_plate_cv_rgb))
    else:
        recognized_texts.append("No text detected.")

    # Convert the OpenCV image (with drawings) back to PIL for Gradio output
    output_image_pil = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    # Return the first cropped plate or a placeholder if none
    if cropped_plates_pil:
        return output_image_pil, ", ".join(recognized_texts), cropped_plates_pil[0]
    else:
        blank_image_placeholder = Image.new('RGB', (100, 50), color = 'red') # Red placeholder for no detected plate
        return output_image_pil, ", ".join(recognized_texts), blank_image_placeholder

# Create the Gradio interface
title = "License Plate Recognition (LPR) Demo with EasyOCR"
description = (
    "Upload an image of a car. This demo uses EasyOCR to detect and recognize text "
    "within the uploaded image. EasyOCR will automatically draw bounding boxes "
    "around detected text and attempt to crop the first detected plate. "
    "For a full LPR system, an object detection model would typically first "
    "locate the license plate, then OCR would be applied to the cropped plate. "
    "Here, EasyOCR is run on the full image."
)
article = "Built using [EasyOCR](https://github.com/JaidedAI/EasyOCR) and [Gradio](https://gradio.app/)."

gr.Interface(
    fn=recognize_license_plate_from_image,
    inputs=gr.Image(type="pil", label="Upload Car Image"),
    outputs=[
        gr.Image(label="Original Image with Detections", type="pil"),
        gr.Textbox(label="Recognized Text"),
        gr.Image(label="Cropped License Plate (First Detected)", type="pil")
    ],
    title=title,
    description=description,
    article=article,
    # allow_flagging="never", # This argument is no longer supported in newer Gradio versions
).launch(debug=True)
