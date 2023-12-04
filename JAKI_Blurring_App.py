import streamlit as st
import cv2
from PIL import Image
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np
import os

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    cfg = get_cfg()
    # add model-specific configurations here if needed
    cfg.merge_from_file(model_zoo.get_config_file("faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("faster_rcnn_R_50_FPN_3x.yaml")
    #cfg.merge_from_file("C:/Users/bonda/OneDrive/Documents/Application After Bachelor/JSC/PROJECT/Project Code/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold
    cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference
    return DefaultPredictor(cfg)

# Function to process and blur the image
def process_and_blur_image(image, predictor):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    outputs = predictor(image_np)
    for bbox, cls in zip(outputs["instances"].pred_boxes, outputs["instances"].pred_classes):
        if cls == 0 or cls == 1:  
            x1, y1, x2, y2 = bbox.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            region = image_np[y1:y2, x1:x2]
            region = cv2.GaussianBlur(region, (99, 99), 0)
            image_np[y1:y2, x1:x2] = region

    processed_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return processed_image

# Load the model
model_path = "detectron2_repo/300_JAKI_FacePlate_Model.pth"
predictor = load_model(model_path)

# Streamlit UI
st.title("Face and Number Plate Blurring App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    st.write("Processing image...")
    processed_image = process_and_blur_image(image, predictor)
    st.image(processed_image, caption='Processed Image', use_column_width=True)
