import io
import base64
import logging
import os
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import cv2
import numpy as np

from utils import process_hvac_drawing, crop_layout_percentage, enhance_cropped_lines, extract_left_half, draw_calibrated_overlays, extract_hvac_skeleton_canvas, process_binary_and_restore_original
from hvac_agent import HVACAgentSystem

# Setup logging formatting
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HVAC_API")

app = FastAPI()

# Best Practice: Pull keys dynamically from environment variables
GEMINI_API_KEY = ""
MISTRAL_API_KEY = "" # Set your Mistral API key in environment

agent_system = HVACAgentSystem(gemini_api_key=GEMINI_API_KEY, mistral_api_key=MISTRAL_API_KEY)


def stitch_crossing_segments(left_items, right_items):
    """
    Combines Left and Right predictions. If an item exists in both sets
    (crosses the centerline boundary), it stretches the box to cover both windows.
    """
    stitched_output = []
    right_lookup = {item["id"]: item for item in right_items}

    # Process items found on the left side
    for l_item in left_items:
        item_id = l_item["id"]
        
        if item_id in right_lookup:
            # THE ITEM CROSSES THE MIDPOINT LINE! Merge their bounding boxes.
            r_item = right_lookup.pop(item_id)
            
            l_ymin, l_xmin, l_ymax, l_xmax = l_item["bbox"]
            r_ymin, r_xmin, r_ymax, r_xmax = r_item["bbox"]
            
            # The merged box takes the maximum span across both windows
            merged_bbox = [
                round(min(l_ymin, r_ymin), 4),
                round(min(l_xmin, r_xmin), 4),
                round(max(l_ymax, r_ymax), 4),
                round(max(l_xmax, r_xmax), 4)
            ]
            
            # Combine any structural target graph connections safely
            combined_connections = l_item.get("connections", []) + [
                conn for conn in r_item.get("connections", []) 
                if conn not in l_item.get("connections", [])
            ]
            
            l_item["bbox"] = merged_bbox
            l_item["connections"] = combined_connections
            l_item["confidence"] = round((l_item["confidence"] + r_item["confidence"]) / 2, 2)
            
            stitched_output.append(l_item)
        else:
            # Item resides completely on the Left half
            stitched_output.append(l_item)

    # Add remaining items that reside completely on the Right half
    for r_item in right_lookup.values():
        stitched_output.append(r_item)

    return stitched_output

DEBUG_DIR = "debug_outputs"
os.makedirs(DEBUG_DIR, exist_ok=True)

import cv2
import numpy as np
import base64
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import uvicorn

app = FastAPI()
logger = logging.getLogger("HVAC_API")

@app.post("/analyze")
async def analyze_drawing(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Received file upload stream: {file.filename}")
        filename_lower = file.filename.lower()
        content = await file.read()
        
        # -------------------------------------------------------------
        # DYNAMIC INGESTION LAYER (PDF vs. Image Routing)
        # -------------------------------------------------------------
        if filename_lower.endswith(".pdf"):
            logger.info("📄 PDF Document detected. Triggering complete processing & layout crop...")
            enhanced_np = process_hvac_drawing(content, poppler_path=r"C:\poppler\Library\bin")
            cropped_np = crop_layout_percentage(enhanced_np)
            # This is your high-fidelity color/grayscale base image matrix
            final_ai_ready_np_raw = enhance_cropped_lines(cropped_np)
            
        elif filename_lower.endswith((".jpg", ".jpeg", ".png")):
            logger.info("Direct Image detected. Bypassing crop matrix transformations...")
            np_arr = np.frombuffer(content, np.uint8)
            decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if decoded_img is None:
                raise ValueError("Could not parse or decode raw file bytes into an image matrix.")
            final_ai_ready_np_raw = decoded_img
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format extension. Please upload a valid .pdf, .jpg, .jpeg, or .png blueprint map."
            )

        # Get dimensions of our cropped base template safely
        master_h, master_w = final_ai_ready_np_raw.shape[:2]
        logger.info(f"📐 Full Cropped Layout Resolution: {master_w}x{master_h}")
        mid_point = master_w // 2
        
        # -------------------------------------------------------------
        # STEP 1: IMAGING - SLICE ORIGINAL CANVASES FOR LLM (ZERO-OVERLAP)
        # -------------------------------------------------------------
        # Slicing the original high-fidelity format for optimal LLM context parsing
        left_half_np = final_ai_ready_np_raw[:, 0:mid_point]
        right_half_np = final_ai_ready_np_raw[:, mid_point:master_w]

        img_pil_left = Image.fromarray(cv2.cvtColor(left_half_np, cv2.COLOR_BGR2RGB))
        img_pil_right = Image.fromarray(cv2.cvtColor(right_half_np, cv2.COLOR_BGR2RGB))
        img_pil_full = Image.fromarray(cv2.cvtColor(final_ai_ready_np_raw, cv2.COLOR_BGR2RGB))
    
        # -------------------------------------------------------------
        # STEP 2: INFERENCE - EXECUTE AGENT RUNS ON IMAGE SPECIMENS
        # -------------------------------------------------------------
        logger.info("🤖 Processing Left Half via Estimator...")
        left_raw_data = agent_system.estimator_agent(img_pil_left)

        logger.info("🤖 Processing Right Half via Estimator...")
        right_raw_data = agent_system.estimator_agent(img_pil_right)

        # -------------------------------------------------------------
        # STEP 3: COORDINATE TRANSLATION MATRIX
        # -------------------------------------------------------------
        calibrated_left = []
        for item in left_raw_data:
            if "bbox" in item and len(item["bbox"]) == 4:
                ymin, xmin, ymax, xmax = item["bbox"]
                item["bbox"] = [round(ymin, 4), round(xmin * 0.5, 4), round(ymax, 4), round(xmax * 0.5, 4)]
            calibrated_left.append(item)

        calibrated_right = []
        for item in right_raw_data:
            if "bbox" in item and len(item["bbox"]) == 4:
                ymin, xmin, ymax, xmax = item["bbox"]
                item["bbox"] = [round(ymin, 4), round(0.5 + (xmin * 0.5), 4), round(ymax, 4), round(0.5 + (xmax * 0.5), 4)]
            calibrated_right.append(item)

        current_data = stitch_crossing_segments(calibrated_left, calibrated_right)

        # -------------------------------------------------------------
        # LLM AGENT FLOW: Self-Correction Loop
        # -------------------------------------------------------------
        logger.info("🤖 Starting Multi-Agent Extraction & Audit Loop...")
        MAX_ATTEMPTS = 3
        attempt = 1
        final_json = None
        audit_passed = False
        report = ""

        while attempt <= MAX_ATTEMPTS:
            logger.info(f"📋 [Attempt {attempt}/{MAX_ATTEMPTS}] Auditing extracted full metadata...")
            report = agent_system.auditor_agent(img_pil_full, current_data)
            
            if "PASSED" in report.upper():
                logger.info(f"✅ Audit PASSED on attempt {attempt}!")
                final_json = current_data
                audit_passed = True
                break
            
            logger.warning(f"❌ Audit FAILED on attempt {attempt}. Report details: {report}")
            if attempt == MAX_ATTEMPTS:
                logger.error("🚨 Reached maximum Gemini retry limit without a clean PASSED audit.")
                break
                
            logger.info("🔄 Routing data to Review Agent for spatial calibration...")
            current_data = agent_system.review_agent(img_pil_full, current_data, report)
            attempt += 1

        if final_json is None:
            final_json = current_data

        # -------------------------------------------------------------
        # OUTPUT PACKAGING LAYER - BASE REF AND AGENT OVERLAYS
        # -------------------------------------------------------------
        logger.info("⚡ Packaging LLM matrix into Base64 for UI validation...")
        
        # Send clean original image canvas to UI as the comparison layer base
        success, buffer = cv2.imencode('.jpg', final_ai_ready_np_raw)
        if not success:
            raise ValueError("OpenCV failed to encode final raw matrix to JPEG bytes.")
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Paint agent bounding box overlay elements onto a separate copy of the raw canvas
        annotated_np = draw_calibrated_overlays(final_ai_ready_np_raw.copy(), final_json)
        _, anno_buffer = cv2.imencode('.jpg', annotated_np)
        anno_base64 = base64.b64encode(anno_buffer).decode('utf-8')        

        # -------------------------------------------------------------
        # STEP 4: DETACHED BINARY OPENCV RULE GENERATION
        # -------------------------------------------------------------
        logger.info("📊 Converting a local matrix slice into binary for OpenCV flood-fill...")
        gray = cv2.cvtColor(final_ai_ready_np_raw, cv2.COLOR_BGR2GRAY)
        _, local_binary_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Execute your binary engine using distinct matrix variables to eliminate target bleeding
        annotated_original_format = process_binary_and_restore_original(
            final_ai_ready_np_raw=final_ai_ready_np_raw, # Pristine original format image
            final_ai_ready_np=local_binary_mask,          # Exact isolated binary mask 
            alpha=0.50
        )

        # Encode the restored layout for the OpenCV pipeline response channel
        _, cv_buffer = cv2.imencode('.jpg', annotated_original_format, [cv2.IMWRITE_JPEG_QUALITY, 95])
        opencv_image_base64 = base64.b64encode(cv_buffer).decode('utf-8')

        return {
            "status": "success" if audit_passed else "partial_success", 
            "cropped_image": img_base64,               # Background original reference layout
            "annotated_image": anno_base64,           # LLM agent coordinate markers
            "opencv_image": opencv_image_base64,       # Flood-fill rule engine outputs returned to original format
            "data": final_json
        }
        
    except Exception as e:
        logger.error(f"❌ API Processing Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))