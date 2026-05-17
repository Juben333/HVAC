import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import os

st.set_page_config(page_title="HVAC Production Estimator", layout="wide")
st.title("🏗️ HVAC Estimator Pro")

# --- INITIALIZE SESSION STATE ---
if "api_result" not in st.session_state:
    st.session_state.api_result = None
if "uploaded_file_type" not in st.session_state:
    st.session_state.uploaded_file_type = None

up_file = st.file_uploader(
    "Upload HVAC Schematic Blueprint", 
    type=["pdf", "png", "jpg", "jpeg"]
)

if up_file and st.button("Start Analysis", type="primary"):
    with st.spinner("Processing drawing assets and executing multi-agent validation audit..."):
        try:
            _, file_extension = os.path.splitext(up_file.name.lower())
            st.session_state.uploaded_file_type = "image" if file_extension in [".jpg", ".jpeg", ".png"] else "pdf"
            
            files = {"file": (up_file.name, up_file.getvalue(), "application/octet-stream")}
            response = requests.post("http://localhost:8000/analyze", files=files)
            
            if response.status_code == 200:
                st.session_state.api_result = response.json()
                st.success("Analysis Complete!")
            else:
                st.error(f"Backend Engine Error: {response.text}")
        except Exception as e:
            st.error(f"Connection to analysis router failed: {e}")

# --- DISPLAY RESULTS FROM SESSION STATE ---
if st.session_state.api_result:
    res_data = st.session_state.api_result
    data = res_data.get("data", [])
    
    # Extract structural base64 rendering assets passed back by the API
    crop_str = res_data.get("cropped_image")
    anno_str = res_data.get("annotated_image")
    cv_str = res_data.get("opencv_image")  # Ingests your new backend property key
    
    is_image = st.session_state.uploaded_file_type == "image"
    step1_header = "🖼️ Step 1: Base Image Matrix Canvas" if is_image else "🖼️ Step 1: Extracted Crop Viewport"
    step1_caption = "Unaltered Source Canvas Map" if is_image else "Pre-processed & Cropped Base Template Viewport"
    step1_filename = "base_canvas.jpg" if is_image else "cropped_layout.jpg"
    step1_btn_label = "📥 Download Base Image" if is_image else "📥 Download Cropped Viewport"

    # Grid: Balanced 3-column layout to inspect images side-by-side
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader(step1_header)
        if crop_str:
            crop_bytes = base64.b64decode(crop_str)
            crop_img = Image.open(io.BytesIO(crop_bytes))
            st.image(crop_img, use_container_width=True, caption=step1_caption)
            
            st.download_button(
                label=step1_btn_label,
                data=crop_bytes,
                file_name=step1_filename,
                mime="image/jpeg",
                key="crop_download"
            )
    
    with col2:
        st.subheader("🤖 Step 2: LLM Output Verification")
        if anno_str:
            anno_bytes = base64.b64decode(anno_str)
            anno_img = Image.open(io.BytesIO(anno_bytes))
            st.image(anno_img, use_container_width=True, caption="Multi-Agent Predicted Coordinate Topology")
            
            st.download_button(
                label="📥 Download LLM Frame",
                data=anno_bytes,
                file_name="llm_annotated_drawing.jpg",
                mime="image/jpeg",
                key="anno_download"
            )

    with col3:
        st.subheader("⚙️ Step 3: OpenCV Rules Processing")
        if cv_str:
            cv_bytes = base64.b64decode(cv_str)
            cv_img = Image.open(io.BytesIO(cv_bytes))
            st.image(cv_img, use_container_width=True, caption="Restored Canvas with Flood-Fill & Contour Overlays")
            
            st.download_button(
                label="📥 Download OpenCV Frame",
                data=cv_bytes,
                file_name="opencv_rules_drawing.jpg",
                mime="image/jpeg",
                key="opencv_download"
            )

    # Interface Row: Global Actions and Metadata Tree Inspector
    st.markdown("---")
    st.subheader("📊 Stitched System Topology Manifest")
    
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        json_string = json.dumps(data, indent=2)
        st.download_button(
            label="📥 Download JSON Data",
            data=json_string,
            file_name="hvac_metadata.json",
            mime="application/json",
            key="json_download"
        )
    with c_btn2:
        st.info(f"Analysis successful. Successfully verified and stitched {len(data)} physical network entities across the pipeline layers.")

    # Live interactive expandable JSON structure element explorer
    with st.expander("🔍 Inspect Extracted Graph Object Nodes", expanded=False):
        st.json(data)

# Clear application execution cache inside sidebar panel
if st.session_state.api_result:
    if st.sidebar.button("Clear Results"):
        st.session_state.api_result = None
        st.session_state.uploaded_file_type = None
        st.rerun()