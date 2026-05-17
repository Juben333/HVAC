import cv2
import numpy as np
import logging
from pdf2image import convert_from_bytes  # Changed from convert_from_path
from skimage.morphology import skeletonize
from skimage.util import invert

logger = logging.getLogger("HVAC_Utils")

def process_hvac_drawing(pdf_content_bytes, dpi=350, poppler_path=None):
    """
    Combines rasterization from bytes, boundary detection, and enhancement.
    """
    try:
        logger.info(f"Rasterizing PDF at {dpi} DPI from bytes")
        
        # Use convert_from_bytes instead of convert_from_path
        pages = convert_from_bytes(
            pdf_content_bytes, 
            dpi=dpi, 
            first_page=1, 
            last_page=1, 
            poppler_path=poppler_path
        )
        
        if not pages:
            raise ValueError("No pages found in PDF")

        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        
        # ... [Rest of your boundary detection and enhancement code remains the same] ...
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = (gray < 50).astype(np.uint8) * 255
        edges = cv2.Canny(binary, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 500, minLineLength=int(W*0.25), maxLineGap=50)
        
        h_lines, v_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle < 5 or angle > 175:
                    h_lines.append((int((y1+y2)/2), min(x1,x2), max(x1,x2), np.linalg.norm([x2-x1, y2-y1])))
                elif 85 < angle < 95:
                    v_lines.append((int((x1+x2)/2), min(y1,y2), max(y1,y2), np.linalg.norm([x2-x1, y2-y1])))

        top_y = next((y for y, x1, x2, l in sorted(h_lines) if l > W*0.5 and y < H*0.3), H//10)
        notes_y = next((y for y, x1, x2, l in sorted(h_lines) if l > W*0.5 and H*0.4 < y < H*0.75), H)
        left_x = next((x for x, y1, y2, l in sorted(v_lines) if l > H*0.4 and x < W*0.1), W//20)
        title_x = next((x for x, y1, y2, l in sorted(v_lines, key=lambda l: -l[0]) if l > H*0.5 and W*0.6 < x < W*0.95), W)
        
        margin = 5
        cropped = img[max(0, top_y+margin):min(H, notes_y-margin), max(0, left_x+margin):min(W, title_x-margin)]

        yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        img_enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        img_enhanced = cv2.bilateralFilter(img_enhanced, d=7, sigmaColor=50, sigmaSpace=50)

        table = np.array([((i / 255.0) ** (1.0/1.2)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        final = cv2.LUT(img_enhanced, table)
        
        return final
    except Exception as e:
        logger.error(f"Failed to process drawing: {str(e)}")
        raise

def crop_layout_percentage(img_bgr):
    """
    Step 1: Focuses exclusively on the core HVAC schematic area
    by slicing away empty margins, grid frameworks, and notes.
    """
    H, W = img_bgr.shape[:2]
    
    y_start = int(H * 0.32)
    y_end = int(H * 0.85)
    x_start = int(W * 0.12)
    x_end = int(W * 0.85)
    
    cropped = img_bgr[y_start:y_end, x_start:x_end]
    return cropped

def enhance_cropped_lines(cropped_bgr):
    """
    Step 2: Takes the tightly cropped image and aggressively 
    darkens/thickens the engineering lines for optimal LLM processing.
    """
    # Convert to grayscale for threshold calculations
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding isolates clean line-art from page gradients
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 7
    )
    
    # Invert to make lines white on a black background for structural adjustments
    inverted = cv2.bitwise_not(thresh)
    
    # Create a 3x3 block kernel to dilate and thicken the line weight
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thickened_inverted = cv2.dilate(inverted, kernel, iterations=1)
    
    # Flip back to standard dark ink layout
    final_binary = cv2.bitwise_not(thickened_inverted)
    
    # Convert back to 3-channel BGR structure to keep image types uniform
    return cv2.cvtColor(final_binary, cv2.COLOR_GRAY2BGR)

def extract_left_half(img_np):
    """
    Splits the image precisely down the center horizontally and returns the left half.
    """
    h, w = img_np.shape[:2]
    mid_point = w // 2
    
    # Slice the numpy array: all rows, from column 0 to the midpoint (512px)
    left_half = img_np[:, 0:mid_point]
    return left_half

def draw_calibrated_overlays(image_np, metadata_json):
    """
    Paints bounding frames and text banner labels directly onto the final cropped canvas matrix.
    Uses dynamic re-scaling based on the current image shape.
    """
    # Create a deep copy to prevent mutating the base canvas matrix frame
    annotated_img = image_np.copy()
    h, w = annotated_img.shape[:2]
    
    for item in metadata_json:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
            
        # Convert normalized 0.0-1.0 coordinate metrics into exact physical pixel boundaries
        ymin = int(bbox[0] * h)
        xmin = int(bbox[1] * w)
        ymax = int(bbox[2] * h)
        xmax = int(bbox[3] * w)
        
        # Ensure values fall securely within array boundaries to avoid image bleeding
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)
        
        # Color matrix configuration (BGR): Supply = Vibrant Blue, Return = Pink
        is_supply = "SUPPLY" in str(item.get("type", "")).upper()
        color = (255, 140, 0) if is_supply else (140, 0, 255)
        
        # Draw target system duct framework bounding box
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Render text identifier background badge banner
        label = f"{item['id']} ({item.get('dimension_label', '')})"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        
        # Draw backdrop solid rectangle box for text readability
        cv2.rectangle(annotated_img, (xmin, ymin - text_h - 4), (xmin + text_w + 4, ymin), color, -1)
        cv2.putText(annotated_img, label, (xmin + 2, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
    return annotated_img

def extract_hvac_skeleton_canvas(img_bgr):
    """
    Processes an image matrix, isolates valid ductwork runs, runs skeletonization
    to discard hatch noise, and dilates the skeleton paths back up to a high-contrast
    thickness optimized for Gemini 3.1 Flash-Lite vision constraints.
    """
    try:
        logger.info("🔬 Transforming image canvas into clean high-contrast skeleton trace...")
        
        # Step 1: Grayscale conversion
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Edge-preserving Bilateral Filter
        gray_filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        
        # Step 3: Adaptive Thresholding (isolates geometry from layout gradients)
        binary = cv2.adaptiveThreshold(
            gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Step 4: Morphological opening to strip loose character noise pixels
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
        
        # Step 5: Morphological closing to bridge microscopic line snaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        
        # Step 6: Connected component analysis to drop dense wall patches & hatch noise
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean_canvas = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Exclude extreme background specs and enormous localized architectural blocks
            if 20 < area < 15000:
                clean_canvas[labels == i] = 255
                
        # Step 7: Skeletonize the filtered trace
        binary_bool = clean_canvas > 0
        skeleton_matrix = skeletonize(binary_bool)
        skeleton_uint8 = (skeleton_matrix * 255).astype(np.uint8)
        
        # -------------------------------------------------------------
        # PRODUCTION ENHANCEMENT: LLM VISION ACCELERATION DILATION
        # -------------------------------------------------------------
        # Pure 1px skeleton strands can vanish inside large canvas footprints during scaling.
        # Thicken the lines to a stable 3px width to ensure high model confidence.
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thickened_skeleton = cv2.dilate(skeleton_uint8, dilation_kernel, iterations=1)
        
        # Invert the image so the layout matches standard engineering style:
        # Dark clean lines printed on a crisp solid white canvas background.
        final_ai_canvas = cv2.bitwise_not(thickened_skeleton)
        
        return cv2.cvtColor(final_ai_canvas, cv2.COLOR_GRAY2BGR)
        
    except Exception as e:
        logger.error(f"❌ Failed calculating clean skeleton matrix: {str(e)}")
        raise


def process_binary_and_restore_original(final_ai_ready_np_raw, final_ai_ready_np, alpha=0.50):
    """
    1. Runs your exact script logic entirely on the binary image matrix.
    2. Isolates the resulting blue and red annotations.
    3. Reverts the background back to your original color drawing format.
    4. Blends the transparent annotations beautifully onto the original canvas for the UI.
    """
    try:
        # Create a working copy of your binary image
        binary_canvas = final_ai_ready_np.copy()
        H, W = binary_canvas.shape[:2]

        # Config parameters from your script
        DX0, DX1 = 150, 7800
        DY0, DY1 = 210, 2450

        exclude = [
            (0,    0,    160,  H),
            (7800, 0,    W,    H),
            (0,    0,    W,    215),
            (0,    2440, W,    H),
            (160,  1480, 760,  2100),
            (4750, 270,  5750, 1150),
        ]

        def in_excl(cx, cy):
            for ex0, ey0, ex1, ey1 in exclude:
                if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                    return True
            return False

        # ── YOUR EXACT INVERSION & MORPHOLOGY LOGIC ─────────────────────────
        inv = cv2.bitwise_not(binary_canvas)

        kern_h = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
        kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,  12))
        inv_c  = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kern_h)
        inv_c  = cv2.morphologyEx(inv_c, cv2.MORPH_CLOSE, kern_v)
        closed = cv2.bitwise_not(inv_c)

        # Colors (BGR)
        RED_BGR  = np.array([60,  60,  220], dtype=np.float32)
        BLUE_BGR = np.array([200, 140, 100], dtype=np.float32)

        # Create a 3-channel version of the binary matrix to receive initial annotations
        binary_bgr = cv2.cvtColor(binary_canvas, cv2.COLOR_GRAY2BGR).astype(np.float32)

        # ══════════════════════════════════════════════════════════════════════
        # 1. LONG THIN RECTANGLES (Duct bodies processed on Binary Canvas)
        # ══════════════════════════════════════════════════════════════════════
        tested   = np.zeros((H, W), dtype=np.uint8)
        filled_g = np.zeros((H, W), dtype=np.uint8)
        rects    = []

        for sy in range(DY0, DY1, 35):
            for sx in range(DX0, DX1, 70):
                if sy >= H or sx >= W: continue
                if closed[sy, sx] != 255: continue
                if tested[sy, sx]:       continue
                if in_excl(sx, sy):      continue

                mask = np.zeros((H+2, W+2), dtype=np.uint8)
                n, _, _, _ = cv2.floodFill(
                    closed.copy(), mask, (sx, sy), 128,
                    loDiff=0, upDiff=0,
                    flags=cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
                )
                fm = mask[1:-1, 1:-1]
                tested[fm > 0] = 1

                if n < 3000 or n > 550000: continue

                ys, xs = np.where(fm > 0)
                if len(ys) == 0: continue
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                fw, fh = x1 - x0, y1 - y0

                if fw < 180 or fh < 8:  continue
                asp = fw / max(fh, 1)
                if asp < 4.0:           continue

                cx, cy = (x0+x1)//2, (y0+y1)//2
                if not (DX0 < cx < DX1 and DY0 < cy < DY1): continue
                if in_excl(cx, cy): continue

                rects.append({
                    'x0':x0,'y0':y0,'x1':x1,'y1':y1,
                    'cx':cx,'cy':cy,'w':fw,'h':fh,'asp':asp,'n':n,
                    'mask': fm.copy()
                })

        rects.sort(key=lambda r: -r['n'])
        dedup_r = []
        used_mask = np.zeros((H, W), dtype=np.uint8)
        for r in rects:
            overlap = np.sum((r['mask'] > 0) & (used_mask > 0))
            if overlap > r['n'] * 0.5: continue
            used_mask[r['mask'] > 0] = 1
            dedup_r.append(r)

        # Apply blue fill strictly to the binary drawing tracking layer
        for r in dedup_r:
            where = (r['mask'] > 0) & (filled_g == 0)
            binary_bgr[where] = binary_bgr[where] * (1 - alpha) + BLUE_BGR * alpha
            filled_g[where] = 1

        binary_bgr = np.clip(binary_bgr, 0, 255).astype(np.uint8)

        # ══════════════════════════════════════════════════════════════════════
        # 2. ROUND PIPE CONNECTORS (Pipe analysis processed on Binary Canvas)
        # ══════════════════════════════════════════════════════════════════════
        contours, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        pipes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 22000: continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            if not (DX0 < cx < DX1 and DY0 < cy < DY1): continue
            if in_excl(cx, cy): continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circ = 4 * np.pi * area / (peri * peri)
            asp  = max(w, h) / max(min(w, h), 1)

            if circ < 0.62 or asp > 1.6 or min(w, h) < 25: continue

            mask2 = np.zeros((H, W), dtype=np.uint8)
            rx, ry = max(w//2 - 6, 4), max(h//2 - 6, 4)
            cv2.ellipse(mask2, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            mc = mask2[y:y+h, x:x+w]
            ic = inv[y:y+h,   x:x+w]
            ms = mc.sum() / 255
            if ms == 0: continue
            dark_pct = np.sum((ic > 0) & (mc > 0)) / ms
            if dark_pct > 0.25: continue

            pipes.append({'cx':cx,'cy':cy,'w':w,'h':h,'area':area,'circ':circ})

        pipes.sort(key=lambda p: -p['area'])
        dedup_p = []
        for p in pipes:
            if not any(abs(p['cx']-d['cx']) < 28 and abs(p['cy']-d['cy']) < 28 for d in dedup_p):
                dedup_p.append(p)

        LW = max(5, W // 1000)
        for p in dedup_p:
            rx, ry = max(p['w']//2, 8), max(p['h']//2, 8)
            mask3 = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(mask3, (p['cx'], p['cy']), (rx, ry), 0, 0, 360, 255, -1)
            
            rf = binary_bgr.astype(np.float32)
            rf[mask3 > 0] = rf[mask3 > 0] * (1 - alpha) + RED_BGR * alpha
            binary_bgr = np.clip(rf, 0, 255).astype(np.uint8)
            
            RED_C = (int(RED_BGR[0]), int(RED_BGR[1]), int(RED_BGR[2]))
            cv2.ellipse(binary_bgr, (p['cx'], p['cy']), (rx, ry), 0, 0, 360, RED_C, LW, cv2.LINE_AA)

        # ══════════════════════════════════════════════════════════════════════
        # 3. REVERT TO ORIGINAL BACKGROUND SHEET FORM
        # ══════════════════════════════════════════════════════════════════════
        logger.info("🔄 Reverting drawing backdrop back to original format...")
        ui_output_img = final_ai_ready_np_raw.copy()

        # Find pixels that are NOT pure white background or pure black lines anymore.
        # This gives us a mask of exactly where your script painted annotations.
        annotation_mask = np.any((binary_bgr != 255) & (binary_bgr != 0), axis=-1)

        # Copy the colorful annotations directly back onto your original layout image
        ui_output_img[annotation_mask] = binary_bgr[annotation_mask]

        # ── 4. DRAW THE LEGEND ON THE FINAL CANVAS ────────────────────────────
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FS   = max(0.65, W / 9000)
        FT   = max(2, W // 3500)
        lx   = W - 760;  ly = H - 190

        cv2.rectangle(ui_output_img, (lx-15, ly-15), (lx+740, ly+170), (255,255,255), -1)
        cv2.rectangle(ui_output_img, (lx-15, ly-15), (lx+740, ly+170), (80, 80, 80),    2)
        cv2.putText(ui_output_img, "LEGEND", (lx, ly+24), FONT, FS*0.9, (30,30,30), FT+1, cv2.LINE_AA)

        items = [
            (tuple(int(v) for v in BLUE_BGR), "Rectangular duct bodies (supply / return trunks)"),
            (tuple(int(v) for v in RED_BGR),  "Round pipe connectors (end caps & transitions)"),
        ]
        for i, (col, label) in enumerate(items):
            iy = ly + 52 + i * 52
            cv2.rectangle(ui_output_img, (lx,    iy), (lx+50, iy+28), col, -1)
            cv2.rectangle(ui_output_img, (lx,    iy), (lx+50, iy+28), (80,80,80), 1)
            cv2.putText(ui_output_img, label, (lx+64, iy+20), FONT, FS*0.72, (30,30,30), FT, cv2.LINE_AA)

        return ui_output_img

    except Exception as e:
        logger.error(f"Error reverting image state: {str(e)}")
        raise