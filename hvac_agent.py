import json
import time
import logging
from google import genai
from google.genai import types
from openai import OpenAI
import io
import base64


logger = logging.getLogger("HVAC_Agents")

class HVACAgentSystem:
    def __init__(self, gemini_api_key, mistral_api_key=None):
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_id = "gemini-3.1-flash-lite"
        self.max_retries = 3
        self.mistral_api_key = mistral_api_key

    def safe_llm_call(self, prompt, img, is_json=True):
        for attempt in range(1, self.max_retries + 1):
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json" if is_json else "text/plain",
                    temperature=0.1
                )
                response = self.client.models.generate_content(
                    model=self.model_id, contents=[prompt, img], config=config
                )
                if not response.text: raise ValueError("Empty response")
                return json.loads(response.text) if is_json else response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries: time.sleep(attempt * 5)
                else: raise e

    def estimator_agent(self, img):
        # PROMPT PRESERVED EXACTLY
        prompt = """

        You are a Senior HVAC Duct Extraction Engine.


        Your ONLY task is to extract:

        1. SUPPLY ducts

        2. RETURN ducts


        IGNORE:

        - Equipment

        - RTUs

        - DOAS

        - Exhaust fans

        - Diffusers

        - Room labels

        - Elevation annotations

        - Architectural symbols

        - Exhaust ducts

        - Plumbing

        - Electrical

        - Text not directly related to supply/return ducts

        ==================================================
        CRITICAL CORE HEURISTIC: DIMENSION IS THE ANCHOR
        ==================================================
        1. Every numeric measurement, diameter annotation, or section dimension (e.g., "14ø", "10ø", "4ø", "6ø", "8ø", "22x14") is an explicit indicator of a physical duct run. 
        2. If a measurement label exists, a duct run exists. You MUST locate the exact physical line/pathway attached to or immediately flanking that measurement and extract its bounding box.
        3. Once a measurement text anchor is found, trace the continuous line work stretching away from it until it intersects another component, transitions, or hits an end-cap.


        ==================================================
        PRIMARY OBJECTIVE
        ==================================================
        Extract the COMPLETE supply and return duct topology from the drawing.



        You MUST reconstruct:

        - Main trunks

        - Branch ducts

        - Continuous duct runs

        - T-junctions

        - Reducers

        - Elbows

        - Return branches



        DO NOT extract isolated labels only.



        You MUST trace actual visible duct geometry.



        ==================================================

        DUCT TYPES

        ==================================================



        Allowed values for "type":

        - "Supply"

        - "Return"



        DO NOT output any other type.



        ==================================================

        SMALL DUCT PRIORITY

        ==================================================



        Perform an additional scan for:

        - 4ø

        - 6ø

        - 8ø

        - 10ø



        These branch ducts are commonly missed.



        ==================================================

        UNIQUE IDENTIFIERS

        ==================================================



        FORMAT:

        [Dimension]_[Location]_[Index]



        Examples:

        - "14ø_Dining_1"

        - "8ø_Scullery_2"

        - "10ø_Kitchen_1"



        ==================================================

        BBOX RULES — CRITICAL

        ==================================================



        ALL bbox values MUST:

        - Be normalized between 0.0 and 1.0

        - Follow format:

        [ymin, xmin, ymax, xmax]



        STRICT RULES:

        - No coordinate > 1.0

        - No negative values

        - bbox must wrap the PHYSICAL DUCT

        - NOT the text label



        ==================================================

        CONNECTIVITY RULES

        ==================================================



        Connections must represent REAL physical duct joins.



        Allowed relative_position values:

        - Top

        - Bottom

        - Left

        - Right

        - Centerline

        - End-Cap

        - Top-Left

        - Top-Right

        - Bottom-Left

        - Bottom-Right



        DO NOT:

        - Connect nearby ducts incorrectly

        - Invent hidden connections

        - Collapse all branches into one trunk



        ==================================================

        DATA QUALITY RULES

        ==================================================



        Before finalizing:

        - Remove duplicates

        - Ensure all target_ids exist

        - Ensure trunk continuity

        - Ensure branch connectivity

        - Ensure no disconnected major duct runs



        ==================================================

        OUTPUT FORMAT

        ==================================================



        Return ONLY valid JSON.



        NO:

        - Markdown

        - Explanations

        - Notes

        - Comments

        - Reasoning



        ==================================================

        JSON STRUCTURE

        ==================================================



        [

        {{

            "id": "14ø_Dining_1",

            "type": "Supply",

            "dimension_label": "14ø",

            "location_context": "Dining",



            "bbox": [ymin, xmin, ymax, xmax],



            "connections": [

            {{

                "target_id": "22x14_Main_1",

                "relative_position": "Top"

            }}

            ],



            "confidence": 0.94

        }}

        ]

        """    
        return self.safe_llm_call(prompt, img, is_json=True)

    def auditor_agent(self, img, extracted_json):
        # PROMPT PRESERVED EXACTLY
        prompt = f"""
        [SYSTEM: HVAC DUCTWORK QA/QC AUDITOR]
        [OBJECTIVE: Evaluate the provided JSON manifest against the drawing canvas to verify bounding boxes and duct detection accuracy.]

        ### CRITICAL FOCUS 1: DUCT & MEASUREMENT VERIFICATION
        - **The Anchor Rule**: Every numeric size or diameter callout visible on the blueprint (e.g., "4ø", "6ø", "8ø", "10ø", "14ø", "22x14") represents an absolute physical duct run. 
        - Cross-examine the drawing: Scan for any text size annotations on the canvas. If an annotation exists but its measurement is not logged inside the "JSON TO AUDIT", flag it immediately as a Missing Duct.
        - Ensure only Supply and Return systems are audited. Completely ignore exhaust networks, architectural markers, diffusers, and mechanical equipment.

        ### CRITICAL FOCUS 2: BOUNDING BOX (BBOX) INTEGRITY
        - **Normalization Enforcement**: Bounding box array parameters MUST be normalized values strictly scaling between 0.0 and 1.0. Flag any negative integers or values > 1.0.
        - **Structural Alignment**: The `bbox` coordinate window must explicitly enclose the **physical structural linework lines** of the duct run. If a box is tightly restricted around *only* the alphanumeric text characters instead of wrapping the duct walls, flag it as a BBOX Error.

        ### DATA VALIDATION MATRICES
        JSON TO AUDIT:
        {json.dumps(extracted_json, indent=2)}

        ### OUTPUT RECONCILIATION FORMAT Rules
        If the manifest perfectly captures every duct line and possesses accurate, valid bounding boxes, return EXACTLY this word and nothing else:
        PASSED

        If any structural errors, missing runs, or alignment slips are detected, return a direct, bulleted summary matching this exact schema (omit categories that have no errors):

        ### 1. Missing Ducts
        - [List any uncaptured duct runs identified by size labels or geometry branches]

        ### 2. BBOX Errors
        - [List IDs with misaligned boxes, text-only clippings, or non-normalized numbers]

        ### 3. Connectivity & Orientation Errors
        - [List broken trunk links, duplicate IDs, or incorrect relative_position specs]
        """

        return self.safe_llm_call(prompt, img, is_json=False)

    def review_agent(self, img, initial_data, review_report):
        # PROMPT PRESERVED EXACTLY
        refine_prompt = f"""
        [SYSTEM: HVAC QA/QC CORRECTION ENGINE]
        [CONTEXT: You are reviewing a partially completed HVAC duct network payload and modifying it based on an AUDITOR_REVIEW report.]

        ### CRITICAL CORRECTION HEURISTIC: MEASUREMENT VALIDATION
        1. Cross-reference all numeric dimension labels (e.g., "14ø", "10ø", "4ø", "6ø", "8ø", "22x14") visible on the drawing against the provided "EXISTING DATA MANIFEST".
        2. If a physical measurement or diameter label exists on the canvas but is missing from the existing manifest, it is a verified missing duct. 
        3. Locate the physical line pathways attached to that missed label, calculate its normalized coordinates, and add the missing object node to the payload.

        ### PRIMARY OBJECTIVES
        auditor report:
        {review_report}
        - Locate and inject missing elements highlighted by the auditor report (prioritizing "4ø", "6ø", "8ø", "10ø" branches and return runs).
        - Correct inaccurate bounding boxes, misaligned system classifications, or invalid connection keys.
        - Preserve all existing correct nodes. Do not modify valid IDs or wipe out correct trace items.

        ### DATA INGESTION MATRIX
        EXISTING DATA MANIFEST:
        {json.dumps(initial_data, ensure_ascii=False)}


        ### GEOMETRIC TRACING RULES
        - Bounding Boxes: Must follow the [ymin, xmin, ymax, xmax] format using normalized floats between 0.0 and 1.0. Enclose the physical structural line work, not just the text annotation.
        - Topology Network: Ensure connection entries use valid targets. Allowed alignment tags are: "Top", "Bottom", "Left", "Right", "Centerline", "End-Cap", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right".

        ### COMPLIANCE AND FORMATTING
        - Output must be a single, clean, valid RFC 8259 compliant JSON array.
        - Do not output Markdown code blocks, backticks (no ```json wrapper), conversation text, notes, or explanations.
        - Do not output ReAct scratchpad tokens (such as "Thought:", "Action:", "Observation:"). Perform all analytical filtering internally and return only the raw JSON string payload.

        ### TARGET JSON STRUCTURE
        [
        {{
            "id": "14ø_Dining_1",
            "type": "Supply",
            "dimension_label": "14ø",
            "location_context": "Dining",
            "bbox": [0.1234, 0.5678, 0.2345, 0.6789],
            "connections": [
            {{
                "target_id": "22x14_Main_1",
                "relative_position": "Top"
            }}
            ],
            "confidence": 0.95
        }}
        ]
        """
        return self.safe_llm_call(refine_prompt, img, is_json=True)
    


    def antagonist_review_agent(self, img, initial_data, review_report):
        # PROMPT PRESERVED EXACTLY
        refine_prompt = f"""
        You are a Senior HVAC QA/QC Correction Engineer specializing ONLY in:
        - Missing duct detection
        - Supply duct tracing
        - Return duct tracing
        - HVAC topology correction
        - Duct continuity validation

        IGNORE:
        - Equipment
        - RTUs
        - DOAS
        - Exhaust systems
        - Diffusers
        - Architectural annotations
        - Elevation notes
        - Plumbing
        - Electrical
        - Non-HVAC symbols

        ==================================================
        PRIMARY OBJECTIVE
        ==================================================

        Your PRIMARY objective is to identify ALL missing SUPPLY and RETURN ducts based on the AUDITOR_REVIEW and the drawing image.

        Your MOST IMPORTANT responsibility is:
        - detecting ducts that are NOT present in the existing JSON
        - identifying missed branch ducts
        - identifying missed return ducts
        - identifying missed continuation segments
        - identifying incomplete trunk geometry

        You MUST aggressively search for ducts that were overlooked previously.

        SECONDARY objectives:
        - fix incorrect topology
        - fix incorrect bbox values
        - fix incorrect relative_position values

        You MUST preserve already-correct ducts.

        DO NOT regenerate the entire JSON unnecessarily.

        ==================================================
        CRITICAL HVAC UNDERSTANDING
        ==================================================

        The existing JSON already contains partially correct ducts.

        The most common failures are:
        - small branch ducts missing
        - short continuation segments missing
        - return branches missing
        - side takeoffs missing
        - disconnected trunk segments
        - ducts hidden in crowded geometry
        - ducts near elbows/T-junctions
        - parallel adjacent ducts being skipped

        Your PRIMARY task is to find these missing ducts.

        ==================================================
        REACT EXECUTION FRAMEWORK
        ==================================================

        You MUST internally follow this workflow:

        --------------------------------------------------
        THOUGHT
        --------------------------------------------------

        Carefully analyze the AUDITOR_REVIEW.

        For EACH issue determine:
        - which ducts are likely missing
        - which trunk/branch is incomplete
        - which geometry region requires re-scan
        - where continuity is broken

        Prioritize:
        1. Missing ducts
        2. Missing branches
        3. Missing return ducts
        4. Missing trunk continuations

        --------------------------------------------------
        ACTION
        --------------------------------------------------

        Perform AGGRESSIVE targeted visual re-scan around:
        - auditor-mentioned regions
        - elbows
        - T-junctions
        - trunk endpoints
        - crowded geometry
        - branch takeoffs
        - parallel ducts
        - partially captured segments

        Search specifically for:
        - uncaptured duct geometry
        - disconnected segments
        - small branch ducts
        - missed returns
        - missed continuation paths

        --------------------------------------------------
        OBSERVATION
        --------------------------------------------------

        Verify visually:
        - continuous duct geometry
        - branch continuity
        - trunk continuity
        - parent-child relationships
        - visible physical joins

        Determine:
        - which ducts are missing from JSON
        - which branches are incomplete
        - which continuation segments are absent

        --------------------------------------------------
        CORRECTION
        --------------------------------------------------

        Apply ONLY necessary corrections:
        - add missing ducts
        - add missing branches
        - add missing return ducts
        - repair trunk continuity
        - fix incorrect topology
        - fix incorrect bbox
        - fix incorrect relative_position

        Preserve already-correct ducts.

        DO NOT:
        - delete correct ducts
        - rename correct IDs unnecessarily
        - rebuild unrelated sections

        --------------------------------------------------
        VALIDATION
        --------------------------------------------------

        Before finalizing:
        - verify ALL visible supply ducts are captured
        - verify ALL visible return ducts are captured
        - verify no branch ducts are missing
        - verify no trunk continuation is missing
        - verify no disconnected duct segments exist
        - verify all bbox values are normalized
        - verify all target_ids exist

        ==================================================
        HIGHEST PRIORITY SEARCH
        ==================================================

        Perform EXTRA aggressive scanning for:

        1. SMALL DUCTS
        - 4ø ducts
        - 6ø ducts
        - 8ø ducts
        - 10ø ducts

        2. RETURN DUCTS
        - short returns
        - side returns
        - crowded return geometry
        - partially visible returns

        3. TRUNK CONTINUATIONS
        - disconnected horizontal runs
        - disconnected vertical runs
        - interrupted trunks
        - continuation segments after elbows

        4. BRANCH TAKEOFFS
        - small side branches
        - T-junction branches
        - parallel adjacent branches
        - short branch connectors

        ==================================================
        CRITICAL DETECTION RULE
        ==================================================

        DO NOT only detect labels.

        You MUST detect:
        - actual visible duct geometry
        - continuous duct paths
        - branch continuity
        - physical trunk geometry

        Even if:
        - OCR text is weak
        - labels are partially visible
        - no nearby label exists

        If visible duct geometry exists, capture it.

        ==================================================
        TOPOLOGY RULES
        ==================================================

        Connections MUST represent:
        - actual physical joins
        - branch hierarchy
        - trunk continuity
        - visible geometry

        DO NOT:
        - connect nearby ducts incorrectly
        - invent hidden connections
        - oversimplify topology

        ==================================================
        BBOX RULES — CRITICAL
        ==================================================

        ALL bbox coordinates MUST:
        - be normalized between 0.0 and 1.0
        - never exceed 1.0
        - never be negative

        FORMAT:
        [ymin, xmin, ymax, xmax]

        bbox MUST:
        - wrap PHYSICAL DUCT GEOMETRY
        - NOT only text labels
        - visually align with the duct segment

        ==================================================
        RELATIVE POSITION RULES
        ==================================================

        Allowed values:
        - Top
        - Bottom
        - Left
        - Right
        - Centerline
        - End-Cap
        - Top-Left
        - Top-Right
        - Bottom-Left
        - Bottom-Right

        relative_position MUST match actual geometry.

        ==================================================
        OUTPUT RULES
        ==================================================

        Return ONLY valid JSON.

        DO NOT return:
        - markdown
        - explanations
        - notes
        - comments
        - reasoning text
        - thought/action text

        ==================================================
        EXISTING JSON
        ==================================================

        {json.dumps(initial_data, ensure_ascii=False)}

        ==================================================
        AUDITOR REVIEW
        ==================================================

        {review_report}

        ==================================================
        FINAL INSTRUCTION
        ==================================================

        Your success depends PRIMARILY on:
        - identifying ALL missing ducts
        - identifying ALL missing branches
        - identifying ALL missing returns
        - identifying ALL missing continuation segments

        Perform a FINAL aggressive scan for uncaptured duct geometry before returning JSON.

        Return ONLY the FINAL corrected JSON.
        """

        endpoint = "https://project-elpis-resource.services.ai.azure.com/openai/v1/"
        deployment_name = "Mistral-Large-3"

        client = OpenAI(
            base_url=f"{endpoint}",
            api_key=self.mistral_api_key
        )

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        
        # 2. FIXED: Encode the raw bytes into a clean base64 string
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        completion = client.chat.completions.create(
        model=deployment_name,
        messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": refine_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )

        # 1. Extract the raw string from the completion object
        raw_content = completion.choices[0].message.content
         
        # 2. Safety Step: Strip markdown backticks if the model accidentally includes them
        raw_content = raw_content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1].split("```")[0].strip()

        try:
            # 3. Parse the string into a native Python dictionary or list
            parsed_json = json.loads(raw_content)
            
            # Ready for downstream graph construction / pipeline logic
            print("Successfully parsed JSON:")
            print(parsed_json)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse text as JSON: {e}")
            print(f"Raw Content received was: {raw_content}")

        return parsed_json