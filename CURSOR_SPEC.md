# OCR Post-Processing Pipeline — Cursor IDE Spec
> Drop this file in your project root as `CURSOR_SPEC.md` or inside `.cursor/rules/postprocessing.md`
> Reference it in Cursor chat: "Follow CURSOR_SPEC.md for this task"

---

## Project Context

This is a **food label data extraction system** built with:
- **OCR Engine:** LightOn OCR-2B (running locally via Ollama)
- **Post-Processing LLM:** Qwen2.5:7b (running locally via Ollama on VM)
- **Barcode Decoder:** pyzbar (runs directly on input image)
- **Backend:** Python / FastAPI
- **Purpose:** Extract structured data from food product label images and return as clean JSON output
- **Scope:** Output only — no database, no persistence, API returns extracted JSON directly to caller

---

## System Design — What This Pipeline Does

```
INPUT:  image file (food product label photo)
OUTPUT: structured JSON with extracted label fields
```

No database. No storage. The FastAPI endpoint receives an image, runs the full pipeline, and returns JSON. That's it.

---

## Target Output Fields

| Field | Type | Notes |
|---|---|---|
| `mrp` | string | e.g. "₹45.00" — extracted by Qwen |
| `net_weight` | string | e.g. "200g", "1.5kg", "500ml" — extracted by Qwen |
| `fssai_lic_no` | string | Always 14 digits — extracted by Qwen, validated by regex |
| `gtin` | string | 8/12/13/14 digit barcode — extracted by pyzbar directly from image |
| `email` | string | Standard email — extracted by Qwen, validated by regex |
| `ingredients` | List[str] | Each ingredient as a separate list item — extracted by Qwen |
| `nutrition` | List[NutritionRow] | Structured rows — extracted by Qwen |
| `confidence` | enum | HIGH / MEDIUM / NEEDS_REVIEW |
| `validation_flags` | List[str] | Any format issues detected post-extraction |
| `barcode_type` | string | e.g. "EAN13" — from pyzbar, null if barcode not found |

### NutritionRow Schema
```python
class NutritionRow(BaseModel):
    name: str
    per_100g: Optional[str] = None
    per_serving: Optional[str] = None
    unit: Optional[str] = None
```

---

## Pipeline Architecture

```
IMAGE FILE
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
LightOn OCR-2B (Ollama)           pyzbar barcode decoder
    │                                     │
    ▼                                     ▼
raw OCR text (string)             GTIN + barcode type
    │                                     │
    ▼                                     │
Qwen2.5:7b (Ollama)                      │
Extracts all fields                       │
from OCR text                             │
    │                                     │
    ▼                                     │
Regex Format Validation                   │
(fssai, email, mrp sanity)               │
    │                                     │
    └──────────────┬──────────────────────┘
                   │
                   ▼
            Merge + Confidence Score
                   │
                   ▼
            JSON Response (API output)
```

**Key design decisions:**
- Qwen handles ALL field extraction — regex is only a post-extraction format validator
- pyzbar decodes GTIN directly from image — never from OCR text (OCR cannot reliably read barcodes)
- pyzbar result always overrides Qwen GTIN if both present — barcode is ground truth
- Pipeline never crashes — on any step failure, that field returns null with a validation flag

---

## File Structure

```
app/
├── postprocessing/
│   ├── __init__.py
│   ├── pipeline.py          ← main entry point, called by API route
│   ├── llm_extractor.py     ← Qwen2.5:7b via Ollama (all text fields)
│   ├── barcode_extractor.py ← pyzbar (GTIN from image directly)
│   ├── validator.py         ← regex format checks post-extraction
│   └── models.py            ← all Pydantic models
├── routes/
│   └── extract.py           ← FastAPI endpoint
└── main.py
```

---

## Environment Variables

```env
OLLAMA_HOST=localhost:11434
OLLAMA_MODEL_OCR=lighton-ocr-2b
OLLAMA_MODEL_EXTRACT=qwen2.5:7b
LOG_LEVEL=INFO
```

No DB url. No storage config. Pipeline is stateless.

---

## models.py

```python
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    NEEDS_REVIEW = "NEEDS_REVIEW"

class NutritionRow(BaseModel):
    name: str
    per_100g: Optional[str] = None
    per_serving: Optional[str] = None
    unit: Optional[str] = None

class LabelData(BaseModel):
    mrp: Optional[str] = None
    net_weight: Optional[str] = None
    fssai_lic_no: Optional[str] = None
    gtin: Optional[str] = None
    barcode_type: Optional[str] = None
    email: Optional[str] = None
    ingredients: Optional[List[str]] = None
    nutrition: Optional[List[NutritionRow]] = None
    confidence: Confidence = Confidence.NEEDS_REVIEW
    validation_flags: Optional[List[str]] = None
```

---

## llm_extractor.py

### Config
```python
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL_EXTRACT", "qwen2.5:7b")
OLLAMA_URL = f"http://{OLLAMA_HOST}/api/generate"

OLLAMA_OPTIONS = {
    "temperature": 0,    # CRITICAL — never change, deterministic output required
    "num_predict": 1500,
    "top_p": 1,
}
```

### Extraction Prompt
```python
EXTRACTION_PROMPT = """
You are extracting structured data from raw OCR text of an Indian food product label.

OCR text is noisy — spelling may be off, line breaks may be mid-word, Hindi/English may be mixed.
Use context clues to identify fields even if formatting is broken.

Field-specific hints:
- MRP: look for "MRP", "M.R.P", "Rs.", "₹" followed by a number. May say "MRP (Incl. of all taxes)"
- Net Weight: look for "Net Wt", "Net Weight", "Nett Weight" followed by g/kg/ml/L
- FSSAI: 14-digit number near "FSSAI", "Lic. No", "Licence No", "Lic No"
- Email: standard email format, often near manufacturer/importer address block
- Ingredients: section after "INGREDIENTS:" — comma or slash separated
- Nutrition: tabular section titled "Nutrition Facts", "Nutritional Information", "Nutrition Info"

Return ONLY this JSON, nothing else, no markdown, no explanation:
{
  "mrp": "₹45.00" or null,
  "net_weight": "200g" or null,
  "fssai_lic_no": "12345678901234" or null,
  "email": "example@brand.com" or null,
  "ingredients": ["Sugar", "Wheat Flour", "Salt"] or null,
  "nutrition": [
    {"name": "Energy", "per_100g": "350 kcal", "per_serving": "175 kcal", "unit": "kcal"},
    {"name": "Protein", "per_100g": "8g", "per_serving": "4g", "unit": "g"}
  ] or null
}

Strict rules:
- fssai_lic_no: return ONLY the 14 digits, no spaces, no dashes, no prefix text
- ingredients: one ingredient per list element, do not split sub-ingredients in brackets
- nutrition names must be normalized to one of:
  Energy, Protein, Total Fat, Saturated Fat, Trans Fat,
  Carbohydrates, Total Sugar, Dietary Fibre, Sodium
- If a value is unclear, missing, or you are not confident: use null — never guess
- Do not add any fields not present in the schema above

OCR TEXT:
{ocr_text}
"""
```

### Extractor Function
```python
import requests, json, logging

logger = logging.getLogger(__name__)

def llm_extract(ocr_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "format": "json",   # forces valid JSON — never remove this
                "stream": False,
                "options": OLLAMA_OPTIONS,
            },
            timeout=60
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        return json.loads(raw)

    except requests.exceptions.Timeout:
        logger.error("Ollama LLM request timed out")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {}
```

**Rules for Cursor:**
- `temperature` must always be 0 — never change this
- `format: "json"` must always be present in the Ollama request body
- On any exception: return empty dict `{}`, never raise, never crash the pipeline
- Never call this function with empty or whitespace-only ocr_text

---

## barcode_extractor.py

### Dependencies
```bash
pip install pyzbar pillow opencv-python
sudo apt-get install libzbar0   # required system library on VM
```

### Extractor with Progressive Preprocessing Fallback
```python
from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

SUPPORTED_BARCODE_TYPES = {"EAN13", "EAN8", "UPCA", "UPCE", "CODE128", "CODE39"}

def extract_gtin_from_image(image_path: str) -> dict:
    """
    Decodes barcode from image using 3-stage fallback preprocessing.
    Returns {"gtin": "...", "barcode_type": "EAN13"} or {"gtin": None, "barcode_type": None}

    Stage 1: Raw image — works ~70% of the time
    Stage 2: CLAHE contrast boost — handles uneven lighting, shiny packaging
    Stage 3: Adaptive threshold — handles shadows, curved labels, low contrast
    """
    for stage, preprocess in enumerate([None, "contrast", "threshold"], 1):
        try:
            result = _decode_image(image_path, preprocess)
            if result:
                logger.info(f"Barcode decoded at stage {stage} ({preprocess or 'raw'})")
                return result
        except Exception as e:
            logger.warning(f"Barcode stage {stage} failed: {e}")

    logger.info("No barcode found in image after all stages")
    return {"gtin": None, "barcode_type": None}


def _decode_image(image_path: str, preprocess: str = None) -> dict:
    if preprocess is None:
        img = Image.open(image_path)
    else:
        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        if preprocess == "contrast":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)

        elif preprocess == "threshold":
            processed = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        img = Image.fromarray(processed)

    barcodes = decode(img)
    for b in barcodes:
        if b.type in SUPPORTED_BARCODE_TYPES:
            return {
                "gtin": b.data.decode("utf-8"),
                "barcode_type": b.type
            }
    return None
```

---

## validator.py

Regex runs AFTER Qwen — only to check format correctness, never to extract.

```python
import re
import logging

logger = logging.getLogger(__name__)

def validate_extracted(data: dict) -> dict:
    """
    Validates format of fields extracted by Qwen.
    Sets invalid fields to None and appends a flag.
    Does NOT re-extract — flags are informational for the API consumer.
    """
    flags = []

    # FSSAI must be exactly 14 digits
    if data.get("fssai_lic_no"):
        cleaned = re.sub(r'\D', '', str(data["fssai_lic_no"]))
        if re.fullmatch(r'\d{14}', cleaned):
            data["fssai_lic_no"] = cleaned  # normalize to pure digits
        else:
            logger.warning(f"FSSAI format invalid: {data['fssai_lic_no']}")
            flags.append("fssai_format_invalid")
            data["fssai_lic_no"] = None

    # Email basic format check
    if data.get("email"):
        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', data["email"]):
            logger.warning(f"Email format invalid: {data['email']}")
            flags.append("email_format_invalid")
            data["email"] = None

    # MRP sanity — must contain a number
    if data.get("mrp"):
        if not re.search(r'\d', data["mrp"]):
            flags.append("mrp_no_numeric_value")
            data["mrp"] = None

    data["validation_flags"] = flags if flags else None
    return data
```

---

## pipeline.py

```python
import logging
from .llm_extractor import llm_extract
from .barcode_extractor import extract_gtin_from_image
from .validator import validate_extracted
from .models import LabelData, Confidence

logger = logging.getLogger(__name__)

CRITICAL_FIELDS = ["mrp", "net_weight", "fssai_lic_no", "ingredients", "nutrition"]

def run_pipeline(ocr_text: str, image_path: str = None) -> LabelData:
    """
    Full extraction pipeline.
    ocr_text:   raw string from LightOn OCR-2B
    image_path: path to original image (needed for barcode decoding)
    Returns:    LabelData (JSON-serializable Pydantic model)
    """

    # Step 1 — LLM extracts all text fields
    llm_result = llm_extract(ocr_text)

    # Step 2 — Validate formats
    validated = validate_extracted(llm_result)

    # Step 3 — GTIN from barcode (overrides LLM if barcode found)
    if image_path:
        barcode = extract_gtin_from_image(image_path)
        validated["gtin"] = barcode.get("gtin")
        validated["barcode_type"] = barcode.get("barcode_type")
    else:
        validated.setdefault("gtin", None)
        validated.setdefault("barcode_type", None)

    # Step 4 — Confidence scoring
    null_count = sum(1 for f in CRITICAL_FIELDS if not validated.get(f))
    has_flags = bool(validated.get("validation_flags"))

    if null_count == 0 and not has_flags:
        confidence = Confidence.HIGH
    elif null_count <= 2 and not has_flags:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.NEEDS_REVIEW

    validated["confidence"] = confidence

    return LabelData(**validated)
```

---

## extract.py (FastAPI Route)

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile, os
from app.postprocessing.pipeline import run_pipeline
from app.postprocessing.models import LabelData

router = APIRouter()

@router.post("/extract", response_model=LabelData)
async def extract_label(image: UploadFile = File(...)):
    """
    Accepts a food label image.
    Returns structured extracted data as JSON.
    No storage — output only.
    """
    suffix = os.path.splitext(image.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # OCR step — plug in your existing LightOn OCR-2B call here
        ocr_text = run_ocr(tmp_path)   # replace with your actual OCR function

        if not ocr_text or not ocr_text.strip():
            raise HTTPException(status_code=422, detail="OCR returned empty text")

        result = run_pipeline(ocr_text=ocr_text, image_path=tmp_path)
        return result

    finally:
        os.unlink(tmp_path)   # always clean up temp file
```

### Sample API Response
```json
{
  "mrp": "₹45.00",
  "net_weight": "200g",
  "fssai_lic_no": "12345678901234",
  "gtin": "8901234567890",
  "barcode_type": "EAN13",
  "email": "care@brand.com",
  "ingredients": ["Sugar", "Wheat Flour", "Edible Vegetable Oil", "Salt"],
  "nutrition": [
    {"name": "Energy", "per_100g": "350 kcal", "per_serving": "175 kcal", "unit": "kcal"},
    {"name": "Protein", "per_100g": "8g", "per_serving": "4g", "unit": "g"},
    {"name": "Total Fat", "per_100g": "12g", "per_serving": "6g", "unit": "g"},
    {"name": "Carbohydrates", "per_100g": "52g", "per_serving": "26g", "unit": "g"},
    {"name": "Sodium", "per_100g": "120mg", "per_serving": "60mg", "unit": "mg"}
  ],
  "confidence": "HIGH",
  "validation_flags": null
}
```

---

## Cursor Coding Rules (Always Follow)

1. **`temperature: 0` on all LLM calls** — never change, deterministic output is required
2. **`format: "json"` always set** in Ollama request — forces valid JSON output from Qwen
3. **Qwen extracts, regex validates** — never use regex to extract field values
4. **pyzbar always overrides Qwen for GTIN** — barcode decoded from image is ground truth
5. **Pipeline never crashes** — every step wrapped in try/except, failures return null for that field
6. **Temp files always deleted** — use try/finally in the route to clean up uploaded images
7. **No DB, no file writes, no persistence** — pipeline is fully stateless, returns JSON only
8. **Never hardcode Ollama URL** — always read from `OLLAMA_HOST` env variable
9. **Empty OCR text** — return HTTP 422, do not call Qwen with empty input
10. **nutrition names always normalized** — must match the standard list in the prompt exactly

---

## VM Deployment & Git Workflow

### Daily Development Loop

```
Local Machine (Cursor IDE)
    │  git push origin main
    ▼
GitHub Repo
    │  SSH into VM + run deploy script
    ▼
VM: git pull + systemd restart
```

### One-Command Deploy from Local

```bash
# On VM — create deploy script once
echo "cd /your/project && git pull origin main && sudo systemctl restart labelextract" > ~/deploy.sh
chmod +x ~/deploy.sh
```

```bash
# From your local terminal after every git push:
ssh user@vm-ip "bash ~/deploy.sh"
```

### Systemd Service (keeps app alive on VM)

```ini
# /etc/systemd/system/labelextract.service
[Unit]
Description=Label Extraction API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/your/project
EnvironmentFile=/your/project/.env
ExecStart=uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable labelextract
sudo systemctl start labelextract
```

### Develop Locally Against VM's Ollama (SSH Tunnel)

If you want to test locally without downloading models:
```bash
# Tunnel VM's Ollama port to your local machine
ssh -L 11434:localhost:11434 user@vm-ip

# Now run FastAPI locally — it will hit Ollama on the VM transparently
uvicorn app.main:app --reload
```

---

## Dependencies

```txt
fastapi
uvicorn
pydantic
requests
pillow
pyzbar
opencv-python
python-multipart
```

```bash
pip install fastapi uvicorn pydantic requests pillow pyzbar opencv-python python-multipart
sudo apt-get install libzbar0
```

---
*Spec version: 2.0 — Full Qwen extraction, pyzbar GTIN, output-only pipeline, no DB*
