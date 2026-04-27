from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
from typing import Any, Final, Optional

import ollama
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from PIL import Image  # type: ignore[import-not-found]
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore[import-not-found]

    BARCODE_LIB_AVAILABLE: Final = True
except Exception:
    Image = None  # type: ignore[assignment]
    zbar_decode = None  # type: ignore[assignment]
    BARCODE_LIB_AVAILABLE: Final = False

OCR_MODEL: Final = os.getenv(
    "OLLAMA_MODEL_OCR",
    os.getenv("OLLAMA_MODEL", "maternion/LightOnOCR-2"),
)
EXTRACT_MODEL: Final = os.getenv("OLLAMA_MODEL_EXTRACT", "qwen2.5:7b")
MAX_BYTES: Final = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
OLLAMA_TIMEOUT_SECONDS: Final = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_HEALTH_TIMEOUT_SECONDS: Final = float(
    os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "5")
)
BARCODE_TIMEOUT_SECONDS: Final = float(os.getenv("BARCODE_TIMEOUT_SECONDS", "5"))
ALLOWED_TYPES: Final = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)
PROMPT: Final = os.getenv(
    "OCR_PROMPT",
    "Transcribe the text in this image exactly as it appears.",
)
EXTRACTION_PROMPT: Final = """You are extracting structured data from raw OCR text of an Indian food product label.

OCR text can be noisy. Return ONLY valid JSON with this exact schema:
{
  "mrp": "45.00" or null,
  "net_weight": "200g" or null,
  "fssai_lic_no": "12345678901234" or null,
  "gtin": "8901234567890" or null,
  "email": "support@brand.com" or null,
  "ingredients": ["ingredient1", "ingredient2"] or null,
  "nutrition": [{"name":"Energy","per_100g":"350 kcal","per_serving":"175 kcal","unit":"kcal"}] or null
}

Rules:
- If unsure, return null and never guess.
- fssai_lic_no must contain only 14 digits.
- Keep mrp numeric only (no currency symbol).
- Do not include markdown.
- Do not include any keys other than the schema above.

OCR TEXT:
{ocr_text}
"""
_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS: Final[list[str]] = [o.strip() for o in _cors.split(",") if o.strip()]

app = FastAPI(title="OCR Extract API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    try:
        await asyncio.wait_for(
            asyncio.to_thread(ollama.list),
            timeout=OLLAMA_HEALTH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Ollama health timeout after {OLLAMA_HEALTH_TIMEOUT_SECONDS:.0f}s",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unreachable: {e}",
        ) from e
    return {
        "status": "ok",
        "ollama": "reachable",
        "ocr_model": OCR_MODEL,
        "extract_model": EXTRACT_MODEL,
    }


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _extract_emails(text: str) -> list[str]:
    emails = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.I)
    seen: set[str] = set()
    out: list[str] = []
    for e in emails:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _extract_phones(text: str) -> list[str]:
    """Extract phone numbers using a strict Indian-mobile pattern only.

    The old broad fallback matched batch codes, serial numbers, and dates.
    Now we only accept numbers that look like genuine Indian phone numbers:
    10-digit numbers starting with 6-9, or +91-prefixed equivalents.
    """
    matches = re.findall(
        r"(?:\+91[\s-]?|0)?[6-9]\d{9}",
        text,
    )
    cleaned: list[str] = []
    seen: set[str] = set()
    for match in matches:
        digits = re.sub(r"\D", "", match)
        # Strip leading country code to normalise
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        if len(digits) != 10:
            continue
        if digits not in seen:
            seen.add(digits)
            cleaned.append(match.strip())
    return cleaned


def _gtin_checksum_valid(digits: str) -> bool:
    """Return True if `digits` passes the GS1 check-digit algorithm.

    Covers EAN-8 (8), UPC-A (12), EAN-13 (13), and GTIN-14 (14).
    The last digit is the check digit; the rest are the payload.
    """
    if not digits.isdigit() or len(digits) not in {8, 12, 13, 14}:
        return False
    total = sum(
        int(d) * (3 if i % 2 == 0 else 1)
        for i, d in enumerate(reversed(digits[:-1]))
    )
    expected_check = (10 - (total % 10)) % 10
    return expected_check == int(digits[-1])


def _extract_fssai(text: str) -> Optional[str]:
    """Extract a 14-digit FSSAI licence number.

    Only returns a value when a recognised keyword (FSSAI / Lic. No.) is
    present nearby.  The old blind "any 14-digit number" fallback has been
    removed because it produced false positives (serial numbers, order codes).
    """
    label_match = re.search(
        r"\b(?:fssai|lic(?:ence|ense)?(?:\s*no\.?)?)\b[^0-9]{0,20}((?:\d[\s-]?){14})\b",
        text,
        flags=re.I,
    )
    if label_match:
        digits = re.sub(r"\D", "", label_match.group(1))
        if len(digits) == 14:
            return digits
    return None


def _extract_gtin(text: str, fssai: Optional[str]) -> Optional[str]:
    """Extract a GTIN from OCR text using a strict three-tier strategy.

    Tier 1 — explicit keyword label (GTIN / EAN / UPC / Barcode): trusted as-is.
    Tier 2 — number alone on its own line: common when OCR reads a barcode
              caption; accepted only if it also passes the GS1 checksum.
    Tier 3 — any number in the text: accepted ONLY when the GS1 checksum
              passes, so random phone numbers, dates, and batch codes are
              rejected.

    Returns None (leave blank) when no confident match is found.
    """
    _VALID_LENGTHS = {8, 12, 13, 14}

    # Tier 1: keyword-labelled — the label is our confidence signal
    labeled = re.search(
        r"\b(?:gtin|ean|upc|barcode)\b[^0-9]{0,20}([0-9][0-9\s-]{7,20})",
        text,
        flags=re.I,
    )
    if labeled:
        digits = re.sub(r"\D", "", labeled.group(1))
        if len(digits) in _VALID_LENGTHS and digits != fssai:
            return digits

    # Tier 2: lone number on its own line + checksum
    for m in re.finditer(r"(?m)^\s*(\d{8,14})\s*$", text):
        digits = m.group(1)
        if len(digits) in _VALID_LENGTHS and digits != fssai and _gtin_checksum_valid(digits):
            return digits

    # Tier 3: number anywhere in text — checksum required to avoid false positives
    for token in re.findall(r"\b(\d{8,14})\b", text):
        if len(token) in _VALID_LENGTHS and token != fssai and _gtin_checksum_valid(token):
            return token

    return None  # don't guess — leave blank


def _is_valid_gtin(value: str, fssai: Optional[str]) -> bool:
    return (
        value.isdigit()
        and len(value) in {8, 12, 13, 14}
        and value != fssai
        and _gtin_checksum_valid(value)
    )


def _decode_barcodes_from_bytes(body: bytes) -> list[str]:
    if not BARCODE_LIB_AVAILABLE:
        return []

    try:
        with Image.open(io.BytesIO(body)) as image:
            decoded = zbar_decode(image)
    except Exception:
        return []

    found: list[str] = []
    seen: set[str] = set()
    for item in decoded:
        raw = item.data.decode("utf-8", errors="ignore").strip()
        if not raw:
            continue

        candidates = re.findall(r"\d{8,14}", raw) or [raw]
        for candidate in candidates:
            digits = re.sub(r"\D", "", candidate)
            if not digits:
                continue
            if digits not in seen:
                seen.add(digits)
                found.append(digits)
    return found


async def _decode_barcodes_with_timeout(
    body: bytes,
    fail_on_timeout: bool = False,
) -> list[str]:
    if not BARCODE_LIB_AVAILABLE:
        return []
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_decode_barcodes_from_bytes, body),
            timeout=BARCODE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as e:
        if fail_on_timeout:
            raise HTTPException(
                status_code=504,
                detail=f"Barcode decoding timed out after {BARCODE_TIMEOUT_SECONDS:.0f}s",
            ) from e
        return []


def _pick_gtin_from_barcodes(barcodes: list[str], fssai: Optional[str]) -> Optional[str]:
    for code in barcodes:
        if _is_valid_gtin(code, fssai):
            return code
    return None


def _extract_mrp(text: str) -> Optional[str]:
    """Extract MRP — only returns a value when the MRP keyword is present.

    The old currency-symbol-only fallback (₹ / Rs.) has been removed because
    it matched prices on ingredient tables, per-unit costs, etc.
    """
    match = re.search(
        r"\bmrp\b[^0-9]{0,40}(?:rs\.?|inr|₹)?\s*([0-9]+(?:\.[0-9]{1,2})?)\s*(?:/-)?",
        text,
        flags=re.I,
    )
    return match.group(1) if match else None


def _extract_net_wt(text: str) -> Optional[str]:
    """Extract net weight / quantity / volume.

    Handles all common label wordings (Net Wt, Net Weight, Net Quantity,
    Net Qty, Net Content, Net Vol, Net Volume) and tolerates the value being
    on a separate line from the keyword (OCR often splits label and value).
    """
    _UNITS = r"(?:kgs?|kg|gms?|gm|mg|ml|ltrs?|ltr|litres?|g|l|oz|lbs?)"
    match = re.search(
        r"\bnet\s*(?:wt\.?|weight|qty\.?|quantity|content|vol\.?|volume)"
        r"[\s\S]{0,60}?"          # non-greedy: cross-line gap, stops at first number
        r"([0-9]+(?:\.[0-9]+)?\s*" + _UNITS + r")\b",
        text,
        flags=re.I,
    )
    if match:
        return _normalize_space(match.group(1))
    return None


def _extract_section(
    text: str,
    start_keywords: tuple[str, ...],
    stop_keywords: tuple[str, ...],
    max_lines: int = 14,
) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = -1
    initial = ""

    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(keyword in lowered for keyword in start_keywords):
            start_idx = idx
            if ":" in line:
                initial = line.split(":", 1)[1].strip()
            break

    if start_idx == -1:
        return None

    collected: list[str] = [initial] if initial else []
    for line in lines[start_idx + 1 :]:
        lowered = line.lower()
        if any(keyword in lowered for keyword in stop_keywords):
            break
        if line:
            collected.append(line)
        if len(collected) >= max_lines:
            break

    section = _normalize_space(" ".join(collected))
    return section or None


def _extract_html_table(text: str) -> Optional[str]:
    match = re.search(r"<table\b[\s\S]*?</table>", text, flags=re.I)
    if not match:
        return None
    return match.group(0).strip()


def _extract_product_fields(
    text: str,
    barcode_gtin: Optional[str] = None,
) -> dict[str, Optional[str]]:
    fssai = _extract_fssai(text)
    emails = _extract_emails(text)
    phones = _extract_phones(text)
    table_html = _extract_html_table(text)
    gtin_from_text = _extract_gtin(text, fssai)
    gtin = barcode_gtin or gtin_from_text

    nutritional = table_html or _extract_section(
        text=text,
        start_keywords=(
            "nutrition facts",
            "nutritional facts",
            "nutrition information",
            "nutritional information",
            "nutrition",
        ),
        stop_keywords=(
            "ingredients",
            "allergen",
            "storage",
            "manufacturer",
            "marketed by",
            "fssai",
            "directions",
            "serving suggestion",
        ),
    )
    ingredients = _extract_section(
        text=text,
        start_keywords=("ingredients", "ingredient"),
        stop_keywords=(
            "nutrition",
            "nutritional",
            "allergen",
            "storage",
            "manufacturer",
            "marketed by",
            "fssai",
            "directions",
            "serving suggestion",
        ),
    )

    return {
        "mrp": _extract_mrp(text),
        "gtin": gtin,
        "GTIN": gtin,
        "gtin_source": "barcode" if barcode_gtin else ("ocr" if gtin_from_text else "none"),
        "fssai": fssai,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "net_wt": _extract_net_wt(text),
        "nutritional": nutritional,
        "nutritable": nutritional,
        "ingredients": ingredients,
    }


def _validate_image_file(file: UploadFile, body: bytes) -> None:
    if not file.content_type or file.content_type not in ALLOWED_TYPES:
        allowed = ", ".join(sorted(ALLOWED_TYPES))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed}",
        )

    if len(body) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_BYTES // (1024 * 1024)} MB)",
        )


async def _extract_text_from_bytes(body: bytes, file_name: str) -> str:
    suffix = os.path.splitext(file_name or "")[1] or ".png"
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.write(fd, body)
        os.close(fd)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.generate,
                    model=OCR_MODEL,
                    prompt=PROMPT,
                    images=[tmp_path],
                ),
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as e:
            raise HTTPException(
                status_code=504,
                detail=f"OCR timed out after {OLLAMA_TIMEOUT_SECONDS:.0f}s",
            ) from e
        except Exception as e:
            msg = str(e).lower()
            if "connection" in msg or "refused" in msg or "timeout" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=f"Ollama error: {e}",
                ) from e
            raise HTTPException(
                status_code=502,
                detail=f"Model error: {e}",
            ) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return response.get("response", "")


def _strip_code_fences(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_ingredients(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) if parts else None
    return _coerce_optional_str(value)


def _normalize_nutrition(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return _coerce_optional_str(value)
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return _coerce_optional_str(value)


async def _extract_structured_with_qwen(ocr_text: str) -> dict[str, Any]:
    if not ocr_text.strip():
        return {}
    prompt = EXTRACTION_PROMPT.format(ocr_text=ocr_text)
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                ollama.generate,
                model=EXTRACT_MODEL,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0,
                    "top_p": 1,
                    "num_predict": 1500,
                },
            ),
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        raw = _strip_code_fences(str(response.get("response", "{}")))
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict[str, Any]:
    body = await file.read()
    _validate_image_file(file, body)

    text = await _extract_text_from_bytes(body, file.filename or "")
    return {"text": text, "model": OCR_MODEL}


@app.post("/api/extract/product-info")
async def extract_product_info(file: UploadFile = File(...)) -> dict[str, Any]:
    body = await file.read()
    _validate_image_file(file, body)

    barcodes = await _decode_barcodes_with_timeout(body)
    text = await _extract_text_from_bytes(body, file.filename or "")
    fssai = _extract_fssai(text)
    barcode_gtin = _pick_gtin_from_barcodes(barcodes, fssai)
    fields = _extract_product_fields(text, barcode_gtin=barcode_gtin)

    return {
        "fields": fields,
        "barcode": {
            "available": BARCODE_LIB_AVAILABLE,
            "decoded": barcodes,
            "gtin": barcode_gtin,
        },
        "text": text,
        "model": OCR_MODEL,
    }


@app.post("/api/extract/required-fields")
async def extract_required_fields(file: UploadFile = File(...)) -> dict[str, Optional[str]]:
    body = await file.read()
    _validate_image_file(file, body)

    barcodes = await _decode_barcodes_with_timeout(body)
    text = await _extract_text_from_bytes(body, file.filename or "")
    fssai = _extract_fssai(text)
    barcode_gtin = _pick_gtin_from_barcodes(barcodes, fssai)
    return _extract_product_fields(text, barcode_gtin=barcode_gtin)


@app.post("/api/decode-barcode")
async def decode_barcode(file: UploadFile = File(...)) -> dict[str, Any]:
    body = await file.read()
    _validate_image_file(file, body)

    if not BARCODE_LIB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Barcode decoder unavailable. Install pyzbar and system zbar library.",
        )

    barcodes = await _decode_barcodes_with_timeout(body, fail_on_timeout=True)
    gtin = _pick_gtin_from_barcodes(barcodes, None)
    return {
        "decoded": barcodes,
        "gtin": gtin,
        "count": len(barcodes),
    }


# ---------------------------------------------------------------------------
# /api/parse — clean structured JSON, no raw OCR text
# ---------------------------------------------------------------------------


def _validate_fields(fields: dict[str, Optional[str]]) -> dict[str, Any]:
    """Return validation result for FSSAI and MRP."""
    errors: list[str] = []

    fssai = fields.get("fssai") or ""
    fssai_valid = bool(re.fullmatch(r"\d{14}", fssai)) if fssai else None
    if fssai and not fssai_valid:
        errors.append(f"FSSAI '{fssai}' is not exactly 14 digits")

    mrp_raw = fields.get("mrp") or ""
    mrp_valid: Optional[bool] = None
    if mrp_raw:
        try:
            mrp_valid = float(mrp_raw) > 0
            if not mrp_valid:
                errors.append(f"MRP '{mrp_raw}' must be a positive number")
        except ValueError:
            mrp_valid = False
            errors.append(f"MRP '{mrp_raw}' is not a valid number")

    return {
        "fssai_valid": fssai_valid,
        "mrp_valid": mrp_valid,
        "errors": errors,
    }


@app.post("/api/parse")
async def parse_product(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a product label image and receive structured product data as JSON.

    Returns only post-processed fields — no raw OCR text.

    Fields returned
    ---------------
    gtin            : GTIN/EAN/UPC barcode number (barcode scan preferred, OCR fallback)
    gtin_source     : "barcode" | "ocr" | "none"
    fssai           : 14-digit FSSAI licence number
    mrp             : MRP value as string (digits only, no currency symbol)
    net_weight      : Net weight/content with unit (e.g. "800g", "1.5kg")
    ingredients     : Ingredients list as a single string
    nutrition       : Nutritional information (HTML table or plain text block)
    email           : First email address found on the label
    phone           : First phone number found on the label
    barcode_decoder_available : Whether pyzbar/zbar is installed on this server
    validation      : { fssai_valid, mrp_valid, errors }
    """
    body = await file.read()
    _validate_image_file(file, body)

    barcodes = await _decode_barcodes_with_timeout(body)
    text = await _extract_text_from_bytes(body, file.filename or "")

    fssai = _extract_fssai(text)
    barcode_gtin = _pick_gtin_from_barcodes(barcodes, fssai)
    regex_fields = _extract_product_fields(text, barcode_gtin=barcode_gtin)
    llm_fields = await _extract_structured_with_qwen(text)

    llm_fssai = re.sub(r"\D", "", str(llm_fields.get("fssai_lic_no") or ""))
    if len(llm_fssai) != 14:
        llm_fssai = ""

    llm_gtin_raw = _coerce_optional_str(llm_fields.get("gtin"))
    llm_gtin = None
    if llm_gtin_raw:
        llm_digits = re.sub(r"\D", "", llm_gtin_raw)
        if _is_valid_gtin(llm_digits, llm_fssai or fssai):
            llm_gtin = llm_digits

    merged_gtin = barcode_gtin or llm_gtin or regex_fields.get("gtin")
    if barcode_gtin:
        gtin_source = "barcode"
    elif llm_gtin:
        gtin_source = "qwen"
    elif regex_fields.get("gtin"):
        gtin_source = "ocr"
    else:
        gtin_source = "none"

    fields: dict[str, Optional[str]] = {
        "gtin": merged_gtin,
        "fssai": llm_fssai or regex_fields.get("fssai"),
        "mrp": _coerce_optional_str(llm_fields.get("mrp")) or regex_fields.get("mrp"),
        "net_wt": _coerce_optional_str(llm_fields.get("net_weight")) or regex_fields.get("net_wt"),
        "ingredients": _normalize_ingredients(llm_fields.get("ingredients")) or regex_fields.get("ingredients"),
        "nutritional": _normalize_nutrition(llm_fields.get("nutrition")) or regex_fields.get("nutritional"),
        "email": _coerce_optional_str(llm_fields.get("email")) or regex_fields.get("email"),
        "phone": regex_fields.get("phone"),
    }

    result: dict[str, Any] = {
        "gtin":                      fields.get("gtin"),
        "gtin_source":               gtin_source,
        "fssai":                     fields.get("fssai"),
        "mrp":                       fields.get("mrp"),
        "net_weight":                fields.get("net_wt"),
        "ingredients":               fields.get("ingredients"),
        "nutrition":                 fields.get("nutritional"),
        "email":                     fields.get("email"),
        "phone":                     fields.get("phone"),
        "barcode_decoder_available": BARCODE_LIB_AVAILABLE,
        "validation":                _validate_fields(fields),
        "extract_model":             EXTRACT_MODEL,
    }
    return result

