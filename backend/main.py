from __future__ import annotations

import asyncio
import io
import json
import time
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
OLLAMA_TIMEOUT_SECONDS: Final = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240"))
OLLAMA_HEALTH_TIMEOUT_SECONDS: Final = float(
    os.getenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "5")
)
BARCODE_TIMEOUT_SECONDS: Final = float(os.getenv("BARCODE_TIMEOUT_SECONDS", "5"))
EXTRACT_NUM_PREDICT: Final = int(os.getenv("EXTRACT_NUM_PREDICT", "600"))
ALLOWED_TYPES: Final = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)
PROMPT: Final = os.getenv(
    "OCR_PROMPT",
    "Transcribe the text in this image exactly as it appears.",
)
EXTRACTION_PROMPT: Final = """You are extracting structured data from raw OCR text of an Indian food product label.

OCR text can be noisy. Return ONLY valid JSON with this exact schema:
{{
  "brand_name": "Haldiram's" or null,
  "product_name": "Delhi Mix" or null,
  "mrp": "45.00" or null,
  "net_weight": "200g" or null,
  "best_before": "06/09/2015" or null,
  "fssai_lic_no": "12345678901234" or null,
  "gtin": "8901234567890" or null,
  "email": "support@brand.com" or null,
  "ingredients": ["ingredient1", "ingredient2"] or null,
  "nutrition": [{{"name":"Energy","per_100g":"350 kcal","per_serving":"175 kcal","unit":"kcal"}}] or null
}}

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

# OpenAPI lives under /api/* so nginx can proxy a single `location /api/` block;
# `/docs` at site root would otherwise be handled by the SPA (try_files → index.html).
app = FastAPI(
    title="OCR Extract API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

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
    """Extract contact numbers commonly printed on product labels.

    Supported patterns:
    - Toll-free: 1800-208-2663, 1800 208 2663, 18002082663
    - Mobile: +91 98765 43210, 09876543210, 9876543210
    - Landline with STD code: 022-23456789, 080 41234567
    """
    patterns = (
        # Toll-free numbers (most common on FMCG packs)
        r"\b(?:1800|1[\s-]?800)(?:[\s-]?\d{1,4}){2,4}\b",
        # +91 / 0-prefixed or plain 10-digit Indian mobile
        r"\b(?:\+91[\s-]?|0)?[6-9]\d{9}\b",
        # Landline with STD code and subscriber number
        r"\b0\d{2,4}[\s-]?\d{6,8}\b",
    )

    seen: set[str] = set()
    cleaned: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(0).strip()
            digits = re.sub(r"\D", "", raw)

            # Canonical keys for de-duplication and validation.
            canonical: Optional[str] = None
            if digits.startswith("1800") and 8 <= len(digits) <= 12:
                canonical = digits
            elif digits.startswith("800") and 7 <= len(digits) <= 11:
                # Handles forms like "1-800-10-22-221" where separators drop the leading 1.
                canonical = f"1{digits}"
            elif digits.startswith("91") and len(digits) == 12 and digits[2] in "6789":
                canonical = digits[2:]
            elif len(digits) == 10 and digits[0] in "6789":
                canonical = digits
            elif digits.startswith("0") and 9 <= len(digits) <= 12:
                canonical = digits

            if not canonical:
                continue
            if canonical in seen:
                continue

            seen.add(canonical)
            cleaned.append(raw)

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
    """Extract MRP with OCR-friendly fallbacks.

    Priority:
    1) Explicit MRP-labelled value (most reliable)
    2) Currency token + `/-` style (`Re 20/-`, `Rs. 20/-`, `₹20/-`)
    3) Bare `20/-` fallback (common on Indian labels when OCR drops symbols)
    """
    labeled = re.search(
        r"\b(?:mrp|m\.r\.p\.?)\b[^0-9]{0,40}(?:rs\.?|re\.?|inr|₹)?\s*([0-9]+(?:\.[0-9]{1,2})?)\s*(?:/-)?",
        text,
        flags=re.I,
    )
    if labeled:
        return labeled.group(1)

    currency_slash = re.search(
        r"\b(?:rs\.?|re\.?|inr|₹)\s*([0-9]+(?:\.[0-9]{1,2})?)\s*/-",
        text,
        flags=re.I,
    )
    if currency_slash:
        return currency_slash.group(1)

    # Common OCR output: plain "Rs. 80.00" without "/-".
    currency_plain = re.search(
        r"\b(?:rs?|re|inr|₹)\.?\s*[:=]?\s*([0-9]+(?:\.[0-9]{1,2})?)\b",
        text,
        flags=re.I,
    )
    if currency_plain:
        return currency_plain.group(1)

    bare_slash = re.search(
        r"\b([0-9]+(?:\.[0-9]{1,2})?)\s*/-",
        text,
        flags=re.I,
    )
    if bare_slash:
        return bare_slash.group(1)

    return None


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


def _extract_best_before(text: str) -> Optional[str]:
    match = re.search(
        r"\bbest\s*before\b[^A-Z0-9]{0,20}([A-Z0-9][A-Z0-9\s:/\-]{2,30})",
        text,
        flags=re.I,
    )
    if not match:
        return None
    value = _normalize_space(match.group(1))
    value = re.split(r"\b(?:batch|mfg|mrp|net)\b", value, maxsplit=1, flags=re.I)[0].strip(" ,;:-")
    return value or None


def _extract_brand_product(text: str) -> tuple[Optional[str], Optional[str]]:
    lines = [_normalize_space(line) for line in text.splitlines() if _normalize_space(line)]
    if not lines:
        return None, None

    skip = (
        "nutritional",
        "nutrition",
        "ingredients",
        "fssai",
        "barcode",
        "mrp",
        "best before",
        "net wt",
        "net qty",
    )
    candidates: list[str] = []
    for line in lines[:30]:
        low = line.lower()
        if any(word in low for word in skip):
            continue
        if len(line) < 3 or len(line) > 60:
            continue
        if re.fullmatch(r"[\d\s\-/.:]+", line):
            continue
        candidates.append(line)
        if len(candidates) >= 6:
            break

    brand = candidates[0] if candidates else None
    product = None
    for line in candidates[1:]:
        if line != brand:
            product = line
            break
    return brand, product


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


def _extract_ingredients_precise(text: str) -> Optional[str]:
    """Extract ingredients with stronger stop rules to avoid tail noise."""
    lines = [line.strip() for line in text.splitlines()]
    if not lines:
        return None

    start_idx = -1
    first_value = ""
    for idx, line in enumerate(lines):
        match = re.search(r"\bingredients?\b\s*:?\s*(.*)$", line, flags=re.I)
        if match:
            start_idx = idx
            first_value = match.group(1).strip()
            break
    if start_idx == -1:
        return None

    stop_pattern = re.compile(
        r"\b("
        r"nutrition|nutritional|fssai|mrp|net\s*(?:qty|wt|weight|quantity|content)|"
        r"batch|b\.?\s*no|mfd|mfg|pkd|packed|best\s*before|barcode|"
        r"customer\s*care|email|website|phone|helpline|imported|manufactured|marketed|"
        r"see\s+.*panel|image\s+on\s+the\s+front|cut\s+here|gst"
        r")\b",
        flags=re.I,
    )

    collected: list[str] = []
    if first_value:
        collected.append(first_value)

    for line in lines[start_idx + 1 :]:
        lowered = line.lower()
        if stop_pattern.search(lowered):
            break
        if not line:
            continue
        # Skip barcode-like / code-only lines.
        if re.fullmatch(r"[\d\s\-]{6,}", line):
            break
        if line.startswith("---"):
            break
        collected.append(line)
        if len(collected) >= 16:
            break

    joined = _normalize_space(" ".join(collected))
    if not joined:
        return None

    # Handle OCR where section repeats "INGREDIENTS:" multiple times.
    parts = [p.strip(" ,;:-") for p in re.split(r"\bingredients?\s*:?\s*", joined, flags=re.I) if p.strip()]
    if parts:
        joined = max(parts, key=len)

    # Hard-trim marketing/footer spillovers if they slipped in same line.
    joined = re.split(
        r"\b(?:image on the front|see (?:centre|center) panel|best before|batch|mrp|net qty|barcode)\b",
        joined,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ,;:-")
    return joined or None


def _extract_html_table(text: str) -> Optional[str]:
    match = re.search(r"<table\b[\s\S]*?</table>", text, flags=re.I)
    if not match:
        return None
    return match.group(0).strip()


def _html_table_to_readable_lines(html: str) -> str:
    """Turn an HTML <table> into plain lines: one row per line, cells separated by ' | '."""
    rows: list[str] = []
    for m in re.finditer(r"<tr\b[^>]*>([\s\S]*?)</tr>", html, flags=re.I):
        row_html = m.group(1)
        cells = re.findall(r"<(?:th|td)\b[^>]*>([\s\S]*?)</(?:th|td)>", row_html, flags=re.I)
        if not cells:
            continue
        parts: list[str] = []
        for cell in cells:
            plain = _normalize_space(re.sub(r"<[^>]+>", " ", cell))
            plain = plain.replace("|", "/").strip()
            if plain:
                parts.append(plain)
        if parts:
            rows.append(" | ".join(parts))
    return "\n".join(rows)


def _nutrition_to_readable(value: Any) -> Optional[str]:
    """Nutrition for JSON responses: no raw HTML tables; readable text or line-oriented rows."""
    if value is None:
        return None
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if isinstance(item, dict):
                pairs = []
                for k, v in item.items():
                    if v is None or str(v).strip() == "":
                        continue
                    label = str(k).replace("_", " ").strip()
                    pairs.append(f"{label}: {v}")
                if pairs:
                    lines.append(", ".join(pairs))
            else:
                t = str(item).strip()
                if t:
                    lines.append(_nutrition_to_readable(t) or t)
        return "\n".join(lines) if lines else None
    text = str(value).strip()
    if not text:
        return None
    low = text.lower()
    if "<table" in low or "<tr" in low or "<td" in low or "<th" in low:
        readable = _html_table_to_readable_lines(text)
        if readable.strip():
            return readable.strip()
        fallback = _normalize_space(re.sub(r"<[^>]+>", " ", text))
        return fallback or None
    return _normalize_space(text)


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
    brand_name, product_name = _extract_brand_product(text)

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
    ingredients = _extract_ingredients_precise(text) or _extract_section(
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

    nutritional_readable = _nutrition_to_readable(nutritional)

    return {
        "mrp": _extract_mrp(text),
        "gtin": gtin,
        "GTIN": gtin,
        "gtin_source": "barcode" if barcode_gtin else ("ocr" if gtin_from_text else "none"),
        "fssai": fssai,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "net_wt": _extract_net_wt(text),
        "best_before": _extract_best_before(text),
        "brand_name": brand_name,
        "product_name": product_name,
        "nutritional": nutritional_readable,
        "nutritable": nutritional_readable,
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
    """Map Qwen nutrition output to the same readable format as regex/HTML paths."""
    return _nutrition_to_readable(value)


def _is_valid_email(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))


def _is_valid_phone(value: Optional[str]) -> bool:
    if not value:
        return False
    digits = re.sub(r"\D", "", value)
    if digits.startswith("1800") and len(digits) in {10, 11, 12}:
        return True
    if digits.startswith("91") and len(digits) == 12 and digits[2] in "6789":
        return True
    if len(digits) == 10 and digits[0] in "6789":
        return True
    if digits.startswith("0") and 9 <= len(digits) <= 12:
        return True
    return False


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
                    "num_predict": EXTRACT_NUM_PREDICT,
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
    llm_fields = await _extract_structured_with_qwen(text)
    fields["brand_name"] = _coerce_optional_str(llm_fields.get("brand_name")) or fields.get("brand_name")
    fields["product_name"] = _coerce_optional_str(llm_fields.get("product_name")) or fields.get("product_name")
    fields["best_before"] = _coerce_optional_str(llm_fields.get("best_before")) or fields.get("best_before")

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


def _listify(value: Optional[str]) -> list[str]:
    cleaned = _coerce_optional_str(value)
    return [cleaned] if cleaned else []


def _nutri_table_lines(value: Optional[str]) -> list[str]:
    """Legacy nutri_table: plain-text rows only (no HTML); one JSON string per table row."""
    cleaned = _coerce_optional_str(value)
    if not cleaned:
        return []
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines if len(lines) > 1 else [cleaned]


def _sanity_score(text: str) -> float:
    # Lightweight proxy score (0-100) based on OCR text richness.
    words = re.findall(r"\b\w+\b", text)
    unique_ratio = (len(set(w.lower() for w in words)) / len(words)) if words else 0.0
    length_factor = min(len(words) / 80.0, 1.0)
    score = max(0.0, min(100.0, (0.65 * unique_ratio + 0.35 * length_factor) * 100.0))
    return round(score, 2)


@app.post("/front")
async def legacy_front(front: UploadFile = File(...)) -> dict[str, Any]:
    request_start = time.perf_counter()
    body = await front.read()
    _validate_image_file(front, body)

    t0 = time.perf_counter()
    text = await _extract_text_from_bytes(body, front.filename or "")
    pre_process = time.perf_counter() - t0
    brand_name, product_name = _extract_brand_product(text)

    total = time.perf_counter() - request_start
    return {
        "front": {
            "brand_name": brand_name or "",
            "product_name": product_name or "",
        },
        "sanity_check": {"front": _sanity_score(text)},
        "filename": {"front": os.path.basename(front.filename or "")},
        "time": {
            "pre_process": round(pre_process, 3),
            "post_process": round(max(0.0, total - pre_process), 3),
            "total_time": round(total, 3),
        },
    }


@app.post("/back")
async def legacy_back(back: UploadFile = File(...)) -> dict[str, Any]:
    request_start = time.perf_counter()
    body = await back.read()
    _validate_image_file(back, body)

    t0 = time.perf_counter()
    barcodes = await _decode_barcodes_with_timeout(body)
    barcode_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    text = await _extract_text_from_bytes(body, back.filename or "")
    ocr_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fssai = _extract_fssai(text)
    barcode_gtin = _pick_gtin_from_barcodes(barcodes, fssai)
    fields = _extract_product_fields(text, barcode_gtin=barcode_gtin)
    regex_time = time.perf_counter() - t0

    back_result = {
        "phone": _listify(fields.get("phone")),
        "mail": _listify(fields.get("email")),
        "fssai": _listify(fields.get("fssai")),
        "GTIN": _listify(fields.get("gtin")),
        "netwt": _listify(fields.get("net_wt")),
        "best_before": _listify(fields.get("best_before")),
        "mrp": _listify(fields.get("mrp")),
        "ingredients": _listify(fields.get("ingredients")),
        "nutri_table": _nutri_table_lines(fields.get("nutritional")),
    }

    flags = {
        "phone": "From regex" if back_result["phone"] else "Not found",
        "mail": "From regex" if back_result["mail"] else "Not found",
        "fssai": "From regex" if back_result["fssai"] else "Not found",
        "GTIN": "From barcode/ocr" if back_result["GTIN"] else "Not found",
        "netwt": "From regex" if back_result["netwt"] else "Not found",
        "best_before": "From regex" if back_result["best_before"] else "Not found",
        "mrp": "From regex" if back_result["mrp"] else "Not found",
    }

    total = time.perf_counter() - request_start
    return {
        "result": {"back": back_result},
        "flags": flags,
        "time": {
            "barcode": round(barcode_time, 3),
            "pre_process": round(barcode_time + ocr_time, 3),
            "regex": round(regex_time, 3),
            "post_process": round(regex_time, 3),
            "total_time": round(total, 3),
        },
        "filename": os.path.basename(back.filename or ""),
        "sanity_check": {"back": _sanity_score(text)},
    }


@app.post("/ingredients")
async def legacy_ingredients(ingredients: UploadFile = File(...)) -> dict[str, Any]:
    request_start = time.perf_counter()
    body = await ingredients.read()
    _validate_image_file(ingredients, body)

    t0 = time.perf_counter()
    text = await _extract_text_from_bytes(body, ingredients.filename or "")
    pre_process = time.perf_counter() - t0
    extracted = _extract_ingredients_precise(text)

    total = time.perf_counter() - request_start
    key = "ingredients"
    return {
        "url": {key: ""},
        "filename": {key: os.path.basename(ingredients.filename or "")},
        "sanity_check": {key: _sanity_score(text)},
        "result": {key: text},
        "time": {
            "pre_process": round(pre_process, 3),
            "post_process": round(max(0.0, total - pre_process), 3),
            "total_time": round(total, 3),
        },
        "Ingredients": extracted or "",
    }


@app.post("/nutritional")
async def legacy_nutritional(nutritional: UploadFile = File(...)) -> dict[str, Any]:
    request_start = time.perf_counter()
    body = await nutritional.read()
    _validate_image_file(nutritional, body)

    t0 = time.perf_counter()
    text = await _extract_text_from_bytes(body, nutritional.filename or "")
    pre_process = time.perf_counter() - t0
    fields = _extract_product_fields(text)

    total = time.perf_counter() - request_start
    key = "nutritional"
    return {
        "url": {key: ""},
        "filename": {key: os.path.basename(nutritional.filename or "")},
        "sanity_check": {key: _sanity_score(text)},
        "result": {key: text},
        "time": {
            "pre_process": round(pre_process, 3),
            "post_process": round(max(0.0, total - pre_process), 3),
            "total_time": round(total, 3),
        },
        "Nutritional Content": fields.get("nutritional") or "",
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
    request_started = time.perf_counter()
    timing: dict[str, float] = {}

    body = await file.read()
    _validate_image_file(file, body)

    t0 = time.perf_counter()
    barcodes = await _decode_barcodes_with_timeout(body)
    timing["barcode_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    text = await _extract_text_from_bytes(body, file.filename or "")
    timing["ocr_sec"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    fssai = _extract_fssai(text)
    barcode_gtin = _pick_gtin_from_barcodes(barcodes, fssai)
    regex_fields = _extract_product_fields(text, barcode_gtin=barcode_gtin)
    llm_fields = await _extract_structured_with_qwen(text)
    timing["qwen_sec"] = round(time.perf_counter() - t0, 3)

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

    llm_email = _coerce_optional_str(llm_fields.get("email"))
    regex_email = regex_fields.get("email")
    if _is_valid_email(llm_email):
        merged_email = llm_email
        email_source = "qwen"
    elif _is_valid_email(regex_email):
        merged_email = regex_email
        email_source = "regex"
    else:
        merged_email = None
        email_source = "none"

    llm_phone = _coerce_optional_str(llm_fields.get("phone"))
    regex_phone = regex_fields.get("phone")
    if _is_valid_phone(llm_phone):
        merged_phone = llm_phone
        phone_source = "qwen"
    elif _is_valid_phone(regex_phone):
        merged_phone = regex_phone
        phone_source = "regex"
    else:
        merged_phone = None
        phone_source = "none"

    llm_brand_name = _coerce_optional_str(llm_fields.get("brand_name"))
    regex_brand_name = regex_fields.get("brand_name")
    if llm_brand_name:
        merged_brand_name = llm_brand_name
        brand_name_source = "qwen"
    elif regex_brand_name:
        merged_brand_name = regex_brand_name
        brand_name_source = "regex"
    else:
        merged_brand_name = None
        brand_name_source = "none"

    llm_product_name = _coerce_optional_str(llm_fields.get("product_name"))
    regex_product_name = regex_fields.get("product_name")
    if llm_product_name:
        merged_product_name = llm_product_name
        product_name_source = "qwen"
    elif regex_product_name:
        merged_product_name = regex_product_name
        product_name_source = "regex"
    else:
        merged_product_name = None
        product_name_source = "none"

    llm_best_before = _coerce_optional_str(llm_fields.get("best_before"))
    regex_best_before = regex_fields.get("best_before")
    if llm_best_before:
        merged_best_before = llm_best_before
        best_before_source = "qwen"
    elif regex_best_before:
        merged_best_before = regex_best_before
        best_before_source = "regex"
    else:
        merged_best_before = None
        best_before_source = "none"

    fields: dict[str, Optional[str]] = {
        "gtin": merged_gtin,
        "fssai": llm_fssai or regex_fields.get("fssai"),
        "mrp": _coerce_optional_str(llm_fields.get("mrp")) or regex_fields.get("mrp"),
        "net_wt": _coerce_optional_str(llm_fields.get("net_weight")) or regex_fields.get("net_wt"),
        "ingredients": _normalize_ingredients(llm_fields.get("ingredients")) or regex_fields.get("ingredients"),
        "nutritional": _normalize_nutrition(llm_fields.get("nutrition")) or regex_fields.get("nutritional"),
        "email": merged_email,
        "phone": merged_phone,
        "best_before": merged_best_before,
        "brand_name": merged_brand_name,
        "product_name": merged_product_name,
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
        "email_source":              email_source,
        "phone":                     fields.get("phone"),
        "phone_source":              phone_source,
        "best_before":               fields.get("best_before"),
        "best_before_source":        best_before_source,
        "brand_name":                fields.get("brand_name"),
        "brand_name_source":         brand_name_source,
        "product_name":              fields.get("product_name"),
        "product_name_source":       product_name_source,
        "barcode_decoder_available": BARCODE_LIB_AVAILABLE,
        "validation":                _validate_fields(fields),
        "extract_model":             EXTRACT_MODEL,
        "timing":                    timing,
    }
    timing["total_sec"] = round(time.perf_counter() - request_started, 3)
    return result

