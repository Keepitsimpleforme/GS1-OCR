# GS1 OCR — HTTP API reference

This document describes the FastAPI backend exposed behind nginx. Paths and ports match a typical VM deployment:

| Listener | Purpose |
|----------|---------|
| **Port 80** | SPA (static frontend) + `/api/*`, `/health` proxied to backend |
| **Port 8080** | Legacy-compatible OCR routes: `/front`, `/back`, `/ingredients`, `/nutritional` |

Replace host with your server, e.g. `http://20.204.169.52`.

### Interactive docs (Swagger UI)

FastAPI serves OpenAPI automatically:

| URL | Description |
|-----|-------------|
| `http://<host>/docs` | **Swagger UI** — try requests in the browser |
| `http://<host>/redoc` | **ReDoc** — alternate layout |
| `http://<host>/openapi.json` | Raw OpenAPI schema |

**Direct to uvicorn (bypass nginx):** `http://127.0.0.1:8000/docs` on the VM.

**Through nginx (port 80):** the repo `nginx.conf` proxies `/docs`, `/redoc`, and `/openapi.json` to the backend. After editing nginx on the VM, run `sudo nginx -t && sudo systemctl reload nginx`.

Legacy routes on **port 8080** (`/front`, `/back`, etc.) are not required to expose Swagger; use port **80** for API explorer.

---

## Common conventions

### Content type

All image upload endpoints expect **`multipart/form-data`** with one file part per request.

### Allowed image types (MIME)

Validated against upload `Content-Type`:

- `image/jpeg`
- `image/png`
- `image/webp`
- `image/gif`

**Filename / extension:** any name is accepted if MIME is one of the above. Clients should send a correct `Content-Type` (Postman usually sets it from the chosen file).

### Max upload size

Default **20 MB** (`MAX_UPLOAD_MB` env × 1 MiB). Oversize returns **413** with a `detail` message.

### Errors (FastAPI)

| HTTP | Typical cause |
|------|-----------------|
| **400** | Invalid file type, bad request body |
| **413** | File too large |
| **422** | Missing or wrong multipart field name (e.g. `front` missing on `/front`) |
| **502** | Ollama / model error during OCR or extraction |
| **503** | Ollama unreachable, or barcode library unavailable (decode endpoint) |
| **504** | OCR or upstream timeout |

Error body is usually JSON:

```json
{
  "detail": "Human readable message"
}
```

Validation errors (422) may use FastAPI’s list form for `detail`.

---

## `GET /health`

**Purpose:** Liveness and Ollama reachability.

**Request:** No body.

**Response `200` JSON:**

```json
{
  "status": "ok",
  "ollama": "reachable",
  "ocr_model": "<OLLAMA_MODEL_OCR or OLLAMA_MODEL>",
  "extract_model": "<OLLAMA_MODEL_EXTRACT>"
}
```

**Example:**

```bash
curl -s "http://20.204.169.52/health"
```

---

## `POST /api/extract`

**Purpose:** Raw OCR text only (no structured fields).

| Part | Type | Required | Description |
|------|------|----------|-------------|
| `file` | file | yes | Image |

**Response `200` JSON:**

```json
{
  "text": "<full OCR transcript>",
  "model": "<OCR model name>"
}
```

**Example:**

```bash
curl -s -X POST "http://20.204.169.52/api/extract" \
  -F "file=@/path/to/label.jpg"
```

---

## `POST /api/extract/product-info`

**Purpose:** Barcode decode + OCR + regex fields + Qwen pass for brand/product/best_before enrichment. Returns **raw OCR text** in the payload.

| Part | Type | Required | Description |
|------|------|----------|-------------|
| `file` | file | yes | Image |

**Response `200` JSON (blueprint):**

```json
{
  "fields": {
    "mrp": "80.00",
    "gtin": "8901512544407",
    "GTIN": "8901512544407",
    "gtin_source": "barcode",
    "fssai": "12345678901234",
    "email": "info@example.com",
    "phone": "18001234567",
    "net_wt": "500 g",
    "nutritional": "<html table or text>",
    "nutritable": "<same as nutritional>",
    "ingredients": "…",
    "best_before": "06/09/2015",
    "brand_name": "Example Brand",
    "product_name": "Delhi Mix"
  },
  "barcode": {
    "available": true,
    "decoded": ["8901512544407"],
    "gtin": "8901512544407"
  },
  "text": "<full OCR transcript>",
  "model": "<OCR model name>"
}
```

**Example:**

```bash
curl -s -X POST "http://20.204.169.52/api/extract/product-info" \
  -F "file=@/path/to/label.jpg"
```

---

## `POST /api/extract/required-fields`

**Purpose:** Same pipeline as product-info but returns **only** the flat `fields` map (no `text`, no `barcode` wrapper).

| Part | Type | Required |
|------|------|----------|
| `file` | file | yes |

**Response `200` JSON:** same shape as `fields` in `/api/extract/product-info`.

---

## `POST /api/decode-barcode`

**Purpose:** Barcode decode only.

| Part | Type | Required |
|------|------|----------|
| `file` | file | yes |

**Response `200` JSON:**

```json
{
  "decoded": ["8901512544407"],
  "gtin": "8901512544407",
  "count": 1
}
```

**Note:** Returns **503** if pyzbar/zbar is not installed on the server.

---

## `POST /api/parse`

**Purpose:** Single-call structured extraction for integrations and batch CSV. **Does not return raw OCR text.**

| Part | Type | Required | Description |
|------|------|----------|-------------|
| `file` | file | yes | Label image |

**Response `200` JSON (blueprint):**

```json
{
  "gtin": "8901512544407",
  "gtin_source": "barcode",
  "fssai": "12345678901234",
  "mrp": "80.00",
  "net_weight": "500 g",
  "ingredients": "Moong Lentils, …",
  "nutrition": "<HTML table string or JSON-serialized nutrition from model>",
  "email": "info@example.com",
  "email_source": "regex",
  "phone": "18001234567",
  "phone_source": "qwen",
  "best_before": "06/09/2015",
  "best_before_source": "regex",
  "brand_name": "Example Brand",
  "brand_name_source": "qwen",
  "product_name": "Delhi Mix",
  "product_name_source": "regex",
  "barcode_decoder_available": true,
  "validation": {
    "fssai_valid": true,
    "mrp_valid": true,
    "errors": []
  },
  "extract_model": "qwen2.5:7b",
  "timing": {
    "barcode_sec": 0.012,
    "ocr_sec": 12.4,
    "qwen_sec": 8.2,
    "total_sec": 20.7
  }
}
```

**`gtin_source` values:** `barcode` | `qwen` | `ocr` | `none`  
**`*_source` for email/phone/brand/product/best_before:** `qwen` | `regex` | `none`

**Example:**

```bash
curl -s -X POST "http://20.204.169.52/api/parse" \
  -F "file=@/path/to/label.jpg"
```

---

## Legacy routes (port **8080**)

These mirror older client expectations. Multipart **field names must match** the route (not `file`).

### `POST http://<host>:8080/front`

| Part | Type | Required |
|------|------|----------|
| `front` | file | yes |

**Response `200` JSON:**

```json
{
  "front": {
    "brand_name": "Brand or empty string",
    "product_name": "Product or empty string"
  },
  "sanity_check": { "front": 66.87 },
  "filename": { "front": "photo.jpg" },
  "time": {
    "pre_process": 7.504,
    "post_process": 0.001,
    "total_time": 7.505
  }
}
```

**Postman:** Body → form-data → key `front` → type File.

---

### `POST http://<host>:8080/back`

| Part | Type | Required |
|------|------|----------|
| `back` | file | yes |

**Response `200` JSON (blueprint):**

```json
{
  "result": {
    "back": {
      "phone": ["18001234567"],
      "mail": ["care@brand.com"],
      "fssai": ["12345678901234"],
      "GTIN": ["8901512544407"],
      "netwt": ["500 g"],
      "best_before": ["06/09/2015"],
      "mrp": ["80.00"],
      "ingredients": ["…"],
      "nutri_table": ["<nutritional snippet or empty>"]
    }
  },
  "flags": {
    "phone": "From regex",
    "mail": "From regex",
    "fssai": "From regex",
    "GTIN": "From barcode/ocr",
    "netwt": "From regex",
    "best_before": "From regex",
    "mrp": "From regex"
  },
  "time": {
    "barcode": 0.01,
    "pre_process": 15.2,
    "regex": 0.05,
    "post_process": 0.05,
    "total_time": 15.3
  },
  "filename": "back.jpg",
  "sanity_check": { "back": 72.5 }
}
```

Empty scalar fields appear as **`[]`**. Flag value **`Not found`** when list is empty.

---

### `POST http://<host>:8080/ingredients`

| Part | Type | Required |
|------|------|----------|
| `ingredients` | file | yes |

**Response `200` JSON:**

```json
{
  "url": { "ingredients": "" },
  "filename": { "ingredients": "ingredients_crop.jpg" },
  "sanity_check": { "ingredients": 70.0 },
  "result": { "ingredients": "<full OCR text of image>" },
  "time": {
    "pre_process": 10.0,
    "post_process": 0.0,
    "total_time": 10.0
  },
  "Ingredients": "Popped Corn, Salt, …"
}
```

`Ingredients` is a **string** (parsed line when possible); `result.ingredients` is the **raw OCR** for that image.

---

### `POST http://<host>:8080/nutritional`

| Part | Type | Required |
|------|------|----------|
| `nutritional` | file | yes |

**Response `200` JSON:**

```json
{
  "url": { "nutritional": "" },
  "filename": { "nutritional": "nutri.jpg" },
  "sanity_check": { "nutritional": 68.0 },
  "result": { "nutritional": "<full OCR text>" },
  "time": {
    "pre_process": 9.5,
    "post_process": 0.0,
    "total_time": 9.5
  },
  "Nutritional Content": "<extracted nutritional block or HTML if present>"
}
```

---

## Batch CSV helper (`scripts/batch_extract_to_csv.py`)

**Endpoint used:** `POST /api/parse` only.

**Multipart field:** `file` (same as API).

**Input:** either `--zip <path>` or `--dir <path>` (mutually exclusive).

**Output CSV:**

1. Summary block (`metric`, `value`) including totals and optional `summary_avg_elapsed_sec`.
2. Blank line.
3. Detail rows: `image_name`, `elapsed_sec`, then parsed fields only (`gtin`, `fssai`, `mrp`, `net_weight`, `ingredients`, `nutrition`, `email`, `phone`, `barcode_decoder_available`, `brand_name`, `product_name`, `best_before`). **No** `status`, `error`, `attempts`, `*_source`, `extract_model`, or validation columns. **Raw OCR is not written.**

**Defaults:** `--timeout 40` and `--retries 0` (single attempt per image; tune for your SLA).

---

## Nginx / deployment checklist

- Port **80**: `location /api/` and `location /health` → `http://127.0.0.1:8000`
- Port **8080**: explicit `location = /front` (etc.) → same upstream
- Timeouts: align `proxy_read_timeout` / `proxy_send_timeout` with long OCR (e.g. 420s) to avoid **504** on slow images

---

## Quick reference — multipart field names

| Endpoint | Form field name |
|----------|-----------------|
| `/api/extract` | `file` |
| `/api/extract/product-info` | `file` |
| `/api/extract/required-fields` | `file` |
| `/api/decode-barcode` | `file` |
| `/api/parse` | `file` |
| `:8080/front` | `front` |
| `:8080/back` | `back` |
| `:8080/ingredients` | `ingredients` |
| `:8080/nutritional` | `nutritional` |
