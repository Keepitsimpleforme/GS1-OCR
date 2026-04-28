#!/usr/bin/env python3
"""
Batch-test images via /api/parse and export structured results to CSV.

Usage:
  python scripts/batch_extract_to_csv.py \
    --zip /path/to/images.zip \
    --api http://20.204.169.52/api/parse \
    --out /path/to/results.csv

Notes:
  - Non-invasive: does not modify backend/frontend code.
  - CSV opens directly in Excel/Google Sheets.
  - One row per image, image name as first column.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _iter_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return sorted(files)


def _as_json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


def _extract_one(image_path: Path, api_url: str, timeout_sec: int) -> tuple[dict[str, Any], str]:
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "application/octet-stream")}
        resp = requests.post(api_url, files=files, timeout=timeout_sec)

    if not resp.ok:
        detail = ""
        try:
            body = resp.json()
            detail = body.get("detail", str(body))
        except Exception:
            detail = resp.text
        return {}, f"HTTP {resp.status_code}: {detail}"

    try:
        data = resp.json()
        if not isinstance(data, dict):
            return {}, "Invalid JSON object response"
        return data, ""
    except Exception as exc:
        return {}, f"JSON parse error: {exc}"


def _is_missing(value: str) -> bool:
    return value.strip() == ""


def _build_summary(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    total = len(rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_count = len(ok_rows)
    err_count = total - ok_count
    success_rate = (ok_count / total * 100.0) if total else 0.0

    tracked_fields = [
        "gtin",
        "fssai",
        "mrp",
        "net_weight",
        "ingredients",
        "nutrition",
        "email",
        "phone",
    ]

    summary: list[tuple[str, str]] = [
        ("summary_total_images", str(total)),
        ("summary_ok_images", str(ok_count)),
        ("summary_error_images", str(err_count)),
        ("summary_success_rate_percent", f"{success_rate:.2f}"),
    ]

    base_rows = ok_rows if ok_rows else rows
    base_n = len(base_rows) if base_rows else 1
    for field in tracked_fields:
        missing = sum(1 for r in base_rows if _is_missing(r.get(field, "")))
        missing_pct = missing / base_n * 100.0
        present_pct = 100.0 - missing_pct
        summary.append((f"{field}_missing_percent", f"{missing_pct:.2f}"))
        summary.append((f"{field}_present_proxy_percent", f"{present_pct:.2f}"))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch extract fields from ZIP images via /api/parse")
    parser.add_argument("--zip", required=True, help="Path to zip file containing images")
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/parse", help="API endpoint URL")
    parser.add_argument("--out", default="batch_results.csv", help="Output CSV path")
    parser.add_argument("--timeout", type=int, default=420, help="Per-image request timeout in seconds")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    api_url = args.api.strip()

    if not zip_path.exists():
        raise SystemExit(f"ZIP file not found: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="gs1-batch-") as tmpdir:
        extract_root = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

        images = _iter_images(extract_root)
        if not images:
            raise SystemExit("No supported image files found in ZIP.")

        rows: list[dict[str, str]] = []
        total = len(images)

        for idx, image in enumerate(images, start=1):
            rel_name = image.relative_to(extract_root).as_posix()
            print(f"[{idx}/{total}] Processing {rel_name}")

            try:
                data, err = _extract_one(image, api_url=api_url, timeout_sec=args.timeout)
            except requests.Timeout:
                data, err = {}, f"Timeout after {args.timeout}s"
            except Exception as exc:
                data, err = {}, f"Request error: {exc}"

            validation = data.get("validation") if isinstance(data, dict) else None
            validation_errors = ""
            if isinstance(validation, dict):
                validation_errors = _as_json_cell(validation.get("errors"))

            rows.append(
                {
                    "image_name": rel_name,
                    "status": "ok" if not err else "error",
                    "error": err,
                    "gtin": _as_json_cell(data.get("gtin")),
                    "gtin_source": _as_json_cell(data.get("gtin_source")),
                    "fssai": _as_json_cell(data.get("fssai")),
                    "mrp": _as_json_cell(data.get("mrp")),
                    "net_weight": _as_json_cell(data.get("net_weight")),
                    "ingredients": _as_json_cell(data.get("ingredients")),
                    "nutrition": _as_json_cell(data.get("nutrition")),
                    "email": _as_json_cell(data.get("email")),
                    "email_source": _as_json_cell(data.get("email_source")),
                    "phone": _as_json_cell(data.get("phone")),
                    "phone_source": _as_json_cell(data.get("phone_source")),
                    "barcode_decoder_available": _as_json_cell(data.get("barcode_decoder_available")),
                    "extract_model": _as_json_cell(data.get("extract_model")),
                    "fssai_valid": _as_json_cell(validation.get("fssai_valid") if isinstance(validation, dict) else ""),
                    "mrp_valid": _as_json_cell(validation.get("mrp_valid") if isinstance(validation, dict) else ""),
                    "validation_errors": validation_errors,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "status",
        "error",
        "gtin",
        "gtin_source",
        "fssai",
        "mrp",
        "net_weight",
        "ingredients",
        "nutrition",
        "email",
        "email_source",
        "phone",
        "phone_source",
        "barcode_decoder_available",
        "extract_model",
        "fssai_valid",
        "mrp_valid",
        "validation_errors",
    ]
    summary_rows = _build_summary(rows)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        summary_writer = csv.writer(f)
        summary_writer.writerow(["metric", "value"])
        for metric, value in summary_rows:
            summary_writer.writerow([metric, value])
        summary_writer.writerow([])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    err_count = len(rows) - ok_count
    print(f"\nDone. Wrote: {out_path}")
    print(f"Total: {len(rows)} | OK: {ok_count} | Errors: {err_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
