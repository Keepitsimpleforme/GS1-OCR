#!/usr/bin/env python3
"""
Batch-test images via /api/parse and export structured results to CSV.

Usage:
  python scripts/batch_extract_to_csv.py \
    --dir /path/to/images \
    --api http://20.204.169.52/api/parse \
    --out /path/to/results.csv

Defaults (suitable for large batch / strict timeout tests):
  --timeout 40   (no retry window beyond one request)
  --retries 0

Notes:
  - CSV contains only extracted business fields + image_name + elapsed_sec.
  - No raw OCR text, no status/error/attempts, no *_source, no model/validation columns.
  - Summary metrics are written at the top; full results follow after a blank line.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Only required output columns (no status, error, attempts, *_source, extract_model, validation).
FIELDNAMES = [
    "image_name",
    "elapsed_sec",
    "gtin",
    "fssai",
    "mrp",
    "net_weight",
    "ingredients",
    "nutrition",
    "email",
    "phone",
    "barcode_decoder_available",
    "brand_name",
    "product_name",
    "best_before",
]


def _iter_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_parts = path.relative_to(root).parts
        if "__MACOSX" in rel_parts:
            continue
        if path.name.startswith("._"):
            continue
        files.append(path)
    return sorted(files)


def _as_json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True)


def _extract_one(image_path: Path, api_url: str, timeout_sec: int) -> tuple[dict[str, Any], str]:
    mime = MIME_BY_EXT.get(image_path.suffix.lower(), "application/octet-stream")
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, mime)}
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


def _should_retry(error_text: str) -> bool:
    low = error_text.lower()
    retry_markers = [
        "timeout",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "connection aborted",
        "connection reset",
        "temporarily unavailable",
    ]
    return any(marker in low for marker in retry_markers)


def _is_missing(value: str) -> bool:
    return value.strip() == ""


def _build_summary(rows: list[dict[str, str]], outcomes: list[bool]) -> list[tuple[str, str]]:
    total = len(rows)
    ok_count = sum(1 for ok in outcomes if ok)
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
        "brand_name",
        "product_name",
        "best_before",
    ]

    summary: list[tuple[str, str]] = [
        ("summary_total_images", str(total)),
        ("summary_ok_images", str(ok_count)),
        ("summary_error_images", str(err_count)),
        ("summary_success_rate_percent", f"{success_rate:.2f}"),
    ]
    if total:
        avg_elapsed = sum(float(r.get("elapsed_sec", "0") or 0) for r in rows) / total
        summary.append(("summary_avg_elapsed_sec", f"{avg_elapsed:.2f}"))

    base_rows = [r for r, ok in zip(rows, outcomes) if ok]
    base_rows = base_rows if base_rows else rows
    base_n = len(base_rows) if base_rows else 1
    for field in tracked_fields:
        missing = sum(1 for r in base_rows if _is_missing(r.get(field, "")))
        missing_pct = missing / base_n * 100.0
        present_pct = 100.0 - missing_pct
        summary.append((f"{field}_missing_percent", f"{missing_pct:.2f}"))
        summary.append((f"{field}_present_proxy_percent", f"{present_pct:.2f}"))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch extract fields from images via /api/parse")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", help="Path to zip file containing images")
    group.add_argument("--dir", help="Path to directory containing images")
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/parse", help="API endpoint URL")
    parser.add_argument("--out", default="batch_results.csv", help="Output CSV path")
    parser.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="Per-image request timeout in seconds (default: 40)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Extra retries after first attempt (default: 0 = no retries)",
    )
    parser.add_argument("--retry-wait", type=float, default=2.0, help="Seconds to wait between retries")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve() if args.zip else None
    dir_path = Path(args.dir).expanduser().resolve() if args.dir else None
    out_path = Path(args.out).expanduser().resolve()
    api_url = args.api.strip()

    if zip_path and not zip_path.exists():
        raise SystemExit(f"ZIP file not found: {zip_path}")
    if dir_path and not dir_path.exists():
        raise SystemExit(f"Input directory not found: {dir_path}")

    rows: list[dict[str, str]] = []
    outcomes: list[bool] = []

    if zip_path:
        with tempfile.TemporaryDirectory(prefix="gs1-batch-") as tmpdir:
            extract_root = Path(tmpdir)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_root)
            images = _iter_images(extract_root)
            root_for_names = extract_root
            source_label = f"ZIP: {zip_path}"
            if not images:
                raise SystemExit("No supported image files found in ZIP.")
            _process_images(images, root_for_names, source_label, api_url, args, rows, outcomes)
    else:
        images = _iter_images(dir_path)
        root_for_names = dir_path
        source_label = f"DIR: {dir_path}"
        if not images:
            raise SystemExit("No supported image files found in directory.")
        _process_images(images, root_for_names, source_label, api_url, args, rows, outcomes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = _build_summary(rows, outcomes)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        summary_writer = csv.writer(f)
        summary_writer.writerow(["metric", "value"])
        for metric, value in summary_rows:
            summary_writer.writerow([metric, value])
        summary_writer.writerow([])

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(outcomes)
    err_count = len(rows) - ok_count
    print(f"\nDone. Wrote: {out_path}")
    print(f"Total: {len(rows)} | OK: {ok_count} | Errors: {err_count}")
    return 0


def _process_images(
    images: list[Path],
    root_for_names: Path,
    source_label: str,
    api_url: str,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    outcomes: list[bool],
) -> None:
    print(f"\nInput source: {source_label}")
    print(f"Images found: {len(images)}")
    print(f"Timeout: {args.timeout}s | Retries: {args.retries}")

    total = len(images)
    max_attempts = args.retries + 1

    for idx, image in enumerate(images, start=1):
        rel_name = image.relative_to(root_for_names).as_posix()
        t0 = time.perf_counter()
        attempts = 0
        data: dict[str, Any] = {}
        err = ""
        while attempts < max_attempts:
            attempts += 1
            if max_attempts > 1:
                print(f"[{idx}/{total}] Processing {rel_name} (attempt {attempts}/{max_attempts})")
            else:
                print(f"[{idx}/{total}] Processing {rel_name}")

            try:
                data, err = _extract_one(image, api_url=api_url, timeout_sec=args.timeout)
            except requests.Timeout:
                data, err = {}, f"Timeout after {args.timeout}s"
            except Exception as exc:
                data, err = {}, f"Request error: {exc}"

            if not err:
                break
            if attempts < max_attempts and _should_retry(err):
                print(f"    transient failure: {err}")
                print(f"    retrying in {args.retry_wait:.1f}s...")
                time.sleep(args.retry_wait)
                continue
            break

        elapsed_sec = time.perf_counter() - t0
        ok = not err

        rows.append(
            {
                "image_name": rel_name,
                "elapsed_sec": f"{elapsed_sec:.2f}",
                "gtin": _as_json_cell(data.get("gtin")),
                "fssai": _as_json_cell(data.get("fssai")),
                "mrp": _as_json_cell(data.get("mrp")),
                "net_weight": _as_json_cell(data.get("net_weight")),
                "ingredients": _as_json_cell(data.get("ingredients")),
                "nutrition": _as_json_cell(data.get("nutrition")),
                "email": _as_json_cell(data.get("email")),
                "phone": _as_json_cell(data.get("phone")),
                "barcode_decoder_available": _as_json_cell(data.get("barcode_decoder_available")),
                "brand_name": _as_json_cell(data.get("brand_name")),
                "product_name": _as_json_cell(data.get("product_name")),
                "best_before": _as_json_cell(data.get("best_before")),
            }
        )
        outcomes.append(ok)
        print(f"    -> {'ok' if ok else 'error'} in {elapsed_sec:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())
