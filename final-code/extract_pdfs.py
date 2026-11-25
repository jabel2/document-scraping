"""Extract a folder of PDFs into Markdown files, manifest.csv, and metadata.csv.

Example (from the project folder):
    uv run --no-sync python final-code/extract_pdfs.py pdfs --output outputs/business --workers 2

Use --estimate-only to inspect the whole input folder without writing results.
Supply --seconds-per-page from representative measurements or use prior page
timings in this output. A real run estimates only queued files. Estimates
exclude overhead and are uncertain if the hardware or page mix changes.
--word-reference-csv accepts source_sha256,page,word,x0,y0,x1,y1 in original
displayed top-left PDF points. IoU and exact-text results describe only those
reviewed labels. A complete status is not a human accuracy certification.
Repeat with the same output to resume or append new PDFs. Keep _state/ for recovery.
Manifest is an event log: use the highest sequence per document_id for current
status. Metadata contains only the current observation set for each document ID.
worker_attempt_seconds includes extraction/checkpoint reuse and staging in the
latest worker attempt, excluding queue and publication. page_extraction_seconds
is the duration originally recorded for the page that supplies a word row.
Input PDFs are read-only. A distinct input path or changed content is a distinct
document version. Markdown retains page headings and unverified table layout.
Metadata contains observed native words and OCR fragments (not guaranteed to cover
every word). Coordinates use source-page display points, top-left origin.
Raw OCR scores are model outputs, not probabilities of correctness.
Optional --reference-csv columns: source_sha256,page,text. Supply one complete
transcription per annotated page. Precision/recall measure casefolded NFKC
whitespace-token count overlap on those pages, ignoring order and geometry;
blank means unmeasured or undefined. No accuracy is inferred from confidence.
Requires Python 3.12, the project's locked dependencies and models, and Tesseract
with English and OSD data. Set TESSERACT_CMD to override executable discovery.
All extraction logic is in this file; no other project script is invoked.
"""
from __future__ import annotations  # Keep type hints readable in the supported Python 3.12 environment.
import math  # Compare displayed text directions and estimate capture time.
import argparse  # Give operators one folder-based command.
import csv  # Exchange audit records in spreadsheet-compatible form.
import hashlib  # Identify exact source bytes and detect changed evidence.
import importlib.metadata  # Record the software versions used for each extraction.
import json  # Retain page evidence and recovery information without rounding.
import multiprocessing  # Isolate model memory and document failures between workers.
import os  # Publish finished files safely and respect machine settings.
import queue  # Wait for the next finished document without busy polling.
import re  # Recognize word boundaries and orientation reports.
import shutil  # Copy approved outputs without touching the original PDFs.
import signal  # Let the coordinator manage interruption and recovery.
import sqlite3  # Preserve the authoritative inventory across interruptions.
import subprocess  # Ask the installed Tesseract program for scan orientation.
import sys  # Report failures to the calling batch job.
import time  # Record elapsed processing time.
import unicodedata  # Compare text after standard Unicode normalization.
import uuid  # Allocate private locations without overwriting previous attempts.
from collections import Counter  # Count repeated words fairly against a supplied reference.
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED  # Limit the number of simultaneously active documents.
from concurrent.futures.process import BrokenProcessPool  # Recognize unexpected model-worker termination and publish partial results.
from datetime import datetime  # Timestamp page progress for operators.
from functools import lru_cache  # Reuse alignment decisions and model fingerprints.
from pathlib import Path  # Keep original and generated file locations distinct.
import psutil  # Size parallelism to the host rather than the dataset.
import pypdfium2 as pdfium  # Read source pages, text layers, and displayed page dimensions.
from pypdfium2 import raw  # Detect images when routing native, mixed, and scanned pages.
TESSERACT = os.environ.get('TESSERACT_CMD') or shutil.which('tesseract') or r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Locate the installed detector without assuming every operator uses the same installation.
ROOT = Path(__file__).resolve().parent.parent  # Locate the installed project models when launched from any folder.
ENGINE = 'rapid-onnx'  # Use the existing English CPU OCR pipeline by default.
RUNTIME = {}  # Keep models in the worker that owns them for subsequent PDFs.


def default_workers() -> int:  # Choose the default parallelism from physical capacity, reserving one core.
    # Physical core count is preferred; logical CPU count is a fallback. Limit
    # the estimate by the process affinity where the OS exposes it.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        count = psutil.cpu_count(logical=False) or os.cpu_count() or 1  # Determine count for this step: choose the default parallelism from physical capacity, reserving one core.
    except (OSError, psutil.Error, NotImplementedError):  # Keep the cause of this failure visible and preserve recoverable progress.
        count = os.cpu_count() or 1  # Determine count for this step: choose the default parallelism from physical capacity, reserving one core.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        allowed = psutil.Process().cpu_affinity()  # Determine allowed for this step: choose the default parallelism from physical capacity, reserving one core.
        if allowed:  # Check eligibility before continuing: choose the default parallelism from physical capacity, reserving one core.
            count = min(count, len(allowed))  # Determine count for this step: choose the default parallelism from physical capacity, reserving one core.
    except (OSError, psutil.Error, NotImplementedError, AttributeError):  # Keep the cause of this failure visible and preserve recoverable progress.
        pass  # Choose the default parallelism from physical capacity, reserving one core.
    return max(1, count - 1)  # Return the result to the next stage: choose the default parallelism from physical capacity, reserving one core.


def configure_worker_threads() -> None:  # Prevent multiple model workers from exhausting CPU threads.
    # Explicit environment settings remain available to users who need different tuning.
    for name in ('DOCLING_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):  # Handle each name separately: prevent multiple model workers from exhausting CPU threads.
        os.environ.setdefault(name, '1')  # Prevent multiple model workers from exhausting CPU threads.


def acquire_run_lock(run_dir: Path):  # Reserve exclusive ownership of the two collection reports.
    """Hold an OS lock so two coordinators cannot schedule the same journal."""
    handle = (run_dir / 'coordinator.lock').open('a+b')  # Determine handle for this step: reserve exclusive ownership of the two collection reports.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        handle.seek(0, os.SEEK_END)  # Reserve exclusive ownership of the two collection reports.
        if handle.tell() == 0:  # Check eligibility before continuing: reserve exclusive ownership of the two collection reports.
            handle.write(b'0'); handle.flush()  # Reserve exclusive ownership of the two collection reports.
        handle.seek(0)  # Reserve exclusive ownership of the two collection reports.
        if os.name == 'nt':  # Check eligibility before continuing: reserve exclusive ownership of the two collection reports.
            import msvcrt  # Load this processing capability only in the worker that needs it.
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # Reserve exclusive ownership of the two collection reports.
        else:  # Use this alternative when earlier evidence does not justify the first route.
            import fcntl  # Load this processing capability only in the worker that needs it.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # Reserve exclusive ownership of the two collection reports.
        return handle  # Return the result to the next stage: reserve exclusive ownership of the two collection reports.
    except OSError as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
        handle.close()  # Release the document or file after its evidence has been preserved.
        raise RuntimeError('This collection is already being processed by another coordinator') from exc  # Surface the inconsistency rather than silently publishing bad evidence.


def inverse_matrix(rotation: int, width: int, height: int) -> list[list[int]]:  # Undo scan rotation so evidence points to the displayed original page.
    """Map corrected image boundary coordinates to original rendered pixels."""
    matrices = {  # Determine matrices for this step: undo scan rotation so evidence points to the displayed original page.
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Undo scan rotation so evidence points to the displayed original page.
        90: [[0, 1, 0], [-1, 0, height], [0, 0, 1]],  # Undo scan rotation so evidence points to the displayed original page.
        180: [[-1, 0, width], [0, -1, height], [0, 0, 1]],  # Undo scan rotation so evidence points to the displayed original page.
        270: [[0, -1, width], [1, 0, 0], [0, 0, 1]],  # Undo scan rotation so evidence points to the displayed original page.
    }  # Undo scan rotation so evidence points to the displayed original page.
    if rotation not in matrices:  # Check eligibility before continuing: undo scan rotation so evidence points to the displayed original page.
        raise ValueError('Rotation must be 0, 90, 180 or 270 degrees')  # Surface the inconsistency rather than silently publishing bad evidence.
    return matrices[rotation]  # Return the result to the next stage: undo scan rotation so evidence points to the displayed original page.


def orient_page(argv, source_digest=None) -> None:  # Detect scan orientation and save the source-to-working-page transform.
    parser = argparse.ArgumentParser(description=__doc__)  # Determine parser for this step: detect scan orientation and save the source-to-working-page transform.
    parser.add_argument('pdf', type=Path)  # Specify the pdf setting for this document-processing step.
    parser.add_argument('--page', type=int, default=1, help='One-based source PDF page')  # Specify the page setting for this document-processing step.
    parser.add_argument('--output-dir', type=Path, required=True)  # Specify the output dir setting for this document-processing step.
    parser.add_argument('--dpi', type=int, default=200)  # Specify the dpi setting for this document-processing step.
    parser.add_argument('--min-confidence', type=float, default=3.0, help='OSD heuristic threshold, not a probability')  # Specify the min confidence setting for this document-processing step.
    parser.add_argument('--tesseract', default=TESSERACT)  # Specify the tesseract setting for this document-processing step.
    args = parser.parse_args(argv)  # Determine args for this step: detect scan orientation and save the source-to-working-page transform.
    if args.page < 1 or args.dpi < 72 or args.min_confidence < 0:  # Check eligibility before continuing: detect scan orientation and save the source-to-working-page transform.
        parser.error('Page must be positive, DPI at least 72, and confidence threshold nonnegative')  # Detect scan orientation and save the source-to-working-page transform.
    if not args.pdf.is_file():  # Check eligibility before continuing: detect scan orientation and save the source-to-working-page transform.
        parser.error('Source PDF does not exist')  # Detect scan orientation and save the source-to-working-page transform.
    args.output_dir.mkdir(parents=True, exist_ok=True)  # Reserve a private location for this processing attempt.
    image_path = args.output_dir / 'source.png'  # Determine image path for this step: detect scan orientation and save the source-to-working-page transform.
    corrected_path = args.output_dir / 'corrected.png'  # Determine corrected path for this step: detect scan orientation and save the source-to-working-page transform.
    output_pdf = args.output_dir / 'corrected.pdf'  # Determine output pdf for this step: detect scan orientation and save the source-to-working-page transform.
    manifest_path = args.output_dir / 'orientation.json'  # Determine manifest path for this step: detect scan orientation and save the source-to-working-page transform.
    if args.pdf.resolve() in {image_path.resolve(), corrected_path.resolve(), output_pdf.resolve(), manifest_path.resolve()}:  # Check eligibility before continuing: detect scan orientation and save the source-to-working-page transform.
        parser.error('Output must not overwrite the source')  # Detect scan orientation and save the source-to-working-page transform.

    pdf = pdfium.PdfDocument(str(args.pdf))  # Determine pdf for this step: detect scan orientation and save the source-to-working-page transform.
    if args.page > len(pdf):  # Check eligibility before continuing: detect scan orientation and save the source-to-working-page transform.
        parser.error('Page number exceeds PDF page count')  # Detect scan orientation and save the source-to-working-page transform.
    page = pdf[args.page - 1]  # Determine page for this step: detect scan orientation and save the source-to-working-page transform.
    width_pt, height_pt = page.get_size()  # Determine width pt, height pt for this step: detect scan orientation and save the source-to-working-page transform.
    image = page.render(scale=args.dpi / 72).to_pil().convert('RGB')  # Determine image for this step: detect scan orientation and save the source-to-working-page transform.
    width, height = image.size  # Determine width, height for this step: detect scan orientation and save the source-to-working-page transform.
    image.save(image_path, dpi=(args.dpi, args.dpi))  # Detect scan orientation and save the source-to-working-page transform.
    proc = subprocess.run([args.tesseract, str(image_path), 'stdout', '--psm', '0', '-l', 'osd'],  # Determine proc for this step: detect scan orientation and save the source-to-working-page transform.
                          capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)  # Determine capture output for this step: detect scan orientation and save the source-to-working-page transform.
    match_rotation = re.search(r'^Rotate:\s*(\d+)', proc.stdout, re.MULTILINE)  # Determine match rotation for this step: detect scan orientation and save the source-to-working-page transform.
    match_confidence = re.search(r'^Orientation confidence:\s*([\d.]+)', proc.stdout, re.MULTILINE)  # Determine match confidence for this step: detect scan orientation and save the source-to-working-page transform.
    rotation = int(match_rotation.group(1)) if match_rotation else None  # Determine rotation for this step: detect scan orientation and save the source-to-working-page transform.
    confidence = float(match_confidence.group(1)) if match_confidence else None  # Determine confidence for this step: detect scan orientation and save the source-to-working-page transform.
    reliable = proc.returncode == 0 and rotation in (0, 90, 180, 270) and confidence is not None and confidence >= args.min_confidence  # Determine reliable for this step: detect scan orientation and save the source-to-working-page transform.
    applied = rotation if reliable else 0  # Determine applied for this step: detect scan orientation and save the source-to-working-page transform.
    corrected = image.rotate(-applied, expand=True)  # Determine corrected for this step: detect scan orientation and save the source-to-working-page transform.
    corrected.save(corrected_path, dpi=(args.dpi, args.dpi))  # Detect scan orientation and save the source-to-working-page transform.
    corrected.save(output_pdf, 'PDF', resolution=float(args.dpi))  # Detect scan orientation and save the source-to-working-page transform.
    original_hash = source_digest or file_sha256(args.pdf)  # Reuse the batch-verified source identity instead of rereading a long PDF for every scan page.
    manifest = {'source_pdf': str(args.pdf.resolve()), 'source_sha256': original_hash, 'source_page': args.page,  # Determine manifest for this step: detect scan orientation and save the source-to-working-page transform.
                'corrected_sha256': hashlib.sha256(output_pdf.read_bytes()).hexdigest(),  # Keep this audit field with the rest of its source evidence.
                'source_pdf_rotation_degrees': page.get_rotation(),  # Keep this audit field with the rest of its source evidence.
                'source_pdf_cropbox': list(page.get_cropbox()), 'source_pdf_mediabox': list(page.get_mediabox()),  # Keep this audit field with the rest of its source evidence.
                'source_display_size_points': [width_pt, height_pt], 'source_rendered_size_pixels': [width, height],  # Keep this audit field with the rest of its source evidence.
                'corrected_size_pixels': list(corrected.size), 'dpi': args.dpi, 'detected_clockwise_rotation': rotation,  # Keep this audit field with the rest of its source evidence.
                'osd_confidence': confidence, 'min_confidence': args.min_confidence, 'needs_review': not reliable,  # Keep this audit field with the rest of its source evidence.
                'applied_clockwise_rotation': applied, 'corrected_pixel_to_source_render_pixel': inverse_matrix(applied, width, height),  # Keep this audit field with the rest of its source evidence.
                'coordinate_note': 'Top-left pixel-boundary coordinates on rendered display page; PDF crop and rotation transforms still require validation.',  # Keep this audit field with the rest of its source evidence.
                'osd_error': proc.stderr.strip(), 'corrected_pdf': str(output_pdf.resolve())}  # Keep this audit field with the rest of its source evidence.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')  # Detect scan orientation and save the source-to-working-page transform.
    page.close()  # Release the document or file after its evidence has been preserved.
    pdf.close()  # Release the document or file after its evidence has been preserved.


def cell_box(rect: dict, height: float) -> tuple[float, float, float, float]:  # Normalize a detected four-corner cell to top-left page coordinates.
    xs = [rect[f'r_x{i}'] for i in range(4)]  # Determine xs for this step: normalize a detected four-corner cell to top-left page coordinates.
    ys = [rect[f'r_y{i}'] for i in range(4)]  # Determine ys for this step: normalize a detected four-corner cell to top-left page coordinates.
    if rect['coord_origin'] == 'BOTTOMLEFT':  # Check eligibility before continuing: normalize a detected four-corner cell to top-left page coordinates.
        ys = [height - y for y in ys]  # Determine ys for this step: normalize a detected four-corner cell to top-left page coordinates.
    elif rect['coord_origin'] != 'TOPLEFT':  # Use this alternative when earlier evidence does not justify the first route.
        raise ValueError(f"Unknown coordinate origin: {rect['coord_origin']}")  # Surface the inconsistency rather than silently publishing bad evidence.
    return min(xs), min(ys), max(xs), max(ys)  # Return the result to the next stage: normalize a detected four-corner cell to top-left page coordinates.


def provenance_box(bbox: dict, height: float) -> tuple[float, float, float, float]:  # Normalize a layout item box to top-left page points.
    xs, ys = [bbox['l'], bbox['r']], [bbox['t'], bbox['b']]  # Determine xs, ys for this step: normalize a layout item box to top-left page points.
    if bbox['coord_origin'] == 'BOTTOMLEFT':  # Check eligibility before continuing: normalize a layout item box to top-left page points.
        ys = [height - y for y in ys]  # Determine ys for this step: normalize a layout item box to top-left page points.
    elif bbox['coord_origin'] != 'TOPLEFT':  # Use this alternative when earlier evidence does not justify the first route.
        raise ValueError(f"Unknown coordinate origin: {bbox['coord_origin']}")  # Surface the inconsistency rather than silently publishing bad evidence.
    return min(xs), min(ys), max(xs), max(ys)  # Return the result to the next stage: normalize a layout item box to top-left page points.


def coverage(cell: tuple, item: tuple) -> float:  # Measure the fraction of a word or cell lying inside a layout item.
    x0, y0, x1, y1 = cell  # Measure the fraction of a word or cell lying inside a layout item.
    area = (x1 - x0) * (y1 - y0)  # Determine area for this step: measure the fraction of a word or cell lying inside a layout item.
    if area <= 0:  # Check eligibility before continuing: measure the fraction of a word or cell lying inside a layout item.
        return 0.0  # Return the result to the next stage: measure the fraction of a word or cell lying inside a layout item.
    overlap = max(0.0, min(x1, item[2]) - max(x0, item[0])) * max(0.0, min(y1, item[3]) - max(y0, item[1]))  # Determine overlap for this step: measure the fraction of a word or cell lying inside a layout item.
    return overlap / area  # Return the result to the next stage: measure the fraction of a word or cell lying inside a layout item.


def source_display_box(box: tuple, corrected_size: tuple, manifest: dict) -> list[float]:  # Convert corrected-page positions back to displayed original PDF points.
    """Map corrected PDF points to the source's displayed, cropped page in points."""
    cw, ch = manifest['corrected_size_pixels']  # Determine cw, ch for this step: convert corrected-page positions back to displayed original PDF points.
    sw, sh = manifest['source_rendered_size_pixels']  # Determine sw, sh for this step: convert corrected-page positions back to displayed original PDF points.
    pw, ph = manifest['source_display_size_points']  # Determine pw, ph for this step: convert corrected-page positions back to displayed original PDF points.
    matrix = manifest['corrected_pixel_to_source_render_pixel']  # Determine matrix for this step: convert corrected-page positions back to displayed original PDF points.
    points = []  # Determine points for this step: convert corrected-page positions back to displayed original PDF points.
    for x in (box[0], box[2]):  # Handle each x separately: convert corrected-page positions back to displayed original PDF points.
        for y in (box[1], box[3]):  # Handle each y separately: convert corrected-page positions back to displayed original PDF points.
            px, py = x * cw / corrected_size[0], y * ch / corrected_size[1]  # Determine px, py for this step: convert corrected-page positions back to displayed original PDF points.
            sx = matrix[0][0] * px + matrix[0][1] * py + matrix[0][2]  # Determine sx for this step: convert corrected-page positions back to displayed original PDF points.
            sy = matrix[1][0] * px + matrix[1][1] * py + matrix[1][2]  # Determine sy for this step: convert corrected-page positions back to displayed original PDF points.
            points.append((sx * pw / sw, sy * ph / sh))  # Convert corrected-page positions back to displayed original PDF points.
    return [min(p[0] for p in points), min(p[1] for p in points),  # Return the result to the next stage: convert corrected-page positions back to displayed original PDF points.
            max(p[0] for p in points), max(p[1] for p in points)]  # Convert corrected-page positions back to displayed original PDF points.


def link(report: dict, threshold: float = 0.5, orientation: dict | None = None, verified_source=None) -> dict:  # Associate observed cells with candidate layout items while retaining ambiguity.
    if orientation:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
        if Path(report['input']).resolve() != Path(orientation['corrected_pdf']).resolve():  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
            raise ValueError('Orientation manifest does not match the processed PDF')  # Surface the inconsistency rather than silently publishing bad evidence.
        if orientation.get('corrected_sha256'):  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
            if report.get('input_sha256') != orientation['corrected_sha256']:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                raise ValueError('Probe does not identify the corrected PDF in the manifest')  # Surface the inconsistency rather than silently publishing bad evidence.
            with Path(report['input']).open('rb') as corrected:  # Associate observed cells with candidate layout items while retaining ambiguity.
                if hashlib.file_digest(corrected, 'sha256').hexdigest() != orientation['corrected_sha256']:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                    raise ValueError('Corrected PDF changed since orientation preparation')  # Surface the inconsistency rather than silently publishing bad evidence.
        if (verified_source or file_sha256(Path(orientation['source_pdf']))) != orientation['source_sha256']:  # Tie page coordinates to the worker's verified source version.
            raise ValueError('Source PDF differs from the orientation record')  # Do not publish boxes against the wrong original.
        if len(report['parsed_pages']) != 1:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
            raise ValueError('Orientation manifest currently supports one corrected page only')  # Surface the inconsistency rather than silently publishing bad evidence.
    pages = {p['page_no']: p for p in report['parsed_pages']}  # Determine pages for this step: associate observed cells with candidate layout items while retaining ambiguity.
    items = []  # Determine items for this step: associate observed cells with candidate layout items while retaining ambiguity.
    for key in ('texts', 'tables', 'pictures'):  # Handle each key separately: associate observed cells with candidate layout items while retaining ambiguity.
        for item in report['document'].get(key, []):  # Handle each item separately: associate observed cells with candidate layout items while retaining ambiguity.
            items.append(item)  # Associate observed cells with candidate layout items while retaining ambiguity.
    candidates: dict[str, list] = {item['self_ref']: [] for item in items}  # Associate observed cells with candidate layout items while retaining ambiguity.
    unlinked = []  # Determine unlinked for this step: associate observed cells with candidate layout items while retaining ambiguity.
    total = 0  # Determine total for this step: associate observed cells with candidate layout items while retaining ambiguity.
    native_total = 0  # Determine native total for this step: associate observed cells with candidate layout items while retaining ambiguity.
    native_unlinked = 0  # Determine native unlinked for this step: associate observed cells with candidate layout items while retaining ambiguity.
    for page_no, page in pages.items():  # Handle each page no, page separately: associate observed cells with candidate layout items while retaining ambiguity.
        parsed = page.get('parsed_page') or {}  # Determine parsed for this step: associate observed cells with candidate layout items while retaining ambiguity.
        height = page['size']['height']  # Determine height for this step: associate observed cells with candidate layout items while retaining ambiguity.
        for kind in ('word_cells', 'textline_cells'):  # Handle each kind separately: associate observed cells with candidate layout items while retaining ambiguity.
            for cell in parsed.get(kind, []):  # Handle each cell separately: associate observed cells with candidate layout items while retaining ambiguity.
                is_ocr = bool(cell.get('from_ocr'))  # Determine is ocr for this step: associate observed cells with candidate layout items while retaining ambiguity.
                if is_ocr:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                    total += 1  # Associate observed cells with candidate layout items while retaining ambiguity.
                else:  # Use this alternative when earlier evidence does not justify the first route.
                    native_total += 1  # Associate observed cells with candidate layout items while retaining ambiguity.
                box = cell_box(cell['rect'], height)  # Determine box for this step: associate observed cells with candidate layout items while retaining ambiguity.
                matches = []  # Determine matches for this step: associate observed cells with candidate layout items while retaining ambiguity.
                for item in items:  # Handle each item separately: associate observed cells with candidate layout items while retaining ambiguity.
                    for prov in item.get('prov', []):  # Handle each prov separately: associate observed cells with candidate layout items while retaining ambiguity.
                        if prov['page_no'] == page_no:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                            value = coverage(box, provenance_box(prov['bbox'], height))  # Determine value for this step: associate observed cells with candidate layout items while retaining ambiguity.
                            if value >= threshold:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                                matches.append((value, item['self_ref']))  # Associate observed cells with candidate layout items while retaining ambiguity.
                record = {'page_no': page_no, 'cell_kind': kind, 'cell_index': cell['index'],  # Determine record for this step: associate observed cells with candidate layout items while retaining ambiguity.
                          'text': cell['text'], 'source': 'ocr' if is_ocr else 'native',  # Keep this audit field with the rest of its source evidence.
                          'confidence': cell.get('confidence') if is_ocr else None,  # Keep this audit field with the rest of its source evidence.
                          'bbox_top_left': list(box)}  # Keep this audit field with the rest of its source evidence.
                if orientation:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                    record['source_page_no'] = orientation['source_page']  # Associate observed cells with candidate layout items while retaining ambiguity.
                    record['source_display_bbox_top_left'] = source_display_box(box, (page['size']['width'], height), orientation)  # Associate observed cells with candidate layout items while retaining ambiguity.
                if not matches:  # Keep observed cells even when no document item can contain them.
                    unlinked.append(record)  # Associate observed cells with candidate layout items while retaining ambiguity.
                    if not is_ocr:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                        native_unlinked += 1  # Associate observed cells with candidate layout items while retaining ambiguity.
                    continue  # Move to the next item without counting unavailable evidence.
                matches.sort(reverse=True)  # Associate observed cells with candidate layout items while retaining ambiguity.
                best = matches[0][0]  # Determine best for this step: associate observed cells with candidate layout items while retaining ambiguity.
                tied = sorted({ref for value, ref in matches if abs(value - best) < 0.01})  # Determine tied for this step: associate observed cells with candidate layout items while retaining ambiguity.
                if len(tied) != 1:  # Keep overlapping candidate items unresolved rather than guessing.
                    unlinked.append({**record, 'ambiguous_candidates': tied})  # Associate observed cells with candidate layout items while retaining ambiguity.
                    if not is_ocr:  # Check eligibility before continuing: associate observed cells with candidate layout items while retaining ambiguity.
                        native_unlinked += 1  # Associate observed cells with candidate layout items while retaining ambiguity.
                    continue  # Move to the next item without counting unavailable evidence.
                candidates[tied[0]].append({**record, 'coverage': round(best, 4)})  # Associate observed cells with candidate layout items while retaining ambiguity.
    return {'source': orientation['source_pdf'] if orientation else report['input'], 'processed_pdf': report['input'], 'engine': report['engine'], 'coordinate_system': 'PDF points, top-left origin, relative to parsed page',  # Return the result to the next stage: associate observed cells with candidate layout items while retaining ambiguity.
            'source_coordinate_system': 'PDF points, top-left of displayed cropped page' if orientation else None,  # Keep this audit field with the rest of its source evidence.
            'method': 'geometric overlap; links are unverified', 'minimum_coverage': threshold,  # Keep this audit field with the rest of its source evidence.
            'ocr_cell_count': total, 'linked_cell_count': total - (len(unlinked) - native_unlinked),  # Keep this audit field with the rest of its source evidence.
            'native_cell_count': native_total, 'native_linked_cell_count': native_total - native_unlinked,  # Keep this audit field with the rest of its source evidence.
            'unlinked_or_ambiguous': unlinked,  # Keep this audit field with the rest of its source evidence.
            'items': [{'ref': item['self_ref'], 'text': item.get('text'), 'cells': candidates[item['self_ref']]} for item in items]}  # Keep this audit field with the rest of its source evidence.


def make_converter(engine, artifacts, force_ocr=False, detect_orientation=False,  # Build a reusable CPU converter for the selected extraction route.
                   tesseract=TESSERACT):  # Determine tesseract for this step: build a reusable CPU converter for the selected extraction route.
    from docling.datamodel.base_models import InputFormat  # Load this processing capability only in the worker that needs it.
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions, TesseractCliOcrOptions  # Load this processing capability only in the worker that needs it.
    from docling.document_converter import DocumentConverter, PdfFormatOption  # Load this processing capability only in the worker that needs it.
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions  # Load this processing capability only in the worker that needs it.
    mode = OcrMode.FULL_PAGE if force_ocr else OcrMode.DEFAULT  # Determine mode for this step: build a reusable CPU converter for the selected extraction route.
    if engine == 'tesseract':  # Check eligibility before continuing: build a reusable CPU converter for the selected extraction route.
        ocr = TesseractCliOcrOptions(lang=['eng'], tesseract_cmd=tesseract, mode=mode)  # Determine ocr for this step: build a reusable CPU converter for the selected extraction route.
    else:  # Use this alternative when earlier evidence does not justify the first route.
        backend = 'onnxruntime' if engine == 'rapid-onnx' else 'paddle'  # Determine backend for this step: build a reusable CPU converter for the selected extraction route.
        ocr = RapidOcrOptions(lang=['en'], backend=backend, mode=mode, detect_orientation=detect_orientation)  # Determine ocr for this step: build a reusable CPU converter for the selected extraction route.
    options = PdfPipelineOptions(artifacts_path=artifacts.resolve(), ocr_options=ocr, generate_parsed_pages=True,  # Determine options for this step: build a reusable CPU converter for the selected extraction route.
                                 accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU))  # Determine accelerator options for this step: build a reusable CPU converter for the selected extraction route.
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})  # Return the result to the next stage: build a reusable CPU converter for the selected extraction route.


def convert_report(converter, pdf, engine, force_ocr=False, detect_orientation=False):  # Keep structured output and parsed cell evidence from the same conversion.
    with pdf.open('rb') as source:  # Keep structured output and parsed cell evidence from the same conversion.
        input_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()  # Determine input sha256 for this step: keep structured output and parsed cell evidence from the same conversion.
    start = time.perf_counter()  # Determine start for this step: keep structured output and parsed cell evidence from the same conversion.
    result = converter.convert(pdf)  # Determine result for this step: keep structured output and parsed cell evidence from the same conversion.
    elapsed = time.perf_counter() - start  # Determine elapsed for this step: keep structured output and parsed cell evidence from the same conversion.
    return {'engine': engine, 'input': str(pdf.resolve()), 'input_sha256': input_sha256, 'force_ocr': force_ocr,  # Return the result to the next stage: keep structured output and parsed cell evidence from the same conversion.
            'detect_orientation': detect_orientation, 'status': str(result.status), 'elapsed_seconds': elapsed,  # Keep this audit field with the rest of its source evidence.
            'document': result.document.export_to_dict(),  # Keep this audit field with the rest of its source evidence.
            'parsed_pages': [p.model_dump(mode='json', exclude={'predictions', 'assembled'}) for p in result.pages]}  # Keep this audit field with the rest of its source evidence.


def normal_fragment(text):  # Compare OCR fragments to item text without punctuation-only mismatches.
    return ''.join(ch.casefold() for ch in text if ch.isalnum())  # Return the result to the next stage: compare OCR fragments to item text without punctuation-only mismatches.


def make_word_model(backend, artifacts):  # Build the independent word-box recognizer once per worker.
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions  # Load this processing capability only in the worker that needs it.
    from docling.datamodel.pipeline_options import RapidOcrOptions  # Load this processing capability only in the worker that needs it.
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel  # Load this processing capability only in the worker that needs it.
    return RapidOcrModel(enabled=True, artifacts_path=artifacts.resolve(), options=RapidOcrOptions(lang=['en'], backend=backend),  # Return the result to the next stage: build the independent word-box recognizer once per worker.
                         accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU))  # Determine accelerator options for this step: build the independent word-box recognizer once per worker.


def extract_words(pdf, data, backend, model, page_number=1, dpi=216, manifest=None, verified_source=None):  # Recognize raw fragments with original-page boxes and model scores.
    import numpy as np  # Load this processing capability only in the worker that needs it.
    with pdf.open('rb') as f:  # Recognize raw fragments with original-page boxes and model scores.
        digest = hashlib.file_digest(f, 'sha256').hexdigest()  # Determine digest for this step: recognize raw fragments with original-page boxes and model scores.
    if digest != data['input_sha256']:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
        raise ValueError('PDF content differs from Docling input')  # Surface the inconsistency rather than silently publishing bad evidence.
    if data['engine'] != ('rapid-onnx' if backend == 'onnxruntime' else 'rapid-paddle'):  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
        raise ValueError('Word backend must match the existing Docling OCR backend')  # Surface the inconsistency rather than silently publishing bad evidence.
    if manifest:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
        if page_number != 1 or Path(manifest['corrected_pdf']).resolve() != pdf.resolve() or manifest.get('corrected_sha256') != digest:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
            raise ValueError('Orientation manifest does not match this corrected single-page PDF')  # Surface the inconsistency rather than silently publishing bad evidence.
        if (verified_source or file_sha256(Path(manifest['source_pdf']))) != manifest['source_sha256']:  # Share the document-level verification across its pages.
            raise ValueError('Source version differs from the scan orientation record')  # Refuse a mismatched source and page pair.
    doc = pdfium.PdfDocument(str(pdf))  # Determine doc for this step: recognize raw fragments with original-page boxes and model scores.
    if page_number < 1 or page_number > len(doc) or dpi < 72:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
        raise ValueError('Invalid page or DPI')  # Surface the inconsistency rather than silently publishing bad evidence.
    page = doc[page_number-1]; width, height = page.get_size()  # Determine page for this step: recognize raw fragments with original-page boxes and model scores.
    image = page.render(scale=dpi/72).to_pil().convert('RGB')  # Determine image for this step: recognize raw fragments with original-page boxes and model scores.
    page.close()  # Release the document or file after its evidence has been preserved.
    doc.close()  # Release the document or file after its evidence has been preserved.
    started = time.perf_counter()  # Determine started for this step: recognize raw fragments with original-page boxes and model scores.
    result = model.reader(np.asarray(image), return_word_box=True)  # Determine result for this step: recognize raw fragments with original-page boxes and model scores.
    elapsed = time.perf_counter()-started  # Determine elapsed for this step: recognize raw fragments with original-page boxes and model scores.
    if result is None or not result.word_results:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
        raise ValueError('RapidOCR returned no word-level results; no synthetic word boxes were fabricated')  # Surface the inconsistency rather than silently publishing bad evidence.
    document = data['document']  # Determine document for this step: recognize raw fragments with original-page boxes and model scores.
    items = [item for k in ('texts', 'tables', 'pictures') for item in document.get(k, [])]  # Determine items for this step: recognize raw fragments with original-page boxes and model scores.
    rows = []  # Determine rows for this step: recognize raw fragments with original-page boxes and model scores.
    for line_index, words in enumerate(result.word_results):  # Handle each line index, words separately: recognize raw fragments with original-page boxes and model scores.
        if not words:  # Keep empty OCR lines out of the word inventory.
            continue  # Move to the next item without counting unavailable evidence.
        for word, score, polygon in words:  # Handle each word, score, polygon separately: recognize raw fragments with original-page boxes and model scores.
            if not polygon:  # A missing polygon cannot support a word-level source box.
                continue  # Move to the next item without counting unavailable evidence.
            xs = [p[0]*width/image.width for p in polygon]; ys = [p[1]*height/image.height for p in polygon]  # Determine xs for this step: recognize raw fragments with original-page boxes and model scores.
            bbox = (min(xs), min(ys), max(xs), max(ys))  # Determine bbox for this step: recognize raw fragments with original-page boxes and model scores.
            candidates = []  # Determine candidates for this step: recognize raw fragments with original-page boxes and model scores.
            for item in items:  # Handle each item separately: recognize raw fragments with original-page boxes and model scores.
                for prov in item.get('prov', []):  # Handle each prov separately: recognize raw fragments with original-page boxes and model scores.
                    if prov['page_no'] == page_number:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
                        overlap = coverage(bbox, provenance_box(prov['bbox'], height))  # Determine overlap for this step: recognize raw fragments with original-page boxes and model scores.
                        if overlap >= 0.5:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
                            candidates.append((overlap, item))  # Recognize raw fragments with original-page boxes and model scores.
            candidates.sort(key=lambda x: x[0], reverse=True)  # Recognize raw fragments with original-page boxes and model scores.
            tied = [x for x in candidates if abs(x[0]-candidates[0][0]) < 0.01] if candidates else []  # Determine tied for this step: recognize raw fragments with original-page boxes and model scores.
            if not candidates:  # Check eligibility before continuing: recognize raw fragments with original-page boxes and model scores.
                status, ref = 'unmatched', ''  # Determine status, ref for this step: recognize raw fragments with original-page boxes and model scores.
            elif len({x[1]['self_ref'] for x in tied}) > 1:  # Use this alternative when earlier evidence does not justify the first route.
                status, ref = 'ambiguous', ''  # Determine status, ref for this step: recognize raw fragments with original-page boxes and model scores.
            else:  # Use this alternative when earlier evidence does not justify the first route.
                item = candidates[0][1]; ref = item['self_ref']  # Determine item for this step: recognize raw fragments with original-page boxes and model scores.
                status = 'candidate' if normal_fragment(word) and normal_fragment(word) in normal_fragment(item.get('text', '')) else 'text_mismatch'  # Determine status for this step: recognize raw fragments with original-page boxes and model scores.
            source_bbox = source_display_box(bbox, (width, height), manifest) if manifest else bbox  # Determine source bbox for this step: recognize raw fragments with original-page boxes and model scores.
            rows.append({'source_pdf': manifest['source_pdf'] if manifest else str(pdf.resolve()),  # Recognize raw fragments with original-page boxes and model scores.
                         'source_sha256': manifest['source_sha256'] if manifest else digest, 'engine': backend,  # Keep this audit field with the rest of its source evidence.
                         'coordinate_system': 'displayed cropped PDF page, top-left points',  # Keep this audit field with the rest of its source evidence.
                         'page': manifest['source_page'] if manifest else page_number, 'word': word, 'confidence': float(score),  # Keep this audit field with the rest of its source evidence.
                         'x0': source_bbox[0], 'y0': source_bbox[1], 'x1': source_bbox[2], 'y1': source_bbox[3],  # Keep this audit field with the rest of its source evidence.
                         'item_ref': ref, 'link_status': status, 'ocr_line_index': line_index})  # Keep this audit field with the rest of its source evidence.
    return rows, elapsed  # Return the result to the next stage: recognize raw fragments with original-page boxes and model scores.


def write_word_csv(output, rows):  # Checkpoint the scan word pass without flattening its confidence or provenance.
    output.parent.mkdir(parents=True, exist_ok=True)  # Reserve a private location for this processing attempt.
    tmp = output.with_suffix(output.suffix + '.tmp')  # Determine tmp for this step: checkpoint the scan word pass without flattening its confidence or provenance.
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:  # Checkpoint the scan word pass without flattening its confidence or provenance.
        writer = csv.DictWriter(f, fieldnames=['source_pdf','source_sha256','engine','coordinate_system','page','word','confidence','x0','y0','x1','y1','item_ref','link_status','ocr_line_index'])  # Determine writer for this step: checkpoint the scan word pass without flattening its confidence or provenance.
        writer.writeheader(); writer.writerows(rows)  # Label the evidence columns for audit and recovery.
    os.replace(tmp, output)  # Publish a finished artifact without exposing a partial write.


def normal(text):  # Normalize Unicode and case while retaining meaningful punctuation for reconciliation.
    # Do not strip punctuation: numeric and punctuation changes require review.
    return unicodedata.normalize('NFKC', str(text)).casefold()  # Return the result to the next stage: normalize Unicode and case while retaining meaningful punctuation for reconciliation.


def word_box(word):  # Read each observed box in its original source-page coordinate frame.
    return [float(word[k]) for k in ('x0', 'y0', 'x1', 'y1')]  # Return the result to the next stage: read each observed box in its original source-page coordinate frame.


def candidate_cell(box, cells):  # Offer a table-cell candidate only when its geometry is sufficiently distinct.
    matches = sorted(((coverage(box, cell['bbox']), cell['id']) for cell in cells), reverse=True)  # Determine matches for this step: offer a table-cell candidate only when its geometry is sufficiently distinct.
    if not matches or matches[0][0] < .7:  # Withhold table-cell assignment below the overlap threshold.
        return ''  # Return the result to the next stage: offer a table-cell candidate only when its geometry is sufficiently distinct.
    if len(matches) > 1 and matches[0][0] - matches[1][0] < .01:  # Withhold placement when two table cells are nearly tied.
        return ''  # Return the result to the next stage: offer a table-cell candidate only when its geometry is sufficiently distinct.
    return matches[0][1]  # Return the result to the next stage: offer a table-cell candidate only when its geometry is sufficiently distinct.


def align(tokens, indexes, words, line_ref, cells):  # Propose exact ordered text matches without merging unrelated or numeric fragments.
    """Return exact, ordered token/fragment groups; unresolved material is not invented."""
    @lru_cache(None)  # Propose exact ordered text matches without merging unrelated or numeric fragments.
    def best(t, i):  # Find the strongest compatible alignment and leave other material unresolved.
        if t == len(tokens) or i == len(indexes):  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
            return (0, 0), ()  # Return the result to the next stage: find the strongest compatible alignment and leave other material unresolved.
        options = [best(t + 1, i), best(t, i + 1)]  # Determine options for this step: find the strongest compatible alignment and leave other material unresolved.
        target = normal(tokens[t])  # Determine target for this step: find the strongest compatible alignment and leave other material unresolved.
        if target and line_ref:  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
            for count in range(1, min(5, len(indexes) - i) + 1):  # Handle each count separately: find the strongest compatible alignment and leave other material unresolved.
                group = [words[indexes[j]] for j in range(i, i + count)]  # Determine group for this step: find the strongest compatible alignment and leave other material unresolved.
                if any(w.get('item_ref') != line_ref or w.get('link_status') in ('unmatched', 'ambiguous') for w in group):  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                    break  # Stop considering this candidate when its eligibility no longer holds.
                if count > 1:  # Apply extra safeguards to proposals that join multiple fragments.
                    # Merging arbitrary line tokens can join distinct words or numbers when
                    # line OCR has lost spaces. Restrict proposals to alphabetic fragments;
                    # they still require review and are never ground truth.
                    if len(target) < 4 or not all(str(w['word']).isalpha() for w in group):  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                        break  # Stop considering this candidate when its eligibility no longer holds.
                    a, b = word_box(group[-2]), word_box(group[-1])  # Determine a, b for this step: find the strongest compatible alignment and leave other material unresolved.
                    if b[0] - a[2] > .33 * max(a[3]-a[1], b[3]-b[1]):  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                        break  # Stop considering this candidate when its eligibility no longer holds.
                    if line_ref.startswith('#/tables/'):  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                        assignments = [candidate_cell(word_box(w), cells.get(line_ref, [])) for w in group]  # Determine assignments for this step: find the strongest compatible alignment and leave other material unresolved.
                        if not assignments[0] or len(set(assignments)) != 1:  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                            break  # Stop considering this candidate when its eligibility no longer holds.
                        assigned_cell = next(c for c in cells[line_ref] if c['id'] == assignments[0])  # Determine assigned cell for this step: find the strongest compatible alignment and leave other material unresolved.
                        if assigned_cell.get('row_span', 1) != 1 or assigned_cell.get('col_span', 1) != 1:  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                            break  # Stop considering this candidate when its eligibility no longer holds.
                combined = normal(''.join(str(w['word']) for w in group))  # Determine combined for this step: find the strongest compatible alignment and leave other material unresolved.
                if combined == target:  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                    score, tail = best(t + 1, i + count)  # Determine score, tail for this step: find the strongest compatible alignment and leave other material unresolved.
                    options.append(((score[0] + len(target), score[1] + 1), ((t, i, count),) + tail))  # Find the strongest compatible alignment and leave other material unresolved.
                if len(combined) > len(target):  # Check eligibility before continuing: find the strongest compatible alignment and leave other material unresolved.
                    break  # Stop considering this candidate when its eligibility no longer holds.
        return max(options, key=lambda x: (x[0], len(x[1])))  # Return the result to the next stage: find the strongest compatible alignment and leave other material unresolved.
    return best(0, 0)[1]  # Return the result to the next stage: propose exact ordered text matches without merging unrelated or numeric fragments.


def reconcile(lines, words, cells=None):  # Keep OCR evidence alongside tentative word merges and disagreement flags.
    cells = cells or {}  # Determine cells for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
    assigned = {i: [] for i in range(len(lines))}  # Determine assigned for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
    ambiguous, no_line = set(), set()  # Determine ambiguous, no line for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
    for wi, word in enumerate(words):  # Handle each wi, word separately: keep OCR evidence alongside tentative word merges and disagreement flags.
        matches = sorted(((coverage(word_box(word), line['bbox']), li) for li, line in enumerate(lines)), reverse=True)  # Determine matches for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        if not matches or matches[0][0] < .5:  # Check eligibility before continuing: keep OCR evidence alongside tentative word merges and disagreement flags.
            no_line.add(wi); continue  # Move to the next item without counting unavailable evidence.
        if len(matches) > 1 and matches[1][0] >= .5 and matches[0][0] - matches[1][0] < .01:  # Check eligibility before continuing: keep OCR evidence alongside tentative word merges and disagreement flags.
            ambiguous.add(wi); continue  # Move to the next item without counting unavailable evidence.
        assigned[matches[0][1]].append(wi)  # Keep OCR evidence alongside tentative word merges and disagreement flags.
    merged, consumed, audit = {}, set(), []  # Keep OCR evidence alongside tentative word merges and disagreement flags.
    for li, line in enumerate(lines):  # Handle each li, line separately: keep OCR evidence alongside tentative word merges and disagreement flags.
        indexes = sorted(assigned[li], key=lambda i: (word_box(words[i])[0], word_box(words[i])[1], i))  # Determine indexes for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        tokens = re.findall(r'\S+', line['text'])  # Determine tokens for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        matched_tokens = set()  # Determine matched tokens for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        for ti, start, count in align(tokens, indexes, words, line.get('item_ref', ''), cells):  # Handle each ti, start, count separately: keep OCR evidence alongside tentative word merges and disagreement flags.
            ids = indexes[start:start + count]; group = [words[i] for i in ids]  # Determine ids for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
            boxes = [word_box(w) for w in group]  # Determine boxes for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
            ref = line.get('item_ref', '')  # Determine ref for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
            status = ('review_proposed_merge' if count > 1 else  # Determine status for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
                      'review_item_text_mismatch' if group[0].get('link_status') == 'text_mismatch' else 'agreement')  # Keep this audit field with the rest of its source evidence.
            row = {'text': tokens[ti], 'status': status,  # Determine row for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
                   'bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)],  # Keep this audit field with the rest of its source evidence.
                   'line_id': line['line_id'], 'line_text': line['text'], 'line_confidence': line.get('confidence'),  # Keep this audit field with the rest of its source evidence.
                   'item_ref': ref, 'table_cell_candidate': candidate_cell(boxes[0], cells.get(ref, [])) if ref.startswith('#/tables/') else '',  # Keep this audit field with the rest of its source evidence.
                   'fragments': [{**w, 'fragment_index': i} for i, w in zip(ids, group)]}  # Keep this audit field with the rest of its source evidence.
            merged[ids[0]] = row; consumed.update(ids); matched_tokens.add(ti)  # Keep OCR evidence alongside tentative word merges and disagreement flags.
        audit.append({'line_id': line['line_id'], 'item_ref': line.get('item_ref', ''), 'line_text': line['text'],  # Keep OCR evidence alongside tentative word merges and disagreement flags.
                      'bbox': line['bbox'], 'unmatched_line_tokens': [t for i, t in enumerate(tokens) if i not in matched_tokens],  # Keep this audit field with the rest of its source evidence.
                      'matched_token_count': len(matched_tokens), 'fragment_count': len(indexes)})  # Keep this audit field with the rest of its source evidence.
    word_line = {word_index: li for li, indexes in assigned.items() for word_index in indexes}  # Determine word line for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
    output = []  # Determine output for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
    for i, word in enumerate(words):  # Handle each i, word separately: keep OCR evidence alongside tentative word merges and disagreement flags.
        if i in merged:  # Check eligibility before continuing: keep OCR evidence alongside tentative word merges and disagreement flags.
            output.append(merged[i]); continue  # Move to the next item without counting unavailable evidence.
        if i in consumed:  # Do not output a raw fragment twice in the proposal inventory.
            continue  # Move to the next item without counting unavailable evidence.
        status = 'review_ambiguous_line' if i in ambiguous else 'review_no_line' if i in no_line else 'review_disagreement'  # Determine status for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        matched_line = lines[word_line[i]] if i in word_line else None  # Determine matched line for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        ref = word.get('item_ref', '')  # Determine ref for this step: keep OCR evidence alongside tentative word merges and disagreement flags.
        output.append({'text': word['word'], 'status': status, 'bbox': word_box(word),  # Keep OCR evidence alongside tentative word merges and disagreement flags.
                       'line_id': matched_line['line_id'] if matched_line else '', 'line_text': matched_line['text'] if matched_line else '',  # Keep this audit field with the rest of its source evidence.
                       'line_confidence': matched_line.get('confidence') if matched_line else None, 'item_ref': ref,  # Keep this audit field with the rest of its source evidence.
                       'table_cell_candidate': candidate_cell(word_box(word), cells.get(ref, [])) if ref.startswith('#/tables/') else '',  # Keep this audit field with the rest of its source evidence.
                       'fragments': [{**word, 'fragment_index': i}]})  # Keep this audit field with the rest of its source evidence.
    return output, audit  # Return the result to the next stage: keep OCR evidence alongside tentative word merges and disagreement flags.


def prepare_inputs(links, parsed, orientation):  # Bring OCR lines and table cells into a common source-page coordinate frame.
    lines = []  # Determine lines for this step: bring OCR lines and table cells into a common source-page coordinate frame.
    for item in links['items']:  # Handle each item separately: bring OCR lines and table cells into a common source-page coordinate frame.
        for c in item['cells']:  # Handle each c separately: bring OCR lines and table cells into a common source-page coordinate frame.
            if c['source'] == 'ocr' and c['cell_kind'] == 'textline_cells':  # Check eligibility before continuing: bring OCR lines and table cells into a common source-page coordinate frame.
                lines.append({'text': c['text'], 'bbox': c['source_display_bbox_top_left'], 'item_ref': item['ref'],  # Bring OCR lines and table cells into a common source-page coordinate frame.
                              'line_id': f"{c['cell_kind']}:{c['cell_index']}", 'confidence': c.get('confidence')})  # Keep this audit field with the rest of its source evidence.
    for c in links['unlinked_or_ambiguous']:  # Handle each c separately: bring OCR lines and table cells into a common source-page coordinate frame.
        if c['source'] == 'ocr' and c['cell_kind'] == 'textline_cells':  # Check eligibility before continuing: bring OCR lines and table cells into a common source-page coordinate frame.
            lines.append({'text': c['text'], 'bbox': c['source_display_bbox_top_left'], 'item_ref': '',  # Bring OCR lines and table cells into a common source-page coordinate frame.
                          'line_id': f"{c['cell_kind']}:{c['cell_index']}", 'confidence': c.get('confidence')})  # Keep this audit field with the rest of its source evidence.
    size = parsed['parsed_pages'][0]['size']; width, height = size['width'], size['height']  # Determine size for this step: bring OCR lines and table cells into a common source-page coordinate frame.
    cells = {}  # Determine cells for this step: bring OCR lines and table cells into a common source-page coordinate frame.
    for table in parsed['document'].get('tables', []):  # Handle each table separately: bring OCR lines and table cells into a common source-page coordinate frame.
        entries = []  # Determine entries for this step: bring OCR lines and table cells into a common source-page coordinate frame.
        for i, cell in enumerate(table['data'].get('table_cells', [])):  # Handle each i, cell separately: bring OCR lines and table cells into a common source-page coordinate frame.
            if not cell.get('bbox'): continue  # Check eligibility before continuing: bring OCR lines and table cells into a common source-page coordinate frame.
            box = provenance_box(cell['bbox'], height)  # Determine box for this step: bring OCR lines and table cells into a common source-page coordinate frame.
            if orientation: box = source_display_box(box, (width, height), orientation)  # Check eligibility before continuing: bring OCR lines and table cells into a common source-page coordinate frame.
            entries.append({'id': f"r{cell.get('start_row_offset_idx')}c{cell.get('start_col_offset_idx')}:cell{i}", 'bbox': box,  # Bring OCR lines and table cells into a common source-page coordinate frame.
                            'row': cell.get('start_row_offset_idx'), 'column': cell.get('start_col_offset_idx'),  # Keep this audit field with the rest of its source evidence.
                            'row_span': cell.get('row_span', 1), 'col_span': cell.get('col_span', 1), 'text': cell.get('text', ''),  # Keep this audit field with the rest of its source evidence.
                            'table_columns': table['data'].get('num_cols')})  # Keep this audit field with the rest of its source evidence.
        cells[table['self_ref']] = entries  # Bring OCR lines and table cells into a common source-page coordinate frame.
    return lines, cells  # Return the result to the next stage: bring OCR lines and table cells into a common source-page coordinate frame.


def table_row_audit(lines, cells, results, audit):  # Flag unsupported table rows, spans, missing slots, and ambiguous placements.
    """Group available line/word evidence by candidate table row without certifying structure."""
    line_audit = {a['line_id']: a for a in audit}  # Determine line audit for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
    output = []  # Determine output for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
    for ref, entries in cells.items():  # Handle each ref, entries separately: flag unsupported table rows, spans, missing slots, and ambiguous placements.
        by_id = {c['id']: c for c in entries}  # Determine by id for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
        grouped = {}  # Determine grouped for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
        for cell in entries:  # Handle each cell separately: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            grouped.setdefault(cell.get('row'), {'cells': [], 'lines': [], 'words': []})['cells'].append(cell)  # Flag unsupported table rows, spans, missing slots, and ambiguous placements.
        unassigned = []  # Determine unassigned for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
        for line in (x for x in lines if x.get('item_ref') == ref):  # Handle each line separately: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            cell_id = candidate_cell(line['bbox'], entries)  # Determine cell id for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            if not cell_id or cell_id not in by_id:  # Check eligibility before continuing: flag unsupported table rows, spans, missing slots, and ambiguous placements.
                unassigned.append(line['line_id']); continue  # Move to the next item without counting unavailable evidence.
            grouped[by_id[cell_id].get('row')]['lines'].append(line)  # Flag unsupported table rows, spans, missing slots, and ambiguous placements.
        for word in (x for x in results if x.get('item_ref') == ref):  # Handle each word separately: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            cell_id = word.get('table_cell_candidate') or candidate_cell(word['bbox'], entries)  # Determine cell id for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            cell = by_id.get(cell_id)  # Determine cell for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            if cell: grouped[cell.get('row')]['words'].append(word)  # Check eligibility before continuing: flag unsupported table rows, spans, missing slots, and ambiguous placements.
        for row_no, group in sorted(grouped.items(), key=lambda x: (x[0] is None, x[0] or 0)):  # Handle each row no, group separately: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            unmatched = sum(len(line_audit[x['line_id']]['unmatched_line_tokens']) for x in group['lines'])  # Determine unmatched for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            reviews = sum(x['status'].startswith('review_') for x in group['words'])  # Determine reviews for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            spans = sum(c.get('row_span', 1) != 1 or c.get('col_span', 1) != 1 for c in group['cells'])  # Determine spans for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            total_columns = next((c['table_columns'] for c in entries if isinstance(c.get('table_columns'), int)), None)  # Determine total columns for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            occupied = {i for c in group['cells'] if isinstance(c.get('column'), int)  # Determine occupied for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
                        for i in range(c['column'], c['column'] + c.get('col_span', 1))}  # Handle each i separately: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            missing = [i for i in range(total_columns) if i not in occupied] if total_columns is not None else []  # Determine missing for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            status = 'review_no_line_evidence' if not group['lines'] else 'review' if unmatched or reviews or spans or missing else 'line_word_agreement_unverified'  # Determine status for this step: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            output.append({'table_ref': ref, 'row': row_no, 'status': status,  # Flag unsupported table rows, spans, missing slots, and ambiguous placements.
                           'candidate_cells': [c['id'] for c in group['cells']], 'cell_texts': [c.get('text', '') for c in group['cells']],  # Keep this audit field with the rest of its source evidence.
                           'line_ids': [x['line_id'] for x in group['lines']], 'line_texts': [x['text'] for x in group['lines']],  # Keep this audit field with the rest of its source evidence.
                           'unmatched_line_tokens': unmatched, 'review_words': reviews, 'spanning_cells': spans,  # Keep this audit field with the rest of its source evidence.
                           'missing_column_slots': missing,  # Keep this audit field with the rest of its source evidence.
                           'note': 'Table structure is unverified; missing slots can be legitimate blank or spanning cells.'})  # Keep this audit field with the rest of its source evidence.
        if unassigned:  # Check eligibility before continuing: flag unsupported table rows, spans, missing slots, and ambiguous placements.
            output.append({'table_ref': ref, 'row': '', 'status': 'review_unassigned_lines', 'candidate_cells': [], 'cell_texts': [],  # Flag unsupported table rows, spans, missing slots, and ambiguous placements.
                           'line_ids': unassigned, 'line_texts': [], 'unmatched_line_tokens': '', 'review_words': '', 'spanning_cells': '', 'missing_column_slots': [],  # Keep this audit field with the rest of its source evidence.
                           'note': 'These lines did not map uniquely to one candidate cell.'})  # Keep this audit field with the rest of its source evidence.
    return output  # Return the result to the next stage: flag unsupported table rows, spans, missing slots, and ambiguous placements.


def write_outputs(directory, source, source_sha256, page, links, parsed, orientation, words):  # Save reconciliation and table evidence without editing the extracted document.
    lines, cells = prepare_inputs(links, parsed, orientation)  # Determine lines, cells for this step: save reconciliation and table evidence without editing the extracted document.
    results, audit = reconcile(lines, words, cells)  # Determine results, audit for this step: save reconciliation and table evidence without editing the extracted document.
    table_rows = table_row_audit(lines, cells, results, audit)  # Determine table rows for this step: save reconciliation and table evidence without editing the extracted document.
    report = {'source': source, 'source_sha256': source_sha256, 'page': page,  # Determine report for this step: save reconciliation and table evidence without editing the extracted document.
              'coordinate_system': 'displayed cropped PDF page, top-left points',  # Keep this audit field with the rest of its source evidence.
              'method': 'exact ordered text agreement with geometry; table cells and all links are unverified',  # Keep this audit field with the rest of its source evidence.
              'results': results, 'lines': audit, 'table_rows': table_rows}  # Keep this audit field with the rest of its source evidence.
    json_path = directory/'reconciliation.json'; tmp = json_path.with_suffix('.json.tmp')  # Determine json path for this step: save reconciliation and table evidence without editing the extracted document.
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'); os.replace(tmp, json_path)  # Publish a finished artifact without exposing a partial write.
    csv_path = directory/'reconciled-words.csv'; tmp = csv_path.with_suffix('.csv.tmp')  # Determine csv path for this step: save reconciliation and table evidence without editing the extracted document.
    fields = ['source_pdf', 'source_sha256', 'page', 'text', 'status', 'x0', 'y0', 'x1', 'y1', 'line_id',  # Determine fields for this step: save reconciliation and table evidence without editing the extracted document.
              'line_text', 'line_confidence', 'item_ref', 'table_cell_candidate', 'fragments_json']  # Keep this audit field with the rest of its source evidence.
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:  # Save reconciliation and table evidence without editing the extracted document.
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()  # Label the evidence columns for audit and recovery.
        for r in results:  # Handle each r separately: save reconciliation and table evidence without editing the extracted document.
            writer.writerow(dict(source_pdf=source, source_sha256=source_sha256, page=page, text=r['text'], status=r['status'],  # Record the evidence while preserving source identifiers and measurements.
                                 **dict(zip(('x0','y0','x1','y1'), r['bbox'])), line_id=r['line_id'], line_text=r['line_text'],  # Keep this audit field with the rest of its source evidence.
                                 line_confidence=r['line_confidence'], item_ref=r['item_ref'], table_cell_candidate=r['table_cell_candidate'],  # Determine line confidence for this step: save reconciliation and table evidence without editing the extracted document.
                                 fragments_json=json.dumps(r['fragments'], ensure_ascii=False)))  # Determine fragments json for this step: save reconciliation and table evidence without editing the extracted document.
    os.replace(tmp, csv_path)  # Publish a finished artifact without exposing a partial write.
    table_path = directory/'table-row-review.csv'; tmp = table_path.with_suffix('.csv.tmp')  # Determine table path for this step: save reconciliation and table evidence without editing the extracted document.
    table_fields = ['table_ref', 'row', 'status', 'candidate_cells', 'cell_texts', 'line_ids', 'line_texts',  # Determine table fields for this step: save reconciliation and table evidence without editing the extracted document.
                    'unmatched_line_tokens', 'review_words', 'spanning_cells', 'missing_column_slots', 'note']  # Keep this audit field with the rest of its source evidence.
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:  # Save reconciliation and table evidence without editing the extracted document.
        writer = csv.DictWriter(f, fieldnames=table_fields); writer.writeheader()  # Label the evidence columns for audit and recovery.
        for table_row in table_rows:  # Handle each table row separately: save reconciliation and table evidence without editing the extracted document.
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in table_row.items()})  # Record the evidence while preserving source identifiers and measurements.
    os.replace(tmp, table_path)  # Publish a finished artifact without exposing a partial write.
    return json_path, csv_path, results, audit  # Return the result to the next stage: save reconciliation and table evidence without editing the extracted document.


def candidates(document):  # Flag possible sentence continuations across page edges without asserting a join.
    pages = document.get('pages', {})  # Determine pages for this step: flag possible sentence continuations across page edges without asserting a join.
    by_page = {}  # Determine by page for this step: flag possible sentence continuations across page edges without asserting a join.
    for item in document.get('texts', []):  # Handle each item separately: flag possible sentence continuations across page edges without asserting a join.
        text = item.get('text', '').strip()  # Determine text for this step: flag possible sentence continuations across page edges without asserting a join.
        if item.get('label') not in ('text', 'list_item') or len(re.sub(r'\W', '', text)) < 8:  # Check eligibility before continuing: flag possible sentence continuations across page edges without asserting a join.
            continue  # Move to the next item without counting unavailable evidence.
        for provenance in item.get('prov', []):  # Handle each provenance separately: flag possible sentence continuations across page edges without asserting a join.
            page = provenance['page_no']  # Determine page for this step: flag possible sentence continuations across page edges without asserting a join.
            height = pages[str(page)]['size']['height']  # Determine height for this step: flag possible sentence continuations across page edges without asserting a join.
            box = provenance_box(provenance['bbox'], height)  # Determine box for this step: flag possible sentence continuations across page edges without asserting a join.
            by_page.setdefault(page, []).append({'item': item, 'box': box, 'height': height})  # Flag possible sentence continuations across page edges without asserting a join.
    results = []  # Determine results for this step: flag possible sentence continuations across page edges without asserting a join.
    for page in sorted(by_page):  # Handle each page separately: flag possible sentence continuations across page edges without asserting a join.
        if page + 1 not in by_page:  # Check eligibility before continuing: flag possible sentence continuations across page edges without asserting a join.
            continue  # Move to the next item without counting unavailable evidence.
        last = max(by_page[page], key=lambda x: x['box'][3])  # Determine last for this step: flag possible sentence continuations across page edges without asserting a join.
        first = min(by_page[page+1], key=lambda x: x['box'][1])  # Determine first for this step: flag possible sentence continuations across page edges without asserting a join.
        end = last['item']['text'].strip()  # Determine end for this step: flag possible sentence continuations across page edges without asserting a join.
        start = first['item']['text'].strip()  # Determine start for this step: flag possible sentence continuations across page edges without asserting a join.
        starts_letter = next((c for c in start if c.isalpha()), '')  # Determine starts letter for this step: flag possible sentence continuations across page edges without asserting a join.
        near_bottom = last['box'][3]/last['height'] >= 0.65  # Determine near bottom for this step: flag possible sentence continuations across page edges without asserting a join.
        near_top = first['box'][1]/first['height'] <= 0.35  # Determine near top for this step: flag possible sentence continuations across page edges without asserting a join.
        has_terminal = bool(re.search(r'[.!?][\s\"\')\]]*$', end))  # Determine has terminal for this step: flag possible sentence continuations across page edges without asserting a join.
        uncertain_possessive = bool(re.search(r"['’]s\.[\s\"\')\]]*$", end, flags=re.IGNORECASE))  # Determine uncertain possessive for this step: flag possible sentence continuations across page edges without asserting a join.
        if near_bottom and near_top and (not has_terminal or uncertain_possessive) and starts_letter.islower():  # Check eligibility before continuing: flag possible sentence continuations across page edges without asserting a join.
            results.append({'from_page': page, 'to_page': page+1, 'from_ref': last['item']['self_ref'],  # Flag possible sentence continuations across page edges without asserting a join.
                            'to_ref': first['item']['self_ref'], 'ending': end[-160:], 'beginning': start[:160],  # Keep this audit field with the rest of its source evidence.
                            'from_bbox_top_left': last['box'], 'to_bbox_top_left': first['box'],  # Keep this audit field with the rest of its source evidence.
                            'status': 'proposed_review_required',  # Keep this audit field with the rest of its source evidence.
                            'reason': ('Possessive ending may have OCR-added punctuation; ' if uncertain_possessive else 'Unfinished ending; ') +  # Keep this audit field with the rest of its source evidence.
                                      'adjacent page-edge text and lowercase beginning; heuristic only.'})  # Keep this audit field with the rest of its source evidence.
    return results  # Return the result to the next stage: flag possible sentence continuations across page edges without asserting a join.


def assemble_page_documents(records):  # Compare neighboring pages using the original displayed coordinates.
    merged = {'pages': {}, 'texts': []}  # Determine merged for this step: compare neighboring pages using the original displayed coordinates.
    for record in records:  # Handle each record separately: compare neighboring pages using the original displayed coordinates.
        if record.get('status') not in ('complete', 'needs_review'):  # Check eligibility before continuing: compare neighboring pages using the original displayed coordinates.
            continue  # Move to the next item without counting unavailable evidence.
        page_no = record['page']  # Determine page no for this step: compare neighboring pages using the original displayed coordinates.
        report = json.loads(Path(record['parsed']).read_text(encoding='utf-8'))  # Determine report for this step: compare neighboring pages using the original displayed coordinates.
        page = report['parsed_pages'][0]  # Determine page for this step: compare neighboring pages using the original displayed coordinates.
        width, height = page['size']['width'], page['size']['height']  # Determine width, height for this step: compare neighboring pages using the original displayed coordinates.
        manifest = json.loads(Path(record['orientation']).read_text(encoding='utf-8')) if record.get('orientation') else None  # Determine manifest for this step: compare neighboring pages using the original displayed coordinates.
        source_width, source_height = manifest['source_display_size_points'] if manifest else (width, height)  # Determine source width, source height for this step: compare neighboring pages using the original displayed coordinates.
        merged['pages'][str(page_no)] = {'size': {'width': source_width, 'height': source_height}}  # Compare neighboring pages using the original displayed coordinates.
        for item in report['document'].get('texts', []):  # Handle each item separately: compare neighboring pages using the original displayed coordinates.
            prov = []  # Determine prov for this step: compare neighboring pages using the original displayed coordinates.
            for position in item.get('prov', []):  # Handle each position separately: compare neighboring pages using the original displayed coordinates.
                bbox = provenance_box(position['bbox'], height)  # Determine bbox for this step: compare neighboring pages using the original displayed coordinates.
                source = source_display_box(bbox, (width, height), manifest) if manifest else bbox  # Determine source for this step: compare neighboring pages using the original displayed coordinates.
                prov.append({'page_no': page_no, 'bbox': {'l': source[0], 't': source[1],  # Compare neighboring pages using the original displayed coordinates.
                                                            'r': source[2], 'b': source[3], 'coord_origin': 'TOPLEFT'}})  # Keep this audit field with the rest of its source evidence.
            merged['texts'].append({'self_ref': f"page-{page_no:05d}:{item['self_ref']}",  # Compare neighboring pages using the original displayed coordinates.
                                    'text': item.get('text', ''), 'label': item.get('label'), 'prov': prov})  # Keep this audit field with the rest of its source evidence.
    return merged  # Return the result to the next stage: compare neighboring pages using the original displayed coordinates.


def write_candidates(run_dir: Path, records: list, source: str, source_hash: str):  # Save possible cross-page continuations as a private review aid.
    result = candidates(assemble_page_documents(records))  # Determine result for this step: save possible cross-page continuations as a private review aid.
    fields = ['source_pdf', 'source_sha256', 'from_page', 'to_page', 'from_ref', 'to_ref', 'ending', 'beginning',  # Determine fields for this step: save possible cross-page continuations as a private review aid.
              'from_bbox_top_left', 'to_bbox_top_left', 'status', 'reason']  # Keep this audit field with the rest of its source evidence.
    path = run_dir/'continuation-candidates.csv'  # Determine path for this step: save possible cross-page continuations as a private review aid.
    tmp = path.with_suffix('.csv.tmp')  # Determine tmp for this step: save possible cross-page continuations as a private review aid.
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:  # Save possible cross-page continuations as a private review aid.
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()  # Label the evidence columns for audit and recovery.
        for row in result:  # Handle each row separately: save possible cross-page continuations as a private review aid.
            writer.writerow({'source_pdf': source, 'source_sha256': source_hash, **row})  # Record the evidence while preserving source identifiers and measurements.
    os.replace(tmp, path)  # Publish a finished artifact without exposing a partial write.
    return path, len(result)  # Return the result to the next stage: save possible cross-page continuations as a private review aid.


def supported_readable_edits(parsed, results):  # Join a visibly split word only when both OCR views support identical letters at its location.
    import copy  # Keep the raw model document unchanged for audits.
    public = dict(parsed,document=copy.deepcopy(parsed['document']))  # Modify only a rendering copy, never the evidence.
    targets = {item['self_ref']:item for item in public['document'].get('texts',[]) if item.get('label') not in ('formula','code')}  # Avoid changing mathematical and code tokens.
    tables = {item['self_ref']:item for item in public['document'].get('tables',[])}  # Restrict each table edit to an existing uniquely identified cell.
    edits = 0  # Count only changes actually made to the readable version.
    for result in results:  # Preserve the reason and original fragments for every decision.
        if result['status'] != 'review_proposed_merge':  # No substitutions, spelling guesses, number edits, or added words are authorized.
            continue  # Treat all other observations as evidence only.
        pieces = [w['word'] for w in result['fragments']]  # Use what the independent word pass actually observed.
        wanted = result['text']  # The line pass must independently support the exact spelling and case.
        if not (len(pieces)>1 and all(x.isalpha() for x in pieces) and ''.join(pieces)==wanted and result['line_text'].split().count(wanted)==1):  # Avoid punctuation, numeric values, case changes, and ambiguous repeated words.
            continue  # Leave the original text and mark the proposal for inspection.
        ref = result.get('item_ref','')  # Do not substitute across document item boundaries.
        target = targets.get(ref)  # Text edits must belong to the same extraction item.
        if ref in tables:  # Table row evidence may validate spelling but cannot validate a column move.
            match = re.fullmatch(r'r(\d+)c(\d+):cell(\d+)',result.get('table_cell_candidate',''))  # Require the exact cell established by geometry.
            cells = tables[ref]['data'].get('table_cells',[])  # Index only existing cells.
            if not match or int(match[3])>=len(cells):  # Ambiguous and absent assignments remain unchanged.
                continue  # Never move text or invent missing table content.
            target = cells[int(match[3])]  # Edit the existing cell, not its neighbors or headers.
            if (target.get('start_row_offset_idx'),target.get('start_col_offset_idx'),target.get('row_span',1),target.get('col_span',1)) != (int(match[1]),int(match[2]),1,1):  # Reject spanning or inconsistent cells.
                continue  # Preserve table structure as emitted by Docling.
        if target is None:  # Unlinked words cannot change an extract.
            continue  # Retain their raw metadata and proposals.
        pattern = r'(?<!\w)' + r'[ \t]+'.join(map(re.escape,pieces)) + r'(?!\w)'  # Change only horizontal spaces between the exact observed fragments.
        original = target.get('text','')  # Keep the comparison within this one item or cell.
        hits = list(re.finditer(pattern,original))  # Unique placement is required.
        if len(hits)!=1 or len(re.findall(r'(?<!\w)'+re.escape(wanted)+r'(?!\w)',original)):  # A repeated location or existing intact word is ambiguous.
            continue  # Do not make a speculative correction.
        target['text'] = original[:hits[0].start()]+wanted+original[hits[0].end():]  # Only whitespace changes; all original letters remain.
        result['readable_action'] = 'joined_whitespace'  # Each raw fragment will expose the applied decision in metadata.csv.
        result['readable_before'] = hits[0].group()  # Keep the exact original token spacing for audit.
        edits += 1  # Count this confirmed operation, not every contributing fragment.
    return public, edits  # Keep raw and readable products separately reproducible.



def render_page(report: dict) -> str:  # Render faithful Markdown, retaining unverified formulas, figures, and tables.
    """Render a copy of the extraction; do not apply unverified word corrections."""
    from docling_core.types.doc import ContentLayer, DoclingDocument  # Load this processing capability only in the worker that needs it.
    document = DoclingDocument.model_validate(report['document'])  # Determine document for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    replacements = {}  # Determine replacements for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    existing_text = json.dumps(report['document'], ensure_ascii=False)  # Determine existing text for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    for index, item in enumerate(document.texts):  # Handle each index, item separately: render faithful Markdown, retaining unverified formulas, figures, and tables.
        if str(item.label) == 'formula' and not item.text and item.orig:  # Check eligibility before continuing: render faithful Markdown, retaining unverified formulas, figures, and tables.
            marker = f'OCRFORMULAFALLBACK{index}END'  # Determine marker for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
            while marker in existing_text:  # Render faithful Markdown, retaining unverified formulas, figures, and tables.
                marker += 'X'  # Render faithful Markdown, retaining unverified formulas, figures, and tables.
            item.text = marker  # Render faithful Markdown, retaining unverified formulas, figures, and tables.
            runs = [len(x) for x in re.findall(r'`+', item.orig)]  # Determine runs for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
            fence = '`' * max(3, max(runs, default=0) + 1)  # Determine fence for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
            replacements[f'$${marker}$$'] = f'**Formula: raw extracted text, unverified**\n\n{fence}text\n{item.orig}\n{fence}'  # Render faithful Markdown, retaining unverified formulas, figures, and tables.
    markdown = document.export_to_markdown(  # Determine markdown for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE, ContentLayer.NOTES},  # Determine included content layers for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
        image_placeholder='*[Image: content not transcribed]*',  # Determine image placeholder for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    )  # Render faithful Markdown, retaining unverified formulas, figures, and tables.
    for marker, replacement in replacements.items():  # Handle each marker, replacement separately: render faithful Markdown, retaining unverified formulas, figures, and tables.
        markdown = markdown.replace(marker, replacement)  # Determine markdown for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    markdown = markdown.replace('<!-- formula-not-decoded -->', '*[Formula not decoded]*')  # Determine markdown for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    if not markdown.strip():  # Check eligibility before continuing: render faithful Markdown, retaining unverified formulas, figures, and tables.
        markdown = '*[No readable text was extracted from this page.]*'  # Determine markdown for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    if report['document'].get('tables'):  # Check eligibility before continuing: render faithful Markdown, retaining unverified formulas, figures, and tables.
        markdown = '> Table structure is extracted and requires review.\n\n' + markdown  # Determine markdown for this step: render faithful Markdown, retaining unverified formulas, figures, and tables.
    return markdown.strip() + '\n'  # Return the result to the next stage: render faithful Markdown, retaining unverified formulas, figures, and tables.


def atomic_text(path: Path, text: str) -> None:  # Publish completed text without exposing a half-written page.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')  # Determine temp for this step: publish completed text without exposing a half-written page.
    temp.write_text(text, encoding='utf-8')  # Publish completed text without exposing a half-written page.
    os.replace(temp, path)  # Publish a finished artifact without exposing a partial write.


def header(source: str, source_sha256: str) -> str:  # Label the readable report with its source and extraction limitations.
    return ('# Extracted document\n\n'  # Return the result to the next stage: label the readable report with its source and extraction limitations.
            f'Source: {Path(source).name}\n\n'  # Label the readable report with its source and extraction limitations.
            f'Source SHA-256: `{source_sha256}`\n\n'  # Label the readable report with its source and extraction limitations.
            '> Machine extraction may contain errors. Check the source for text, reading order, tables, and figures. '  # Keep this audit field with the rest of its source evidence.
            'Only independently supported alphabetic whitespace joins may be applied; raw observations remain in metadata.csv. '  # Keep this audit field with the rest of its source evidence.
            'check manifest.csv for document status and missing word evidence.\n\n')  # Keep this audit field with the rest of its source evidence.


def page_section(record: dict, page_text: str | None) -> str:  # Label each original page and explicitly mark missing or uncertain text.
    result = f"---\n\n## Page {record['page']}\n\n"  # Determine result for this step: label each original page and explicitly mark missing or uncertain text.
    if record.get('status') == 'needs_review':  # Check eligibility before continuing: label each original page and explicitly mark missing or uncertain text.
        result += '> Review: ' + ('; '.join(record.get('warnings',[])) or 'inspect source-page evidence') + '\n\n'  # Label each original page and explicitly mark missing or uncertain text.
    elif record.get('status') in ('failed', 'interrupted'):  # Use this alternative when earlier evidence does not justify the first route.
        result += f"> Page processing {record['status']}; any text shown may be incomplete.\n\n"  # Label each original page and explicitly mark missing or uncertain text.
    if page_text is not None:  # Check eligibility before continuing: label each original page and explicitly mark missing or uncertain text.
        result += page_text.strip() + '\n\n'  # Label each original page and explicitly mark missing or uncertain text.
    else:  # Use this alternative when earlier evidence does not justify the first route.
        detail = record.get('readable_error') or record.get('error') or record.get('status', 'unavailable')  # Determine detail for this step: label each original page and explicitly mark missing or uncertain text.
        result += f'*[Page text unavailable: {detail}]*\n\n'  # Label each original page and explicitly mark missing or uncertain text.
    return result  # Return the result to the next stage: label each original page and explicitly mark missing or uncertain text.


def write_page(report: dict, path: Path) -> None:  # Save readable content while the page evidence is available.
    atomic_text(path, render_page(report))  # Save readable content while the page evidence is available.


def saved_page_text(record: dict) -> str | None:  # Retrieve validated content from a previously processed page.
    if record.get('readable_status') == 'failed':  # Do not reuse unreadable page results.
        return None  # Return the result to the next stage: retrieve validated content from a previously processed page.
    if record.get('readable_page'):  # Check eligibility before continuing: retrieve validated content from a previously processed page.
        return Path(record['readable_page']).read_text(encoding='utf-8')  # Return the result to the next stage: retrieve validated content from a previously processed page.
    if record.get('status') in ('complete', 'needs_review') and record.get('parsed'):  # Check eligibility before continuing: retrieve validated content from a previously processed page.
        return render_page(json.loads(Path(record['parsed']).read_text(encoding='utf-8')))  # Return the result to the next stage: retrieve validated content from a previously processed page.
    return None  # Return the result to the next stage: retrieve validated content from a previously processed page.


def rebuild_document(path: Path, summary: dict) -> None:  # Assemble a complete readable extract in original page order.
    """Regenerate a document extract from completed page artifacts in source order."""
    by_page = {r['page']: r for r in summary['pages']}  # Determine by page for this step: assemble a complete readable extract in original page order.
    temp = path.with_suffix(path.suffix + '.tmp')  # Determine temp for this step: assemble a complete readable extract in original page order.
    with temp.open('w', encoding='utf-8', newline='\n') as f:  # Assemble a complete readable extract in original page order.
        f.write(header(summary['source'], summary['source_sha256']))  # Assemble a complete readable extract in original page order.
        for number in range(1, summary['page_count'] + 1):  # Handle each number separately: assemble a complete readable extract in original page order.
            record = by_page.get(number, {'page': number, 'status': 'pending'})  # Determine record for this step: assemble a complete readable extract in original page order.
            f.write(page_section(record, saved_page_text(record)))  # Assemble a complete readable extract in original page order.
    os.replace(temp, path)  # Publish a finished artifact without exposing a partial write.


CSV_FIELDS = ('page', 'status', 'route', 'started_at', 'finished_at', 'duration_seconds',  # Determine CSV FIELDS for this step: show the processing state of all source pages, including pending pages.
              'ocr_cells', 'unlinked_or_ambiguous', 'word_fragments', 'word_review', 'word_file',  # Keep this audit field with the rest of its source evidence.
              'reconciled_words', 'reconciliation_review', 'reconciliation_unmatched_line_tokens', 'reconciled_file', 'table_row_file',  # Keep this audit field with the rest of its source evidence.
              'readable_page', 'readable_extract', 'warnings', 'error', 'result_file', 'source_file')  # Carry uncertainty into the downstream review status.


def timestamp() -> str:  # Record local timestamps for audit and progress.
    return datetime.now().astimezone().isoformat(timespec='seconds')  # Return the result to the next stage: record local timestamps for audit and progress.


def classify_page(page) -> dict:  # Use the visible page, not the mere presence of searchable text, to choose the extraction route.
    layer = page.get_textpage()  # Inspect a bounded sample of glyphs to keep routing inexpensive.
    try:  # Always release the PDF text resources.
        text = layer.get_text_range()  # Count the native layer even if it is only a hidden OCR overlay.
        count = layer.count_chars()  # PDFium glyph indexes can differ from Python Unicode offsets.
        directions, visible, hidden, inspected = Counter(), 0, 0, 0  # Keep positive visibility and cardinal angle evidence separate.
        for idx in range(0, count, max(1, math.ceil(count / 192))):  # Sample across the full page, not just its header.
            code = raw.FPDFText_GetUnicode(layer, idx)  # Inspect only actual letters and numbers for direction.
            if not (0 < code <= 0x10ffff and chr(code).isalnum()):  # Punctuation and generated spaces have unreliable orientation.
                continue  # Use the next sampled glyph.
            obj = layer.get_textobj(idx)  # PDF drawing mode distinguishes a hidden search layer from visible text.
            if obj is None:  # Unsupported text structure does not establish visibility.
                continue  # Rely on raster evidence if no usable glyphs remain.
            inspected += 1  # Count how many real glyphs were inspected.
            if raw.FPDFTextObj_GetTextRenderMode(obj) in (raw.FPDF_TEXTRENDERMODE_INVISIBLE, raw.FPDF_TEXTRENDERMODE_CLIP):  # These glyphs do not paint visible text.
                hidden += 1  # A predominantly invisible layer should not replace the scanned image.
                continue  # Exclude hidden search text when deciding orientation.
            visible += 1  # Include visible glyphs even when their angle is ambiguous.
            angle = (math.degrees(raw.FPDFText_GetCharAngle(layer,idx)) - page.get_rotation()) % 360  # Translate the native baseline to the displayed scan frame.
            cardinal = int(round(angle / 90) * 90) % 360  # Only cardinal rotations can be applied losslessly here.
            if min((angle-cardinal)%360,(cardinal-angle)%360) <= 8:  # Discard skewed or decorative text as orientation evidence.
                directions[cardinal] += 1  # Vote for a clockwise correction in display space.
    finally:  # Do not retain PDF text objects across pages.
        layer.close()  # Release page text resources promptly.
    alnum = sum(ch.isalnum() for ch in text)  # Count searchable characters, not bytes.
    images = list(page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE]))  # Distinguish scanned page images from small figures or logos.
    page_box = page.get_bbox()  # Measure coverage against the visible crop rectangle.
    image_share = max((coverage(page_box,obj.get_bounds()) for obj in images),default=0.0)  # A full-page raster plus suspect text indicates an OCR overlay.
    dominant = image_share >= .65  # Small logos alone must not force the scan workflow.
    suspect = bool(text.strip()) and (text.count('\ufffd') / max(1,len(text)) > .02 or (inspected and hidden >= .60*inspected))  # Identify corrupt Unicode or invisible glyphs.
    scan = alnum < 20 or (bool(images) and suspect) or (dominant and visible < 6)  # Preserve healthy visible native text on ordinary mixed pages.
    route = 'scan_or_sparse' if scan else 'mixed' if images else 'native'  # Choose the layout/OCR path once per source page.
    vote, votes = directions.most_common(1)[0] if directions else (None,0)  # Dominant glyph direction is stronger than the PDF rotate tag alone.
    reliable = visible >= 6 and votes >= .80*visible  # Withhold rotation when text directions are mixed.
    return dict(route=route,native_alphanumeric_chars=alnum,image_objects=len(images),image_coverage=round(image_share,3),suspect_text_layer=bool(suspect),visible_glyph_sample=visible,hidden_glyph_sample=hidden,native_clockwise_correction=vote if reliable else None)  # Expose routing evidence for diagnosis.


def source_stamp(path):  # Cheaply detect replacement or modification while a long source is open.
    st = path.stat()  # Record identity and change metadata without rereading the entire PDF.
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)  # Final whole-file hashing still checks the published version.



def atomic_json(path: Path, obj: dict) -> None:  # Publish complete checkpoints that can be trusted after interruption.
    tmp = path.with_suffix(path.suffix + '.tmp')  # Determine tmp for this step: publish complete checkpoints that can be trusted after interruption.
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')  # Publish complete checkpoints that can be trusted after interruption.
    os.replace(tmp, path)  # Publish a finished artifact without exposing a partial write.


def write_progress_csv(path: Path, summary: dict) -> None:  # Show the processing state of all source pages, including pending pages.
    """Write a complete page snapshot; run.json remains the resume source of truth."""
    by_page = {record['page']: record for record in summary['pages']}  # Determine by page for this step: show the processing state of all source pages, including pending pages.
    tmp = path.with_suffix(path.suffix + '.tmp')  # Determine tmp for this step: show the processing state of all source pages, including pending pages.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        with tmp.open('w', encoding='utf-8-sig', newline='') as stream:  # Show the processing state of all source pages, including pending pages.
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)  # Determine writer for this step: show the processing state of all source pages, including pending pages.
            writer.writeheader()  # Label the evidence columns for audit and recovery.
            for page in range(1, summary['page_count'] + 1):  # Handle each page separately: show the processing state of all source pages, including pending pages.
                record = by_page.get(page, {})  # Determine record for this step: show the processing state of all source pages, including pending pages.
                writer.writerow({'page': page, 'status': record.get('status', 'pending'),  # Record the evidence while preserving source identifiers and measurements.
                                 'route': record.get('route', ''), 'started_at': record.get('started_at', ''),  # Keep this audit field with the rest of its source evidence.
                                 'finished_at': record.get('finished_at', ''), 'duration_seconds': record.get('duration_seconds', ''),  # Keep this audit field with the rest of its source evidence.
                                 'ocr_cells': record.get('ocr_cells', ''), 'unlinked_or_ambiguous': record.get('unlinked_or_ambiguous', ''),  # Keep this audit field with the rest of its source evidence.
                                 'word_fragments': record.get('word_fragments', ''), 'word_review': record.get('word_review', ''),  # Keep this audit field with the rest of its source evidence.
                                 'word_file': record.get('word_file', ''),  # Keep this audit field with the rest of its source evidence.
                                 'reconciled_words': record.get('reconciled_words', ''),  # Keep this audit field with the rest of its source evidence.
                                 'reconciliation_review': record.get('reconciliation_review', ''),  # Keep this audit field with the rest of its source evidence.
                                 'reconciliation_unmatched_line_tokens': record.get('reconciliation_unmatched_line_tokens', ''),  # Keep this audit field with the rest of its source evidence.
                                 'reconciled_file': record.get('reconciled_file', ''), 'table_row_file': record.get('table_row_file', ''),  # Keep this audit field with the rest of its source evidence.
                                 'readable_page': record.get('readable_page', ''), 'readable_extract': summary.get('readable_extract', ''),  # Keep this audit field with the rest of its source evidence.
                                 'warnings': ' | '.join(record.get('warnings', [])), 'error': record.get('error', ''),  # Carry uncertainty into the downstream review status.
                                 'result_file': record.get('links', ''), 'source_file': summary['source']})  # Keep this audit field with the rest of its source evidence.
        os.replace(tmp, path)  # Publish a finished artifact without exposing a partial write.
    finally:  # Release resources even if this processing step did not finish.
        if tmp.exists():  # Check eligibility before continuing: show the processing state of all source pages, including pending pages.
            tmp.unlink()  # Show the processing state of all source pages, including pending pages.


def save_progress(run_dir: Path, summary: dict) -> None:  # Checkpoint page progress even if a spreadsheet viewer locks the private CSV.
    atomic_json(run_dir / 'run.json', summary)  # Checkpoint page progress even if a spreadsheet viewer locks the private CSV.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        write_progress_csv(run_dir / 'progress.csv', summary)  # Checkpoint page progress even if a spreadsheet viewer locks the private CSV.
    except OSError as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
        print(f'Warning: could not update progress.csv: {exc}. Progress remains in run.json.', file=sys.stderr, flush=True)  # Expose progress or a recoverable failure to the operator.


def file_sha256(path: Path) -> str:  # Identify exact input or artifact bytes for traceability.
    with path.open('rb') as stream:  # Identify exact input or artifact bytes for traceability.
        return hashlib.file_digest(stream, 'sha256').hexdigest()  # Return the result to the next stage: identify exact input or artifact bytes for traceability.


def valid_readable_extract(summary: dict) -> bool:  # Verify that the finalized readable extract still matches its saved digest.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        return (summary.get('readable_status') == 'generated' and  # Return the result to the next stage: verify that the finalized readable extract still matches its saved digest.
                file_sha256(Path(summary['readable_extract'])) == summary['readable_extract_sha256'])  # Verify the exact bytes associated with the evidence.
    except (OSError, KeyError, TypeError):  # Keep the cause of this failure visible and preserve recoverable progress.
        return False  # Return the result to the next stage: verify that the finalized readable extract still matches its saved digest.


def execution_config(engine: str, artifacts: Path, min_confidence: float, word_boxes: bool = True) -> dict:  # Fingerprint code, packages, OCR installation, settings, and model artifacts.
    from docling.datamodel.accelerator_options import AcceleratorOptions  # Load this processing capability only in the worker that needs it.
    files = sorted(p for p in artifacts.rglob('*') if p.is_file())  # Determine files for this step: fingerprint code, packages, OCR installation, settings, and model artifacts.
    model_digest = hashlib.sha256()  # Determine model digest for this step: fingerprint code, packages, OCR installation, settings, and model artifacts.
    for path in files:  # Handle each path separately: fingerprint code, packages, OCR installation, settings, and model artifacts.
        model_digest.update(path.relative_to(artifacts).as_posix().encode('utf-8'))  # Fingerprint code, packages, OCR installation, settings, and model artifacts.
        model_digest.update(bytes.fromhex(file_sha256(path)))  # Verify the exact bytes associated with the evidence.
    packages = ('docling', 'docling-core', 'rapidocr', 'onnxruntime', 'paddlepaddle', 'pypdfium2', 'Pillow')  # Determine packages for this step: fingerprint code, packages, OCR installation, settings, and model artifacts.
    tess_exe = Path(TESSERACT).resolve()  # Determine tess exe for this step: fingerprint code, packages, OCR installation, settings, and model artifacts.
    tessdata = Path(os.environ.get('TESSDATA_PREFIX') or (tess_exe.parent / 'tessdata'))  # Determine tessdata for this step: fingerprint code, packages, OCR installation, settings, and model artifacts.
    return {'engine': engine, 'artifacts': str(artifacts.resolve()), 'model_files_sha256': model_digest.hexdigest(),  # Return the result to the next stage: fingerprint code, packages, OCR installation, settings, and model artifacts.
            'model_file_count': len(files), 'min_orientation_confidence': min_confidence, 'word_boxes': word_boxes,  # Keep this audit field with the rest of its source evidence.
            'accelerator_num_threads': AcceleratorOptions().num_threads,  # Keep this audit field with the rest of its source evidence.
            'scripts': {'extract_pdfs.py': file_sha256(Path(__file__))},  # Verify the exact bytes associated with the evidence.
            'packages': {name: importlib.metadata.version(name) for name in packages},  # Keep this audit field with the rest of its source evidence.
            'python': sys.version, 'tesseract_sha256': file_sha256(tess_exe),  # Verify the exact bytes associated with the evidence.
            'tesseract_languages': {lang: file_sha256(tessdata / (lang + '.traineddata')) if (tessdata / (lang + '.traineddata')).is_file() else 'system-managed' for lang in ('eng', 'osd')}}  # Verify the exact bytes associated with the evidence.


def valid_completed_page(record: dict, source_hash: str, require_reconciliation: bool = False, require_readable: bool = False) -> bool:  # Reuse a finished page only when all required evidence passes its checks.
    if record.get('status') not in ('complete', 'needs_review'):  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
        return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        links, parsed = Path(record['links']), Path(record['parsed'])  # Determine links, parsed for this step: reuse a finished page only when all required evidence passes its checks.
        if file_sha256(links) != record['links_sha256'] or file_sha256(parsed) != record['parsed_sha256']:  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('word_status') == 'failed':  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('word_file') and file_sha256(Path(record['word_file'])) != record['word_file_sha256']:  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if require_reconciliation and record.get('word_file') and record.get('reconciliation_status') != 'generated_unverified':  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('reconciliation_status') == 'failed':  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('reconciliation_status') == 'generated_unverified':  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            for field in ('reconciliation_json', 'reconciled_file', 'table_row_file'):  # Handle each field separately: reuse a finished page only when all required evidence passes its checks.
                if file_sha256(Path(record[field])) != record[field + '_sha256']:  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
                    return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if require_readable and record.get('readable_status') != 'generated':  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('readable_status') == 'failed':  # Do not reuse unreadable page results.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('readable_status') == 'generated':  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            if file_sha256(Path(record['readable_page'])) != record['readable_page_sha256']:  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
                return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        if record.get('orientation') and file_sha256(Path(record['orientation'])) != record['orientation_sha256']:  # Check eligibility before continuing: reuse a finished page only when all required evidence passes its checks.
            return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
        data = json.loads(links.read_text(encoding='utf-8'))  # Determine data for this step: reuse a finished page only when all required evidence passes its checks.
        return data['source_sha256'] == source_hash and data['source_page_no'] == record['page']  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.
    except (OSError, KeyError, ValueError, TypeError):  # Keep the cause of this failure visible and preserve recoverable progress.
        return False  # Return the result to the next stage: reuse a finished page only when all required evidence passes its checks.


def process_document(argv=None, runtime=None):  # Extract and checkpoint a single PDF without losing the original page identity.
    run_started = time.perf_counter()  # Determine run started for this step: extract and checkpoint a single PDF without losing the original page identity.
    parser = argparse.ArgumentParser(description=__doc__)  # Determine parser for this step: extract and checkpoint a single PDF without losing the original page identity.
    parser.add_argument('pdf', type=Path)  # Specify the pdf setting for this document-processing step.
    parser.add_argument('--engine', choices=['tesseract', 'rapid-onnx', 'rapid-paddle'], default='rapid-onnx')  # Specify the engine setting for this document-processing step.
    parser.add_argument('--output-root', type=Path, default=Path('outputs/runs'))  # Specify the output root setting for this document-processing step.
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'models')  # Specify the artifacts setting for this document-processing step.
    parser.add_argument('--min-orientation-confidence', type=float, default=3.0)  # Specify the min orientation confidence setting for this document-processing step.
    parser.add_argument('--resume', type=Path, help='Resume an existing run directory with matching input and settings')  # Specify the resume setting for this document-processing step.
    parser.add_argument('--no-word-boxes', action='store_true', help='Skip the additional scanned-page OCR word pass')  # Specify the no word boxes setting for this document-processing step.
    args = parser.parse_args(argv)  # Determine args for this step: extract and checkpoint a single PDF without losing the original page identity.
    if not args.pdf.is_file() or not args.artifacts.is_dir() or args.min_orientation_confidence < 0:  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
        parser.error('Existing PDF and model directory and a nonnegative orientation threshold are required')  # Extract and checkpoint a single PDF without losing the original page identity.
    source = args.pdf.resolve()  # Determine source for this step: extract and checkpoint a single PDF without losing the original page identity.
    stamp = source_stamp(source)  # Detect changes between page checkpoints.
    source_hash = file_sha256(source)  # Verify the exact bytes associated with the evidence.
    if (runtime or {}).get('expected_digest',source_hash) != source_hash:  # Refuse a source that changed after discovery before publishing any page.
        raise ValueError('Source changed after discovery; rerun to process its new version.')  # Leave this version failed and independently traceable.
    document = pdfium.PdfDocument(str(source))  # Determine document for this step: extract and checkpoint a single PDF without losing the original page identity.
    config = (runtime or {}).get("fingerprint") or execution_config(args.engine, args.artifacts.resolve(), args.min_orientation_confidence, not args.no_word_boxes)  # Expose progress or a recoverable failure to the operator.
    if runtime is None:  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
        runtime = {}  # Determine runtime for this step: extract and checkpoint a single PDF without losing the original page identity.
    if runtime.get('config') is not None and runtime['config'] != config:  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
        raise RuntimeError('Processing configuration changed while models were loaded; start a new worker')  # Surface the inconsistency rather than silently publishing bad evidence.
    runtime['config'] = config  # Extract and checkpoint a single PDF without losing the original page identity.
    if args.resume:  # Reuse prior page evidence only for the same source and settings.
        run_dir = args.resume.resolve()  # Determine run dir for this step: extract and checkpoint a single PDF without losing the original page identity.
        summary = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))  # Determine summary for this step: extract and checkpoint a single PDF without losing the original page identity.
        if summary.get('source') != str(source) or summary.get('source_sha256') != source_hash or summary.get('config') != config or summary.get('page_count') != len(document):  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
            parser.error('Cannot resume: source, page count, code, models, versions, or settings changed')  # Extract and checkpoint a single PDF without losing the original page identity.
    else:  # Use this alternative when earlier evidence does not justify the first route.
        run_dir = args.output_root.resolve() / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])  # Determine run dir for this step: extract and checkpoint a single PDF without losing the original page identity.
        run_dir.mkdir(parents=True, exist_ok=False)  # Reserve a private location for this processing attempt.
        summary = {'source': str(source), 'source_sha256': source_hash, 'engine': args.engine, 'config': config,  # Determine summary for this step: extract and checkpoint a single PDF without losing the original page identity.
               'page_count': len(document), 'pages': [], 'status': 'running',  # Keep this audit field with the rest of its source evidence.
               'limitations': ['Experimental sentence continuation candidates require review; no table or citation linking.',  # Keep this audit field with the rest of its source evidence.
                               'Converters and the separate word OCR model are reused by this worker.',  # Keep this audit field with the rest of its source evidence.
                               'Native glyph direction and visual orientation are heuristics; ambiguous directions require inspection.',  # Keep this audit field with the rest of its source evidence.
                               'Displayed-page coordinates; links are geometric estimates.',  # Keep this audit field with the rest of its source evidence.
                               'Word boxes and reconciliation proposals are unverified; preserve and review original OCR evidence.',  # Keep this audit field with the rest of its source evidence.
                               'Table row and cell assignments are unverified and may be merged incorrectly.',  # Keep this audit field with the rest of its source evidence.
                               'Single process only; OCR backend caches outside model directory are not fingerprinted.']}  # Expose progress or a recoverable failure to the operator.
    summary['status'] = 'running'  # Extract and checkpoint a single PDF without losing the original page identity.
    readable_path = run_dir / 'extracted.md'  # Determine readable path for this step: extract and checkpoint a single PDF without losing the original page identity.
    summary['readable_extract'] = str(readable_path)  # Extract and checkpoint a single PDF without losing the original page identity.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        atomic_text(readable_path, header(str(source), source_hash))  # Extract and checkpoint a single PDF without losing the original page identity.
    except OSError as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
        print(f'Warning: could not initialize readable extract: {exc}', file=sys.stderr, flush=True)  # Expose progress or a recoverable failure to the operator.
    save_progress(run_dir, summary)  # Extract and checkpoint a single PDF without losing the original page identity.
    print(f'Progress: {run_dir / "progress.csv"}', flush=True)  # Expose progress or a recoverable failure to the operator.

    def append_readable(record):  # Make finished page text available while a longer PDF is still processing.
        try:  # Handle a failure in this step without silently accepting incomplete evidence.
            text = saved_page_text(record)  # Determine text for this step: make finished page text available while a longer PDF is still processing.
            with readable_path.open('a', encoding='utf-8', newline='\n') as stream:  # Make finished page text available while a longer PDF is still processing.
                stream.write(page_section(record, text))  # Make finished page text available while a longer PDF is still processing.
                stream.flush()  # Make finished page text available while a longer PDF is still processing.
        except (OSError, ValueError, KeyError, TypeError) as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
            print(f'Warning: could not update readable extract for page {record["page"]}: {exc}', file=sys.stderr, flush=True)  # Expose progress or a recoverable failure to the operator.

    converters = runtime.setdefault('converters', {})  # Determine converters for this step: extract and checkpoint a single PDF without losing the original page identity.
    for index in range(len(document)):  # Handle each index separately: extract and checkpoint a single PDF without losing the original page identity.
        page_number = index + 1  # Determine page number for this step: extract and checkpoint a single PDF without losing the original page identity.
        previous = next((p for p in summary['pages'] if p['page'] == page_number), None)  # Determine previous for this step: extract and checkpoint a single PDF without losing the original page identity.
        if previous and valid_completed_page(previous, source_hash, require_reconciliation=True, require_readable=True):  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
            if runtime.get('progress') is not None:  # Resume with verified pages visible in the current publication.
                runtime['progress'].put((runtime['current_document'],previous,len(document)))  # The coordinator deduplicates notifications by page.
            append_readable(previous)  # Extract and checkpoint a single PDF without losing the original page identity.
            print(f'Page {page_number}/{len(document)}: reused validated result', flush=True)  # Expose progress or a recoverable failure to the operator.
            continue  # Move to the next item without counting unavailable evidence.
        page_dir = run_dir / f'page-{page_number:05d}' / ('attempt-' + uuid.uuid4().hex[:8])  # Determine page dir for this step: extract and checkpoint a single PDF without losing the original page identity.
        page_dir.mkdir(parents=True)  # Reserve a private location for this processing attempt.
        record = {'page': page_number, 'status': 'processing', 'warnings': [], 'started_at': timestamp()}  # Carry uncertainty into the downstream review status.
        start = time.perf_counter()  # Determine start for this step: extract and checkpoint a single PDF without losing the original page identity.
        summary['pages'] = [p for p in summary['pages'] if p['page'] != page_number] + [record]  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['pages'].sort(key=lambda p: p['page'])  # Extract and checkpoint a single PDF without losing the original page identity.
        save_progress(run_dir, summary)  # Extract and checkpoint a single PDF without losing the original page identity.
        try:  # Handle a failure in this step without silently accepting incomplete evidence.
            record.update(classify_page(document[index]))  # Extract and checkpoint a single PDF without losing the original page identity.
            scan = record['route'] == 'scan_or_sparse'  # Run the independent OCR passes for image pages and untrustworthy text overlays.
            visual_check = scan or record['native_clockwise_correction'] is None or record['image_coverage'] >= .65  # Assess picture-dominated mixed pages using their visible content.
            correction = record['native_clockwise_correction'] or 0  # Healthy native text can be oriented using its displayed glyph baselines.
            manifest, orient_args = None, []  # Maintain an inverse map only where the working page needs one.
            if visual_check:  # Use visual orientation before layout when native glyphs cannot settle it.
                orient_page([str(source),'--page',str(page_number),'--output-dir',str(page_dir),'--min-confidence',str(args.min_orientation_confidence)],source_hash)  # Preserve original source identity without rehashing it per page.
                manifest_path = page_dir/'orientation.json'  # The visual check records both its decision and uncertainty.
                manifest = read_json(manifest_path)  # Inspect evidence before deciding how to prepare a working page.
                if manifest['needs_review']:  # A weak visual vote does not overturn reliable visible glyphs.
                    if scan or record['native_clockwise_correction'] is None or record['image_coverage'] >= .65:  # Flag the page if its dominant orientation is unresolved.
                        record['warnings'].append('Orientation uncertain; inspect the displayed page.')  # Surface a concrete review reason.
                elif scan or record['native_clockwise_correction'] in (None,manifest['applied_clockwise_rotation']):  # Use supported visual orientation without overriding conflicting native text.
                    correction = manifest['applied_clockwise_rotation']  # This is clockwise relative to the source as displayed.
                else:  # A visible diagram and the native text can face different directions.
                    record['warnings'].append('Visual and native orientation disagree; retained native direction.')  # Let the operator inspect a genuine conflict.
                record.update(orientation=str(manifest_path),orientation_sha256=file_sha256(manifest_path))  # Retain the exact orientation evidence.
            if scan:  # Render visible content rather than trusting sparse or invisible search text.
                pdf_path = page_dir/'corrected.pdf'  # The corrected image preserves the page appearance.
                orient_args = ['mapped']  # Map raster OCR observations back to the displayed source.
                ocr_args = ['--force-ocr']  # Apply the two-pass scan pipeline.
            else:  # Keep vector content and selectable native text instead of rasterizing good native data.
                pdf_path = page_dir/'source-page.pdf'  # Isolate this page for Docling layout.
                page_doc = pdfium.PdfDocument.new()  # Preserve the full native page structure.
                page_doc.import_pages(document,pages=[index])  # Copy the original page unchanged before setting its display rotation.
                native_page = page_doc[0]  # Retain text and pictures in a mixed document.
                native_page.set_rotation((native_page.get_rotation()+correction)%360)  # Correct the display angle losslessly before layout.
                native_page.close()  # Finish updating the cloned page.
                page_doc.save(str(pdf_path))  # Never modify the source PDF.
                page_doc.close()  # Release the working document.
                ocr_args = []  # Let Docling read native text and OCR its embedded image regions.
                if correction:  # Reverse the applied displayed-page rotation when publishing coordinates.
                    w,h = document[index].get_size()  # The source frame already includes its own rotation and crop.
                    manifest = dict(source_pdf=str(source),source_sha256=source_hash,source_page=page_number,source_display_size_points=[w,h],source_rendered_size_pixels=[w,h],corrected_size_pixels=[h,w] if correction in (90,270) else [w,h],corrected_pixel_to_source_render_pixel=inverse_matrix(correction,w,h),corrected_pdf=str(pdf_path),corrected_sha256=file_sha256(pdf_path))  # Use an exact scale-one transform for this vector page.
                    orient_args = ['mapped']  # Use the same source-coordinate contract for rotated native and raster pages.
            record['orientation_correction_degrees'] = correction if not scan else manifest['applied_clockwise_rotation']  # Record what the worker actually applied.
            parsed = page_dir / 'parsed.json'  # Determine parsed for this step: extract and checkpoint a single PDF without losing the original page identity.
            linked = page_dir / 'links.json'  # Determine linked for this step: extract and checkpoint a single PDF without losing the original page identity.
            force_ocr = bool(ocr_args)  # Determine force ocr for this step: extract and checkpoint a single PDF without losing the original page identity.
            if force_ocr not in converters:  # Load each route-specific model only once per worker.
                converters[force_ocr] = make_converter(args.engine, args.artifacts, force_ocr)  # Extract and checkpoint a single PDF without losing the original page identity.
            try:  # Handle a failure in this step without silently accepting incomplete evidence.
                parsed_data = convert_report(converters[force_ocr], pdf_path, args.engine, force_ocr)  # Determine parsed data for this step: extract and checkpoint a single PDF without losing the original page identity.
            except Exception:  # Keep the cause of this failure visible and preserve recoverable progress.
                converters.pop(force_ocr, None)  # Extract and checkpoint a single PDF without losing the original page identity.
                raise  # Surface the inconsistency rather than silently publishing bad evidence.
            atomic_json(parsed, parsed_data)  # Extract and checkpoint a single PDF without losing the original page identity.
            try:  # Handle a failure in this step without silently accepting incomplete evidence.
                readable_page = page_dir / 'extracted.md'  # Determine readable page for this step: extract and checkpoint a single PDF without losing the original page identity.
                write_page(parsed_data, readable_page)  # Extract and checkpoint a single PDF without losing the original page identity.
                record.update({'readable_status': 'generated', 'readable_page': str(readable_page),  # Extract and checkpoint a single PDF without losing the original page identity.
                               'readable_page_sha256': file_sha256(readable_page)})  # Verify the exact bytes associated with the evidence.
            except Exception as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
                record['readable_status'] = 'failed'  # Extract and checkpoint a single PDF without losing the original page identity.
                record['readable_error'] = str(exc)[-500:]  # Extract and checkpoint a single PDF without losing the original page identity.
                record['warnings'].append('Readable extract failed: ' + record['readable_error'])  # Carry uncertainty into the downstream review status.
            data = link(parsed_data, orientation=manifest if orient_args else None, verified_source=source_hash)  # Determine data for this step: extract and checkpoint a single PDF without losing the original page identity.
            if parsed_data['status'] != 'ConversionStatus.SUCCESS':  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
                record['warnings'].append(f"Docling conversion status: {parsed_data['status']}")  # Carry uncertainty into the downstream review status.
            data['source'] = str(source)  # Extract and checkpoint a single PDF without losing the original page identity.
            data['source_sha256'] = source_hash  # Extract and checkpoint a single PDF without losing the original page identity.
            data['source_page_no'] = page_number  # Extract and checkpoint a single PDF without losing the original page identity.
            for cell in [c for item in data['items'] for c in item['cells']] + data['unlinked_or_ambiguous']:  # Handle each cell separately: extract and checkpoint a single PDF without losing the original page identity.
                cell['source_page_no'] = page_number  # Extract and checkpoint a single PDF without losing the original page identity.
                if not orient_args:  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
                    cell['source_display_bbox_top_left'] = cell['bbox_top_left']  # Extract and checkpoint a single PDF without losing the original page identity.
            data['source_coordinate_system'] = 'PDF points, top-left of displayed cropped page'  # Extract and checkpoint a single PDF without losing the original page identity.
            atomic_json(linked, data)  # Extract and checkpoint a single PDF without losing the original page identity.
            if record['route'] == 'scan_or_sparse' and not args.no_word_boxes and args.engine != 'tesseract':  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
                word_file = page_dir / 'word-candidates.csv'  # Determine word file for this step: extract and checkpoint a single PDF without losing the original page identity.
                try:  # Handle a failure in this step without silently accepting incomplete evidence.
                    backend = 'onnxruntime' if args.engine == 'rapid-onnx' else 'paddle'  # Determine backend for this step: extract and checkpoint a single PDF without losing the original page identity.
                    if runtime.get('word_model') is None:  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
                        runtime['word_model'] = make_word_model(backend, args.artifacts)  # Extract and checkpoint a single PDF without losing the original page identity.
                    words, _ = extract_words(pdf_path, parsed_data, backend, runtime['word_model'], manifest=manifest, verified_source=source_hash)  # Determine words,   for this step: extract and checkpoint a single PDF without losing the original page identity.
                    write_word_csv(word_file, words)  # Extract and checkpoint a single PDF without losing the original page identity.
                    record.update({'word_file': str(word_file), 'word_file_sha256': file_sha256(word_file),  # Verify the exact bytes associated with the evidence.
                                   'word_fragments': len(words), 'word_review': sum(w['link_status'] != 'candidate' for w in words),  # Keep this audit field with the rest of its source evidence.
                                   'word_status': 'generated_unverified'})  # Keep this audit field with the rest of its source evidence.
                    try:  # Handle a failure in this step without silently accepting incomplete evidence.
                        recon_started = time.perf_counter()  # Determine recon started for this step: extract and checkpoint a single PDF without losing the original page identity.
                        recon_json, recon_csv, recon_rows, recon_lines = write_outputs(  # Extract and checkpoint a single PDF without losing the original page identity.
                            page_dir, str(source), source_hash, page_number, data, parsed_data, manifest, words)  # Extract and checkpoint a single PDF without losing the original page identity.
                        public, applied = supported_readable_edits(parsed_data,recon_rows)  # Apply only narrowly supported text-spacing decisions.
                        if applied:  # Keep raw Docling JSON unchanged, including its original table structure.
                            write_page(public,readable_page)  # Render the corrected item or cell into the business extract.
                            record.update(readable_status='generated',readable_page=str(readable_page),readable_page_sha256=file_sha256(readable_page),readable_joins=applied)  # Bind this exact readable output to its supporting decisions.
                        audit = read_json(recon_json)  # Attach the rendering decision to the existing raw fragment evidence.
                        audit['results'] = recon_rows  # Preserve all original observations, accepted edits, and remaining proposals.
                        if applied:  # Avoid rewriting unchanged evidence.
                            atomic_json(recon_json,audit)  # Persist applied decisions before publishing their word rows.
                        unresolved = sum(r['status'].startswith('review_') and not r.get('readable_action') for r in recon_rows)  # Count disagreements and unapplied proposals for targeted inspection.
                        if unresolved or any(r['unmatched_line_tokens'] for r in recon_lines):  # Independent readings did not fully agree.
                            record['warnings'].append(f'OCR line/word disagreement or unplaced evidence ({unresolved} flagged word groups).')  # Provide a concrete review reason.
                        if any(r['status'].startswith('review') for r in audit['table_rows']):  # Missing or uncertain row/column evidence must remain visible.
                            record['warnings'].append('Table row, column, or span assignments need inspection.')  # Avoid treating correct numbers as proof of correct placement.
                        table_csv = page_dir / 'table-row-review.csv'  # Determine table csv for this step: extract and checkpoint a single PDF without losing the original page identity.
                        record.update({'reconciliation_status': 'generated_unverified', 'reconciliation_json': str(recon_json),  # Extract and checkpoint a single PDF without losing the original page identity.
                                       'reconciliation_json_sha256': file_sha256(recon_json), 'reconciled_file': str(recon_csv),  # Verify the exact bytes associated with the evidence.
                                       'reconciled_file_sha256': file_sha256(recon_csv), 'table_row_file': str(table_csv),  # Verify the exact bytes associated with the evidence.
                                       'table_row_file_sha256': file_sha256(table_csv), 'reconciled_words': len(recon_rows),  # Verify the exact bytes associated with the evidence.
                                       'reconciliation_review': sum(r['status'].startswith('review_') for r in recon_rows),  # Keep this audit field with the rest of its source evidence.
                                       'reconciliation_unmatched_line_tokens': sum(len(r['unmatched_line_tokens']) for r in recon_lines),  # Keep this audit field with the rest of its source evidence.
                                       'reconciliation_seconds': round(time.perf_counter() - recon_started, 3)})  # Keep this audit field with the rest of its source evidence.
                    except Exception as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
                        record['reconciliation_status'] = 'failed'  # Extract and checkpoint a single PDF without losing the original page identity.
                        record['warnings'].append('Reconciliation failed: ' + str(exc)[-500:])  # Carry uncertainty into the downstream review status.
                except Exception as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
                    runtime['word_model'] = None  # Extract and checkpoint a single PDF without losing the original page identity.
                    record['word_status'] = 'failed'  # Extract and checkpoint a single PDF without losing the original page identity.
                    record['warnings'].append('Word-level pass failed: ' + str(exc)[-500:])  # Carry uncertainty into the downstream review status.
            if source_stamp(source) != stamp:  # Check eligibility before continuing: extract and checkpoint a single PDF without losing the original page identity.
                raise RuntimeError('Source PDF changed during processing')  # Surface the inconsistency rather than silently publishing bad evidence.
            if data['unlinked_or_ambiguous']:  # Unplaced source elements weaken traceability.
                record['warnings'].append(f"{len(data['unlinked_or_ambiguous'])} extracted elements have unresolved locations or item assignments.")  # Identify the specific missing provenance.
            if parsed_data['document'].get('tables') and not record.get('table_row_file'):  # Native tables may have no independent row evidence.
                record['warnings'].append('Table structure lacks independent row validation.')  # Make missing table evidence explicit.
            if record.get('readable_status')=='generated' and not parsed_data['document'].get('texts') and not parsed_data['document'].get('tables'):  # Empty output can mean blank paper or missing content.
                record['warnings'].append('No text or table elements detected; verify blank or image-only page.')  # Never silently certify an empty extraction.
            record.update({'status': 'needs_review' if record['warnings'] else 'complete',  # Carry uncertainty into the downstream review status.
                           'links': str(linked), 'links_sha256': file_sha256(linked), 'parsed': str(parsed),  # Verify the exact bytes associated with the evidence.
                           'parsed_sha256': file_sha256(parsed), 'ocr_cells': data['ocr_cell_count'],  # Verify the exact bytes associated with the evidence.
                           'unlinked_or_ambiguous': len(data['unlinked_or_ambiguous'])})  # Keep this audit field with the rest of its source evidence.
        except KeyboardInterrupt:  # Keep the cause of this failure visible and preserve recoverable progress.
            record.update({'status': 'interrupted', 'finished_at': timestamp(), 'duration_seconds': round(time.perf_counter() - start, 2)})  # Extract and checkpoint a single PDF without losing the original page identity.
            summary['status'] = 'interrupted'  # Extract and checkpoint a single PDF without losing the original page identity.
            save_progress(run_dir, summary)  # Extract and checkpoint a single PDF without losing the original page identity.
            append_readable(record)  # Extract and checkpoint a single PDF without losing the original page identity.
            print(f'Interrupted. Progress: {run_dir / "progress.csv"}', flush=True)  # Expose progress or a recoverable failure to the operator.
            document.close()  # Release the document or file after its evidence has been preserved.
            raise  # Surface the inconsistency rather than silently publishing bad evidence.
        except Exception as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
            record.update({'status': 'failed', 'error': str(exc)})  # Extract and checkpoint a single PDF without losing the original page identity.
        record.update({'finished_at': timestamp(), 'duration_seconds': round(time.perf_counter() - start, 2)})  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['pages'] = [p for p in summary['pages'] if p['page'] != page_number] + [record]  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['pages'].sort(key=lambda p: p['page'])  # Extract and checkpoint a single PDF without losing the original page identity.
        save_progress(run_dir, summary)  # Extract and checkpoint a single PDF without losing the original page identity.
        append_readable(record)  # Finish private text before notifying the public writer.
        if runtime.get('progress') is not None:  # A page is complete even when other pages remain.
            runtime['progress'].put((runtime['current_document'],record,len(document)))  # Send only a compact record; artifacts remain on disk.
        print(f"Page {page_number}/{len(document)}: {record['route'] if 'route' in record else 'unknown'} — {record['status']}", flush=True)  # Expose progress or a recoverable failure to the operator.
    summary['status'] = 'partial_failure' if any(p['status'] == 'failed' for p in summary['pages']) else 'needs_review' if any(p['status']=='needs_review' for p in summary['pages']) else 'complete'  # Separate actionable review from successfully processed, unverified output.
    summary['elapsed_seconds_this_invocation'] = round(time.perf_counter() - run_started, 3)  # Extract and checkpoint a single PDF without losing the original page identity.
    summary.pop('continuation_error', None)  # Extract and checkpoint a single PDF without losing the original page identity.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        continuation_path, count = write_candidates(run_dir, summary['pages'], str(source), source_hash)  # Determine continuation path, count for this step: extract and checkpoint a single PDF without losing the original page identity.
        summary['continuation_candidates'] = str(continuation_path)  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['continuation_candidate_count'] = count  # Extract and checkpoint a single PDF without losing the original page identity.
    except Exception as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
        summary['continuation_error'] = str(exc)  # Extract and checkpoint a single PDF without losing the original page identity.
    try:  # Handle a failure in this step without silently accepting incomplete evidence.
        rebuild_document(readable_path, summary)  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['readable_status'] = 'generated'  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['readable_extract_sha256'] = file_sha256(readable_path)  # Verify the exact bytes associated with the evidence.
        summary.pop('readable_error', None)  # Extract and checkpoint a single PDF without losing the original page identity.
    except (OSError, ValueError, KeyError, TypeError) as exc:  # Keep the cause of this failure visible and preserve recoverable progress.
        summary['readable_status'] = 'failed'  # Extract and checkpoint a single PDF without losing the original page identity.
        summary['readable_error'] = str(exc)  # Extract and checkpoint a single PDF without losing the original page identity.
        print(f'Warning: could not finalize readable extract: {exc}', file=sys.stderr, flush=True)  # Expose progress or a recoverable failure to the operator.
    save_progress(run_dir, summary)  # Extract and checkpoint a single PDF without losing the original page identity.
    print(f"Results: {run_dir / 'run.json'}")  # Expose progress or a recoverable failure to the operator.
    print(f"Progress: {run_dir / 'progress.csv'}")  # Expose progress or a recoverable failure to the operator.
    print(f'Readable extract: {readable_path}')  # Expose progress or a recoverable failure to the operator.
    document.close()  # Release the document or file after its evidence has been preserved.
    return run_dir / 'run.json'  # Return the result to the next stage: extract and checkpoint a single PDF without losing the original page identity.


MANIFEST = 'sequence recorded_at document_id source_path source_sha256 extract_path status estimated_worker_seconds worker_attempt_seconds actual_minus_estimate_seconds page_processing_seconds page_count pages_with_word_evidence pages_without_word_evidence word_rows native_words ocr_words missing_word_pages review_pages failed_pages precision recall reference_pages predicted_tokens reference_tokens matched_tokens reference_method reviewed_boxes boxes_iou50 boxes_iou50_exact engine detail'.split()  # Give users one inventory row per source version.
WORDS = 'page_token_precision page_token_recall metric_scope page_extraction_seconds document_id source_path source_sha256 extract_path page word_id word provenance raw_confidence x0 y0 x1 y1 coordinate_system page_route page_status word_status item_ref reconciliation_status proposed_text table_cell_candidate readable_action readable_before reviewed_box_id reviewed_word box_iou reviewed_text_exact'.split()  # Store observed words separately from proposals or document-level metrics.
COORDS = 'PDF points; top-left of the displayed cropped source page'  # Identify the source coordinate frame for every box.

def read_json(path):  # Read saved evidence without repeating inference.
    return json.loads(Path(path).read_text(encoding='utf-8'))  # Keep numbers and Unicode as written by the extraction engine.

def fingerprint(*parts):  # Build unambiguous keys across file paths, content versions, and processing settings.
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()  # Encode each component separately so different paths cannot collide by concatenation.

def references(path):  # Accept only explicit full-page transcriptions supplied by the operator.
    by_hash = {}  # Index annotations by original content and one-based page.
    if path:  # Without a reference, precision and recall are unknown.
        with path.open(encoding='utf-8-sig', newline='') as stream:  # Read Excel-compatible UTF-8 reference files.
            reader = csv.DictReader(stream)  # Bind annotations by named columns, not position.
            if not {'source_sha256','page','text'} <= set(reader.fieldnames or ()):  # Reject inputs that cannot identify a full source page.
                raise ValueError('Reference CSV requires source_sha256,page,text (one complete page transcription per row).')  # Show the expected annotation contract.
            for row in reader:  # Preserve every explicitly annotated page, including a verified empty page.
                key, number, text = row['source_sha256'].strip().lower(), int(row['page']), row['text']  # Normalize the key without changing the reference transcription.
                if not re.fullmatch('[0-9a-f]{64}', key) or number < 1 or text is None or number in by_hash.setdefault(key, {}):  # Reject ambiguous or malformed annotations.
                    raise ValueError(f'Invalid or duplicate reference on CSV line {reader.line_num}')  # Point the operator to the conflicting entry.
                by_hash[key][number] = text  # Keep independent full-page text for a later measured comparison.
    return by_hash  # Leave unannotated PDFs and pages unscored.

def add_box_references(by_hash,path):  # Accept reviewed word samples bound to exact source versions.
    if not path:  # Without labels, location accuracy is unknown.
        return  # Leave all geometry evaluation fields blank.
    with path.open(encoding='utf-8-sig',newline='') as stream:  # Read the project's reviewed-box CSV format.
        reader = csv.DictReader(stream)  # Use named columns to keep each coordinate unambiguous.
        if not {'source_sha256','page','word','x0','y0','x1','y1'} <= set(reader.fieldnames or []):  # Require source identity and original displayed-page coordinates.
            raise ValueError('Word references require source_sha256,page,word,x0,y0,x1,y1 (top-left displayed original PDF points).')  # Explain the expected input.
        seen = set()  # Prevent duplicated labels from receiving independent credit.
        for row in reader:  # Validate each reviewed word before any output is written.
            sha,n,word = row['source_sha256'].strip().lower(),int(row['page']),row['word']  # Review labels apply only to this source and page.
            box = tuple(float(row[k]) for k in ('x0','y0','x1','y1'))  # Keep sub-point precision for small words.
            if not re.fullmatch('[0-9a-f]{64}',sha) or n<1 or not word or len(word.split())!=1 or not all(math.isfinite(v) for v in box) or not (0<=box[0]<box[2] and 0<=box[1]<box[3]) or (sha,n,box) in seen:  # Reject degenerate boxes and repeated references.
                raise ValueError(f'Invalid or duplicate reviewed box on line {reader.line_num}')  # Identify the correction needed.
            seen.add((sha,n,box))  # Reserve the reviewed location once.
            pages = by_hash.setdefault(sha,{})  # Keep source versions independent.
            if not isinstance(pages.get(n),dict):  # Keep existing full-page transcription when both reference types exist.
                pages[n] = dict(text=pages.get(n),boxes=[])  # Partial labels never substitute for a full page.
            pages[n]['boxes'].append(dict(word=word,box=box))  # Preserve original label order for stable reference IDs.


def box_iou(a,b):  # Measure spatial agreement without using word spelling.
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))  # Compute shared area.
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter  # Compute the combined area occupied by either rectangle.
    return inter/area if area>0 else 0  # A degenerate prediction cannot establish a match.


def apply_reviewed_boxes(rows,refs,number):  # Compare independent reviewed locations to the observed word rectangles.
    boxes = [tuple(float(r[k]) for k in ('x0','y0','x1','y1')) for r in rows]  # Both sets use the same displayed source-page coordinates.
    pairs = sorted((-score,i,j) for i,ref in enumerate(refs) for j,b in enumerate(boxes) if (score:=box_iou(ref['box'],b))>0)  # Match geometry first, with deterministic ties and no spelling bias.
    used_ref,used_pred = set(),set()  # A label or prediction may be used at most once.
    for neg,i,j in pairs:  # Assign strongest overlaps first.
        if i in used_ref or j in used_pred:  # Refuse weaker competing candidates.
            continue  # Leave unassigned predictions unscored.
        used_ref.add(i); used_pred.add(j)  # Reserve this one-to-one match.
        rows[j].update(reviewed_box_id=f'{number}:{i+1}',reviewed_word=refs[i]['word'],box_iou=-neg,reviewed_text_exact=normal(rows[j]['word'])==normal(refs[i]['word']))  # Keep spelling comparison separate from rectangle IoU.



def tokens(text):  # Define document accuracy as case-insensitive, punctuation-preserving word overlap.
    return Counter(unicodedata.normalize('NFKC', text).casefold().split())  # Count repeated words separately; this metric does not measure order or geometry.

def observations(page):  # Emit only actual word boxes; line boxes are never duplicated across words.
    if not page.get('links'):  # A page can fail before any trustworthy coordinate mapping exists.
        return  # Leave the page in the missing-evidence manifest count.
    if page.get('word_file'):  # Prefer the independent word pass on scanned pages.
        proposals = {}  # Index optional reconciliation without changing what OCR observed.
        if page.get('reconciliation_json'):  # A failed reconciliation does not invalidate raw observations.
            for result in read_json(page['reconciliation_json'])['results']:  # Keep each unverified grouping traceable to its fragments.
                for fragment in result['fragments']:  # A proposed merge can involve several distinct raw boxes.
                    proposals[fragment['fragment_index']] = result  # Preserve the raw row while attaching its proposed reading.
        with Path(page['word_file']).open(encoding='utf-8-sig', newline='') as stream:  # Stream the original scores and source-page boxes.
            for index, row in enumerate(csv.DictReader(stream)):  # Keep the engine's original order for reproducible word IDs.
                proposal = proposals.get(index, {})  # A missing proposal is normal for an unverified fragment.
                yield dict(word=row['word'], provenance='rapidocr_word_fragment', raw_confidence=row['confidence'], x0=row['x0'], y0=row['y0'], x1=row['x1'], y1=row['y1'], word_status=row['link_status'], item_ref=row['item_ref'], reconciliation_status=proposal.get('status',''), proposed_text=proposal.get('text',''), table_cell_candidate=proposal.get('table_cell_candidate',''),readable_action=proposal.get('readable_action',''),readable_before=proposal.get('readable_before',''))  # Publish raw evidence alongside, never replaced by, tentative corrections.
        return  # Do not double count OCR output from Docling's parallel line pass.
    linked = read_json(page['links'])  # Use Docling's native word cells, including unlinked evidence.
    groups = [(item['ref'], item['cells']) for item in linked['items']] + [('', linked['unlinked_or_ambiguous'])]  # Preserve words that could not be assigned to a document item.
    cells = sorted(((cell['cell_index'], ref, cell) for ref, group in groups for cell in group if cell['cell_kind'] == 'word_cells'), key=lambda r:r[0])  # Exclude text-line cells; their boxes are not word boxes.
    for _, ref, cell in cells:  # Keep each real word in source order.
        word = cell['text']  # Preserve the observed string without correcting spelling or case.
        if len(word.split()) != 1:  # Some engines label multiword or blank snippets as word cells.
            continue  # Never invent individual boxes by dividing a line.
        box = cell['source_display_bbox_top_left']  # Use the mapped original display page, not rotated working coordinates.
        ocr = cell['source'] == 'ocr'  # Only OCR observations can have an OCR confidence.
        yield dict(word=word, provenance='docling_ocr_word_cell' if ocr else 'native_pdf_word_cell', raw_confidence=cell['confidence'] if ocr else '', x0=box[0], y0=box[1], x1=box[2], y1=box[3], word_status='linked_unverified' if ref else 'ambiguous' if cell.get('ambiguous_candidates') else 'unlinked', item_ref=ref)  # Preserve uncertain links and leave native confidence empty.

def page_evidence(page, reference):  # Compute page overlap once using raw observations and a supplied complete transcription.
    rows = observations(page)  # Unannotated pages can stream directly to disk.
    counts, truth, metrics = Counter(),Counter(),{}  # Leave accuracy fields blank when no annotation is supplied.
    if isinstance(reference,dict):  # A reviewed word sample is not a full-page transcription.
        rows = list(rows)  # Limit label matching to one page of observations.
        apply_reviewed_boxes(rows,reference['boxes'],page['page'])  # Keep raw word observations unchanged.
        reference = reference.get('text')  # Only this optional full-page field can support precision/recall.
    if reference is not None:  # Only an independent complete page reference supports measured text overlap.
        rows = list(rows)  # Buffer at most this one reference page.
        for row in rows:  # Count the observed tokens before applying any proposed display corrections.
            counts.update(tokens(row['word']))  # Preserve repeated tokens as separate observations.
        truth = tokens(reference)  # Apply the exact same comparison contract to the reference.
        hits = (counts & truth).total()  # Exclude duplicate predictions beyond the reference multiplicity.
        metrics = dict(page_token_precision=hits/counts.total() if counts.total() else '',page_token_recall=hits/truth.total() if truth.total() else '',metric_scope='whole annotated page; NFKC casefold whitespace-token multiset; ignores order/boxes')  # These page ratios are not word confidences.
    return rows,counts,truth,metrics  # Both progress and final publication use these identical values.


def word_record(identity, page, index, raw_word, metrics):  # Give a raw word one stable ID and the same evidence on live and final export.
    number = page['page']  # Keep the one-based original source page number.
    return dict(**identity,**metrics,page_extraction_seconds=page.get('duration_seconds',''),page=number,word_id=f"{identity['document_id']}:{number}:{index}",coordinate_system=COORDS,page_route=page.get('route',''),page_status=page.get('status','failed'),**raw_word)  # Never manufacture geometry or confidence.



def extract_path(document_id, source):
    """Give each source version a stable, source-based output filename."""
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, document_id).hex[:8]
    return Path(Path(source).stem + '-extract-' + suffix + '.md').as_posix()


def package_document(job, summary, detail=''):  # Stage a readable document and its shared-report rows without touching collection files.
    doc_id, source, digest, destination, engine, model_dir, config, reference = job  # Retain immutable identity and the exact reviewed pages for this source.
    stage = Path(destination) / 'packages' / uuid.uuid4().hex  # A new package cannot damage an earlier committed one.
    stage.mkdir(parents=True)  # Reserve the private output before assembling either deliverable.
    relative = extract_path(doc_id, source)  # A stable relative link still works when the report bundle is copied.
    identity = dict(document_id=doc_id, source_path=source, source_sha256=digest, extract_path=relative)  # Give each word the same join keys as its manifest entry.
    pages = {p['page']:p for p in summary.get('pages', [])}  # Locate page evidence without assuming all pages finished.
    count = summary.get('page_count')  # Unknown page counts stay blank when a PDF cannot be opened.
    missing, review, failed = [], [], []  # Record exact page numbers for operator follow-up.
    reviewed_boxes = supported_boxes = exact_boxes = 0  # Include missed reference boxes in the labelled denominator.
    native = ocr = found = predicted = expected = matched = 0  # Aggregate observed counts without keeping the collection's words in memory.
    problems = [detail] if detail else []  # Keep input failures distinct from extraction warnings.
    if count is not None and set(reference) - set(range(1, count+1)):  # Reject a reference page outside this exact original PDF.
        problems.append('Reference includes page numbers outside the PDF; metrics withheld.')  # Do not silently discard an invalid scored page.
        reference = {}  # Retain extract and metadata even when annotations need fixing.
    with (stage/'words.csv').open('w', encoding='utf-8', newline='') as stream:  # Stage rows privately until this document is ready to publish.
        writer = csv.DictWriter(stream, WORDS)  # Maintain one stable schema for all workers.
        writer.writeheader()  # Make the private staging file independently interpretable on recovery.
        for number in range(1, (count or 0)+1):  # Include failed, blank, and unprocessed pages in coverage denominators.
            page = pages.get(number, {})  # An absent record means no page evidence exists.
            page_words, page_total = Counter(), 0  # Keep reference comparisons within the same source page.
            page_failed = page.get('status') not in ('complete','needs_review') or page.get('readable_status') != 'generated' or page.get('word_status') == 'failed' or page.get('reconciliation_status') == 'failed'  # Track engine failures even if a readable page was still recovered.
            if page_failed:  # Failed processing must remain eligible for retry.
                failed.append(number)  # Identify the affected source page.
            if page.get('status') == 'needs_review' or page_failed:  # Retain the pipeline's explicit review outcome.
                review.append(number)  # Never present machine output as reviewed text.
            annotation = reference.get(number)  # Count reviewed labels whether a prediction exists or not.
            reviewed_boxes += len(annotation['boxes']) if isinstance(annotation,dict) else 0  # A missed word remains in the coverage denominator.
            evidence,page_words,truth,metrics = page_evidence(page,reference.get(number))  # Score raw words only when the reviewed full-page text exists.
            for page_total, row in enumerate(evidence,1):  # Preserve each observation's stable page-local identity.
                writer.writerow(word_record(identity,page,page_total,row,metrics))  # Final rows exactly match already published page rows.
                supported_boxes += int(row.get('box_iou',0)>=.5)  # Count reviewed boxes that overlap an observation by at least 50 percent.
                exact_boxes += int(row.get('box_iou',0)>=.5 and row.get('reviewed_text_exact',False))  # Distinguish matching location from matching location and spelling.
                native += int(row['provenance'] == 'native_pdf_word_cell')  # Count native text without adding a model score.
                ocr += int(row['provenance'] != 'native_pdf_word_cell')  # Count OCR evidence separately from native PDF text.
            if not page_total:  # No trustworthy word observations exist on this page.
                missing.append(number)  # Do not call an unobserved native or scanned page complete.
            else:  # Any observed words establish availability but not completeness.
                found += 1  # Name this count "pages with word evidence" rather than verified coverage.
            if metrics:  # Partial word labels never contribute full-page precision or recall.
                matched += (page_words & truth).total()  # Count measured text overlap on the same page.
                predicted += page_words.total()  # Include every prediction on a fully annotated page.
                expected += truth.total()  # Include all supplied full-page reference words.
    text_reference = {n:v for n,v in reference.items() if not isinstance(v,dict) or v.get('text') is not None}  # Report full-transcription page numbers separately from word-box labels.
    readable = stage/'extract.md'  # Freeze the matching document extract alongside its word evidence.
    if summary.get('readable_status') == 'generated' and valid_readable_extract(summary):  # Publish only an extract the page runner finalized successfully.
        shutil.copyfile(summary['readable_extract'], readable)  # Preserve original text, layout warnings, and source page ordering.
    else:  # Each unreadable PDF still receives an explicit failure notice.
        problems.append('No finalized readable extract.')  # Make a missing document distinguishable from a blank one.
        readable.write_text(f'# Extraction unavailable\n\nSource: {source}\n\nSHA-256: {digest}\n\n' + '\n'.join(problems), encoding='utf-8')  # Provide a traceable placeholder rather than inventing text.
    reasons = [f"p{n}: " + '; '.join(p['warnings']) for n,p in pages.items() if p.get('warnings') and n not in failed]  # Publish actionable reasons for otherwise successful pages.
    if missing and count is not None:  # No word evidence is an explicit completeness signal.
        reasons.append('No word locations on pages: '+','.join(map(str,missing)))  # Distinguish missing coverage from a healthy unreviewed result.
    errors = [f"p{n}: {p.get('error') or p.get('readable_error') or '; '.join(p.get('warnings',[]))}" for n,p in pages.items() if n in failed]  # Explain page failures in the shared manifest.
    status = 'failed' if count is None or not valid_readable_extract(summary) else 'partial_failure' if failed else 'needs_review' if review or missing or reasons or problems else 'complete'  # Treat all unverified machine results as requiring review.
    manifest = dict(**identity, status=status, page_processing_seconds=round(sum(p.get('duration_seconds',0) for p in pages.values() if p.get('status') in ('complete','needs_review')),3), page_count=count, pages_with_word_evidence=found, pages_without_word_evidence=len(missing) if count is not None else '', word_rows=native+ocr, native_words=native, ocr_words=ocr, missing_word_pages=';'.join(map(str,missing)), review_pages=';'.join(map(str,review)), failed_pages=';'.join(map(str,failed)), precision=matched/predicted if predicted else '', recall=matched/expected if expected else '', reference_pages=';'.join(map(str,sorted(text_reference))) if count is not None else '', predicted_tokens=predicted if text_reference and count is not None else '', reference_tokens=expected if text_reference and count is not None else '', matched_tokens=matched if text_reference and count is not None else '', reference_method='pagewise NFKC casefold whitespace token multiset; complete reference pages only' if text_reference and count is not None else '', engine=engine, detail=' | '.join(problems+errors+reasons))  # Leave unmeasured ratios blank, including undefined zero denominators.
    manifest.update(reviewed_boxes=reviewed_boxes or '',boxes_iou50=supported_boxes if reviewed_boxes else '',boxes_iou50_exact=exact_boxes if reviewed_boxes else '')  # Denominators refer to reviewed labels, not to the entire population of words.
    return manifest, str(stage)  # Send only compact inventory data and paths back to the coordinator.


def extract_job(job):  # Run one source version using this process's warm models and private checkpoints.
    started = time.perf_counter()  # Measure this worker attempt, including checkpoint validation and package staging, but excluding coordinator queue and shared publication.
    doc_id, source, digest, destination, engine, model_dir, config, reference = job  # Keep document, model, and annotation identity together.
    directory = Path(destination)/'runs'/doc_id/config  # Isolate changed PDF bytes or model settings from earlier checkpoints.
    directory.mkdir(parents=True, exist_ok=True)  # Create only private worker space; the parent owns the shared reports.
    RUNTIME['current_document'] = doc_id  # Tag page progress so that late notifications cannot affect another PDF.
    args = [source, '--output-root', str(directory), '--engine', engine, '--artifacts', model_dir]  # Pass the original PDF directly to the page engine.
    runs = sorted(directory.glob('*/run.json'), key=lambda p:p.stat().st_mtime_ns, reverse=True)  # Find the latest page checkpoint after an interruption.
    try:  # A malformed PDF should not stop the rest of the collection.
        if runs:  # The engine verifies source, model settings and each saved page before reuse.
            args += ['--resume', str(runs[0].parent)]  # Continue from validated pages instead of repeating all inference.
        RUNTIME['expected_digest'] = digest  # Verify the discovered version once at worker startup, then again at final publication.
        summary = read_json(process_document(args, RUNTIME))  # Keep model instances alive between PDF jobs in this worker.
        if summary['source_sha256'] != digest or file_sha256(Path(source)) != digest:  # Check discovery against the material actually extracted.
            raise ValueError('Source changed during extraction; rerun to discover its new version.')  # Never link new bytes to an old source hash.
        published = package_document(job, summary)  # Stage the completed output for serial publication.
    except (Exception, SystemExit) as exc:  # Preserve failures, including invalid PDF or incompatible checkpoints.
        published = package_document(job, {}, f'{type(exc).__name__}: {exc}')  # Emit one traceable failure record and a readable notice.

    published[0]['worker_attempt_seconds'] = round(time.perf_counter()-started,3)  # Report this invocation only; a resumed attempt does not claim earlier processing time.
    return published  # The parent will publish this timed result once it is durable.

def publish_extract(root, row, stage):  # Make the readable file available before exposing its manifest link.
    target = root/row['extract_path']  # Keep file names deterministic and collision-free.
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix('.md.tmp')  # Readers never see a partly copied Markdown file.
    shutil.copyfile(stage/'extract.md', pending)  # Copy the immutable committed version from private storage.
    os.replace(pending, target)  # Replace a prior report only when the new one is ready.

def write_records(manifests, words, row, stage):  # Project a committed document into both shared reports.
    with (stage/'words.csv').open(encoding='utf-8', newline='') as source:  # Reuse staged words without loading them all into memory.
        words.writerows(csv.DictReader(source))  # Append this PDF's observed words as a single writer.
    manifests.writerow(row)  # Record the accompanying source version and coverage summary.

def log_row(db, row):  # Store ordered status history alongside each committed document.
    row['recorded_at'] = datetime.now().astimezone().isoformat(timespec='seconds')  # Timestamp the state change where it becomes durable.
    cursor = db.execute('INSERT INTO events(id,status,manifest) VALUES(?,?,?)',(row['document_id'],row['status'],'{}'))  # Allocate a strictly increasing sequence for latest-status selection.
    row['sequence'] = cursor.lastrowid  # Include the ordering key in both the journal and its public projection.
    encoded = json.dumps(row,ensure_ascii=False)  # Preserve names, error messages, and metric values exactly.
    db.execute('UPDATE events SET manifest=? WHERE seq=?',(encoded,row['sequence']))  # Save the full event in the same transaction.
    db.execute("UPDATE settings SET value='1' WHERE key='dirty'")  # Recover through a rebuild if the CSV projection is interrupted.
    return encoded  # Reuse the exact same serialized data for publication.


def clean(db):  # Acknowledge that both public CSV files reflect the committed journal.
    with db:  # Keep the dirty marker set until all projection writes have succeeded.
        db.execute("UPDATE settings SET value='0' WHERE key='dirty'")  # Future new sources can append without rebuilding history.


def flush(stream):  # Make acknowledged report changes durable.
    stream.flush()  # Push buffered CSV rows to the operating system.
    os.fsync(stream.fileno())  # Wait until those bytes are persisted before marking the projection clean.


def live_schema(db):  # Keep the recoverable source of each live page separate from completed document packages.
    db.execute('CREATE TABLE IF NOT EXISTS live_pages(id TEXT,page INTEGER,stage TEXT,rows INTEGER,PRIMARY KEY(id,page))')  # A retry can never create two live pages for the same source.


def write_live_page(root,row,stage,first):  # Update the public extract as soon as a completed page is available.
    target = root/row['extract_path']  # Keep the path stable throughout the run.
    if first:  # Replace the pending notice or an earlier run's partial extract.
        atomic_text(target,header(row['source_path'],row['source_sha256']))  # Publish the original identity before appending page sections.
    with target.open('a',encoding='utf-8',newline='') as out,(stage/'extract.md').open(encoding='utf-8') as src:  # Only the coordinator writes the shared folder.
        shutil.copyfileobj(src,out)  # Append this page, never the entire previous document.
        flush(out)  # Commit readable progress before exposing its manifest event.


def publish_page(db, root, job, page, count):  # Durably append one completed source page to all business outputs.
    live_schema(db)  # Unit tests and older journals may not yet have the live page table.
    identity,source,digest,_,engine,_,_,reference = job  # Keep the same version and reference as the assigned worker.
    number = page['page']  # A committed page can be observed again when a worker restarts.
    if not db.execute('SELECT 1 FROM active WHERE id=?',(identity,)).fetchone() or db.execute('SELECT 1 FROM live_pages WHERE id=? AND page=?',(identity,number)).fetchone():  # Ignore late and duplicate notifications.
        return  # Never append superseded page rows after final publication.
    stage = root/'_state'/'pages'/uuid.uuid4().hex  # A partially written staging area is never authoritative.
    stage.mkdir(parents=True)  # Reserve a unique durable location for this page.
    base = dict(document_id=identity,source_path=source,source_sha256=digest,extract_path=extract_path(identity,source))  # Use the exact document join keys on every row.
    rows,_,_,metrics = page_evidence(page,reference.get(number) if all(1<=n<=count for n in reference) else None)  # Reject out-of-source annotations consistently with final publication.
    total = 0  # A failed or blank page can legitimately have no observed words.
    with (stage/'words.csv').open('w',encoding='utf-8',newline='') as stream:  # Stage the exact word rows for crash recovery.
        writer = csv.DictWriter(stream,WORDS)  # Both normal processing and recovery share one schema.
        writer.writeheader()  # Make this evidence independently reusable.
        for total,raw_word in enumerate(rows,1):  # Assign the same stable index as final packaging.
            writer.writerow(word_record(base,page,total,raw_word,metrics))  # Attach source coordinates and any genuine metrics.
    (stage/'extract.md').write_text(page_section(page,saved_page_text(page)),encoding='utf-8')  # Freeze this one page's text and its warnings.
    first = not db.execute('SELECT 1 FROM live_pages WHERE id=? LIMIT 1',(identity,)).fetchone()  # The first page replaces the placeholder.
    has_old = first and db.execute('SELECT 1 FROM documents WHERE id=?',(identity,)).fetchone()  # A previous version of these word rows must be superseded first.
    event = dict(**base,status='processing',page_count=count,engine=engine,detail=f"Page {number}/{count}: {page['status']}; {total} located words. "+'; '.join(page.get('warnings',[])))  # Explicitly identify completed page progress in the manifest.
    with db:  # Commit page identity and the event before touching public files.
        db.execute('INSERT INTO live_pages VALUES(?,?,?,?)',(identity,number,stage.relative_to(root).as_posix(),total))  # Retain one authoritative staging package for this page.
        log_row(db,event)  # A dirty journal triggers a clean reconstruction after a torn append.
    if has_old:  # Superseded rows cannot safely be removed by appending to CSV.
        project(db,root)  # Rebuild once at the start of a replacement, not after every page.
    else:  # The normal path appends only this page.
        write_live_page(root,event,stage,first)  # Keep an immediately readable public document.
        with (root/'metadata.csv').open('a',encoding='utf-8',newline='') as w,(root/'manifest.csv').open('a',encoding='utf-8',newline='') as m:  # Serialize all shared writes through the coordinator.
            write_records(csv.DictWriter(m,MANIFEST),csv.DictWriter(w,WORDS),event,stage)  # Add the page's words and its event exactly once.
            flush(w); flush(m)  # Persist both reports before marking this page published.
        clean(db)  # An interruption before this point triggers reconstruction from the journal.



def project(db, root, record=None):  # Append new word packages; rebuild from the journal only for recovery or replacement.
    live_schema(db)  # Support recovery of partially published documents.
    if record:  # A first completed document can append without revisiting previous metadata.
        serialized, relative = record  # Obtain the exact committed inventory and word package.
        row, stage = json.loads(serialized), root/relative  # Resolve the immutable saved package.
        publish_extract(root,row,stage)  # Publish the readable extract before its result event.
        with (root/'metadata.csv').open('a',encoding='utf-8',newline='') as w, (root/'manifest.csv').open('a',encoding='utf-8',newline='') as m:  # Give the coordinator sole write access to the shared projections.
            write_records(csv.DictWriter(m,MANIFEST),csv.DictWriter(w,WORDS),row,stage)  # Append raw words and the result event once.
            flush(w)  # Persist the new word evidence first.
            flush(m)  # Persist its matching completion event.
    else:  # Repair an interrupted append or replace evidence from a prior failed attempt.
        mw, ww = root/'manifest.csv.tmp', root/'metadata.csv.tmp'  # Reconstruct away from the files that people are reading.
        with mw.open('w',encoding='utf-8',newline='') as m, ww.open('w',encoding='utf-8',newline='') as w:  # Create a complete replacement projection.
            manifests, words = csv.DictWriter(m,MANIFEST), csv.DictWriter(w,WORDS)  # Set public columns consistently on repair.
            manifests.writeheader()  # Establish headers even for an empty collection.
            words.writeheader()  # Allow a zero-observation corpus to remain machine readable.
            for encoded, in db.execute('SELECT manifest FROM events ORDER BY seq'):  # Reconstruct every status transition in its original order.
                manifests.writerow(json.loads(encoded))  # Keep interrupted and superseded results visible in the audit trail.
            for encoded, relative in db.execute('SELECT manifest,stage FROM documents d WHERE NOT EXISTS(SELECT 1 FROM live_pages p WHERE p.id=d.id) ORDER BY seq'):  # Use only the current published word set for each identity.
                row, stage = json.loads(encoded), root/relative  # Resolve the current document package.
                publish_extract(root,row,stage)  # Restore any missing or stale readable deliverable.
                with (stage/'words.csv').open(encoding='utf-8',newline='') as staged:  # Read one document's word file at a time.
                    words.writerows(csv.DictReader(staged))  # Avoid duplicate rows from superseded attempts.
            previous = None  # Reconstruct in page order without loading the whole collection into memory.
            for identity,relative,job_json in db.execute('SELECT p.id,p.stage,a.job FROM live_pages p JOIN active a ON p.id=a.id ORDER BY p.id,p.page'):  # Restore pages whose worker had not finalized its document.
                job = json.loads(job_json)  # Recover source identity from its durable assignment.
                event = db.execute('SELECT manifest FROM events WHERE id=? AND status=? ORDER BY seq DESC LIMIT 1',(identity,'processing')).fetchone()
                row = dict(extract_path=json.loads(event[0])['extract_path'] if event else extract_path(identity,job[1]),source_path=job[1],source_sha256=job[2])  # Preserve paths from older active journals.
                stage = root/relative  # Read only the package selected by the journal.
                write_live_page(root,row,stage,previous!=identity)  # Replace old content before restoring the first page.
                with (stage/'words.csv').open(encoding='utf-8',newline='') as src:  # Recover exactly the observed rows.
                    words.writerows(csv.DictReader(src))  # Append to the rebuilt metadata file once.
                previous = identity  # Later pages append to this document's extract.
            flush(w)  # Persist all reconstructed positional evidence before installation.
            flush(m)  # Persist the reconstructed audit trail.
        os.replace(ww,root/'metadata.csv')  # Install a complete word table.
        os.replace(mw,root/'manifest.csv')  # Install its paired manifest after the word table.
    clean(db)  # Mark the projection clean only after both files have been published.


def commit_result(db, root, config_key, manifest, stage):  # Commit the latest word package and its status together.
    live_schema(db)  # Old journals can be opened safely.
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='forecasts'").fetchone():  # Preserve compatibility with isolated recovery fixtures.
        prediction = db.execute('SELECT seconds FROM forecasts WHERE id=?',(manifest['document_id'],)).fetchone()  # Compare against the pre-run estimate for this version.
        if prediction and prediction[0] is not None:  # An uncalibrated run cannot report a meaningful difference.
            manifest['estimated_worker_seconds'] = prediction[0]  # Retain the per-document prediction, not a wall-time guarantee.
            if manifest.get('worker_attempt_seconds') is not None:  # Interrupted snapshots may lack a complete attempt duration.
                manifest['actual_minus_estimate_seconds'] = round(manifest['worker_attempt_seconds']-prediction[0],3)  # Includes setup and packaging for this worker attempt.
        if manifest['status'] in ('complete','needs_review') and manifest.get('page_processing_seconds') and manifest.get('page_count'):  # Train only on measured page work, never cached retry wall time.
            db.execute('INSERT OR REPLACE INTO timings VALUES(?,?,?,?)',(manifest['document_id'],manifest['engine'],manifest['page_count'],manifest['page_processing_seconds']))  # Reprocessing one version does not double count its measured pages.
    live_count,live_words = db.execute('SELECT count(*),coalesce(sum(rows),0) FROM live_pages WHERE id=?',(manifest['document_id'],)).fetchone()  # Check whether all expected live words are already present.
    previous = db.execute('SELECT manifest FROM documents WHERE id=?',(manifest['document_id'],)).fetchone()  # Determine whether old published words need replacement.
    relative = Path(stage).relative_to(root).as_posix()  # Keep the private package path movable with this output folder.
    with db:  # Commit the authoritative event, word package, and recovery marker together.
        db.execute('DELETE FROM live_pages WHERE id=?',(manifest['document_id'],))  # Promote the single final document package to the new authority.
        db.execute('DELETE FROM active WHERE id=?',(manifest['document_id'],))  # Remove current assignment in the same durable result transaction.
        encoded = log_row(db,manifest)  # Append a durable completion, failure, or interruption.
        db.execute('INSERT INTO documents(id,config,manifest,stage) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET config=excluded.config,manifest=excluded.manifest,stage=excluded.stage', (manifest['document_id'],config_key,encoded,relative))  # Retain just one current word package per source version.
    if live_count and live_count == manifest.get('page_count') and live_words == manifest.get('word_rows') and manifest['status'] in ('complete','needs_review','partial_failure'):  # All live rows came from the same immutable page checkpoints.
        publish_extract(root,manifest,Path(stage))  # Replace the live text with its final page-ordered rendition.
        with (root/'manifest.csv').open('a',encoding='utf-8',newline='') as stream:  # The final status needs one event, not another copy of every word.
            csv.DictWriter(stream,MANIFEST).writerow(manifest)  # Preserve the full live page history.
            flush(stream)  # Guarantee the final event is durable.
        clean(db)  # The live files already match the current final authority.
    else:  # Missing notifications, interruption, or failed source validation need a fresh projection.
        project(db,root,None if live_count or (previous and json.loads(previous[0])['word_rows']) else (encoded,relative))  # Recovery cannot double-count partial results.
    print(f"{manifest['status']}: {manifest['source_path']} ({manifest['word_rows']} word rows)",flush=True)  # Report completion once the public files have caught up.


def status_event(db,root,job,status,detail=''):  # Make pending, active, and interrupted work visible in the public manifest.
    identity, source, digest, destination, engine, model_dir, cfg, annotations = job  # Carry the same provenance keys as eventual word rows.
    row = dict(document_id=identity,source_path=source,source_sha256=digest,extract_path=extract_path(identity,source),engine=engine,status=status,detail=detail)  # Keep unmeasured counts blank before completion.
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='forecasts'").fetchone():  # Unit tests and older journals may predate forecasts.
        predicted = db.execute('SELECT pages,seconds FROM forecasts WHERE id=?',(identity,)).fetchone()  # Link estimates to the same source version as the event.
        if predicted:  # Discovery failures do not have usable workload estimates.
            row.update(page_count=predicted[0],estimated_worker_seconds=predicted[1])  # Make the document estimate visible from its pending status onward.
    target = root/row['extract_path']  # Offer a portable readable link from the first discovery event.
    if not target.exists():  # Preserve any already published partial or completed content during a retry.
        atomic_text(target,f'# Extraction pending\n\nSource: {source}\n\nSHA-256: {digest}\n\nSee manifest.csv for the latest processing status.\n')  # Give a pending source a truthful, clearly marked placeholder.
    with db:  # Save the new state before appending it to the CSV.
        log_row(db,row)  # Flag recovery if this append is interrupted.
    with (root/'manifest.csv').open('a',encoding='utf-8',newline='') as stream:  # Progress events never touch the growing word table.
        csv.DictWriter(stream,MANIFEST).writerow(row)  # Append the visible state under the existing header.
        flush(stream)  # Persist the event before starting or resuming document work.
    clean(db)  # Acknowledge successful projection of this state change.


def interrupted(db,root):  # Recover all active work after Ctrl+C, a worker crash, or a killed coordinator.
    for identity,key,encoded in list(db.execute("SELECT id,config,job FROM active")):  # Active assignments survive abrupt shutdown in the journal.
        job = json.loads(encoded)  # Get the original path, version, and reference used when work was assigned.
        job[-1] = {int(k):v for k,v in job[-1].items()}  # Restore one-based integer reference page numbers.
        runs = sorted((root/'_state'/'runs'/identity/job[6]).glob('*/run.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)  # Locate the latest durable page checkpoint.
        if runs:  # A worker may have finished some pages before the rest of the document stopped.
            summary = read_json(runs[0])  # Read only checkpointed state after workers have been stopped.
            partial = runs[0].parent/'interrupted.md'  # Keep the partial public report separate from the engine's resumable file.
            rebuild_document(partial,summary)  # Add explicit missing-page notices for all unfinished pages.
            summary.update(readable_extract=str(partial),readable_status='generated',readable_extract_sha256=file_sha256(partial))  # Link the partial report to the checkpointed source content.
            row, stage = package_document(job,summary,'Interrupted; repeat the command to retry unfinished pages.')  # Preserve already saved text and raw word observations.
            row['status'] = 'interrupted'  # Never label the incomplete source done.
            commit_result(db,root,key,row,stage)  # Publish the partial evidence without duplicating words on retry.
        else:  # An assignment may have ended before its first page checkpoint.
            status_event(db,root,job,'interrupted','Stopped before a page checkpoint; rerun to retry.')  # Record even very early interruptions.
        with db:  # Remove only work whose interruption has been published.
            db.execute('DELETE FROM active WHERE id=?',(identity,))  # Allow a fresh assignment on resume.


def workload_size(path):  # Inspect page count and file size without rendering pages or starting OCR.
    try:  # One damaged or inaccessible file must not hide the rest of the workload.
        size = path.stat().st_size  # Report the amount of PDF data in the inventory.
        with pdfium.PdfDocument(str(path)) as doc:  # Open only the PDF directory and page tree.
            return len(doc),size  # Return exact page count where the file is readable.
    except (OSError,pdfium.PdfiumError,ValueError):  # Encrypted or invalid files have unknown page counts.
        return None,0  # Exclude unknown pages from time estimates and report their count separately.


def historical_rate(db,engine):  # Use observed time from this output collection when the operator gives no calibration.
    if db is None or not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='timings'").fetchone():  # Fresh outputs and earlier versions have no reliable history.
        return None  # Unknown is more honest than a fabricated runtime.
    pages,seconds = db.execute('SELECT SUM(pages),SUM(seconds) FROM timings WHERE engine=?',(engine,)).fetchone()  # Pool measured page seconds for this engine; workload mix can still change.
    return seconds/pages if pages and seconds else None  # Omit an invalid or empty estimate.


class Forecast:  # Estimate the scheduled workload while retaining only one timing slot per worker.
    def __init__(self,workers,rate):  # Separate the operator's measured calibration from known file and page counts.
        self.slots,self.rate = [0.0]*workers,rate  # Use document-level parallelism; one long PDF still runs on one worker.
        self.files,self.pages,self.bytes,self.unknown = 0,0,0,0  # Count inputs that could not be assessed separately.
    def add(self,path):  # Simulate assigning this PDF to the next available worker.
        import heapq  # Maintain the earliest available worker without scanning all workers per document.
        count,size = workload_size(path)  # Estimate workload from the page tree without loading OCR models.
        self.files+=1; self.bytes+=size; self.pages+=count or 0; self.unknown+=int(count is None)  # Keep the upfront inventory and limitations visible.
        estimate = round(count*self.rate,3) if count is not None and self.rate is not None else None  # Estimate measured page time only; this is not a deadline.
        if estimate is not None:  # Files with unreadable page trees cannot be estimated.
            heapq.heapreplace(self.slots,self.slots[0]+estimate)  # Reflect serial processing within a PDF and parallel work between PDFs.
        return count,estimate  # Carry each document's forecast into the manifest for later comparison.
    def report(self,scope):  # Explain assumptions before PDF inference starts.
        eta = f'{max(self.slots):,.1f} seconds at {self.rate:.3f} worker-seconds/page' if self.rate is not None else 'unknown; supply --seconds-per-page from representative measurements or finish a calibration run'  # Do not invent an uncalibrated duration.
        print(f'Preflight ({scope}): {self.files:,} PDFs; {self.pages:,} known pages; {self.bytes/1048576:,.1f} MiB; {self.unknown} unreadable page counts. Estimated wall time: {eta}. {len(self.slots)} worker(s), document scheduling; excludes startup, I/O overhead, retries and unknown pages. CPU, scan complexity, and contention can change the result.',flush=True)  # Keep this clearly separate from actual elapsed time.



def schedule(db, root, input_dir, args, reviewed, cfg):  # Discover recursively, skip current publications and bound simultaneous work.
    db.execute('CREATE TEMP TABLE queue(seq INTEGER PRIMARY KEY,config TEXT,job TEXT)')  # Bound memory while listing all discovered sources before workers start.
    rate = args.seconds_per_page or historical_rate(db,args.engine)  # Prefer an operator-provided representative rate; otherwise use original page timings.
    forecast = Forecast(args.workers,rate)  # Queue only new or retryable sources for the batch estimate.
    private = root/'_state'  # Send every auxiliary file below one clearly identified directory.
    for path in input_dir.rglob('*'):  # Discover additions each time the same folder command is run.
        if path.suffix.lower() != '.pdf' or not path.is_file():  # Limit extraction to ordinary PDF file entries.
            continue  # Leave other business records untouched.
        source = str(path.absolute())  # Preserve each discovered path even when distinct names have identical bytes.
        try:  # Report inaccessible PDFs independently of accessible neighbors.
            digest = file_sha256(path)  # Identify the PDF version without trusting modification times.
            error = ''  # No discovery error affects this particular source.
        except OSError as exc:  # A locked or unreadable file is still part of the discovered inventory.
            digest, error = '', str(exc)  # Leave its unknown content hash blank and show the access failure.
        identity = fingerprint(os.path.normcase(source),digest)  # Keep duplicate contents at distinct source paths as separate documents.
        annotations = reviewed.get(digest,{})  # A page reference can only score the exact source bytes.
        key = fingerprint(cfg, annotations)  # A changed model or reference requires a refreshed publication.
        prior = db.execute('SELECT config,manifest,stage FROM documents WHERE id=?',(identity,)).fetchone()  # Check the journal without reading either growing CSV.
        if prior and prior[0] == key:  # Only equivalent settings and annotations can be skipped.
            row = json.loads(prior[1])  # Read the status and portable extract link.
            if row['status'] in ('complete','needs_review') and (root/prior[2]/'extract.md').is_file():  # A completed, previously unverified result can be reused without rerunning OCR.
                target, backup = root/row['extract_path'], root/prior[2]/'extract.md'  # Compare public content against the immutable published package.
                if not target.is_file() or file_sha256(target) != file_sha256(backup):  # Repair a deleted or externally changed managed extract.
                    publish_extract(root,row,root/prior[2])  # Recover the generated report from the staged copy.
                if not (root/prior[2]/'words.csv').is_file():  # Missing recovery evidence requires restaging rather than skipping.
                    raise FileNotFoundError(f'Missing recovery package for {source}: {root/prior[2]}')  # Stop rather than discard the authoritative word evidence.
                continue  # Never append a second copy on a no-change rerun.
        job = (identity, source, digest, str(private), args.engine, str(args.models), cfg, annotations)  # Carry only one document's paths and annotations into the worker.
        if error:  # No worker can safely run extraction without a known source version.
            commit_result(db,root,key,*package_document(job,{},error))  # Preserve the access failure as an inventory row.
        else:  # Intact PDFs can be scheduled normally.
            page_count,estimate = forecast.add(path)  # Estimate document time before model work begins.
            db.execute('INSERT OR REPLACE INTO forecasts VALUES(?,?,?)',(identity,page_count,estimate))  # Record the exact forecast used so results can be compared later.
            last = db.execute('SELECT status FROM events WHERE id=? ORDER BY seq DESC LIMIT 1',(identity,)).fetchone()  # Avoid a duplicate pending event when an earlier run stopped before assignment.
            if not last or last[0] != 'pending':  # Newly discovered and explicitly retried work is visible before model startup.
                status_event(db,root,job,'pending')  # Record every queued source in the public manifest.
            db.execute('INSERT INTO queue(config,job) VALUES(?,?)',(key,json.dumps(job)))  # Keep even very large pending inventories off the Python heap.
    forecast.report('queued files only')  # Show the pending workload before workers start.
    for key, encoded in db.execute('SELECT config,job FROM queue ORDER BY seq'):  # Dispatch the full discovered inventory in its recorded order.
        job = json.loads(encoded)  # Restore one source assignment.
        job[-1] = {int(k):v for k,v in job[-1].items()}  # JSON object keys must be converted back to integer reference page numbers.
        yield key,job  # Let the coordinator enforce the active worker limit.

def start_worker(config, progress=None):  # Reuse the fingerprint validated by the coordinator for this processing session.
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Only the coordinator decides when to stop shared publication.
    parent = multiprocessing.parent_process()  # Stop an orphan worker if its coordinator is forcibly terminated.
    if parent is not None:  # Direct calls used for diagnostics do not have a worker parent.
        import threading  # Watch the coordinator independently of model calls.
        def monitor():  # Avoid competing writes to checkpoints if a replacement coordinator starts.
            parent.join()  # Sleep until the process that owns this collection has exited.
            os._exit(1)  # Abandon the worker, leaving atomic page checkpoints for recovery.
        threading.Thread(target=monitor,daemon=True).start()  # Monitoring must not delay a normal worker exit.
    RUNTIME['progress'] = progress  # Deliver completed page records to the single public writer.
    RUNTIME['fingerprint'] = config  # Avoid repeatedly reading gigabytes of unchanged model files for every PDF.

def run_pool(jobs, db, root, workers, config):  # Keep at most one active document per requested model process.
    context = multiprocessing.get_context('spawn')  # Use the same worker policy on Windows and Unix.
    progress = context.Queue(maxsize=max(2,workers*2))  # Bound unpublished page notifications to limit memory use.
    active_jobs = {}  # Keep live job identities only for active workers.
    def drain():  # Publish queued page checkpoints before finalizing a completed document.
        while True:  # Write every available page without delaying ready documents.
            try:  # No worker or parent should wait on an empty queue.
                identity,page,count = progress.get_nowait()  # Retrieve a fully written page artifact.
            except queue.Empty:  # There are no additional pages ready for publication.
                return  # Wait for more worker activity.
            if identity in active_jobs:  # Ignore late notifications once a document outcome is final.
                publish_page(db,root,active_jobs[identity],page,count)  # Single-writer publication protects the shared CSVs.
    pool = ProcessPoolExecutor(workers, mp_context=context, initializer=start_worker, initargs=(config,progress))  # Give each worker isolated reusable models on Windows and Unix.
    pending = {}  # Hold only a bounded number of future results and their version keys.
    try:  # A broken worker or Ctrl+C must leave published rows recoverable.
        while True:  # Refill free slots until discovery and extraction both finish.
            while len(pending) < workers:  # Do not queue an unbounded list of PDFs on large collections.
                entry = next(jobs,None)  # Discover the next PDF only when capacity exists.
                if entry is None:  # The input directory has no more pending jobs.
                    break  # Drain outstanding documents before returning.
                key, job = entry  # Retain the configuration key until durable publication.
                with db:  # Persist assignment before handing a PDF to a potentially long model call.
                    db.execute('INSERT OR REPLACE INTO active VALUES(?,?,?)',(job[0],key,json.dumps(job)))  # Allow restart to distinguish interrupted from unstarted work.
                status_event(db,root,job,'processing')  # Make active work visible in the public register.
                active_jobs[job[0]] = job  # Bind page notifications to exact inputs and annotations.
                pending[pool.submit(extract_job,job)] = (key,job[0])  # Workers only create private output packages.
            if not pending:  # No pending input or outstanding result remains.
                return  # Every discovered source now has an up-to-date outcome.
            finished, _ = wait(pending,timeout=.25,return_when=FIRST_COMPLETED)  # Publish the next finished document without waiting for slower PDFs.
            drain()  # Publish completed pages during a long document, not only at its end.
            for future in finished:  # Multiple workers can finish at the same time.
                key, identity = pending.pop(future)  # Release this worker slot once its staged result arrives.
                active_jobs.pop(identity,None)  # Late page events will not replace a final publication.
                commit_result(db,root,key,*future.result())  # Only the parent serializes the shared reports.
                with db:  # Release only assignments whose finished result is durable.
                    db.execute('DELETE FROM active WHERE id=?',(identity,))  # Do not mark a committed job interrupted on a clean exit.
    except BaseException:  # Stop workers before releasing the output folder to another invocation.
        for worker in multiprocessing.active_children():  # This CLI owns its child model processes.
            worker.terminate()  # Leave atomic page checkpoints intact for the next invocation.
        raise  # Preserve the real error or interruption exit status.
    finally:  # Finish child cleanup even if publication or a worker failed.
        pool.shutdown(wait=True,cancel_futures=True)  # No worker can write after the lock is released.
        progress.close()  # No worker should write after the coordinator releases the collection lock.

def main(argv=None):  # Supply one command that discovers, extracts, resumes, and reports.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # Put the operating contract and example directly in --help.
    parser.add_argument('--estimate-only',action='store_true',help='Inspect the whole input folder and estimate time without running OCR or changing output files.')  # Offer a read-only preflight before committing CPU time.
    parser.add_argument('--seconds-per-page',type=float,help='Representative measured worker-seconds per page for your machine; otherwise reuse page timings from this output.')  # State calibration explicitly rather than promising a generic throughput.
    parser.add_argument('input_dir',type=Path,help='Folder of source PDFs, searched recursively.')  # Accept the business collection rather than a list of files.
    parser.add_argument('--output','--output-root',type=Path,default=Path('outputs/final'),help='New/empty folder or an existing output from this script. Repeat the command to resume.')  # Use one stable destination for future additions.
    parser.add_argument('--workers',type=int,default=default_workers(),help='Parallel PDF workers (default: physical cores minus one, capped by affinity). Each loads its own models.')  # Let operators control memory use and throughput.
    parser.add_argument('--engine',choices=('rapid-onnx','rapid-paddle'),default=ENGINE)  # Offer the two supported engines with independent word observations.
    parser.add_argument('--models',type=Path,default=ROOT/'models',help='Installed project model directory.')  # Allow standalone placement without hardcoded project imports.
    parser.add_argument('--word-reference-csv',type=Path,help='Reviewed word sample: source_sha256,page,word,x0,y0,x1,y1; top-left PDF points on the original displayed page.')  # Compare word geometry only where independent labels exist.
    parser.add_argument('--reference-csv',type=Path,help='UTF-8 source_sha256,page,text; one full reviewed transcription per annotated source page.')  # Require independent evidence before reporting precision or recall.
    args = parser.parse_args(argv)  # Reject invalid options before changing any output.
    source, root, args.models = args.input_dir.resolve(), args.output.resolve(), args.models.resolve()  # Resolve paths once for consistent version and provenance keys.
    if not source.is_dir() or (not args.estimate_only and not args.models.is_dir()) or args.workers < 1 or root.is_relative_to(source) or (args.seconds_per_page is not None and (not math.isfinite(args.seconds_per_page) or args.seconds_per_page<=0)):  # Guard against recursive ingestion or missing processing resources.
        parser.error('Existing input and models folders, positive --workers, and an output outside input are required.')  # Prevent accidental writes into the original dataset.
    if args.estimate_only:  # Preflight must not create, lock or modify any files.
        journal = root/'_state'/'collection.sqlite3'  # Only existing measurements may calibrate the forecast.
        rate = args.seconds_per_page  # Explicit calibration takes priority.
        if rate is None and journal.is_file():  # Avoid opening or creating a collection just to estimate it.
            with sqlite3.connect(journal.as_uri()+'?mode=ro',uri=True) as history:  # Read previously measured durations without modifying the output.
                rate = historical_rate(history,args.engine)  # The printed rate is heuristic, not proof of future throughput.
        forecast = Forecast(args.workers,rate)  # Estimate the whole input folder, including previously completed versions.
        for path in source.rglob('*'):  # Mirror the real run's recursive input selection.
            if path.suffix.lower()=='.pdf' and path.is_file():  # Ignore other business records in the folder.
                forecast.add(path)  # Inspect the page tree without loading OCR.
        forecast.report('all files, including previously completed')  # Distinguish this capacity estimate from a resumed run's remaining work.
        return 0  # No models, new output directory or OCR work were required.
    wall_start = time.perf_counter()  # Capture total run time including discovery, model setup and final publication.
    reviewed = references(args.reference_csv)  # Validate annotation shape before opening shared files.
    add_box_references(reviewed,args.word_reference_csv)  # Load the optional reviewed word-location sample.
    configure_worker_threads()  # Set conservative inference parallelism before importing model libraries.
    config = execution_config(args.engine,args.models,3.0)  # Identify model files, software settings, and the single script once per run.
    cfg = fingerprint(config)  # Isolate changed model or software versions without long path components.
    root.mkdir(parents=True,exist_ok=True)  # Create the collection folder without modifying any input PDF.
    private = root/'_state'  # Keep all operational data apart from the three user-facing outputs.
    if not private.is_dir() and any(root.iterdir()):  # A new invocation must not adopt unrelated existing outputs.
        parser.error('Use a new/empty output folder, or resume an output created by this script.')  # Refuse to overwrite pre-existing user files.
    private.mkdir(exist_ok=True)  # Reserve shared operational state beneath the output folder.
    with acquire_run_lock(private):  # A second launching process cannot interleave writes to this collection.
        journal = private/'collection.sqlite3'  # Keep the authority for recovering the CSVs.
        if not journal.exists() and any(p.name != '_state' for p in root.iterdir()):  # A missing journal cannot authorize overwriting existing public files.
            parser.error('Output contains files but no collection journal; choose a new folder.')  # Protect data with no known ownership.
        with sqlite3.connect(journal) as db:  # Restrict journal writes to the process that owns the output lock.
            db.executescript('CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS documents(seq INTEGER PRIMARY KEY,id TEXT UNIQUE,config TEXT,manifest TEXT,stage TEXT); CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,id TEXT,status TEXT,manifest TEXT); CREATE INDEX IF NOT EXISTS event_source ON events(id,seq); CREATE TABLE IF NOT EXISTS active(id TEXT PRIMARY KEY,config TEXT,job TEXT);')  # Record one durable package per document identity.
            with db:  # Create initial settings without disturbing an existing journal.
                db.executemany('INSERT OR IGNORE INTO settings VALUES(?,?)',[('dirty','1'),('input',str(source)),('schema','4')])  # Bind this output to its original input folder and projection format.
            settings = dict(db.execute('SELECT key,value FROM settings'))  # Read the binding and the previous publication outcome.
            if settings['input'] != str(source) or settings['schema'] not in ('3','4'):  # Avoid mixing unrelated collections or incompatible schemas.
                parser.error('Input directory or output schema differs from the original collection.')  # Require a separate output for a different collection.
            db.executescript('CREATE TABLE IF NOT EXISTS forecasts(id TEXT PRIMARY KEY,pages INTEGER,seconds REAL); CREATE TABLE IF NOT EXISTS timings(id TEXT PRIMARY KEY,engine TEXT,pages INTEGER,seconds REAL);')  # Persist forecasts and original page timings for later calibration.
            if settings['schema']=='3':  # Preserve earlier outcomes and expand report columns without requiring new output folders.
                with db:  # Recovery must not append rows using mismatched CSV headers.
                    db.execute("UPDATE settings SET value='4' WHERE key='schema'")  # Adopt the new export contract.
                    db.execute("UPDATE settings SET value='1' WHERE key='dirty'")  # Rebuild the CSV pair from existing journal evidence.
                settings['dirty']='1'  # Ensure this invocation immediately translates earlier reports.
            if settings['dirty'] == '1' or not all((root/n).is_file() for n in ('manifest.csv','metadata.csv')):  # Recover from interruption, missing snapshots, or an empty collection.
                project(db,root)  # Rebuild from committed staging files without rerunning OCR.
            interrupted(db,root)  # Close out assignments orphaned by a previous abrupt exit before retrying them.
            jobs = schedule(db,root,source,args,reviewed,cfg)  # Discover new and retryable documents lazily.
            try:  # Keep interrupted work distinguishable from final results.
                run_pool(jobs,db,root,args.workers,config)  # Extract in parallel while publishing reports serially.
            except (KeyboardInterrupt, BrokenProcessPool):  # The pool first stops all active workers and synchronizes their exit.
                if dict(db.execute('SELECT key,value FROM settings')).get('dirty') == '1':  # An interrupt may have landed during publication.
                    project(db,root)  # Restore all acknowledged CSV rows before recording interrupted assignments.
                interrupted(db,root)  # Expose partial page evidence and mark every interrupted assignment.
                raise  # Preserve the interruption or worker-failure exit status.
            failed = sum(json.loads(m)['status'] in ('failed','partial_failure') for m, in db.execute('SELECT manifest FROM documents'))  # Surface any unresolved document failure to batch schedulers.
            print(f"Manifest: {root/'manifest.csv'}\nWord metadata: {root/'metadata.csv'}\nFailed or partial documents: {failed}",flush=True)  # Show exactly where to review the collection.
            print(f'Actual collection wall time: {time.perf_counter()-wall_start:.1f} seconds.',flush=True)  # Include real setup, scheduling, and publication costs.
            return int(bool(failed))  # Unverified results are reviewable; actual processing failures return a nonzero exit code.

if __name__ == '__main__':  # Importing the file in worker processes must never start another coordinator.
    try:  # Keep interruption clear for interactive and automated operators.
        raise SystemExit(main())  # Propagate the collection outcome to the calling process.
    except KeyboardInterrupt:  # A deliberate stop does not erase durable checkpoints.
        print('Interrupted. Repeat the same command to resume.',file=sys.stderr)  # Tell the operator how to continue safely.
        raise SystemExit(130)  # Return the conventional interruption exit status.
