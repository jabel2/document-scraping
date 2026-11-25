# PDF extraction workflow

## Business purpose

This program turns a folder of PDFs into readable text and records where the extracted information came from. It is intended for collections containing both ordinary text-based PDFs and scanned images, including collections where the type is not known in advance. Each extract can be traced back to its original file and page; word-level records include source-page locations where available.

The workflow prioritizes extraction quality while supporting multiple CPU workers and restartable batch processing. The current implementation is a proof of concept: it does not guarantee accuracy or a particular daily throughput. Review important text, tables, and figures against the source documents.

## What the program does

1. Finds PDFs in the input folder and its subfolders without changing the originals.
2. Examines each page to choose a native-text or scan-processing route. Suspect hidden text layers can be routed to OCR.
3. Assesses page orientation before layout extraction and applies a supported rotation where appropriate. Uncertain cases are flagged for review.
4. Uses Docling and, for scanned pages, RapidOCR with the ONNX backend by default. A separate word-level OCR pass supplies word fragments, locations, and raw model confidence scores. Models are reused within each worker.
5. Compares line and word observations and makes only narrowly supported alphabetic spacing corrections. Ambiguous text, numbers, table structure, and possible cross-page relationships still require review.
6. Publishes readable page text, document status, and word metadata as processing progresses. It keeps checkpoints so an interrupted collection can be resumed.

PDFs can contain complex layouts, hidden text, images, tables, and citations. Classification and layout are heuristic, so the result may omit or misorder content. Word fragments may not cover every word, and a `complete` status does not mean a person has verified accuracy.

## Output structure

With `--output final-code\output`, the results are organized as follows:

| Location | Purpose |
| --- | --- |
| `output\<source>-extract-<id>.md` | One readable Markdown extract per PDF version, with page headings and any review notices. The filename uses the source PDF's name without its extension and an eight-character UUID suffix; use `extract_path` in the CSVs to find the matching file. |
| `output\manifest.csv` | A history of document status and page progress, linked to the source and extract. It also contains timing, page counts, review details, and reference-based summary metrics when available. |
| `output\metadata.csv` | The current word or OCR-fragment observations, including document ID, source, page, text, bounding box, provenance, and available confidence or reference-comparison fields. |
| `output\_state\` | Processing journal, checkpoints, and intermediate evidence used for recovery. Keep this folder with the output if the collection may be resumed or extended. |

The manifest is an **event history**: a document can have several rows. For its current status, find the row with the highest `sequence` for that `document_id`. Statuses include `pending`, `processing`, `complete`, `needs_review`, `partial_failure`, `failed`, and `interrupted`. The `detail` column explains specific issues where available. `needs_review` means the extract is available but one or more issues merit inspection.

`document_id`, `source_path`, `source_sha256`, and `extract_path` link the outputs. Different source paths or changed PDF contents produce distinct document identities. Extract paths are relative to the output folder. Metadata is maintained as the current observation set for each document; recovery or a changed processing configuration can cause it to be rebuilt rather than simply appended. Avoid editing generated files in place, since recovery can replace them. If a spreadsheet program prevents CSV updates on Windows, close the file while processing continues.

The UUID suffix is stable for a document version, so retries use the same filename. Extracts created by an earlier version of the script are not automatically moved or renamed; their recorded `extract_path` remains the way to find them.

Bounding boxes use PDF points with the origin at the **top-left of the displayed, cropped original page**. A raw OCR confidence is a model score, not a verified probability that a word is correct. Native PDF words generally have no OCR confidence. Blank metrics mean unavailable, not zero.

## System setup

These instructions use Windows Command Prompt and assume the complete project is present, including `pyproject.toml`, `uv.lock`, and `final-code\extract_pdfs.py`. The script is self-contained application code, but it still needs the installed Python dependencies, model files, and Tesseract described below. The examples run from the **project root**, the directory containing `pyproject.toml`.

### 1. Install Python and uv

Install Python **3.12** and [uv](https://docs.astral.sh/uv/). Open a new Command Prompt and check:

```cmd
uv --version
py -3.12 --version
```

From the project root, install the locked dependencies into the project's `.venv`:

```cmd
uv sync --locked --python 3.12
```

If Python 3.12 is not discoverable by `uv`, provide its full executable path, for example `uv sync --locked --python C:\Python312\python.exe`. The resolved package versions are recorded in `uv.lock`; later examples use `--no-sync` to use this prepared environment.

### 2. Install Tesseract and its language data

Install the Tesseract executable separately. It is used for page orientation detection; the default text-recognition engine is RapidOCR ONNX. Ensure the installation includes the English (`eng`) and orientation/script detection (`osd`) trained data. Verify:

```cmd
tesseract --version
tesseract --list-langs
```

If Tesseract is installed but is not on `PATH`, start a new terminal after updating `PATH`, or set its full path for the current Command Prompt session before running the script:

```cmd
set "TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
"%TESSERACT_CMD%" --list-langs
```

The script also checks for the language files under the Tesseract installation's `tessdata` folder, or under `TESSDATA_PREFIX` if set. Use the actual location on your machine.

### 3. Download the models

From the project root, download the layout, table, and English RapidOCR ONNX artifacts into `models\`:

```cmd
uv run --no-sync docling-tools models download --output-dir models layout tableformer rapidocr --rapidocr-backend-lang onnxruntime:en
```

The initial download requires network access and disk space. Existing project model files can be reused. To use a different model directory, pass `--models "C:\path\to\models"` to the script. The optional `--engine rapid-paddle` backend requires its corresponding RapidOCR Paddle model artifacts as well; the default and examples below use ONNX.

### 4. Prepare storage and compute

Create `final-code\scans` if it does not exist, then place the source PDFs there. Allocate space for the input, readable extracts, CSVs, and potentially substantial intermediate evidence in `output\_state`. The input and output folders must be separate; the output must be a new or empty folder, or an existing collection created by this script.

The program runs on CPU. By default it requests the number of physical CPU cores minus one, subject to CPU affinity, with a minimum of one. Each worker loads its own models and processes one PDF at a time, so more workers require more memory; pages within a single PDF are processed sequentially. Start with a smaller explicit worker count if machine capacity is uncertain.

## Run the program

Open Command Prompt in the project root. To inspect the input count and page count without processing or modifying output files:

```cmd
uv run --no-sync python final-code\extract_pdfs.py final-code\scans --output final-code\output --workers 4 --estimate-only
```

If the output has no prior timing history, the estimated duration is shown as unknown. To supply a representative measured rate, add `--seconds-per-page 12`, replacing `12` with measured **worker-seconds per page** for a similar workload and machine. This is a rough planning estimate, not a deadline; model startup, I/O, retries, hardware, and page complexity affect elapsed time. Estimate-only covers all input files, including previously completed files. A real run estimates queued files.

Run the extraction:

```cmd
uv run --no-sync python final-code\extract_pdfs.py final-code\scans --output final-code\output --workers 4
```

Adjust `--workers 4` to suit the machine. The console shows progress and final elapsed time. Open `final-code\output\manifest.csv` to monitor status and use its `extract_path` to find each readable extract. Page-level text and metadata can appear before a multi-page document finishes; treat a `processing` document as incomplete.

To resume after an interruption or process newly added PDFs, run the same command again with the same input and output folders. Keep `_state` intact. Already completed equivalent work is reused; failed or interrupted work can be retried. Only one coordinator can write to the same output collection at a time. A different model or processing configuration can require refreshed results.

For all options:

```cmd
uv run --no-sync python final-code\extract_pdfs.py --help
```

## Optional accuracy checks

The extraction does not know the correct answer for an arbitrary PDF. For measured text overlap, supply `--reference-csv` with UTF-8 columns `source_sha256,page,text`, containing a complete independently reviewed transcription for each annotated page. Reported precision and recall are page-level, case-insensitive normalized whitespace-token overlaps; they ignore word order and bounding-box geometry. They are not per-word confidence or complete-document accuracy unless the relevant pages were reviewed.

For reviewed word locations, supply `--word-reference-csv` with columns `source_sha256,page,word,x0,y0,x1,y1`, using the original displayed, cropped page's top-left PDF-point coordinates. The output reports matched-box intersection-over-union (`box_iou`) and text agreement for the reviewed sample; manifest counts include the number of reviewed boxes and how many met the IoU threshold. These results describe the supplied labels only. Reference fields stay blank when no applicable references were provided.
