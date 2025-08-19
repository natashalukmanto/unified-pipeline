"""
Unified Pipeline API (FastAPI)
- Ingests a benefits PDF (or other doc → PDF), converts to Markdown via Reducto,
  then runs these LLM phases via OpenRouter:
    1) Classification (pass 1) — carrier candidates, LOCs, page ranges
    2) Plan Name Identification — uses separate prompt with pass-1 output + document
    3) Parameter extraction per plan — uses a different Vellum prompt per Line of Coverage (LOC)
- Prompts are fetched from Vellum (**REQUIRED by spec**). OpenRouter handles model execution.
- All LLM raw outputs are written to .txt and zipped; final structured output returned as JSON.

Requirements (install):
  pip install fastapi uvicorn pydantic requests python-multipart

Environment variables (set these):
  OPENROUTER_API_KEY=...
  VELLUM_API_KEY=...
  VELLUM_BASE_URL=https://api.vellum.ai
  CLOUDCONVERT_API_KEY=...
  REDUCTO_API_KEY=...
  REDUCTO_BASE_URL=https://api.reducto.ai
  MODEL_ID=anthropic/claude-3.5-sonnet

Run:
  uvicorn main:app --reload

Test:
  curl -X POST "http://127.0.0.1:8000/process" -F "file=@/path/to/benefits_packet.pdf"

Notes:
- CloudConvert integration is provided but optional. If missing API key and file is not PDF, request will error.
- Reducto call uploads the PDF as multipart/form-data to /parse to get Markdown back.
- OpenRouter requests are structured to be stable (same system+user templates, temp=0) to encourage prompt caching.
- To serve artifacts locally, this app exposes /files/{job_id}/{filename}.
"""
from __future__ import annotations

import os
import re
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import threading
import time
import json
import requests
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
from dotenv import load_dotenv

# load .env from the project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# -----------------------------
# Config & constants
# -----------------------------
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/tmp/unified_pipeline"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_ID = os.environ.get("MODEL_ID", "anthropic/claude-3.5-sonnet")

VELLUM_API_KEY = os.environ.get("VELLUM_API_KEY")
VELLUM_BASE_URL = os.environ.get("VELLUM_BASE_URL", "https://api.vellum.ai")

CLOUDCONVERT_API_KEY = os.environ.get("CLOUDCONVERT_API_KEY")

REDUCTO_API_KEY = os.environ.get("REDUCTO_API_KEY")
REDUCTO_BASE_URL = os.environ.get("REDUCTO_BASE_URL", "https://api.reducto.ai")

# Used for OpenRouter best practices
HTTP_REFERER = os.environ.get("HTTP_REFERER", "http://localhost")
X_TITLE = os.environ.get("X_TITLE", "Unified Pipeline")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Unified Pipeline API", version="1.0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class ProcessResponse(BaseModel):
    job_id: str
    classification_output: str = Field(
        serialization_alias="Classification step output"
    )
    plan_identification_output: str = Field(
        serialization_alias="Plan identification output"
    )
    # NEW:
    classification_prompt_text: str = Field(
        serialization_alias="Classification prompt text"
    )
    plan_identification_prompt_text: str = Field(
        serialization_alias="Plan identification prompt text"
    )
    json_result: Dict[str, Any]

# -----------------------------
# Utilities
# -----------------------------
def write_text(job_dir: Path, name: str, content: str) -> Path:
    p = job_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def zip_dir(job_dir: Path, zip_name: str = "artifacts.zip") -> Path:
    zip_path = job_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in job_dir.iterdir():
            if p.name != zip_name:
                z.write(p, arcname=p.name)
    return zip_path


def slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s)[:120]

# -----------------------------
# CloudConvert (optional) - non-PDF → PDF
# -----------------------------
def ensure_pdf(src_path: Path, job_dir: Path) -> Path:
    """If file is PDF, return it. If not, try CloudConvert; else raise HTTP 400."""
    if src_path.suffix.lower() == ".pdf":
        return src_path
    if not CLOUDCONVERT_API_KEY:
        raise HTTPException(status_code=400, detail="Non-PDF upload requires CLOUDCONVERT_API_KEY or upload a PDF.")

    headers = {"Authorization": f"Bearer {CLOUDCONVERT_API_KEY}", "Content-Type": "application/json"}
    create_job_payload = {
        "tasks": {
            "import-1": {"operation": "import/upload"},
            "convert-1": {"operation": "convert", "input": "import-1", "output_format": "pdf"},
            "export-1": {"operation": "export/url", "input": "convert-1"}
        }
    }
    job_resp = requests.post("https://api.cloudconvert.com/v2/jobs", headers=headers, json=create_job_payload)
    if job_resp.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"CloudConvert job create failed: {job_resp.text}")
    job = job_resp.json()["data"]

    import_task = next(t for t in job["tasks"] if t["name"] == "import-1")
    upload_url = import_task["result"]["form"]["url"]
    form_params = import_task["result"]["form"]["parameters"]

    files = {"file": (src_path.name, src_path.open("rb"))}
    form_data = form_params
    upload_resp = requests.post(upload_url, data=form_data, files=files)
    if upload_resp.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"CloudConvert upload failed: {upload_resp.text}")

    job_id = job["id"]
    for _ in range(30):
        jr = requests.get(f"https://api.cloudconvert.com/v2/jobs/{job_id}", headers=headers)
        jr.raise_for_status()
        j = jr.json()["data"]
        if j["status"] == "finished":
            export_task = next(t for t in j["tasks"] if t["name"] == "export-1")
            file_url = export_task["result"]["files"][0]["url"]
            pdf_path = job_dir / "converted.pdf"
            pdf_bin = requests.get(file_url).content
            pdf_path.write_bytes(pdf_bin)
            return pdf_path
        elif j["status"] in {"error", "failed"}:
            raise HTTPException(status_code=502, detail=f"CloudConvert job failed: {j}")

    raise HTTPException(status_code=504, detail="CloudConvert job timed out.")


# -----------------------------
# Reducto: PDF → Markdown
# -----------------------------
def _download_text_from_url(url: str, timeout: int = 180) -> str:
    r = requests.get(url, timeout=timeout)
    if r.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download text from {url}: {r.status_code} {r.text[:200]}"
        )
    # If the signed URL returns JSON with text fields, use them; else raw body.
    try:
        j = r.json()
        for k in ("markdown", "text", "content"):
            v = j.get(k)
            if isinstance(v, str) and v.strip():
                return v
    except Exception:
        pass
    return r.text


def reducto_result_to_text(pj: dict) -> Optional[str]:
    """
    Normalize Reducto's structured 'result.chunks' into a single text blob.
    Prefers plain 'embed' (clean text), falls back to 'content'.
    """
    res = pj.get("result") or (pj.get("data") or {}).get("result")
    if not isinstance(res, dict):
        return None
    chunks = res.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return None

    parts: list[str] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        piece = ch.get("embed") or ch.get("content")
        if isinstance(piece, str) and piece.strip():
            parts.append(piece.strip())

    if not parts:
        return None
    # Separate sections to keep the LLM context readable.
    return "\n\n---\n\n".join(parts)

def extract_text_from_reducto_payload(pj: dict) -> Optional[str]:
    """
    Try all common places Reducto may put text, then fall back to result.chunks,
    then to downloadable URLs.
    """
    # 1) Inline
    for k in ("markdown", "text", "content"):
        v = pj.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 2) Nested (data.*)
    data = pj.get("data")
    if isinstance(data, dict):
        for k in ("markdown", "text", "content"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v

    # 3) Structured chunks
    v = reducto_result_to_text(pj)
    if v:
        return v

    # 4) Downloadable URLs
    url_keys = ("markdown_url", "text_url", "content_url", "md_url", "result_url")
    for k in url_keys:
        url = pj.get(k)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_text_from_url(url)
    if isinstance(data, dict):
        for k in url_keys:
            url = data.get(k)
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return _download_text_from_url(url)

    return None


def _poll_job_for_markdown(job_id: str, max_wait_s: int = 180) -> Optional[str]:
    """
    Poll a few likely job endpoints until text materializes.
    Different tenants expose different paths; we try several.
    """
    headers = {"Authorization": f"Bearer {REDUCTO_API_KEY}"}
    base = REDUCTO_BASE_URL
    candidates = [
        f"{base}/jobs/{job_id}",
        f"{base}/job/{job_id}",
        f"{base}/parse/jobs/{job_id}",
        f"{base}/results/{job_id}",
    ]
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        for url in candidates:
            try:
                r = requests.get(url, headers=headers, timeout=20)
            except requests.exceptions.RequestException:
                continue
            if r.status_code >= 300:
                continue
            try:
                j = r.json()
            except Exception:
                if r.text.strip():
                    return r.text
                continue

            txt = extract_text_from_reducto_payload(j)
            if txt:
                return txt

            status = (j.get("status") or (j.get("data") or {}).get("status") or "").lower()
            if status in {"queued", "processing", "running", "in_progress"}:
                continue
        time.sleep(2)
    return None


def pdf_to_markdown(pdf_path: Path) -> str:
    """
    Robust Reducto flow:
      1) POST /upload  -> returns document_url or file_id (reducto://...)
      2) POST /parse   -> asks for markdown/text; normalize any response shape
      3) If needed, poll job endpoints or try /extract
    Returns a single large text string ready for LLM input.
    """
    if not (REDUCTO_API_KEY and REDUCTO_BASE_URL):
        raise HTTPException(status_code=500, detail="Reducto not configured (REDUCTO_API_KEY/REDUCTO_BASE_URL).")

    # --- Step 1: upload ---
    upload_url = f"{REDUCTO_BASE_URL}/upload"
    up_headers = {"Authorization": f"Bearer {REDUCTO_API_KEY}"}
    files = {"file": (pdf_path.name, pdf_path.open("rb"), "application/pdf")}
    try:
        up_resp = requests.post(upload_url, headers=up_headers, files=files, timeout=180)
    finally:
        files["file"][1].close()

    if up_resp.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Reducto upload failed ({up_resp.status_code}) at {upload_url}: {up_resp.text}",
        )

    try:
        up_json = up_resp.json()
    except Exception:
        up_json = {}

    # Prefer document_url; accept file_id (reducto://...pdf) as a handle.
    document_url = (
        up_json.get("document_url")
        or up_json.get("url")
        or (up_json.get("data") or {}).get("document_url")
        or up_json.get("file_id")
    )
    if not document_url:
        raise HTTPException(
            status_code=502,
            detail=f"Reducto upload succeeded but no document_url found. Body: {up_resp.text[:300]}",
        )

    # --- Step 2: parse ---
    parse_url = f"{REDUCTO_BASE_URL}/parse"
    parse_headers = {
        "Authorization": f"Bearer {REDUCTO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
    "document_url": document_url,
    "options": {
        "force_url_result": False,
        "ocr_mode": "standard",
        "extraction_mode": "ocr",
        "chunking": {"chunk_mode": "page"},
        "table_summary": {"enabled": False},
        "figure_summary": {"enabled": False}
    },
    "advanced_options": {
        "enable_change_tracking": False,
        "ocr_system": "highres",
        "table_output_format": "html",
        "merge_tables": False,
        "include_color_information": False,
        "continue_hierarchy": False,
        "keep_line_breaks": True,
        "large_table_chunking": {"enabled": False},
        "add_page_markers": True,
        "exclude_hidden_sheets": True,
        "exclude_hidden_rows_cols": True
    },
    "experimental_options": {
        "danger_filter_wide_boxes": False,
        "rotate_pages": True,
        "enable_scripts": True
    },
    "priority": True
}
    try:
        p_resp = requests.post(parse_url, headers=parse_headers, json=payload, timeout=300)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Reducto request to {parse_url} failed: {e}")

    if p_resp.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=f"Reducto parse failed: {p_resp.status_code} — {p_resp.text}",
        )

    # Normalize response into a single string
    try:
        pj = p_resp.json()
    except Exception:
        return p_resp.text

    txt = extract_text_from_reducto_payload(pj)
    if txt:
        return txt

    job_id = pj.get("job_id") or (pj.get("data") or {}).get("job_id")
    if job_id:
        txt2 = _poll_job_for_markdown(job_id, max_wait_s=180)
        if txt2:
            return txt2

    # --- Step 3: fallback to /extract if tenant requires it ---
    try:
        extract_url = f"{REDUCTO_BASE_URL}/extract"
        ex_headers = {"Authorization": f"Bearer {REDUCTO_API_KEY}", "Content-Type": "application/json"}
        ex_payload = {
            "document_url": document_url,
            "output": {"format": "markdown"},
            "priority": True
        }
        ex_resp = requests.post(extract_url, headers=ex_headers, json=ex_payload, timeout=300)
        if ex_resp.status_code < 300:
            try:
                exj = ex_resp.json()
            except Exception:
                return ex_resp.text
            txt3 = extract_text_from_reducto_payload(exj)
            if txt3:
                return txt3
    except Exception:
        pass

    raise HTTPException(
        status_code=502,
        detail=f"Reducto parse returned no markdown/text. Body: {str(pj)[:400]}",
    )

# -----------------------------
# Vellum: prompt retrieval (REQUIRED)
# -----------------------------
_PROMPT_CACHE: Dict[str, str] = {}

def _join_vellum_message_text(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract plain prompt text from Vellum provider-payload response:
      payload -> messages[] -> content[] -> {type: "text", text: "..."}
    If multiple text parts exist, join them with newlines.
    """
    try:
        msgs = payload.get("messages") or []
        if not msgs:
            return None
        # Usually the prompt template is in the first user message
        contents = msgs[0].get("content") or []
        parts: List[str] = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        if parts:
            return "\n".join(parts)
        return None
    except Exception:
        return None


def get_prompt_from_vellum(deployment_name: str, version: Optional[str] = None) -> Optional[str]:
    """
    Fetches prompt text for a *deployed* Vellum prompt using the provider-payload endpoint.
    This does NOT execute the prompt; it only returns the provider payload (including the
    static compiled text template). We extract the prompt text from messages[].content[].text.
    """
    if not VELLUM_API_KEY:
        raise HTTPException(status_code=500, detail="VELLUM_API_KEY not set.")

    base = (VELLUM_BASE_URL or "https://api.vellum.ai").rstrip("/")
    url = f"{base}/v1/deployments/provider-payload"

    headers = {
        # Vellum's deployments API uses X-API-KEY (not Authorization: Bearer)
        "X-API-KEY": VELLUM_API_KEY,
        "Content-Type": "application/json",
    }

    # Many deployments don't require inputs to build the provider payload.
    # If your deployment *does* declare required vars, pass dummies here.
    body = {
        "deployment_name": deployment_name,
        # Provide an empty list by default. If your deployment needs a variable, add it here.
        "inputs": []
    }
    # Optional: If you version your deployments, you can pass it through:
    if version:
        body["tag"] = version  # Vellum may treat this as a release tag; harmless if unused.

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Vellum provider-payload request failed: {e}")

    if resp.status_code >= 300:
        # Some deployments require at least one variable; retry with a dummy if 400 tells us.
        if resp.status_code == 400:
            try:
                j = resp.json()
            except Exception:
                j = {}
            # try a second attempt with a placeholder input if required by your prompt
            body_retry = dict(body)
            body_retry["inputs"] = [{"type": "STRING", "name": "document", "value": ""}]
            try:
                resp2 = requests.post(url, json=body_retry, headers=headers, timeout=30)
                if 200 <= resp2.status_code < 300:
                    j2 = resp2.json()
                    payload = (j2 or {}).get("payload") or {}
                    txt = _join_vellum_message_text(payload)
                    if txt:
                        return txt
            except requests.exceptions.RequestException:
                pass

        raise HTTPException(
            status_code=502,
            detail=f"Vellum prompt fetch failed for '{deployment_name}': {resp.status_code} {resp.text[:300]}",
        )

    data = resp.json()
    payload = (data or {}).get("payload") or {}
    text = _join_vellum_message_text(payload)
    if text:
        return text

    # Fallback: sometimes prompt text sits under different keys—try a few quick ones
    for k in ("compiled_prompt", "prompt", "template", "text", "content", "body"):
        v = (data.get(k) or payload.get(k))
        if isinstance(v, str) and v.strip():
            return v

    raise HTTPException(
        status_code=502,
        detail=f"Vellum provider-payload returned no text for '{deployment_name}'. Body keys: {list(data.keys())}",
    )


def get_prompt(slug: str, version: Optional[str] = None) -> str:
    """
    Cached accessor that uses provider-payload by deployment name (slug).
    """
    key = f"{slug}:{version or 'latest'}"
    cached = _PROMPT_CACHE.get(key)
    if cached:
        return cached
    remote = get_prompt_from_vellum(slug, version)
    if not remote:
        raise HTTPException(status_code=500, detail=f"Missing prompt '{slug}' from Vellum.")
    _PROMPT_CACHE[key] = remote
    return remote

# -----------------------------
# OpenRouter client
# -----------------------------
def openrouter_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": X_TITLE,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or MODEL_ID,
        "messages": messages,
        # "reasoning": {
        #     "max_tokens": 1024,
        #     "enabled": True,
        #     "exclude": False
        # },
        "max_output_tokens": 64000,
        "temperature": 1,
        "top_p": 0.999,
        "top_k": 250,
        "extra_body": {"cache": True},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {r.text}")
    data = r.json()
    try:
        choice = data["choices"][0]
        print("finish_reason:", choice.get("finish_reason"), "stop_reason:", choice.get("stop_reason"))
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Malformed OpenRouter response: {data}")


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_classification(raw: str) -> Dict[str, Any]:
    """
    Expected signals from classification output:
      - Basic Info::Carrier Candidates::[Aetna, Kaiser]
      - Basic Info::Page Numbers::<LOC>::[1,3,4]
    """
    carriers: List[str] = []
    loc_pages: Dict[str, List[int]] = {}

    # Carriers
    m = re.search(r"Basic Info::Carrier(?: Candidates)?::\[(.*?)\]", raw, re.IGNORECASE)
    if m:
        carriers = [c.strip() for c in m.group(1).split(",") if c.strip()]

    # Page numbers per LOC
    for loc in [
        "Medical", "Dental", "Vision", "STD", "LTD", "LifeADD", "VLifeADD",
        "VL", "VA", "VCI", "VHI", "Other"
    ]:
        pattern = rf"Basic Info::Page Numbers::{re.escape(loc)}::\[(.*?)\]"
        mm = re.search(pattern, raw)
        if mm:
            nums = [int(n.strip()) for n in mm.group(1).split(",") if n.strip().isdigit()]
            if nums:
                loc_pages[loc] = nums

    return {"carriers": carriers, "loc_pages": loc_pages}


def parse_plan_names(raw: str) -> Dict[str, List[str]]:
    """
    Parse plan lines like:
      1::Plans::Medical::Aetna PPO 250::$0.00::2
    Returns: { LOC: [plan_name, ...], ... }
    """
    plans_by_loc: Dict[str, List[str]] = {}
    for line in raw.splitlines():
        m = re.match(r"\d+::Plans::([A-Za-z]+)::(.+?)::\$0\.00::\d+", line.strip())
        if m:
            loc = m.group(1).strip()
            plan = m.group(2).strip()
            plans_by_loc.setdefault(loc, []).append(plan)
    return plans_by_loc

# -----------------------------
# Pipeline stages
# -----------------------------
def run_classification(md_text: str, job_dir: Path) -> tuple[str, Dict[str, Any], str]:
    """
    Step 3–5: fetch classification prompt; inject <Document>{md_text}</Document>;
    send to OpenRouter; return (raw_output, used_prompt).
    """
    # 1) Fetch the base prompt from Vellum
    base_prompt = get_prompt("classify-document-and-identify-carrier-loc-and-plan-names-v-1-0-variant-1")
    used_prompt = base_prompt.replace("<Document></Document>", f"<Document>{md_text}</Document>", 1)

    system = (
        "You are a precise, deterministic parser. Output exactly the requested key::value lines, "
        "no commentary, no markdown."
    )

    raw_output = openrouter_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": used_prompt},
    ])

    write_text(job_dir, "classification.txt", raw_output)
    write_text(job_dir, "classification_prompt.txt", used_prompt)

    return raw_output, parse_classification(raw_output), used_prompt

def extract_loc_from_classification(classification_output: str) -> list[str]:
    """
    Extract the LOC list from classification output.
    Example line looks like:
      Basic Info::Line of Coverage::[Medical, Dental, Vision, STD, LTD, LifeADD, VLifeADD, VL, VA, VCI, VHI, Other]
    Returns a list without 'Other'.
    """
    m = re.search(r"Basic Info::Line of Coverage::\[(.*?)\]", classification_output or "")
    if not m:
        return []
    locs = [x.strip() for x in m.group(1).split(",")]
    return [x for x in locs if x.lower() != "other"]

def run_plan_identification(md_text: str, previous_output: str, job_dir: Path) -> tuple[str, Dict[str, List[str]], str]:
    """Step 6–7: fetch plan-ident prompt; send prompt + classification output + markdown; return raw + plans + prompt_text."""
    base_prompt = get_prompt("plan-name-identification-prompt-v-18-0-variant-1")
    used_prompt = base_prompt.replace("<Document></Document>", f"<Document>{md_text}</Document>", 1)
    print(md_text)
    used_prompt = used_prompt.replace("<Classification Output></Classification Output>", f"<Classification Output>{previous_output}</Classification Output>", 1)

    locs = extract_loc_from_classification(previous_output)
    loc_str = ", ".join(locs)

    used_prompt = used_prompt.replace("{{LOC}}", loc_str)

    system = "You are a precise, deterministic parser. Output on...ly <index>::Plans::<LOC>::<Plan Name>::$0.00::<page_ref> lines."

    raw_output = openrouter_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": used_prompt},
    ])

    write_text(job_dir, "plan_identification.txt", raw_output)
    write_text(job_dir, "plan_identification_prompt.txt", used_prompt)
    return raw_output, parse_plan_names(raw_output), used_prompt

# Map LOC → Vellum prompt slug for extraction
PROMPT_MAP: Dict[str, str] = {
    "Medical": "medical-unified-refiner-v-5-0-variant-1",
    "Dental": "dental-unified-refiner-v-3-0-variant-1",
    "Vision": "vision-unified-refiner-v-3-0-variant-1",
    "LifeADD": "life-add-life-add-unified-refiner-v-2-0-variant-1",
    "STD": "std-unified-refiner-v-3-0-variant-1",
    "LTD": "ltd-unified-refiner-v-3-0-variant-1",
    "VLifeADD": "vol-life-and-add-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VSTD": "vol-std-generic-unified-refiner-reducto-v-8-0-variant-1",
    "VLTD": "vol-ltd-generic-unified-refiner-reducto-v-8-0-variant-1",
    "VA": "vol-accident-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VHI": "vol-hospital-indemnity-generic-unified-refiner-reducto-v-7-0-variant-1",
    "VCI": "vol-critical-illness-generic-unified-refiner-reducto-v-7-0-variant-1"
}   

def run_extraction(md_text: str, loc: str, plan_name: str, job_dir: Path) -> Dict[str, Any]:
    """Step 8: for each (LOC, plan), fetch per-LOC prompt and extract parameters."""
    prompt_slug = PROMPT_MAP.get(loc, "medical-unified-refiner-v-5-0-variant-1")
    prompt = get_prompt(prompt_slug)
    system = (
        "You are a precise, deterministic parser. Output only 'Category::Name::Value' lines. "
        "Include 'Source::Pages::<[list]>' if possible."
    )
    used_prompt = prompt.replace("<Document></Document>", f"<Document>{md_text}</Document>")
    used_prompt = used_prompt.replace("{{plan_name}}", plan_name)
    raw = openrouter_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": used_prompt},
    ])
    safe = f"extract_{slug(loc)}_{slug(plan_name)}.txt"
    write_text(job_dir, safe, raw)
    # Minimal normalizer (you can swap in your richer normalize_extraction later)
    params: Dict[str, Any] = {}
    source_pages: List[int] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "::" not in line:
            continue
        if line.lower().startswith("source::pages::"):
            nums = re.findall(r"\d+", line)
            source_pages = [int(n) for n in nums]
            continue
        parts = line.split("::")
        if len(parts) >= 3:
            cat, name, value = parts[0], parts[1], "::".join(parts[2:])
            params[f"{cat}::{name}"] = value
    return {
        "loc": loc,
        "plan_name": plan_name,
        "parameters": params,
        "source_pages": source_pages,
        "raw": raw,
    }

# ---------- Extraction concurrency helpers with verbose logging ----------
def _elog(job_dir: Path, msg: str) -> None:
    print(f"[EXTRACT] {msg}", flush=True)
    try:
        (job_dir / "extraction_debug.log").write_text(
            ((job_dir / "extraction_debug.log").read_text(encoding="utf-8", errors="ignore") if (job_dir / "extraction_debug.log").exists() else "")
            + f"{msg}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

def _progress(job_dir: Path, event: dict) -> None:
    try:
        with (job_dir / "extraction_progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass

def extraction_file_for(loc: str, plan_name: str, job_dir: Path) -> Path:
    return job_dir / f"extract_{slug(loc)}_{slug(plan_name)}.txt"

def run_extraction_scoped(md_text: str, loc: str, plan_name: str,
                          job_dir: Path, loc_pages: list[int] | None) -> dict:
    # (Keeping scope=nop to avoid changing your Reducto flow. If you later want page scoping,
    #  slice here and keep the log lines as-is.)
    return run_extraction(md_text, loc, plan_name, job_dir)

def run_all_extractions(
    md_text: str,
    plans_by_loc: dict[str, list[str]],
    loc_pages: dict[str, list[int]] | None,
    job_dir: Path,
    max_workers: int = 10,
) -> list[dict]:
    """Parallelize extraction with detailed progress logging."""
    # Build task list
    tasks_to_run: list[tuple[str, str, list[int] | None]] = []
    results: list[dict] = []
    skipped_count = 0

    _elog(job_dir, "Building extraction task list …")
    for loc, plan_list in (plans_by_loc or {}).items():
        unique_plans = sorted({(p or "").strip() for p in plan_list if (p or "").strip()})
        pages = (loc_pages or {}).get(loc)
        for plan_name in unique_plans:
            out_path = extraction_file_for(loc, plan_name, job_dir)
            if out_path.exists() and out_path.stat().st_size > 0:
                raw = out_path.read_text(encoding="utf-8", errors="ignore")
                results.append({"loc": loc, "plan_name": plan_name, "raw": raw})
                skipped_count += 1
                _elog(job_dir, f"SKIP existing: LOC={loc} plan='{plan_name}' file={out_path.name}")
                _progress(job_dir, {"type": "skip", "loc": loc, "plan": plan_name, "file": out_path.name})
            else:
                tasks_to_run.append((loc, plan_name, pages))

    total = len(tasks_to_run)
    _elog(job_dir, f"Extraction queue ready: to_run={total}, skipped={skipped_count}, total_plans={total + skipped_count}")

    if total == 0:
        _elog(job_dir, "No new extraction calls needed.")
        return results

    # Submit work
    in_flight = 0
    done = 0
    start_all = time.perf_counter()

    def _worker(idx: int, total: int, loc: str, plan_name: str, pages: list[int] | None):
        tname = threading.current_thread().name
        _elog(job_dir, f"START {idx}/{total} [{tname}] LOC={loc} plan='{plan_name}' pages={pages or 'all'}")
        _progress(job_dir, {"type": "start", "idx": idx, "total": total, "loc": loc, "plan": plan_name, "pages": pages})
        t0 = time.perf_counter()
        try:
            res = run_extraction_scoped(md_text, loc, plan_name, job_dir, pages)
            dt = time.perf_counter() - t0
            _elog(job_dir, f"DONE  {idx}/{total} [{tname}] LOC={loc} plan='{plan_name}' in {dt:.1f}s")
            _progress(job_dir, {"type": "done", "idx": idx, "loc": loc, "plan": plan_name, "secs": round(dt, 3)})
            return res
        except Exception as e:
            dt = time.perf_counter() - t0
            _elog(job_dir, f"ERROR {idx}/{total} [{tname}] LOC={loc} plan='{plan_name}' after {dt:.1f}s -> {type(e).__name__}: {e}")
            _progress(job_dir, {"type": "error", "idx": idx, "loc": loc, "plan": plan_name, "secs": round(dt, 3), "err": f"{type(e).__name__}: {e}"})
            # Re-raise so the caller records it to errors.log too
            raise

    # Queue + run
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, (loc, plan, pages) in enumerate(tasks_to_run, 1):
            futures.append(pool.submit(_worker, i, total, loc, plan, pages))
            in_flight += 1
            _elog(job_dir, f"QUEUED {i}/{total} inflight={in_flight}")

        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                write_text(job_dir, "errors.log", f"Extraction error: {type(e).__name__}: {e}\n")
            finally:
                done += 1
                in_flight -= 1
                _elog(job_dir, f"PROGRESS {done}/{total} complete; inflight={in_flight}")

    _elog(job_dir, f"All extraction tasks finished in {time.perf_counter() - start_all:.1f}s "
                   f"(ran={total}, skipped={skipped_count}, total_plans={total + skipped_count})")
    _progress(job_dir, {"type": "summary", "ran": total, "skipped": skipped_count,
                        "total_plans": total + skipped_count,
                        "secs": round(time.perf_counter() - start_all, 3)})
    return results

# ---------- /helpers ----------

# -----------------------------
# Routes
# -----------------------------
@app.post("/process", response_model=ProcessResponse)
def process_document(file: UploadFile = File(...)):
    # Validate env
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is required.")
    if not VELLUM_API_KEY or not VELLUM_BASE_URL:
        raise HTTPException(status_code=500, detail="VELLUM_API_KEY and VELLUM_BASE_URL are required.")
    if not REDUCTO_API_KEY:
        raise HTTPException(status_code=500, detail="REDUCTO_API_KEY is required.")

    job_id = str(uuid4())
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Save upload
        src_path = job_dir / (file.filename or "upload.bin")
        with src_path.open("wb") as f:
            f.write(file.file.read())

        # 1) If not PDF → CloudConvert → PDF; else keep as-is
        pdf_path = ensure_pdf(src_path, job_dir)

        # 2) PDF → Markdown (Reducto)
        md_text = pdf_to_markdown(pdf_path)
        write_text(job_dir, "markdown.txt", md_text)
        print("DEBUG: Reducto to MD Stage Finished")

        # 3–5) Classification (single pass)
        raw_cls, cls, cls_prompt_text = run_classification(md_text, job_dir)

        # 6–7) Plan identification (separate prompt + previous output)
        raw_plans, plans_by_loc, plan_ident_prompt_text = run_plan_identification(md_text, raw_cls, job_dir)

        # 8) Extraction step: parallel + dedup + skip existing
        cls_loc_pages = cls.get("loc_pages", {}) if isinstance(cls, dict) else {}
        extracted: List[Dict[str, Any]] = run_all_extractions(
            md_text=md_text,
            plans_by_loc=plans_by_loc,
            loc_pages=cls_loc_pages,
            job_dir=job_dir,
            max_workers=10,  # your current setting
        )
        print(f"DEBUG: Extraction Stage Finished: {len(extracted)} plans")

        # Final JSON
        result = {"plans": extracted}
        write_text(job_dir, "result.json", json.dumps(result, indent=2))

        resp = ProcessResponse(
            job_id=job_id,
            classification_output=raw_cls,
            plan_identification_output=raw_plans,
            classification_prompt_text=cls_prompt_text,
            plan_identification_prompt_text=plan_ident_prompt_text,
            json_result=result,
        )
        return resp

    except HTTPException:
        raise
    except Exception as e:
        write_text(job_dir, "errors.log", f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{job_id}/{filename}")
def get_artifact(job_id: str, filename: str):
    p = ARTIFACTS_DIR / job_id / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p)