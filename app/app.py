import argparse
import base64
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, Template
from PIL import Image
from slugify import slugify


# Create a basic logger that prints to stdout.
# Adjust the level to DEBUG if you want even more detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger()

app = FastAPI()

# ----------------------------------------------------------------------
# CLI flags – original ones + new Open‑WebUI options
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="AI‑wizard for Hugo posts")
parser.add_argument("--dest", default=os.getenv("DEST", ".."),
                    help="Base destination directory for Hugo site (env: DEST)")
parser.add_argument("--content", default=os.getenv("CONTENT", "content/blog/esemenyek"),
                    help="Sub‑directory under dest for markdown files (env: CONTENT)")
parser.add_argument("--static", default=os.getenv("STATIC", "static/img/plakatok"),
                    help="Sub‑directory under dest for images (env: STATIC)")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "43264")),
                    help="Port for the web server (env: PORT)")

# ---- NEW flags ---------------------------------------------------------
parser.add_argument("--openwebui-url", default=os.getenv("OPENWEBUI_URL", "http://localhost:3000"),
                    help="Base URL of the Open‑WebUI server (env: OPENWEBUI_URL)")
parser.add_argument("--openwebui-token", default=os.getenv("OPENWEBUI_TOKEN", ""),
                    help="Bearer token for Open‑WebUI (env: OPENWEBUI_TOKEN)")
parser.add_argument("--openwebui-model", default=os.getenv("OPENWEBUI_MODEL", "granite3.1-dense:8b"),
                    help="Model name to query (env: OPENWEBUI_MODEL)")
# ----------------------------------------------------------------------
args = parser.parse_args()

DEST_ROOT = Path(args.dest).resolve()
CONTENT_DIR = DEST_ROOT / args.content
STATIC_DIR = DEST_ROOT / args.static

STATIC_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Jinja2 env
# ----------------------------------------------------------------------
templates = Environment(loader=FileSystemLoader("templates"))

# ----------------------------------------------------------------------
# Helper utilities (unchanged)
# ----------------------------------------------------------------------
def resize_image(src_path: Path):
    with Image.open(src_path) as im:
        im.thumbnail((1920, 1080))
        im.save(src_path, quality=85)

def convert_pdf_to_jpg(pdf_path: Path) -> Path:
    jpg_path = pdf_path.with_suffix(".jpg")
    cmd = [
        "convert", "-density", "72",
        f"{pdf_path}[0]", "-resize", "1920x1080", str(jpg_path)
    ]
    subprocess.run(cmd, check=True)
    return jpg_path

from datetime import datetime

def iso_to_local_input(iso_str: str) -> str:
    """
    Convert an ISO‑8601 string (with or without a trailing Z) to the
    format required by <input type="datetime-local">:
        YYYY‑MM‑DDTHH:MM
    The function assumes the ISO string is UTC; it simply drops the
    seconds and the timezone indicator.
    """
    # Strip the trailing Z (if present) and any microseconds
    iso_str = iso_str.rstrip("Z")
    # Parse – datetime.fromisoformat can handle “YYYY‑MM‑DDTHH:MM:SS” or “YYYY‑MM‑DDTHH:MM”
    dt = datetime.fromisoformat(iso_str)
    # Return the format the widget expects
    return dt.strftime("%Y-%m-%dT%H:%M")


# ----------------------------------------------------------------------
# 1️⃣  Change the payload to request a streamed response
# ----------------------------------------------------------------------
def call_openwebui_ai(image_path: Path) -> dict:
    """
    Sends *image_path* to Open-WebUI to generate a Hugo blog post.
    Returns a dict with keys: date (ISO-8601), title, summary, content.
    """
    # Validate image path
    if not image_path.exists() or not image_path.is_file():
        raise ValueError(f"Image path {image_path} does not exist or is not a file")

    # Build data URL
    ext = image_path.suffix.lower().lstrip(".")
    mime, _ = mimetypes.guess_type(image_path) or (f"image/{'jpeg' if ext == 'jpg' else ext}", None)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    # Prompt
    prompt = """
        Analyze the attached image and create a Hugo blog post describing its content or context. 
        Return **only** a JSON object with the keys: `date` (ISO-8601 format, e.g., 2025-09-29T00:00:00Z), 
        `title`, `summary`, `content` in Hungarian language using the same words / vocabulary as on the image. 
        Do not include any extra text, or backticks. Summary and content fields should be Markdown formatted.
        The summary can be even 400 hundred characters, there is "enough" place to display it. 
        It is evident, that the event will be in the Budavári Evangélikus Templom (unless other specified) no need to add that.
    """

    print('=======')
    print(data_url[:120])
    print('=======')

    # Assemble payload (set stream = True)
    payload = {
        "model": args.openwebui_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        "stream": True          # ← enable streaming
    }

    print(json.dumps(payload)[:500], "...")

     # Section 4: Send a JSON request
    if 'lxs.cloud' in args.openwebui_url:
        url = f"{args.openwebui_url.rstrip('/')}/chat/completions"
    else:
        url = f"{args.openwebui_url.rstrip('/')}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {args.openwebui_token}",
        "Content-Type": "application/json"
    }

    analysis_output = []
    full_text = ""
    finish = False
    with requests.post(url, json=payload, headers=headers, stream=True) as r:
        r.encoding = "utf-8"
        print(r.text[:1000])  # show first KB
        r.raise_for_status()
        
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data)
                
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    analysis_output.append(delta)
                    print(delta, end="", flush=True)  # live output

                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason", "")
                if finish_reason == "stop":
                    finish = True
                    print("STOP")
                    break

                if chunk.get("choices", []) == []:
                    print(chunk.get("usage"))
                    if finish:
                        print("END")
                        break
                

            except json.JSONDecodeError:
                print("JSON error " + data)
                continue
            
        # Remove thinking from analysis_output
        full_text = "".join(analysis_output)
        # Remove content between <thinking> and </thinking> tags
        import re
        full_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL)
        
        raw_content = full_text.strip()
        raw_content = (
            raw_content.removeprefix("```json\n")
            .removeprefix("```\n")
            .removesuffix("\n```")
            .strip()
        )

    try:
        logger.info(raw_content)
        result = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to decode JSON from streamed response: {exc}\n"
            f"Raw streamed text: {raw_content[:500]}"
        ) from exc

    # ------------------------------------------------------------------
    # 5️⃣  Validate the result (unchanged)
    # ------------------------------------------------------------------
    required_keys = {"date", "title", "summary", "content"}
    if not isinstance(result, dict) or not all(k in result for k in required_keys):
        raise RuntimeError(
            f"Response JSON missing required keys: {required_keys}\nGot: {result}"
        )
    logger.info(f"{result}")
    return result


# ----------------------------------------------------------------------
# Routes (unchanged except for the call above)
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def upload_page():
    """Render the upload form – the very first step of the wizard."""
    tmpl = templates.get_template("upload.html")
    return HTMLResponse(content=tmpl.render())

@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    # Save temporarily
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    temp_path = tmp_dir / f"{slugify(file.filename)}{Path(file.filename).suffix}"
    with open(temp_path, "wb") as out:
        out.write(await file.read())


    ext = temp_path.suffix.lower()
    if ext == ".pdf":
        temp_path = convert_pdf_to_jpg(temp_path)


    served_static_dir = Path("static")
    preview_name = f"tmp_{temp_path.name}"
    preview_path = served_static_dir / preview_name
    shutil.copy2(temp_path, preview_path)

    # Build the URL that FastAPI will serve (mounted at /static)
    preview_image_url = f"/static/{preview_name}"

    # Call AI
    try:
        ai_data = call_openwebui_ai(temp_path)
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Open‑WebUI call failed: {exc}")

    # Render wizard with AI‑generated values
    raw_date = ai_data.get("date", "")
    date_for_input = iso_to_local_input(raw_date) if raw_date else ""

    tmpl = templates.get_template("wizard.html")
    html = tmpl.render(
        temp_path=str(temp_path),
        preview_image_url=preview_image_url,
        datetime_str=date_for_input,
        title=ai_data.get("title", ""),
        summary=ai_data.get("summary", ""),
        content=ai_data.get("content", ""),
    )
    return HTMLResponse(content=html)

# ----------------------------------------------------------------------
# Finalize (unchanged – uses the same logic as before)
# ----------------------------------------------------------------------
@app.post("/finalize")
async def finalize(
    temp_path: str = Form(...),
    datetime_str: str = Form(...),
    title: str = Form(...),
    summary: str = Form(...),
    content: str = Form(...)
):
    temp_path = Path(temp_path)
    if not temp_path.exists():
        raise HTTPException(status_code=400, detail="Temp file missing")

    try:
        dt = datetime.fromisoformat(datetime_str)
    except Exception:
        raise HTTPException(status_code=400,
                            detail="Invalid datetime format (YYYY‑MM‑DDTHH:MM)")

    date_str = dt.strftime("%Y%m%d")
    safe_title = slugify(title)
    ext = temp_path.suffix.lower()

    # PDF → JPG conversion if needed
    print(f'extension is {ext} from {temp_path}')
    if ext == ".pdf":
        final_image_path = convert_pdf_to_jpg(temp_path)
        ext = ".jpg"
    else:
        final_image_path = temp_path

    resize_image(final_image_path)

    final_image_name = f"{date_str}_{safe_title}{ext}"
    final_image_path = STATIC_DIR / final_image_name
    shutil.move(temp_path, final_image_path)

    # Build markdown (same template you already had)
    hugo_template = """---
title: {{ title }}
date: '{{ date }}'
publishDate: '{{ publish_date }}'
expiryDate: '{{ expiry_date }}'
categories:
    - hirdetések
tags:
    - hirdetések
summary: "{{ summary }}"
banner: {{ banner_path }}
---

# {{ title }}

![{{ banner_path }}]({{ banner_path }})

{{ content }}
"""
    banner_path = "/" + STATIC_DIR.relative_to(DEST_ROOT / "static").as_posix() + "/" + final_image_name
    publish_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d") # Use the imported datetime class
    expiry_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

    from jinja2 import Template
    md = Template(hugo_template).render(
        title=title,
        date=dt.isoformat(),
        publish_date=publish_date,
        expiry_date=expiry_date,
        summary=summary,
        banner_path=banner_path,
        content=content,
    )

    md_filename = f"{date_str}_{safe_title}.md"
    md_path = CONTENT_DIR / md_filename
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    # --------------------------------------------------------------
    # Git handling – run only if DEST_ROOT is a Git repository
    # --------------------------------------------------------------
    def _run_git(cmd: list[str]) -> subprocess.CompletedProcess:
        """Run a git command inside DEST_ROOT, raise on failure."""
        return subprocess.run(
            cmd,
            cwd=str(DEST_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    try:
        # Check for a .git folder (quick repo detection)
        if (DEST_ROOT / ".git").exists():
            # Stage the new/changed files
            _run_git(["git", "add", str(md_path.relative_to(DEST_ROOT))])
            _run_git(["git", "add", str(final_image_path.relative_to(DEST_ROOT))])

            # Commit – use a generic message that includes the title
            commit_msg = f"Add post: {title}"
            _run_git(["git", "commit", "-m", commit_msg])

            # Pull with rebase (force‑rebase to avoid conflicts)
            _run_git(["git", "pull", "--rebase", "--autostash"])

            # Push the new commit
            _run_git(["git", "push"])
    except subprocess.CalledProcessError as exc:
        # Log the failure but do not abort the request – the post is still created
        logger.error(
            "Git operation failed: %s\nstdout: %s\nstderr: %s",
            exc.cmd,
            exc.stdout,
            exc.stderr,
        )

    # Cleanup temp folder
    if temp_path.parent == Path("tmp"):
        shutil.rmtree("tmp", ignore_errors=True)

    return JSONResponse(
        {
            "detail": "Bejegyzés sikeresen létrehozva",
            "markdown_path": str(md_path),
            "image_path": str(final_image_path),
        }
    )

# ----------------------------------------------------------------------
# Serve static assets (optional)
# ----------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=False)
