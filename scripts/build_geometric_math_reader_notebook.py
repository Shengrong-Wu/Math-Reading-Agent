import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "geometric_math_reader_workflow.ipynb"


def lines(block: str) -> list[str]:
    return textwrap.dedent(block).lstrip("\n").splitlines(keepends=True)


notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                """
                # The Geometric Math Reader

                This notebook tests the core workflow for the math-reading agent before the web UI exists. It does four things:

                1. Ingests a PDF, LaTeX, markdown, text file, or URL.
                2. Sends the extracted content to Gemini with a math-specific system prompt.
                3. Intercepts `[GENERATE_IMAGE: ...]` and `[GENERATE_PLOT: ...]` tags during streaming.
                4. Renders the augmented summary inline and exports Markdown/HTML, plus PDF when `weasyprint` is installed.

                Edit the config cell for your API key and source document, then run from top to bottom.
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
                # Optional dependency install if your notebook kernel is missing packages.
                # %pip install -q google-genai pymupdf requests beautifulsoup4 markdown pillow matplotlib weasyprint
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
                import ast
                import base64
                import html
                import io
                import json
                import math
                import mimetypes
                import os
                import re
                import textwrap
                import uuid
                from dataclasses import dataclass, field
                from pathlib import Path
                from typing import Any

                import fitz
                import matplotlib.pyplot as plt
                import numpy as np
                import requests
                from bs4 import BeautifulSoup
                from google import genai
                from google.genai import types
                from IPython.display import Image as IPyImage
                from IPython.display import Markdown as IPyMarkdown
                from IPython.display import clear_output, display
                from PIL import Image

                try:
                    import markdown as markdown_lib
                except ImportError:
                    markdown_lib = None

                try:
                    from weasyprint import HTML as WeasyHTML
                except ImportError:
                    WeasyHTML = None

                BASE_DIR = Path.cwd()
                ARTIFACT_DIR = BASE_DIR / "artifacts"
                PLOTS_DIR = ARTIFACT_DIR / "plots"
                IMAGES_DIR = ARTIFACT_DIR / "images"
                EXPORTS_DIR = ARTIFACT_DIR / "exports"
                for directory in (ARTIFACT_DIR, PLOTS_DIR, IMAGES_DIR, EXPORTS_DIR):
                    directory.mkdir(parents=True, exist_ok=True)

                API_KEY = os.getenv("GEMINI_API_KEY", "")
                TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3-flash-preview")
                IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
                USE_JINA_READER = True
                REQUEST_TIMEOUT = 60
                MAX_SOURCE_CHARS = 120_000

                SYSTEM_PROMPT = \"\"\"You are a PhD-level Mathematical Assistant. Your goal is to summarize the provided text for a researcher.

                Structure: Provide an Outline, Key Propositions, and Sketch Proofs.
                Notation: Use strict LaTeX for all mathematical expressions.
                Visualization Logic: For geometry-heavy papers, prefer 4-8 visuals when they materially improve understanding. When an important definition introduces a central geometric object, singularity model, or flow, add a short geometric example or visual intuition paragraph and then place a [GENERATE_IMAGE: <minimalist_prompt>] tag immediately after it. After major theorem or proposition statements, insert a visual whenever the statement describes a dichotomy, perturbation, stability mechanism, asymptotic profile, singularity type, or before/after transition.
                If the concept is conceptual or geometric (for example, a manifold, a fiber bundle, rescaled mean curvature flow, a neck pinch, or a cylindrical versus spherical singularity), use [GENERATE_IMAGE: <minimalist_prompt>]. Prefer prompts that depict one clean mathematical situation or a short progression, such as smooth surface -> small perturbation -> spherical or nondegenerate cylindrical singularities.
                If the concept is functional or quantitative (for example, a vector field, a monotonicity quantity, a profile curve, or a graph of a function), use [GENERATE_PLOT: <matplotlib_python_code>].
                Visual Minimalism: Every GENERATE_IMAGE prompt must include: Minimalist diagram, white background, high contrast, no labels, single focus, clean geometric lines. Do not add clutter or text.
                Plot Safety: In every GENERATE_PLOT tag, emit runnable matplotlib code only. Do not import math, numpy, or matplotlib; math, np, and plt are already available. Do not call plt.show() or savefig(). Avoid square brackets in the Python code body so the tag can be parsed reliably while streaming; prefer tuples over lists when possible.
                Output Format: Return Markdown prose with inline GENERATE tags exactly where the visual should appear. For important definitions and theorem statements, add 1-3 sentences of geometric example or intuition around the visual. Do not wrap the tags in code fences.\"\"\"

                TAG_PATTERN = re.compile(r"\\[(GENERATE_IMAGE|GENERATE_PLOT):\\s*(.*?)\\]", re.DOTALL)
                TAG_PREFIX = "[GENERATE_"
                MINIMALIST_IMAGE_SUFFIX = "Minimalist diagram, white background, high contrast, no labels, single focus, clean geometric lines. Do not add clutter or text."
                PLOT_BANNED_NAMES = {
                    "open",
                    "exec",
                    "eval",
                    "compile",
                    "__import__",
                    "input",
                    "help",
                    "os",
                    "sys",
                    "subprocess",
                    "pathlib",
                    "shutil",
                    "socket",
                    "requests",
                }


                @dataclass
                class IngestedDocument:
                    source: str
                    kind: str
                    title: str
                    text: str
                    metadata: dict[str, Any] = field(default_factory=dict)


                def slugify(value: str) -> str:
                    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
                    return slug or f"document-{uuid.uuid4().hex[:8]}"


                def is_url(value: str) -> bool:
                    return value.startswith("http://") or value.startswith("https://")


                def clean_document_text(text: str) -> str:
                    text = text.replace("\\x00", "")
                    text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
                    text = re.sub(r"\\n{3,}", "\\n\\n", text)
                    return text.strip()


                def build_jina_reader_url(url: str) -> str:
                    stripped = url.split("://", 1)[-1]
                    return f"https://r.jina.ai/http://{stripped}"


                def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                        return clean_document_text("\\n\\n".join(page.get_text("text") for page in doc))


                def extract_text_from_pdf_path(path: Path) -> str:
                    with fitz.open(path) as doc:
                        return clean_document_text("\\n\\n".join(page.get_text("text") for page in doc))


                def read_text_file(path: Path) -> str:
                    return clean_document_text(path.read_text(encoding="utf-8"))


                def html_to_markdownish(html_text: str) -> str:
                    soup = BeautifulSoup(html_text, "html.parser")
                    for element in soup(["script", "style", "noscript", "svg"]):
                        element.decompose()

                    lines_out = []
                    if soup.title and soup.title.get_text(strip=True):
                        lines_out.append(f"# {soup.title.get_text(' ', strip=True)}")

                    body = soup.body or soup
                    for element in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"], limit=500):
                        text = element.get_text(" ", strip=True)
                        if not text:
                            continue
                        if element.name == "h1":
                            lines_out.append(f"# {text}")
                        elif element.name == "h2":
                            lines_out.append(f"## {text}")
                        elif element.name == "h3":
                            lines_out.append(f"### {text}")
                        elif element.name == "h4":
                            lines_out.append(f"#### {text}")
                        elif element.name == "li":
                            lines_out.append(f"- {text}")
                        else:
                            lines_out.append(text)
                    return clean_document_text("\\n\\n".join(lines_out))


                def ingest_url(url: str) -> IngestedDocument:
                    headers = {"User-Agent": "GeometricMathReader/0.1"}
                    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "").lower()
                    if url.lower().endswith(".pdf") or "application/pdf" in content_type:
                        title = Path(url.split("?")[0]).stem or "url-pdf"
                        return IngestedDocument(
                            source=url,
                            kind="pdf-url",
                            title=title,
                            text=extract_text_from_pdf_bytes(response.content),
                            metadata={"content_type": content_type},
                        )

                    text = ""
                    if USE_JINA_READER:
                        try:
                            reader_response = requests.get(
                                build_jina_reader_url(url),
                                timeout=REQUEST_TIMEOUT,
                                headers=headers,
                            )
                            reader_response.raise_for_status()
                            text = clean_document_text(reader_response.text)
                        except Exception:
                            text = ""

                    if not text:
                        text = html_to_markdownish(response.text)

                    title = Path(url.split("?")[0]).stem or "url-document"
                    return IngestedDocument(
                        source=url,
                        kind="url",
                        title=title,
                        text=text,
                        metadata={"content_type": content_type},
                    )


                def ingest_file(path_string: str) -> IngestedDocument:
                    path = Path(path_string).expanduser().resolve()
                    if not path.exists():
                        raise FileNotFoundError(f"Source file does not exist: {path}")

                    suffix = path.suffix.lower()
                    if suffix == ".pdf":
                        kind = "pdf"
                        text = extract_text_from_pdf_path(path)
                    elif suffix in {".tex", ".md", ".markdown", ".txt"}:
                        kind = suffix.lstrip(".")
                        text = read_text_file(path)
                    else:
                        raise ValueError(f"Unsupported file type for notebook workflow: {suffix}")

                    return IngestedDocument(
                        source=str(path),
                        kind=kind,
                        title=path.stem,
                        text=text,
                        metadata={"path": str(path)},
                    )


                def ingest_source(source: str) -> IngestedDocument:
                    if is_url(source):
                        return ingest_url(source)
                    return ingest_file(source)


                def build_summary_prompt(document: IngestedDocument) -> str:
                    trimmed_text = document.text[:MAX_SOURCE_CHARS]
                    truncated_note = ""
                    if len(document.text) > len(trimmed_text):
                        truncated_note = (
                            "The source text was truncated before sending to the model. Prioritize the most central definitions, propositions, and proofs that remain in the excerpt.\\n\\n"
                        )

                    return textwrap.dedent(
                        f\"\"\"
                        Summarize the following mathematical source.

                        Source: {document.source}
                        Title: {document.title}
                        Detected kind: {document.kind}

                        Additional instructions:
                        - Produce a readable vertical document with sections for Outline, Key Propositions, and Sketch Proofs.
                        - Use strict LaTeX for mathematics.
                        - For geometry-heavy material, prefer roughly 4 to 8 visual tags when they materially clarify the paper.
                        - After an important definition, add a short geometric example or visual intuition paragraph and then insert a visual tag if the object is central to the topic.
                        - After a major theorem or proposition statement, add a visual whenever the statement describes a dichotomy, perturbation, stability result, asymptotic model, singularity type, or before/after transition.
                        - For geometric PDE or flow papers, actively add extra visualizations for central objects such as rescaled flows, neck pinches, cylindrical versus spherical singularities, and perturbation mechanisms.
                        - Keep image prompts concise, geometric, and stylistically minimalist. Sequence-style prompts are encouraged, for example: smooth surface -> small perturbation -> surface with spherical or nondegenerate cylindrical singularities.
                        - For plot tags, emit matplotlib code that creates a figure but does not save or show it. Do not import math, numpy, or matplotlib because math, np, and plt are already available.
                        - Avoid square brackets inside plot code so the tag parser can safely detect the closing delimiter.
                        - Do not wrap GENERATE tags in code fences.

                        {truncated_note}Source text:
                        {trimmed_text}
                        \"\"\"
                    ).strip()


                def append_markdown(blocks: list[dict[str, Any]], text: str) -> None:
                    if not text:
                        return
                    if blocks and blocks[-1]["type"] == "markdown":
                        blocks[-1]["content"] += text
                    else:
                        blocks.append({"type": "markdown", "content": text})


                def parse_stream_chunk(chunk_text: str, pending: str) -> tuple[list[tuple[str, ...]], str]:
                    pending += chunk_text
                    events: list[tuple[str, ...]] = []

                    while pending:
                        start = pending.find(TAG_PREFIX)
                        if start == -1:
                            safe_length = max(0, len(pending) - (len(TAG_PREFIX) - 1))
                            if safe_length:
                                events.append(("text", pending[:safe_length]))
                                pending = pending[safe_length:]
                            break

                        if start > 0:
                            events.append(("text", pending[:start]))
                            pending = pending[start:]

                        closing = pending.find("]")
                        if closing == -1:
                            break

                        candidate = pending[: closing + 1]
                        match = TAG_PATTERN.fullmatch(candidate)
                        if not match:
                            events.append(("text", pending[0]))
                            pending = pending[1:]
                            continue

                        events.append(("tag", match.group(1), match.group(2).strip()))
                        pending = pending[closing + 1 :]

                    return events, pending


                def flush_stream_buffer(pending: str) -> list[tuple[str, str]]:
                    if pending:
                        return [("text", pending)]
                    return []


                def render_blocks(blocks: list[dict[str, Any]], status: str | None = None) -> None:
                    clear_output(wait=True)
                    if status:
                        display(IPyMarkdown(f"> {status}"))

                    for block in blocks:
                        if block["type"] == "markdown":
                            display(IPyMarkdown(block["content"]))
                        elif block["type"] == "image":
                            display(IPyImage(filename=str(block["path"])))
                            if block.get("caption"):
                                display(IPyMarkdown(f"*{block['caption']}*"))
                        elif block["type"] == "placeholder":
                            display(IPyMarkdown(f"> {block['message']}"))


                def normalize_image_prompt(prompt: str) -> str:
                    prompt = prompt.strip()
                    if not prompt:
                        return MINIMALIST_IMAGE_SUFFIX
                    if "Minimalist diagram" in prompt:
                        return prompt
                    if prompt.endswith("."):
                        return f"{prompt} {MINIMALIST_IMAGE_SUFFIX}"
                    return f"{prompt}. {MINIMALIST_IMAGE_SUFFIX}"


                def extract_inline_image(response: Any) -> Image.Image:
                    candidates = getattr(response, "candidates", None) or []
                    for candidate in candidates:
                        content = getattr(candidate, "content", None)
                        parts = getattr(content, "parts", None) or []
                        for part in parts:
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data and getattr(inline_data, "data", None):
                                image = Image.open(io.BytesIO(inline_data.data))
                                image.load()
                                return image
                    raise RuntimeError("The image model response did not include inline image bytes.")


                def generate_image(client: genai.Client, prompt: str, model: str = IMAGE_MODEL) -> Path:
                    response = client.models.generate_content(
                        model=model,
                        contents=normalize_image_prompt(prompt),
                    )
                    image = extract_inline_image(response)
                    output_path = IMAGES_DIR / f"image-{uuid.uuid4().hex[:8]}.png"
                    image.save(output_path)
                    return output_path


                SAFE_PLOT_IMPORT_LINES = {
                    "import math",
                    "import numpy as np",
                    "import matplotlib.pyplot as plt",
                }


                def sanitize_plot_code(plot_code: str) -> str:
                    cleaned_lines = []
                    for line in plot_code.splitlines():
                        if line.strip() in SAFE_PLOT_IMPORT_LINES:
                            continue
                        cleaned_lines.append(line)
                    return "\\n".join(cleaned_lines).strip()


                def validate_plot_code(plot_code: str) -> None:
                    tree = ast.parse(plot_code, mode="exec")
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            raise ValueError("Plot code may not import modules.")
                        if isinstance(node, ast.Name):
                            if node.id.startswith("__") or node.id in PLOT_BANNED_NAMES:
                                raise ValueError(f"Disallowed name in plot code: {node.id}")
                        if isinstance(node, ast.Attribute):
                            if node.attr.startswith("__"):
                                raise ValueError("Dunder attribute access is not allowed in plot code.")
                            if node.attr in {"savefig", "show"}:
                                raise ValueError("Plot code should not call savefig() or show().")


                def execute_plot_code(plot_code: str) -> Path:
                    plot_code = sanitize_plot_code(plot_code)
                    validate_plot_code(plot_code)
                    plt.close("all")

                    safe_builtins = {
                        "abs": abs,
                        "float": float,
                        "int": int,
                        "len": len,
                        "max": max,
                        "min": min,
                        "pow": pow,
                        "range": range,
                        "round": round,
                        "sum": sum,
                        "tuple": tuple,
                    }
                    globals_dict = {"__builtins__": safe_builtins}
                    locals_dict = {"math": math, "np": np, "plt": plt}

                    exec(compile(plot_code, "<GENERATE_PLOT>", "exec"), globals_dict, locals_dict)
                    if not plt.get_fignums():
                        raise RuntimeError("The plot code finished without creating a matplotlib figure.")

                    figure = plt.gcf()
                    figure.tight_layout()
                    output_path = PLOTS_DIR / f"plot-{uuid.uuid4().hex[:8]}.png"
                    figure.savefig(output_path, dpi=200, bbox_inches="tight")
                    plt.close("all")
                    return output_path


                def image_path_to_data_uri(path: Path) -> str:
                    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                    return f"data:{mime_type};base64,{encoded}"


                def markdown_to_html(markdown_text: str) -> str:
                    if markdown_lib is None:
                        escaped = html.escape(markdown_text)
                        return f"<pre>{escaped}</pre>"
                    return markdown_lib.markdown(
                        markdown_text,
                        extensions=["fenced_code", "tables", "sane_lists"],
                        output_format="html5",
                    )


                def blocks_to_markdown(blocks: list[dict[str, Any]], export_root: Path | None = None) -> str:
                    parts: list[str] = []
                    for block in blocks:
                        if block["type"] == "markdown":
                            parts.append(block["content"].rstrip())
                        elif block["type"] == "image":
                            path = block["path"]
                            image_ref = str(path)
                            if export_root is not None:
                                image_ref = os.path.relpath(path, start=export_root)
                            parts.append(f"![{block.get('caption', 'generated image')}]({image_ref})")
                            if block.get("caption"):
                                parts.append(f"*{block['caption']}*")
                        elif block["type"] == "placeholder":
                            parts.append(f"> {block['message']}")
                    return "\\n\\n".join(parts).strip() + "\\n"


                def build_review_markdown(blocks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
                    parts: list[str] = []
                    visuals: list[dict[str, Any]] = []
                    visual_index = 0
                    for block in blocks:
                        if block["type"] == "markdown":
                            parts.append(block["content"].rstrip())
                        elif block["type"] == "image":
                            visual_index += 1
                            caption = block.get("caption", "generated visual")
                            visuals.append(
                                {
                                    "visual_index": visual_index,
                                    "path": block["path"],
                                    "caption": caption,
                                }
                            )
                            parts.append(f"[VISUAL {visual_index}: {caption}]")
                        elif block["type"] == "placeholder":
                            parts.append(f"[PLACEHOLDER: {block['message']}]")
                    return "\\n\\n".join(part for part in parts if part).strip(), visuals


                def parse_json_response(text: str) -> dict[str, Any]:
                    cleaned = text.strip()
                    if cleaned.startswith("```"):
                        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
                        cleaned = re.sub(r"\\s*```$", "", cleaned)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        match = re.search(r"\\{.*\\}", cleaned, re.DOTALL)
                        if not match:
                            raise
                        return json.loads(match.group(0))


                def run_final_review(
                    client: genai.Client,
                    blocks: list[dict[str, Any]],
                    text_model: str = TEXT_MODEL,
                ) -> dict[str, Any]:
                    report_markdown, visuals = build_review_markdown(blocks)
                    review_prompt = textwrap.dedent(
                        \"\"\"
                        You are performing one final QA pass on an already generated mathematical reading report.

                        Important constraints:
                        - You do not have the original article text anymore.
                        - Do not fact-check article-specific mathematical details.
                        - Do not invent new theorem content, hypotheses, or proof steps.
                        - Only evaluate overall report logic, section flow, whether the statements read coherently, whether visuals are suitable for the nearby text, and whether one or two additional visuals would materially help the reader.
                        - If an existing visual is too generic, mismatched, or misleading, suggest a replacement prompt.
                        - If another argument still needs a visual, suggest at most two additional image prompts.
                        - Produce a polished reader-facing title for the report.
                        - Keep comments concise and high signal.

                        Return JSON only with this schema:
                        {
                          "title": string,
                          "summary_assessment": string,
                          "logic_warnings": [string],
                          "visual_reviews": [
                            {"visual_index": integer, "suitable": boolean, "reason": string}
                          ],
                          "replacement_visuals": [
                            {"visual_index": integer, "reason": string, "prompt": string}
                          ],
                          "additional_visuals": [
                            {"placement_hint": string, "reason": string, "prompt": string}
                          ]
                        }

                        Every prompt must describe a minimalist mathematical image on a white background with clean geometric lines and no text labels.
                        \"\"\"
                    ).strip()

                    parts = [
                        types.Part.from_text(text=review_prompt),
                        types.Part.from_text(text=f"Report for review:\\n\\n{report_markdown}"),
                    ]
                    for visual in visuals[:8]:
                        path = Path(visual["path"])
                        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                        parts.append(
                            types.Part.from_text(
                                text=f"Visual {visual['visual_index']} caption: {visual['caption']}"
                            )
                        )
                        parts.append(
                            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)
                        )

                    response = client.models.generate_content(
                        model=text_model,
                        contents=[types.Content(role="user", parts=parts)],
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            response_mime_type="application/json",
                        ),
                    )
                    if not (response.text or "").strip():
                        raise RuntimeError("Final QA pass returned an empty response.")
                    return parse_json_response(response.text)


                def update_report_title(blocks: list[dict[str, Any]], title: str) -> None:
                    clean_title = title.strip()
                    if not clean_title:
                        return
                    for block in blocks:
                        if block["type"] != "markdown":
                            continue
                        lines = block["content"].splitlines()
                        if lines and lines[0].startswith("# "):
                            lines[0] = f"# {clean_title}"
                            block["content"] = "\\n".join(lines)
                            return
                        block["content"] = f"# {clean_title}\\n\\n" + block["content"]
                        return


                def apply_final_review(
                    client: genai.Client,
                    blocks: list[dict[str, Any]],
                    review_data: dict[str, Any],
                    image_model: str = IMAGE_MODEL,
                    fallback_title: str = "Mathematical Reading Report",
                ) -> str:
                    final_title = str(review_data.get("title") or fallback_title).strip() or fallback_title
                    update_report_title(blocks, final_title)

                    image_block_indices = [
                        index for index, block in enumerate(blocks) if block["type"] == "image"
                    ]
                    replacement_notes: list[str] = []
                    for item in review_data.get("replacement_visuals", [])[:2]:
                        try:
                            visual_index = int(item.get("visual_index", 0))
                        except (TypeError, ValueError):
                            continue
                        if visual_index < 1 or visual_index > len(image_block_indices):
                            continue
                        prompt = str(item.get("prompt") or "").strip()
                        if not prompt:
                            continue
                        block_index = image_block_indices[visual_index - 1]
                        try:
                            new_path = generate_image(client, prompt, model=image_model)
                            blocks[block_index]["path"] = new_path
                            blocks[block_index]["caption"] = prompt
                            reason = str(item.get("reason") or "Improved visual alignment.").strip()
                            replacement_notes.append(
                                f"- Replaced visual {visual_index}: {reason}"
                            )
                        except Exception as exc:
                            replacement_notes.append(
                                f"- Failed to replace visual {visual_index}: {exc}"
                            )

                    qa_lines = ["## Final QA Pass"]
                    summary_assessment = str(review_data.get("summary_assessment") or "").strip()
                    if summary_assessment:
                        qa_lines.append(summary_assessment)

                    logic_warnings = [
                        str(item).strip()
                        for item in review_data.get("logic_warnings", [])
                        if str(item).strip()
                    ]
                    if logic_warnings:
                        qa_lines.append("### Logic Review")
                        qa_lines.extend(f"- {warning}" for warning in logic_warnings)

                    visual_reviews = review_data.get("visual_reviews", [])
                    if visual_reviews:
                        qa_lines.append("### Visual Review")
                        for item in visual_reviews:
                            try:
                                visual_index = int(item.get("visual_index", 0))
                            except (TypeError, ValueError):
                                continue
                            reason = str(item.get("reason") or "").strip()
                            if not reason:
                                continue
                            verdict = "suitable" if item.get("suitable", False) else "needs improvement"
                            qa_lines.append(f"- Visual {visual_index}: {verdict}. {reason}")

                    if replacement_notes:
                        qa_lines.append("### Applied Visual Revisions")
                        qa_lines.extend(replacement_notes)

                    blocks.append(
                        {"type": "markdown", "content": "\\n\\n".join(qa_lines).strip() + "\\n"}
                    )

                    extra_visuals = review_data.get("additional_visuals", [])[:2]
                    if extra_visuals:
                        blocks.append({"type": "markdown", "content": "## Final Visual Additions\\n"})
                    for item in extra_visuals:
                        placement_hint = str(item.get("placement_hint") or "Later in the report").strip()
                        reason = str(
                            item.get("reason")
                            or "Additional visualization requested by the final QA pass."
                        ).strip()
                        prompt = str(item.get("prompt") or "").strip()
                        if not prompt:
                            continue
                        blocks.append(
                            {
                                "type": "markdown",
                                "content": f"### Suggested placement: {placement_hint}\\n\\n{reason}\\n",
                            }
                        )
                        try:
                            image_path = generate_image(client, prompt, model=image_model)
                            blocks.append(
                                {"type": "image", "path": image_path, "caption": prompt}
                            )
                        except Exception as exc:
                            blocks.append(
                                {
                                    "type": "placeholder",
                                    "message": f"Final QA visual generation failed: {exc}",
                                }
                            )

                    return final_title


                def blocks_to_html(blocks: list[dict[str, Any]], title: str) -> str:
                    body_parts: list[str] = []
                    for block in blocks:
                        if block["type"] == "markdown":
                            body_parts.append(markdown_to_html(block["content"]))
                        elif block["type"] == "image":
                            data_uri = image_path_to_data_uri(block["path"])
                            caption = html.escape(block.get("caption", ""))
                            body_parts.append(
                                f"<figure><img src='{data_uri}' alt='{caption}' /><figcaption>{caption}</figcaption></figure>"
                            )
                        elif block["type"] == "placeholder":
                            body_parts.append(f"<blockquote>{html.escape(block['message'])}</blockquote>")

                    body_html = "\\n".join(body_parts)
                    return f\"\"\"<!DOCTYPE html>
                    <html lang='en'>
                    <head>
                      <meta charset='utf-8' />
                      <meta name='viewport' content='width=device-width, initial-scale=1' />
                      <title>{html.escape(title)}</title>
                      <style>
                        :root {{
                          color-scheme: light;
                          --page-width: 860px;
                          --text: #101418;
                          --muted: #5f6b76;
                          --line: #dbe2e8;
                          --surface: #ffffff;
                          --bg: #f6f7f9;
                        }}
                        body {{
                          margin: 0;
                          background: var(--bg);
                          color: var(--text);
                          font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
                          line-height: 1.7;
                        }}
                        main {{
                          max-width: var(--page-width);
                          margin: 0 auto;
                          padding: 48px 24px 80px;
                        }}
                        article {{
                          background: var(--surface);
                          border: 1px solid var(--line);
                          border-radius: 20px;
                          padding: 40px 44px;
                          box-shadow: 0 10px 30px rgba(10, 20, 30, 0.05);
                        }}
                        h1, h2, h3, h4 {{
                          line-height: 1.2;
                          margin-top: 1.6em;
                        }}
                        p, li {{
                          font-size: 1.05rem;
                        }}
                        img {{
                          display: block;
                          width: 100%;
                          max-width: 760px;
                          margin: 18px auto 8px;
                          border: 1px solid var(--line);
                          border-radius: 12px;
                          background: #fff;
                        }}
                        figure {{
                          margin: 28px 0;
                        }}
                        figcaption {{
                          color: var(--muted);
                          font-size: 0.95rem;
                          text-align: center;
                        }}
                        blockquote {{
                          margin: 24px 0;
                          padding: 16px 18px;
                          border-left: 4px solid #d97706;
                          background: #fff7ed;
                          color: #7c2d12;
                        }}
                        code {{
                          background: #f3f4f6;
                          padding: 0.1rem 0.3rem;
                          border-radius: 6px;
                        }}
                        pre {{
                          overflow-x: auto;
                          padding: 16px;
                          border-radius: 12px;
                          background: #111827;
                          color: #f9fafb;
                        }}
                      </style>
                      <script>
                        window.MathJax = {
                          tex: {
                            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                            processEscapes: true,
                            processEnvironments: true
                          },
                          options: {
                            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
                          },
                          svg: {
                            fontCache: 'global'
                          }
                        };
                      </script>
                      <script id='MathJax-script' async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js'></script>
                    </head>
                    <body>
                      <main>
                        <article>
                          {body_html}
                        </article>
                      </main>
                    </body>
                    </html>
                    \"\"\"


                def export_augmented_summary(blocks: list[dict[str, Any]], title: str) -> dict[str, Path]:
                    slug = slugify(title)
                    markdown_path = EXPORTS_DIR / f"{slug}.md"
                    html_path = EXPORTS_DIR / f"{slug}.html"

                    markdown_path.write_text(
                        blocks_to_markdown(blocks, export_root=markdown_path.parent),
                        encoding="utf-8",
                    )
                    html_text = blocks_to_html(blocks, title=title)
                    html_path.write_text(html_text, encoding="utf-8")

                    exported = {"md": markdown_path, "html": html_path}
                    if WeasyHTML is not None:
                        pdf_path = EXPORTS_DIR / f"{slug}.pdf"
                        WeasyHTML(string=html_text).write_pdf(pdf_path)
                        exported["pdf"] = pdf_path
                    return exported


                def run_workflow(
                    source: str,
                    api_key: str,
                    text_model: str = TEXT_MODEL,
                    image_model: str = IMAGE_MODEL,
                    system_prompt: str = SYSTEM_PROMPT,
                ) -> dict[str, Any]:
                    if not api_key.strip():
                        raise ValueError(
                            "Set API_KEY in the config cell or export GEMINI_API_KEY before running the workflow."
                        )

                    client = genai.Client(api_key=api_key.strip())
                    document = ingest_source(source)
                    blocks: list[dict[str, Any]] = [
                        {
                            "type": "markdown",
                            "content": f"# {document.title}\\n\\n_Source:_ `{document.source}`\\n\\n",
                        }
                    ]
                    render_blocks(
                        blocks,
                        status="Document ingested. Requesting structured summary from Gemini...",
                    )

                    raw_output = ""
                    pending = ""
                    image_count = 0
                    plot_count = 0
                    final_title = document.title
                    review_data: dict[str, Any] | None = None

                    stream = client.models.generate_content_stream(
                        model=text_model,
                        contents=build_summary_prompt(document),
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.3,
                        ),
                    )

                    for chunk in stream:
                        chunk_text = chunk.text or ""
                        if not chunk_text:
                            continue

                        raw_output += chunk_text
                        events, pending = parse_stream_chunk(chunk_text, pending)
                        for event in events:
                            if event[0] == "text":
                                append_markdown(blocks, event[1])
                                render_blocks(blocks, status="Streaming summary...")
                                continue

                            _, tag_name, payload = event
                            if tag_name == "GENERATE_IMAGE":
                                render_blocks(blocks, status="Generating inline image...")
                                try:
                                    image_path = generate_image(client, payload, model=image_model)
                                    blocks.append({"type": "image", "path": image_path, "caption": payload})
                                    image_count += 1
                                except Exception as exc:
                                    blocks.append(
                                        {
                                            "type": "placeholder",
                                            "message": f"Image generation failed: {exc}",
                                        }
                                    )
                            elif tag_name == "GENERATE_PLOT":
                                render_blocks(
                                    blocks,
                                    status="Executing matplotlib code for inline plot...",
                                )
                                try:
                                    plot_path = execute_plot_code(payload)
                                    blocks.append(
                                        {
                                            "type": "image",
                                            "path": plot_path,
                                            "caption": "AI-generated matplotlib plot",
                                        }
                                    )
                                    plot_count += 1
                                except Exception as exc:
                                    blocks.append(
                                        {
                                            "type": "placeholder",
                                            "message": f"Plot generation failed: {exc}",
                                        }
                                    )
                            render_blocks(blocks, status="Continuing summary stream...")

                    for event in flush_stream_buffer(pending):
                        append_markdown(blocks, event[1])

                    render_blocks(blocks, status="Running final QA pass...")
                    try:
                        review_data = run_final_review(client, blocks, text_model=text_model)
                        final_title = apply_final_review(
                            client,
                            blocks,
                            review_data,
                            image_model=image_model,
                            fallback_title=document.title,
                        )
                    except Exception as exc:
                        blocks.append(
                            {"type": "placeholder", "message": f"Final QA pass failed: {exc}"}
                        )

                    render_blocks(
                        blocks,
                        status=(
                            f"Complete. Generated {image_count} image(s) and {plot_count} plot(s), "
                            "then finished one final QA pass."
                        ),
                    )
                    return {
                        "document": document,
                        "blocks": blocks,
                        "raw_output": raw_output,
                        "image_count": image_count,
                        "plot_count": plot_count,
                        "review": review_data,
                        "final_title": final_title,
                    }


                print("Notebook helpers loaded. Fill the config cell, then run the workflow cell.")
                print(f"Artifacts will be written under: {ARTIFACT_DIR}")
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
                # Fill these values before running.
                # If your account exposes a different text model alias, replace TEXT_MODEL here.

                SOURCE = ""
                # Examples:
                # SOURCE = "/absolute/path/to/paper.pdf"
                # SOURCE = "/absolute/path/to/notes.tex"
                # SOURCE = "https://arxiv.org/pdf/2401.00001"
                # SOURCE = "https://example.com/math-article"

                API_KEY = API_KEY or ""
                TEXT_MODEL = TEXT_MODEL
                IMAGE_MODEL = IMAGE_MODEL

                # Optional: edit the system prompt for your experiments.
                SYSTEM_PROMPT = SYSTEM_PROMPT

                if not SOURCE:
                    print("Set SOURCE and API_KEY, then run the next cell.")
                else:
                    print(f"Ready to run with source: {SOURCE}")
                    print(f"Text model: {TEXT_MODEL}")
                    print(f"Image model: {IMAGE_MODEL}")
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
                result = run_workflow(
                    source=SOURCE,
                    api_key=API_KEY,
                    text_model=TEXT_MODEL,
                    image_model=IMAGE_MODEL,
                    system_prompt=SYSTEM_PROMPT,
                )

                result["document"]
                """
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
                exported_files = export_augmented_summary(
                    result["blocks"],
                    title=result.get("final_title") or result["document"].title,
                )
                display(IPyMarkdown("## Exported Files"))
                for kind, path in exported_files.items():
                    print(f"{kind}: {path}")

                if "pdf" not in exported_files:
                    print("PDF export skipped because weasyprint is not installed in this notebook environment.")
                """
            ),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
