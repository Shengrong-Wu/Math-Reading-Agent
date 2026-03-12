from __future__ import annotations

import ast
import base64
import html
import json
import math
import mimetypes
import os
import re
import shutil
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import fitz
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image

from .providers import (
    ImageAttachment,
    ProviderConfig,
    ProviderError,
    generate_image,
    generate_text,
    supports_image_generation,
    supports_vision_review,
)

try:
    import markdown as markdown_lib
except ImportError:
    markdown_lib = None


DEFAULT_SYSTEM_PROMPT = """You are a PhD-level Mathematical Assistant. Your goal is to summarize the provided text for a researcher.

Structure: Provide an Outline, Key Propositions, and Sketch Proofs.
Notation: Use strict LaTeX for all mathematical expressions.
Visualization Logic: For geometry-heavy papers, prefer 4-8 visuals when they materially improve understanding. When an important definition introduces a central geometric object, singularity model, or flow, add a short geometric example or visual intuition paragraph and then place a [GENERATE_IMAGE: <minimalist_prompt>] tag immediately after it. After major theorem or proposition statements, insert a visual whenever the statement describes a dichotomy, perturbation, stability mechanism, asymptotic profile, singularity type, or before/after transition.
If the concept is conceptual or geometric (for example, a manifold, a fiber bundle, rescaled mean curvature flow, a neck pinch, or a cylindrical versus spherical singularity), use [GENERATE_IMAGE: <minimalist_prompt>]. Prefer prompts that depict one clean mathematical situation or a short progression, such as smooth surface -> small perturbation -> spherical or nondegenerate cylindrical singularities.
If the concept is functional or quantitative (for example, a vector field, a monotonicity quantity, a profile curve, or a graph of a function), use [GENERATE_PLOT: <matplotlib_python_code>].
Visual Minimalism: Every GENERATE_IMAGE prompt must include: Minimalist diagram, white background, high contrast, no labels, single focus, clean geometric lines. Do not add clutter or text.
Plot Safety: In every GENERATE_PLOT tag, emit runnable matplotlib code only. Do not import math, numpy, or matplotlib; math, np, and plt are already available. Do not call plt.show() or savefig(). Avoid square brackets in the Python code body so the tag can be parsed reliably while streaming; prefer tuples over lists when possible.
Output Format: Return Markdown prose with inline GENERATE tags exactly where the visual should appear. For important definitions and theorem statements, add 1-3 sentences of geometric example or intuition around the visual. Do not wrap the tags in code fences."""

MINIMALIST_IMAGE_SUFFIX = (
    "Minimalist diagram, white background, high contrast, no labels, single focus, "
    "clean geometric lines. Do not add clutter or text."
)
TAG_PATTERN = re.compile(r"\[(GENERATE_IMAGE|GENERATE_PLOT):\s*(.*?)\]", re.DOTALL)
TAG_PREFIX = "[GENERATE_"
FENCED_JSON_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
INVALID_JSON_ESCAPE_PATTERN = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')
MARKDOWN_IMAGE_PATTERN = re.compile(r"^!\[(?P<alt>.*?)\]\((?P<path>.+?)\)$")
ITALIC_CAPTION_PATTERN = re.compile(r"^\*(.+)\*$")
STATE_FILENAME = "report_state.json"
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
SAFE_PLOT_IMPORT_LINES = {
    "import math",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
}
MATH_TOKEN_PATTERN = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$[^$\n].*?(?<!\\)\$)",
    re.DOTALL,
)
TEX_NOISE_PATTERNS = (
    r"^\\newcommand\b",
    r"^\\renewcommand\b",
    r"^\\providecommand\b",
    r"^\\DeclareMathOperator\b",
    r"^\\def\b",
    r"^\\let\b",
    r"^\\usepackage\b",
    r"^\\documentclass\b",
    r"^\\input\b",
    r"^\\include\b",
    r"^\\bibliographystyle\b",
    r"^\\bibliography\b",
    r"^\\setlength\b",
    r"^\\theoremstyle\b",
    r"^\\newtheorem\b",
)


@dataclass
class IngestedDocument:
    source: str
    kind: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    title: str
    document: IngestedDocument
    blocks: list[dict[str, Any]]
    markdown: str
    html: str
    image_count: int
    plot_count: int
    review: dict[str, Any] | None
    export_paths: dict[str, Path]
    run_dir: Path


@dataclass
class RefinementResult:
    title: str
    blocks: list[dict[str, Any]]
    markdown: str
    html: str
    chat_history: list[dict[str, str]]
    applied_operations: list[dict[str, Any]]
    export_paths: dict[str, Path]


@dataclass
class LoadedArtifactResult:
    workflow_result: WorkflowResult
    chat_history: list[dict[str, str]]
    last_refinement: list[dict[str, Any]] | None


def run_workflow(
    *,
    source_url: str | None,
    upload_name: str | None,
    upload_bytes: bytes | None,
    llm_config: ProviderConfig,
    image_config: ProviderConfig | None,
    app_dir: Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    run_final_review: bool = True,
    status_callback: Callable[[str], None] | None = None,
) -> WorkflowResult:
    _status(status_callback, "Preparing artifact directories...")
    run_dir = _create_run_directory(app_dir)
    document = ingest_source(
        source_url=source_url,
        upload_name=upload_name,
        upload_bytes=upload_bytes,
        run_dir=run_dir,
    )

    _status(status_callback, f"Generating summary with {llm_config.provider}:{llm_config.model}...")
    summary_text = generate_text(
        llm_config,
        system_prompt=system_prompt,
        user_prompt=build_summary_prompt(document),
        json_mode=False,
        temperature=0.3,
    )

    blocks: list[dict[str, Any]] = [
        {
            "type": "markdown",
            "content": f"# {document.title}\n\n_Source:_ `{document.source}`\n\n",
        }
    ]
    plot_count = 0

    _status(status_callback, "Rendering visuals and plots...")
    events, pending = parse_stream_chunk(summary_text, "")
    for event in events:
        if event[0] == "text":
            append_markdown(blocks, event[1])
            continue

        _, tag_name, payload = event
        if tag_name == "GENERATE_IMAGE":
            _append_generated_image(
                blocks,
                prompt=payload,
                image_config=image_config,
                image_dir=run_dir / "images",
                error_prefix="Image generation failed",
            )
        elif tag_name == "GENERATE_PLOT":
            try:
                plot_path = execute_plot_code(payload, plot_dir=run_dir / "plots")
                blocks.append(
                    {
                        "type": "image",
                        "path": plot_path,
                        "caption": "AI-generated matplotlib plot",
                        "origin": "plot",
                    }
                )
                plot_count += 1
            except Exception as exc:  # pragma: no cover - defensive runtime surface
                blocks.append({"type": "placeholder", "message": f"Plot generation failed: {exc}"})

    for event in flush_stream_buffer(pending):
        append_markdown(blocks, event[1])

    review_data: dict[str, Any] | None = None
    final_title = document.title
    if run_final_review:
        _status(status_callback, "Running one final QA pass...")
        try:
            review_data = review_report(llm_config, blocks)
            final_title = apply_final_review(
                blocks=blocks,
                review_data=review_data,
                fallback_title=document.title,
                image_config=image_config,
                image_dir=run_dir / "images",
            )
        except Exception as exc:  # pragma: no cover - provider/runtime dependent
            blocks.append({"type": "placeholder", "message": f"Final QA pass failed: {exc}"})

    export_dir = run_dir / "exports"
    markdown_text = blocks_to_markdown(blocks, export_root=export_dir)
    html_text = blocks_to_html(blocks, title=final_title)
    export_paths = write_exports(
        markdown_text=markdown_text,
        html_text=html_text,
        title=final_title,
        export_dir=export_dir,
    )

    image_count = sum(1 for block in blocks if block["type"] == "image")
    _status(status_callback, "Report ready.")
    result = WorkflowResult(
        title=final_title,
        document=document,
        blocks=blocks,
        markdown=markdown_text,
        html=html_text,
        image_count=image_count,
        plot_count=plot_count,
        review=review_data,
        export_paths=export_paths,
        run_dir=run_dir,
    )
    save_report_state(result, chat_history=[], last_refinement=None)
    return result


def refine_report(
    *,
    workflow_result: WorkflowResult,
    llm_config: ProviderConfig,
    image_config: ProviderConfig | None,
    user_request: str,
    chat_history: list[dict[str, str]] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> RefinementResult:
    history = list(chat_history or [])
    _status(status_callback, "Planning targeted report edits...")
    plan = build_refinement_plan(
        llm_config=llm_config,
        title=workflow_result.title,
        document=workflow_result.document,
        blocks=workflow_result.blocks,
        user_request=user_request,
        chat_history=history,
    )

    blocks = clone_blocks(workflow_result.blocks)
    _status(status_callback, "Applying report edits...")
    applied_operations = apply_edit_operations(
        blocks=blocks,
        operations=plan.get("operations", []),
        image_config=image_config,
        image_dir=workflow_result.run_dir / "images",
        plot_dir=workflow_result.run_dir / "plots",
    )

    if plan.get("title"):
        update_report_title(blocks, str(plan["title"]))

    title = extract_report_title(blocks) or workflow_result.title
    markdown_text = blocks_to_markdown(blocks, export_root=workflow_result.export_paths["md"].parent)
    html_text = blocks_to_html(blocks, title=title)
    export_paths = overwrite_exports(
        markdown_path=workflow_result.export_paths["md"],
        html_path=workflow_result.export_paths["html"],
        markdown_text=markdown_text,
        html_text=html_text,
    )

    assistant_summary = str(plan.get("assistant_summary") or "Applied targeted report edits.").strip()
    history.append({"role": "user", "content": user_request})
    history.append({"role": "assistant", "content": assistant_summary})

    updated_result = WorkflowResult(
        title=title,
        document=workflow_result.document,
        blocks=blocks,
        markdown=markdown_text,
        html=html_text,
        image_count=sum(1 for block in blocks if block["type"] == "image"),
        plot_count=sum(1 for block in blocks if block.get("origin") in {"plot", "chat-plot"}),
        review=workflow_result.review,
        export_paths=export_paths,
        run_dir=workflow_result.run_dir,
    )
    save_report_state(updated_result, chat_history=history, last_refinement=applied_operations)

    _status(status_callback, "Refinement complete.")
    return RefinementResult(
        title=title,
        blocks=blocks,
        markdown=markdown_text,
        html=html_text,
        chat_history=history,
        applied_operations=applied_operations,
        export_paths=export_paths,
    )


def ingest_source(
    *,
    source_url: str | None,
    upload_name: str | None,
    upload_bytes: bytes | None,
    run_dir: Path,
) -> IngestedDocument:
    if upload_name and upload_bytes is not None:
        return ingest_uploaded_file(upload_name=upload_name, upload_bytes=upload_bytes, run_dir=run_dir)
    if source_url and source_url.strip():
        return ingest_url(source_url.strip())
    raise ValueError("Provide either a source URL or an uploaded file.")


def ingest_uploaded_file(*, upload_name: str, upload_bytes: bytes, run_dir: Path) -> IngestedDocument:
    uploads_dir = run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / Path(upload_name).name
    path.write_bytes(upload_bytes)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        kind = "pdf"
        text = extract_text_from_pdf_path(path)
    elif suffix in {".tex", ".md", ".markdown", ".txt"}:
        kind = suffix.lstrip(".")
        text = clean_document_text(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        raise ValueError(f"Unsupported upload type: {suffix}")

    return IngestedDocument(
        source=str(path),
        kind=kind,
        title=path.stem,
        text=text,
        metadata={"path": str(path)},
    )


def ingest_url(url: str) -> IngestedDocument:
    headers = {"User-Agent": "GeometricMathReaderApp/0.1"}
    response = requests.get(url, timeout=60, headers=headers)
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
    try:
        reader_response = requests.get(build_jina_reader_url(url), timeout=60, headers=headers)
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


def build_summary_prompt(document: IngestedDocument) -> str:
    trimmed_text = document.text[:120_000]
    truncated_note = ""
    if len(document.text) > len(trimmed_text):
        truncated_note = (
            "The source text was truncated before sending to the model. Prioritize the most central "
            "definitions, propositions, and proofs that remain in the excerpt.\n\n"
        )

    return textwrap.dedent(
        f"""
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
        """
    ).strip()


def review_report(llm_config: ProviderConfig, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    report_markdown, attachments = build_review_payload(blocks, vision_enabled=supports_vision_review(llm_config.provider))
    review_prompt = textwrap.dedent(
        """
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
        """
    ).strip()

    response_text = generate_text(
        llm_config,
        system_prompt="You are a careful mathematical report reviewer.",
        user_prompt=f"{review_prompt}\n\nReport for review:\n\n{report_markdown}",
        attachments=attachments,
        json_mode=True,
        temperature=0.2,
    )
    return parse_json_response(response_text)


def build_refinement_plan(
    *,
    llm_config: ProviderConfig,
    title: str,
    document: IngestedDocument,
    blocks: list[dict[str, Any]],
    user_request: str,
    chat_history: list[dict[str, str]],
) -> dict[str, Any]:
    report_snapshot = build_refinement_block_context(blocks)
    source_context = build_source_refinement_context(document.text, user_request)
    history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in chat_history[-6:])
    prompt = textwrap.dedent(
        f"""
        You are a report editor for a mathematical reading assistant.

        You must edit an existing report in place instead of regenerating the whole report.
        The user may ask for field-specific emphasis, extra intuition, more or fewer visuals, or section rewrites.

        Constraints:
        - Preserve correct existing content when possible.
        - Prefer targeted edits over broad rewrites.
        - Work from the saved report and saved source snippets instead of regenerating the whole document.
        - If the user asks for a new visual, insert it near the relevant markdown block using an image prompt or plot code.
        - Only use plots when the request is clearly quantitative. Use image prompts for geometric or conceptual content.
        - Keep the current title, source line, and unaffected sections unless the user clearly asks to change them.
        - When editing a section, use the full nearby block text below. Do not invent article details that are not supported by the saved report or source snippets.
        - Return JSON only.

        Allowed operation types:
        - replace_markdown: replace one markdown block's content
        - append_markdown: append extra markdown to an existing markdown block
        - insert_markdown_after: insert a new markdown block after a target block
        - insert_image_after: insert a generated image after a target block using a prompt
        - insert_plot_after: insert a generated plot after a target block using matplotlib code
        - delete_block: remove a block
        - replace_image: replace an existing image with a new image prompt

        Current title: {title}

        Existing report blocks with full content:
        {report_snapshot}

        Relevant saved source snippets:
        {source_context or "No saved source snippets available for this request. Work from the report content only."}

        Recent refinement chat:
        {history_text or "No previous refinement turns."}

        New user request:
        {user_request}

        Return JSON with this schema:
        {{
          "title": string,
          "assistant_summary": string,
          "operations": [
            {{
              "type": string,
              "target_block_id": integer,
              "content": string,
              "prompt": string,
              "code": string
            }}
          ]
        }}

        For operations, include only the fields that are needed by that type.
        Every JSON string must be valid JSON. Escape literal backslashes inside LaTeX or TeX snippets by doubling them, for example write \\\\alpha inside JSON strings.
        """
    ).strip()

    response_text = generate_text(
        llm_config,
        system_prompt="You are a precise structured editor. Return JSON only.",
        user_prompt=prompt,
        json_mode=True,
        temperature=0.2,
    )
    return parse_json_response(response_text)


def apply_final_review(
    *,
    blocks: list[dict[str, Any]],
    review_data: dict[str, Any],
    fallback_title: str,
    image_config: ProviderConfig | None,
    image_dir: Path,
) -> str:
    final_title = str(review_data.get("title") or fallback_title).strip() or fallback_title
    update_report_title(blocks, final_title)

    image_block_indices = [index for index, block in enumerate(blocks) if block["type"] == "image"]
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
            path = write_image_bytes(
                image_dir=image_dir,
                prompt=prompt,
                image_config=image_config,
                prefix="review-image",
            )
            blocks[block_index]["path"] = path
            blocks[block_index]["caption"] = prompt
            reason = str(item.get("reason") or "Improved visual alignment.").strip()
            replacement_notes.append(f"- Replaced visual {visual_index}: {reason}")
        except Exception as exc:  # pragma: no cover - runtime/provider dependent
            replacement_notes.append(f"- Failed to replace visual {visual_index}: {exc}")

    qa_lines = ["## Final QA Pass"]
    summary_assessment = str(review_data.get("summary_assessment") or "").strip()
    if summary_assessment:
        qa_lines.append(summary_assessment)

    logic_warnings = [str(item).strip() for item in review_data.get("logic_warnings", []) if str(item).strip()]
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

    blocks.append({"type": "markdown", "content": "\n\n".join(qa_lines).strip() + "\n"})

    extra_visuals = review_data.get("additional_visuals", [])[:2]
    if extra_visuals:
        blocks.append({"type": "markdown", "content": "## Final Visual Additions\n"})
    for item in extra_visuals:
        placement_hint = str(item.get("placement_hint") or "Later in the report").strip()
        reason = str(
            item.get("reason") or "Additional visualization requested by the final QA pass."
        ).strip()
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        blocks.append(
            {
                "type": "markdown",
                "content": f"### Suggested placement: {placement_hint}\n\n{reason}\n",
            }
        )
        try:
            path = write_image_bytes(
                image_dir=image_dir,
                prompt=prompt,
                image_config=image_config,
                prefix="review-extra",
            )
            blocks.append({"type": "image", "path": path, "caption": prompt, "origin": "review-image"})
        except Exception as exc:  # pragma: no cover - runtime/provider dependent
            blocks.append({"type": "placeholder", "message": f"Final QA visual generation failed: {exc}"})

    return final_title


def build_review_payload(
    blocks: list[dict[str, Any]],
    *,
    vision_enabled: bool,
) -> tuple[str, list[ImageAttachment]]:
    parts: list[str] = []
    attachments: list[ImageAttachment] = []
    visual_index = 0
    for block in blocks:
        if block["type"] == "markdown":
            parts.append(block["content"].rstrip())
            continue
        if block["type"] == "placeholder":
            parts.append(f"[PLACEHOLDER: {block['message']}]")
            continue
        if block["type"] == "image":
            visual_index += 1
            caption = block.get("caption", "generated visual")
            parts.append(f"[VISUAL {visual_index}: {caption}]")
            if vision_enabled:
                path = Path(block["path"])
                mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                attachments.append(ImageAttachment(mime_type=mime_type, data=path.read_bytes(), caption=caption))

    return "\n\n".join(part for part in parts if part).strip(), attachments


def parse_json_response(text: str) -> dict[str, Any]:
    for candidate in iter_json_candidates(text):
        for parser in (json.loads, load_json_with_escape_repair):
            try:
                loaded = parser(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
            raise ValueError("Expected a JSON object response from the model.")
    raise ValueError("Model response did not contain valid JSON.")


def iter_json_candidates(text: str) -> list[str]:
    cleaned = FENCED_JSON_PATTERN.sub("", text.strip())
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    return candidates


def load_json_with_escape_repair(text: str) -> dict[str, Any]:
    repaired = INVALID_JSON_ESCAPE_PATTERN.sub(r"\\\\", text)
    return json.loads(repaired)


def clone_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for block in blocks:
        copied = dict(block)
        if "path" in copied:
            copied["path"] = Path(copied["path"])
        cloned.append(copied)
    return cloned


def build_refinement_block_context(blocks: list[dict[str, Any]], *, max_chars: int = 28_000) -> str:
    parts: list[str] = []
    total = 0
    for index, block in enumerate(blocks, start=1):
        if block["type"] == "markdown":
            block_body = block["content"].strip()
        elif block["type"] == "image":
            origin = str(block.get("origin") or "image")
            block_body = f"caption: {block.get('caption', '')}\norigin: {origin}"
        else:
            block_body = str(block.get("message") or "").strip()

        chunk = f"--- BLOCK {index} ({block['type']}) ---\n{block_body}\n"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 120:
                parts.append(chunk[:remaining].rstrip() + "\n[TRUNCATED]")
            parts.append("\n--- CONTEXT TRUNCATED ---")
            break
        parts.append(chunk.rstrip())
        total += len(chunk)
    return "\n\n".join(parts)


def build_source_refinement_context(source_text: str, user_request: str, *, max_snippets: int = 3) -> str:
    if not source_text.strip():
        return ""

    keywords = extract_refinement_keywords(user_request)
    if not keywords:
        excerpt = source_text[:1400].strip()
        return excerpt if excerpt else ""

    lowered = source_text.lower()
    snippets: list[str] = []
    seen_spans: list[tuple[int, int]] = []
    for keyword in keywords[:6]:
        start = lowered.find(keyword)
        if start == -1:
            continue
        span = (max(0, start - 450), min(len(source_text), start + 850))
        if any(not (span[1] <= left or span[0] >= right) for left, right in seen_spans):
            continue
        seen_spans.append(span)
        snippet = source_text[span[0] : span[1]].strip()
        if snippet:
            snippets.append(f"[keyword: {keyword}]\n{snippet}")
        if len(snippets) >= max_snippets:
            break
    return "\n\n".join(snippets)


def extract_refinement_keywords(user_request: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", user_request.lower()):
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def describe_blocks_for_editing(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, block in enumerate(blocks, start=1):
        block_type = block["type"]
        if block_type == "markdown":
            preview = re.sub(r"\s+", " ", block["content"]).strip()
            lines.append(f"[{index}] markdown: {preview[:600]}")
        elif block_type == "image":
            caption = block.get("caption", "generated image")
            lines.append(f"[{index}] image: {caption}")
        elif block_type == "placeholder":
            lines.append(f"[{index}] placeholder: {block['message']}")
    return "\n".join(lines)


def extract_report_title(blocks: list[dict[str, Any]]) -> str:
    for block in blocks:
        if block["type"] != "markdown":
            continue
        lines = block["content"].splitlines()
        if lines and lines[0].startswith("# "):
            return lines[0][2:].strip()
    return ""


def apply_edit_operations(
    *,
    blocks: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    image_config: ProviderConfig | None,
    image_dir: Path,
    plot_dir: Path,
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for operation in operations[:12]:
        op_type = str(operation.get("type") or "").strip()
        if not op_type:
            continue
        target_index = resolve_target_index(blocks, operation.get("target_block_id"))
        if op_type == "replace_markdown" and target_index is not None and blocks[target_index]["type"] == "markdown":
            blocks[target_index]["content"] = str(operation.get("content") or "").strip() + "\n"
            applied.append({"type": op_type, "target_block_id": target_index + 1})
        elif op_type == "append_markdown" and target_index is not None and blocks[target_index]["type"] == "markdown":
            addition = str(operation.get("content") or "").strip()
            if addition:
                blocks[target_index]["content"] = blocks[target_index]["content"].rstrip() + "\n\n" + addition + "\n"
                applied.append({"type": op_type, "target_block_id": target_index + 1})
        elif op_type == "insert_markdown_after" and target_index is not None:
            content = str(operation.get("content") or "").strip()
            if content:
                blocks.insert(target_index + 1, {"type": "markdown", "content": content + "\n"})
                applied.append({"type": op_type, "target_block_id": target_index + 1})
        elif op_type == "insert_image_after" and target_index is not None:
            prompt = str(operation.get("prompt") or "").strip()
            if prompt:
                try:
                    path = write_image_bytes(image_dir=image_dir, prompt=prompt, image_config=image_config, prefix="chat-image")
                    blocks.insert(target_index + 1, {"type": "image", "path": path, "caption": prompt, "origin": "chat-image"})
                    applied.append({"type": op_type, "target_block_id": target_index + 1, "prompt": prompt})
                except Exception as exc:
                    blocks.insert(target_index + 1, {"type": "placeholder", "message": f"Chat image generation failed: {exc}"})
                    applied.append({"type": op_type, "target_block_id": target_index + 1, "error": str(exc)})
        elif op_type == "insert_plot_after" and target_index is not None:
            code = str(operation.get("code") or "").strip()
            if code:
                try:
                    path = execute_plot_code(code, plot_dir=plot_dir)
                    blocks.insert(
                        target_index + 1,
                        {
                            "type": "image",
                            "path": path,
                            "caption": "AI-generated matplotlib plot",
                            "origin": "chat-plot",
                        },
                    )
                    applied.append({"type": op_type, "target_block_id": target_index + 1})
                except Exception as exc:
                    blocks.insert(target_index + 1, {"type": "placeholder", "message": f"Chat plot generation failed: {exc}"})
                    applied.append({"type": op_type, "target_block_id": target_index + 1, "error": str(exc)})
        elif op_type == "delete_block" and target_index is not None:
            deleted = blocks.pop(target_index)
            applied.append({"type": op_type, "target_block_id": target_index + 1, "deleted_type": deleted["type"]})
        elif op_type == "replace_image" and target_index is not None and blocks[target_index]["type"] == "image":
            prompt = str(operation.get("prompt") or "").strip()
            if prompt:
                try:
                    path = write_image_bytes(image_dir=image_dir, prompt=prompt, image_config=image_config, prefix="chat-replace")
                    blocks[target_index]["path"] = path
                    blocks[target_index]["caption"] = prompt
                    applied.append({"type": op_type, "target_block_id": target_index + 1, "prompt": prompt})
                except Exception as exc:
                    applied.append({"type": op_type, "target_block_id": target_index + 1, "error": str(exc)})
    return applied


def resolve_target_index(blocks: list[dict[str, Any]], target_block_id: Any) -> int | None:
    try:
        index = int(target_block_id) - 1
    except (TypeError, ValueError):
        return None
    if 0 <= index < len(blocks):
        return index
    return None


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


def sanitize_plot_code(plot_code: str) -> str:
    cleaned_lines = []
    for line in plot_code.splitlines():
        if line.strip() in SAFE_PLOT_IMPORT_LINES:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


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


def execute_plot_code(plot_code: str, *, plot_dir: Path) -> Path:
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

    plot_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.gcf()
    figure.tight_layout()
    output_path = plot_dir / f"plot-{uuid.uuid4().hex[:8]}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return output_path


def blocks_to_markdown(
    blocks: list[dict[str, Any]],
    export_root: Path | None = None,
    *,
    embed_images: bool = False,
) -> str:
    parts: list[str] = []
    assets_dir = None
    if export_root is not None:
        assets_dir = export_root / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        if block["type"] == "markdown":
            parts.append(block["content"].rstrip())
        elif block["type"] == "image":
            path = Path(block["path"])
            caption = str(block.get("caption", "generated image"))
            if embed_images:
                data_uri = image_path_to_data_uri(path)
                safe_caption = html.escape(caption)
                parts.append(
                    f"<figure><img src=\"{data_uri}\" alt=\"{safe_caption}\" />"
                    f"<figcaption>{safe_caption}</figcaption></figure>"
                )
                continue
            image_ref = str(path)
            if export_root is not None:
                copied_path = copy_export_asset(path=path, assets_dir=assets_dir)
                image_ref = os.path.relpath(copied_path, start=export_root)
            parts.append(f"![{caption}]({image_ref})")
            if caption:
                parts.append(f"*{caption}*")
        elif block["type"] == "placeholder":
            parts.append(f"> {block['message']}")
    return "\n\n".join(parts).strip() + "\n"


def blocks_to_html(blocks: list[dict[str, Any]], *, title: str) -> str:
    body_parts: list[str] = []
    for block in blocks:
        if block["type"] == "markdown":
            body_parts.append(markdown_to_html(block["content"]))
        elif block["type"] == "image":
            data_uri = image_path_to_data_uri(Path(block["path"]))
            caption = html.escape(block.get("caption", ""))
            body_parts.append(
                f"<figure><img src='{data_uri}' alt='{caption}' /><figcaption>{caption}</figcaption></figure>"
            )
        elif block["type"] == "placeholder":
            body_parts.append(f"<blockquote>{html.escape(block['message'])}</blockquote>")

    body_html = "\n".join(body_parts)
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --page-width: 880px;
      --text: #101418;
      --muted: #5f6b76;
      --line: #dbe2e8;
      --surface: #ffffff;
      --bg: #f6f7f9;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 28%),
        linear-gradient(180deg, #fcfdfd 0%, var(--bg) 100%);
      color: var(--text);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      line-height: 1.75;
    }}
    main {{
      max-width: var(--page-width);
      margin: 0 auto;
      padding: 48px 20px 96px;
    }}
    article {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 40px 44px;
      box-shadow: 0 16px 40px rgba(9, 22, 30, 0.06);
    }}
    h1, h2, h3, h4 {{
      line-height: 1.2;
      margin-top: 1.6em;
      letter-spacing: -0.02em;
    }}
    h1 {{
      margin-top: 0;
      font-size: 2.3rem;
    }}
    h2 {{
      font-size: 1.65rem;
      color: #0b3b37;
    }}
    p, li {{
      font-size: 1.05rem;
    }}
    img {{
      display: block;
      width: 100%;
      max-width: 780px;
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
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
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
"""


def markdown_to_html(markdown_text: str) -> str:
    if markdown_lib is None:
        return f"<pre>{html.escape(markdown_text)}</pre>"
    protected_text, replacements = protect_math_segments(markdown_text)
    rendered = markdown_lib.markdown(
        protected_text,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )
    return restore_math_segments(rendered, replacements)


def image_path_to_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def copy_export_asset(*, path: Path, assets_dir: Path | None) -> Path:
    if assets_dir is None:
        return path
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / path.name
    if not target.exists() or path.stat().st_mtime > target.stat().st_mtime:
        shutil.copy2(path, target)
    return target


def protect_math_segments(markdown_text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = f"@@MATH_TOKEN_{len(replacements)}@@"
        replacements[token] = match.group(0)
        return token

    protected = MATH_TOKEN_PATTERN.sub(_replace, markdown_text)
    return protected, replacements


def restore_math_segments(rendered_html: str, replacements: dict[str, str]) -> str:
    restored = rendered_html
    for token, math_text in replacements.items():
        restored = restored.replace(token, math_text)
    return restored


def write_exports(*, markdown_text: str, html_text: str, title: str, export_dir: Path) -> dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    markdown_path = export_dir / f"{slug}.md"
    html_path = export_dir / f"{slug}.html"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return {"md": markdown_path, "html": html_path}


def overwrite_exports(*, markdown_path: Path, html_path: Path, markdown_text: str, html_text: str) -> dict[str, Path]:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return {"md": markdown_path, "html": html_path}


def save_report_state(
    workflow_result: WorkflowResult,
    *,
    chat_history: list[dict[str, str]],
    last_refinement: list[dict[str, Any]] | None,
) -> Path:
    state_path = workflow_result.run_dir / STATE_FILENAME
    payload = {
        "version": 1,
        "title": workflow_result.title,
        "document": serialize_document(workflow_result.document),
        "blocks": serialize_blocks(workflow_result.blocks, root=workflow_result.run_dir),
        "markdown": workflow_result.markdown,
        "image_count": workflow_result.image_count,
        "plot_count": workflow_result.plot_count,
        "review": workflow_result.review,
        "export_paths": {
            kind: os.path.relpath(path, start=workflow_result.run_dir)
            for kind, path in workflow_result.export_paths.items()
        },
        "chat_history": chat_history,
        "last_refinement": last_refinement,
    }
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return state_path


def serialize_document(document: IngestedDocument) -> dict[str, Any]:
    return {
        "source": document.source,
        "kind": document.kind,
        "title": document.title,
        "text": document.text,
        "metadata": document.metadata,
    }


def serialize_blocks(blocks: list[dict[str, Any]], *, root: Path) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for block in blocks:
        item = dict(block)
        if "path" in item:
            item["path"] = os.path.relpath(Path(item["path"]), start=root)
        serialized.append(item)
    return serialized


def deserialize_document(data: dict[str, Any]) -> IngestedDocument:
    return IngestedDocument(
        source=str(data.get("source") or ""),
        kind=str(data.get("kind") or "artifact"),
        title=str(data.get("title") or "Loaded report"),
        text=str(data.get("text") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def deserialize_blocks(data: list[dict[str, Any]], *, root: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in data:
        item = dict(block)
        if "path" in item:
            item["path"] = (root / str(item["path"])).resolve()
        blocks.append(item)
    return blocks


def list_saved_runs(app_dir: Path) -> list[dict[str, Any]]:
    artifacts_dir = app_dir / "artifacts"
    if not artifacts_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(
        (path for path in artifacts_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        title, has_state = inspect_saved_run(run_dir)
        runs.append(
            {
                "run_dir": run_dir,
                "title": title,
                "label": f"{run_dir.name} | {title}",
                "has_state": has_state,
            }
        )
    return runs


def inspect_saved_run(run_dir: Path) -> tuple[str, bool]:
    state_path = run_dir / STATE_FILENAME
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            title = str(payload.get("title") or run_dir.name).strip() or run_dir.name
            return title, True
        except Exception:
            pass

    markdown_path = first_export_path(run_dir, suffix=".md")
    if markdown_path is None:
        return run_dir.name, False
    title = extract_title_from_markdown(markdown_path.read_text(encoding="utf-8", errors="ignore"))
    return title or run_dir.name, False


def load_saved_run(run_dir: Path) -> LoadedArtifactResult:
    state_path = run_dir / STATE_FILENAME
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        blocks = deserialize_blocks(list(payload.get("blocks") or []), root=run_dir)
        export_paths = {
            kind: (run_dir / str(rel_path)).resolve()
            for kind, rel_path in dict(payload.get("export_paths") or {}).items()
        }
        markdown_text = str(payload.get("markdown") or "")
        html_path = export_paths.get("html")
        html_text = (
            html_path.read_text(encoding="utf-8", errors="ignore")
            if html_path is not None and html_path.exists()
            else blocks_to_html(blocks, title=str(payload.get("title") or run_dir.name))
        )
        workflow_result = WorkflowResult(
            title=str(payload.get("title") or run_dir.name),
            document=deserialize_document(dict(payload.get("document") or {})),
            blocks=blocks,
            markdown=markdown_text,
            html=html_text,
            image_count=int(payload.get("image_count") or 0),
            plot_count=int(payload.get("plot_count") or 0),
            review=payload.get("review"),
            export_paths=export_paths,
            run_dir=run_dir,
        )
        return LoadedArtifactResult(
            workflow_result=workflow_result,
            chat_history=list(payload.get("chat_history") or []),
            last_refinement=payload.get("last_refinement"),
        )

    return load_legacy_saved_run(run_dir)


def load_legacy_saved_run(run_dir: Path) -> LoadedArtifactResult:
    markdown_path = first_export_path(run_dir, suffix=".md")
    if markdown_path is None:
        raise FileNotFoundError(f"No exported Markdown file found in {run_dir}.")
    html_path = first_export_path(run_dir, suffix=".html")

    markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_exported_markdown(markdown_text, base_dir=markdown_path.parent)
    title = extract_report_title(blocks) or extract_title_from_markdown(markdown_text) or run_dir.name
    html_text = (
        html_path.read_text(encoding="utf-8", errors="ignore")
        if html_path is not None
        else blocks_to_html(blocks, title=title)
    )
    document = IngestedDocument(
        source=extract_source_from_markdown(markdown_text) or str(markdown_path),
        kind="artifact",
        title=title,
        text="",
        metadata={"loaded_from": str(markdown_path), "legacy_artifact": True},
    )
    workflow_result = WorkflowResult(
        title=title,
        document=document,
        blocks=blocks,
        markdown=markdown_text,
        html=html_text,
        image_count=sum(1 for block in blocks if block["type"] == "image"),
        plot_count=sum(
            1
            for block in blocks
            if block["type"] == "image" and Path(block["path"]).name.startswith("plot-")
        ),
        review=None,
        export_paths={
            "md": markdown_path.resolve(),
            "html": html_path.resolve() if html_path is not None else (run_dir / "exports" / f"{slugify(title)}.html"),
        },
        run_dir=run_dir,
    )
    return LoadedArtifactResult(workflow_result=workflow_result, chat_history=[], last_refinement=None)


def first_export_path(run_dir: Path, *, suffix: str) -> Path | None:
    exports_dir = run_dir / "exports"
    if not exports_dir.exists():
        return None
    candidates = sorted(
        (path for path in exports_dir.iterdir() if path.is_file() and path.suffix.lower() == suffix),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_exported_markdown(markdown_text: str, *, base_dir: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    markdown_lines: list[str] = []
    lines = markdown_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        image_match = MARKDOWN_IMAGE_PATTERN.match(line.strip())
        if image_match:
            flush_markdown_buffer(blocks, markdown_lines)
            caption = image_match.group("alt")
            path = (base_dir / image_match.group("path")).resolve()
            next_index = index + 1
            if next_index < len(lines):
                caption_match = ITALIC_CAPTION_PATTERN.match(lines[next_index].strip())
                if caption_match:
                    caption = caption_match.group(1).strip()
                    index += 1
            blocks.append({"type": "image", "path": path, "caption": caption, "origin": infer_block_origin(path)})
            index += 1
            continue
        if line.startswith("> "):
            flush_markdown_buffer(blocks, markdown_lines)
            blocks.append({"type": "placeholder", "message": line[2:].strip()})
            index += 1
            continue
        markdown_lines.append(lines[index])
        index += 1

    flush_markdown_buffer(blocks, markdown_lines)
    return blocks


def flush_markdown_buffer(blocks: list[dict[str, Any]], markdown_lines: list[str]) -> None:
    if not markdown_lines:
        return
    content = "\n".join(markdown_lines).strip()
    markdown_lines.clear()
    if content:
        blocks.append({"type": "markdown", "content": content + "\n"})


def extract_title_from_markdown(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_source_from_markdown(markdown_text: str) -> str:
    match = re.search(r"_Source:_\s*`([^`]+)`", markdown_text)
    return match.group(1).strip() if match else ""


def infer_block_origin(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("plot-"):
        return "plot"
    if name.startswith("chat-image"):
        return "chat-image"
    if name.startswith("chat-replace"):
        return "chat-replace"
    if name.startswith("review-"):
        return "review-image"
    return "image"


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
            block["content"] = "\n".join(lines)
            return
        block["content"] = f"# {clean_title}\n\n" + block["content"]
        return


def write_image_bytes(
    *,
    image_dir: Path,
    prompt: str,
    image_config: ProviderConfig | None,
    prefix: str,
) -> Path:
    if image_config is None:
        raise ProviderError("No image provider is configured.")
    if not supports_image_generation(image_config.provider):
        raise ProviderError(f"Image generation is not supported for provider: {image_config.provider}")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = generate_image(image_config, prompt=normalize_image_prompt(prompt))
    output_path = image_dir / f"{prefix}-{uuid.uuid4().hex[:8]}.png"
    output_path.write_bytes(image_bytes)
    return output_path


def normalize_image_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return MINIMALIST_IMAGE_SUFFIX
    if "Minimalist diagram" in prompt:
        return prompt
    if prompt.endswith("."):
        return f"{prompt} {MINIMALIST_IMAGE_SUFFIX}"
    return f"{prompt}. {MINIMALIST_IMAGE_SUFFIX}"


def clean_document_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = strip_tex_noise(text)
    return text.strip()


def strip_tex_noise(text: str) -> str:
    cleaned_lines: list[str] = []
    previous_normalized = ""
    repeated_count = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            previous_normalized = ""
            repeated_count = 0
            continue

        if is_tex_noise_line(line):
            continue

        normalized = re.sub(r"\s+", " ", line)
        if normalized == previous_normalized:
            repeated_count += 1
            if repeated_count >= 2:
                continue
        else:
            previous_normalized = normalized
            repeated_count = 0

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def is_tex_noise_line(line: str) -> bool:
    if any(re.match(pattern, line) for pattern in TEX_NOISE_PATTERNS):
        return True

    # Drop extracted macro soup such as repeated \operatorname or command aliases.
    if line.count("\\") >= 3 and (
        "\\operatorname" in line
        or "\\newcommand" in line
        or "\\DeclareMathOperator" in line
        or "\\def" in line
    ):
        return True

    # Heuristic for lines dominated by TeX command tokens rather than prose.
    backslash_count = line.count("\\")
    alpha_count = sum(char.isalpha() for char in line)
    space_count = line.count(" ")
    if backslash_count >= 4 and alpha_count > 0 and space_count <= 2:
        return True

    return False


def build_jina_reader_url(url: str) -> str:
    stripped = url.split("://", 1)[-1]
    return f"https://r.jina.ai/http://{stripped}"


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return clean_document_text("\n\n".join(page.get_text("text") for page in doc))


def extract_text_from_pdf_path(path: Path) -> str:
    with fitz.open(path) as doc:
        return clean_document_text("\n\n".join(page.get_text("text") for page in doc))


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
    return clean_document_text("\n\n".join(lines_out))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or f"document-{uuid.uuid4().hex[:8]}"


def _create_run_directory(app_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = app_dir / "artifacts" / f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_generated_image(
    blocks: list[dict[str, Any]],
    *,
    prompt: str,
    image_config: ProviderConfig | None,
    image_dir: Path,
    error_prefix: str,
) -> None:
    try:
        path = write_image_bytes(image_dir=image_dir, prompt=prompt, image_config=image_config, prefix="image")
        blocks.append({"type": "image", "path": path, "caption": prompt, "origin": "image"})
    except Exception as exc:  # pragma: no cover - runtime/provider dependent
        blocks.append({"type": "placeholder", "message": f"{error_prefix}: {exc}"})


def _status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
