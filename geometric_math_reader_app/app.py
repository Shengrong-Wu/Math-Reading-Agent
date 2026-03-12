from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from math_reader import (
    DEFAULT_SYSTEM_PROMPT,
    IMAGE_PROVIDERS,
    LLM_PROVIDERS,
    ProviderConfig,
    blocks_to_markdown,
    discover_models,
    list_saved_runs,
    load_saved_run,
    refine_report,
    run_workflow,
)


APP_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Startup-speed helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _cached_saved_runs(app_dir: Path) -> list[dict]:
    """Scan the artifacts folder at most once every 30 s instead of on every rerun."""
    return list_saved_runs(app_dir)


# ---------------------------------------------------------------------------
# Secrets helpers — pre-fill API keys from .streamlit/secrets.toml when present
# ---------------------------------------------------------------------------

_PROVIDER_SECRET_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _secret_for_provider(provider: str) -> str:
    """Return the API key stored in st.secrets for *provider*, or '' if absent."""
    secret_name = _PROVIDER_SECRET_KEYS.get(provider, "")
    if not secret_name:
        return ""
    try:
        return st.secrets.get(secret_name, "")  # type: ignore[return-value]
    except Exception:
        return ""


st.set_page_config(
    page_title="Geometric Math Reader",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1450px;
      }
      [data-testid="stSidebar"] {
        background:
          radial-gradient(circle at top, rgba(15, 118, 110, 0.08), transparent 30%),
          linear-gradient(180deg, #fbfcfc 0%, #f5f7f8 100%);
      }
      .app-shell {
        padding: 1.2rem 0 0.6rem;
      }
      .app-kicker {
        color: #0f766e;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }
      .app-title {
        font-size: 2.4rem;
        line-height: 1.05;
        margin: 0.2rem 0 0.75rem;
      }
      .app-copy {
        max-width: 780px;
        color: #425466;
        font-size: 1.05rem;
      }
    </style>
    <div class="app-shell">
      <div class="app-kicker">Math Reading Agent</div>
      <div class="app-title">The Geometric Math Reader</div>
      <div class="app-copy">
        Generate an augmented report once, then refine it in place with a chatbox. The chat editor
        patches the current report instead of regenerating a new file tree on every turn.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _load_models_into_state(*, state_key: str, provider: str, api_key: str, purpose: str, base_url: str | None) -> None:
    st.session_state[state_key] = discover_models(provider, api_key, purpose=purpose, base_url=base_url)


def _provider_base_url(provider: str, raw_value: str) -> str | None:
    if provider not in {"openai", "deepseek"}:
        return None
    value = raw_value.strip()
    return value or None


def _selected_model(options: list[str], dropdown_value: str, override: str) -> str:
    return override.strip() or dropdown_value or (options[0] if options else "")


def _build_llm_config(
    *,
    provider: str,
    api_key: str,
    dropdown_value: str,
    override: str,
    options: list[str],
    base_url_raw: str,
) -> ProviderConfig:
    if not api_key.strip():
        raise ValueError("Enter an LLM API key.")
    model = _selected_model(options, dropdown_value, override)
    if not model:
        raise ValueError("Choose an LLM model or enter a custom model id.")
    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=_provider_base_url(provider, base_url_raw),
    )


def _build_image_config(
    *,
    provider: str,
    api_key: str,
    dropdown_value: str,
    override: str,
    options: list[str],
    base_url_raw: str,
) -> ProviderConfig | None:
    if provider == "disabled":
        return None
    if not api_key.strip():
        raise ValueError("Enter an image API key or disable image generation.")
    model = _selected_model(options, dropdown_value, override)
    if not model:
        raise ValueError("Choose an image model or enter a custom image model id.")
    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=_provider_base_url(provider, base_url_raw),
    )


def _save_runtime_config(llm_config: ProviderConfig, image_config: ProviderConfig | None, system_prompt: str) -> None:
    st.session_state["runtime_llm_config"] = llm_config
    st.session_state["runtime_image_config"] = image_config
    st.session_state["runtime_system_prompt"] = system_prompt


def _resolve_runtime_configs(
    *,
    llm_provider: str,
    llm_api_key: str,
    llm_model: str,
    llm_model_override: str,
    llm_models: list[str],
    llm_base_url: str,
    image_provider: str,
    image_api_key: str,
    image_model: str,
    image_model_override: str,
    image_models: list[str],
    image_base_url: str,
) -> tuple[ProviderConfig, ProviderConfig | None]:
    runtime_llm = st.session_state.get("runtime_llm_config")
    runtime_image = st.session_state.get("runtime_image_config")

    try:
        llm_config = _build_llm_config(
            provider=llm_provider,
            api_key=llm_api_key,
            dropdown_value=llm_model,
            override=llm_model_override,
            options=llm_models,
            base_url_raw=llm_base_url,
        )
    except Exception:
        if runtime_llm is None:
            raise
        llm_config = runtime_llm

    try:
        image_config = _build_image_config(
            provider=image_provider,
            api_key=image_api_key,
            dropdown_value=image_model,
            override=image_model_override,
            options=image_models,
            base_url_raw=image_base_url,
        )
    except Exception:
        if image_provider == "disabled":
            image_config = None
        elif runtime_image is None:
            raise
        else:
            image_config = runtime_image

    return llm_config, image_config


def _render_chat_history(chat_history: list[dict[str, str]]) -> None:
    if not chat_history:
        st.caption("No refinement turns yet.")
        return
    for item in chat_history:
        with st.chat_message("assistant" if item["role"] == "assistant" else "user"):
            st.markdown(item["content"])


with st.sidebar:
    st.header("Source")
    source_mode = st.radio("Input type", options=("Upload file", "URL"), horizontal=True)
    source_url = ""
    uploaded_file = None
    if source_mode == "Upload file":
        uploaded_file = st.file_uploader("Paper file", type=["pdf", "tex", "md", "markdown", "txt"])
    else:
        source_url = st.text_input("Paper URL", placeholder="https://arxiv.org/pdf/...")

    st.divider()
    st.header("Saved Reports")
    saved_runs = _cached_saved_runs(APP_DIR)
    selected_saved_run = ""
    if saved_runs:
        saved_run_options = [str(item["run_dir"]) for item in saved_runs]
        saved_run_labels = {str(item["run_dir"]): item["label"] for item in saved_runs}
        selected_saved_run = st.selectbox(
            "Artifact run",
            options=saved_run_options,
            format_func=lambda value: saved_run_labels.get(value, value),
        )
        load_saved_clicked = st.button("Load saved report", use_container_width=True)
        if selected_saved_run:
            selected_info = next(item for item in saved_runs if str(item["run_dir"]) == selected_saved_run)
            st.caption("Structured state available." if selected_info["has_state"] else "Legacy export only. Refinement uses the saved Markdown plus images.")
    else:
        load_saved_clicked = False
        st.caption("No saved reports in `artifacts/` yet.")

    st.divider()
    st.header("Reasoning Provider")
    llm_provider = st.selectbox("LLM provider", options=LLM_PROVIDERS, format_func=str.title)
    llm_api_key = st.text_input("LLM API key", type="password", value=_secret_for_provider(llm_provider))
    llm_base_url = ""
    if llm_provider in {"openai", "deepseek"}:
        llm_base_url = st.text_input("LLM base URL (optional)", placeholder="Leave blank for the official endpoint")

    if st.button("Load LLM models", use_container_width=True):
        try:
            _load_models_into_state(
                state_key="llm_models",
                provider=llm_provider,
                api_key=llm_api_key,
                purpose="text",
                base_url=_provider_base_url(llm_provider, llm_base_url),
            )
            st.success("Loaded LLM models.")
        except Exception as exc:
            st.error(str(exc))

    llm_models = st.session_state.get("llm_models", [])
    llm_model = st.selectbox("LLM model", options=llm_models or [""], disabled=not llm_models)
    llm_model_override = st.text_input("Custom LLM model (optional)", placeholder="Type a model id if it is not listed")

    st.divider()
    st.header("Image Provider")
    image_provider = st.selectbox(
        "Image provider",
        options=IMAGE_PROVIDERS,
        format_func=lambda value: value.title() if value != "disabled" else "Disabled",
    )
    reuse_llm_key = st.checkbox("Reuse LLM credentials when provider matches", value=True)
    if image_provider == "disabled":
        image_api_key = ""
        image_base_url = ""
    else:
        default_image_key = llm_api_key if reuse_llm_key and image_provider == llm_provider else _secret_for_provider(image_provider)
        image_api_key = st.text_input("Image API key", type="password", value=default_image_key)
        image_base_url = ""
        if image_provider == "openai":
            image_base_url = st.text_input("Image base URL (optional)", placeholder="Leave blank for the official endpoint")

        if st.button("Load image models", use_container_width=True):
            try:
                _load_models_into_state(
                    state_key="image_models",
                    provider=image_provider,
                    api_key=image_api_key,
                    purpose="image",
                    base_url=_provider_base_url(image_provider, image_base_url),
                )
                st.success("Loaded image models.")
            except Exception as exc:
                st.error(str(exc))

    image_models = st.session_state.get("image_models", [])
    image_model = (
        st.selectbox("Image model", options=image_models or [""], disabled=image_provider == "disabled" or not image_models)
        if image_provider != "disabled"
        else ""
    )
    image_model_override = (
        st.text_input("Custom image model (optional)", placeholder="Type a model id if it is not listed")
        if image_provider != "disabled"
        else ""
    )

    st.divider()
    st.header("Workflow")
    run_final_review = st.checkbox("Run final QA pass", value=True)
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=320)
    run_clicked = st.button("Generate augmented report", type="primary", use_container_width=True)


status_box = st.empty()

if load_saved_clicked and selected_saved_run:
    try:
        loaded = load_saved_run(Path(selected_saved_run))
        st.session_state["workflow_result"] = loaded.workflow_result
        st.session_state["chat_history"] = loaded.chat_history
        st.session_state["last_refinement"] = loaded.last_refinement
        status_box.success(f"Loaded saved report from {Path(selected_saved_run).name}.")
    except Exception as exc:
        status_box.error(str(exc))

if run_clicked:
    try:
        if source_mode == "Upload file" and uploaded_file is None:
            raise ValueError("Upload a source file before running the app.")
        if source_mode == "URL" and not source_url.strip():
            raise ValueError("Enter a source URL before running the app.")

        llm_config = _build_llm_config(
            provider=llm_provider,
            api_key=llm_api_key,
            dropdown_value=llm_model,
            override=llm_model_override,
            options=llm_models,
            base_url_raw=llm_base_url,
        )
        image_config = _build_image_config(
            provider=image_provider,
            api_key=image_api_key,
            dropdown_value=image_model,
            override=image_model_override,
            options=image_models,
            base_url_raw=image_base_url,
        )

        result = run_workflow(
            source_url=source_url if source_mode == "URL" else None,
            upload_name=uploaded_file.name if uploaded_file is not None else None,
            upload_bytes=uploaded_file.getvalue() if uploaded_file is not None else None,
            llm_config=llm_config,
            image_config=image_config,
            app_dir=APP_DIR,
            system_prompt=system_prompt,
            run_final_review=run_final_review,
            status_callback=lambda message: status_box.info(message),
        )

        st.session_state["workflow_result"] = result
        st.session_state["chat_history"] = []
        st.session_state["last_refinement"] = None
        _save_runtime_config(llm_config, image_config, system_prompt)
        status_box.success("Generation finished.")
    except Exception as exc:
        status_box.error(str(exc))


result = st.session_state.get("workflow_result")
if result:
    downloadable_markdown = blocks_to_markdown(result.blocks, embed_images=True)

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Final title", result.title)
    meta_col2.metric("Image blocks", result.image_count)
    meta_col3.metric("Plot blocks", result.plot_count)
    st.caption(f"Source: {result.document.source}")

    components.html(result.html, height=1350, scrolling=True)

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download Markdown",
        data=downloadable_markdown,
        file_name=result.export_paths["md"].name,
        mime="text/markdown",
        use_container_width=True,
        help="This download embeds generated images directly into the Markdown so it stays portable.",
    )
    download_col2.download_button(
        "Download HTML",
        data=result.html,
        file_name=result.export_paths["html"].name,
        mime="text/html",
        use_container_width=True,
    )

    with st.expander("Refine Report", expanded=False):
        st.caption("Use the chatbox to patch the current report in place. The app overwrites the current HTML/Markdown export instead of creating a new report run.")
        _render_chat_history(st.session_state.get("chat_history", []))

        refinement_prompt = st.chat_input("Ask for a targeted revision, field-specific emphasis, or extra visualization")
        if refinement_prompt:
            try:
                llm_config, image_config = _resolve_runtime_configs(
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    llm_model_override=llm_model_override,
                    llm_models=llm_models,
                    llm_base_url=llm_base_url,
                    image_provider=image_provider,
                    image_api_key=image_api_key,
                    image_model=image_model,
                    image_model_override=image_model_override,
                    image_models=image_models,
                    image_base_url=image_base_url,
                )
                _save_runtime_config(llm_config, image_config, system_prompt)

                refinement = refine_report(
                    workflow_result=result,
                    llm_config=llm_config,
                    image_config=image_config,
                    user_request=refinement_prompt,
                    chat_history=st.session_state.get("chat_history", []),
                    status_callback=lambda message: status_box.info(message),
                )

                result.blocks = refinement.blocks
                result.title = refinement.title
                result.markdown = refinement.markdown
                result.html = refinement.html
                result.export_paths = refinement.export_paths
                result.image_count = sum(1 for block in refinement.blocks if block["type"] == "image")
                result.plot_count = sum(1 for block in refinement.blocks if block.get("origin") in {"plot", "chat-plot"})

                st.session_state["workflow_result"] = result
                st.session_state["chat_history"] = refinement.chat_history
                st.session_state["last_refinement"] = refinement.applied_operations
                status_box.success("Applied targeted report edits.")
                st.rerun()
            except Exception as exc:
                status_box.error(str(exc))

        last_refinement = st.session_state.get("last_refinement")
        if last_refinement:
            st.markdown("**Last applied edit plan**")
            st.json(last_refinement)

    with st.expander("Markdown output", expanded=False):
        st.code(result.markdown, language="markdown")

    with st.expander("Run details", expanded=False):
        st.write(
            {
                "run_dir": str(result.run_dir),
                "document_kind": result.document.kind,
                "review_present": bool(result.review),
                "exports": {kind: str(path) for kind, path in result.export_paths.items()},
            }
        )
else:
    st.info("Configure the providers in the sidebar, generate a report, then refine it in the chatbox.")
