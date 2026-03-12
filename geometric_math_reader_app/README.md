# Geometric Math Reader App

This folder contains the Streamlit app used by the main repository.

## What it does

- Uploads a PDF, LaTeX, Markdown, or text file, or ingests a URL.
- Lets you choose a reasoning provider from OpenAI, Gemini, Anthropic, or DeepSeek.
- Lets you choose a separate image provider from OpenAI or Gemini, or disable images.
- Loads available text and image models after you enter API keys.
- Generates an augmented math summary with inline visuals and a final QA pass.
- Saves each run into `artifacts/` so reports can be reopened and refined later.
- Refines the current report in place through a chatbox instead of regenerating a fresh report on every turn.
- Exports Markdown and HTML.

## Run Locally

```bash
cd ~/Math_Reading_Agent/geometric_math_reader_app
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Image generation is intentionally separate from the main LLM provider because Claude and DeepSeek do not expose the same image-generation path as OpenAI and Gemini.
- The final QA pass is single-shot and focuses on overall report logic and visualization quality, not article-level re-derivation.
- New artifact runs save structured report state for stronger later refinement.
