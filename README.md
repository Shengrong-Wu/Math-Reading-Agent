# The Geometric Math Reader

The Geometric Math Reader is a Streamlit app for reading mathematics papers with LLM assistance. It ingests PDFs, LaTeX, Markdown, or URLs, generates a structured summary with LaTeX notation, inserts AI-generated visuals and plots, and lets you refine saved reports in place.

## What It Does

- Reads papers from a URL or uploaded file
- Supports OpenAI, Gemini, Anthropic, and DeepSeek as reasoning providers
- Supports OpenAI or Gemini for image generation
- Produces a report with sections such as outline, key propositions, and sketch proofs
- Adds visual explanations through image prompts and matplotlib plots
- Runs a final QA pass to improve title and visual suitability
- Saves report artifacts so they can be reopened and refined later
- Exports Markdown and HTML

## Repository Structure

- `geometric_math_reader_app/`: Streamlit app and workflow code
- `notebooks/`: notebook prototype for testing the workflow
- `scripts/`: helper script for rebuilding the notebook
- `streamlit_app.py`: root app entrypoint
- `requirements.txt`: root dependency file

## Setup

```bash
cd ~/Math_Reading_Agent
pip install -r requirements.txt
```

Then run:

```bash
streamlit run streamlit_app.py
```

## Usage

1. Open the app in your browser.
2. Choose an input source:
   - upload a PDF, `.tex`, `.md`, or `.txt` file
   - or paste a paper URL
3. Choose a reasoning provider and enter its API key.
4. Load the accessible models and choose the LLM model.
5. Choose an image provider if you want image generation.
6. Optionally edit the system prompt.
7. Click `Generate augmented report`.
8. Review the generated HTML report and export Markdown or HTML if needed.

## Refinement

After a report is generated, you can refine it in place:

- open the `Refine Report` panel
- ask for a targeted rewrite, extra intuition, or more visualizations
- the app updates the saved report instead of creating a full new run

Saved runs are stored under `geometric_math_reader_app/artifacts/`. Newer runs include structured report state, which makes refinement more reliable than editing from export files alone.

## Notes

- API keys are entered in the app UI and are not stored in the repository.
- Local artifact output is ignored by Git.
- The app works best on newly generated runs, because they preserve saved source text and structured block state for later refinement.
