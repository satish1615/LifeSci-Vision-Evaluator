# LifeSci Vision Evaluator

A **working Streamlit prototype** for reviewing scientific image quality and authoring structured visual question-answering (VQA) records. It is designed for practice with research workflows in **histopathology, cytology, hematology, molecular imaging and agricultural microbiology**. This is a general image annotation workbench, **not a medically validated system**.

## Features

- Upload PNG/JPEG/TIFF imagery, or choose locally generated **synthetic demonstration** images.
- Display image with deterministic and inspectable image-quality proxies: luminance, percentile contrast, edge variance and numeric review flags.
- Create a structured VQA annotation with four answer options, gold answer, evidence, explanation, source rights, SHA-256 hash, image description, quality notes and user-reviewed artifact labels.
- Validate record completeness; export JSON metadata and compare external model predictions from a CSV against manually authored answers.
- Transparent safeguards: no disease prediction, no automatic contamination detector, no remote LLM usage, no patient examples.
- Tests for numeric behavior, image decoding, record validation, and prediction scoring.

## Start locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
streamlit run app.py
```

Browse `http://localhost:8501`. Run tests with `pytest -q`.

For Streamlit Community Cloud, deploy this repository and choose `app.py` as the entrypoint. The application adds the local `src` directory to its import path, so the package also imports when Streamlit Community Cloud installs only `requirements.txt`.

## Research accuracy and privacy

**Synthetic means synthetic.** The generated cell-like figures are graphical test images and cannot substantiate specimen type, pathology, diagnosis, presence of contamination, staining quality or any measured experiment. Quality metrics are basic image statistics that produce *review prompts*, not artifact classifiers. Manual validation checks formatting, **not the scientific truth** of an answer. Do not upload identifiable health information; store external image licenses and permissions.

Task metadata downloads do not include the associated image, only a SHA-256 ID. For reproducibility keep the licensed original separately and document modality, acquisition, magnification, stain, sample preparation and human review. An independent domain expert should review descriptions and questions before research publication.

## Tools

Python, Streamlit, NumPy, Pillow, Pandas, pytest. No paid API keys needed.

## Roadmap

- Pixel/region annotations and polygon export for expert-audited evidence.
- Optional supported image datasets with retained license, references and assay metadata.
- Expert consensus and calibration for artifact grading.
- Stratified held-out evaluation by assay and quality characteristic.

## License

MIT; synthetic assets are illustrative rather than biological reference imagery.
