"""LifeSci Vision Evaluator: scientific image review and VQA annotation."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lifesci.core import (
    ARTIFACTS, LETTERS, MODALITIES, compare_image_quality, image_metrics, load_image,
    review_flags, score_predictions, sha256_bytes, validate_record,
)
from lifesci.demo import synthetic_field
from lifesci.samples import example_record

st.set_page_config(page_title="LifeSci Vision Evaluator", layout="wide", page_icon="🔬")
st.title("LifeSci Vision Evaluator")
st.caption("Scientific image quality review | evidence-based VQA | reproducible metadata")
st.warning("Research / education prototype only. Not for diagnosis, triage, clinical decisions, or automatic artifact identification. Numeric quality indicators cannot establish image interpretability.")

if "records" not in st.session_state:
    st.session_state.records = []

inspect, annotate, review, compare, guide = st.tabs(["Image QC", "VQA Annotation", "Review & Export", "Compare Image Quality", "Methods & Limits"])

with inspect:
    st.subheader("1. Select an image")
    st.info("Use appropriately licensed and de-identified scientific images. Demonstration images are generated shapes, not genuine tissue, blood, cells, or experiments.")
    src = st.radio("Input", ["Synthetic demonstration", "Local image file"], horizontal=True)
    payload = None
    if src == "Synthetic demonstration":
        sample = st.selectbox("Synthetic image condition", ["baseline", "defocused", "low_contrast", "overexposed"])
        selected_image = synthetic_field(sample)
        buffer = io.BytesIO()
        selected_image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        provenance_default = f"Procedurally generated synthetic demonstration ({sample}); not a laboratory image"
        source_default = "Synthetic demonstration"
    else:
        uploaded = st.file_uploader("PNG, JPEG, TIFF (maximum 25 MB, 24 MP)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if uploaded:
            payload = uploaded.getvalue()
            try:
                selected_image = load_image(payload)
            except ValueError as exc:
                st.error(str(exc))
                payload = None
        provenance_default = uploaded.name if uploaded else ""
        source_default = "Licensed research image"
    if payload is not None:
        st.session_state.image_info = {
            "hash": sha256_bytes(payload), "source_type": source_default,
            "provenance": provenance_default, "metrics": image_metrics(selected_image),
        }
        left, right = st.columns([3, 2])
        with left:
            st.image(selected_image, caption=f"Source: {source_default}", use_container_width=True)
        with right:
            metrics = st.session_state.image_info["metrics"]
            st.metric("Dimensions", f"{metrics['width']} × {metrics['height']}")
            st.metric("Mean luminance (0–1)", f"{metrics['mean_luminance']:.3f}")
            st.metric("5th to 95th percentile contrast", f"{metrics['p05_p95_contrast']:.3f}")
            st.metric("Discrete Laplacian variance", f"{metrics['laplacian_variance']:.5f}")
            contrast = st.slider("Contrast threshold", 0.01, 0.50, 0.15, 0.01)
            edge = st.slider("Edge threshold", 0.0001, 0.1000, 0.003, 0.0001, format="%.4f")
            st.markdown("**Numeric review prompts (not diagnoses)**")
            for flag in review_flags(metrics, contrast_floor=contrast, edge_floor=edge):
                st.write(f"• {flag}")
            st.code(st.session_state.image_info["hash"], language=None)
        st.download_button("Download numeric QC report", json.dumps(st.session_state.image_info, indent=2),
                           file_name="image_qc_report.json", mime="application/json")
    else:
        st.session_state.pop("image_info", None)
        st.info("Select a demonstration or upload an image to inspect.")

with annotate:
    st.subheader("2. Create an image-grounded scientific evaluation record")
    image_info = st.session_state.get("image_info")
    if not image_info:
        st.info("First select an image in the Image QC tab.")
    else:
        st.caption("A draft must cite the visible evidence. Scientific accuracy and image rights need independent verification.")
        with st.form("annotation"):
            rid = st.text_input("Unique question ID", value=f"LS-{len(st.session_state.records)+1:03d}")
            modality = st.selectbox("Modality", MODALITIES)
            sample_type = st.text_input("Specimen or assay type", placeholder="E.g., H&E section, peripheral blood smear; use 'synthetic demonstration' for demo")
            source_type = st.selectbox("Image source type", ["Synthetic demonstration", "Licensed research image", "Own research image"], index=["Synthetic demonstration", "Licensed research image", "Own research image"].index(image_info['source_type']))
            provenance = st.text_input("Provenance, source URI or identifier, acquisition notes", value=image_info["provenance"])
            rights = st.text_input("License / use permission", value="Original synthetic demonstration" if source_type == "Synthetic demonstration" else "")
            description = st.text_area("Detailed image description (35+ words): modality, staining, structures, artifacts, quality", height=130)
            qc_notes = st.text_area("Quality observations: focus, illumination, staining, signal and uncertainty")
            visible = st.multiselect("Artifact labels confirmed or marked as uncertain by reviewer", ARTIFACTS)
            question = st.text_area("Image-dependent question")
            cols = st.columns(2)
            choices = {}
            for index, key in enumerate(LETTERS):
                with cols[index % 2]:
                    choices[key] = st.text_input(f"Option {key}")
            answer = st.selectbox("Gold answer (only after expert confirmation)", LETTERS)
            reasoning = st.text_area("Reasoning: why answer follows from image; address distractors")
            evidence = st.text_area("Visible image evidence (position, appearance, relevant limitation)")
            confirmed = st.checkbox("I verified the image rights and checked there is no identifiable patient information")
            save = st.form_submit_button("Validate and save to this session")
        if save:
            record = dict(id=rid, modality=modality, sample_type=sample_type, source_type=source_type,
                          image_sha256=image_info['hash'], provenance=provenance, rights=rights,
                          image_description=description, quality_observations=qc_notes,
                          visible_artifacts=visible, question=question, choices=choices,
                          answer=answer, reasoning=reasoning, image_evidence=evidence,
                          review_status="Draft", numeric_quality=image_info['metrics'])
            errors = validate_record(record)
            if not confirmed:
                errors.append("Confirm image rights and privacy review")
            if errors:
                st.error("Record needs changes")
                for error in errors:
                    st.write(f"• {error}")
            elif rid in {r['id'] for r in st.session_state.records}:
                st.error("Use a unique record ID; this ID is already in the session")
            else:
                st.session_state.records.append(record)
                st.success("Record passes structural checks and was saved in this browser session. It has not been scientifically reviewed.")

with review:
    st.subheader("3. Dataset audit and evaluation")
    records = st.session_state.records
    st.metric("Saved user-authored records", len(records))
    if records:
        st.dataframe(pd.DataFrame([{k: r[k] for k in ('id','modality','source_type','review_status')} for r in records]), use_container_width=True)
        st.download_button("Export dataset JSON", json.dumps(records, indent=2, ensure_ascii=False),
                           file_name="lifesci_vqa_records.json", mime="application/json")
        st.caption("Export is metadata only. Uploaded images are not included. Store image files separately and retain their SHA-256 hashes to preserve linkage.")
        st.markdown("**Score externally generated model predictions**")
        st.caption("Upload a CSV with exactly two columns: id,prediction. Predictions use A, B, C or D. Missing IDs count as incorrect.")
        pred_file = st.file_uploader("Prediction CSV", type=["csv"], key="prediction_csv")
        if pred_file:
            try:
                data = pred_file.getvalue().decode("utf-8-sig")
                predicted = list(csv.DictReader(io.StringIO(data)))
                report = score_predictions(records, predicted)
                st.metric("Exact-match accuracy", f"{report['accuracy']:.1%}")
                st.dataframe(pd.DataFrame(report['details']), use_container_width=True)
            except (ValueError, UnicodeError, KeyError) as exc:
                st.error(f"Invalid prediction file: {exc}")
    else:
        st.info("Create a VQA record in the annotation tab to export or score predictions.")
    st.markdown("**Illustrative record**")
    st.json(example_record("0" * 64), expanded=False)

with guide:
    st.markdown("""
### Scientific scope and limitations

- **Modalities:** Histopathology, cytology, hematology, molecular imaging, and agricultural microbiology are annotation categories. The app does not determine which modality an image truly represents.
- **QC metrics:** Mean display luminance, a percentile contrast proxy, and discrete Laplacian variance flag potential issues to inspect. Edge variation is strongly affected by staining, magnification, texture and noise. A sharpness threshold cannot be applied blindly across assays.
- **Artifacts:** Debris, contamination, staining inconsistency, motion, noise and failed experiments require an experienced researcher's judgment. The software does not assert that these artifacts are present.
- **Synthetic demonstrations:** Artificial blobs let you test the interface. They are not pathology slides, microscope specimens or experimental measurements.
- **Privacy and rights:** Do not upload identifiable patient data. Use only images for which you have documented rights. This app does not call remote AI APIs; Streamlit hosting choices can still affect where uploaded data is processed.
- **Evaluation:** Use independent human-reviewed gold answers, blind scoring, and provenance records before reporting model performance. Exact-match accuracy alone is not proof of scientific reliability.

### Local reproducibility

```bash
pip install -e '.[dev]'
streamlit run app.py
pytest -q
```
""")


with compare:
    st.subheader("Interactive scientific-image comparison")
    st.caption("Compare numerical image-quality indicators across simulated conditions or your own locally selected images. These are imaging statistics, not microscopy diagnoses.")
    demo_mode = st.toggle("Use generated synthetic examples", value=True)
    labeled_images = []
    if demo_mode:
        labeled_images = [(condition, synthetic_field(condition)) for condition in
                          ("baseline", "defocused", "low_contrast", "overexposed")]
        st.info("These samples are generated illustrations. They are not blood, tissue, clinical specimens, or measured experimental images.")
    else:
        compare_files = st.file_uploader("Choose 2–6 PNG, JPEG, or TIFF images to compare", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                         accept_multiple_files=True, key="quality_compare_files")
        if 2 <= len(compare_files) <= 6:
            for index, file in enumerate(compare_files):
                try:
                    labeled_images.append((f"{index + 1}: {file.name}", load_image(file.getvalue())))
                except ValueError as exc:
                    st.error(f"{file.name}: {exc}")
                    labeled_images = []
                    break
        else:
            st.info("Select between two and six images for a comparison.")
    if labeled_images:
        contrast_cutoff = st.slider("Contrast review threshold", 0.01, 0.50, 0.15, 0.01, key="compare_contrast")
        edge_cutoff = st.slider("Edge variation threshold", 0.0001, 0.10, 0.003, 0.0001, format="%.4f", key="compare_edge")
        comparisons = compare_image_quality(labeled_images, contrast_floor=contrast_cutoff, edge_floor=edge_cutoff)
        table = pd.DataFrame(comparisons)
        st.dataframe(table, use_container_width=True, hide_index=True)
        metric = st.selectbox("Compare metric", ["mean_luminance", "p05_p95_contrast", "laplacian_variance",
                                                       "dark_pixel_fraction", "bright_pixel_fraction"], key="comparison_metric")
        st.bar_chart(table.set_index("condition")[[metric]], use_container_width=True)
        st.download_button("Download comparison as CSV", table.to_csv(index=False).encode("utf-8"),
                           "image_quality_comparison.csv", "text/csv")
        st.caption("Numerical comparisons are not adjusted for differences in stain, magnification, tissue density, image compression, or acquisition settings. Expert review is still required.")
