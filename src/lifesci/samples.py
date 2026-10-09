"""A synthetic-only example record. No diagnostic claims."""

def example_record(image_hash: str) -> dict:
    return {
        "id": "LS-DEMO-001",
        "modality": "Other Life Sciences",
        "sample_type": "Artificial microscopy-like field generated for software demonstration",
        "source_type": "Synthetic demonstration",
        "image_sha256": image_hash,
        "provenance": "Generated locally using lifesci.demo.synthetic_field(seed=12)",
        "rights": "Original synthetic material, MIT repository license",
        "image_description": (
            "This digitally generated square image imitates a microscopy field for interface testing only. "
            "Numerous purple round and oval outlines appear over a pale pink background, with darker centers "
            "and overlapping boundaries. The displayed features are drawn shapes, not genuine cells, "
            "tissue structures, stains, organisms, or experimental observations. No clinical or "
            "biological inference can be drawn from this synthetic scene."
        ),
        "quality_observations": "Edges and contrast can be compared across synthetic variants; values are nonclinical proxies.",
        "visible_artifacts": ["Uncertain / requires expert review"],
        "question": "Which statement is directly supported by the provenance and the appearance of this image?",
        "choices": {
            "A": "It confirms malignant histology",
            "B": "It shows generated cell-like shapes but cannot establish a tissue diagnosis",
            "C": "It proves active bacterial contamination",
            "D": "It confirms a validated hematology assay",
        },
        "answer": "B",
        "reasoning": "The visual features are explicitly computer-generated; neither cell identity nor disease status is established.",
        "image_evidence": "Multiple digitally drawn purple oval boundaries are visible on a pale background.",
        "review_status": "Draft",
    }
