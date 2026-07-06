# OCR Text Occlusion Risk Design

## Goal

Add first-version detection for hand or object occlusion over text in the OCR quality gate. The feature must stay local, low-latency, explainable, configurable, and conservative: occlusion by itself is a risk, not a hard rejection, unless OCR probing also shows poor readability.

## Scope

The first version detects likely occlusion risk from image masks and text-box overlap. It does not call VLMs, remote vision models, or train a custom model. It does not attempt full semantic understanding of the blocking object. It only answers whether suspicious non-text regions are likely covering detected text areas.

## Pipeline Placement

The new layer runs after text detection and text-region summarization, before OCR probe selection and final page decision:

```text
image metrics -> text detection -> text region metrics -> occlusion analysis -> optional OCR probe -> decision
```

Occlusion analysis consumes the analysis image and `TextDetectionResult`. It returns a stable `OcclusionResult` that is attached to the page result as `occlusionSummary`.

## Detection Strategy

The first implementation uses deterministic NumPy/Pillow image processing:

- Convert RGB pixels into a simple skin-likelihood mask using configurable channel rules.
- Build a generic non-background obstruction mask for saturated or low-texture regions that are not likely black text or white page background.
- Use connected components to produce candidate boxes.
- Filter candidates by area ratio, minimum dimensions, and overlap with text areas.
- Compute overlap metrics against expanded text boxes, not against the full page.

The skin-like path produces `HAND_OCCLUSION_RISK`. The generic obstruction path produces `TEXT_OCCLUSION_RISK`.

## Metrics

`OcclusionResult` reports:

- `candidateCount`
- `handLikeCandidateCount`
- `occlusionAreaRatio`
- `textOverlapRatio`
- `affectedTextBoxRatio`
- `maxBoxOverlapRatio`
- `source`

All raw ratios are kept in the result object and rounded only in `to_summary()`.

## Decision Rules

Occlusion risks do not directly reject a page. They enter the existing warning and probe flow:

- If occlusion risk is detected and OCR probe is enabled, the page must run OCR probe even if other risk counts are low.
- If occlusion risk is detected and OCR probe passes, the page returns `ACCEPT_WITH_WARNINGS`.
- If occlusion risk is detected and real OCR probe fails, the page returns `REJECT` through the existing `OCR_PROBE_LOW_CONFIDENCE` hard condition.
- If skin-like or obstruction regions do not overlap text boxes, no occlusion reason code is emitted.

## Configuration

New thresholds live in `QualityConfig` and `config/quality.default.json`:

- `enable_occlusion_detection`
- `occlusion_min_area_ratio`
- `occlusion_min_text_overlap_ratio`
- `occlusion_min_affected_box_ratio`
- `occlusion_min_box_overlap_ratio`
- `occlusion_text_box_padding_ratio`
- `occlusion_skin_min_red`
- `occlusion_skin_min_red_green_delta`
- `occlusion_skin_min_red_blue_delta`

## Result Contract

Add stable reason codes:

- `HAND_OCCLUSION_RISK`
- `TEXT_OCCLUSION_RISK`

Add `occlusionSummary` to each page result. File-level `detectedRisks` continues to aggregate reason codes ending in `_RISK`.

## Testing

Tests use synthetic images so the feature is deterministic:

- Normal text-like image does not report occlusion.
- Skin-colored shape over a text box reports hand occlusion risk.
- Skin-colored shape away from text does not report occlusion.
- Non-skin dark obstruction over a text box reports generic text occlusion risk.
- Occlusion risk forces OCR probe.
- Occlusion risk alone returns warning, not rejection.
- Occlusion risk plus real low-confidence OCR probe rejects.

