# OCR Quality Design

This design implements a generic OCR image-material quality gate. It checks whether uploaded images or rendered PDF pages are usable for downstream text detection, text recognition, and structured parsing. It does not bind to a single document template, fixed field coordinates, remote multimodal models, or self-trained models.

The first version uses a layered pipeline:

- Layer 0 checks file processability: supported type, size, decodability, page count, and image dimensions.
- Layer 1 computes fast image metrics with Pillow and NumPy: blank or black page, effective resolution, exposure, contrast, sharpness, local quality blocks, noise estimate, illumination unevenness, and simple skew proxy.
- Layer 2 exposes a text detection interface. The default implementation is a deterministic lightweight estimator; production deployments can inject a pre-warmed PP-OCR or equivalent adapter without changing the result contract.
- Layer 3 exposes an optional OCR probe interface. It runs only for boundary cases and is disabled by default unless an adapter is supplied.

Hard failures are reserved for reliable conditions such as corrupted files, unsupported formats, blank pages, severe exposure problems, severe blur, extreme low resolution, or no effective text when text is required. Risk items such as local blur, small text, border crop risk, low local contrast, shadow, or glare do not reject by themselves. They enter combination rules or OCR probing.

All thresholds are centralized in versioned configuration. Output objects contain stable decisions, reason codes, user messages, raw metrics, auto-fix records, per-page results, timings, config version, OCR probe status, and file-level aggregation.

