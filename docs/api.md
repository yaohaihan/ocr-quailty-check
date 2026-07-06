# API

## `GET /`

Returns the simple upload page for manually testing image and PDF quality checks.

## `GET /health`

Returns service health and the active threshold configuration version.

Example response:

```json
{
  "status": "ok",
  "thresholdConfigVersion": "ocr-quality-v2-paddle"
}
```

## `GET /api/quality/config`

Returns the current quality configuration summary, including supported extensions, file limits, aggregation policy, and threshold fields.

## `POST /api/quality/check`

Runs the quality pipeline for one uploaded file.

Request:

- Content type: `multipart/form-data`
- File field: `file`
- Supported inputs: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.pdf`

PowerShell example:

```powershell
$form = @{ file = Get-Item 'path\to\page.png' }
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/quality/check' -Method Post -Form $form
```

The response is the same unified result object used by the CLI and core pipeline.
