# PolicyPulse — Secure Render Deployment Package

This package keeps the Gemini API key on the server. The key is read from the `GEMINI_API_KEY` environment variable and is never requested in the public Gradio interface.

## Render deployment

1. Push the contents of this folder to the Git repository connected to your Render Web Service.
2. In Render, open your PolicyPulse service.
3. Go to **Environment**.
4. Add `GEMINI_API_KEY` and paste your Google AI Studio API key as its value.
5. Optionally set `GEMINI_MODEL` to a Gemini model available to your API key. If omitted, the app uses `gemini-flash-latest`.
6. Save changes and trigger a redeploy.

The `render.yaml` file declares `GEMINI_API_KEY` as a secret (`sync: false`). Add the actual value in the Render dashboard; do not put it in this file or in source code.

## Security notes

- Never commit your API key to GitHub or include it in a ZIP shared publicly.
- Do not paste the key into the PolicyPulse chat interface; the interface no longer has an API-key field.
- If the key has ever been exposed publicly, revoke it in Google AI Studio and create a replacement.
- Anyone with access to the Render service's environment settings may be able to manage the secret. Restrict Render workspace access.
- Keep the API key private. The application sends it from the server to Google's Gemini API over HTTPS.

## Features

- CSV, XLSX and XLS dataset upload
- Text-column selection and validation
- URL removal and whitespace normalisation
- VADER sentiment scoring and Positive / Negative / Neutral classification
- Dashboard charts, summary statistics, record preview and CSV export
- Optional Gemini AI assistant, configured securely through Render environment variables

## Research and implementation limitations

- The application analyses datasets uploaded by users. It does not continuously collect live posts from X/Twitter.
- “Real-time” in this prototype means rapid processing after a dataset is uploaded.
- Results describe only the uploaded dataset and are not automatically representative of all Nigerians.
- VADER may misinterpret Nigerian Pidgin, slang, sarcasm, irony and context.
- Uploaded files and session data are not stored in a permanent database. Render Free services may sleep after inactivity and use an ephemeral filesystem.

## Render service settings

- Service type: Web Service
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Plan: Free (subject to current Render availability)
