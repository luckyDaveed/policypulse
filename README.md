# PolicyPulse — Render deployment package

This package is configured for a Python/Gradio web service on Render.

## Render service settings
- Service type: Web Service
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Instance plan: Free (if available in the account/region)

The app binds to `0.0.0.0` and reads Render's `PORT` environment variable.

## Important limitations
- Render Free web services sleep after 15 minutes without traffic and may take about one minute to wake.
- The free service has an ephemeral filesystem; uploaded data and session files should not be treated as permanent storage.
- This application analyses uploaded datasets. It does not connect directly to X/Twitter or collect live posts continuously.
