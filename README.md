---
title: PolicyPulse
emoji: 📊
colorFrom: indigo
colorTo: teal
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# PolicyPulse – Policy Sentiment Intelligence Platform

A Gradio application for analysing user-uploaded CSV, XLSX and XLS datasets with VADER sentiment analysis.

## Scope
- The application analyses datasets uploaded by users.
- It does not have a direct X/Twitter API connection and does not collect live posts continuously.
- Frequent terms are calculated from the uploaded file; they are not live X trends.
- Results describe only the uploaded dataset and are not automatically representative of Nigeria.
- VADER may misinterpret Nigerian Pidgin, slang, sarcasm, irony and context.
- The sample/demo dataset is synthetic and must not be treated as real public opinion.
- The application does not include a permanent database.

## Run
The Space uses the Gradio SDK. Additional Python dependencies are listed in `requirements.txt`.
