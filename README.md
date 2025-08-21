# 🧠 Neuro Insight From MRI & EEG Visual Analytics with LLM Insights

This Streamlit app lets you upload **brain MRI** (NIfTI / DICOM) and **EEG** (EDF / CSV) data, visualize them interactively, and generate **human-friendly insights** using a Large Language Model (LLM).  
By default it uses **Groq’s hosted open-source LLM (`openai/gpt-oss-20b`)** for fast inference, with an optional local HuggingFace fallback.

---

## ✨ Features

### 🧠 MRI Analysis
- Load **NIfTI (.nii/.nii.gz)** or **DICOM series (.zip)**
- Show central orthogonal slices (axial, coronal, sagittal)
- Interactive **slice scroller**
- **Intensity histogram**
- **Maximum Intensity Projections (MIP)**

### ⚡ EEG Analysis
- Load **EDF** or **CSV** EEG recordings
- Compute robust QC features (artifact fraction, band powers, dominant frequency band)
- Visualizations:
  - Raw traces (multi-channel)
  - Power Spectral Density (PSD)
  - Spectrogram
  - Relative band-power bars
  - Alpha topomap (if montage available via MNE)

### 🤖 LLM Insights
- Summarizes extracted features into clear, human-friendly explanations
- Options for **layperson** vs **clinician** tone
- Multilingual support (English, Arabic, Urdu)
- Backends:
  - **Groq (`openai/gpt-oss-20b`)** → fast, hosted
  - **Local HuggingFace model** → offline fallback

### 📑 Reports
- Download complete Markdown report:
  - MRI features (JSON)
  - EEG features (JSON)
  - LLM-generated summary

