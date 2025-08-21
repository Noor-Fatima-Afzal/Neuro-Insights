# app.py — MRI+EEG Visual Analytics with Groq LLM (fast) and optional Local fallback
# -----------------------------------------------------------------------------
# Quick setup:
#   pip install -U streamlit nibabel mne numpy scipy pandas pillow SimpleITK matplotlib
#   pip install -U groq                 # for Groq hosted LLM (fast)
#   # Optional local fallback:
#   pip install -U torch transformers accelerate bitsandbytes
#
# Run:
#   # Set your Groq key (or paste it in the sidebar field):
#   # Linux/Mac: export GROQ_API_KEY=your_key
#   # Windows:   setx GROQ_API_KEY your_key
#   streamlit run app.py
#
# Notes:
# - Educational demo; NOT medical advice. Use anonymized data only.
# - Visuals: MRI (slices, scroller, histogram, MIPs), EEG (raw, PSD, spectrogram, band bars, topomap if montage).
# - LLM: Groq ("openai/gpt-oss-20b") by default; Local HF as optional fallback.
# -----------------------------------------------------------------------------

import os, io, zipfile, tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Soft deps
_has_nib = _has_sitk = _has_mne = _has_torch = _has_hf = _has_groq = False
try:
    import nibabel as nib; _has_nib = True
except Exception: pass
try:
    import SimpleITK as sitk; _has_sitk = True
except Exception: pass
try:
    import mne; _has_mne = True
except Exception: pass
try:
    import torch; _has_torch = True
except Exception: pass
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    _has_hf = True
except Exception: pass
try:
    from groq import Groq
    _has_groq = True
except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neuro Insight", layout="wide", page_icon="🧠")
st.title("🧠 Neuro Insight From MRI & EEG with Visual Analysis")
st.caption("Runs with Groq’s hosted OSS LLM for speed. Educational only — **not** medical advice.")

with st.expander("What this app does / doesn’t do"):
    st.markdown(
        "- Loads anonymized MRI (NIfTI or DICOM ZIP) and/or EEG (EDF/CSV).\n"
        "- Computes lightweight QC/summary features; **no** diagnostic model.\n"
        "- Visualizes MRI/EEG interactively.\n"
        "- Uses Groq’s **openai/gpt-oss-20b** to turn metrics into human-friendly insights.\n"
        "- Optional local model fallback if you prefer fully offline."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: LLM settings (Groq + optional Local)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("LLM Settings")
backend = st.sidebar.selectbox("Backend", ["Groq (fast, recommended)", "Local HF (offline fallback)"])

# Groq settings
groq_model = st.sidebar.text_input("Groq model", value="openai/gpt-oss-20b")
groq_api_key_input = st.sidebar.text_input("GROQ_API_KEY (optional if set in env)", type="password")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
max_new_tokens = st.sidebar.slider("Max output tokens", 128, 4096, 800, 64)

# Local settings (only used if Local HF selected)
default_local_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_model = st.sidebar.text_input("Local HF model (fallback)", value=default_local_model)
use_4bit = st.sidebar.toggle("Local: 4-bit quantization", value=True)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95, 0.05)

audience = st.sidebar.selectbox("Audience tone", ["Plain language (layperson)", "Clinician-style (technical)"])
lang = st.sidebar.selectbox("Language", ["English", "عربي (Arabic)", "اردو (Urdu)"])

# Apply Groq key from sidebar if provided
if groq_api_key_input:
    os.environ["GROQ_API_KEY"] = groq_api_key_input

# ─────────────────────────────────────────────────────────────────────────────
# Groq LLM client
# ─────────────────────────────────────────────────────────────────────────────
def call_groq_stream(system_msg: str, user_msg: str,
                     model: str = "openai/gpt-oss-20b",
                     temperature: float = 0.3, max_tokens: int = 1024,
                     placeholder=None) -> str:
    if not _has_groq:
        raise RuntimeError("groq package not installed. Install with: pip install groq")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set (env) or provided in sidebar.")
    client = Groq()  # uses env var
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    acc = ""
    for chunk in completion:
        delta = ""
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        acc += delta
        if placeholder is not None and delta:
            placeholder.markdown(acc)
    return acc.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Local HF fallback (optional)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def get_local_pipe(model_name: str, use_4bit: bool):
    if not _has_hf:
        raise RuntimeError("Install transformers to use local fallback: pip install -U transformers accelerate")
    quant = None
    if use_4bit and _has_torch:
        try:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                       bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception:
            quant = None
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = dict(trust_remote_code=True, device_map="auto")
    if _has_torch and torch.cuda.is_available() and quant is None:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if quant is not None:
        kwargs["quantization_config"] = quant
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return pipeline("text-generation", model=mdl, tokenizer=tok), tok

def chat_generate(pipe, tok, system: str, user: str, **gen_kwargs) -> str:
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"
    out = pipe(
        prompt,
        max_new_tokens=gen_kwargs.get("max_new_tokens", 800),
        do_sample=(gen_kwargs.get("temperature", 0.3) > 0),
        temperature=gen_kwargs.get("temperature", 0.3),
        top_p=gen_kwargs.get("top_p", 0.95),
        eos_token_id=tok.eos_token_id
    )[0]["generated_text"]
    if "[ASSISTANT]" in out:
        out = out.split("[ASSISTANT]", 1)[-1].strip()
    elif out.startswith(prompt):
        out = out[len(prompt):].strip()
    return out.strip()

# ─────────────────────────────────────────────────────────────────────────────
# MRI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_slice(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    if p99 <= p1:
        p1, p99 = x.min(), max(x.max(), x.min()+1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)

def load_nifti(file_bytes: bytes):
    if not _has_nib: raise RuntimeError("Install nibabel")
    # More robust: write to temp file to ensure nibabel compatibility
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush()
        path = tmp.name
    img = nib.load(path)
    try:
        data = img.get_fdata(dtype=np.float32)
    finally:
        try: os.remove(path)
        except Exception: pass
    hdr = img.header
    zooms = hdr.get_zooms()[:3] if hdr is not None else (np.nan, np.nan, np.nan)
    return data, zooms, img.affine

def load_dicom_zip(zip_bytes: bytes):
    if not _has_sitk: raise RuntimeError("Install SimpleITK")
    with tempfile.TemporaryDirectory() as tmpd:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf: zf.extractall(tmpd)
        reader = sitk.ImageSeriesReader()
        sids = reader.GetGDCMSeriesIDs(tmpd)
        if not sids: raise RuntimeError("No DICOM series found in ZIP.")
        best = max(sids, key=lambda sid: len(reader.GetGDCMSeriesFileNames(tmpd, sid)))
        files = reader.GetGDCMSeriesFileNames(tmpd, best)
        reader.SetFileNames(files); img = reader.Execute()
        vol = sitk.GetArrayFromImage(img)  # [z,y,x]
        spacing = img.GetSpacing()         # (x,y,z)
        vol = np.transpose(np.moveaxis(vol, 0, 2), (1, 0, 2))  # -> [x,y,z]
        return vol.astype(np.float32), (spacing[0], spacing[1], spacing[2]), None

def central_slices(vol: np.ndarray):
    sx, sy, sz = np.array(vol.shape) // 2
    return (
        Image.fromarray(_normalize_slice(vol[:, :, sz])),
        Image.fromarray(_normalize_slice(vol[:, sy, :])),
        Image.fromarray(_normalize_slice(vol[sx, :, :]))
    )

def mri_features(vol: np.ndarray, zooms: Tuple[float, float, float]) -> Dict:
    vox = np.prod(zooms) if all(z > 0 for z in zooms) else np.nan
    mu, sd = float(np.mean(vol)), float(np.std(vol) + 1e-6)
    return {
        "shape_xyz": list(map(int, vol.shape)),
        "voxel_size_mm": [float(z) for z in zooms],
        "voxel_volume_mm3": float(vox) if np.isfinite(vox) else None,
        "intensity_mean": mu,
        "intensity_std": float(sd),
        "intensity_p01_p99": [float(np.percentile(vol, 1)), float(np.percentile(vol, 99))],
        "high_intensity_outlier_frac": float(np.mean(vol > (mu + 3*sd))),
        "low_intensity_outlier_frac": float(np.mean(vol < (mu - 3*sd))),
    }

# MRI visuals
def plot_mri_hist(vol: np.ndarray):
    fig, ax = plt.subplots()
    v = vol[np.isfinite(vol)].ravel()
    ax.hist(v, bins=100)
    ax.set_title("MRI Intensity Histogram")
    ax.set_xlabel("Intensity"); ax.set_ylabel("Voxel count")
    st.pyplot(fig, use_container_width=True)

def plot_mip(vol: np.ndarray):
    mip_ax = _normalize_slice(np.max(vol, axis=2))
    mip_cor = _normalize_slice(np.max(vol, axis=1))
    mip_sag = _normalize_slice(np.max(vol, axis=0))
    c1, c2, c3 = st.columns(3)
    c1.image(mip_ax, caption="MIP — Axial", use_container_width=True)
    c2.image(mip_cor, caption="MIP — Coronal", use_container_width=True)
    c3.image(mip_sag, caption="MIP — Sagittal", use_container_width=True)

def slice_scroller(vol: np.ndarray):
    axis = st.radio("Scroll axis", ["Axial (z)", "Coronal (y)", "Sagittal (x)"], horizontal=True)
    if "Axial" in axis:
        idx = st.slider("Slice (z)", 0, vol.shape[2]-1, vol.shape[2]//2)
        img = _normalize_slice(vol[:, :, idx])
    elif "Coronal" in axis:
        idx = st.slider("Slice (y)", 0, vol.shape[1]-1, vol.shape[1]//2)
        img = _normalize_slice(vol[:, idx, :])
    else:
        idx = st.slider("Slice (x)", 0, vol.shape[0]-1, vol.shape[0]//2)
        img = _normalize_slice(vol[idx, :, :])
    st.image(img, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# EEG helpers
# ─────────────────────────────────────────────────────────────────────────────
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}

def load_eeg_edf(file_bytes: bytes):
    if not _has_mne: raise RuntimeError("Install mne")
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes); tmp.flush()
        path = tmp.name
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    finally:
        try: os.remove(path)
        except Exception: pass
    return raw

def load_eeg_csv(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "time" in df.columns:
        t = df["time"].values.astype(float); dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
        sfreq = float(1.0 / dt) if dt > 0 else 200.0
        data = df.drop(columns=["time"]).to_numpy(dtype=np.float64).T
        ch_names = list(df.columns.drop("time"))
    else:
        sfreq = 200.0; data = df.to_numpy(dtype=np.float64).T; ch_names = list(df.columns)
    if _has_mne:
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        return mne.io.RawArray(data, info, verbose=False)
    return {"data": data, "sfreq": sfreq, "ch_names": ch_names, "_kind": "numpy"}

def eeg_features(raw_obj) -> Dict:
    if _has_mne and hasattr(raw_obj, "copy"):
        raw = raw_obj.copy().load_data()
        sfreq = float(raw.info["sfreq"]); ch_names = list(raw.ch_names)
        data = raw.get_data(); duration = float(raw.n_times / sfreq)
    else:
        data = raw_obj["data"]; sfreq = float(raw_obj["sfreq"])
        ch_names = raw_obj["ch_names"]; duration = float(data.shape[1] / sfreq)

    med = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - med), axis=1, keepdims=True) + 1e-6
    robust_sigma = 1.4826 * mad
    artifact_frac = float(np.mean(np.abs(data - med) > (5.0 * robust_sigma)))

    def welch_psd(x, fs, nperseg=1024):
        step = nperseg // 2
        if x.size < nperseg:
            pad = np.zeros(nperseg, dtype=np.float64); pad[:x.size] = x; x = pad
        segs = []
        for start in range(0, x.size - nperseg + 1, step):
            segs.append(np.fft.rfft(x[start:start+nperseg] * np.hanning(nperseg)))
        if not segs: return np.array([0.0]), np.array([0.0])
        S = np.mean([np.abs(s)**2 for s in segs], axis=0)
        freqs = np.fft.rfftfreq(nperseg, 1.0/fs)
        return freqs, S

    k = min(8, len(ch_names)); idxs = np.linspace(0, len(ch_names)-1, k, dtype=int)
    psd_sum, freqs = None, None
    for i in idxs:
        f, P = welch_psd(data[i].astype(np.float64), sfreq, nperseg=min(2048, int(sfreq*4)))
        psd_sum = P if psd_sum is None else (psd_sum + P); freqs = f
    psd_avg = psd_sum / max(1, k)
    total_power = float(np.trapz(psd_avg, freqs)) if freqs is not None else 1.0

    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi) if freqs is not None else np.array([False])
        power = float(np.trapz(psd_avg[mask], freqs[mask])) if mask.any() else 0.0
        band_powers[name] = {"abs": power, "rel": float(power / (total_power + 1e-9))}
    dom_band = max(band_powers.items(), key=lambda kv: kv[1]["rel"])[0] if band_powers else None

    return {
        "n_channels": int(len(ch_names)),
        "sfreq_hz": float(sfreq),
        "duration_sec": float(duration),
        "artifact_fraction": float(artifact_frac),
        "band_powers": band_powers,
        "dominant_band": dom_band,
        "_cache": {"freqs": freqs, "psd_avg": psd_avg, "ch_names": ch_names, "data": data, "sfreq": sfreq}
    }

# EEG visuals
def plot_eeg_raw(data: np.ndarray, sfreq: float, ch_names, t_start: float, t_end: float, channels: list):
    s0 = int(max(0, t_start) * sfreq); s1 = int(min(data.shape[1], t_end * sfreq))
    if s1 <= s0: s0, s1 = 0, min(data.shape[1], s0 + int(5*sfreq))
    fig, ax = plt.subplots()
    offset = 0.0; step = np.nanstd(data[:, s0:s1]) * 3.0 + 1e-6
    t = np.arange(s0, s1) / sfreq
    for ch in channels:
        idx = ch_names.index(ch)
        ax.plot(t, data[idx, s0:s1] + offset, linewidth=0.8)
        ax.text(t[0], offset, ch, va="bottom", fontsize=8)
        offset += step
    ax.set_title("EEG Raw (stacked)"); ax.set_xlabel("Time (s)"); ax.set_yticks([])
    st.pyplot(fig, use_container_width=True)

def plot_psd(freqs: np.ndarray, psd_avg: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(freqs, 10*np.log10(psd_avg + 1e-12), linewidth=1.0)
    ax.set_xlim(0, min(60, freqs.max() if freqs is not None else 60))
    ax.set_title("Power Spectral Density (avg of channels)")
    ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    st.pyplot(fig, use_container_width=True)

def plot_band_bars(bp: Dict):
    names = list(bp.keys()); rel = [bp[k]["rel"] for k in names]
    fig, ax = plt.subplots()
    ax.bar(names, rel)
    ax.set_title("Relative Band Powers"); ax.set_ylabel("Fraction of total")
    st.pyplot(fig, use_container_width=True)

def plot_spectrogram(one_channel: np.ndarray, sfreq: float):
    fig, ax = plt.subplots()
    ax.specgram(one_channel, NFFT=256, Fs=sfreq, noverlap=128)
    ax.set_ylim(0, min(60, sfreq/2)); ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
    st.pyplot(fig, use_container_width=True)

def plot_topomap_if_possible(raw_like, freqs: np.ndarray, psd_avg: np.ndarray):
    if not _has_mne or not hasattr(raw_like, "copy"):
        st.info("Topomap requires MNE Raw with montage."); return
    raw = raw_like
    try:
        if raw.get_montage() is None:
            st.info("No montage set; cannot compute topomap.")
            return
    except Exception:
        st.info("No montage set; cannot compute topomap.")
        return
    with st.spinner("Computing alpha topomap (8–13 Hz)…"):
        f = np.logical_and(freqs >= 8, freqs <= 13)
        data = raw.get_data()[:, :int(5*raw.info['sfreq'])]
        psds, ff = mne.time_frequency.psd_array_welch(data, sfreq=raw.info['sfreq'], n_fft=512, average='mean')
        alpha = psds[:, np.logical_and(ff >= 8, ff <= 13)].mean(axis=1)
        ret = mne.viz.plot_topomap(alpha, raw.info, outlines="head", contours=0, show=False)
        fig = ret[0] if isinstance(ret, (tuple, list)) else ret
        st.pyplot(fig, use_container_width=True)


# ---- JSON sanitizers ----
import json
import numpy as np

def _to_jsonable(obj, drop_cache: bool = False):
    """
    Recursively convert NumPy types to plain Python.
    Optionally drop any '_cache' keys with large arrays.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if drop_cache and k == "_cache":
                continue  # remove heavy arrays
            out[k] = _to_jsonable(v, drop_cache=drop_cache)
        return out
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(x, drop_cache=drop_cache) for x in obj]
    elif isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, (np.ndarray, )):
        # If you ever keep arrays, convert to list (but we prefer dropping them for LLM)
        return obj.tolist()
    else:
        return obj

def dump_json(obj, drop_cache: bool = False, **kwargs) -> str:
    return json.dumps(_to_jsonable(obj, drop_cache=drop_cache), **kwargs)




# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_messages(mri_stats: Optional[dict], eeg_stats: Optional[dict]) -> tuple[str, str]:
    audience_str = "Layperson" if "Plain" in audience else "Clinician"
    system = (
        f"You are a careful neuroimaging/EEG explainer. Speak in {lang}. "
        f"Audience: {audience_str}. Provide concise conclusions without revealing internal reasoning steps. "
        "Avoid diagnoses; highlight data quality caveats and next-step suggestions to discuss with clinicians."
    )
    # Build a lean payload: drop '_cache' (freqs/psd arrays), and sanitize NumPy types
    payload = {"MRI": mri_stats, "EEG": eeg_stats}
    payload = _to_jsonable(payload, drop_cache=True)

    user = (
        "Summarize these derived features from MRI/EEG:\n"
        "1) Short overall summary.\n"
        "2) Bullet insights: typical/atypical patterns, quality flags.\n"
        "3) Gentle next-step suggestions (non-diagnostic).\n"
        "Be clear and human-friendly.\n\n"
        f"DATA (JSON):\n```json\n{dump_json(payload, drop_cache=True, indent=2)}\n```"
    )
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generators
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("No files? Generate small demo data"):
    make_demo = st.checkbox("Create synthetic demo MRI & EEG")
    demo_size = st.slider("Demo MRI size", 32, 96, 64, 8)

def demo_mri(d: int = 64):
    x, y, z = np.meshgrid(np.linspace(-1,1,d), np.linspace(-1,1,d), np.linspace(-1,1,d), indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    vol = np.exp(-(r**2)*6) + 0.05*np.random.randn(d, d, d).astype(np.float32)
    return vol.astype(np.float32), (1.0, 1.0, 1.5)

def demo_eeg(n_ch=8, sfreq=200.0, dur=20.0):
    t = np.arange(0, dur, 1/sfreq)
    data = []
    for _ in range(n_ch):
        sig = 10*np.sin(2*np.pi*10*t) + 2*np.sin(2*np.pi*3*t) + np.random.randn(t.size)*0.5
        data.append(sig.astype(np.float64))
    data = np.stack(data, axis=0)
    if _has_mne:
        info = mne.create_info([f"Ch{i+1}" for i in range(n_ch)], sfreq, "eeg")
        return mne.io.RawArray(data, info, verbose=False)
    return {"data": data, "sfreq": sfreq, "ch_names": [f"Ch{i+1}" for i in range(n_ch)], "_kind": "numpy"}

# ─────────────────────────────────────────────────────────────────────────────
# Uploads
# ─────────────────────────────────────────────────────────────────────────────
colL, colR = st.columns(2)

with colL:
    st.subheader("📥 Upload Brain MRI")
    mri_file = st.file_uploader("NIfTI (.nii/.nii.gz) or DICOM series as .zip", type=["nii", "nii.gz", "zip"])
    mri_data = mri_zooms = None
    mri_stats = None
    mri_imgs = None
    if mri_file is not None:
        try:
            if mri_file.name.lower().endswith((".nii", ".nii.gz")):
                mri_data, mri_zooms, _ = load_nifti(mri_file.read())
            else:
                mri_data, mri_zooms, _ = load_dicom_zip(mri_file.read())
            mri_stats = mri_features(mri_data, mri_zooms)
            a, c, s = central_slices(mri_data); mri_imgs = {"Axial": a, "Coronal": c, "Sagittal": s}
            st.success(f"MRI loaded. Shape: {tuple(mri_data.shape)}, voxel size (mm): {mri_zooms}")
            cols = st.columns(3)
            for (k, img), cc in zip(mri_imgs.items(), cols):
                cc.image(img, caption=k, use_container_width=True)
            with st.expander("MRI numeric summary"):
                st.json(mri_stats)
        except Exception as e:
            st.error(f"Error loading MRI: {e}")

with colR:
    st.subheader("📥 Upload EEG")
    eeg_file = st.file_uploader("EDF (.edf) or CSV (.csv)", type=["edf", "csv"])
    eeg_stats = None; raw_for_topomap = None
    if eeg_file is not None:
        try:
            raw = load_eeg_edf(eeg_file.read()) if eeg_file.name.lower().endswith(".edf") else load_eeg_csv(eeg_file.read())
            raw_for_topomap = raw if (_has_mne and hasattr(raw, "copy")) else None
            eeg_stats = eeg_features(raw)
            st.success(f"EEG loaded. Channels: {eeg_stats['n_channels']}, duration: {eeg_stats['duration_sec']:.1f}s, fs: {eeg_stats['sfreq_hz']:.1f}Hz")
            with st.expander("EEG numeric summary"):
                st.json(eeg_stats)
        except Exception as e:
            st.error(f"Error loading EEG: {e}")

# Optional demo
if make_demo and (mri_stats is None and eeg_stats is None):
    st.info("Generating synthetic demo MRI & EEG…")
    mri_data, mri_zooms = demo_mri(demo_size)
    mri_stats = mri_features(mri_data, mri_zooms)
    a, c, s = central_slices(mri_data)
    st.image([a, c, s], caption=["Axial", "Coronal", "Sagittal"], use_container_width=True)

    eeg_demo = demo_eeg()
    eeg_stats = eeg_features(eeg_demo)
    raw_for_topomap = eeg_demo if (_has_mne and hasattr(eeg_demo, "copy")) else None
    with st.expander("Demo features"):
        st.json({"MRI": mri_stats, "EEG": eeg_stats})

# ─────────────────────────────────────────────────────────────────────────────
# Visualization panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Visualization — pick what you want to see")

vcol1, vcol2 = st.columns(2)

with vcol1:
    st.subheader("MRI Visuals")
    if mri_stats is None:
        st.info("Upload an MRI (or enable demo) to view MRI visuals.")
    else:
        mri_visuals = st.multiselect(
            "Select MRI visualizations",
            ["Slice scroller", "Intensity histogram", "MIP projections", "Central orthogonal slices"],
            default=["Central orthogonal slices"]
        )
        if "Central orthogonal slices" in mri_visuals:
            a, c, s = central_slices(mri_data)
            c1, c2, c3 = st.columns(3)
            c1.image(a, caption="Axial", use_container_width=True)
            c2.image(c, caption="Coronal", use_container_width=True)
            c3.image(s, caption="Sagittal", use_container_width=True)
        if "Slice scroller" in mri_visuals:
            slice_scroller(mri_data)
        if "Intensity histogram" in mri_visuals:
            plot_mri_hist(mri_data)
        if "MIP projections" in mri_visuals:
            plot_mip(mri_data)

with vcol2:
    st.subheader("EEG Visuals")
    if eeg_stats is None:
        st.info("Upload an EEG (or enable demo) to view EEG visuals.")
    else:
        eeg_visuals = st.multiselect(
            "Select EEG visualizations",
            ["Raw traces", "PSD", "Spectrogram", "Band-power bars", "Alpha topomap (if montage)"],
            default=["Band-power bars", "PSD"]
        )
        cache = eeg_stats["_cache"]; freqs = cache["freqs"]; psd_avg = cache["psd_avg"]
        data = cache["data"]; sfreq = cache["sfreq"]; ch_names = cache["ch_names"]

        if "Raw traces" in eeg_visuals:
            pick = st.multiselect("Channels", ch_names, default=ch_names[:min(5, len(ch_names))])
            t0, t1 = st.slider("Time window (s)", 0.0, float(eeg_stats["duration_sec"]), (0.0, min(10.0, float(eeg_stats["duration_sec"]))), 0.5)
            if pick:
                plot_eeg_raw(data, sfreq, ch_names, t0, t1, pick)
            else:
                st.info("Pick at least one channel.")
        if "PSD" in eeg_visuals:
            plot_psd(freqs, psd_avg)
        if "Spectrogram" in eeg_visuals:
            ch0 = st.selectbox("Channel for spectrogram", ch_names, index=0)
            plot_spectrogram(data[ch_names.index(ch0)], sfreq)
        if "Band-power bars" in eeg_visuals:
            plot_band_bars(eeg_stats["band_powers"])
        if "Alpha topomap (if montage)" in eeg_visuals:
            plot_topomap_if_possible(raw_for_topomap, freqs, psd_avg)

# ─────────────────────────────────────────────────────────────────────────────
# LLM Insights
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📝 LLM Insights")

if (mri_stats is None) and (eeg_stats is None):
    st.warning("Upload MRI and/or EEG (or enable demo data) to generate insights.")
else:
    system, user = build_messages(mri_stats, eeg_stats)
    go = st.button("Generate Insights", type="primary", use_container_width=True)
    if go:
        try:
            if backend.startswith("Groq"):
                with st.spinner("Calling Groq model…"):
                    live = st.empty()
                    text = call_groq_stream(system, user,
                                            model=groq_model,
                                            temperature=temperature,
                                            max_tokens=max_new_tokens,
                                            placeholder=live)
                st.success("Insights ready.")
                st.markdown(text)
            else:
                with st.spinner("Loading local model & generating…"):
                    pipe, tok = get_local_pipe(local_model, use_4bit)
                    text = chat_generate(pipe, tok, system, user,
                                         max_new_tokens=max_new_tokens,
                                         temperature=temperature, top_p=top_p)
                st.success("Insights ready (local).")
                st.markdown(text)

            import datetime
            safe_mri = _to_jsonable(mri_stats, drop_cache=True) if mri_stats else None
            safe_eeg = _to_jsonable(eeg_stats, drop_cache=True) if eeg_stats else None
            report = (
                "# Neuro Insight Report\n\n"
                f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"- Backend: {backend}\n"
                f"- Model: {groq_model if backend.startswith('Groq') else local_model}\n"
                f"- Audience: {audience}\n"
                f"- Language: {lang}\n\n"
                "## MRI Features\n"
                f"```json\n{dump_json(safe_mri, drop_cache=True, indent=2) if safe_mri is not None else 'null'}\n```\n"
                "## EEG Features\n"
                f"```json\n{dump_json(safe_eeg, drop_cache=True, indent=2) if safe_eeg is not None else 'null'}\n```\n"
                "## LLM Summary\n\n"
                f"{text}\n\n"
                "> *Educational use only; not medical advice.*\n"
            )
            st.download_button("Download report (.md)", report, "neuro_insight_report.md")
        except Exception as e:
            st.error(f"Insight generation failed: {e}")

st.markdown("---")
st.caption("© 2025 • Groq-powered demo (with optional offline fallback). No diagnoses provided.")
