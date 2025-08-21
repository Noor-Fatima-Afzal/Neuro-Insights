# Neuro Insight Report

- Generated: 2025-08-21T18:10:06
- Backend: Groq (fast, recommended)
- Model: openai/gpt-oss-20b
- Audience: Plain language (layperson)
- Language: English

## MRI Features
```json
{
  "shape_xyz": [
    64,
    64,
    64
  ],
  "voxel_size_mm": [
    1.0,
    1.0,
    1.5
  ],
  "voxel_volume_mm3": 1.5,
  "intensity_mean": 0.045007459819316864,
  "intensity_std": 0.128225177526474,
  "intensity_p01_p99": [
    -0.10900459438562393,
    0.6470541954040527
  ],
  "high_intensity_outlier_frac": 0.027088165283203125,
  "low_intensity_outlier_frac": 0.0
}
```
## EEG Features
```json
{
  "n_channels": 8,
  "sfreq_hz": 200.0,
  "duration_sec": 20.0,
  "artifact_fraction": 0.0,
  "band_powers": {
    "delta": {
      "abs": 60251.50946657463,
      "rel": 0.0385113856274076
    },
    "theta": {
      "abs": 298.45192538587844,
      "rel": 0.00019076363881230545
    },
    "alpha": {
      "abs": 1497482.115940034,
      "rel": 0.9571562894885853
    },
    "beta": {
      "abs": 1285.1410741924858,
      "rel": 0.0008214327563246255
    },
    "gamma": {
      "abs": 1100.5797603852413,
      "rel": 0.0007034653893514398
    }
  },
  "dominant_band": "alpha"
}
```
## LLM Summary

**1. Short overall summary**  
The MRI shows a normal‑looking brain volume with typical gray‑white contrast and no obvious outliers. The EEG, recorded with 8 channels at 200 Hz for 20 seconds, is clean (no artifacts) and dominated by alpha activity, which is common in relaxed wakefulness.

---

**2. Bullet insights**

* **MRI**
  * **Shape & size** – 64 × 64 × 64 voxels, 1 mm × 1 mm × 1.5 mm voxels → good spatial resolution.
  * **Intensity statistics** – mean 0.045, standard deviation 0.128; intensity range from –0.109 to 0.647. These values fall within expected ranges for a healthy brain scan.
  * **Outliers** – 2.7 % of voxels are unusually bright; none are unusually dark. This small fraction is typical and unlikely to indicate pathology.
  * **Quality flag** – No major artifacts or signal loss detected.

* **EEG**
  * **Sampling & duration** – 200 Hz, 20 s; adequate for basic spectral analysis.
  * **Artifact fraction** – 0 % → clean recording.
  * **Band powers**  
    * Alpha: 95.7 % of total power – normal for a relaxed, eyes‑closed state.  
    * Delta, theta, beta, gamma: each < 1 % – within expected limits for a short, artifact‑free recording.
  * **Dominant band** – Alpha, which is typical for a relaxed adult.
  * **Quality flag** – No channel drop‑outs or excessive noise.

---

**3. Gentle next‑step suggestions (non‑diagnostic)**  

1. **Discuss the findings with your clinician** – share that the MRI and EEG appear normal and that the alpha dominance is expected in a relaxed state.  
2. **Consider a longer EEG session** if you want to explore sleep or task‑related patterns; a 20‑second snapshot is limited.  
3. **If you have specific concerns** (e.g., headaches, seizures), ask whether additional imaging sequences or a full‑night EEG might be helpful.  
4. **Keep a symptom diary** – correlating daily symptoms with these objective measures can provide useful context for future visits.  

These steps can help you and your healthcare team interpret the data in the context of your overall health.

> *Educational use only; not medical advice.*
