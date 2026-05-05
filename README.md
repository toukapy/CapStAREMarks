# CapStARE-LM: Capsule-based Spatiotemporal Architecture for Calibration-Free Gaze Estimation using Facial Landmarks

CapStARE-LM is a landmark-guided extension of CapStARE that introduces explicit facial geometry into capsule-based spatiotemporal gaze estimation. By grounding capsule formation in facial landmarks, CapStARE-LM improves cross-subject robustness, calibration-free deployment, and real-world generalization while preservig lightweight, real-time performance. Based on the published paper, facial landmarks act as a structural prior rather than auxiliary supervision, defining anatomically meaningful capsule regions for more stable gaze estimation.

# 🔥 Key Features

- **Landmark-guided capsule formation** using 468 MediaPipe facial landmarks grouped into anatomical regions.
- **Calibration-free gaze estimation** with strong cross-dataset ransferability.
- **Region-wise Gaussian heatmaps** for geometry-aware feature aggregation.
- **Dual-path GRU temporal decoders** for modeling fast eye and slow head dynamics separately.
- **Real-time efficiency** ~2.08 ms per 12-fraeme sequence.
- **Improved generalization:** significantly stronger cross-dataset performance on MPIIFaceGaze and RT-GENE.
- **Robust anatomical grounding:** more stable and interpretable capsule activations across users.

# Benchmark Performance

| Method       | ETH-XGaze ↑ | MPIIFaceGaze ↓ | RT-GENE ↓ | Inference ↑ | Calibration |
| ------------ | ----------- | -------------- | --------- | ----------- | ----------- |
| CapStARE     | **3.65°**   | 7.98°          | 7.36°     | **1.126 ms**| Required    |
| CapStARE-LM  | 5.35º       | **4.48º**      | **5.46º** | 2.080 ms    | No          |
