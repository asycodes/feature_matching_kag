Feature Matching → Geometric Verification → (Optional) SfM with COLMAP

This repo implements a two-stage image matching pipeline:

1) Global retrieval with DINOv2
2) Local matching with SuperPoint + LightGlue, followed by geometric verification (RANSAC)
3) 3D reconstruction via pycolmap/COLMAP

It’s designed for fast candidate generation (global) and precise correspondence estimation (local), and can export to a COLMAP database for SfM.



![Image Pairing](graphs/pairs_graph.png)
![3D Reconstruction](3d_reconstruction.jpg)
