# CFL-Phish

**CFL-Phish** is a privacy-preserving Continual Federated Learning framework for multimodal phishing detection. The framework integrates URL, HTML, and visual webpage evidence with Differential Privacy (DP), Zero-Trust (ZT) inspired trust-aware aggregation, experience replay and Elastic Weight Consolidation (EWC), and selective DistilBERT-based semantic verification.

The framework is designed for decentralized edge environments where raw webpage data remain local to participating clients and only protected model updates are exchanged. Experiments are conducted on the Phish360 dataset under heterogeneous Non-IID client distributions, with scalability evaluated from 3 to 50 clients.

This repository provides the implementation, experimental configurations, dataset split manifests, preprocessing and feature-extraction components, federated training pipeline, privacy accounting, evaluation scripts, and reproducibility resources associated with the CFL-Phish study.

<img width="2040" height="934" alt="main-model (2)" src="https://github.com/user-attachments/assets/a4bf2e19-c2d8-451e-8c6f-03be36c0acdb" />
