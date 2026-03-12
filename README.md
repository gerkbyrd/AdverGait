## The Adversarial Gait: Detecting Adversarial Attacks against Vision-Language Models via Self-Targeted Gradient Characterization
[![PDF Preview](diagram/preview.png)](diagram/document.pdf)

This repository contains the implementation for the core components of the paper. The code is organized as follows:

directories:
- archive/, data/, existing_attacks/, utils/: helper files required to run our implementation

main files:
- utils/transfer_attack.py: algorithm used for both our adaptive attacks and AdverGait
- pip.py, nearside.py, mirrorcheck.py: implementation of our baseline detectors
- Demo.ipynb: short demo that runs AdverGait and the baselines on a small dataset
- pipsvm_3.joblib, nearside_adv_dir_vlm_3.joblib: pre-trained components to perform detection with PIP and Nearside



