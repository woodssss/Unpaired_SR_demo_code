# Unpaired_SR_Fluid_Dynamics
This project provides codes for the paper [DIFFUSION-BASED MODELS FOR UNPAIRED SUPER-RESOLTUON IN FLUID DYNAMICS]. Currently, we are publishing sample code for the Navier–Stokes equations. The full code will be released once the paper is officially published.

# Usage of code
This section outlines the standard procedure for utilizing the code, with a focus on the implementation of the 2D Navier-stokes snapshot data. The Jupyter notebook step by step demonstration is in the 'Demo.ipynb'.  

# **Step 1.** Data preparation

  - ## Option A (recommended): you can download the prepared datasets from Google Drive by
    ` python download_data.py  `
    - ## The datasets are stored in the 'data' folder

  - ## Option B: prepare your own datasets.
    - ## First, generate high-fidelity high-resolution (HFHR) datasets (256x256) and low-fidelity low-resolution (LFLR) datasets (32x32) (here we use JAX-CFD)
    - ## This resulting two unpaired HFHR dataset and LFLR dataset for training, and one paired LFLR and HFHR dataset for testing. These datasets should store in the raw_data folder.
    - ## Then prepare the intermidiate datasets HFLR datasets at resolution 128x128, 64x64, 32x32 by downsampling the HFHR dataset using cubic interpolation: 
    - ` python prepare_data.py `
    - ## The resulting datasets are stored in the data folder.
   
# **Step 2.** Training for debiasing at LR level. 

- ## Option A (recommended): you can download the pretrained models from Google Drive by
    ` python download_chckpt.py  `
    - ## The models are stored in the 'mdls' folder
## Option B: For completeness, the following commands outline how to train each model.


 # 2.1. EDDIB(ours):
   - ## Train two unconditional diffuion models for the HFLR and LFLR datasets, respectively.
   - ## ` python train_gen.py --flag 0` for LFLR dataset
   - ## ` python train_gen.py --flag 1` for HFLR dataset.
 # 2.2. SDEdit:
   - ## Train one unconditional diffuion model for the HFLR dataset. (Note: Retraining is not needed since it was already performed in the previous step; this is provided for completeness.)
   - ## ` python train_gen.py --flag 1` for HFLR dataset
 # 2.3. OT (OTT-JAX)
   - ## ` python train_ot.py `
 # 2.4. NOT (Neural OT)
   - ## ` python train_not.py `

---

# **Step 3.** Training for cascaded SR3.
## Train three conditional diffusion models for the HFLR datasets, respectively.
   -  ` python train_sup.py --flag 0` for SR from 32x32 to 64x64
   -  ` python train_sup.py --flag 1` for SR from 64x64 to 128x128
   -  ` python train_sup.py --flag 2` for SR from 128x128 to 256x256

# **Step 4.** Inference.
## Unpaired SR for a testing LFLR dataset.
  - ` python Bridge.py`
