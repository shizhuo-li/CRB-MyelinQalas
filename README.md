# 🧠 Omni-QALAS Project

This repository contains the full framework for **Omni-QALAS sequence generation, optimization, and reconstruction**, implemented using the open-source **[Pulseq](https://pulseq.github.io/)** platform.  

With this repo, you can:
- 🎛️ Generate different QALAS-based sequences (QALAS, MWF-QALAS, Omni-QALAS)  
- 🤖 Optimize sequence parameters using deep learning (MLP) and CRB-based objectives  
- 🖼️ Reconstruct quantitative maps (T1, T2, MWF, etc.) from k-space data  

---

## 📂 Repository Structure

### 1. `OmniSeq/`
This folder contains the code for generating the **Omni-QALAS sequence**.  
- All sequences in this study were implemented with **Pulseq**.  
- Run **`OmniSeq.m`** to generate the corresponding `.seq` file.    

---

### 2. `Optimization/`
This folder contains the optimization pipeline for sequence design.  
- Uses an **MLP network** together with a **CRB-based objective function**.  
- Run **`optimization_main.py`** to output optimized sequence parameters:  
  - Flip Angle (**FA**)  
  - Gap (**GAP**)  
  - Echo Time (**TE**)  
- The optimized parameters can then be used with the `OmniSeq` code to generate new sequences.  

---

### 3. `SubspaceRecon/`
This folder contains the reconstruction framework from k-space to quantitative maps.  
- **`subspace_recon.m`** performs reconstruction from k-space to multiple contrasts.  
- Results are stored in **`fit.m`**, which can be further processed to obtain **T1**, **T2**, **MWF**, and other quantitative maps.  

---


For access to the data, please contact me via email: shizhuo0908@zju.edu.cn
