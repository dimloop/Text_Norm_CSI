# Cover Song Similarity (Da-TACOS) 

Experiments on **cover song similarity** using the **Da-TACOS** dataset.  
We compare two Siamese models:

- **CNN (1D, residual)** – learns L2-normalized embeddings from chroma-like features and scores pairs with a learned squared-difference head.
- **CNN + Quantum layer (hybrid)** – same backbone with a small Pennylane variational circuit inserted on top of the embedding.

Both models are trained with **binary cross-entropy** and evaluated with accuracy and ROC–AUC. 



---

## Repository structure
```
.
├── figures/                     # saved plots (loss/acc/ROC )
├── models/                      # saved checkpoints (e.g., best_model.pth)
├── src/
│   ├── dataset.py               # Da-TACOS feature loading & preprocessing
│   ├── globals.py               # global constants (path data)
│   ├── losses.py                # Similarity loss (BCE)
│   ├── models.py                # Siamese backbones (CNN, CNN+Quantum)
│   ├── train.py                 # training / validation / test loops 
│   └── utils.py                 # plotting etc
├── CSI_notebook.ipynb           # quick experiments & visualizations
├── data_notebook.ipynb          # data inspection (distributions, lengths)
├── Cover_song_similarity.pdf    # report
└── README.md                    # this file
```


---

## Data: Da-TACOS
We use the **Da-TACOS** dataset: <https://github.com/MTG/da-tacos> – features + metadata only (no audio).  
Key points we use in code:
- **Features (HDF5):** `hpcp`, `crema` (CREMA-PCP), etc. Loaded per performance ID (PID).
- **Metadata (JSON):** mappings from work IDs (WID) to PIDs, titles, artists, year, etc.

### Preprocessing 
- L2-normalize each time frame (row) to **unit norm**.
- Downsample / crop each sequence to **2000 frames** for consistency.
- If shorter, **right-pad with zeros** to reach 2000 frames.

Update paths in `src/dataset.py` to point to your feature folders/H5 files.

---

##  Models
- **Siamese 1D CNN** with residual blocks → global pooling → MLP → **128-dim** embedding.
- Comparator: apply a linear head on the **squared difference** `(v1 - v2)^2` and **sigmoid** → pair probability.
- **Hybrid variant:** insert a PennyLane **quantum block** after the embedding. Uses `AmplitudeEmbedding` on `q` qubits + entanglers, then map back to 128-dim.

Loss: **BCE** on pair labels (`0 = similar`, `1 = dissimilar` unless configured otherwise).

---

## Training

You can run and inspect the full training pipeline in the **CSI** notebook, which imports the training loop from `train.py`.

- **Notebook:** `notebooks/CSI.ipynb`  
  Walks through data loading, model setup, and calls into `train_siamese_network` from `train.py`, so you can see logs, plots (loss/accuracy/ROC), and tweak hyperparameters.

- **Script:** `train.py`  
  Contains the reusable training/evaluation functions (`train_siamese_network`, `evaluate_siamese_network`). 


**Checkpoints**: best validation is saved into `models/best_model.pth` by default.

---

## Evaluation
- During training we log **loss** and **accuracy** for train/val.
- On test, we report **accuracy** and plot **ROC**; ROC is computed using the **best-validation** checkpoint.

Example snippet to load a trained model:
```python
import torch
from src.models import SiameseSimple  # or your chosen class

model = SiameseQuantum(in_channels=12, emb_dim=128)
state = torch.load("models/best_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

---

## Results
- **CNN:** Test **accuracy ≈ 0.62**, **ROC–AUC ≈ 0.70**
- **CNN + Quantum:** Test **accuracy ≈ 0.65**, **ROC–AUC ≈ 0.71**

The hybrid shows a small but consistent improvement in ranking (AUC) and a modest accuracy bump. Use **early stopping** and **threshold calibration** for best accuracy.

---

## References
- Da-TACOS dataset: <https://github.com/MTG/da-tacos>
- Jullien et al., *Cover Song Identification with Siamese Networks*, 2020.  
- Serrano & Bellogín, *Siamese neural networks in recommendation*, NCA 2023. <https://doi.org/10.1007/s00521-023-08610-0>





