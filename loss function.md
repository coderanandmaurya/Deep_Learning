# Loss Function vs Cost Function in Deep Learning

## 1. What is a Loss Function?
A **loss function** measures how wrong your model’s predictions are compared to the actual target values.

- **Input:** model predictions (y_hat) and actual labels (y)
- **Output:** a single number representing the error for **one sample**
- **Goal:** Minimize this value during training.

**Example (Regression – MSE for one sample):**
![MSE Example](https://latex.codecogs.com/png.image?L%20=%20(y%20-%20\hat{y})^2)

---

## 2. Loss Function vs Cost Function

| Aspect         | Loss Function (Per Sample) | Cost Function (Dataset-wide) |
|----------------|----------------------------|-------------------------------|
| Definition     | Error for **one training example** | Average (or sum) of loss over the **entire dataset** |
| Scope          | Per-sample measure         | Dataset-wide measure |
| Example        | Squared error for one prediction | Mean Squared Error (MSE) over all samples |
| Usage          | Calculated before averaging | Used for optimization in gradient descent |

## 3. Types of Loss Functions in Deep Learning

### A. Regression Loss Functions
1. **Mean Squared Error (MSE)**  
![MSE](https://latex.codecogs.com/png.image?MSE%20=%20rac{1}{N}%20\sum_{i=1}^N%20(y_i%20-%20\hat{y}_i)^2)

2. **Mean Absolute Error (MAE)**  
![MAE](https://latex.codecogs.com/png.image?MAE%20=%20rac{1}{N}%20\sum_{i=1}^N%20|y_i%20-%20\hat{y}_i|)

3. **Huber Loss**  
Hybrid between MSE and MAE.

---

### B. Classification Loss Functions
1. **Binary Cross-Entropy**  
![BCE](https://latex.codecogs.com/png.image?BCE%20=%20-rac{1}{N}%20\sum_{i=1}^N%20[y_i%20\log(\hat{y}_i)%20+%20(1-y_i)%20\log(1-\hat{y}_i)])

2. **Categorical Cross-Entropy**  
![CCE](https://latex.codecogs.com/png.image?CCE%20=%20-\sum_{c=1}^C%20y_{ic}%20\log(\hat{y}_{ic}))

3. **Sparse Categorical Cross-Entropy** — same as CCE but uses integer labels.

4. **Hinge Loss** — margin-based loss.

5. **Focal Loss** — focuses training on hard examples.

---

### C. Encoding / Embedding Losses
1. **Triplet Loss**  
![Triplet](https://latex.codecogs.com/png.image?Triplet%20=%20\max(0,%20d(a,%20p)%20-%20d(a,%20n)%20+%20lpha))

2. **Contrastive Loss** — pairs-based similarity/dissimilarity loss.

3. **Center Loss** — pulls features toward class centers.

---

### D. Autoencoder Losses
1. **Reconstruction Loss**  
MSE or MAE between input and reconstruction.

2. **VAE Loss**  
![VAE](https://latex.codecogs.com/png.image?L%20=%20\mathbb{E}_{q(z|x)}[-\log%20p(x|z)]%20+%20KL(q(z|x)\,\|\,p(z)))

---

### E. GAN Losses
1. **Minimax GAN Loss**
![GAN Minimax](https://latex.codecogs.com/png.image?\max_D%20\mathbb{E}[\log%20D(x)]%20+%20\mathbb{E}[\log(1-D(G(z)))])

2. **Non-saturating Generator Loss**
![GAN NS](https://latex.codecogs.com/png.image?L_G%20=%20-\mathbb{E}[\log%20D(G(z))])

---

### F. Object Detection Losses
1. **Classification Loss** — cross-entropy or focal loss.  
2. **Localization Loss** — Smooth L1 (Huber) or IoU-based loss.  
3. **Objectness Loss** — binary cross-entropy for object presence.  
4. **Mask Loss** — segmentation masks.  

---

## 4. Tree Representation of Loss Functions in Deep Learning
```
Loss Functions in DL
│
├── Object Detection
│   ├── Classification Loss (cross-entropy, focal)
│   ├── Localization Loss (Smooth L1, IoU, GIoU, DIoU, CIoU)
│   ├── Objectness Loss
│   └── Mask Loss
│
├── Regression
│   ├── MSE
│   ├── MAE
│   └── Huber Loss
│
├── Classification
│   ├── Binary Crossentropy
│   ├── Categorical Crossentropy
│   ├── Sparse Categorical Crossentropy
│   ├── Hinge Loss
│   └── Focal Loss
│
├── Encoding / Embedding
│   ├── Triplet Loss
│   ├── Contrastive Loss
│   └── Center Loss
│
├── Autoencoders
│   ├── Reconstruction Loss (MSE / MAE / BCE)
│   └── VAE Loss (Reconstruction + KL Divergence)
│
└── GAN
    ├── Discriminator Loss
    ├── Generator Loss
    └── Variants (LSGAN, WGAN, Hinge-GAN)
```

---

## ✅ Summary
- **Loss function** → error for one sample.  
- **Cost function** → average error over all samples.  
- Choose the loss function based on your **task**.
