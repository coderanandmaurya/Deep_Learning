# 🧠 Day 2: Introduction to Deep Learning & Representation Learning

Welcome to Day 2 of the AI/ML Bootcamp!

Today, we explore how Deep Learning has become the dominant method in AI by learning powerful **representations** from data, powered by large datasets, better hardware, and open-source libraries. We'll end with a historical look at the **Perceptron**, the ancestor of neural networks.

---

## 📘 Contents

1. [What is Deep Learning?](#1-what-is-deep-learning)
2. [Representation Learning](#2-representation-learning)
3. [Deep Learning vs Classical ML](#3-deep-learning-vs-classical-ml)
4. [Why Deep Learning is Popular](#4-why-deep-learning-is-popular)
5. [Frameworks: TensorFlow vs PyTorch](#5-frameworks-tensorflow-vs-pytorch)
6. [Transfer Learning](#6-transfer-learning)
7. [Popular Architectures & Use Cases](#7-popular-architectures--use-cases)
8. [GANs (Generative Adversarial Networks)](#8-gans-generative-adversarial-networks)
9. [Historical Roots: The Perceptron](#9-historical-roots-the-perceptron)
10. [Practice & Assignments](#10-practice--assignments)

---

## 1. What is Deep Learning?

Deep Learning is a subset of machine learning where models learn directly from **raw data** using multiple layers of processing called **neural networks**.

- Handles unstructured data: images, text, audio.
- Learns **hierarchical** features automatically.
- Scales well with large datasets and compute.

---

## 2. Representation Learning

Instead of hand-crafted features, deep learning learns representations such as:

- **Low-level:** edges, textures
- **Mid-level:** shapes, patterns
- **High-level:** objects, semantics

### Example:
In image classification:
- Classical ML: manual features like HOG or SIFT.
- Deep Learning: CNNs learn filters automatically from data.

---

## 3. Deep Learning vs Classical ML

| Feature               | Classical ML         | Deep Learning             |
|-----------------------|----------------------|----------------------------|
| Features              | Hand-engineered      | Automatically learned      |
| Input type            | Tabular, structured  | Images, audio, text        |
| Model type            | Shallow (e.g. SVM)   | Deep (CNN, RNN, Transformer) |
| Data requirement      | Small to medium      | Large datasets             |
| Training time         | Short                | Long, but reusable         |

---

## 4. Why Deep Learning is Popular

### ✅ Big Data Availability
- ImageNet, Common Crawl, YouTube-8M, etc.

### ✅ Hardware Advancements
| Platform | Hardware           |
|----------|--------------------|
| PC       | GPU (NVIDIA CUDA)  |
| Mobile   | NNAPI, Core ML, TFLite |
| Edge     | Jetson Nano, Edge TPU |
| Cloud    | TPUs (Google), ASICs |

### ✅ Libraries
- TensorFlow (Google): production, mobile
- PyTorch (Meta): research, flexible

---

## 5. Frameworks: TensorFlow vs PyTorch

| Feature             | TensorFlow        | PyTorch            |
|---------------------|-------------------|--------------------|
| Graph Type          | Static (TF 1.x)   | Dynamic            |
| Mobile Support      | TFLite            | TorchScript        |
| Community           | Mature, industry  | Research-oriented  |
| Best Use            | Deployment        | Prototyping, research |

---

## 6. Transfer Learning

Use a **pre-trained model** on a new task to:
- Reduce training time
- Require less data
- Improve accuracy

### Examples:
- Use **ResNet50** for disease classification
- Use **BERT** for text classification
- Fine-tune **U-Net** on medical segmentation

---

## 7. Popular Architectures & Use Cases

| Model     | Task                      | Use Case Example                         |
|-----------|---------------------------|------------------------------------------|
| ResNet    | Image Classification       | Classify animals, medical images         |
| U-Net     | Image Segmentation         | Tumor detection in MRIs                  |
| YOLO      | Object Detection           | Real-time vehicle/pedestrian detection   |
| Pix2Pix   | Image-to-Image Translation | Sketch to realistic image                |
| BERT      | NLP Understanding          | Chatbots, Q&A, sentiment classification  |
| WaveNet   | Audio Generation           | Natural-sounding speech (TTS)            |
| GAN       | Image/Audio Generation     | Deepfakes, art, upscaling, super-resolution |

---

## 8. GANs (Generative Adversarial Networks)

Invented by Ian Goodfellow (2014), a **GAN** consists of:
- **Generator**: Creates fake data
- **Discriminator**: Judges real vs fake

### GAN Use Cases:
- Deepfake video generation
- Image enhancement (Super-resolution)
- Style transfer (turn photo into art)
- Synthetic data creation

---

## 9. Historical Roots: The Perceptron

### 🏛️ What is a Perceptron?

The **Perceptron** is the first artificial neuron model, introduced by **Frank Rosenblatt** (1958). It laid the foundation for neural networks.

### Equation:
\[
y = \begin{cases}
1 & \text{if } \sum w_i x_i + b > 0 \\
0 & \text{otherwise}
\end{cases}
\]

### ✅ Solvable Logic Gates:
- AND
- OR
- NAND

### ❌ Limitations:
- Cannot solve XOR (not linearly separable)
- Led to the **AI Winter** after criticism by Minsky and Papert (1969)

### 🔁 Solution:
Multi-layer perceptrons (MLPs) introduced **backpropagation** to solve XOR and nonlinear problems.

---

## 10. Practice & Assignments

### 🧪 Code Practice
1. Implement AND, NAND, XOR using perceptron logic.
2. Fine-tune **ResNet** on your own small image dataset.
3. Use **BERT** from HuggingFace to classify text.
4. Run a **YOLOv5** notebook for real-time detection.
5. Try **StyleGAN2** for face generation.

### 🔧 Tools Setup
- Install PyTorch or TensorFlow
- Try Google Colab or Kaggle for free GPUs
- Explore ONNX models for cross-framework deployment

---

## 🧠 Knowledge Checkpoints

✅ What is representation learning?  
✅ How does deep learning differ from classical ML?  
✅ What are GANs, and how do they work?  
✅ When and why did the Perceptron fail?  
✅ Which model would you use for segmentation/classification/text/audio?

---

## 📚 Resources

- [📘 DeepLearning.ai](https://www.deeplearning.ai/)
- [📘 PyTorch Tutorials](https://pytorch.org/tutorials/)
- [📘 TensorFlow Guide](https://www.tensorflow.org/learn)
- [📘 HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [🎮 Run DL models online](https://colab.research.google.com)

---

🔜 **Coming Next: Day 3 - Neural Network Training, Backpropagation, and Optimizers**

