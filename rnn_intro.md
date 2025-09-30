# 📘 Recurrent Neural Networks (RNN)

## 1. Introduction
A **Recurrent Neural Network (RNN)** is a specialized type of Artificial Neural Network (ANN) designed to process **sequential data**. Unlike feedforward networks, RNNs have **loops** that allow information to persist, making them suitable for tasks where the **order of inputs matters**.  

---

## 2. Types of ANN
Artificial Neural Networks come in different architectures, each suitable for different data types:  

- **Feedforward Neural Network (FNN)**  
  - Data flows in one direction, no memory.  
  - Suitable for tabular/structured data.  

- **Convolutional Neural Network (CNN)**  
  - Designed for spatial data (images, video).  
  - Captures local patterns using filters.  

- **Recurrent Neural Network (RNN)**  
  - Designed for sequential/time-dependent data.  
  - Maintains a hidden state (memory).  

---

## 3. Sequential vs. Non-Sequential Data

| **Data Type**        | **Examples**                     | **Property**                                |
|-----------------------|-----------------------------------|---------------------------------------------|
| **Sequential Data**  | Text, speech, stock prices, DNA  | Order matters; elements depend on each other |
| **Non-Sequential Data** | Images, tabular data, independent features | Order does not matter                        |

👉 RNNs excel in sequential data because they **remember context across steps**.  

---

## 4. Why RNN is Different from Other NNs
- **Memory**: Retains previous information via hidden state.  
- **Weight Sharing**: Same weights across all time steps → efficient training.  
- **Context Awareness**: Learns dependencies like grammar, tone, or trends.  

Unlike traditional NNs that process inputs independently, RNNs **connect past to present**.  

---

## 5. RNN Architecture

At each time step *t*:

- Input: \( x_t \)  
- Previous hidden state: \( h_{t-1} \)  
- Current hidden state:  

\[
h_t = f(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)
\]

- Output:  

\[
y_t = g(W_{hy} \cdot h_t + b_y)
\]

Where:  
- \( W_{xh}, W_{hh}, W_{hy} \) are weight matrices  
- \( f \) = activation (tanh/ReLU)  
- \( g \) = softmax/sigmoid  

**Diagram (conceptual):**  

```

x1 → [RNN Cell] → h1 → y1
x2 → [RNN Cell] → h2 → y2
x3 → [RNN Cell] → h3 → y3

```

---

## 6. Working of RNN
1. Input sequence is processed step by step.  
2. Each RNN cell takes **current input (x_t)** + **previous hidden state (h_{t-1})**.  
3. Produces a **new hidden state (h_t)** and an **output (y_t)**.  
4. Hidden state passes forward in the sequence.  

This recurrence allows RNNs to model **temporal dependencies**.  

---

## 7. Applications of RNN
### ✅ Natural Language Processing (NLP)
- Sentiment Analysis  
- Machine Translation  
- Text Generation  

### ✅ Speech & Audio
- Speech Recognition  
- Voice Synthesis  
- Music Generation  

### ✅ Time Series
- Stock Price Prediction  
- Weather Forecasting  
- Sensor Data Monitoring  

### ✅ Control & Decision Making
- Robotics  
- Game AI  

---

## 8. Limitations of Vanilla RNN
- **Vanishing Gradient Problem**: Hard to learn long-term dependencies.  
- **Exploding Gradient Problem**: Training instability.  
- **Slow Training**: Sequential nature makes it less parallelizable.  

👉 These issues led to advanced RNN variants:  
- **LSTM (Long Short-Term Memory)**  
- **GRU (Gated Recurrent Unit)**  

---

## 9. Summary
- RNNs are **designed for sequential data**.  
- They differ from traditional NNs because they have **memory (hidden state)**.  
- Widely used in **NLP, speech, time series, and decision-making systems**.  
- Advanced versions like **LSTM/GRU** solve the limitations of vanilla RNNs.  
```

---
