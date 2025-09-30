---

# **Recurrent Neural Networks (RNN) – Detailed Lecture**

---

## **1. Introduction to RNN**

* RNNs are **neural networks for sequential or temporal data**, where **order matters**.
* Unlike feedforward networks, RNNs have **loops in their architecture**, allowing them to **store past information**.
* **Applications:** language modeling, speech recognition, stock prediction, IoT sensor data.

**Example:**

* Input sequence: “I am going to the …” → RNN predicts the next word based on **context**.

---

## **2. Historical Context**

* **1980s–1990s:** Early RNNs (Elman and Jordan networks).
* **1997:** LSTM introduced to solve **vanishing gradient problem**.
* **2010s:** Widely used in NLP and speech recognition.
* **Today:** Transformers dominate NLP, but RNNs still used for **time-series and real-time applications**.

---

## **3. Comparison with Other Neural Networks**

| NN Type        | Input       | Memory       | Best Use Case             |
| -------------- | ----------- | ------------ | ------------------------- |
| Feedforward NN | Independent | None         | Images, tabular data      |
| CNN            | Spatial     | None         | Images, feature maps      |
| RNN            | Sequential  | Hidden state | Text, speech, time-series |

---

## **4. Sequential vs Non-Sequential Data**

* **Non-Sequential:** Independent, order irrelevant. Example: Image classification.
* **Sequential:** Order matters, previous inputs affect current output. Example: Predicting the next word.

**Example:**

* “I love this movie” → Positive
* “I don’t love this movie” → Negative
* RNN remembers **“don’t”** to predict sentiment correctly.

---

## **5. Why RNN is Different**

* Maintains a **hidden state** to store information from previous inputs.
* Can handle **variable-length input sequences**.
* Learns **temporal dependencies**.

**Example:** Stock price prediction depends on several previous days.

---

## **6. Architecture of RNN**

### **Components**

1. **Input Layer:** Takes sequential data (x_t).
2. **Hidden Layer:** Combines current input (x_t) and previous hidden state (h_{t-1}).
3. **Output Layer:** Produces prediction (y_t).

### **Equations**

* Hidden state:
  [
  h_t = f(W_h h_{t-1} + W_x x_t + b)
  ]
* Output:
  [
  y_t = g(W_y h_t + c)
  ]

Where (f) is usually **tanh/ReLU**, (g) depends on task (**softmax, sigmoid, linear**).

---

## **7. Working of RNN**

1. Input (x_1) → hidden state (h_1) → output (y_1) (if many-to-many).
2. Input (x_2) + (h_1) → hidden (h_2) → output (y_2).
3. Continue for all time steps.

**Types of sequence processing:**

* Many-to-one: Sentiment analysis (output after last input).
* Many-to-many: Machine translation (output at each step).

**Diagram:**

* [RNN cell unrolled across time steps]
* Shows (x_t → h_t → y_t) and loop back to (h_{t+1}).

---

## **8. Activation Functions in RNN**

| Layer  | Activation Function      | Reason                                    |
| ------ | ------------------------ | ----------------------------------------- |
| Hidden | tanh, ReLU               | Control gradient, capture non-linearities |
| Output | Softmax, Sigmoid, Linear | Task-specific (classification/regression) |

---

## **9. Loss Functions**

* **Classification:** Cross-Entropy (binary/multi-class).
* **Sequence prediction:** Token-wise Cross-Entropy.
* **Regression / Forecasting:** MSE, MAE.

**Example:**

* Sentiment analysis → Binary cross-entropy
* Stock price → MSE

---

## **10. RNN Variants**

### **10.1 Vanilla RNN**

* Basic recurrent unit.
* Pros: Simple, easy to implement.
* Cons: Vanishing/exploding gradients.

### **10.2 LSTM (Long Short-Term Memory)**

* Solves **long-term dependency problem**.
* **Gates:** Forget, Input, Output.
* Maintains **cell state** (C_t).
* Pros: Captures long-term dependencies.
* Cons: More parameters → slower training.

**Equations:**
[
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
]
[
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
]
[
\tilde{C}*t = \tanh(W_C[h*{t-1}, x_t] + b_C)
]
[
C_t = f_t * C_{t-1} + i_t * \tilde{C}*t
]
[
o_t = \sigma(W_o[h*{t-1}, x_t] + b_o)
]
[
h_t = o_t * \tanh(C_t)
]

---

### **10.3 GRU (Gated Recurrent Unit)**

* Simplified LSTM → **faster, fewer parameters**.
* Pros: Performs similarly to LSTM.
* Use case: Time-series, NLP.

---

### **10.4 Bidirectional RNN (BRNN)**

* Processes sequence **forward and backward**.
* Captures **past and future context**.
* Use case: NLP, sentiment analysis, speech recognition.

---

### **10.5 Deep / Stacked RNN**

* Multiple RNN layers stacked → learns **higher-level sequence features**.

---

### **10.6 Echo State Network (ESN)**

* Recurrent weights **fixed/random**, only output weights trained.
* Fast training, used in **time-series prediction**.

---

### **10.7 Attention-based RNN**

* Combines **RNN with attention** to focus on **important parts of sequence**.
* Use case: Machine translation, summarization.

---

### **RNN Variants Summary Table**

| Type          | Key Feature         | Pros              | Use Case                   |
| ------------- | ------------------- | ----------------- | -------------------------- |
| Vanilla RNN   | Basic recurrence    | Simple            | Short sequences            |
| LSTM          | Gates, cell state   | Long-term memory  | Text, speech, time-series  |
| GRU           | Simplified LSTM     | Faster            | Time-series, NLP           |
| BRNN          | Forward+Backward    | Full context      | NLP                        |
| Deep RNN      | Stacked layers      | Abstract features | Complex tasks              |
| ESN           | Reservoir computing | Fast training     | Dynamic systems            |
| Attention-RNN | Attention           | Focus on sequence | Translation, summarization |

---

## **11. Layers in RNN**

* **Input:** One-hot, embeddings.
* **Hidden:** SimpleRNN / LSTM / GRU.
* **Stacked layers:** Multiple hidden layers.
* **Dropout:** Prevent overfitting.
* **Output:** Dense + activation.

---

## **12. Optimization**

* **Optimizers:** Adam, RMSProp, SGD.
* **Regularization:** Dropout, gradient clipping.

---

## **13. Evaluation Metrics**

* Classification: Accuracy, Precision, Recall, F1.
* Sequence: Perplexity, BLEU score.
* Regression: RMSE, MAE, R².

---

## **14. Time-Series Forecasting Example (LSTM)**

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sine wave data
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps)

# Create sequences
def create_sequences(data, seq_len=50):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(data)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM model
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(X.shape[1],1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=50, batch_size=32)

# Predict
pred = model.predict(X)

# Plot
plt.plot(y, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.show()
```

---

## **15. Advantages & Limitations**

**Advantages:**

* Captures temporal dependencies.
* Handles variable-length sequences.
* Strong for sequential data tasks.

**Limitations:**

* Slow training for long sequences.
* Vanilla RNN suffers from vanishing gradients.
* Transformers often outperform in NLP.

---

## **16. Future Directions**

* Still relevant for **time-series forecasting, IoT, real-time systems**.
* NLP tasks → Largely replaced by **Transformers**.
* Lightweight, efficient RNNs can be used on **edge devices**.

---

✅ **Lecture Highlights:**

* Full theory, math, architecture, and working.
* Variants of RNN with pros, cons, and use-cases.
* Activation functions, loss functions, layers, optimization.
* Real-world application with **LSTM time-series code**.

---

