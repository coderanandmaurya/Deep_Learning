# 📘 Lecture: Recurrent Neural Networks (RNN)

---

## 1. Introduction to RNN

* **Recurrent Neural Networks (RNNs)** are a class of neural networks specifically designed for **sequential and temporal data**.
* Unlike feedforward networks, RNNs **retain information from previous steps** through hidden states.
* Useful when **context and order matter** (e.g., language, speech, stock trends).

👉 Example:

* Feedforward NN predicting image = "cat" or "dog" → order doesn’t matter.
* Predicting the next word in “I am going to the …” → order **does** matter.

---

## 2. Historical Context

* **1980s–1990s**: Early RNN models by **Jordan** and **Elman**.
* **1997**: Hochreiter & Schmidhuber proposed **LSTM**, solving vanishing gradients.
* **2010s**: RNNs widely used in NLP and speech.
* **Today**: Transformers dominate NLP, but RNNs remain strong for **time-series, real-time, and edge applications**.

---

## 3. Types of Artificial Neural Networks

* **Feedforward Neural Network (FNN)** → Input → Hidden → Output (no memory).
* **Convolutional Neural Network (CNN)** → Best for spatial data (images).
* **Recurrent Neural Network (RNN)** → Best for sequential/temporal data.

---

## 4. Sequential vs Non-Sequential Data

* **Non-sequential data**: Independent samples.

  * Example: Image classification.
* **Sequential data**: Order matters, past influences future.

  * Examples: Speech, text, time-series.

---

## 5. Why RNN is Different

* Traditional NN processes each input **independently**.
* RNN introduces **feedback loops**:

  * Stores past context in **hidden state**.
  * Can handle **variable-length sequences**.
* Learns **temporal dependencies**.

👉 Example:

* Sentiment depends on sequence:

  * “I love this movie” → positive.
  * “I don’t love this movie” → negative (requires context memory).

---

## 6. Applications of RNN

* **Natural Language Processing (NLP)**: Machine translation, text generation, sentiment analysis.
* **Speech Recognition**: Converting audio → text.
* **Time-Series Forecasting**: Stock market, weather prediction.
* **Music & Text Generation**: Create melodies, poems, or stories.
* **Healthcare**: Predict patient health from sequential data.
* **Video Analysis**: Captioning, action recognition.

---

## 7. Architecture of RNN

### Components

1. **Input Layer** → Sequential data ((x_t)).
2. **Hidden Layer** → Combines (x_t) + previous hidden state (h_{t-1}).
3. **Output Layer** → Produces prediction ((y_t)).

### Equations

* Hidden state update:
  [
  h_t = f(W_h h_{t-1} + W_x x_t + b)
  ]
* Output:
  [
  y_t = g(W_y h_t + c)
  ]

Where:

* (f) = activation function (tanh/ReLU).
* (g) = output activation (softmax/sigmoid/linear).

---

## 8. Working of RNN

1. First input (x_1) → hidden state (h_1).
2. Second input (x_2) + (h_1) → hidden state (h_2).
3. Repeat for all steps in sequence.
4. Output at each step (many-to-many) or final step (many-to-one).

👉 Example:

* Machine translation → output at every time step.
* Sentiment analysis → output only at last step.

---

## 9. Variants of RNN

* **Vanilla RNN** → Basic model, suffers from vanishing gradients.
* **LSTM (Long Short-Term Memory)** → Has input, output, forget gates.
* **GRU (Gated Recurrent Unit)** → Simplified LSTM.
* **Bidirectional RNN** → Processes sequence in both directions.
* **Deep RNN** → Stacked multiple layers for complexity.

---

## 10. Training RNN – Backpropagation Through Time (BPTT)

* **Unroll RNN** across time steps.
* Apply backpropagation to update weights.
* **Problems**:

  * Vanishing gradients (info lost over long sequences).
  * Exploding gradients (unstable updates).
* **Solutions**: Gradient clipping, LSTM, GRU.

---

## 11. Activation Functions in RNN

* **Hidden state**: tanh, ReLU, sigmoid.
* **Output layer**:

  * Softmax → multi-class classification.
  * Sigmoid → binary classification.
  * Linear → regression tasks.

👉 Example:

* Sentiment analysis → Softmax.
* Stock price prediction → Linear.

---

## 12. Loss Functions in RNN

* **Classification**: Cross-Entropy Loss (categorical/binary).
* **Sequence Prediction**: Cross-Entropy (token-by-token).
* **Regression/Forecasting**: MSE, MAE.

👉 Example:

* Next word prediction → Cross-Entropy.
* Predicting temperature → MSE.

---

## 13. Layers in RNN

* **Input layer**: One-hot encoding or Embedding.
* **Hidden layer**: SimpleRNN, LSTM, GRU.
* **Stacked RNN layers**: For deeper learning.
* **Dropout layers**: Prevent overfitting.
* **Output layer**: Dense with activation (sigmoid/softmax/linear).

👉 Example (Keras):

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## 14. Optimization in RNN

* **Optimizers**: SGD, Adam (most common), RMSProp (good for sparse updates).
* **Regularization**: Dropout, weight decay, gradient clipping.

---

## 15. Evaluation Metrics

* **Classification**: Accuracy, Precision, Recall, F1.
* **Sequence tasks**: Perplexity, BLEU score.
* **Regression**: RMSE, MAE, R².

---

## 16. Real-World Example

### Task: Sentiment Analysis using RNN

* **Input**: Movie reviews.
* **Process**:

  * Convert words → embeddings.
  * RNN processes words sequentially.
  * Final hidden state → sentiment prediction.
* **Loss**: Binary Cross-Entropy.
* **Activation**: Sigmoid (positive/negative).
* **Output**: Sentiment score (0 = negative, 1 = positive).

👉 Example Sentence:

* “The movie was not good.”

  * RNN remembers **“not”** before **“good”** → predicts negative sentiment.

---

## 17. Advantages & Limitations

✅ Advantages:

* Captures temporal dependencies.
* Works with variable-length input.
* Strong in sequence-based tasks.

❌ Limitations:

* Vanishing/exploding gradients.
* Struggles with long-term dependencies.
* Slow to train on long sequences.
* Replaced in NLP by Transformers.

---

## 18. Future Directions

* Still used in **time-series forecasting, speech recognition, IoT**.
* In NLP, **Transformers (BERT, GPT)** are preferred.
* RNNs remain useful where **efficiency and low-latency** are needed.

---

## 19. Summary

* RNNs → Designed for sequential data, with hidden state memory.
* Key components → Input, hidden, output layers.
* Training → Backpropagation Through Time (BPTT).
* Variants → LSTM, GRU, Bidirectional RNN.
* Loss functions & activations depend on task type.
* Applications → NLP, speech, forecasting, healthcare, video analysis.
* Future → Transformers dominate, but RNNs still important in real-time systems.

---
