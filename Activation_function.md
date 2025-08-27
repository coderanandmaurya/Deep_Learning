---

# 📖 Lecture on Activation Functions in Neural Networks

---

## 1. Introduction

Artificial Neural Networks (ANNs) are inspired by biological neurons.
A neuron in the brain fires only when it receives a strong enough input.
In ANNs, this “firing” is controlled by **activation functions**.

Without activation functions, a neural network would only be a **linear model**.
That means no matter how many layers you add, the network would behave like **simple linear regression**.

So, activation functions add **non-linearity**, enabling networks to solve complex problems like image recognition, speech, and natural language processing.

---

## 2. Why Do We Need Activation Functions?

* They introduce **non-linear behavior**.
* They help networks **learn complex mappings** between input and output.
* They control **gradient flow** during training.
* They determine **task suitability**: classification, regression, or clustering.

---

## 3. Step Function

The oldest and simplest activation.

### Formula:

$$
f(x) = 
\begin{cases} 
1 & \text{if } x \geq 0 \\ 
0 & \text{if } x < 0
\end{cases}
$$

### Intuition:

* Like a switch: ON (1) or OFF (0).
* Used in early perceptrons.

### Pros:

* Simple and intuitive.

### Cons:

* Not differentiable → can’t be used with gradient descent.
* No probability meaning.
* Poor at learning complex patterns.

### Use:

* Only in **theoretical models** or perceptrons.
* Not used in modern deep learning.

---

## 4. Sigmoid Function

Squashes input into the range (0,1).

### Formula:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

### Intuition:

* Converts inputs into probability-like values.
* Useful for binary decisions.

### Pros:

* Good for probability outputs.

### Cons:

* Causes **vanishing gradients**.
* Outputs not zero-centered.
* Expensive exponential computation.

### Use:

* **Binary classification output layer**.
* Example: Spam (1) vs Not Spam (0).

---

## 5. Tanh (Hyperbolic Tangent)

### Formula:

$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

### Range: (-1, 1)

### Intuition:

* Like sigmoid but outputs negative values too.
* Centers data around zero.

### Pros:

* Zero-centered → better gradient flow than sigmoid.

### Cons:

* Still suffers vanishing gradient for large |x|.

### Use:

* **Hidden layers** in older networks.
* Sometimes in **autoencoders**.

---

## 6. ReLU (Rectified Linear Unit)

The most widely used activation today.

### Formula:

$$
f(x) = \max(0, x)
$$

### Intuition:

* Negative inputs → 0.
* Positive inputs → same as input.

### Pros:

* Very fast.
* Reduces vanishing gradient problem.
* Produces sparse activations.

### Cons:

* **Dying ReLU problem**: some neurons always output 0.

### Use:

* **Hidden layers** in classification and regression.

---

## 7. Leaky ReLU

Improvement over ReLU with a small slope for negative values.

### Formula:

$$
f(x) = 
\begin{cases} 
x & x > 0 \\ 
\alpha x & x \leq 0
\end{cases}
$$

Here, **α** is a small constant (e.g., 0.01).

### Pros:

* Fixes dying ReLU by allowing small gradients when $x < 0$.

### Cons:

* Requires selecting α.

### Use:

* Hidden layers in **classification** and **regression**.

---

## 8. Parametric ReLU (PReLU)

A variant of Leaky ReLU where α is **learnable**.

### Formula:

$$
f(x) = 
\begin{cases} 
x & x > 0 \\ 
\alpha x & x \leq 0
\end{cases}
$$

* Here α is trained along with network weights.

### Pros:

* Learns slope automatically.

### Cons:

* Adds parameters → more complexity.

### Use:

* CNNs where flexibility improves performance.

---

## 9. Exponential Linear Unit (ELU)

### Formula:

$$
f(x) = 
\begin{cases} 
x & x > 0 \\ 
\alpha (e^x - 1) & x \leq 0
\end{cases}
$$

Where α > 0.

### Pros:

* Smooth curve.
* Allows negative outputs.
* Faster convergence.

### Cons:

* More expensive than ReLU.

### Use:

* Hidden layers in **classification** tasks.

---

## 10. Swish

Introduced by Google Brain.

### Formula:

$$
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### Pros:

* Smooth and non-monotonic.
* Outperforms ReLU in many deep networks.

### Cons:

* Heavier computation than ReLU.

### Use:

* Modern **classification and regression** problems.

---

## 11. GELU (Gaussian Error Linear Unit)

Popular in Transformers (BERT, GPT).

### Formula:

$$
f(x) = x \cdot \Phi(x)
$$

Where $\Phi(x)$ = Gaussian CDF.

### Pros:

* Combines ReLU and Sigmoid properties.
* Smooth, probabilistic.

### Cons:

* Complex math.

### Use:

* **NLP models** and transformers.

---

## 12. Maxout

### Formula:

$$
f(x) = \max(w_1^Tx+b_1, w_2^Tx+b_2)
$$

### Pros:

* Very flexible, learns its own activation.

### Cons:

* Doubles parameters → heavy model.

### Use:

* **CNNs for classification**.

---

## 13. Softmax

Special for multi-class classification.

### Formula:

$$
f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

### Intuition:

* Outputs probabilities across classes.
* Values sum to 1.

### Pros:

* Perfect for multi-class classification.

### Cons:

* Computationally expensive for large outputs.

### Use:

* **Output layer in multi-class classification**.

---

## 14. Choosing the Right Activation Function

* **Binary Classification** → Sigmoid (output).
* **Multi-class Classification** → Softmax (output).
* **Regression** → Linear (no activation in output).
* **Hidden Layers** → ReLU, Leaky ReLU, PReLU, ELU, Swish, GELU.
* **Clustering / Autoencoders** → Tanh or ReLU.

---

## 15. Rule of Thumb

* Use **ReLU (or variant)** in hidden layers.
* Use **Sigmoid** for binary outputs.
* Use **Softmax** for multi-class outputs.
* Use **Linear** for regression outputs.
* Use **GELU or Swish** in advanced research models.

---

## 16. Closing Notes

Activation functions are the **soul of deep learning**.

* They enable non-linear transformations.
* They make neural networks powerful enough to model speech, vision, and language.
* Choosing the right activation depends on **task type**: classification, regression, clustering.

To summarize:

* **Step** → Only theory.
* **Sigmoid & Tanh** → Older networks, still useful for probabilities.
* **ReLU and variants (Leaky, PReLU, ELU)** → Standard for hidden layers.
* **Swish & GELU** → Advanced, used in modern architectures.
* **Softmax** → Multi-class classification.
* **Linear** → Regression.

---
