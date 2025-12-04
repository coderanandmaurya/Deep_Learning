# python-fundamentals-for-data-analytics.md

```markdown
# Python Fundamentals for Data Analytics

This document covers the foundational elements of Python programming, with a focus on how it supports data analytics tasks. Python is a versatile, beginner-friendly programming language widely used in data analytics due to its simplicity, extensive libraries, and community support. It's free, open-source, and runs on multiple platforms.

## Why Python for Data Analytics?

- **Readability**: Code is clean and easy to understand.
- **Libraries**: Pandas for data manipulation, NumPy for numerical computing, Matplotlib/Seaborn for plotting, and Scikit-learn for machine learning.
- **Integration**: Works with web data, databases, and big data tools.
- **Community**: Vast resources like Stack Overflow and Jupyter Notebooks for interactive coding.

## Basic Python Syntax

Python uses indentation (spaces or tabs) to define code blocks—no curly braces like in other languages.

### Variables and Data Types

Variables store data; no need to declare types explicitly.

```python
# Integer (whole number)
age = 25

# Float (decimal)
height = 5.9

# String (text)
name = "Alice"

# Boolean (True/False)
is_student = True

# Print output
print(f"{name} is {age} years old.")  # Output: Alice is 25 years old.
```

### Control Structures

- **If-Else Statements**: For conditional logic.

```python
score = 85
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")  # Output: Grade: B
else:
    print("Grade: C")
```

- **Loops**: For repetition.

```python
# For loop (iterate over a range)
for i in range(3):
    print(i)  # Output: 0 1 2

# While loop
count = 0
while count < 3:
    print(count)
    count += 1  # Output: 0 1 2
```

### Functions

Reusable blocks of code. Define with `def`.

```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

grades = [80, 90, 70]
avg = calculate_average(grades)
print(avg)  # Output: 80.0
```

### Data Structures

Essential for handling data collections.

- **Lists**: Ordered, mutable (changeable) arrays.

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("date")  # Add item
print(fruits[0])  # Output: apple
```

- **Dictionaries**: Key-value pairs for structured data.

```python
person = {"name": "Bob", "age": 30}
print(person["name"])  # Output: Bob
```

- **Tuples**: Ordered, immutable lists (useful for fixed data).

```python
coordinates = (10, 20)
```

### Key Libraries for Data Analytics

- **NumPy**: For arrays and math operations.

```python
import numpy as np
array = np.array([1, 2, 3])
print(np.mean(array))  # Output: 2.0
```

- **Pandas**: For data frames (like Excel tables).

```python
import pandas as pd
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print(df.head())  # Displays the table

Practice in a Jupyter Notebook for interactive exploration—install via Anaconda for a full setup.
```
## Test your knowledge with these questions. .

1. **What is the primary goal of descriptive analytics?**  
   a) Predict future trends  
   b) Explain past events  
   c) Summarize historical data  
   d) Recommend actions  
   **Answer: c) Summarize historical data** (It focuses on "what happened.")

2. **In Python, which data type is used for text?**  
   **Answer: String** (e.g., `name = "Hello"`)

3. **Write a simple if-statement in Python to check if a number `x` is even.**  
   **Answer:**  
   ```python
   if x % 2 == 0:
       print("Even")
   else:
       print("Odd")
   ```

4. **What does the `append()` method do to a list?**  
   **Answer: Adds an element to the end of the list** (e.g., `my_list.append(5)`)

5. **Name one key advantage of using Python for data analytics over Excel.**  
   **Answer: Automation and scalability for large datasets** (or handling complex computations via libraries like Pandas.)

6. **What is the output of `print(range(3))` in Python?**  
   **Answer: range(0, 3)** (It creates a range object; use `for i in range(3): print(i)` to see 0 1 2.)

7. **In a dictionary, how do you access a value using its key?**  
   **Answer: Using square brackets, e.g., `my_dict["key"]`** 

8. **What library is commonly used in Python for data manipulation, similar to Excel?**  
   **Answer: Pandas**

9. **Explain the difference between a list and a tuple in Python.**  
   **Answer: Lists are mutable (can be changed), tuples are immutable (cannot be changed after creation.)**

10. **What is prescriptive analytics? Give a brief example.**  
    **Answer: It recommends actions based on analysis. Example: An algorithm suggesting the best route for delivery trucks to minimize fuel use.** 
