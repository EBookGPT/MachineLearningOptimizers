# Chapter 4: Stochastic Gradient Descent (SGD)

Welcome back! In the previous chapter, we covered Gradient Descent, which is a widely used algorithm for minimizing a cost function in Machine Learning. However, Gradient Descent can be impractical when the dataset is large. That's where Stochastic Gradient Descent (SGD) comes in.

SGD is an iterative method for optimizing an objective function with a large dataset. Unlike Gradient Descent, which involves computing gradients for the entire dataset, SGD estimates the gradients using a subset of the data. This makes SGD much faster than Gradient Descent and more scalable to large datasets.

The name "stochastic" comes from the fact that the gradient estimate is based on a random subset of data. The subset, also known as a mini-batch, typically contains several hundred or a few thousand examples. By using this subset, the algorithm can escape from local minima and converge to the global minimum more quickly compared to batch processing.

But how does SGD actually work? Let's dive in and find out.

```python
# Stochastic Gradient Descent (SGD) code example
import numpy as np

def stochastic_gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    for epoch in range(epochs):
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X[rand_ind,:].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)
            grad = X_i.T.dot(X_i.dot(theta) - y_i)
            theta = theta - (alpha * grad)
    return theta
```

Above is a sample Python code for implementing SGD. The function takes in the feature matrix `X`, target variable vector `y`, current parameter vector `theta`, learning rate `alpha`, and number of epochs `epochs`. Inside the nested `for` loops, we randomly select a single example from the dataset, compute its gradient, and update the parameter vector. The process is repeated for the specified number of epochs until the model converges.

As with any optimization algorithm, there are certain trade-offs to be aware of. We will explore these further in the following section.
# The Parable of the Stochastic Wizard

Once upon a time in the magical land of Oz, there lived a young wizard named Glinda who dreamt of building the most powerful spellcasting machine in all the land. She knew that the key to her success lay in a powerful algorithm called Stochastic Gradient Descent, which would allow her to train her machine on vast datasets in record time.

One day, Glinda embarked on a perilous journey to the far corners of Oz to retrieve the secret ingredients for her master algorithm. She passed through treacherous forests, battled fearsome monsters, and braved the scorching heat of the desert until she arrived at the dark wizard's lair.

The dark wizard was intrigued by Glinda's quest and decided to test her. He challenged her to train a spellcasting machine to predict the outcome of a complex spell. He offered her a vast dataset with thousands of magic spells, each with dozens of different features.

Glinda knew that traditional machine learning algorithms would never work in such a scenario due to the vast size of the dataset. But she was undaunted. She summoned her courage and proceeded to implement Stochastic Gradient Descent.

Using the magic of Python, Glinda programmed a function that would train her machine on the vast dataset using mini-batches of data, incrementally updating her algorithm's parameters as it iterated over each batch. Her cunning strategy allowed her to converge on the perfect algorithm in mere hours, whereas the dark wizard's machine took weeks to train.

In the end, Glinda emerged victorious, having proven the power of her Stochastic Gradient Descent algorithm. She built the most powerful spellcasting machine in all the land, and wizards from far and wide came to learn her secrets. From that day forward, Stochastic Gradient Descent became the standard algorithm for all manner of magical incantations, and Glinda's legend lived on for generations to come.

The moral of the story? When it comes to training complex algorithms on massive datasets, Stochastic Gradient Descent is the way to go. By leveraging a smaller subset of the dataset to estimate gradients, SGD allows us to train models more efficiently and effectively, regardless of how much data we have at our disposal.
Sure, let me explain the code we used in our parable to implement the Stochastic Gradient Descent (SGD) algorithm.

```python
# Stochastic Gradient Descent (SGD) code example
import numpy as np

def stochastic_gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    for epoch in range(epochs):
        for i in range(m):
            rand_ind = np.random.randint(0, m)
            X_i = X[rand_ind,:].reshape(1, X.shape[1])
            y_i = y[rand_ind].reshape(1, 1)
            grad = X_i.T.dot(X_i.dot(theta) - y_i)
            theta = theta - (alpha * grad)
    return theta
```

The code defines a function called `stochastic_gradient_descent` that takes in four arguments:

* `X`: the feature matrix of our dataset
* `y`: the target variable vector of our dataset
* `theta`: the current parameter vector
* `alpha`: the learning rate
* `epochs`: the number of iterations over the full dataset

The function initializes the number of examples in our dataset `m`. It then enters two nested loops, one for each epoch and one for each example in our dataset.

During each iteration, the function selects a random example from our dataset using the `np.random.randint` function. It then shrinks the X_i and y_i matrices into a column vector each, so we can dot product and work with them.

The function computes the gradient for the selected example and updates the parameter vector using the learning rate `alpha`. This process is repeated `epochs` times until the function converges to an optimal parameter value.

In conclusion, the SGD function performs an efficient optimization of the cost function by randomly selecting a subset of the examples and computing the grandients based on that smaller subset of examples. This makes it an excellent choice for dealing with large datasets where other gradient-based optimization algorithms may become impractical.


[Next Chapter](05_Chapter05.md)