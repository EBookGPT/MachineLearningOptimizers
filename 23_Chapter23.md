# Chapter 23: Proximal Gradient Method

Welcome back, dear reader! In the previous chapter, we learned about Truncated Newton (TNewton) method. Today, we explore another powerful optimization algorithm- the Proximal Gradient method.

To help us fully grasp the Proximal Gradient method, we have a special guest with us today- Simon Du. Simon is a renowned expert in optimization & machine learning, and his work has been published in several top-tier journals such as JMLR, NIPS, and ICML.

Simon, would you like to say a few words before we get started?

```python
Simon: Thank you for having me, and it's my pleasure to help. Proximal Gradient method is one of my favorite optimization algorithms. It has a wide range of applications in both convex and non-convex optimization, and is particularly useful for solving problems where the objective function has a proper lower semi-continuous convex part, and a non-smooth but easy-to-proximate regularization penalty.
```

Perfect! So, let's dive into the Proximal Gradient method and understand its workings.

```python
def proximal_gradient_method(objective_function, gradient_function, proximal_operator, learning_rate, threshold, max_iterations):
    """
    Proximal Gradient Method.
    
    objective_function (function): Function to optimize.
    gradient_function (function): Function to compute the gradient of objective function.
    proximal_operator (function): Function to compute proximal operator of regularization penalty.
    learning_rate (float): Learning rate used for weight update.
    threshold (float): Convergence threshold.
    max_iterations (int): Maximum number of iterations.
    
    Returns:
    Optimal weights for objective function.
    """
    
    weights = initialize_weights() # Set initial weights
    for i in range(max_iterations):
        gradient = gradient_function(weights) # Compute gradient
        weights = proximal_operator(weights - learning_rate * gradient, learning_rate) # Update weights
        if np.linalg.norm(weights - old_weights) < threshold: # Check convergence
            break
        old_weights = weights
    return weights
```

The Proximal Gradient method follows a simple idea- at every iteration, it combines two steps- a gradient descent step on the convex part of the objective function, and a proximal step on the non-smooth regularization penalty. The proximal operator computes the point in the proximity of the current weights that minimizes the sum of the regularization penalty and a weighted L2-norm.

```python
def l1_proximal_operator(weights, learning_rate):
    """
    L1-norm proximal operator.
    
    weights (numpy array): Weights to compute proximal operator.
    learning_rate (float): Learning rate.
    
    Returns:
    Proximal operator of L1-norm regularization penalty.
    """
    
    return np.sign(weights) * np.maximum(np.abs(weights) - learning_rate, 0)
```

The Proximal Gradient method is a powerful tool in the optimization toolbox and can be used to solve several real-world problems. Stay tuned for the next chapter, where we will explore another exciting optimization algorithm.

Once again, many thanks to Simon Du for sharing his expertise with us. That's all for now, folks!
# Chapter 23: Proximal Gradient Method - The Wizard of Oz Parable

Once upon a time, in the Land of Machine Learning, the Scarecrow was looking for a way to optimize his learning algorithm. He had heard about the Truncated Newton method from the Tin Man and learned quite a bit about it, but he was aware that it had some limitations in terms of the size of the training dataset. 

However, his quest led him to meet Glinda the Good Witch, who told him about an alternative optimization method called the Proximal Gradient method. She explained how it was more robust and could handle large datasets more effectively.

Excited about this new method, the Scarecrow immediately set out to find the Proximal Gradient method, but he was lost and didn't know where to start. However, just as in the original story, the Good Witch of the North appeared and gave him a map to follow.

As he started his journey, the Scarecrow was joined by a special guest, Simon Du, a wise sage with extensive experience in optimization and machine learning. Simon accompanied him on his journey and shared his insights and knowledge about the Proximal Gradient method. 

The Scarecrow learned that the Proximal Gradient method is a powerful optimization algorithm that is useful when the objective function has a proper lower semi-continuous convex part and a non-smooth but easy-to-proximate regularization penalty. Simon showed him how to combine a gradient descent step on the convex part of the objective function with a proximal step on the regularization penalty, which led to a faster convergence.

The Scarecrow was grateful for Simon's guidance and used the Proximal Gradient method to optimize his learning algorithm successfully. He knew that he could use this powerful method to solve several real-world problems, and was pleased to have added Proximal Gradient method to his arsenal of optimization tools.

With Simon by his side, the Scarecrow continued on his journey, eager to discover new optimization methods and improve his machine learning skills.
# Explanation of the Code

In the Wizard of Oz parable, we introduced the Proximal Gradient method as a way for the Scarecrow to optimize his learning algorithm. We used Python code to show how the Proximal Gradient method works:

```python
def proximal_gradient_method(objective_function, gradient_function, proximal_operator, learning_rate, threshold, max_iterations):
    """
    Proximal Gradient Method.
    
    objective_function (function): Function to optimize.
    gradient_function (function): Function to compute the gradient of objective function.
    proximal_operator (function): Function to compute proximal operator of regularization penalty.
    learning_rate (float): Learning rate used for weight update.
    threshold (float): Convergence threshold.
    max_iterations (int): Maximum number of iterations.
    
    Returns:
    Optimal weights for objective function.
    """
    
    weights = initialize_weights() # Set initial weights
    for i in range(max_iterations):
        gradient = gradient_function(weights) # Compute gradient
        weights = proximal_operator(weights - learning_rate * gradient, learning_rate) # Update weights
        if np.linalg.norm(weights - old_weights) < threshold: # Check convergence
            break
        old_weights = weights
    return weights
```

The `proximal_gradient_method()` function takes as inputs the objective function, gradient function, proximal operator, learning rate, convergence threshold, and maximum number of iterations. The function returns the optimal weights (or values) for the objective function.

In the Proximal Gradient method, the weights are updated by combining a gradient descent step on the convex part of the objective function with a proximal step on the non-smooth regularization penalty. The proximal operator computes the point in the proximity of the current weights that minimizes the sum of the regularization penalty and a weighted L2-norm. In our example, we used the L1-norm proximal operator:

```python
def l1_proximal_operator(weights, learning_rate):
    """
    L1-norm proximal operator.
    
    weights (numpy array): Weights to compute proximal operator.
    learning_rate (float): Learning rate.
    
    Returns:
    Proximal operator of L1-norm regularization penalty.
    """
    
    return np.sign(weights) * np.maximum(np.abs(weights) - learning_rate, 0)
```

The `l1_proximal_operator()` function computes the proximal operator of the L1-norm regularization penalty.

Finally, we want to reiterate that the Proximal Gradient method is a powerful optimization method that can be used to solve several real-world problems. We encourage readers to explore this method further and apply it to their own problems.


[Next Chapter](24_Chapter24.md)