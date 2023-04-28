# Chapter 22: The Parable of the Truncated Newton (TNewton) Method

Welcome, dear reader, to the twenty-second chapter of our book on Machine Learning Optimizers. In the previous chapter, we explored the Dogleg Method, which is a popular optimization algorithm used in unconstrained and constrained nonlinear optimization problems. Now, we will delve into the Truncated Newton (TNewton) Method, which is another widely used optimization algorithm in the field of machine learning.

The Truncated Newton Method, also known as the Hessian-free optimization method, is a second-order optimization algorithm that could compute the second derivative of the objective function (Hessian) without explicitly calculating it. Instead of computing the Hessian matrix, it estimates it efficiently by means of conjugate gradient techniques. This makes the TNewton method computationally cheaper than the standard Newton's method, which requires solving a linear system of equations involving the Hessian matrix.

The Truncated Newton Method is particularly useful when the objective function is non-convex, and there are many local minima. The TNewton method performs well in such situations by converging to the global minimum or at least a good local minimum. Moreover, it has been shown that the TNewton method is superior to other optimization algorithms in deep learning tasks such as training neural networks.

In this chapter, we will illustrate the TNewton method through the parable of the Wizard of Oz. Our main character, Dorothy, with the help of her friends, will navigate through the Land of Oz to find the optimal solution for a nonlinear optimization problem. We will provide Python code examples to demonstrate the TNewton algorithm in action.

So, fasten your seatbelts as we embark on another exciting journey into the world of optimization algorithms. Let the adventure begin!
# The Parable of the Truncated Newton (TNewton) Method: Dorothy's Quest for the Optimal Solution

Once upon a time, in the magical land of Oz, there lived a young girl named Dorothy. She was a curious and adventurous soul who loved to explore new places and solve problems. One day, she stumbled upon a nonlinear optimization problem that needed solving.

Dorothy knew that the problem was complex and required a powerful optimization algorithm. She decided to seek the help of her trusted friends, Scarecrow, Tin Man, and Cowardly Lion, who were known for their problem-solving skills and bravery. Together, they set out on a quest to find the optimal solution.

As they traversed through the lush forests and towering mountains of Oz, they encountered many obstacles and challenges. The problem was non-convex, which meant that it had many local minima, and finding the global minimum was a daunting task. But Dorothy and her friends were determined to find the best solution possible.

After many days of wandering, they stumbled upon a wise old wizard who went by the name of TNewton. TNewton was a master of optimization algorithms and had helped many travelers solve complex optimization problems.

Upon hearing the dilemma of Dorothy and her friends, TNewton offered to help them solve the problem using his powerful optimization algorithm. TNewton explained that his algorithm, the Truncated Newton Method, could efficiently estimate the second derivative of the objective function without explicitly calculating it, making it computationally cheaper and faster.

Excited by the prospect, Dorothy and her friends eagerly agreed to TNewton's proposal. TNewton then taught them how to use his algorithm to find the optimal solution.

First, they initialized their starting point, which was arbitrary. They then computed the gradient of the objective function at that point using automatic differentiation techniques. They used the gradient information to update their current point by taking a small step in the direction of steepest descent.

Next, they estimate the curvature information by using the conjugate gradient method to solve a sequence of linear systems with the Jacobian of the gradient as the coefficient matrix. This step allows the algorithm to adapt according to the curvature of the objective function at each point.

Finally, they took a step along the conjugate gradient direction, truncated based on the predicted improvement in the objective function, and repeat the above until convergence.

After applying the TNewton method for a few iterations, Dorothy and her friends found the optimal solution to the problem. They were amazed by the efficiency and effectiveness of the TNewton method and grateful to TNewton for his help.

As they bid farewell to TNewton and continued on their journey, Dorothy and her friends realized the importance of having the right tools and techniques to solve complex optimization problems. They also learned that with determination and the right mindset, any problem could be solved, no matter how challenging.
# The Code Behind the Parable of the Truncated Newton (TNewton) Method

In the previous parable, we followed Dorothy and her friends as they use the Truncated Newton (TNewton) Method to solve a nonlinear optimization problem. In this section, we will illustrate the TNewton method using Python code examples.

The TNewton method is a powerful optimization algorithm that can efficiently estimate the Hessian matrix without explicitly calculating it, making it faster and more computationally efficient than other second-order methods. However, implementing the TNewton method requires some advanced knowledge of optimization algorithms.

Here's an example implementation of the TNewton method in Python:

``` python
import numpy as np
from scipy import optimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2 + x[2]**2

# Define the gradient of the objective function
def gradient_function(x):
    return np.array([2*x[0], 2*x[1], 2*x[2]])

# Define the Hessian vector product function
def hessian_vector_product(x, v):
    eps = 1e-6
    result = np.zeros_like(x)

    for i in range(len(x)):
        x_plus_eps = x.copy()
        x_plus_eps[i] += eps
        x_minus_eps = x.copy()
        x_minus_eps[i] -= eps
        result[i] = (objective_function(x_plus_eps) - objective_function(x_minus_eps)) / (2 * eps)

    return result + v

# Define the TNewton method
def truncated_newton_method(x0):
    # Define the maximum number of iterations
    max_iter = 100

    # Define the tolerance for stopping criterion
    tol = 1e-6

    # Initialize the current point and the increment
    x = x0
    inc = np.zeros_like(x)

    # Initialize the scaling parameter
    scale = 1.0

    # Perform the iterations until convergence or maximum number of iterations
    for i in range(max_iter):
        # Compute the gradient at the current point
        g = gradient_function(x)

        # Check convergence criterion
        if np.linalg.norm(g) < tol:
            break

        # Compute the product of the Hessian and the increment
        Hv = hessian_vector_product(x, inc)

        # Compute the step direction using the conjugate gradient method
        d = optimize.minimize(fun=lambda d: np.dot(d, Hv) + np.dot(g, d), x0=inc, method='CG', jac=lambda d: np.dot(hessian_vector_product(x, d), inc) + gradient_function(x)).x

        # Compute the predicted improvement
        pred = -np.dot(g, inc) - 0.5 * np.dot(inc, Hv)

        # Compute the actual improvement
        actual = objective_function(x + d) - objective_function(x)

        # Compute the scaling parameter
        rho = actual / pred

        # Update the current point and the increment
        x += d
        inc += rho * d

        # Update the scaling parameter
        if rho < 0.25:
            scale *= 0.25
        elif rho > 0.75:
            scale *= 2.0

    # Return the optimal solution
    return x
```

In this code, we first define our objective function, gradient function, and Hessian vector product function. The Hessian vector product function uses finite differences to estimate the Hessian matrix by approximating the Jacobian of the gradient function.

Next, we define the TNewton method, which takes an initial point as input and performs iterations until convergence or a maximum number of iterations. In each iteration, we compute the gradient at the current point, approximate the product of the Hessian and the increment vector, and use the conjugate gradient method to compute the step direction. We then compute the predicted and actual improvements, update the current point and the increment vector, and update the scaling parameter.

Finally, we return the optimal solution.

By using this Python code, we can efficiently implement the TNewton method and solve complex nonlinear optimization problems like the one that Dorothy and her friends encountered in the Land of Oz.


[Next Chapter](23_Chapter23.md)