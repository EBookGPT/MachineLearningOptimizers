# Chapter 19: Introduction to Quasi-Newton Methods

Welcome back, dear readers, to another informative chapter on Machine Learning Optimizers. In our previous chapter, we discussed Broyden's method in detail. Today, we will continue our journey towards understanding optimization algorithms with another prominent method – the Quasi-Newton method.

Quasi-Newton methods, based on the Newton-Raphson optimization algorithm, are widely used for unconstrained optimization problems. It is a popular choice because it doesn't require second-order derivatives, making it less computationally expensive than the classical Newton-Raphson method. Additionally, Quasi-Newton methods can handle non-linear and non-smooth optimization problems, making them a favorite among machine learning experts.

Throughout this chapter, we will follow the yellow brick road towards understanding the fundamental concepts of Quasi-Newton methods. We will explain how it works, highlight its advantages, and touch upon its implementation. Furthermore, we will provide sample code to help you get started in building your own implementations.

So, slip on your ruby red slippers and get ready as we embark on yet another exciting adventure towards understanding optimization algorithms.


> "If you want to understand the universe, think in terms of energy, frequency, and vibration." - Nikola Tesla
# Chapter 19: Quasi-Newton Methods - A Wizard of Oz Parable

Once upon a time, in the magical land of Oz, the Great Wizard tried to find the optimal route to Emerald City. He traveled through fields of poppies, crossed streams, and climbed mountains. However, the journey was long and tedious, and he wanted to find a quicker way to get there. The Good Witch of the North came to his aid and offered him a solution to his problem.

"You need to optimize your journey, Great Wizard," said the Good Witch of the North. "Use the Quasi-Newton method to find the optimal route."

The Great Wizard was puzzled. "What is the Quasi-Newton method? How does it work?"

The Good Witch of the North replied, "It's an optimization algorithm used to find the minimum of a function. It's a modification of the Newton-Raphson method, but instead of using the exact Hessian matrix, it approximates it with a positive-definite matrix. The resulting algorithm is computationally less expensive and well-suited for unconstrained optimization."

The Great Wizard listened intently, and the Good Witch of the North continued, "The Quasi-Newton method works by iteratively updating an approximation of the inverse Hessian matrix using the difference between the gradient vectors of the objective function. This approximation is then used to find the minimum of the function efficiently."

The Great Wizard felt enlightened. He thanked the Good Witch of the North and set about implementing the Quasi-Newton method. With each iteration, the optimization algorithm got better and found the optimal route to Emerald City.

As the Great Wizard arrived in Emerald City, he was delighted with the speed at which he arrived. He realized that the Quasi-Newton method was not only powerful but also practical in solving optimization problems. From that day forward, the Great Wizard used the Quasi-Newton method for all his optimization needs.

And so dear readers, the moral of the story is that optimization algorithms like the Quasi-Newton method can help you find the optimal route to your destination. As you journey towards optimization, remember to use the best tools available and take the advice of wise wizards and witches.

> "All we have to decide is what to do with the time that is given us." - J.R.R. Tolkien
# Chapter 19: Implementing Quasi-Newton Methods - Code Explanation

Now that we have explored the Quasi-Newton method as a Wizard of Oz parable, it's time to dive into the implementation details. In this section, we will explain the code used to solve our optimization problem in the parable.

First, let's review the basics of the Quasi-Newton method. The algorithm involves iteratively updating an approximation of the inverse Hessian matrix using the difference between the gradient vectors of the objective function. This approximation is then used to find the minimum of the function efficiently.

The implementation in Python can be done using the `scipy.optimize.minimize` function. We can specify the method to be `BFGS` (short for Broyden-Fletcher-Goldfarb-Shanno method), which is a popular Quasi-Newton method. Here is the code for solving a simple optimization problem using the `BFGS` method:

```python
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return x[0]**2 + 2 * x[1]**2

# Define the gradient of the objective function
def gradient(x):
    return np.array([2 * x[0], 4 * x[1]])

# Initial guess for the optimization variable
x0 = np.array([1, 1])

# Call the `minimize` function from `scipy.optimize`
result = minimize(fun=objective, x0=x0, method='BFGS', jac=gradient, options={'disp': True})

print(result.x)
```

In this code, we define the objective function as `x[0]**2 + 2 * x[1]**2`, and the gradient as `[2 * x[0], 4 * x[1]]`. We then set an initial guess for the optimization variable using `x0 = np.array([1, 1])`. Finally, we call the `minimize` function from `scipy.optimize`, passing in the objective function, initial guess and gradient function, and setting the method to be `BFGS`. The `disp` option is also set to `True`, so that we can see some output from the optimization process.

When we run the code, we get the following output:

```
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 3
         Function evaluations: 4
         Gradient evaluations: 4
[ 0.  0.]
```

The output tells us that the optimization terminated successfully, and we found the minimum of the function to be `0.000`. It took 3 iterations and 4 function/gradient evaluations to converge to the solution. The optimal solution for the optimization variable is `[0., 0.]`.

And that's it for the implementation of the Quasi-Newton method using Python! As you can see, implementing the method is relatively straightforward using the `scipy.optimize` library. With the help of the Good Witch of the North and the Quasi-Newton method, we can solve optimization problems with ease.

> "A good code is its own best documentation." - Steve McConnell


[Next Chapter](20_Chapter20.md)