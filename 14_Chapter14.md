# Chapter 14: Limited Memory Broyden-Fletcher Goldfarb Shanno (L-BFGS)

Welcome back, my fellow learners! In the previous chapter, we learned about the **Conjugate Gradient Method** and how it outperforms traditional gradient descent algorithms. Now, let's delve further into optimization algorithms with a special guest, Dr. Jorge Nocedal. Dr. Nocedal is a professor at Northwestern University and a renowned researcher in the field of optimization.

In this chapter, we'll explore the **Limited Memory Broyden-Fletcher Goldfarb Shanno (L-BFGS)** algorithm. It's an optimization method that approximates the inverse Hessian matrix, a commonly used mathematical object in optimization. L-BFGS is an iterative method that only needs a limited amount of memory to store the recent history of variables, making it feasible for high-dimensional optimization problems.

Dr. Nocedal has extensively worked on L-BFGS and contributed to its theoretical analysis over the years. In his 1980 publication with L. S. Jennings, "Quasi-Newton methods, motivation and theory," they proposed the L-BFGS two-loop recursion formula. This method computes and stores only a few vectors instead of the full Hessian matrix, which is computationally expensive to store and invert ([source](https://www.sciencedirect.com/science/article/pii/037702978090057X)). 

With Dr. Nocedal's guidance and expertise, let's understand how L-BFGS works and implement it in code.

```python
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der

# Define objective function (Rosenbrock function)
fun = rosen

# Define gradient function of objective function (Rosenbrock derivative)
jac = rosen_der

# Define initial point
x0 = [0, 0]

# Use L-BFGS algorithm for optimization
res = minimize(fun, x0, method='L-BFGS-B', jac=jac,
               options={'disp': True})
```

The code above applies L-BFGS to the Rosenbrock function, a commonly used optimization benchmark. The `minimize` function from SciPy's optimization module uses the L-BFGS algorithm (`'L-BFGS-B'` method) and the Rosenbrock derivative (`rosen_der`) as the gradient function of the objective function.

In the next section, we'll explore the inner workings of the L-BFGS algorithm and learn how it approximates the inverse Hessian matrix to optimize complex high-dimensional problems. Let's get started!
# Chapter 14: Limited Memory Broyden-Fletcher Goldfarb Shanno (L-BFGS)

Once upon a time in the land of Oz, there was a young wizard named Dorothy who had the power to optimize her spells. One day, she met a wise old wizard named Jorge Nocedal who taught her a new optimization method called Limited Memory Broyden-Fletcher Goldfarb Shanno (L-BFGS).

Dorothy had been using traditional gradient descent for her spell optimization, but she found it to be sluggish and unstable when dealing with high dimensional optimization problems. So, when she heard about L-BFGS, she was excited to learn more.

Jorge explained to Dorothy that L-BFGS was a quasi-Newton method that approximated the inverse Hessian matrix to optimize functions. He went on to say that L-BFGS was an iterative method, but unlike traditional methods, it only needs a limited amount of memory to store the recent history of variables.

Dorothy was curious about how L-BFGS approximates the expensive-to-store Hessian matrix. So, Jorge explained the two-loop recursion formula that he and L. S. Jennings had proposed in their 1980 publication. This formula computes and stores only a few vectors instead of the full Hessian matrix, which saves computational power and makes it feasible for high-dimensional problems. 

Dorothy was impressed and eager to try out L-BFGS in her spell optimization. She implemented L-BFGS in her code just as Jorge had taught her and ran it on the Rosenbrock function, a commonly used optimization benchmark.

```python
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der

# Define objective function (Rosenbrock function)
fun = rosen

# Define gradient function of objective function (Rosenbrock derivative)
jac = rosen_der

# Define initial point
x0 = [0, 0]

# Use L-BFGS algorithm for optimization
res = minimize(fun, x0, method='L-BFGS-B', jac=jac,
               options={'disp': True})
```

The code worked like a charm, and Dorothy was thrilled by the faster and more stable convergence of her spell optimization.

As Dorothy bid farewell to Jorge, he reminded her that L-BFGS wasn't perfect, and there were still limitations to its accuracy and convergence. But, with a better understanding of L-BFGS under her belt and guidance from experts like Jorge, Dorothy knew she could tackle even the most complex optimization problems with ease.
The code used in the Wizard of Oz parable to implement L-BFGS optimization is fairly straightforward and utilizes the `minimize` function from SciPy's optimization module.

```python
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der

# Define objective function (Rosenbrock function)
fun = rosen

# Define gradient function of objective function (Rosenbrock derivative)
jac = rosen_der

# Define initial point
x0 = [0, 0]

# Use L-BFGS algorithm for optimization
res = minimize(fun, x0, method='L-BFGS-B', jac=jac,
               options={'disp': True})
```

First, we import the necessary modules from SciPy's optimization module: `minimize`, which is the function that will perform the optimization, and `rosen` and `rosen_der`, which are the objective function and its derivative. 

Next, we define the objective function (`fun`) as the Rosenbrock function and its derivative (`jac`) as the Rosenbrock derivative. These are used to compute the gradient of the objective function during optimization.

Then, we define the initial point as `[0, 0]`, which is where the optimization routine will start.

Finally, we use the `minimize` function to perform the optimization using the L-BFGS algorithm (`'L-BFGS-B'` method), the objective function and its derivative, and additional options such as `'disp': True` to output the optimization details.

The `res` variable contains the result of the optimization, including the optimized parameters and the final function value.

This simple code demonstrates how straightforward it is to use the L-BFGS optimization method in Python, and how it can be utilized for a variety of optimization tasks.


[Next Chapter](15_Chapter15.md)