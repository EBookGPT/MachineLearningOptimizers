# Chapter 13: Conjugate Gradient Method

Welcome back, my fellow learners! In our last chapter, we explored Krylov Subspace Methods, which are commonly used iterative methods in solving linear systems of equations in machine learning techniques. 

In Chapter 13, we will investigate another widely used optimization algorithm called Conjugate Gradient Method (CG). This algorithm is an iterative method that works particularly well for solving large, symmetric positive definite sets of equations. 

To help us gain a deeper understanding of the concept, we have a special guest who will be joining us - Shai Shalev-Shwartz! Shai is a professor in the School of Computer Science and Engineering at the Hebrew University of Jerusalem. He is also the co-author of the books 'Understanding Machine Learning: From Theory to Algorithms' and 'Online Learning and Online Convex Optimization,' both published by Cambridge University Press. Let's hear from him as he shares some insights about CG.

---

### Shai Shalev-Shwartz on Conjugate Gradient Method

"Conjugate Gradient Method in its modern form was originally presented by Magnus Hestenes and Eduard Stiefel. The basic idea behind CG is to iteratively solve a set of equations by minimizing the error between the current solution and the ideal solution. This is done by using conjugate directions to iteratively update the current solution estimate. 

In machine learning applications, CG is commonly used to optimize quadratic functions, with many useful applications such as Principal Component Analysis (PCA) and Support Vector Machines (SVM). Moreover, CG can be used for non-linear optimization as well (by applying CG to the gradient, a.k.a. Newton-CG), as well as for solving linear systems of equations (using the preconditioned CG variant).

It's important to note that CG has advantages over other optimization methods not only in terms of convergence rates and efficiency but also in terms of simplicity and ease of implementation. Moreover, recent advances in optimization have presented extensions to the classic CG, like the CG-Steihaug method for constrained optimization problems, and its brother, the Lanczos method. Both are widely used in modern second-order optimization algorithms."

---

In the next sections, we'll dive into the details of the CG method and learn how to implement it to solve optimization problems. Let's get started! 

## The Algorithm

The Conjugate Gradient Method is a popular iterative algorithm for optimization that can be described as follows:

1. Initialize the estimate of the solution x0.
2. Compute the first residual r0 = b - Ax0, where b is the known column vector, A is known as the positive-definite symmetric matrix, and x0 is the initial guess.
3. Compute the search direction vector d0 = r0.
4. For k = 0, 1, 2,... until convergence is achieved (or a maximum number of iterations is reached):
   - Compute alpha_k = (r_k^T r_k) / (d_k^T A d_k), where r_k is the residual vector and A is the known positive-definite symmetric matrix.
   - Compute the solution estimate x_k+1 = x_k + alpha_k d_k.
   - Compute r_k+1 = r_k - alpha_k A d_k.
   - Compute beta_k = (r_k+1^T r_k+1) / (r_k^T r_k), where r_k, r_k+1 are the residual vectors.
   - Compute the new search direction d_k+1 = r_k+1 + beta_k d_k.

Note that the convergence of CG algorithm depends on the condition number of the matrix A. In practice, the maximum number of iterations is often set empirically, with typical values ranging between 100 and 1000.

## Implementing Conjugate Gradient Method in Python

Here's an example of an implementation in Python:

```python
import numpy as np

def conjugate_gradient(A, b, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    r = b - A @ x
    d = r.copy()
    for k in range(max_iter):
        alpha = (r @ r) / (d @ A @ d)
        x += alpha * d
        r_kp1 = r - alpha * A @ d
        if np.linalg.norm(r_kp1) < tol: # check convergence
            break
        beta = (r_kp1 @ r_kp1) / (r @ r)
        d = r_kp1 + beta * d
        r = r_kp1
    return x
```

As you can see, the code is quite simple and easy to understand. We start with an initial guess of the solution (x0), compute the first residual (r0), and then start the loop for updating the current solution estimate iteratively. We follow the algorithm outlined in the previous section.

## Conclusion

In summary, the Conjugate Gradient Method is a popular optimization algorithm that works particularly well for solving large, symmetric positive definite sets of equations. The algorithm is not only efficient and fast but also simple to implement. By using this algorithm, we can solve optimization problems related to Principal Component Analysis, Support Vector Machines, and other machine learning applications. With the knowledge gained in this chapter, you will be better equipped to implement CG in your own optimization problems.
# Chapter 13: Conjugate Gradient Method - The Wizard of Oz

Once Dorothy landed in the land of Oz, she found herself in a peculiar situation. The Wicked Witch of the West had trapped her in a maze that she needed to solve to leave. To her aid came a special guest, Shai Shalev-Shwartz, who was known as the Conjugate Gradient Wizard. 

Dorothy's dilemma was how to solve the maze in the most optimal way. Shai suggested that she use the Conjugate Gradient algorithm to solve this problem. He explained to her that the Conjugate Gradient Method is an efficient optimization algorithm that iteratively solves equations of the form Ax=b. This algorithm works particularly well when dealing with large, symmetric positive-definite matrices, which we might encounter in our journey through machine learning.

To teach Dorothy, Shai started by transforming the maze problem into an optimization problem. The objective function that we need to minimize is the distance from Dorothy's current position to the exit of the maze. Dorothy's location in the maze can be represented by a vector x, and the distance to the exit is a quadratic function of x. Once we have the objective function, we can apply the Conjugate Gradient algorithm to solve for the optimal path to the exit.

Shai then went on to explain the steps involved in the Conjugate Gradient algorithm to Dorothy. Each iteration of the algorithm has two main steps: first, we calculate the search direction based on the previous search direction and residue, and second, we update the current solution estimate to bring us closer to the optimal solution. Shai highlighted that conjugacy is the key to the algorithm's efficiency. The search directions are carefully chosen to be mutually conjugate, which means that they are all orthogonal with respect to the matrix A. This orthogonality allows us to find the optimal solution with fewer iterations than other optimization algorithms.

Dorothy was thrilled to learn about the Conjugate Gradient algorithm, and she was eager to try it out in the maze. She quickly wrote a Python program to implement the algorithm and started running it. To her delight, the algorithm was able to find the shortest path to the exit in just a few iterations. She was amazed at how efficient the algorithm is and thanked Shai for teaching her this powerful technique.

With the help of the Conjugate Gradient Wizard, Dorothy was able to solve the maze and continue on her journey to find her way back home. She knew that she could always rely on the Conjugate Gradient algorithm to help her solve even the most challenging optimization problems she might encounter.

With this parable, we conclude our discussion on Conjugate Gradient Method in Machine Learning Optimizers. We hope that you found it educational and entertaining. Go forth and use the Conjugate Gradient algorithm to tackle your optimization problems with confidence!
Certainly! In the Wizard of Oz parable, we used the Conjugate Gradient algorithm to solve the maze problem that Dorothy was facing. Here's a brief explanation of the Python code used to implement the algorithm:

```python
import numpy as np

def conjugate_gradient(A, b, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    r = b - A @ x
    d = r.copy()
    for k in range(max_iter):
        alpha = (r @ r) / (d @ A @ d)
        x += alpha * d
        r_kp1 = r - alpha * A @ d
        if np.linalg.norm(r_kp1) < tol: # check convergence
            break
        beta = (r_kp1 @ r_kp1) / (r @ r)
        d = r_kp1 + beta * d
        r = r_kp1
    return x
```

Let's go over each line of the code:

```python
import numpy as np
```
This line imports the NumPy library, a popular library for scientific computing in Python. We will use NumPy to do matrix vector operations and some other array operations.


```python
def conjugate_gradient(A, b, x0, max_iter=1000, tol=1e-6):
```

This line starts the definition of the `conjugate_gradient` function which takes in four arguments:
* `A`: a square, symmetric positive-definite matrix `A` that defines the optimization problem.
* `b`: a column vector `b`, which represents the right-hand side of the equation.
* `x0`: an initial guess for the solution vector `x`.
* `max_iter`: an optional argument that specifies the maximum number of iterations to perform. This is to avoid infinite loops that may occur if the algorithm fails to converge.
* `tol`: an optional argument for the convergence tolerance, which is the desired level of accuracy that we want to achieve.

```python
    x = x0.copy()
```

This line initializes the solution vector by making a copy of the initial guess.

```python
    r = b - A @ x
    d = r.copy()
```

These lines calculate the initial residual vector `r` and set the initial search direction vector `d`.

```python
    for k in range(max_iter):
```

This line starts the loop for updating the solution estimate iteratively. We use a `for` loop, which iterates from 0 to `max_iter`.

```python
        alpha = (r @ r) / (d @ A @ d)
```

This line calculates the step size `alpha` that we take in the direction of the search vector `d`.

```python
        x += alpha * d
```

This line updates the current solution estimate `x` by taking a step of size `alpha` in the direction of the search vector `d`.

```python
        r_kp1 = r - alpha * A @ d
```

These lines calculate the new residual vector `r_kp1` and check for convergence.

```python
        if np.linalg.norm(r_kp1) < tol: # check convergence
            break
```

If the norm of the residual `r_kp1` is less than the specified tolerance `tol`, we break out of the loop.

```python
        beta = (r_kp1 @ r_kp1) / (r @ r)
```

This line calculates the scaling factor `beta` that we need to apply to the search vector `d` to obtain a new conjugate direction.

```python
        d = r_kp1 + beta * d
        r = r_kp1
```

These lines update the search direction vector `d` and residual vector `r` for the next iteration. 

```python
    return x
```

Finally, this line returns the solution vector `x` as the output of the function.

That's a brief overview of the Python code used for the Conjugate Gradient algorithm. We hope this helps you understand how the algorithm works and how to implement it in Python!


[Next Chapter](14_Chapter14.md)