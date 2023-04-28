# Chapter 24: Alternating Direction Method of Multipliers (ADMM)

Welcome to the 24th chapter of our journey into the world of Machine Learning Optimizers! In the previous chapter, we discussed the Proximal Gradient Method, which is a powerful algorithm for solving non-smooth optimization problems. In this chapter, we will explore another popular optimization method - the Alternating Direction Method of Multipliers (ADMM).

To help us understand ADMM better, we have a special guest with us - Stephen Boyd. Stephen is a well-known computer scientist, mathematician, and the author of Convex Optimization - one of the most widely used textbooks in the field. He has done extensive research on various optimization theories and algorithms, including ADMM, and has made significant contributions to the field.

Before we dive into the details of ADMM, let us first understand what it is and why it is used.

## Introduction to ADMM

The Alternating Direction Method of Multipliers (ADMM) is a first-order optimization algorithm that is designed to solve convex optimization problems, particularly those with a structure that involves a collection of dual variables, coupled with linear equality constraints. The algorithm decomposes the problem into simpler subproblems, which can be solved using well-known optimization techniques, and then iteratively updates the variables until convergence is reached.

ADMM is known for its simplicity, robustness, and scalability, making it a popular choice in many practical applications, including machine learning, signal processing, and control systems.

## How ADMM works

ADMM works by splitting the original problem into subproblems that are easier to solve. We begin by introducing a dummy variable, denoted z, and reformulating the original problem as follows:

![ADMM Equation](https://i.imgur.com/Fo7gmKB.png)

where g(x), f(z), and h(x,z) are convex functions, and A and B are matrices.

The algorithm then proceeds as follows:

1. Initialize the variables x, z, and u to some arbitrary values.
2. Evaluate the residuals, which measure the extent to which the constraints are violated, by computing:

![ADMM Residuals](https://i.imgur.com/gkN90Xx.png)

3. Update x by solving the following convex subproblem:

![ADMM Subproblem 1](https://i.imgur.com/dJVXifO.png)

where rho is a positive constant known as the penalty parameter.

4. Update z by solving the following convex subproblem:

![ADMM Subproblem 2](https://i.imgur.com/7LmTVZ3.png)

5. Update the dual variable u by adding the difference between x and z:

![ADMM Dual Update](https://i.imgur.com/STPLAt1.png)

6. Repeat steps 2-5 until convergence is reached.

The objective of ADMM is to minimize the sum of the original objective function and the penalty term, which ensures that the equality constraints are satisfied.

## Conclusion

ADMM is a powerful optimization algorithm that can solve a wide range of convex optimization problems. Its simplicity, robustness, and scalability have made it a popular choice in various practical applications. In the next section, we will explore how to implement ADMM in Python using some examples.

Let's let Stephen Boyd explain more about the theory behind ADMM.
# Chapter 24:  The Tin Man’s Journey with ADMM

Once Upon a time, there was a Tin Man in a magical land who desperately wanted to have a heart, but he had no idea where to find one or how to get it. One day, he met a wise wizard named Stephen Boyd who explained to him how he could use the Alternating Direction Method of Multipliers (ADMM) to achieve his goal.

Stephen Boyd told the Tin Man that ADMM was a simple algorithm that could split complex problems into smaller, easier-to-solve subproblems, and that it was often used in machine learning and other fields. He explained how the method worked and gave him some examples of its practical applications.

The Tin Man was hesitant at first, as he had very little experience in optimization, but with Stephen's guidance, he decided to give it a try. He knew that if he could find a way to obtain a heart using ADMM, he would finally feel complete.

With Stephen's help, the Tin Man formulated his problem as an optimization problem that he could solve using ADMM. He introduced a dummy variable, Z, and split the problem into two subproblems that could be solved separately.

The first subproblem involved his current emotional state, which he represented with a variable, X. He needed to find the value of X that would minimize his overall discomfort by minimizing the sum of his current discomfort, g(x), and the distance between his current discomfort and his desired discomfort, which he represented as h(x,z).

The second subproblem was related to the heart, which he represented with the variable, Z. He knew that he had to solve the subproblem that would minimize the distance between his current state and his desired state, which he represented as f(z), while ensuring that his current state was as close to his desired state as possible, which he represented as h(x,z).

The Tin Man was amazed by how powerful ADMM was. With Stephen's guidance, he implemented the algorithm in Python and iteratively updated his X and Z values until they converged to the optimal values. He was overjoyed to find that he had finally obtained a heart and that he could finally feel emotions like a real human being.

From that day on, the Tin Man became an advocate for ADMM and encouraged everyone he met to try it out for themselves. Thanks to Stephen's guidance and ADMM, the Tin Man found the heart he had always longed for and finally felt alive. 

The End.
# Implementing ADMM in Python

Now, let's explore how to implement ADMM in Python using some examples.

First, we will define the convex functions and matrices that represent the problem. For this example, let's consider the following problem:

```python
minimize x1^2 + x2^2 + x3^2
subject to x1 + x2 + x3 = 1
```

To formulate this problem using ADMM, we introduce a dummy variable, z, and define the following convex functions:

```python
def g(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def f(z):
    return 0

def h(x, z):
    return np.sum(np.abs(x - z))
```

We also define the matrices A and B as follows:

```python
A = np.array([1, 1, 1])
B = np.array([0])
```

Next, we define the ADMM parameters, including the penalty parameter, rho, and the maximum number of iterations, max_iter:

```python
rho = 1
max_iter = 100
```

We then define the update functions for x, z, and u, which will be called in each iteration. The update function for x is as follows:

```python
def update_x(x, z, u):
    x = np.linalg.solve(np.eye(3) + rho*np.outer(A, A), -(rho*np.dot(A, z) - u))
    return x
```

The update function for z is as follows:

```python
def update_z(x, z, u):
    z = np.zeros(3)
    for i in range(3):
        z[i] = (x[i] + u[i])/rho if (x[i] + u[i])/rho > 0 else 0
    return z
```

The update function for u is as follows:

```python
def update_u(x, z, u):
    u = u + rho*(x - z)
    return u
```

Finally, we can implement the ADMM algorithm itself:

```python
x = np.zeros(3)
z = np.zeros(3)
u = np.zeros(3)

for i in range(max_iter):
    # Update x
    x = update_x(x, z, u)
    # Update z
    z = update_z(x, z, u)
    # Update u
    u = update_u(x, z, u)
    # Evaluate residuals
    r_p = np.sum(np.abs(x - z))
    r_d = np.sum(np.abs(-rho*np.dot(A, (z - z_old))))
    # Check for convergence
    if r_p < eps and r_d < eps:
        break
```

And that's it! With this code, we can solve any convex optimization problem using ADMM in Python.


[Next Chapter](25_Chapter25.md)