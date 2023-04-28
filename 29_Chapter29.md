# Chapter 29: The Armijo-Goldstein Condition

Welcome back! In the previous chapter, we learned about backtracking line search, a method for finding the appropriate step size in gradient descent. Continuing on this journey through machine learning optimizers, we now turn our attention to the Armijo-Goldstein Condition.

The Armijo-Goldstein Condition is a line search method that is similar to backtracking line search. While backtracking line search shrinks the step size by a fixed factor when the current step size does not result in sufficient decrease in the objective function, Armijo-Goldstein adds an additional condition to ensure that the decrease is "sufficient enough". 

The condition in question is a lower bound on the amount of decrease we require from our objective function. In other words, it specifies the minimum amount of improvement in order for us to accept a new point as the next iteration of our algorithm. The condition is as follows:

```math
f(x_k + \alpha_k p_k) \leq f(x_k) + c_1\alpha_k\nabla f(x_k)^Tp_k
```
where `f` is our objective function, `p_k` is our search direction, `c_1` is a constant between 0 and 1, and `alpha_k` is our step size. 

The Armijo-Goldstein Condition is used as a safeguard to ensure that the search stays in the general direction of the minimum. It does this by looking at the slope of the function at the current point and "pushing" the search direction in the opposite direction. If the search direction ends up pointing to a point that is lower than our current value, we take that as the next step. If not, we shrink our step size and iterate again.

Intuitively, the Armijo-Goldstein Condition tries to balance exploration (taking a larger step) and exploitation (taking small steps to ensure safe exploration). With each iteration, we "explore" the search space to see if we can find a new promising point, while the condition helps us to make sure that our exploration doesn't accidentally lead us to a worse point.

In practice, the Armijo-Goldstein Condition is an excellent method for optimizing nonconvex objective functions, and it is a cornerstone of many optimizer algorithms such as the L-BFGS algorithm. Indeed, in several published journals, the Armijo-Goldstein Condition is cited as a successful approach for optimizing challenging nonconvex optimization problems.

With that, we will now look at how the Armijo-Goldstein Condition works in code!
# The Wizard of Oz and the Search for the Emerald City

Once again, we turn to the land of Oz for inspiration on our journey through machine learning optimizers. Our protagonist, Dorothy, sets out on a quest to find the fabled Emerald City. But in order to get there, she must navigate a treacherous forest full of twists and turns.

At each turn, she encounters a yellow brick road with an uphill slope. Being brave and determined, she sets off on the road, but soon finds herself exhausted with her progress dwindling. But then, she meets the Wizard of Oz, who offers to help her find the right path to the Emerald City.

The Wizard gives her a staff with a magical property—each time she takes a step in a certain direction, the staff can tell her if the path will bring her closer or farther to the Emerald City. The Wizard also tells Dorothy that there is a condition she must meet in order to find the path to the Emerald City.

"The condition," the Wizard explains, "is that you must ensure that for each step you take, you must be going downhill, at least a little bit. But you don't want to take too big of a step or you might fall off a cliff. So, take small steps, but not too small, and make sure that each step you take, the road has a slope downwards. If it does, then keep on taking steps, until you reach the Emerald City."

Dorothy heeds the Wizard’s advice and starts moving along the yellow brick road. She takes the step with the staff, and if the staff doesn't tell her that she's going downhill, she takes a smaller step. If the staff tells her she is making progress, she takes the next step in that direction with a slightly larger step size. 

With each step, Dorothy is able to traverse the forest in search of the Emerald City. The staff helps her evaluate her progress, and the condition ensures that she makes safe and steady progress towards her goal.

The Armijo-Goldstein Condition is analogous to the condition set by the Wizard of Oz. It helps us move downhill and make progress towards the minimum of the objective function in a safe and steady manner. The magic staff is analogous to the gradient of the objective function, which helps us determine the direction of the slope. And Dorothy represents our algorithm, taking small and cautious steps towards the optimum with the help of the Armijo-Goldstein Condition.

Let's see how implementing the Armijo-Goldstein Condition in code can help us find the path to our own "Emerald City" of optimization!
# Code Implementation of the Armijo-Goldstein Condition

We will now demonstrate how to implement the Armijo-Goldstein Condition in Python code. First, we define a function to evaluate the objective function and another to calculate its gradient. These functions will be used to determine if we are meeting the condition.

```python

def obj_func(x):
    """Objective function to minimize."""
    return (x ** 2) + (2 * x) + 1

def gradient(x):
    """Gradient of the objective function."""
    return (2 * x) + 2
 
```

Next, we define a function to implement the Armijo-Goldstein Condition. The function will take as input our current position `x`, the search direction `p`, the current objective value `f_x`, and the current gradient `grad`. It returns the step size that satisfies both the Armijo-Goldstein Condition and a specified tolerance.

```python

def armijo_goldstein(x, p, f_x, grad, c1=0.1, alpha=1, tolerance=1e-5):
    """Find the step size that satisfies the Armijo-Goldstein condition."""
    while obj_func(x + alpha * p) > obj_func(x) + c1 * alpha * grad.T @ p:
        alpha *= 0.5
    return alpha
 
```

The while loop in this function performs the Armijo-Goldstein Condition check. We first compute the next candidate point given by the step size `alpha` using the search direction `p`. We then check if `f(x_k + alpha_k * p_k)` is less than or equal to `f(x_k) + c_1 * alpha_k * grad.T @ p_k`, where `f(x_k)` is the current objective value and `grad.T @ p_k` is the directional derivative, `c_1` is a decay parameter, `0 < c1 < 1`. If this inequality holds true, we have met the Armijo-Goldstein condition and can return the step size.

Finally, we can run our optimization algorithm using the Armijo-Goldstein Condition as our line search method. We can use the following function to perform gradient descent, using the Armijo-Goldstein Condition to update the step size at each iteration.

```python

def gradient_descent(x_init, learning_rate, num_iters):
    """Perform gradient descent with Armijo-Goldstein line search."""
    x = x_init
    for i in range(num_iters):
        grad = gradient(x)
        p = -grad
        alpha = armijo_goldstein(x, p, obj_func(x), grad)
        x = x + alpha * p
    return x

```
In this function, we set `x_init` to an initial starting value for the objective function `obj_func`, `learning_rate` to a specified learning rate, and `num_iters` as the number of iterations to perform. In each iteration, we compute the gradient at the current point `x` and update our position by taking a step in the opposite direction of the gradient by an amount that satisfies the Armijo-Goldstein Condition.

There you have it! With these Python functions, we can now implement the Armijo-Goldstein Condition in our optimization algorithms, and use it to help us make safe and steady progress towards finding our own "Emerald City" of optimization.


[Next Chapter](30_Chapter30.md)