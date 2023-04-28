# Chapter 5: Momentum Based Gradient Descent

Welcome to the momentum based gradient descent chapter of our journey through the world of Machine Learning optimizers!

In the previous chapter, we discussed the popular Stochastic Gradient Descent (SGD) optimizer. We learned that SGD is a powerful algorithm that is widely used for optimizing various machine learning algorithms. We also discovered that SGD has some limitations, such as its slow convergence rate and unstable behavior near local minima.

In this chapter, we will learn about Momentum Based Gradient Descent, which is a variant of SGD. Momentum-based Gradient Descent attempts to solve the slow convergence rate problem of SGD by helping the optimizer to gain momentum in the relevant direction and dampening oscillations. This optimizer is especially useful when the objective function has high curvature, and the learning rate is relatively small. 

We will start this chapter by discussing the theory behind Momentum Based Gradient Descent. We will learn how it differs from Standard Gradient Descent and how it works to achieve faster convergence. 

So, buckle up and come along as we delve into another exciting topic in the world of optimization! 

Let's begin!
# The Wizard of Oz Parable: Momentum-Based Gradient Descent

Once upon a time, in a land far, far away, there was a young data scientist named Dorothy who lived in the small town of Kansas. She had a burning desire to become a master of optimization techniques in Machine Learning. She knew that the journey ahead was full of challenges and obstacles, but she was determined to learn and improve.

One day, Dorothy received an invitation to attend the "Optimization Techniques Workshop" hosted by "The Wizard of Oz." The Wizard was widely known for his mastery of all things optimization, and Dorothy knew that this was an opportunity she couldn't afford to miss. 

So, she packed her bags, put on her red shoes, and traveled to the wizard's castle. 

When she reached the castle, she was welcomed by the wizard himself. The wizard led her to a large room filled with many different types of optimizers, each with its unique characteristics.

Dorothy was naturally curious and asked the wizard, "Which optimizer is the best and will help me achieve faster convergence?"

The Wizard replied, "Momentum-Based Gradient Descent, my dear. It is one of the most efficient optimizers when it comes to converging quickly."

The Wizard then explained to Dorothy how Momentum-Based Gradient Descent works. 

"Momentum-Based Gradient Descent is like having a ball rolling down a hill. When the ball picks up momentum, it becomes increasingly challenging for it to be stopped, and it quickly reaches the valley. Similarly, momentum-based Gradient Descent accelerates the convergence towards the minimum loss function by gaining momentum in the relevant direction and dampening oscillations."

He then pulled out his magic wand and cast a spell, revealing the following pseudocode for Momentum-Based Gradient Descent.

```
v = 0
beta = 0.9
for t = 1, 2, ..., T:
    v = beta*v + (1-beta)*dL/dw
    w = w - alpha*v
```
Dorothy was awestruck at how simple the optimizer's magic seen to be. She then understood how Momentum-Based Gradient Descent could achieve faster convergence by taking advantage of the previous gradients.

From that day, she set out on a new journey of marvel and mastery, equipped with this knowledge. Along the way, she encountered many obstacles but always remembered the lesson from The Wizard of Oz: with the help of the right optimizer, she could achieve faster convergence and become a true master of Machine Learning.

And so, dear reader, always remember that Momentum-Based Gradient Descent is your friend on your journey to optimal convergence.
Sure, let me explain the pseudocode used to solve the Wizard of Oz parable. 

The pseudocode provided by the wizard for momentum-based gradient descent is as follows:

```python
v = 0
beta = 0.9
for t = 1, 2, ..., T:
    v = beta*v + (1-beta)*dL/dw
    w = w - alpha*v
```

Here, `v` is the velocity of the optimization process, `w` represents the weights that we are trying to learn, `beta` is a hyperparameter that controls the velocity of the optimization process, and `alpha` is the learning rate.

At each step of the optimization process, the velocity `v` is updated using a weighted average of the previous velocities and the current gradient `dL/dw`. The weight given to the previous velocity is controlled by the hyperparameter `beta`. When `beta` is high, the previous velocity is given more weight, and when `beta` is low, the current gradient is given more weight. This ensures that the optimizer gains momentum in the relevant direction and dampens oscillations. 

Finally, the weights are updated by subtracting the product of the learning rate `alpha` and the velocity `v` from the current weights `w`. This ensures that the optimization moves along the direction of the velocity with the step size controlled by the learning rate.

In summary, the pseudocode for momentum-based gradient descent provides an efficient way to converge quickly towards the minimum loss function by taking advantage of previous gradients. When used with the appropriate hyperparameters, this algorithm can significantly improve the training process of machine learning models.


[Next Chapter](06_Chapter06.md)