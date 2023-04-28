# Chapter 3: Gradient Descent - The First Optimizer

Welcome back, learner! You have already gained an understanding of the different types of Machine Learning Optimizers in the last chapter. In this chapter, we will focus our attention on one particular optimizer - Gradient Descent, which is considered as the workhorse of optimization algorithms.

Gradient Descent is the first optimization algorithm that every beginner in Machine Learning usually encounters. This technique iteratively tweaks the parameters of a function in the direction of steepest descent of the cost function. It is named as "gradient descent" because we use gradient (slope) of the cost function to move towards the minimum cost.

Do you know Gradient Descent dates back to the early 1800s? A mathematician named Carl Friedrich Gauss invented it to help him find the orbit of the asteroid Ceres! Information on Gradient Descent dates back all the way to the Gauss-Newton method developed in the 1800s.

Understanding Gradient Descent is critical to comprehend more advanced optimization algorithms like Stochastic Gradient Descent, Newton's Method, and Adam. It forms the foundation of computational graphs, backpropagation, and neural networks. Isn't that fascinating?

To help us dive deeper into Gradient Descent, we have a special guest joining us, Yann LeCun! Yann LeCun is the Director of AI Research at Facebook, and he focuses mainly on deep learning, computer vision, and natural language processing.

Now, let's get started and learn how Gradient Descent works!
# Chapter 3: Gradient Descent - The First Optimizer: A Parable

Once upon a time, in the land of Machine Learning, lived Dorothy, an enthusiastic young data scientist. She was on a quest to master the art of optimization algorithms. One day, she set out on a journey to the Emerald City to learn from the greatest of them all - the Wizard of Optimization.

On her way, Dorothy came across a scarecrow who seemed to be in great distress. "What's wrong?" asked Dorothy. The Scarecrow replied, "I am just a dummy with no intelligence or knowledge. I want to learn how to optimize my brain to think bigger and better."

Dorothy remembered how she learned about Gradient Descent in her previous chapter and thought it could help the Scarecrow. "Don't worry, Scarecrow," she said, "I know a technique called Gradient Descent that will help you optimize your thinking process."

With new hope, the Scarecrow and Dorothy set out on their journey to meet the Wizard of Optimization. Along the way, they met another traveler, Yann LeCun, who offered to join them on their adventure. Yann was an expert in optimization algorithms, specifically deep learning, and had a lot of knowledge to share.

As they journeyed together, Dorothy taught the Scarecrow about Gradient Descent. She explained how it works by taking small steps in the direction of the steepest descent of the cost function, and how it can help optimize his thinking process. Yann then chimed in, "Gradient Descent is a simple yet powerful optimization algorithm. It forms the basis of neural networks and deep learning."

With the help of Gradient Descent, the Scarecrow began to optimize his thinking process, and he started to think bigger and better thoughts. When they finally reached the Emerald City, the Wizard of Optimization was amazed at the Scarecrow's newfound ability, and he granted Dorothy and her friends with a magical reward.

From that day forward, Dorothy became known as the Master of Gradient Descent, and the Scarecrow continued to optimize his brain using this technique. And whenever they needed help, they knew they could count on their friend Yann LeCun, the expert of optimization algorithms.

The End.
To resolve the Wizard of Oz parable, we used Gradient Descent, one of the most fundamental optimization algorithms used in Machine Learning. 

Gradient Descent minimizes the cost function by iteratively adjusting the parameters towards the optimum. In the case of the story, the Scarecrow wanted to improve his brain's performance. The Scarecrow's brain can be thought of as a function, and the parameters can be considered as the connections between the neurons in the brain.

We can optimize the Scarecrow's brain network by iteratively adjusting the parameters towards the optimum using Gradient Descent.

```
# Pseudo Code for Gradient Descent Algorithm
# Let `f` be the function that we want to minimize
# Let `lr` be the learning rate
# Let `iters` be the total number of iterations

def gradient_descent(f, init_params, lr, iters):
    params = init_params
    for i in range(iters):
        grad_params = compute_gradient(f, params)
        params = params - lr * grad_params
    return params
 ```
 
In the code, `gradient_descent` takes in the function `f` that we want to optimize, the initial set of parameters `init_params`, the learning rate `lr`, and the total number of iterations `iters`. 

At each iteration, we compute the gradient of the function `f` with respect to the parameters using `compute_gradient`, and adjust the parameters in the direction of the negative gradient (i.e., the direction where the function is decreasing the most) by multiplying the gradient with the learning rate `lr`. We do this until we reach the minimum possible value of the function.

By optimizing with Gradient Descent, the Scarecrow was able to optimize his brain and start thinking more effectively, just like how gradient descent is used to optimize the weights in a neural network.

Hence, Gradient Descent forms the foundation of a myriad of optimization algorithm and is a must-know technique for any data scientist or machine learning engineer.


[Next Chapter](04_Chapter04.md)