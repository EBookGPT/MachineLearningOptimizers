# Chapter 6: Nesterov Accelerated Gradient (NAG)

Greetings reader, welcome back to our journey through the world of machine learning optimizers. In the previous chapter, we learned about the Momentum Based Gradient Descent optimizer, which helps the gradient descent method converge faster towards the minimum. 

Today we will be discussing the Nesterov Accelerated Gradient (NAG) optimizer, named after special guest Yurii Nesterov who proposed the algorithm in his paper 'A method of solving a convex programming problem with convergence rate O(1/k2)' in 1983. Nesterov is a renowned mathematician and computer scientist who has made significant contribution to the field of optimization.

Nesterov proposed a modification to the original Momentum Based Gradient Descent algorithm which takes into account the future gradient. Unlike Momentum Based Gradient Descent which first computes the gradient and then take the gradient's direction, NAG takes a peek at what the gradient will be on the next step, and use that to correct the momentum.

Nesterov Accelerated Gradient (NAG) is widely used in modern deep learning training pipelines, particularly when training with large datasets. It improves upon Momentum Based Gradient Descent algorithm by sharpening the optimal step thus allowing faster convergence to the optimum.

Let's take a closer look at how Nesterov Accelerated Gradient (NAG) works and use code samples to illustrate its implementation.
# The Wizard of NAG

Once upon a time, there was a young machine learning enthusiast named Dorothy who wished to become a master of the art. She had already traversed through the deadly fields of high dimensional data sets and had successfully passed through the swamp of overfitting. However, the path ahead was still uncertain and she needed guidance.

One day, she stumbled upon a wise old man named Yurii. Recognizing him as a renowned mathematician and computer scientist, Dorothy was elated to have found such a wise leader in her journey.

Yurii saw the eagerness to learn in Dorothy's eyes and hence, he agreed to share with her a powerful technique he had developed - The Nesterov Accelerated Gradient (NAG) optimizer.

To help Dorothy understand NAG, Yurii shared a story from his past. He explained to her that NAG works like a person going downhill while trying to take corners without slowing down too much. Instead of anticipating the next corner from a distance, the driver goes a little ahead towards the supposed corner, slows down, takes the turn and speeds up on the other side. Similarly, in NAG, instead of calculating the gradient at t and taking a step towards that direction, one anticipates the gradient at t+1, takes a step towards that direction to slow down and then calculate the gradient and take a step towards that direction. 

Dorothy was amazed and inspired by Yurii's words, and she quickly grasped the technique. She implemented NAG in her machine learning model, and it helped the model reach convergence faster, ultimately improving its performance.

From that day on, Dorothy continued her journey ahead and became one of the greatest machine learning experts - thanks to her mentor and the powerful technique he shared.

Thus, the Wizard of NAG helped Dorothy unlock a new realm of possibilities in the world of machine learning.
To implement Nesterov Accelerated Gradient (NAG), we first need to initialize the required hyperparameters `learning_rate`, `momentum_factor` and `prev_delta_theta`. Then, we can use the following code:

```
learning_rate = 0.01
momentum_factor = 0.9
prev_delta_theta = 0

for i in range(num_iterations):
    # calculate the next gradient at the predicted position
    next_theta = theta - momentum_factor * prev_delta_theta
    next_gradient = calculate_gradient(next_theta)
    
    # apply the correction to the momentum direction
    delta_theta = momentum_factor * prev_delta_theta + learning_rate * next_gradient
    theta = theta - delta_theta
    
    prev_delta_theta = delta_theta
```

Here, we first calculate the predicted position of `theta` by subtracting `momentum_factor * prev_delta_theta` from `theta`. Then, we calculate the gradient at the predicted position `next_gradient`. Next, we use this gradient to compute the new optimal direction `delta_theta` by applying a corrective term to the momentum direction. Finally, we update the current position of `theta` by subtracting `delta_theta`.

The process is repeated for a fixed number of iterations, iterating through a loop. At each iteration, we update `prev_delta_theta` to hold the last direction used for correcting the momentum.

In practice, the implementation of NAG can be used to optimize various differentiable objectives in Machine Learning. Its efficacy in high dimensional datasets and shallow neural networks is remarkable, as it has been shown to outperform the traditional Momentum Based Gradient Descent algorithm in reducing errors and improving convergence.


[Next Chapter](07_Chapter07.md)