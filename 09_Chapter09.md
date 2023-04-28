# Chapter 9: Adaptive Moment Estimation (Adam)

Welcome back, my fellow Machine Learning enthusiasts! In the last chapter, we discussed RMSprop, a powerful optimization algorithm designed to tackle non-convex optimization problems. We learned how RMSprop leverages the concept of exponentially weighted moving averages to scale the learning curves of each parameter separately, resulting in faster convergence and a more stable training process. But, as always, there is always room for improvement. In this chapter, we will introduce you to a more advanced optimizer - Adam, which stands for Adaptive Moment Estimation.

Adam is a more sophisticated variant of RMSprop that takes into account the first and second moments of the gradients. It was introduced in a 2014 paper by Diederik Kingma and Jimmy Ba called "Adam: A Method for Stochastic Optimization" and quickly became one of the most widely used optimization algorithms in Machine Learning. In fact, a survey conducted by the ACM in 2021 found that Adam was the most popular optimizer used by Machine Learning practitioners. 

We are honored to have special guest Diederik Kingma, one of the co-authors of the original Adam paper, join us to explain Adam in further detail. Diederik is a researcher at OpenAI, co-founder of the e-commerce site Booking.com, and was named one of Forbes’ 30 under 30 in Europe in 2017 for his contributions to the field of Deep Learning. So, without further ado, let's dive into Adam! 

```python
#Adam Implementation in Python
class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad_wrt_w, 2)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))
        w = w - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w
```

In our implementation of Adam in Python, we instantiate the optimizer with a learning rate, and two exponential decay rates for the first and second moments of the gradients, respectively. We also set a small epsilon value to prevent division by zero. In the update method, Adam first computes the first and second moments of the gradients, and then adjusts them by dividing by their respective bias-corrected terms. Finally, it performs the weight update by scaling the gradients by the square root of the second moment estimate. 

That’s it for now, folks! We hope this introduction to Adam whets your appetite for the rest of the chapter. Stay tuned for more Machine Learning fun in the next sections.
# The Wonderful Wizard of Adam: A Machine Learning Parable

Once upon a time, there was a young Machine Learning researcher named Dorothy. She was curious and passionate about her work, but often found herself lost and confused when trying to navigate the complex world of optimization algorithms.

One day, while lost in thought, Dorothy stumbled upon a kind and knowledgeable wizard named Diederik Kingma. With a smile, Diederik welcomed Dorothy into his magical laboratory and offered to teach her the secrets of a powerful optimization algorithm he had created called Adam.

Excited to learn more, Dorothy eagerly followed Diederik as he explained to her how Adam worked. He began by telling her that Adam stood for Adaptive Moment Estimation and that it was essentially a more sophisticated version of the RMSprop optimizer she had learned about previously.

Diederik then pulled out a large chalkboard and began drawing out the math behind Adam. He explained that like RMSprop, Adam also computed an exponentially weighted moving average of the gradients. However, it also took into account two additional terms: the first moment, which is the exponential weighted average of the gradients, and the second moment, which is the exponential weighted average of the squared gradients. In other words, Adam was essentially keeping track of both the gradient and its variance.

Dorothy was amazed as she watched Diederik's chalkboard drawing come to life, and couldn't wait to see the effect of Adam for herself. Diederik then demonstrated how Adam adjusted the learning rate of each parameter during training and how it allowed for faster convergence and more stable optimization in non-convex settings.

Feeling confident in her understanding of this powerful optimizer, Dorothy asked Diederik if he could help her implement Adam on a deep neural network she was working on. He happily obliged and together they wrote code that encapsulated the essence of the algorithm.

```python
# Dorothy's Adam Implementation in Python
class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad_wrt_w, 2)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))
        w = w - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w
```

With Diederik's help, Dorothy was thrilled to see how fast and efficient their deep neural network was training using Adam.

As they wrapped up their work together, Diederik encouraged Dorothy to continue exploring different optimization algorithms and to never stop learning. With a warm smile and newfound confidence, Dorothy left the magical laboratory of Diederik Kingma, eager to return to her research and discover even more wonder and excitement in the world of Machine Learning.
Sure, let's dive into the code that was used to implement the Adam optimizer in the Wizard of Oz parable!

The implementation we used is written in Python, and extends the `Optimizer` class. It has four hyperparameters, which are commonly used in Adam implementations:

- `alpha`: The learning rate - this controls how much the optimizer adjusts the parameters based on the gradients.
- `beta1`: The exponential decay rate for the first moment of the gradient (i.e., the moving average of the gradient).
- `beta2`: The exponential decay rate for the second moment of the gradient (i.e., the moving average of the squared gradient).
- `epsilon`: A small value used to prevent division by zero when computing the weight update.

The Adam optimizer maintains two moving averages of the gradients. The first moment (i.e., moving average) of the gradient is calculated as an exponentially damped sum of past gradients, while the second moment is calculated similarly but for past squared gradients. 

The `update` method in the implementation of Adam then computes the bias-corrected first- and second- moment estimates, as well as the weight update for the current iteration.

Let's take a closer look at the code to see all of this in action:

```python
# Dorothy's Adam Implementation in Python
class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, w, grad_wrt_w):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad_wrt_w, 2)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))
        w = w - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w
```

In this code, the `Adam` class inherits from the `Optimizer` base class and overrides two of its methods: the `__init__` that initializes the hyperparameters, and the `update` method that updates the weights in each iteration of the optimization process.

In `__init__`, the values of the hyperparameters are specified, and the variables `m`, `v`, and `t` are initialized. The variables `m` and `v` are used to store the first and second moments of the gradients, while `t` is an iteration counter.

In `update`, we first check if we have computed the variables `m` and `v` before. If we haven't, we initialize them with zeros of the same shape as `w`, which are the weights to be updated. Then, we update `m` and `v` using the gradients `grad_wrt_w`, and compute their bias-corrected estimates `m_hat` and `v_hat`.

Finally, the weights are updated using the AdaDelta update rule.

This implementation of Adam is a simple yet effective way to harness the power of this optimizer. I hope this explanation helps you gain a deeper understanding of how Adam works and how to implement it in Python!


[Next Chapter](10_Chapter10.md)