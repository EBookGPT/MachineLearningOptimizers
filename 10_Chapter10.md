# Chapter 10: AdaDelta - The Unpredictable Sorcerer

Welcome to the tenth chapter of our book on Machine Learning Optimizers. In the previous chapter (Chapter 9), we learned about the Adaptive Moment Estimation (Adam) optimizer, which achieved state-of-the-art performance on many deep learning tasks. 

In this chapter, we will explore another popular optimizer called AdaDelta (Adaptive Delta). AdaDelta is a derivative-free, adaptive learning rate optimizer that aims to mitigate the need for manual tuning of hyperparameters. It was introduced in the 2012 paper, "ADADELTA: An Adaptive Learning Rate Method" by Matthew Zeiler. 

The name AdaDelta stands for "adaptive delta", which refers to the use of delta updates in its update rule. The optimizer dynamically adjusts the learning rate for each parameter based on the moving average of its past gradient updates. AdaDelta uses two exponential moving averages: 

- **E[g^2]**: The running average of the square of past gradients.
- **E[∆x^2]**: The running average of the square of past parameter updates.

Using these two averages, AdaDelta computes the root mean squared (RMS) error of the gradient (E[g^2]_t) and the RMS error of the parameter updates (E[∆x^2]_t). It then computes the scaled gradient by dividing the current gradient by the RMS error of the past gradients. Finally, it updates the parameters using the scaled gradient and the RMS error of the past parameter updates. 

AdaDelta is known for its robustness to the choice of hyperparameters, making it a popular optimizer in deep learning. Additionally, it has been shown to outperform other adaptive learning rate methods, such as AdaGrad, on certain tasks. However, despite its success, the mechanism behind its effective performance is not yet fully understood. 

Are you intrigued by the unpredictable sorcery of AdaDelta? Let's dive in and learn how to implement it in code.
# AdaDelta - The Unpredictable Sorcerer

Once upon a time in the land of Machine Learning, there was a young Wizard named Dorothy. Dorothy was a curious Wizard who loved to experiment with new magic spells. One day, she stumbled upon a mysterious wand called AdaDelta. The wand was unlike any other wand she had seen before. 

As she started to study the magic behind AdaDelta, the wand revealed its mysteries to her. "I am AdaDelta, the unpredictable sorcerer," the wand said. "My magic lies in adapting the learning rate of your neural network to mitigate the need for manual tuning of hyperparameters. I will help you achieve the best performance on deep learning tasks with my adaptive learning rate method."

Dorothy was intrigued by the wand's promise and decided to put it to the test. She went into her laboratory and started to implement AdaDelta's magic in code. She defined the two exponential moving averages, E[g^2] and E[∆x^2], and used them to compute the RMS errors of past gradients and parameter updates. 

She added magic to scale the gradients by dividing the current gradient by the RMS error of past gradients. Finally, Dorothy used AdaDelta's magic to update the parameters using the scaled gradient and the RMS error of past parameter updates.

As she ran her neural network with AdaDelta, she was amazed to see how well her model was performing. The magic of AdaDelta had indeed helped her achieve state-of-the-art performance on deep learning tasks, without having to manually tune the learning rate. 

But the young Wizard was left puzzled by the unpredictability of AdaDelta's sorcery. "Why do you work so well, AdaDelta? What is the secret behind your magic?" Dorothy asked the wand.

AdaDelta smiled and replied, "My youthful Wizard, the mechanism behind my effective performance is still a mystery. Researchers are still trying to understand how and why I work so well. But one thing is for sure, my magic lies in my ability to adapt the learning rate in a way that is unparalleled by other optimizers."

And so, Dorothy continued to use the magic of AdaDelta in her deep learning experiments, marveling at its unpredictable and effective sorcery.
# Implementation of AdaDelta in Code

Implementing AdaDelta optimizer in code is fairly straightforward. Here's how Dorothy did it in her laboratory:

```python
import torch.optim as optim

learning_rate = 1.0

# Initialize the optimizer
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute the loss
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update the parameters
        optimizer.step()
``` 

In the above code snippet, `optim.Adadelta` is used to initialize the optimizer with a specified learning rate. The `model.parameters()` argument is passed to ensure that all model parameters are optimized.

During training, the optimizer is used to handle the gradients of the loss function with respect to the model parameters by calling `optimizer.zero_grad()` to clear the gradients, `loss.backward()` to compute the gradients, and `optimizer.step()` to update the parameters based on the computed gradients.

By using AdaDelta in her neural network training, Dorothy was able to automatically adapt the learning rate without the need for manual tuning of hyperparameters. The unpredictable sorcery of AdaDelta had helped her achieve great performance on deep learning tasks.


[Next Chapter](11_Chapter11.md)