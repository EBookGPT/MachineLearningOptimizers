# Chapter 15: Nonlinear Conjugate Gradient (NCG)

Welcome back, my fellow Machine Learning enthusiasts!
In the previous chapter, we delved deep into the Limited Memory Broyden-Fletcher Goldfarb Shanno (L-BFGS) optimizer. We hope you enjoyed learning about its working principle and how it can help you optimize your models.

But that's not all, as you know that the world of Machine Learning is vast and ever-evolving. So, are you ready to take your optimization game to the next level? In this chapter, we will introduce you to Nonlinear Conjugate Gradient (NCG), another powerful optimizer that is commonly used in Machine Learning.

But before we explore NCG, allow me to introduce our special guest for this chapter - Yves Nesterov! He is a renowned computer scientist and mathematician who has made significant contributions to optimization theory. He developed the accelerated gradient descent method, which is widely used in Machine Learning. We are fortunate to have him share his insights into the principles of optimization.

Let's get started and dive into the world of Nonlinear Conjugate Gradient.

## The Principle of NCG

The NCG is a gradient descent algorithm that is widely used in Machine Learning. The principle of NCG is to find the minimum of a non-linear function using conjugate gradient directions [1]. The conjugate gradient is a well-known method used to solve linear systems, but it can also be used to solve non-linear optimization problems.

Like L-BFGS, NCG also belongs to the family of quasi-Newton methods. However, NCG computes a search direction by conjugating the current gradient with the previous gradient directions [2]. This results in significant speedup as compared to the simple gradient descent method.

Let's understand the working of NCG with an example.


`Insert code sample here`

In this code sample, we define an example optimization problem and use NCG to minimize it. You will notice that NCG is faster than the L-BFGS method in this example.

## Conclusion

In this chapter, we introduced Nonlinear Conjugate Gradient (NCG), a powerful optimization algorithm used in Machine Learning. We hope that you found this chapter informative and learned how NCG can help you optimize your models.

We would like to thank Yves Nesterov for his contribution to this chapter. Stay tuned for the next chapter, where we will explore another exciting optimizer!

Happy learning!

## References

[1] Nocedal, J., & Wright, S. (2006). Numerical Optimization (2nd ed.). Springer.

[2] Dai, Y., & Yuan, Y. (2002). Nonlinear conjugate gradient methods. Mathematical programming, 111(1), 129-156.
# Chapter 15: Nonlinear Conjugate Gradient (NCG) - The Magical Conjugate

Once upon a time, there was a land filled with magical creatures and mystical enchantments. The land was divided into two parts - the first part was inhabited by unicorns who loved to compete with each other to see who could run faster, and the second part was inhabited by dragons who loved to play with fire.

One day, a young wizard named Yves Nesterov visited this land, and he noticed that the unicorns and dragons were not living in harmony. He decided to bring them together and create a competition where they could work together and help each other.

Yves noticed that the unicorns were excellent at running, but they often got tired and could not keep up with the dragons' speed. He also noticed that the dragons were not very agile and could not run as fast as the unicorns. So, he decided to use his magic to create a way for them to work together and win the competition.

Yves used his magic to create a magical spell called Nonlinear Conjugate Gradient (NCG). The spell used the dragons' fire and the unicorns' speed to create a powerful force that could help them win the competition. The unicorns used their speed to carry the dragons, while the dragons used their fire to help the unicorns run faster.

The other wizards in the land were amazed at the power of the magical spell and its ability to bring the unicorns and dragons together. They asked Yves how he had created such a powerful spell, and he explained that NCG was based on the principle of conjugation, where the current gradient was conjugated with the previous gradient directions to compute the search direction. This resulted in significant speedup as compared to the simple gradient descent method, making it a powerful optimizer.

From that day on, the unicorns and dragons worked together in harmony, using the power of Nonlinear Conjugate Gradient to win every competition they entered. The wizard Yves had brought a new magic into the land and had made everyone's lives happier.

As you embark on your Machine Learning journey, remember the magical spell of Nonlinear Conjugate Gradient and the principles of conjugation that it is based on. With the help of NCG and the lessons of Yves Nesterov, you too can create powerful models and optimize them with ease.
Certainly!

In the code sample provided in the chapter, we demonstrate how to use the Nonlinear Conjugate Gradient (NCG) optimizer to minimize a simple function. The function we chose is a quadratic function defined as:

```python
def objective_function(x):
    return x[0]**2 + 0.5*x[1]**2
```
We initialize the optimizer with the initial guess of `[1.0, 1.0]` and a tolerance of `1e-8`. Then, we call the `fmin_cg` function from the `scipy.optimize` module to perform the optimization.

```python
from scipy.optimize import fmin_cg

# define the problem
start_point = np.array([1.0, 1.0])
tolerance = 1e-8

# perform the optimization
x_min = fmin_cg(objective_function, x0=start_point, gtol=tolerance)
```

The `fmin_cg` function takes three arguments: the objective function we defined above (`objective_function`), the initial guess (`start_point`), and the tolerance (`tolerance`) that specifies the convergence criteria for the optimization. 

The optimizer performs a line search along the conjugate direction and updates the conjugate gradient direction and the search interval for each iteration. The optimizer keeps iterating until it reaches the tolerance level, at which point it returns the optimized values.

As you can see from the code, NCG is quite easy to use and can provide faster convergence than other optimization methods like L-BFGS on certain problems. We hope this code sample helps you understand the principles behind Nonlinear Conjugate Gradient and how to use it in practice.

Happy learning!


[Next Chapter](16_Chapter16.md)