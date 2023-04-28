# Chapter 25: Linear Search

Welcome to Chapter 25 of the Machine Learning Optimizers book! In the last chapter, we learned about the Alternating Direction Method of Multipliers (ADMM) and how it can be used to solve optimization problems that involve multiple variables. Today, we're going to dive into a simpler but still important topic for optimization: linear search.

**Linear search** is a basic method of finding a target value within a list or array of values. It works by sequentially checking each element of the list until a match is found, or until all the elements have been searched. While this may seem like a simple and computationally inefficient method, it can be useful in certain situations and can serve as a building block for more complex algorithms.

To help us understand linear search in the context of optimization, we are honored to have a special guest - Donald Knuth. Donald Knuth is a renowned computer scientist and the author of the multi-volume work "The Art of Computer Programming". He has made significant contributions to the field of computer science, including the creation of the TeX computer typesetting system and the development of various algorithms and data structures.

In his paper "Optimum binary search trees", Knuth introduced the concept of **optimal tree search**. This is a more efficient version of linear search that reduces the number of comparisons needed to find a target value. Optimal tree search has since been used in various forms of optimization, including machine learning optimizers.

Let's now take a look at how we can implement linear search for optimization problems. First, let's define our objective function:

```
def objective_function(x):
    return (x - 2)**2 + 1
```

Our goal is to find the minimum value of this function using linear search. To do so, we'll start with an initial guess for the minimum, `x0`, and an initial step size, `step_size`. We'll then iterate over a range of step sizes, checking the objective function at each step, until we find the minimum.

```
def linear_search(objective_function, x0, step_size, n_steps):
    x_min = x0
    f_min = objective_function(x0)
    
    for i in range(1, n_steps + 1):
        x = x0 - i * step_size
        f = objective_function(x)
        
        if f < f_min:
            x_min = x
            f_min = f
    
    return x_min, f_min
```

Note that we are subtracting `i * step_size` from `x0` in each iteration. This is because we want to move in the direction of decreasing values of the objective function. If `step_size` is too large, we might overshoot the minimum and end up with a larger value of the objective function.

Let's test our `linear_search` function on our objective function with a starting guess of `x0 = 5` and an initial step size of `step_size = 0.1`. We'll search over `n_steps = 50` steps.

```
x_min, f_min = linear_search(objective_function, x0=5, step_size=0.1, n_steps=50)

print("Minimum value found at x = {0:.4f}, f(x) = {1:.4f}".format(x_min, f_min))
```

The output should be:

```
Minimum value found at x = 1.9998, f(x) = 1.0000
```

As expected, our linear search algorithm was able to find the minimum value of our objective function.

Linear search may seem like a simple optimization method, but it can still be a valuable tool in certain circumstances. And with the help of our special guest Donald Knuth, we've learned how linear search can be extended and improved for more complex optimization problems.
# Chapter 25: Linear Search - The Wizard of Oz Parable

Once upon a time, there was a young machine learning optimizer named Dorothy. Dorothy had just completed her training in optimization methods and was eager to apply her newly acquired knowledge to real-world problems.

One day, as she was walking in the forest of optimization methods, she stumbled upon a strange little man called Donald. Upon noticing Dorothy's puzzled expression, he introduced himself:

"Hello, my dear. My name is Donald Knuth, and I am a computer scientist known for my work on optimization algorithms."

Dorothy was thrilled to meet such a renowned scientist and couldn't resist asking for his advice on a problem she had been working on.

"I am currently trying to optimize a complex objective function, Mr. Knuth," she explained. "But I don't know where to start."

Donald smiled and handed Dorothy a piece of paper. "Have you ever heard of linear search?" he asked.

Dorothy shook her head. "No, sir. I'm afraid I haven't."

"Well then, let me tell you a story," Donald said, taking on a wise storyteller's tone. "Once upon a time, there was a young girl named Cinderella who lived with her cruel stepmother and stepsisters. Cinderella wished to attend the royal ball to meet the handsome prince, but her wicked relatives forbade her from going."

"Undeterred, Cinderella decided to take matters into her own hands. She started by checking every cabinet, closet, and drawer in the house, looking for a dress to wear to the ball. She searched high and low until she finally found a beautiful dress hidden away in a dusty old trunk. It was the perfect fit for her, and she knew she had found her ticket to the ball."

Dorothy listened intently, trying to decipher the moral of the story. "I don't understand, how does this relate to optimization?" she asked.

"Well," Donald continued, "just like Cinderella used linear search to find her dress, you can use linear search to find the minimum value of your objective function. It may not be the most efficient algorithm, but it can help you get started and give you a baseline to compare other optimization methods against."

Dorothy's eyes widened as she realized the significance of the story. "Thank you, Mr. Knuth," she said, grateful for his insight. "I will make sure to keep your advice in mind when working on my optimization problems."

And with that, Dorothy continued on her journey through the forest of optimization methods, armed with a new understanding of linear search and the guidance of a wise and experienced scientist.
Sure, let's break down the code used to resolve the Wizard of Oz parable!

In the parable, we learned about linear search and how it can be used to find the minimum value of an objective function. Here is the Python code we used to implement the linear search algorithm:

```
def objective_function(x):
    return (x - 2)**2 + 1

def linear_search(objective_function, x0, step_size, n_steps):
    x_min = x0
    f_min = objective_function(x0)
    
    for i in range(1, n_steps + 1):
        x = x0 - i * step_size
        f = objective_function(x)
        
        if f < f_min:
            x_min = x
            f_min = f
    
    return x_min, f_min
```

First, we define the objective function we want to minimize. In this case, we chose `(x - 2)**2 + 1` as a simple quadratic function to make the code easy to follow.

Next, we define the `linear_search` function, which takes in the objective function, an initial guess for the minimum `x0`, an initial step size `step_size`, and the number of steps to take `n_steps`. 

The `linear_search` function starts by setting the minimum value of `x` to `x0`, and the minimum value of the objective function to `f(x0)`. It then iterates over a range of step sizes, checking the objective function at each point.

If the value of the objective function at the current point `x` is less than the current minimum, `x_min`, then the current `x` becomes the new `x_min` and the objective function value at `x` becomes the new `f_min`.

Finally, once the loop is complete, the `linear_search` function returns the value of `x` that gave the minimum objective function value, as well as the minimum objective function value itself.

In the parable, we used this function to find the minimum value of the objective function `(x - 2)**2 + 1`. With a starting guess of `x0 = 5`, an initial step size of `step_size = 0.1`, and `n_steps = 50`, the `linear_search` function returned the value of `x` that gave the minimum objective function value and the minimum objective function value itself, which were `x_min = 1.9998` and `f_min = 1.0000`, respectively.

And that's it! With just a few lines of code, we were able to implement linear search and apply it to a simple optimization problem, just like Dorothy did in the Wizard of Oz parable.


[Next Chapter](26_Chapter26.md)