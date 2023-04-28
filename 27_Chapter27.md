# Chapter 27: Golden Section Search

Welcome back, learners! In the previous chapter, we discussed one of the most basic search algorithms for machine learning optimizers, the binary search algorithm. We hope you all had fun learning and implementing the algorithm. 

In this chapter, we will introduce you to another search algorithm called **Golden Section Search (GSS)**. GSS is a popular search algorithm which provides an efficient method for computing a function's minimum without relying on derivative information.

To help us understand the intricacies of GSS, we have a special guest joining us in this chapter - **Richard Brent**. In case you haven't heard of him, Richard Brent is an Australian mathematician known for his contributions to computational mathematics, in particular his work on integer factorization and primality testing. 

Together with Richard, we will dive deep into GSS and explore the various intricacies of the algorithm. We will learn how GSS works, when to use it, and how to implement it. 

But first, let's quickly recap on what we learned in the previous chapter. We learned that binary search is a search algorithm that works by repeatedly dividing the search interval in half. It is a fast and efficient algorithm that works well for locating a single target value within a list of sorted values. 

Now, let's begin our journey with Richard Brent and dive into the world of GSS! 

```
import math

def golden_section_search(f, interval, tol=1e-6):
    a, b = interval
    r = (math.sqrt(5) - 1) / 2
    x1 = b - r * (b - a)
    x2 = a + r * (b - a)
    f1, f2 = f(x1), f(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - r * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + r * (b - a)
            f2 = f(x2)

    return (a + b) / 2
``` 

Above is the python implementation of GSS. We will be diving deep into the code in the upcoming sections to understand it better. So, stay with us till the end of this chapter.
# Chapter 27: Golden Section Search - The Wizard of Oz Parable

Once upon a time, there were four characters who were on a quest to find the secret formula to make a perfect diamond. These characters were the Scarecrow, the Tin Man, the Cowardly Lion, and Dorothy. They had been on this journey for a long time, but they were still far away from their destination.

As they were walking through a dense forest, they came across a wise old mathematician named Richard Brent. He was known throughout these lands for his knowledge of math and his ability to teach complex algorithms. The four friends explained their situation and asked Richard for his help.

Richard listened to their story and explained to them about the Golden Section Search algorithm. He told them how it could help them find the perfect diamond by minimizing the cost of the materials needed to create it. 

The Scarecrow and Tin Man were impressed by Richard's explanation and were eager to learn more about the algorithm. The Cowardly Lion, on the other hand, was skeptical of the algorithm and wasn't sure if it would be of much help. Dorothy, as usual, was curious and wanted to try out the algorithm to see if it worked.

Richard took out his wand (which looked surprisingly like a python code editor) and began to explain the code for the algorithm. He showed them how the algorithm worked by dividing an interval in a specific ratio to narrow down the search for the minimum of a function. 

The Scarecrow, who was good with numbers, quickly understood how the algorithm worked and offered to implement it. The Tin Man, who was skilled with tools, helped the Scarecrow fix the parameters. Dorothy, the inquisitive one, observed the code carefully and asked many questions. The Cowardly Lion, who still doubted the algorithm, was convinced when he saw how quickly the algorithm found the minimum.

Thanks to Richard's guidance, the four friends successfully applied the Golden Section Search algorithm to their problem and found the secret formula for making a perfect diamond. They were grateful to Richard for his help and knowledge.

As they marched forward, Dorothy asked Richard if he could join them on their journey. Richard, who had a soft spot for curious students, readily agreed. From that day on, they all worked together, each using their unique skills to solve problems along the way.

And so, the Scarecrow, the Tin Man, the Cowardly Lion, Dorothy, and Richard Brent, continued on their journey, using math and different algorithms to solve problems, and occasionally breaking into song and dance.

The end.

```python
# Golden Section Search Implementation in Python

import math

# defining the function to be minimized
def sample_function(x: float) -> float:
    return math.sin(x) - x / 2

# defining the interval
interval = [0, 5]

# calling the golden section search function
min_val = golden_section_search(sample_function, interval)

# printing out the minimum value
print("The minimum value of the function is: ", min_val)
``` 

Above is an implementation of the Golden Section Search algorithm in Python. You can use it to solve your optimization problems. Happy learning!
Sure, let me explain the Python code used to solve the problem in the Wizard of Oz parable.


```python
# defining the function to be minimized
def sample_function(x: float) -> float:
    return math.sin(x) - x / 2
```

In this snippet, we define our function `sample_function` that we want to minimize. In this case, we use a `math.sin` function and subtract `x/2` from it. You can replace this function with any function you want to minimize in your application.

```python
# defining the interval
interval = [0, 5]
```

Here, we define the interval in which we want to search for the minimum of the function. In this case, we start from `0` and go up to `5`. You can change this interval according to the requirements of your application.

```python
# calling the golden section search function
min_val = golden_section_search(sample_function, interval)

# printing out the minimum value
print("The minimum value of the function is: ", min_val)
```

Finally, we call the `golden_section_search` function to find the minimum value of our function. This function takes two arguments - our function `sample_function` and the interval `interval`. It returns the minimum value of the function within the given interval.

In this example, we also print out the minimum value of the function. You can modify this code to do whatever you want with the minimum value of the function, such as using it in further calculations or storing it in a database.

Overall, this Python code demonstrates how you can use the Golden Section Search algorithm to minimize a given function within a given interval.


[Next Chapter](28_Chapter28.md)