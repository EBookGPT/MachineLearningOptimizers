# Chapter 26: Binary Search

Welcome back, my dear reader! In the previous chapter, we learned about Linear Search, an algorithm that sequentially searches for an element in the dataset. While linear search is a simple and intuitive algorithm, it can be time-consuming for large datasets. That is why we need to explore other search algorithms that can optimize the search process.

In this chapter, we will delve into Binary Search, a classic search algorithm widely used in computer science. Binary search is a divide and conquer algorithm that works by dividing the dataset in half, then repeatedly narrowing down the search range until the desired element is found or determined to be absent.

Binary search is particularly useful when working with ordered datasets (ascending or descending), as it can quickly locate the desired element in logarithmic time. In fact, binary search has a time complexity of O(log n), making it one of the most efficient search algorithms for large datasets.

We will start this chapter with a brief introduction to the algorithm's concept and idea. We will then explain how to implement binary search in Python and provide some code samples to illustrate the algorithm's execution sequence. Additionally, we will highlight the strengths and weaknesses of binary search algorithm and provide some use cases where this algorithm is particularly useful.

So, sit tight and enjoy the ride as we uncover the Binary Search algorithm!
# Chapter 26: Binary Search - The Search Within

Once upon a time, in the Land of Oz, there lived a wise old wizard named Merlin. Merlin was known for his magical powers, especially his ability to find anything that was lost. One day, the Scarecrow, the Tin Man, and Dorothy asked Merlin for advice on how to find a hidden treasure. Merlin, being the wise old wizard he was, suggested they use the Binary Search algorithm to locate the treasure.

Merlin explained to them that Binary Search was a magical algorithm that worked by searching for a specific element within a dataset by dividing the dataset in half repeatedly, continuing the search in the half that could contain the target value. He told them that this algorithm was particularly powerful for searching in large datasets and that it worked best on sorted datasets.

The Scarecrow, who was fascinated by the idea of Binary Search, asked Merlin to teach him how the algorithm worked. Merlin, being the kind and patient wizard he was, took out a long parchment and started to explain the algorithm step by step.

First, Merlin wrote down a list of numbers and asked the Scarecrow to pick a number to search for. The Scarecrow chose the number 7.

```
Our dataset: [1, 3, 4, 5, 7, 8, 10, 12, 14, 17]
Our target value: 7
```

Merlin then explained that they would compare the target value to the middle element of the dataset (in this case, the element with index 4). As the target value is smaller than the middle element, they would continue the search within the first half of the dataset.

```
[1, 3, 4, 5, 7, 8, 10, 12, 14, 17]
        ^
Target: 7
```

The next step was to repeat the process, dividing the first half of the dataset in half and comparing the target value to the middle element (in this case, the element with index 2). As the target value is greater than the middle element, they would continue the search within the second half of the first half of the dataset.

```
[1, 3, 4, 5, 7, 8, 10, 12, 14, 17]
    ^   
Target: 7
```

They would then repeat this process until they found the target value or determined that it wasn't in the dataset. The Scarecrow was amazed by how quickly they found the target value using the Binary Search algorithm. He realized that this magical algorithm could help them find the hidden treasure.

Dorothy and Tin Man were also impressed by the Binary Search algorithm. They asked Merlin to help them incorporate it into their machine learning models. Merlin was more than happy to help them and showed them how to translate this algorithm into Python code.

```python
def binary_search(arr, target):
  low = 0
  high = len(arr) - 1

  while low <= high:
    mid = (low + high) // 2

    if arr[mid] == target:
      return mid
    elif arr[mid] < target:
      low = mid + 1
    else:
      high = mid - 1

  return -1
```

The Tin Man and Dorothy were amazed by how easy the Binary Search algorithm was to implement. They immediately started using it in their predictive models and saw great results.

From that day on, the Scarecrow, Tin Man, and Dorothy used Binary Search to locate all sorts of treasures and optimized their machine learning models using this magical algorithm. They became well-known for their ability to find what was lost, all thanks to Merlin's wise advice and the Binary Search algorithm.
# Explanation of the Binary Search Code

In the Wizard of Oz parable, Merlin introduced the Scarecrow, Tin Man, and Dorothy to the magical Binary Search algorithm. To help them incorporate the Binary Search algorithm into their machine learning models, Merlin showed them how to write a Python implementation of the algorithm.

The code for the Binary Search algorithm is a function called `binary_search` that takes in two parameters: an ordered array and a target value that needs to be searched. The function then splits the array in half and compares the target value to the middle element of the array. Depending on the comparison, the function will then narrow down the search range by either searching the first half or the second half of the array. This process is repeated until the target value is found or determined to be absent.

Here's a line-by-line explanation of the code:

```python
def binary_search(arr, target):
```

- `binary_search`: This is the function name, which is used to implement the Binary Search algorithm.
- `arr`: This is the ordered array that contains the elements to be searched.
- `target`: This is the target value that needs to be searched within the array.

```python
low = 0
high = len(arr) - 1
```

- `low`: This initializes a counter to 0. This counter represents the lowest index of the array that still needs to be searched.
- `high`: This initializes a counter to the last index of the array. This counter represents the highest index of the array that still needs to be searched.

```python
while low <= high:
```

- This starts a while loop that continues until the search range is either empty (i.e., the target value was not found) or the target is found.

```python
mid = (low + high) // 2
```

- `mid`: This is the middle value of the current search range. The `//` operator performs integer division, ensuring that the middle index is not a decimal number.

```python
if arr[mid] == target:
  return mid
elif arr[mid] < target:
  low = mid + 1
else:
  high = mid - 1
```

- This block of code compares the target value to the middle element of the array.
- `arr[mid] == target`: This checks if the middle element is equal to the target value. If it is, the function returns the index of the middle element.
- `arr[mid] < target`: This checks if the middle element is less than the target value. If it is, then the target value is in the second half of the array. Therefore, the function updates `low` by setting it to `mid + 1`.
- `arr[mid] > target`: This checks if the middle element is greater than the target value. If it is, then the target value is in the first half of the array. Therefore, the function updates high by setting it to `mid - 1`.

```python
return -1
```

- This line of code executes if the target value is not found within the search range. It returns -1 to indicate that the target value is not found in the array.

Overall, the Binary Search algorithm provides a very efficient method for finding elements in large, ordered datasets. The provided Python code is a straightforward implementation of the algorithm, which can be easily incorporated into any machine learning algorithm that requires searching an ordered list.


[Next Chapter](27_Chapter27.md)