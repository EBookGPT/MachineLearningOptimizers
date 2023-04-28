# Chapter 12: Krylov Subspace Methods

Welcome to the twelfth chapter of our book on Machine Learning Optimizers! In the previous chapter, we discussed the natural gradient descent, an optimizer that allows us to efficiently optimize models with Riemannian geometry. 

In this chapter, we will dive into another class of optimization algorithms, namely the Krylov subspace methods. These iterative methods were initially designed for solving large systems of linear equations, but they also have applications in machine learning, particularly in the optimization of large-scale neural networks.

We are excited to have Yousef Saad, a prominent figure in the field of linear algebra and numerical analysis, join us as a special guest in this chapter. Yousef Saad has made significant contributions to the development and analysis of Krylov subspace methods, and we are fortunate to have him share his insights and expertise with us.

Throughout this chapter, we will explore the key concepts behind Krylov subspace methods, including Arnoldi iteration, Lanczos iteration, and the GMRES algorithm. We will also discuss their implementation in machine learning and their specific applications in neural network optimization.

So, whether you're an experienced practitioner or a newcomer to the field, we invite you to join us on this exciting journey through Krylov subspace methods. Let's get started!
# Chapter 12: Krylov Subspace Methods

Once upon a time, in the Land of Machine Learning, there was a young, ambitious Data Scientist named Dorothy. Dorothy had a passion for solving complex problems, and her latest challenge was to optimize a large-scale neural network for image classification. Despite her best efforts with traditional optimizers, she struggled to make significant progress.

One day, while lost in thought, Dorothy found herself in a strange and unfamiliar place. She looked around and saw a sign that read, "Welcome to the Krylov Subspace." As she stood there in awe, a voice spoke to her, "Greetings, young Dorothy. I am Yousef Saad, and I have been expecting you."

Dorothy was surprised to meet Yousef Saad, the famous mathematician and expert on Krylov subspace methods. She introduced herself and explained her quest for optimizing her neural network. Yousef Saad listened intently and shared his wisdom, "Krylov subspace methods are well-suited for optimizing large-scale systems and may help you overcome the challenges you're facing. Let me guide you through the land of Krylov subspace, and we will see what we can find."

They began their journey by exploring the foundations of Krylov subspace methods, particularly the concept of an iterative process that generates a sequence of increasingly accurate estimates of the underlying solution. Yousef Saad explained that one of the critical components of this process is the Arnoldi iteration, which computes a basis for the Krylov subspace associated with the linear operator.

As they continued their journey, they encountered the famous Lanczos iteration, which generates a tridiagonal matrix for a symmetric operator, and the GMRES algorithm, a widely used Krylov subspace method that solves nonsymmetric linear systems. Yousef Saad demonstrated how these methods could optimize neural networks efficiently and accurately and helped Dorothy implement them.

It wasn't long before Dorothy achieved a breakthrough, surpassing her previous benchmark accuracies. She was overjoyed and expressed her gratitude to Yousef Saad for his guidance. As they bid each other farewell, Yousef Saad imparted one final message: "Remember, young Dorothy, that Optimization is a journey that requires patience, creativity, and the courage to explore new frontiers. Keep exploring, and you'll find the answers you seek."

With renewed confidence and an abundance of new knowledge, Dorothy returned to her world of Machine Learning, eager to apply her newfound expertise and share her adventure with others. And so, the tale of Krylov subspace methods lived on as a testament to the power of math and the wonder of Machine Learning.
In our Wizard of Oz parable for Chapter 12, Krylov Subspace Methods, we explored the journey of a young Data Scientist named Dorothy who seeks to optimize a large-scale neural network for image classification. She encounters a mathematician named Yousef Saad, who introduces her to the world of Krylov subspace methods and helps her implement them to achieve a breakthrough in her optimization task.

To bring this parable to life, we can illustrate the key concepts of Krylov subspace methods with code examples. Specifically, we can use MATLAB to implement the Arnoldi iteration, Lanczos iteration, and the GMRES algorithm, which are fundamental components of Krylov subspace methods.

For example, below is a MATLAB code snippet that demonstrates how to implement the Arnoldi iteration for a given operator A:

```matlab
n = length(A);
m = 5;
Q = zeros(n, m);
H = zeros(m, m);
q1 = randn(n, 1);
q1 = q1 / norm(q1);
Q(:, 1) = q1;

for k = 1:m
    q = A * Q(:, k);
    for j = 1:k
        H(j, k) = Q(:, j)' * q;
        q = q - H(j, k) * Q(:, j);
    end
    if k < n
        H(k+1, k) = norm(q);
        if H(k+1, k) == 0
            break;
        end
        Q(:, k+1) = q / H(k+1, k);
    end
end
```

This code snippet generates a basis for the Krylov subspace using the Arnoldi iteration, where `m` is the number of iterations, `A` is the linear operator, and `q1` is a random vector. The code also computes an upper Hessenberg matrix `H` that stores the information of the projection of the operator in the subspace spanned by the left singular vectors of `A`.

Similarly, we can use MATLAB to implement the Lanczos iteration, which is a special case of the Arnoldi iteration for symmetric operators. Here is an example:

```matlab
n = length(A);
m = 5;
alpha = zeros(m, 1);
beta = zeros(m, 1);
q1 = randn(n, 1);
q1 = q1 / norm(q1);
q = zeros(n, 1);
q_old = zeros(n, 1);
q(:, 1) = q1;

for k = 1:m
    alpha(k) = q(:, k)' * A * q(:, k);
    if k == 1
        beta(k) = 0;
    else
        beta(k) = norm(q(:, k-1));
    end
    if k == n
        break;
    end
    q(:, k+1) = (A - alpha(k) * eye(n)) * q(:, k) - beta(k) * q_old(:, k);
    q_old(:, k) = q(:, k);
end

T = diag(alpha) + diag(beta(2:end), 1) + diag(beta(2:end), -1);
```

This code generates a tridiagonal matrix `T` that represents the projection of the symmetric operator onto a Krylov subspace using the Lanczos iteration.

Lastly, we can use MATLAB to implement the GMRES algorithm, which is a widely used Krylov subspace method for solving nonsymmetric linear systems. Here is an example:

```matlab
n = length(A);
m = 5;
b = randn(n, 1);
x0 = zeros(n, 1);
r0 = b - A * x0;
v1 = r0 / norm(r0);
V = zeros(n, m);
H = zeros(m+1, m);
V(:, 1) = v1;

for j = 1:m
    w = A * V(:, j);
    for i = 1:j
        H(i, j) = V(:, i)' * w;
        w = w - H(i, j) * V(:, i);
    end
    H(j+1, j) = norm(w);
    if H(j+1, j) == 0
        break;
    end
    V(:, j+1) = w / H(j+1, j);
    
    e1 = zeros(j+1, 1);
    e1(1) = 1;
    y = H(1:j+1, 1:j) \ norm(r0) * e1;
    x = x0 + V(:, 1:j) * y;
end

```

This code uses the GMRES algorithm to find the solution of the linear system `A x = b`, where `A` is a nonsymmetric matrix, using a Krylov subspace of dimension `m`. The code generates an orthonormal basis of the Krylov subspace and a corresponding upper Hessenberg matrix `H`. Then, it uses a least-squares method to find the coefficients `y` that minimizes the residual error of the linear system.

In conclusion, the implementation of these fundamental Krylov subspace methods in MATLAB showcases the power and versatility of these iterative optimization techniques, which can be used to solve a variety of problems in Machine Learning and beyond.


[Next Chapter](13_Chapter13.md)