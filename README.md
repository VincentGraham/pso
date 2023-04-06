#  Swarm Intelligence Optimization
My senior project at Case Western Reserve University in the math department. It applies bio-inspired optimization techniques to clustering problems on elementary data sets. The full writeup can be found [here](/eho/eho_final.pdf) with some interesting LaTeX character tuning techniques [here](/eho/main.tex#L67-L88). 

A further example is done with image segmentation through a level set/Chan-Vese inspired method.
A Python and MATLAB version are provided.

## Requirements
The Python version uses

* [Python 3.11.0](https://www.python.org/downloads/release/python-3110/)
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy) or [CPython](https://github.com/python/cpython)/C compiler

MATLAB uses

* [MATLAB r2022b](https://www.mathworks.com/products/matlab.html)
* [Image Processing Toolbox](https://www.mathworks.com/products/image.html)
* [Statistics Toolbox](https://www.mathworks.com/products/statistics.html)

## Particle Swarm Optimization

* [MATLAB PSO](/eho/pso.m)
* [Python PSO](/eho/python/pso.py)

Implements the following iterative algorithms

* R. Eberhart and J. Kennedy, A new optimizer using particle swarm
theory, in MHS’95. Proceedings of the Sixth International Symposium on
Micro Machine and Human Science, 1995, pp. 39–43
* A. P. Engelbrecht, Computational Intelligence: An Introduction, John
Wiley & Sons, 2 ed., 2007

Particle Swarm Optimization originally appears in Eberhart and Kennedy and an informant based modification is provided in Engelbrecht.
Particles are drawn towards a global maximum or minimum of an objective function, $f$, and are influenced by environmental factors.
These factors can include a previous best position of each particular particle, the current best particle, or a mathematically determined better position in the neighborhood of each particle. 
The score of the particle is determined by the objective function $f$.


### Method
Let $x_j^{(t)}\in\mathbb{R}^n$ denote the $j$th particle in the swarm at iteration $t$ and $f:\mathbb{R}^n \to \mathbb{R}_+$ be the objective function which we are minimizing.
Set parameters $c_1$, $c_2$, and $c_3$ and define $r_1$, $r_2$ to be uniform random variables: $r\sim \mathcal{U}(0, 1)$.
At a particular time step $t$, particle motion is then described by the following formulas

$$ v_j^{(t+1)} = c_1v_j^{(t)} + 2r_1c_2\left(p_{j, 1} - x_j^{(t)}\right) + 2r_2c_3\left(g-x_j^{(t)}\right) $$

$$ x_j^{(t+1)} = x_j^{(t)} + v_j^{(t+1)} $$

where $p_{j, 1}$ is the best position in the history of the particle's previous positions and $g$ is the best overall position in the history of every particle in the swarm. These are called informants.

A small improvement can be made to utilize more informants by adding terms for $p_{j, 2}$ and $p_{j, 3}$, the second and third best positions in the particles history.

Convergence is achieved when either the particle positions are all identical, or their velocity is zero.
$g$ is returned as the minimizer of $f$.


### Example

Here, the classical Particle Swarm Optimization algorithm is used to minimize a 2D ridge function, a class of common difficult benchmarking problems.
The function I used was

$$ h(\mathbf{x}) = \left[\left(\Vert{\mathbf{x}}\Vert^2 - 2 \right)^2\right]^\frac{1}{8} + \frac{1}{2} \left( \frac{1}{2} \Vert \mathbf{x} \Vert^2 + \mathbf{1}^T\mathbf{x} \right)$$

with the true minimizer being $\mathbf{x} = ( \textrm{-}1,  \textrm{-}1 )$.

<figure>
<img src="./figures/pso_2d.gif", alt="mising"/>
</figure>

### $lbest$ pso
$lbest$ pso uses a local best topology based on the three best positions in the history of each particle (three informants).

The update for a particle $x$ at time step $t$ is done by

$$ x^{(t+1)} \leftarrow x^{(t)} + c_1(x_{pbest_1} - x^{(t)}) + c_2(x_{pbest_2} - x^{(t)}) + c_3(x_{pbest_3} - x^{(t)}) $$

where $c_i$ are parameters calculated from a tunable $\varphi$ and 3 uniform random variables $r_i$.
$\varphi = 2.07$ is the value found in the literature.




# Elephant Herding Optimization
A modification of Particle Swarm Optimization based on the behavior of elephants.
## Motivation
* Elephants live in herds led by a matriarch
* Male elephants leave the herd once they have reached adulthood
* Elephants live in a home-range and seldom leave

## Algorithm
Given an objective function $F:\Omega \to \mathbb{R}$, elephants live in the domain $\Omega$ called the solution space, bounded by $[x_{min}, x_{max}]$.
Elephants are divided into herds, every herd usually has the same number of elephants. 
Herds do not interact with eachother.
A score is assigned to each elephant using $F(x)$.

* $x_{best}$ the overall best elephant at any iteration
* $x_{ci, best}$ the best elephant of the $ci$th herd at the current iteration
* $x_{ci, worst}$ the worst elephant of the $ci$th herd at the current iteration

$$ x_{ci, j} \leftarrow x_{ci, j} + \alpha r \left( x_{ci, best} - x_{ci, j} \right)$$

$$ x_{ci, best} \leftarrow x_{ci, best} + \beta \sigma\hspace{2em} \sigma\sim \text{Cauchy} $$

$$ x_{ci, worst} \leftarrow x_{min} + r\left(x_{max}-x_{min}\right) $$

$r\sim \mathcal{U}(0, 1)$.



## Clustering
Given $d$ dimensional data in $\mathbb{R}^d$, 
an elephant is treated as a set of centroids $\mathbf{c_1}, \mathbf{c_2},\ldots$ with each $\mathbf{c_i}\in\mathbb{R}^d$.
For example, if the data is 2D and we want to find 3 clusters an elephant contains $3\times 2 = 6$ elements.

### Notation
Often in data science, the variable $x$ refers to a data point; however in particle swarm optimization algorithms, $x$ will be used to denote a particle.
For centroids $\mathbf{c_1, c_2, c_3}$ with $\mathbf{c_i} = \begin{pmatrix} c_{i,1} & c_{i,2} \end{pmatrix}$, an elephant is represented as 

$$ x = \begin{pmatrix} c_{1,1} & c_{2, 1} & c_{3, 1} \\ c_{2, 1} & c_{2, 2} & c_{3, 2} \end{pmatrix} $$

## Gradient based EHO
Applying a gradient operator to the elephant positions can be used to improve the quality of the result.
Suppose we want to find the minimum of $f(x)$.
Beginning from the taylor series around a point $x_0$ and descending $f(x)$, the next point $x_1$ can be caluclated from the first two terms of
$f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)(x-x_0)^2}{2!} + \cdots$.
In this case, $f'(x)' can be approximated by $(x_1-x_0)$, $x_{best} - x_0$ or $x_{best} - x_{worst}$.

$$ x_1 = x_0 - \frac{f(x_0)}{f'(x_0)} $$


## Implementation details
In PSO, the particles are stored in a single matrix.
To determine the distance from each centroid to the data points, a pairwise distance function such as `cdist` from scipy.
However, `cdist` does not work on paginated matrices.
A sample implementation of a page-wise distance calculation is provided in the `transform_reduce3d` method of [distance_metrics2.h](/eho/python/metric/distance_metrics2.h#L156).
But it doesn't offer any performance improvement from using `cdist` in a loop.