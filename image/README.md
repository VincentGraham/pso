### Image Processing with Swarm Intelligence Algorithms

Image segmentation can be done through a level set method that maximizes 
The zero level set is encoded as a matrix which is non-zero only on the boundary of the level set contour.

The objective function used is an L2 norm regularization, applied to the cluster indexing vector.



Given a gaussian kernel $K$, a within cluster distance, given a centroid $c_i$, can be defined as $K\ast (c_i-x)^2$.
Thus, for $k$ centroids and data points $x$, the energy function to be minimized is

$$\sum_{i=0}^k \lambda \Vert\partial C_i \Vert + \int_{x\in C_i}K\ast(c_i - x)^2 $$

where $x\in C_i$ is all data points assigned to centroid $i$ and $\partial C_i$ is a smooth boundary regularization term.

