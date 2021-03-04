# NBNC
#### **N**ishimori **B**ethe Hessian for **N**ode **C**lassification


These are the codes to reproduce the results appearing in the article 
* Dall'Amico, Couillet and Tremblay - *[Nishimori meets Bethe: a spectral method for node classification in sparse weighted graphs]()* (arXiv:)

The source code represents our implementation of Algorithm 2, appear in the article cited above and can be used to perform node classification on a weighted graph, as well as to perform binart classification of high dimensional vectors. 

> If you make use of these codes, please consider to cite the related article.



##### Content of the package

* The directory ```src``` contains the source codes. The file ```basic_functions.jl``` contains some functions needed to build a (degree-corrected) Erdos-Renyi random graph, together with the non-backtracking and Bethe-Hessian matrix. Overall these functions are needed to reproduce the main theoretical results of our paper. The file ```NBNC.jl``` provides our implementation of Algorithm 2 with all the functions associated to it. Finally ```clustering_algorithms.jl``` contains the implementation of the three spectral algorithms we use as a benchmark.
* Create a directory ```data``` and [download from this link](https://mega.nz/folder/jh1QQAjQ#vT0f9c9sLTOSG1nEAxqGLg) the files ```features.dat``` and ```features_small.dat``` containing the features of 40k (resp. 6k) GAN images that can be used to test our algorithm and. Save the two files in the directory ```data```.
* The notebook ```Figure 2-5``` provides the code needed to reproduce the Figures 2,3,4,5 of our article. These figures constitute the support to our main theoretical findings.
* The notebook ```DEMO``` explains how to: i) generate a synthetic graph on which node classification can be performed; ii) use the files contained in the directory ```data```; iii) use the algorithms for node classifications provided in the ```src``` folder. 

### Required packages

Our code require the following packages

```
LightGraphs, SparseArrays, LinearAlgebra, StatsBase, DataFrames, Distributions, 
KrylovKit, ParallelKMeans
```

### Functions documentation

Each function has a minimal documentation on the inputs, outputs and basic usage. To access it, type
```
?name_of_the_function
```



## Authors

[Lorenzo Dall'Amico](https://lorenzodallamico.github.io/)
[Nicolas Tremblay](http://www.gipsa-lab.fr/~nicolas.tremblay/)
