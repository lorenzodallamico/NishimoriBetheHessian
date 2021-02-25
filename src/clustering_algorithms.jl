using LightGraphs, SparseArrays, LinearAlgebra, StatsBase, DataFrames, Distributions, KrylovKit, ParallelKMeans



"""
This function performs spectral clustering on a weighted graph using the naive mean field approximation

----------
```X, estimated_ℓ = clustering_MF(edge_list, J_edge_list, n, N_repeat, verbose)```

Entry
----------
* ```edge_list```   : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```           : size of the matrix (Int64)

Optional entries
------------------
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output. By default set to 2
* ```N_repeat```     : number of repetitions of the k-means algorithm. By default set to 8

Returns
-------
* ```X```              : Informative eigenvector (Array{Float64,1})
* ```estimated_ℓ```    : Array ∈ {-1,1} containing the label assignement (Array{Int64,1})
"""
function clustering_MF(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64; N_repeat = 8, verbose = 1)
    
    
    J = sparse(edge_list[:,1],edge_list[:,2], J_edge_list, n,n) # create sparse matrix J
    J = J+J'
    
    s, X = eigsolve(J, 1, :LR, ishermitian = true) # compute the largest eigenpair
    if verbose >= 1
        printstyled("o Running kmeans\n"; color = 2)
    end
    fKM = [ParallelKMeans.kmeans(X[1]', 2) for r in 1:N_repeat] # run k-means on the entries of the eigenvector 
    f = [fKM[r].totalcost for r=1:N_repeat]
    best = argmin(f) # pick the best trial
    KM = fKM[best]
    estimated_ℓ = KM.assignments # find the label assignments as an output of kmeans.
    
    if verbose >= 1
        printstyled("o Done!\n"; color = 2)
    end
    
    return X, (estimated_ℓ .- 1)*2 .- 1
    
end


##########################################################################################################


"""
This function performs spectral clustering on a weighted graph using the Bethe Hessian matrix at the spin-glass phase transition temperature
Usage

----------
```X, estimated_ℓ = clustering_BH_SG(edge_list, J_edge_list, n)```

Entry
----------
* ```edge_list```   : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```           : size of the matrix (Int64)

Optional entries
------------------
* ```ϵ```            : precision of the estimates (Float64). By default set to 2*10^(-5) 
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output. By default set to 2
* ```N_repeat```     : number of repetitions of the k-means algorithm. By default set to 8
* ```t```            : the temperature adopted is β = β_SG*t. The value of x is sqrt{cΦ*E[tanh^2(β*J)]} (Float64). By default set to 1
Returns
-------
* ```X```             : Informative eigenvector (Array{Float64,1})
* ```estimated_ℓ```   : Array ∈ {-1,1} containing the label assignement (Array{Int64,1})
"""
function clustering_BH_SG(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64; ϵ = 2*10^(-5), N_repeat = 8, verbose = 1, t = 1.)
    
    
    A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix
    A = A+A' # symmetrize the adjacency matrix
    d = A*ones(n) # find the degree vector
    c = mean(d) # estimate the average degree
    Φ = mean(d.^2)/c^2 # second moment of the normalized degree distribution
    
    β_SG =  find_β_SG(J_edge_list, c, Φ, ϵ) # find the spin-glass phase transition
    if verbose >= 1
        printstyled("o The value of β_SG is ", round(β_SG, digits = 2), "\n"; color = 2)
    end
    
    β = t*β_SG
    w_edge_list = tanh.(β*J_edge_list) # build the vector w
    x = sqrt(c*Φ*mean(w_edge_list.^2))
    H = H_matrix(edge_list, w_edge_list, x, n) # create the matrix H at the Nishimori temperature
    s, X = eigsolve(H, 1, :SR, ishermitian = true) # compute the smallest eigenpair
    if verbose >= 1
        printstyled("o Running kmeans\n"; color = 2)
    end
    fKM = [ParallelKMeans.kmeans(X[1]', 2) for r in 1:N_repeat] # run k-means on the entries of the eigenvector 
    f = [fKM[r].totalcost for r=1:N_repeat]
    best = argmin(f) # pick the best trial
    KM = fKM[best]
    estimated_ℓ = KM.assignments # find the label assignments as an output of kmeans.
    
    if verbose >= 1
        printstyled("o Done!\n"; color = 2)
    end
    
    return X, (estimated_ℓ .- 1)*2 .- 1
    
end


################################################################################################################


"""
This function performs spectral clustering on a weighted graph using the Bethe Hessian matrix at the spin-glass phase transition temperature
Usage

----------
```X, estimated_ℓ = clustering_BH_SG(edge_list, J_edge_list, n, ϵ, N_repeat, verbose)```

Entry
----------
* ```edge_list```   : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```           : size of the matrix (Int64)

Optional entries
------------------
* ```ϵ```            : precision of the estimates (Float64). By default set to 2*10^(-5) 
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output. By default set to 2
* ```N_repeat```     : number of repetitions of the k-means algorithm. By default set to 8

Returns
-------
* ```X```             : Informative eigenvector (Array{Float64,1})
* ```estimated_ℓ```   : Array ∈ {-1,1} containing the label assignement (Array{Int64,1})
"""
function clustering_signed_Lap(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64; ϵ = 2*10^(-5), N_repeat = 8, verbose = 1)
    
    
    L = Lap_matrix(edge_list, J_edge_list, n) # create the Laplacian matrix
    s, X = eigsolve(L, 1, :SR, ishermitian = true) # compute the smallest eigenpair
    if verbose >= 1
        printstyled("o Running kmeans\n"; color = 2)
    end
    fKM = [ParallelKMeans.kmeans(X[1]', 2) for r in 1:N_repeat] # run k-means on the entries of the eigenvector 
    f = [fKM[r].totalcost for r=1:N_repeat]
    best = argmin(f) # pick the best trial
    KM = fKM[best]
    estimated_ℓ = KM.assignments # find the label assignments as an output of kmeans.
    
    if verbose >= 1
        printstyled("o Done!\n"; color = 2)
    end
    
    return X, (estimated_ℓ .- 1)*2 .- 1
    
end


##########################################################################################################


"""
This function generates the weighted Laplacian matrix corresponding to a given weighted edge list

Usage
----------

```L = Lap_matrix(edge_list, J_edge_list, n)```


Entry
----------
* ```edge_list```   : undirected edge list containinf the non zero entries of H (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```           : size of the matrix (Int64)

Returns
-------
```L```         : sparse weighted matrix L (SparseMatrixCSC{Float64,Int64})
"""
function Lap_matrix(edge_list::Array{Int64,2}, w::Array{Float64,1}, n::Int64)

        
    W = sparse(edge_list[:,1],edge_list[:,2], w, n,n)
    W = W+W'
    
    absW = sparse(edge_list[:,1],edge_list[:,2], abs.(w), n,n)
    absW = absW+absW'
    
    Λ = spdiagm(0 => absW*ones(n))
    
    L = Λ - W

    return L
    
end
