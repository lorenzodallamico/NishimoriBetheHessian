using LightGraphs, SparseArrays, LinearAlgebra, StatsBase, DataFrames, Distributions, KrylovKit




"""
This function generates an Erdos-Renyi graph with average degree ```c``` and return the graph with the adjacency matrix representation as well as the edge list representation

Usage
----------
```A, edge_list = adjacency_matrix_ER(c,n)```


Entry
----------
* ```c``` : average degree (Float64)
* ```n``` : number of nodes (Int64)

Returns
-------
* ```A```         : sparse representation of the adjacency matrix (SparseMatrixCSC{Float64,Int64})
* ```edge_list``` : edge list representation of the graph (Array{Int64,2})
"""
function adjacency_matrix_ER(c::Float64,n::Int64)
    
    g = erdos_renyi(n,c/n) # erdos-renyi graph taken from the package LightGraphs.jl

    first = zeros(g.ne) # first column of the edge list
    second = zeros(g.ne) # second column of the edge list
    counter = 1
    for i=1:n
        v = g.fadjlist[i][g.fadjlist[i] .> i]
        m = length(v)
        first[counter:counter+m-1] .= i
        second[counter:counter+m-1] = v
        counter = counter + m
    end

    edge_list = hcat(first,second) # create edge list
    edge_list = convert(Array{Int64}, edge_list)
    A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix


    return A+A', edge_list
    
end


################################################################################################################################		



"""
This function generates a sparse Erdos-Renyi graph with degree correction. Note that this implementation is not adapted for dense graphs
Usage

----------
```A, edge_list =  adjacency_matrix_DCER(c,θ)```

Entry
----------
* ```c```   : expected average degree (Float64)
* ```θ```   : expected normalized degree vector (Array{Float64,1})

Returns
-------
* ```A```         : sparse representation of the adjacency matrix (SparseMatrixCSC{Float64,Int64})
* ```edge_list``` : edge list representation of the graph (Array{Int64,2})
"""
function adjacency_matrix_DCER(c::Float64,θ::Array{Float64,1})
    
    n = length(θ) # number of nodes, n
    fs = []
    ss = []
    
    fs = sample(collect(1:n), Weights(θ/n),Int(n*c)) # select nc nodes w.p. θ/n
    ss  = sample(collect(1:n), Weights(θ/n),Int(n*c)) # select the nodes to connect to the considered ones
    
    idx = fs.> ss # keep only edges (ij) in which i>j
    fs2 = fs[idx]
    ss2 = ss[idx]


    edge_list = hcat(fs2,ss2) # create edge list
    edge_list = Array(unique(DataFrame(edge_list))) # remove repeated edges
    A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix

    return A+A', edge_list
end


############################################################################################################################



"""
This function generates the weighted adjacency matrix given the (directed) edge list of a graph and the weight corresponding to each edge
This implementation is an adaptation of the code appearing in the package Erdos.jl

Usage
----------
```B = B_matrix(edge_list, w_edge_list)```


Entry
----------
* ```edge_list```   :  array containing the edge list of the graph (Array{Int64,2})
* ```d_edge_list``` :  array containing the weight associated to each edge of the graph (Array{Float64,1})

Returns
-------
* ```B``` : sparse representation of the non-backtracking matrix (SparseMatrixCSC{Float64,Int64})
"""
function B_matrix(edge_list::Array{Int64,2}, w_edge_list::Array{Float64,1})
    
    d_edge_list = zeros(2*length(edge_list[:,1]), 2) # create the directed edge list
    d_edge_list[1:length(edge_list[:,1]),:] = edge_list
    d_edge_list[length(edge_list[:,1])+1:end,:] = edge_list[:,end:-1:1]  

    d_edge_list = convert(Array{Int64,2},d_edge_list)

    w_d_edge_list = zeros(2*length(edge_list[:,1])) # associate the weights to the directed edge list
    w_d_edge_list[1:length(edge_list[:,1])] = w_edge_list
    w_d_edge_list[length(edge_list[:,1])+1:end] = w_edge_list[:,end:-1:1]

    E = Edge{Int64}
    edge_id_map = Dict{E, Int}()
    for i=1:length(d_edge_list[:,1])
        edge_id_map[E(d_edge_list[i,1],d_edge_list[i,2])] = i # create a mapping between nodes and edges
    end

    nb1 = []
    nb2 = []
    nb_w = []

    G = Graph(A)
    neighbours = [all_neighbors(G, i) for i=1:n] # set of neighbours of each node

    for (e,u) in edge_id_map 
        i, j = src(e), dst(e) 
        for k in neighbours[i] 
            k == j && continue
            v = edge_id_map[E(k,i)]
            append!(nb1,u)
            append!(nb2,v)
            append!(nb_w, w_d_edge_list[v])
        end

    end


    B = sparse(nb1,nb2, nb_w, length(d_edge_list[:,1]),length(d_edge_list[:,1])) # create sparse adjacency matrix


    return convert(SparseMatrixCSC{Float64,Int64},B)

end


###############################################################################################################################


"""
This function generates the matrix F(λ) as detailed in Appendix A

Usage
----------

```F = F_matrix(edge_list, w_edge_list, λ, n)```


Entry
----------
* ```edge_list```   : undirected edge list containinf the non zero entries of H (Array{Int64,2})
* ```w_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```λ```           : eigenvalue of B considered (Complex{Float64})
* ```n```           : size of the matrix (Int64)

Returns
-------
```F```         : sparse representation of the matrix F(λ) (SparseMatrixCSC{Float64,Int64})
"""
function F_matrix(edge_list::Array{Int64,2}, w_edge_list::Array{Float64,1}, λ::Complex{Float64}, n::Int64)

        
    
    wd = w_edge_list.^4 ./(λ^2 .- w_edge_list.^2)
    w = λ*w_edge_list.^3 ./(λ^2 .- w_edge_list.^2)

    W = sparse(edge_list[:,1],edge_list[:,2], w, n,n)
    W = W+transpose(W)

    Λ = sparse(edge_list[:,1],edge_list[:,2], wd, n,n) 
    Λ = Λ+transpose(Λ)
    Λ = spdiagm(0 => (Λ*ones(n) .+ 1))
    
    F = Λ - W

    return F
    
end

#########################################################################################################################

"""
This function generates the matrix H(x) defined in Equation 12

Usage
----------

```H = H_matrix(edge_list, w_edge_list, x, n)```


Entry
----------
* ```edge_list```   : undirected edge list containing the non zero entries of H (Array{Int64,2})
* ```w_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```x```           : value of the parameter x considered (Float64)
* ```n```           : size of the matrix (Int64)

Returns
-------
```H```         : sparse representation of the matrix H(x) (SparseMatrixCSC{Float64,Int64})
"""
function H_matrix(edge_list::Array{Int64,2}, w_edge_list::Array{Float64,1}, x::Float64, n::Int64)

        
    
    wd = w_edge_list.^2 ./(x^2 .- w_edge_list.^2)
    w = x*w_edge_list ./(x^2 .- w_edge_list.^2)

    W = sparse(edge_list[:,1],edge_list[:,2], w, n,n)
    W = W+transpose(W)

    Λ = sparse(edge_list[:,1],edge_list[:,2], wd, n,n) 
    Λ = Λ+transpose(Λ)
    Λ = spdiagm(0 => (Λ*ones(n) .+ 1))
    
    H = Λ - W

    return H
    
end


#########################################################################################################################



"""
This function finds the value of the ferromagnetic - spin glass transition β_F. 
This function keeps into account a possibly non-homogeneous degree distribution through the parameter 
Φ = E[d^2]/E^2[d], where ``d`` indicates the degree.

Usage
----------

```β_F =  find_β_F(J_edge_list, c, Φ, ϵ)```


Entry
----------
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```c```           : average degree (Float64)
* ```Φ```           : second moment of the normalized degree distribution (Float64)
* ```ϵ```           : precision of the estimation (Float64)

Returns
-------
```β_F```         : value of the transition temperature (Float64)
"""
function find_β_F(J_edge_list::Array{Float64,1}, c::Float64, Φ::Float64, ϵ :: Float64)
    
    β_small = 0 # β_F > β_small 
    β_large = 1/c # initialization of the right edge of the interval containing β_F
    
    flag = 0

    while flag == 0
        f_now = c*Φ*mean(tanh.(β_large*J_edge_list)) - 1
        if f_now > 0
            flag = 1
        else
            β_large += 1/c # increase the value of β_large until c*Φ*mean(tanh.(β_large*J_edge_list)) - 1 > 0
        end
    end

    
    δ = 1 # find the solution to c*Φ*mean(tanh.(β_large*J_edge_list)) - 1 = 0 with the bisection method
    while δ > ϵ
        global β_new = (β_small + β_large)/2
        δ = abs(β_large - β_small)
        if c*Φ*mean(tanh.(β_new*J_edge_list)) - 1 > 0
            β_large = β_new
        else
            β_small = β_new
        end
        
    end 
    
    return β_new
end

###############################################################################################################################



"""
This function finds the value of the paramagnetic - spin glass transition β_SG. This function keeps into account a possibly non-homogeneous degree distribution through the parameter 
Φ = E[d^2]/E^2[d], where ``d`` indicates the degree.

Usage
----------

```β_SG =  find_β_SG(J_edge_list, c ,Φ, ϵ)```


Entry
----------
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```c```           : average degree (Float64)
* ```Φ```           : second moment of the normalized degree distribution (Float64)
* ```ϵ```           : precision of the estimation (Float64)

Returns
-------
```β_SG```         : value of the transition temperature (Float64)
"""
function find_β_SG(J_edge_list::Array{Float64,1}, c::Float64, Φ::Float64, ϵ ::Float64)
    
    β_small = 0 # β_SG > β_small 
    β_large  = 1/c # initialization of the right edge of the interval containing β_SG
    flag = 0

    while flag == 0
        f_now = c*Φ*mean(tanh.(β_large*J_edge_list).^2) - 1
        if f_now > 0
            flag = 1
        else
            β_large += 1/c # increase the value of β_large until c*Φ*mean(tanh.(β_large*J_edge_list).^2) - 1 > 0
        end
    end

     # find the solution to c*Φ*mean(tanh.(β_large*J_edge_list).^2) - 1 = 0 with the bisection method
    
    δ = 1 
    while δ > ϵ 
        global β_new = (β_small + β_large)/2
        δ = abs(β_large - β_small)
        if c*Φ*mean(tanh.(β_new*J_edge_list).^2) - 1 > 0
            β_large = β_new
        else
            β_small = β_new
        end
        
    end 
    
    return β_new
end



################################################################################################################################

































