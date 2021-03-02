using LightGraphs, SparseArrays, LinearAlgebra, StatsBase, DataFrames, Distributions, KrylovKit, ParallelKMeans


"""
This function performs spectral clustering on a weighted graph using the Bethe Hessian matrix at the Nishimori temperature
according to Algorithm 2

Usage
----------
```estimated_ℓ = clustering_BH_Nishimori(edge_list, J_edge_list, n)```

Entry
----------
* ```edge_list```   : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```           : size of the matrix (Int64)

Optional entries
------------------
* ```ϵ```            : precision of the estimates (Float64). By default set to 2*10^(-5) 
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output. By default set to 2 (Int64)
* ```N_repeat```     : number of repetitions of the k-means algorithm. By default set to 8
* ```is_signed_th``` : if variance(th(β_SG*J_edge_list).^2) < is_signed_th, the signed representation will be used (Float64). By default set to 10^(-3)
* ```β_threshold```  : the largest value of β_N that can be computed is sqrt(c)*β_SG*β_times (Float64). By default set to 2
* ```n_max```        : maximal number of iterations admitteted to achieve convergence in the estimation of β_N (Int64). By default set to 25

Returns
-------
```estimated_ℓ```   : Array ∈ {-1,1} containing the label assignement (Array{Int64,1})
"""
function clustering_BH_Nishimori(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64; ϵ = 2*10^(-5), N_repeat = 8, is_signed_th = 10^(-3), β_threshold = 2., verbose = 1, n_max = 25)
    
    A = sparse(edge_list[:,1],edge_list[:,2], ones(length(edge_list[:,1])), n,n) # create sparse adjacency matrix
    A = A+A' # symmetrize the adjacency matrix
    d = A*ones(n) # find the degree vector
    c = mean(d) # estimate the average degree
    Φ = mean(d.^2)/c^2 # second moment of the normalized degree distribution
    
    
    β_SG =  find_β_SG(J_edge_list, c, Φ, ϵ) # find the spin-glass phase transition
    if verbose >= 1
        printstyled("o The value of β_SG is ", round(β_SG, digits = 2), ". Computing β_N\n"; color = 2)
    end
    
    # We now determine if we need to use the signed representation
    
    variance_J = sqrt(var(tanh.(β_SG*J_edge_list).^2))/β_SG # computing how far the vector ω is from being constant
    
    if variance_J < is_signed_th
        signed = true
        J_edge_list = sign.(J_edge_list)
        if verbose >= 1
            printstyled("\nThe signed representation of J is adopted. If you want to use the weighted one, increase the value of `is_signed_th`. The algorithm might have a sensible slow down\n"; color = 166)
        end
    else
        signed = false
    end
    
    # Now we compute the Nishimori temperature
    
    if signed == false
        β_N = find_β_N(edge_list, J_edge_list, n, c, β_SG, ϵ = ϵ, verbose = verbose, β_threshold = β_threshold, n_max = n_max) # compute the Nishimori temperature
        L =  L_matrix(edge_list, J_edge_list, β_N, n)
        W_Λ = sparse(edge_list[:,1],edge_list[:,2], (sinh.(β_N*J_edge_list)).^2, n,n)
        W_Λ = W_Λ+W_Λ'
        Λ = spdiagm(0 => (W_Λ*ones(n) .+ 1).^(-1/2))
        s, X = eigsolve(L, 1, :SR, ishermitian = true) # compute the smallest eiegnpair
        X = [Λ*X[1]] # we recover the eigenvector of the Bethe-Hessian
    else
        β_N = find_β_N_signed(edge_list, J_edge_list, n, c, β_SG, ϵ = ϵ, verbose = verbose) # compute the Nishimori temperature
        r = 1/tanh(β_N)
        H = H_matrix_signed(edge_list, J_edge_list, r, n)
        s, X = eigsolve(H, 1, :SR, ishermitian = true) # compute the smallest eiegnpair
           
    end
    
    if verbose >= 1
        printstyled("\no The value of β_N is ", round(β_N, digits = 2); color = 2)
    end
        

    if verbose >= 1
        printstyled("\no Running kmeans\n"; color = 2)
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


######################################################################################################



"""
This function estimates the Nishimori temperature on a random graph, according to Algorithm 1

Usage
----------

```β_N = find_β_N(edge_list, J_edge_list, n, c, β_SG)```


Entry
----------
* ```edge_list```    : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list```  : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```            : size of the matrix (Int64)
* ```c```            : average degree of the graph (Float64)
* ```β_SG```         : value of β_SG, the spin-glass phase transition temperature (Float64)

Optional entries
----------
* ```n_max```        : maximal number of iterations run by the algorithm (Int64). By default set to 25 
* ```ϵ```            : precision of the estimates (Float64). By default set to 2*10^(-5) 
* ```β_threshold```  : The largest value of β_N that can be computed is sqrt(c)*β_SG*β_threshold (Float64). By default set to 2.
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output. By default set to 2 (Int64)

Returns
-------
```β_N```            : estimate of the Nishimori temperature (Float64)
"""
function find_β_N(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64, c::Float64, β_SG::Float64; n_max = 25, ϵ = 2*10^(-5), β_threshold = 2., verbose = 1)
    
    flag = 0 # the algorithm runs as  long as this flag is equal to zero
    counter = 1
    β_old = 2*β_SG # initialization of β_old to enter the loop the first time

    # First of all we check that β_N > β_SG, so that at β_SG there is at least a negative eigenvalue in the spectrum of L
    
    L = L_matrix(edge_list, J_edge_list, β_SG, n) # build the matrix L as described in Appendix B
    s, X = eigsolve(L, 1, :SR, ishermitian = true) # compute the smallest eigenvalue of L
    x = X[1] # take the associated eigenvector
    
    """ If the smallest eigenvalue of L computed at β = β_SG is not negative, then the detectability condition 
    β_F < β_SG < β_N is not met and the algorithm is stopped"""

    if s[1] > 0 
        flag = 1 # raise the flag
        if verbose >= 1 
            printstyled("The Nishimori temperature cannot be estimated on this matrix\n" ; color = 9)
        end
        return β_SG
        
    end
    
    """ If β_N can be estimated, we proceed """
    
    
    β = β_SG # we initialize the value  of  β
    
    # If β_N can be estimated, we begin the iterations
    
    while flag == 0
          
        
        if counter > 1 # we already diagonalized the matrix at β_SG
        
            L = L_matrix(edge_list, J_edge_list, β, n) # build the matrix L
            s, X = eigsolve(L, 1, :SR, ishermitian = true) # compute the smallest eigenvalue
            x = X[1] # take the associated eigenvector
            
        end
            
        ### Here we write the conditions that allow to exit from the loop

        flag = exitFromLoop(counter, n_max, verbose, s, c, β, J_edge_list, ϵ, β_old, β_threshold, β_SG)
        
        
        ### If the flag has not been raised, we continue the iterations
        
        
        if flag == 0
        
            β_large = find_β_large(β, β_SG, edge_list, J_edge_list, n, x) # estimate a value of β so that x'*L*x is positive
            β_old = β
            β = find_zero(β, β_large, 10^(-12), c, edge_list, J_edge_list, n, x) # find the zero of the function x'*L*x and update β
            
        end
            
        if verbose >= 2
            printstyled("\nIteration # ", counter, ": ",
                "\nThe current estimate of β_N is ", β,
                "\nThe smallest eigenvalue is ", s[1]*c*(1+mean(sinh.(β_SG*J_edge_list).^2)), "\n" 
                ; color = 4)
            
        end
        
        counter += 1 # update the counter
        
        
    end

    
    return β
    
end


##############################################################################################################

"""
This function generates the Bethe-Hessian inspired regularized Laplacian matrix corresponding to a given weighted edge list,
described in Appendix B

Usage
----------

```L =  L_matrix(edge_list, J_edge_list, β, n)```


Entry
----------
* ```edge_list```   : undirected edge list containinf the non zero entries of H (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```β```           : value of β (Float64)
* ```n```           : size of the matrix (Int64)

Returns
-------
```L```         : sparse weighted matrix L (SparseMatrixCSC{Float64,Int64})
"""
function L_matrix(edge_list::Array{Int64,2}, w_edge_list::Array{Float64,1}, β::Float64, n::Int64)

        
    
    w = 1/2*sinh.(2*β*w_edge_list)
    w2 = (sinh.(β*w_edge_list)).^2

    W = sparse(edge_list[:,1],edge_list[:,2], w, n,n)
    W = W+W'

    Λ = sparse(edge_list[:,1],edge_list[:,2], w2, n,n) 
    Λ = Λ+Λ'
    Λ_05 = spdiagm(0 => (Λ*ones(n) .+ 1).^(-1/2))
    
    Id = spdiagm(0 => ones(n))
    
    L = Id - Λ_05*W*Λ_05

    return L
    
end


##############################################################################################################


"""
This function computes a value of β so that x'*L*x is positive

Usage
----------

```β_large =  find_β_large(β_small, β_SG, edge_list, J_edge_list, n, x)```


Entry
----------
*```β_small```      : left edge of the interval containing the zero of the function x'*L*x (Float64)
*```β_SG```         : spin-glass phase transition temperature (Float64)
* ```edge_list```   : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```x```           : vector x (Array{Float64,1})


Returns
-------
```β_large```       : right edge of the interval containing the zero of the function x'*L*x (Float64)
"""
function find_β_large(β_small::Float64, β_SG::Float64, edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64, x)
    
    
    flag = 0
    β_large = β_small
    
    while flag == 0
    
        β_large = β_large + β_SG
        L = L_matrix(edge_list, J_edge_list, β_large, n) # compute the matrix L for the new value of β
        f = x'*L*x
        if f > 0
            flag = 1
        end
        
    end
    
    return β_large
    
end


##############################################################################################################


"""
This function estimates the value of β so that x'*L*x = 0

Usage
----------

```β =  find_zero(β_small, β_large, ϵ, edge_list, J_edge_list, n, x)```


Entry
----------
* ```β_small```      : left edge of the interval containing the zero
* ```β_large```      : right edge of the interval containing the zero
* ```ϵ```            : precision of the estimation
* ```edge_list```    : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list```  : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```            : size of the matrix (Int64)
* ```x```            : vector x (Array{Float64,1})


Returns
-------
```β```             : estimate of zero of the function x'*L*x (Float64)
"""
function find_zero(β_small::Float64, β_large::Float64, ϵ::Float64, c::Float64, edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64, x::Array{Float64,1})
    
    δ = 1 # initialize the error

    while δ > ϵ

        β = (β_large + β_small)/2 # use the bisection method to find the zero
        L = L_matrix(edge_list, J_edge_list, β, n) # compute the matrix L for the new value of β
        y = x'*L*x # using Courant-Fischer thoerem
        f = y*c*(1+mean(sinh.(β*J_edge_list).^2))

        if f > 0 # update the value of β according to the value of f
            β_large = β
        else
            β_small = β
        end

        δ = minimum([abs(f), β_large - β_small]) # update the error
        
    end
    
    return β_small
    
end


##################################################################################################################



"""
This function checks the conditions to exit from the loop to estimate β_N in the weighted case

Usage
----------

```flag = exitFromLoop(counter, n_max, verbose, s, c, β, J_edge_list, ϵ, β_old, β_threshold)```


Entry
----------
* ```counter```      : number of iterations run by the algorithm (Int64)
* ```n_max```        : maximal number of iterations allowed
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output
* ```s```            : smallest converged eigenvalues of L ({Int64,1})
* ```c```            : average degree of the graph (Float64)
* ```β```            : value of the temperature (Float64)
* ```J_edge_list```  : weights associated to the edge list ({Float64, 1})
* ```ϵ```            : precision error (Float64)
* ```β_old```        : last estimate of β_N (Float64)
* ```β_threshold```  : The largest value of β_N that can be computed is sqrt(c)*β_SG*β_threshold (Float64).
* ```β_SG```         : spin-glass phase transition temperature (Float64)

Returns
-------
```flag```           : boolean (0,1). If flag = 1 the algorithm will exit from the loop
"""
function exitFromLoop(counter::Int64, n_max::Int64, verbose::Int64, s::Array{Float64,1}, c::Float64, β::Float64, J_edge_list::Array{Float64,1}, ϵ::Float64, β_old::Float64, β_threshold::Float64, β_SG::Float64)
    
    flag = 0

    #################

    if counter > n_max # here we get out of the loop because we reached the maximal number of iterations
            
        if verbose >= 1
            printstyled("\nMaximal number of iterations reached. The algorithm is stopped. If you want to obtain a higher precision, increase the maximal number of iterations allowed\n"; color = 166)
        end
        flag = 1
        
    end
    
    #################
    
    if s[1] > 0
            
        flag = 1 # if the smallest eigenvalue of L for the last computed value of β is positive (but it was 
                           # negative at the first iteration), then we get out of the loop
    end
    
    
    #################
    
    if abs(s[1]*(1+c*mean(sinh.(β_SG*J_edge_list).^2))) < ϵ # if the smallest eigenvalue of L is sufficiently 
                                                                    # close tozero, then we get out of the  loop
        flag = 1
        
    end
    
    
    #######################
    
    if abs(β_old - β) < ϵ/4
            
        if verbose >= 1
            printstyled("\nThe variation in the estimates of β is below ϵ/4. The algorithm is stopped because it is unlikely to converge. Increase the value of ϵ to avoid this situation\n"; color = 166)
        end
        flag = 1
    end
    
    #########################

    if β > β_threshold*sqrt(c)*β_SG

        if verbose >= 1
        printstyled("\nThe estimated β is too large: this means that you are in a scenario where Nishimori Bethe Hessuian becomes numerically unstable but where the problem is sufficiently easy for a mean field approximation. The estimation of β_N is interrupted.\n"; color = 166)
        end
        flag = 1
    end
    
    return flag
    
end



##############################################################################################################



"""
This function estimates the Nishimori temperature on a signed graph

Usage
----------

```β_N = find_β_N_signed(edge_list, J_edge_list, n, c, β_SG)```


Entry
----------
* ```edge_list```   : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```           : size of the matrix (Int64)
* ```c```           : average degree of the graph (Float64)
* ```β_SG```        : value of β_SG, the spin-glass phase transition temperature (Float64)

Optional entries
------------------
* ```n_max```       : maximal number of iterations run by the algorithm (Int64). By default set to 25 
* ```ϵ```           : precision of the estimates (Float64). By default set to 2*10^(-5) 
* ```verbose```     : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output. By default set to 2 (Int64)

Returns
-------
```β_N```            : estimate of the Nishimori temperature (Float64)
"""
function find_β_N_signed(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64, c::Float64, β_SG::Float64; n_max = 25, ϵ = 2*10^(-5), verbose = 2)

    flag = 0 # the algorithm runs as long as this flag is equal to zero
    counter = 1
    r_old = 1/tanh(2*β_SG) # initialization for the first time we enter the loop
    
    r = 1/tanh(β_SG) # we initialize the value of r
    
    
    H = H_matrix_signed(edge_list, J_edge_list, r, n) # build the matrix H
    s, X = eigsolve(H, 1, :SR, ishermitian = true) # compute the smallest eiegnpair
    x = X[1] 
    
    

    if s[1] > 0
        flag = 1 # raise the flag
        if verbose >= 1 
            printstyled("The Nishimori temperature cannot be estimated on this matrix\n" ; color = 9)
        end
        return β_SG
        
    end
    
    
    # If β_N can be estimated, we begin the iterations
    
    while flag == 0
          
        # First of all we check that β_N > β_SG, so that at β_SG there is at least a negative eigenvalue in the spectrum of L
    
        if counter > 1 # we already diagonalized the matrix at β_SG
        
            H = H_matrix_signed(edge_list, J_edge_list, r, n) # build the matrix H
            s, X = eigsolve(H, 1, :SR, ishermitian = true) # compute the smallest eiegnpair
            x = X[1] 
            
        end
            
        # Here we write the condition to exit the loop
        
        flag =  exitFromLoopSigned(counter, n_max, verbose, s, r_old, r, ϵ)
         
        # If the flag has not been raised, we continue the iterations
        
        
        if flag == 0
        
            r_old = r # store the last value of r
            r = find_zero_signed(r, 10^(-12), c, edge_list, J_edge_list, n, x) # find the zero of the function x'*H*r
            
        end
        
        if verbose >= 2
            printstyled("\nIteration # ", counter, ": ",
                "\nThe current estimate of β_N is ", atanh(1/r),
                "\nThe smallest eigenvalue is ", s[1], "\n"
                ; color = 4)
        end
        
        counter += 1 # update the counter
        
    end
    
    return atanh(1/r)
        
end


###########################################################################################################



"""
This function generates the signed Bethe-Hessian matrix of Equation 31. The parameter r = 1/th(β|J|)

Usage
----------

```H = H_matrix_signed(edge_list, J_edge_list, r, n)```


Entry
----------
* ```edge_list```   : undirected edge list containinf the non zero entries of H (Array{Int64,2})
* ```J_edge_list``` : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```r```           : value of r (Float64)
* ```n```           : size of the matrix (Int64)

Returns
-------
```H```         : sparse weighted matrix H (SparseMatrixCSC{Float64,Int64})
"""
function H_matrix_signed(edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, r::Float64, n::Int64)

        
    J = sparse(edge_list[:,1],edge_list[:,2], J_edge_list, n,n) # build the matrix J
    J = J+J'

    D = spdiagm(0 => abs.(J)*ones(n)) # degree matrix
    Id = spdiagm(0 => ones(n)) # identity matrix
    
    H = (r^2-1)*Id + D - r*J

    return H
    
end


################################################################################################################



"""
This function estimates the value of r so that x'*H*x = 0 in the case H is signed

Usage
----------

```r =  find_zero_signed(r_large, ϵ, c, edge_list, J_edge_list, n, x)```


Entry
----------
* ```r_large```      : right edge of the interval containing the zero
* ```ϵ```            : precision of the estimation
* ```c```            : average degree (Float64)
* ```edge_list```    : undirected edge list containing the non zero entries of J (Array{Int64,2})
* ```J_edge_list```  : weight corresponding to each edge contained in ```edge_list``` (Array{Float64,1})
* ```n```            : size of the matrix (Int64)
* ```x```            : vector x (Array{Float64,1})


Returns
-------
```r```             : estimate of zero of the function x'*H*x (Float64)
"""
function find_zero_signed(r_large::Float64, ϵ::Float64, c::Float64, edge_list::Array{Int64,2}, J_edge_list::Array{Float64,1}, n::Int64, x::Array{Float64,1})
    
    δ = 1 # initialize the error

    A = sparse(edge_list[:,1],edge_list[:,2], J_edge_list, n,n) # build the adjacency matrix 
    A = A+A'
    
    D = spdiagm(0 => abs.(A)*ones(n)) # build the degree matrix
    
    a = x'*A*x # compute the projection of A over the vector x
    d = x'*D*x # compute the projection of D over the vector x
    
    r_small = 1 # initialization of the left edge of the interval
    
    while δ > ϵ

        r = (r_large + r_small)/2 # use the bisection method to find the zero
        f = r^2 - 1 + d - r*a

        if f > 0 # update the value of r according to the value of f
            r_small = r
        else
            r_large = r
        end

        δ = minimum([abs(f), r_large - r_small]) # update the error
        
    end
    
    return r_large
    
end


###################################################################################################

"""
This function checks the conditions to exit from the loop to estimate β_N in the signed case case

Usage
----------

```flag =  exitFromLoopSigned(counter, n_max, verbose, s, r_old, r, ϵ)```


Entry
----------
* ```counter```      : number of iterations run by the algorithm (Int64)
* ```n_max```        : maximal number of iterations allowed
* ```verbose```      : equal to 0,1,2 sets an increasing level of verbosity. 0 indicates no output
* ```s```            : smallest converged eigenvalues of L ({Int64,1})
* ```r_old```        : last estimate of r_N (Float64)
* ```r```            : current value of r (Float64)
* ```ϵ```            : precision error (Float64)


Returns
-------
```flag```           : boolean (0,1). If flag = 1 the algorithm will exit from the loop
"""
function exitFromLoopSigned(counter::Int64, n_max::Int64, verbose::Int64, s::Array{Float64,1}, r_old::Float64, r::Float64, ϵ::Float64)
    
    flag = 0
    
    if counter > n_max # here we get out of the loop because we reached the maximal number of iterations
            
        if verbose >= 1
            printstyled("\nMaximal number of iterations reached. The algorithm is stopped. If you want to obtain a higher precision, increase the maximal number of iterations allowed\n"; color = 166)
        end
        flag = 1
        
    end
    
    #################

    if s[1] > 0

        flag = 1 # if the smallest eigenvalue of L for the last computed value of β is positive (but it was 
                       # negative at the first iteration), then we get out of the loop
        
    end
    
    ###################

    if abs(s[1]) < ϵ # if the smallest eigenvalue of H is sufficiently 
                        # close to zero, then we get out of the  loop
        flag = 1
        
    end
    
    ###################

    if abs(r_old - r) < ϵ/4

        if verbose >= 1
            printstyled("\nThe variation in the estimates of β is below ϵ/4. The algorithm is stopped because it is unlikely to converge. Increase the value of ϵ to avoid this situation"; color = 166)
        end
        flag = 1
    end
    
    return flag
        
end
