
# Implementation of Dongarra et al., "Improving the accuracy...," SINUM 1983

# find peak index
function _normalizeInf!(x)
    n = length(x)
    s=1
    xm = abs(x[1])
    for j=1:n
        t = abs(x[j])
        if t > xm
            xm = t
            s = j
        end
    end
    x ./= xm
    s
end

# simple version for an isolated eigenvalue
"""
    rfeigen(A,x,λ,DT) => λnew, xnew, status

Improve the precision of a computed eigenpair `(x,λ)` for matrix `A`
via multi-precision iterative refinement, using more-precise real type `DT`.

The higher precision `DT` is only used for residual computation
(i.e. matrix-vector products), so this can be much faster than a full
eigensystem solution with precise eltype.  This method works on a
single eigenpair, and can fail spectacularly if there is another
eigenvalue nearby.
"""
function rfeigen(A::AbstractMatrix{T},
                 x::AbstractVector{Tx},
                 λ::Tλ,
                 DT::Type{<:AbstractFloat} = widen(real(T));
                 maxiter=5,
                 tol=1,
                 factor = lu,
                 scale = true,
                 verbose = false
                 ) where {T,Tλ,Tx}
    Tr = promote_type(promote_type(Tx,DT),Tλ)
    res = _rfeigen(A, x, λ, Tr, factor, maxiter, tol, scale, verbose)
    return res
end

"""
    rfeigen(A,λ,DT) => λnew, xnew, status

Like `rfeigen(A,x,λ,DT)`, but initialize `x` via one step of inverse
iteration.
"""
function rfeigen(A::AbstractMatrix{T},
                 λ::Tλ,
                 DT::Type{<:AbstractFloat} = widen(real(T));
                 maxiter=5,
                 tol=1,
                 factor = lu,
                 scale = true,
                 verbose = false
                 ) where {T,Tλ}
    # CHECKME: is this condition adequate?
    if issymmetric(A) && (Tλ <: Real)
        Tx = Tλ
    else
        Tx = complex(Tλ)
    end
    # There may not be a sampler for types of interest (hello, Quadmath)
    # so let's promote.
    x = normalize!((A - λ * I) \ Tx.(rand(size(A,1))))
    Tr = promote_type(promote_type(Tx,DT))
    res = _rfeigen(A, x, λ, Tr, factor, maxiter, tol, scale, verbose)
    return res
end

function _rfeigen(A::AbstractMatrix{T},
                      x::AbstractVector{Tx},
                      λ::Tλ,
                      ::Type{DT},
                      factor,
                      maxiter,
                      tol,
                      scale,
                      verbose
                      ) where {T,Tx,Tλ,DT}

    status = :unfinished
    λd = convert(DT,λ)
    n = LinearAlgebra.checksquare(A)
    tol1 = tol * eps(real(DT)) # CHECKME: factor of n?
    Ad = DT.(A)
    B = Ad - λd * I

    s = _normalizeInf!(x)
    xd = DT.(x)
    B[:,s] .= -xd
    Btmp = A - λ * I # do it over to get the type right
    Btmp[:,s] .= -x
    FB = factor(Btmp)

    # initial residual
    r::Vector{DT} = λd * xd - Ad * xd
    y = zeros(Tx,n)
    ys = zero(Tx)
    δ = similar(y)
    δp = similar(y)
    yp = similar(y)
    rt = similar(y)
    prevnorm = convert(real(DT),Inf)
    for p = 1:maxiter
        verbose && println("iter $p resnorm: ",norm(r))
        δ = FB \ Tx.(r) # ldiv!(δ,FB,Tx.(r))
        y .= y .+ δ
        δnorm = norm(δ)
        ynorm = norm(y)
        yp .= y
        ys = y[s]
        yp[s] = zero(T)
        if δnorm > prevnorm
            if δnorm > 10.0 * prevnorm
                status = :diverging
            else
                status = :stalled
            end
            verbose && println("$status at iter $p; early exit")
            break
        end
        prevnorm = δnorm
        if δnorm / ynorm < tol1
            status = :converged
            verbose && println("converged")
            # println("iter $p ratio ",δnorm / ynorm)
            break
        end
        δp .= δ
        δs = δ[s]

        r .= r .- B * DT.(δ)
        δp[s] = zero(T)
        r .= r .+ ys * δp .+ δs * y
    end
    xnew = x + yp
    λnew = λ + ys
    return λnew, xnew, status
end

"""
    rfeigen(A, S::Schur, idxλ, DT, maxiter=5) -> vals, vecs, status

Improves the precision of a cluster of eigenvalues of matrix `A`
via multi-precision iterative refinement, using more-precise real type `DT`.
Returns improved estimates of eigenvalues and vectors generating
the corresponding invariant subspace.

This method works on the set of eigenvalues in `S.values` indexed by `idxλ`.
It is designed to handle (nearly) defective cases, but will fail if
the matrix is extremely non-normal or the initial estimates are poor.
Note that `S` must be a true Schur decomposition, not a "real Schur".
"""
function rfeigen(A::AbstractMatrix{T}, S::Schur{TS}, idxλ,
                 DT = widen(real(T)), maxiter=5;
                 tol = 1, verbose = false) where {T, TS <: Complex}
    n = size(A,1)
    m = length(idxλ)
    λ = [S.values[idxλ[i]] for i in 1:m]
    Tw = promote_type(T,eltype(λ))
    DTw = promote_type(DT,Tw)
    tol1 = tol * eps(real(DT)) # CHECKME: factor of n?
    status = :unfinished

    # compute X, M
    # X is an adequately-conditioned set spanning the invariant subspace
    # M is an upper-triangular matrix of mixing coefficients

    # Most of the work is in the space of Schur vectors
    Z = zeros(Tw, n, m)
    z = zeros(Tw, n)
    idxz = Vector{Int}(undef, m)

    k = idxλ[1]
    if k==1
        z[1] = one(Tw)
    else
        x0 = (S.T[1:k-1,1:k-1] - λ[1] * I) \ S.T[1:k-1,k]
        z[1:k-1] .= -x0
        z[k] = one(Tw)
    end
    zm,zi = findmax(abs.(z))
    z ./= zm
    idxz[1] = zi
    Z[:,1] .= z
    M = zeros(Tw, m, m)
    M[1,1] = λ[1]
    for l=2:m
        kp = k
        k = idxλ[l]
        @assert k > kp
        x0 = (S.T[1:k-1,1:k-1] - λ[l]*I) \ S.T[1:k-1,k]
        X1 = (S.T[1:k-1,1:k-1] - λ[l]*I) \ Z[1:k-1,1:l-1]
        z[1:k-1] .= -x0
        z[k] = one(Tw)
        # pick mixing coeffts so that each vector has a good dominant index
        rhs = [-z[idxz[i]] for i=1:l-1]
        mtmp = [X1[idxz[i],j] for i=1:l-1,j=1:l-1]
        mv = mtmp \ rhs
        M[l,l] = λ[l]
        M[1:l-1,l] .= mv
        z[1:k-1] .= z[1:k-1] .+ X1 * mv
        zm, zi = findmax(abs.(z))
        idxz[l] = zi
        Z[:,l] .= z
    end
    X = S.Z * Z

    s = Int[]
    for j=1:m
        xm = zero(real(Tw))
        xi = 0
        for i=1:n
            xt = abs(X[i,j])
            if xt > xm && i ∉ s
                xm = xt
                xi = i
            end
        end
        push!(s, xi)
    end

    λd = DTw.(λ)
    Ad = DTw.(A)
    Xd = DTw.(X)
    Md = DTw.(M)
    # TODO: if cluster is tight enough, only need a singleton B
    # How tight is tight enough?
    B = Vector{Matrix{DTw}}(undef, m)
    for j=1:m
        B[j] = Ad - λd[j] * I
    end
    for j=1:m
        for i=1:m
            B[j][:,s[i]] .= -Xd[:,i]
        end
    end

    r = zeros(DTw, n, m)
    for j=1:m
        r[:,j] = λd[j] * Xd[:,j] - Ad * Xd[:,j]
        for i=1:j-1
            r[:,j] .= r[:,j] .+ M[i,j] * Xd[:,i]
        end
    end
    verbose && println("at iter 0 res norm = ", norm(r))
    FB0 = lu(Tw.(B[1]))
    FB = Vector{typeof(FB0)}(undef, m)
    FB[1] = FB0
    for j=1:m
        FB[j] = lu(Tw.(B[j]))
    end

    y = zeros(Tw, n, m)
    yp = zeros(Tw, n, m)
    ys = zeros(Tw, m, m)
    δp = zeros(Tw, n, m)
    δs = zeros(Tw, m, m)
    prevnorm = convert(real(DT),Inf)
    for p=1:maxiter
        δnorm = zero(real(DT))
        for j=1:m
            rhs = Tw.(r[:,j])
            # for jj=1:j
            #    rhs -= M[jj,j] * yp[:,jj]
            # end
            δ = FB[j] \ rhs
            δnorm += norm(δ)
            y[:,j] .+= δ
            yp[:,j] .= y[:,j]
            δp[:,j] .= δ
            for i=1:m
                δs[i,j] = δ[s[i]]
                δp[s[i],j] = zero(Tw)
                yp[s[i],j] = zero(Tw)
            end
            r[:,j] .= (r[:,j] - B[j] * DTw.(δ))
        end
        # there are occasional strange transients
        if (p > 3) && (δnorm > prevnorm)
            if δnorm > 10.0 * prevnorm
                status = :diverging
            else
                status = :stalled
            end
            verbose && println("$status at iter $p; early exit")
            break
        end
        prevnorm = δnorm
        for j=1:m
            for jj=1:m
                r[:,j] .= r[:,j] + (DTw(ys[jj,j]) * DTw.(δp[:,jj])
                                    + DTw(δs[jj,j]) * DTw.(yp[:,jj]))
            end
            for jj=1:j-1
                r[:,j] .= r[:,j] + Md[jj,j] * DTw.(δp[:,jj])
            end
        end
        # this update is done late to avoid doubly adding δ δ terms
        for j=1:m
            for i=1:m
                ys[i,j] = y[s[i],j]
            end
        end
        verbose && println("at iter $p res norm = ", norm(r))
        # println("ew: ",eigvals(Md + DTw.(ys)))
        verbose && println("DP subspace error: ",
                norm((Xd + DTw.(yp))*(Md+DTw.(ys)) - Ad * (Xd + DTw.(yp))))
        ynorm = norm(y)
        if δnorm / ynorm < tol1
            status = :converged
            verbose && println("converged")
            # println("iter $p ratio ",δnorm / ynorm)
            break
        end
    end
    Xnew = X + yp
    verbose && println("final subspace error norm: ", norm(Xnew*(M+ys) - A * Xnew))
    λbar = (1/m)*sum(λ)
    Mnew = Tw.(Md + DTw.(ys) - DTw(λbar) * I)
    dλ = eigvals(Mnew)
    λnew = λbar .+ dλ
    λnew, Xnew, status
end
