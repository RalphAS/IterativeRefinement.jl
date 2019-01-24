
# Implementation of Dongarra et al., "Improving the accuracy...," SINUM 1983

# find peak index
function _normalize!(x)
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

# version for an isolated eigenvalue
"""
    refineeigen(A,x,λ,DT) => xnew, λnew

Improve the precision of a computed eigenpair `(x,\lambda)` for matrix `A`
via multi-precision iterative refinement, using more-precise real type `DT`.

The higher precision `DT` is only used for residual computation
(i.e. matrix-vector products), so this can be much faster than a full
eigensystem solution with `eltype == DT`.  This method works on a
single eigenpair, and can fail spectacularly if there is another
eigenvalue nearby.
"""
function refineeigen(A::AbstractMatrix{T},
                     x::AbstractVector{Tx},
                     λ::Tλ,
                     DT::Type{<:AbstractFloat} = Float64;
                     maxiter=5,
                     factor = lu,
                     # DT = _widen(T),
                     scale = true,
                     verbose = false
                     ) where {T,Tλ,Tx}
    Tr = promote_type(promote_type(Tx,DT),Tλ)
    res = _refineeigen(A, x, λ, Tr, factor, maxiter, scale, verbose)
    res
end
function _refineeigen(A::AbstractMatrix{T},
                     x::AbstractVector{Tx},
                     λ::Tλ,
                     DT,
                     factor = lu,
                     maxiter = 5,
                     scale = true,
                     verbose = false
                ) where {T,Tx,Tλ}

    λd = convert(DT,λ)
    n = LinearAlgebra.checksquare(A)
    Ad = DT.(A)
    B = Ad - λd * I

    s = _normalize!(x)
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
    for p = 1:maxiter
        verbose && println("iter $p resnorm: ",norm(r))
        δ = FB \ Tx.(r) # ldiv!(δ,FB,Tx.(r))
        y .= y .+ δ
        yp .= y
        δp .= δ
        ys = y[s]
        δs = δ[s]

        r .= r .- B * DT.(δ)
        δp[s] = zero(T)
        yp[s] = zero(T)
        r .= r .+ ys * δp .+ δs * y
    end
    xnew = x + yp
    λnew = λ + ys
    return xnew, λnew
end
