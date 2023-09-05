"""
    mkmat(n,log10κ=5,T=Float32)

construct a matrix of size `(n,n)` and eltype `T` with log-spaced
singular values from `10^(-log10κ)` to 1.
"""
function mkmat(n, log10κ, ::Type{T}; verbose=false) where {T}
    verbose && println("Matrix{$T} N=$n cond=10^$log10κ")
    if T <: Real
        q1,_ = qr(randn(n,n))
        q2,_ = qr(randn(n,n))
    else
        q1,_ = qr(randn(ComplexF64,n,n))
        q2,_ = qr(randn(ComplexF64,n,n))
    end
    DT = real(T)
    s = 10.0 .^(-shuffle(0:(n-1))*log10κ/(n-1))
    A = T.(Matrix(q1)*Diagonal(s)*Matrix(q2)')
end

# matrix with simple eigenvalues, separated by at least `dmin`
function mkmat_simple(n, dmin, ::Type{T}; verbose=false) where {T}
    if dmin > 1 / (2*n)
        throw(ArgumentError("unrealistic value of dmin, I give up."))
    end
    dmin1 = 0.0
    local ews
    while dmin1 < dmin
        ews = rand(n)
        dmin1 = minimum(abs.((ews .- ews') + I))
    end
    verbose && println("Matrix{$T} N=$n simple gap=$dmin1")
    X = rand(n,n)
    # println("cond X: ",cond(X))
    A1 = X * diagm(0 => ews) * inv(X)
    A = T.(A1)
end

function mkmat_multiple(n, k, target, dmin, ::Type{T}; verbose=false) where {T <: Real}
    if dmin > 1 / (2*(n-k))
        throw(ArgumentError("unrealistic value of dmin, I give up."))
    end
    dmin1 = 0.0
    local ews
    while dmin1 < dmin
        ews = rand(n-k)
        dmin1 = minimum(abs.((ews .- ews') + I))
    end
    verbose && println("Matrix{$T} N=$n $k-multiple for λ=$target gap=$dmin1")
    append!(ews,fill(Float64(target),k))
    X = rand(n,n)
    # println("cond X: ",cond(X))
    A1 = X * diagm(0 => ews) * inv(X)
    A = T.(A1)
end
function mkmat_multiple(n, k, target, dmin, ::Type{T}; verbose=false) where {T <: Complex}
    dmin1 = 0.0
    local ews
    while dmin1 < dmin
        ews = rand(ComplexF64,n-k)
        dmin1 = minimum(abs.((ews .- ews') + I))
    end
    verbose && println("Matrix{$T} N=$n $k-multiple for λ=$target gap=$dmin1")
    append!(ews,fill(ComplexF64(target),k))
    X = rand(ComplexF64,n,n)
    # println("cond X: ",cond(X))
    A1 = X * diagm(0 => ews) * inv(X)
    A = T.(A1)
end


"""
construct a matrix similar to one with one Jordan block of size `k`,
eigenvalue `w1` and other eigenvalues random, likely simple, in [0,1).
"""
function mkmat_defective(n, k, w1, ::Type{T}; verbose=false) where {T <: Real}
    # putting defective ones at end seems to make final location more random
    verbose && println("Matrix{$T} N=$n $k-Jordan for λ=$w1")
    ews = vcat(rand(n-k), w1 * ones(k))
    Ts = diagm(0=>ews) + diagm(1 => vcat(zeros(n-k), ones(k-1)))
    X = rand(n,n)
    # println("cond X: ",cond(X))
    A1 = X * Ts * inv(X)
    A = T.(A1)
end
function mkmat_defective(n, k, w1, ::Type{T}; verbose=false) where {T <: Complex}
    verbose && println("Matrix{$T} N=$n $k-Jordan for λ=$w1")
    ews = vcat(rand(ComplexF64, n-k), w1 * ones(ComplexF64, k))
    Ts = diagm(0=>ews) + diagm(1 => vcat(zeros(n-k), ones(k-1)))
    X = rand(ComplexF64,n,n)
    A1 = X * Ts * inv(X)
    A = T.(A1)
end

"""
construct a matrix with a cluster of eigenvalues with specified condition.
`lbdiag` specifies whether lower block is normal. (Otherwise it is likely
to have worse condition than the cluster of interest, which may be
undesirable.)
"""
function mkmat_cond(n, targets, cond, ::Type{T}; lbdiag=false, verbose=false) where T
    if (cond < 1)
        throw(ArgumentError("condition cannot be < 1"))
    end
    k = length(targets)
    verbose && println("Matrix{$T} N=$n $k-cluster w/ cond $cond")
    Tw = (T <: Real) ? Float64 : ComplexF64
    A11 = diagm(0=>Tw.(targets)) + triu(rand(Tw,k,k),1)
    ews = rand(n-k)
    if lbdiag
        A22 = diagm(0=>rand(Tw,n-k))
    else
        A22 = triu(rand(Tw,n-k,n-k))
    end
    R = rand(Tw,k,n-k)
    condr = sqrt(cond^2 - 1.0)
    lmul!(condr/opnorm(R,2),R)
    A12 = -A11 * R + R * A22
    U,_ = qr(randn(Tw,n,n))
    At = [A11 A12; zeros(Tw,n-k,k) A22]
    # norm(A12) / norm(R) might be a good estimate for sep(A11,A22)
    A = T.(U' * At * U)
end
