using LinearAlgebra, Random
using Test
using Quadmath
using IterativeRefinement

const verbose = (get(ENV,"VERBOSITY","0") == "1")

Random.seed!(1101)

include("utils.jl")

function runone(A::Matrix{T},x0::AbstractVector) where {T}
    n = size(A,1)
    DT = widen(T)
    # println("wide type is $DT")
    Ad = DT.(A)
    xd = DT.(x0)
    b = T.(Ad * xd)
    # checkme: Demmel et al. use refined solver here
    xtrue = Ad \ DT.(b)
    xt = T.(xtrue)
    Rv, Cv = equilibrators(A)
    if maximum(abs.(Cv)) > 10
        RA = Diagonal(Rv)*A
    else
        RA = A
    end
    a = opnorm(RA,Inf)
    F = lu(RA)
    κnorm = condInfest(RA,F,a)
    RAx = RA*Diagonal(xt)
    a = opnorm(RAx,Inf)
    F = lu(RAx)
    κcomp = condInfest(RAx,F,a)
    crit = 1 / (max(sqrt(n),10) * eps(real(T)))
    if verbose
        println("problem difficulty (rel. to convergence criterion):")
        println("normwise: ", κnorm/crit, " componentwise: ", κcomp/crit)
    end
    xhat,Bnorm,Bcomp = @inferred(rfldiv(A,b))
    # xhat,Bnorm,Bcomp = rfldiv(A,b)
    Enorm = norm(xhat-xtrue,Inf)/norm(xtrue,Inf)
    Ecomp = maximum(abs.(xhat-xtrue) ./ abs.(xtrue))
    if verbose
        println("Bounds: $Bnorm $Bcomp")
        println("Errors: $Enorm $Ecomp")
    end
    if Bnorm > 0.1
        @test κcomp > 100 * crit
    else
        γ = max(10,sqrt(n))
        @test Enorm < 1.1*Bnorm
        if κnorm < crit
            @test Bnorm < γ * eps(real(T))
        end
        @test Ecomp < 1.1*Bcomp
        if κcomp < crit
            @test Bcomp < γ * eps(real(T))
        end
    end
end

# pick log10(condition-number) for various cases
function lkval(class,T)
    if class == :easy
        if real(T) <: Float32
            return 5.0
        elseif real(T) <: Float64
            return 13.0
        elseif real(T) <: Float128
            return 29.0
        end
    elseif class == :moderate
        if real(T) <: Float32
            return 7.5
        elseif real(T) <: Float64
            return 16.0
        end
    elseif class == :painful
        if real(T) <: Float32
            return 9.0
        elseif real(T) <: Float64
            return 18.0
        end
    end
    throw(ArgumentError("undefined lkval"))
end

@testset "matrix rhs $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    for n in [10]
        A = mkmat(n,lkval(:easy,T),T)
        nrhs = 3
        X = rand(T,n,nrhs)
        B = copy(X)
        X1 = copy(X)
        bn1 = zeros(T,nrhs)
        bc1 = zeros(T,nrhs)
        # check validity w/ view arg (someday maybe more tricky AbstractArrays)
        runone(A,view(X,:,2))
        for j=1:nrhs
            x,bnorm,bcomp = @inferred(rfldiv(A,view(X,:,j)))
            X1[:,j] .= x
            bn1[j] = bnorm
            bc1[j] = bcomp
        end
        X2, bn2, bc2 = @inferred(rfldiv(A,B))
        @test X1 ≈ X2
        @test bn1 ≈ bn2
        @test bc1 ≈ bc2
    end
end

@testset "preprocessed args $T" for T in (Float32, Float128)
    n = 16
    A = mkmat(n,lkval(:easy,T),T)
    # make it badly scaled
    s = 1 / sqrt(floatmax(T))
    A = s * A
    x = rand(T,n)
    b = A * x
    # basic usage for comparison
    x1, bn1, bc1 = rfldiv(A,b)

    # example of use with precomputed factor
    Rv, Cv = equilibrators(A)
    R = Diagonal(Rv)
    As = R * A * Diagonal(Cv)
    bs = R * b
    F = lu(As)
    a = opnorm(As,Inf)
    κnorm = condInfest(As,F,a)
    x2, bn2, bc2 = rfldiv(As,bs; F=F, κ = κnorm, equilibrate = false)
    cx2 = Diagonal(Cv) * x2
    @test cx2 ≈ x1
    @test bn2 ≈ bn1
    @test bc2 ≈ bc1
    # make sure this was not an empty test
    x3, bn3, bc3 = @test_logs (:warn, r"no convergence.*") rfldiv(A,b; F=F, κ = κnorm, equilibrate = false)
    @test ! (x3 ≈ x1)
end

@testset "well-conditioned $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    for n in [10,30,100]
        A = mkmat(n,lkval(:easy,T),T)
        x = rand(n)
        runone(A,x)
    end
end

@testset "marginally-conditioned $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    for n in [10,30,100]
        A = mkmat(n,lkval(:moderate,T),T)
        x = rand(n)
        runone(A,x)
    end
end

@info "The next block of tests is expected to produce warnings"

@testset "badly-conditioned $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    # We don't test for convergence failure here because
    # the method occasionally works in this regime.
    for n in [10,30,100]
        A = mkmat(n,lkval(:painful,T),T)
        x = rand(n)
        runone(A,x)
    end
end

include("eigen.jl")

