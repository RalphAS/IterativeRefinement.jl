# TODO: possibly add GenericSchur to deps and test T == Float64 etc.

@testset "simple eigenvalues $T" for T in (Float32, ComplexF32)
    DT = widen(T)
    maxit = 5
    tol = 20.0
    e = eps(real(T))
    dmin = 1e3 * e
    for n in [8,32]
        A = mkmat_simple(n,dmin,T)
        Ad = convert.(DT,A)
        ewd = eigvals(Ad)
        ew, ev = eigen(A)

        if verbose
            ewerrs = [minimum(abs.(ew[j] .- ewd)) for j in 1:n]
            println("initial errors ", ewerrs / (e * n))
        end
        newvecs = similar(ev)
        newews = similar(ew)
        for j=1:n
            λ, x = rfeigen(A, ev[:,j], ew[j], real(DT), maxiter=maxit)
            newvecs[:,j] .= x
            newews[j] = λ
        end
        # allow for different orderings because of roundoff
        ewerrs = [minimum(abs.(newews[j] .- ewd)) for j in 1:n]
        if verbose
            println("final errors ", ewerrs / (e * n))
        end
        @test maximum(abs.(ewerrs))  < tol * e * n * norm(A)
        # TODO: check newvecs against DP version
    end
end

@testset "(nearly) defective eigenvalues $T" for T in (Float32, ComplexF32)
    DT = widen(T)
    etarget = T(2)
    maxit = 5
    tol = 20.0
    for n in [5,10,32]
        for k in [2,3]
            @label retry
            A = mkmat_defective(n,k,etarget,T)
            # we need a true Schur here
            S = schur(A .+ 0im)
            Ad = convert.(DT,A)
            ew = eigvals(Ad)
            idx = findall(abs.(S.values .- etarget) .< 0.2)
            # don't try if A is so nonnormal that initial estimates are bad
            if length(idx) != k
                @goto retry
            end
            println("n=$n k=$k A[1,1]=",A[1,1])
            e = eps(real(T))
            if verbose
                ewerrs = [minimum(abs.(S.values[j] .- ew)) for j in idx]
                println("initial errors ", ewerrs / (e * n))
            end
            newew, newvecs = rfeigen(A, S, idx, DT, maxit)
            ewerrs = [minimum(abs.(newew[j] .- ew)) for j in 1:k]
            if verbose
                println("final errors ", ewerrs / (e * n))
            end
            @test maximum(ewerrs) / abs(etarget) < tol * e * n
        end
    end
end

@testset "multiple eigenvalues $T" for T in (Float32, ComplexF32)
    DT = widen(T)
    etarget = T(2)
    dmin = 1e3 * eps(real(T))
    maxit = 5
    tol = 20.0
    for n in [5,10,32]
        for k in [2,3]
            @label retry
            A = mkmat_multiple(n,k,etarget,dmin,T)
            # we need a true Schur here
            S = schur(A .+ 0im)
            Ad = convert.(DT,A)
            ew = eigvals(Ad)
            idx = findall(abs.(S.values .- etarget) .< 0.2)
            # don't try if A is so nonnormal that initial estimates are bad
            if length(idx) != k
                @goto retry
            end
            e = eps(real(T))
            if verbose
                ewerrs = [minimum(abs.(S.values[j] .- ew)) for j in idx]
                println("initial errors ", ewerrs / (e * n))
            end
            newew, newvecs = rfeigen(A, S, idx, DT, maxit)
            ewerrs = [minimum(abs.(newew[j] .- ew)) for j in 1:k]
            if verbose
                println("final errors ", ewerrs / (e * n))
            end
            @test maximum(ewerrs) / abs(etarget) < tol * e * n
        end
    end
end
