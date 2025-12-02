# IterativeRefinement

This package is an implementation of multi-precision iterative refinement for
certain dense-matrix linear algebra problems.

## Background
The purpose of iterative refinement (IR) is to improve the accuracy of a
solution.  If `x` is the exact solution of `A*x=b`, a simple solve of
the form `y = A \ b` will have a relative forward error
(`norm(y-x)/norm(x)`) of approximately `ϵ * O(n) * cond(A)` where `ϵ`
is the unit roundoff error in the standard precision. Iterative
refinement with higher precision residuals can reduce this to
 `ϵ * O(n)`, as long as the matrix `A` is not very badly conditioned
relative to `ϵ`.

Why not do everything in high precision? The factorization step is
typically *very* expensive (`O(n^3)`) in high precision, whereas the
residual computation is relatively cheap (`O(n^2)`). Furthermore, IR
schemes often provide useful error bounds.

For typical use, one would have a basic working precision of `Float64`
(`ϵ = 2.2e-16`), so that fast LAPACK/BLAS routines dominate the runtime.
`rfldiv` will then (by default) use `BigFloat` for residuals.
One might alternatively use `Double64` from
[DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)
or `Float128` from
[Quadmath.jl](https://github.com/JuliaMath/Quadmath.jl)

# Linear systems

This package provides a function `rfldiv`, which
handles linear matrix-vector problems of the form

`A x = b`.

## Basic Usage
```julia
julia> using LinearAlgebra, IterativeRefinement
julia> x, bnorm, bcomp = rfldiv(A,b)
```
This provides an accurate solution vector `x` and estimated bounds
on norm-wise and component-wise relative error. By default `LU` decomposition
is used.

## Advanced Usage
See the function docstring for details.

If one has several right-hand-sides, one can equilibrate and factor
`A` in advance; see the tests for an example.

## Reference
J.Demmel et al., "Error bounds from extra precise iterative refinement,"
LAPACK Working Note Nr. 165 (2005), also published as
ACM TOMS, 32, 325 (2006).  The work
described therein eventually turned into a collection of subroutines
included in some versions of LAPACK.  This implementation is based on
the paper; minor modifications were introduced based on experimentation.
To be precise, this package implements Algorithm 3.

# Eigensystems

Additional methods (`rfeigen`) are provided for improving estimates of
eigenvalue/subspace pairs of the form

`A X = X λ`.

For a simple eigenvalue, `X` is the corresponding eigenvector, and
the user provides coarse estimates of both. In the case of
multiple or defective eigenvalues, columns of `X` are generators for the
corresponding invariant subspace, and the user provides a Schur decomposition
with a list of indices for the cluster of interest.

Problem-specific error bound estimates are not yet provided for eigensystems.

## Basic Usage
### isolated eigenvalue
```julia
julia> using LinearAlgebra, IterativeRefinement, Quadmath
julia> E = eigen(A)
julia> j = your_index_selection()
julia> λrefined, xrefined = rfeigen(A, E.vectors[:,j], E.values[j], Float128)
```

### eigenvalue cluster
```julia
julia> using LinearAlgebra, IterativeRefinement, Quadmath
julia> S = schur(A)
julia> S = LinearAlgebra.Schur{ComplexF64}(S) # if eltype of A is real
julia> idx = findall(abs.(S.values .- target) .< 0.1)
julia> λrefined, Vrefined = rfeigen(A, S, idx, Float128)
```

## Reference
J.J.Dongarra, C.B.Moler, and J.H.Wilkinson, "Improving the accuracy of computed
eigenvalues and eigenvectors," SIAM J. Numer. Anal. 20, 23-45 (1983).

