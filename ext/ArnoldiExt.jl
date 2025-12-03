module ArnoldiExt

using LinearAlgebra
using IterativeRefinement
const IR = IterativeRefinement
using ArnoldiMethod

"""
    rfeigen(A, S::partialschur, idxλ, DT, maxiter=5) -> vals, vecs, status

Uses a partial Schur decomposition from ArnoldiMethod for cluster refinement.
"""
function IR.rfeigen(A::AbstractMatrix{T}, ps::ArnoldiMethod.PartialSchur{TQ,TR}, idxλ,
                    DT = widen(real(T)), maxiter=5; kwargs...
) where {T, TQ, TR <: AbstractMatrix{TRe}} where {TRe <: Complex}
    S = LinearAlgebra.Schur(Matrix(ps.R), Matrix(ps.Q), ps.eigenvalues)
    IterativeRefinement.rfeigen(A, S, idxλ, DT, maxiter; kwargs...)
end

function IR.rfeigen(A::AbstractMatrix{T}, ps::ArnoldiMethod.PartialSchur{TQ,TR}, idxλ,
                    DT = widen(real(T)), maxiter=5; kwargs...
) where {T, TQ, TR <: AbstractMatrix{TRe}} where {TRe <: Real}
    Sr = LinearAlgebra.Schur(Matrix(ps.R), Matrix(ps.Q), ps.eigenvalues)
    S = LinearAlgebra.Schur{complex(TRe)}(Sr)
    IterativeRefinement.rfeigen(A, S, idxλ, DT, maxiter; kwargs...)
end

end
