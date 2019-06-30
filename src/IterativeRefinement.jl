module IterativeRefinement
# this file is part of IterativeRefinement.jl, released under the MIT Expat license.

using LinearAlgebra

export rfldiv, equilibrators, condInfest, rfeigen

include("infra.jl")

# Algorithm 3 from
# J.Demmel et al., "Error bounds from extra precise iterative refinement",
# LAPACK Working Note Nr. 165 (2005), also published as
# ACM TOMS, 32, 325 (2006) (henceforth "the paper").

"""
    rfldiv(A,b,f=lu; kwargs...) -> x,bnorm,bcomp,flags

Compute an accurate solution to a linear system ``A x = b`` using
extra-precise iterative refinement, with error bounds.

Returns solution `x`, a normwise relative forward error estimate `bnorm`,
and maximum componentwise relative error estimate `bcomp`.
Specifically,  `bnorm` is an estimate of  ``‖xtrue - x‖ / ‖x‖`` (max norms).
If the problem is so ill-conditioned that a good solution is unrealizable,
`bnorm` and `bcomp` are set to unity (unless `expert`).
`flags` contains convergence diagnostics potentially interesting to
specialists.

# Arguments
- `A`: a matrix,
- `b`: a vector with the same `eltype`,
- `f`: a factorization function such as `lu`.

## Keywords

- `DT`: higher-precision type for refinement; defaults to `widen(eltype(A))`
- `verbosity`: 0(default): quiet, 1: report on iterations, 2: details.
- `equilibrate::Bool`: whether the function should equilibrate `A`
  (default `true`).
- `maxiter`: default 20.
- `tol`: relative tolerance for convergence, in units of `eps(T)`.
- `expert::Bool`: whether to return questionable bounds in extreme cases.
- `κ`: the (max-norm) condition of `A` (see below).
- `F`: a factorization of `A` (see below).

If `A` has already been equilibrated, and a `Factorization` object `F`
and condition estimate `κ` have already been computed, they may be
provided as keyword arguments; no check for consistency is done here.

Uses the algorithm of Demmel et al. ACM TOMS, 32, 325 (2006).
"""
function rfldiv(A::AbstractMatrix{T},
                b::AbstractVecOrMat{T},
                factor = lu;
                DT = widen(T),
                maxiter=20, tol=max(10.0,sqrt(size(A,1))),
                equilibrate = true,
                verbosity = 0,
                ρthresh = 0.5, # "what Wilkinson used"
                expert = false,
                κ = -one(real(T)),
                F::Union{Nothing, Factorization} = nothing
                ) where {T}
    RT = real(T)
    rfldiv_(A,b,lu,DT,RT,maxiter,tol,equilibrate,verbosity,ρthresh,expert,κ,F)
end
function rfldiv_(A::AbstractMatrix{T},
                 b::AbstractVecOrMat{T},
                 factor,
                 ::Type{DT},
                 ::Type{RT},
                 maxiter, tol, equilibrate, verbosity, ρthresh, expert, κ, F
                ) where {T, DT, RT}
    # maxiter is ithresh in paper
    # tol is γ in paper

    m,n = size(A,1), size(A,2)
    if size(b,1) != m
        throw(DimensionMismatch("first dimension of A, $n, does not match that of b, $(size(b,1))"))
    end
    nrhs = size(b,2)

    cvtok = true

    # the use of this variable in closures subverts inference, as of v1.1
    ϵw::RT = RT(2)^(-precision(RT)) # typically eps(T) / 2
    tol1 = 1 / (tol * ϵw) # $1/γϵ_w$ in the paper

    if equilibrate
        Rv,Cv = equilibrators(A)
        cnorm = maximum(abs.(Cv))
        equil = cnorm > 10
    else
        equil = false
    end
    if equil
        (verbosity > 1) && println("equilibrating, ‖C‖=$cnorm")
        C = Diagonal(Cv)
        R = Diagonal(Rv)
        As = R * A * C
    else
        C = I
        As = A
    end

    local Asd
    try
        Asd = DT.(As)
    catch
        cvtok = false
    end
    cvtok || throw(ArgumentError("unable to convert to "
                                 * "designated wide type $DT"))

    if F === nothing
        Fs = factor(As)
    else
        Fs = F
    end
    if κ < 0
        anorm = opnorm(As, Inf)
        κs = condInfest(As,Fs,anorm)
        if verbosity > 1
            equil && print("equilibrated ")
            println("norm: $anorm condition: $κs; compare to $tol1")
        end
    else
        κs = κ
    end

    dzthresh = 1/4 # "works well for binary arithmetic"

    nsingle = 1
    ndouble = 0
    relnormx = relnormz = RT(Inf)
    dxnorm = dznorm = RT(Inf)
    ρmax_x = ρmax_z = zero(RT)
    xstate = :working
    zstate = :unstable
    yscheme = :single
    incrprec = false

    if nrhs > 1
        X = zeros(T,n,nrhs)
        normwisebounds = zeros(RT,nrhs)
        termwisebounds = zeros(RT,nrhs)
        flagss = zeros(Int,3,nrhs)
    end

    function newxstate(state, xnorm, dxnorm, dxprev)
        curnorm = relnormx
        dxratio = dxnorm / dxprev
        dxrel = dxnorm / xnorm
        if (state == :noprogress) && (dxratio <= ρthresh)
            state = :working
        end
        if state == :working
            if dxrel <= ϵw
                # tiny dx, criterion (18) in paper
                state = :converged
                (verbosity > 1) && println("convergence (in norm)")
            elseif dxratio > ρthresh
                if yscheme == :single
                    (verbosity > 1) && println("increasing precision(x)")
                    incrprec = true
                elseif ndouble > 1
                    # lack of progress, criterion (17) in paper
                    state = :noprogress
                    (verbosity > 1) && println("x stalled")
                end
            else
                ρmax_x = max( ρmax_x, dxratio)
            end
            (state != :working) && (curnorm = dxrel)
        end
        state, ρmax_x, curnorm
    end
    function newzstate(state, dznorm, dzprev)
        curnorm = relnormz
        dzratio = dznorm / dzprev
        if (state == :unstable) && (dznorm <= dzthresh)
            state = :working
        end
        if (state == :noprogress) && (dzratio <= ρthresh)
            state = :working
        end
        if state == :working
            if dznorm <= ϵw
                # tiny dz
                state = :converged
                (verbosity > 1) && println("convergence (component-wise)")
            elseif dznorm > dzthresh
                state = :unstable
                relnormz = RT(Inf)
                ρmax_z = zero(RT)
            elseif dzratio > ρthresh
                if yscheme == :single
                    (verbosity > 1) && println("increasing precision(z)")
                    incrprec = true
                elseif ndouble > 1
                    state = :noprogress
                    (verbosity > 1) && println("z stalled")
                end
            else
                ρmax_z = max(ρmax_z, dzratio)
            end
            (state != :working) && (curnorm = dznorm)
        end
        state, ρmax_z, curnorm
    end

    # simple "for" loop w/o scoping
    irhs = 1

    @label NEXTRHS

    if equil
        bs = R * b[:,irhs]
    else
        bs = b[:,irhs]
    end
    bd = DT.(bs)
    y = Fs \ bs

    local yd, xnorm
    for iter=1:maxiter
        # compute residual in appropriate precision
        if yscheme == :single
            r = As * y - bs
            nsingle += 1
        else
            r = T.(Asd * yd - bd)
            ndouble += 1
        end
        # compute correction to y
        dy = Fs \ r
        # check error-related stopping criteria
        xnorm = norm(C*y,Inf)
        dxprev = dxnorm
        dxnorm = norm(C*dy,Inf)
        dzprev = dznorm
        dznorm = maximum( abs.(dy) ./ abs.(y))
        (verbosity > 0) && println("iter $iter |dx|=$dxnorm |dz|=$dznorm")
        ay0,ay1 = extrema(abs.(y))
        if (yscheme == :single) && (κs * ay1 / ay0 >= tol1)
            (verbosity > 1) && println("increasing precision")
            incrprec = true
        end
        xstate, ρmax_x, relnormx = newxstate(xstate, xnorm, dxnorm, dxprev)
        zstate, ρmax_z, relnormz = newzstate(zstate, dznorm, dzprev)
        # the unstable z case is not in the paper but seems
        # necessary to prevent early stalling
        if ((xstate != :working) && !(zstate ∈ [:working,:unstable]))
            break
        end
        if incrprec
# with modified logic above:
#            if yscheme == :double
#                @warn "secondary widening is indicated but not implemented"
#            end
            yscheme = :double
            incrprec = false
            yd = DT.(y)
        end
        # update solution
        if yscheme == :single
            y .-= dy
        else
            yd .-= DT.(dy)
            y = T.(yd)
        end
    end
    if xstate == :working
        relnormx = dxnorm / xnorm
    end
    if zstate == :working
        relnormz = dznorm
    end
    x::Vector{T} = C * y
    min1 = max(10,sqrt(n)) * ϵw # value from paper
    min2 = ϵw
    normwisebound = RT(max( relnormx/(1-ρmax_x), min2))
    termwisebound = RT(max( relnormz/(1-ρmax_z), min1))
    if !expert
        flag = false
        if normwisebound > sqrt(ϵw)
            flag = true
            normwisebound = one(RT)
        end
        if termwisebound > sqrt(ϵw)
            flag = true
            termwisebound = one(RT)
        end
        if flag && (verbosity >= 0)
            @warn "no convergence: result is not meaningful"
        end
    end
    fval = Dict(:converged => 0, :working => 1,
                :noprogress => 2, :unstable => 3)
    flags = [10*fval[xstate]+fval[zstate],nsingle,ndouble]

    # let's see if we can make this type-stable.
    if !(b isa AbstractVector)

        X[:,irhs] .= x
        normwisebounds[irhs] = normwisebound
        termwisebounds[irhs] = termwisebound
        flagss[:,irhs] .= flags

        if irhs == nrhs
            return (X, normwisebounds, termwisebounds, flagss)
        end

    else
        # return (X, normwisebounds, termwisebounds, flagss)
        # take that, you poor confused compiler!
        nb::RT = convert(RT,normwisebound)
        tb::RT = convert(RT,termwisebound)
        ff::Vector{Int} = convert.(Int,flags)
        return x, nb, tb, ff
    end

    # re-initialize state
    nsingle = 1
    ndouble = 0
    relnormx = relnormz = RT(Inf)
    dxnorm = dznorm = RT(Inf)
    ρmax_x = ρmax_z = zero(RT)
    xstate = :working
    zstate = :unstable
    yscheme = :single
    incrprec = false

    irhs += 1
    @goto NEXTRHS
    # no path to this location
end

include("eigen.jl")
end # module
