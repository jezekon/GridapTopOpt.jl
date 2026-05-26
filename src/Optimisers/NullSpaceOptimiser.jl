# Direction-computation map for the null-space method
struct NullSpaceDirectionMap{A}
  vel_ext :: AbstractVelocityExtension
  caches
  params
  function NullSpaceDirectionMap(
    nC :: Int,
    vel_ext::AbstractVelocityExtension;
    alphaJ=1.0, alphaC=1.0, CFL=0.9, itnormalisation=1
  )
    dJ_proj = allocate_in_domain(vel_ext.K)
    dC_proj = [allocate_in_domain(vel_ext.K) for _ = 1:nC]
    xiJ     = allocate_in_domain(vel_ext.K)
    xiC     = allocate_in_domain(vel_ext.K)
    vel_buf = allocate_in_domain(vel_ext.K)
    K_buf   = allocate_in_domain(vel_ext.K)
    M       = zeros(Float64,nC,nC)
    b_J     = zeros(Float64,nC)
    muls_J  = zeros(Float64,nC)
    muls_C  = zeros(Float64,nC)
    muls    = zeros(Float64,nC)
    uhd     = zero(vel_ext.U_reg)
    normxiJ_save = Ref(Inf)
    caches  = (;dJ_proj,dC_proj,xiJ,xiC,vel_buf,K_buf,M,b_J,muls_J,muls_C,muls,uhd,normxiJ_save)
    params  = (;alphaJ,alphaC,CFL,itnormalisation)
    return new{typeof(vel_ext.K)}(vel_ext,caches,params)
  end
end

# Compute null-space direction ξ_J and range-space direction ξ_C via the
# Schur-complement of the KKT system:
#   [ K   Aᵀ ] [ X ]   [ r ]
#   [ A   0  ] [ μ ] = [ s ]
# After projection, dJ_proj ≈ K⁻¹ dJ_raw and dC_proj[i] ≈ K⁻¹ dC_raw[i].
# Then ξ_J = dJ_proj - Σ μ_J[i]·dC_proj[i] with μ_J = M\b_J, b_J[i] = dot(dC_proj[i], K·dJ_proj),
# and ξ_C = Σ (M\C)[i]·dC_proj[i],  where M[i,j] = dot(dC_proj[i], K·dC_proj[j]).
function update_descent_direction!(m::NullSpaceDirectionMap,dJ,C,dC,K,V_φ)
  c = m.caches
  U_reg = m.vel_ext.U_reg

  _interpolate_onto_rhs!(c.dJ_proj,U_reg,c.uhd,dJ,V_φ)
  for i ∈ eachindex(dC)
    _interpolate_onto_rhs!(c.dC_proj[i],U_reg,c.uhd,dC[i],V_φ)
  end

  nC = length(C)
  if nC == 0
    copy!(c.xiJ,c.dJ_proj); fill!(c.xiC,zero(eltype(c.xiC)))
    return c.xiJ, c.xiC, c.muls
  end

  # Build Gram matrix M[i,j] = (dC_proj[i], K·dC_proj[j]) and rhs b_J[i] = (dC_proj[i], K·dJ_proj)
  for j in 1:nC
    mul!(c.K_buf,K,c.dC_proj[j])
    for i in 1:nC
      c.M[i,j] = dot(c.dC_proj[i],c.K_buf)
    end
  end
  mul!(c.K_buf,K,c.dJ_proj)
  for i in 1:nC
    c.b_J[i] = dot(c.dC_proj[i],c.K_buf)
  end

  # Solve M·μ_J = b_J and M·μ_C = C. Share the factorisation; fall back to
  # the Moore-Penrose pseudo-inverse if M is (near-)singular (e.g. when two
  # constraints become locally linearly dependent).
  copyto!(c.muls_J,c.b_J); copyto!(c.muls_C,C)
  F = lu(c.M; check=false)
  if issuccess(F)
    ldiv!(F,c.muls_J)
    ldiv!(F,c.muls_C)
  else
    Mpinv = pinv(c.M)
    mul!(c.muls_J,Mpinv,c.b_J)
    mul!(c.muls_C,Mpinv,collect(C))
  end

  # Null-space step
  copy!(c.xiJ,c.dJ_proj)
  for i in 1:nC
    c.xiJ .-= c.muls_J[i] .* c.dC_proj[i]
  end
  c.muls .= c.muls_J

  # Range-space step
  fill!(c.xiC,zero(eltype(c.xiC)))
  for i in 1:nC
    c.xiC .+= c.muls_C[i] .* c.dC_proj[i]
  end

  return c.xiJ, c.xiC, c.muls
end

# Compute the step sizes AJ, AC and the velocity = -AJ·ξ_J - AC·ξ_C on U_reg.
# Mirrors nullspace.py:461-472. Note: the absolute magnitude of vel_buf is
# rescaled again by the CFL condition inside `evolve!`, so AJ and AC chiefly
# control the *ratio* between the null-space and range-space directions.
function _compute_step_sizes_and_velocity!(m::NullSpaceDirectionMap,γ,it)
  c = m.caches
  alphaJ, alphaC, CFL, itnormalisation = m.params

  normxiJ = sqrt(dot(c.xiJ,c.xiJ))
  normxiC = sqrt(dot(c.xiC,c.xiC))

  if it < itnormalisation || !isfinite(c.normxiJ_save[])
    c.normxiJ_save[] = normxiJ
    AJ = alphaJ * γ / (1e-9 + normxiJ)
  else
    AJ = alphaJ * γ / max(1e-9 + normxiJ, c.normxiJ_save[])
  end
  AC = min(CFL, alphaC * γ / max(normxiC, 1e-9))

  @. c.vel_buf = -AJ * c.xiJ - AC * c.xiC
  normdx = sqrt(dot(c.vel_buf,c.vel_buf))
  return normxiJ, normxiC, normdx, c.vel_buf
end

"""
    struct NullSpaceOptimiser <: Optimiser

A null-space gradient-flow optimiser based on Feppon, Allaire & Dapogny, 2019
([link](https://doi.org/10.1051/cocv/2020015)). The method discretises the ODE
``\\dot x(t) = -\\alpha_J \\xi_J(x) - \\alpha_C \\xi_C(x)`` with Forward Euler,
where ``\\xi_J`` is the best feasible descent direction for the objective and
``\\xi_C`` corrects the constraint violation. ``\\xi_J`` is computed by
Schur-complement reduction of the KKT system on the Hilbertian inner-product
matrix ``K`` from `vel_ext`; ``\\xi_C`` is computed directly as the X-block of
the same KKT system, equivalent to ``K^{-1}\\,\\mathrm{d}C^{\\top}\\,M^{-1}\\,C``
with ``M = \\mathrm{d}C\\,K^{-1}\\,\\mathrm{d}C^{\\top}``.

This implementation supports equality constraints. The `AbstractPDEConstrainedFunctionals`
API in this codebase does not distinguish equalities from inequalities, so all
constraints are treated as equalities.

# Parameters

- `problem::AbstractPDEConstrainedFunctionals`: The objective and constraint setup.
- `ls_evolver::AbstractLevelSetEvolution`: Solver for the evolution and reinitialisation equations.
- `vel_ext::AbstractVelocityExtension`: The velocity-extension method for extending
  shape sensitivities onto the computational domain.
- `projector::NullSpaceDirectionMap`: Direction-computation caches and parameters.
- `history::OptimiserHistory{Float64}`: Historical information for optimisation problem.
- `converged::Function`: A function to check optimiser convergence.
- `has_oscillations::Function`: A function to check for oscillations.
- `params::NamedTuple`: Optimisation parameters.
"""
struct NullSpaceOptimiser <: Optimiser
  problem           :: AbstractPDEConstrainedFunctionals
  ls_evolver        :: AbstractLevelSetEvolution
  vel_ext           :: AbstractVelocityExtension
  projector         :: NullSpaceDirectionMap
  history           :: OptimiserHistory{Float64}
  converged         :: Function
  has_oscillations  :: Function
  params            :: NamedTuple
  φ0

  @doc """
      NullSpaceOptimiser(
        problem    :: AbstractPDEConstrainedFunctionals{N},
        ls_evolver :: AbstractLevelSetEvolution,
        vel_ext    :: AbstractVelocityExtension,
        φ0;
        alphaJ=1.0, alphaC=1.0, CFL=0.9, itnormalisation=1,
        γ=0.1, reinit_mod=1,
        maxtrials=3, strict_trial=false, Gtol=1e-7,
        os_γ_mult=0.5, has_oscillations::Function=default_has_oscillations,
        tol=1e-5*γ, maxiter=1000,
        converged::Function=default_ns_converged,
        verbose=false, debug=false,
        constraint_names = map(i -> Symbol("C_\$i"), 1:N),
        γ_reinit=NaN,
      ) where N

  Create an instance of `NullSpaceOptimiser`.

  # Required

  - `problem::AbstractPDEConstrainedFunctionals`: The objective and constraint setup.
  - `ls_evolver::AbstractLevelSetEvolution`: Solver for the evolution and reinitialisation equations.
  - `vel_ext::AbstractVelocityExtension`: The velocity-extension method for extending
    shape sensitivities onto the computational domain.
  - `φ0`: An initial level-set function defined as an `FEFunction` (or GridapDistributed equivalent).

  # Algorithm defaults

  - `alphaJ = 1.0`: Scaling of the null-space step (controls the relative magnitude
    of ``\\xi_J`` inside the velocity buffer; the absolute step size of the
    level-set update is determined by `γ` and the CFL inside `evolve!`).
  - `alphaC = 1.0`: Scaling of the range-space step (same caveat as `alphaJ`).
  - `CFL = 0.9`: Upper bound on `AC`.
  - `itnormalisation = 1`: Forces `α_J` normalisation for the first `itnormalisation` iterations.
  - `γ = 0.1`: Time step for the Hamilton-Jacobi evolution equation.
  - `reinit_mod = 1`: How often to solve the reinitialisation equation.

  # Line search defaults (Python null-space style; halve `dx`)

  - `maxtrials = 3`: Maximum number of halving trials.
  - `strict_trial = false`: If `true`, additionally require `\\|G\\| < max(Gtol, \\|G_old\\|)`.
  - `Gtol = 1e-7`: Constraint tolerance for `strict_trial`.

  # Oscillation handling

  - `os_γ_mult = 0.5`: Decrease multiplier for `γ` when `has_oscillations` returns true.
  - `has_oscillations::Function = default_has_oscillations`: Oscillation detector.

  # Stopping

  - `tol = 1e-5*γ`: Convergence threshold on the L² norm of the velocity step.
  - `maxiter = 1000`: Maximum number of algorithm iterations.
  - `converged::Function = default_ns_converged`: Convergence criterion.

  # Reporting

  - `verbose = false`: Verbosity flag.
  - `debug = false`: Debug flag; if `true`, `iterate` yields `(it, uh, φh, state)`.
  - `constraint_names`: Constraint names for history output. The history additionally
    exposes the null-space Lagrange multipliers under the keys `μ1, μ2, ..., μN`
    bundled as `:μ` (accessible via `h[:μ, it]`).
  """
  function NullSpaceOptimiser(
    problem    :: AbstractPDEConstrainedFunctionals{N},
    ls_evolver :: AbstractLevelSetEvolution,
    vel_ext    :: AbstractVelocityExtension,
    φ0;
    alphaJ=1.0, alphaC=1.0, CFL=0.9, itnormalisation=1,
    γ=0.1, reinit_mod=1,
    maxtrials=3, strict_trial=false, Gtol=1e-7,
    os_γ_mult=0.5, has_oscillations::Function=default_has_oscillations,
    tol=1e-5*γ, maxiter=1000,
    converged::Function=default_ns_converged,
    verbose=false, debug=false,
    constraint_names = map(i -> Symbol("C_$i"), 1:N),
    γ_reinit=NaN,
  ) where N

    @assert isnan(γ_reinit) "γ_reinit has been removed from all optimisers. Please set this
      in the corresponding reinitialiser (i.e., FiniteDifferenceReinitialiser)"
    @assert maxtrials ≥ 1 "maxtrials must be ≥ 1"
    @assert reinit_mod ≥ 1 "reinit_mod must be ≥ 1"

    constraint_names = map(Symbol,constraint_names)
    μ_names = map(i -> Symbol("μ$i"), 1:N)
    ns_keys = [:J, constraint_names..., :γ, :normxiJ, :normxiC, :normdx, μ_names...]
    ns_bundles = Dict(:C => constraint_names, :μ => μ_names)
    history = OptimiserHistory(Float64,ns_keys,ns_bundles,maxiter,verbose)

    projector = NullSpaceDirectionMap(N,vel_ext;alphaJ,alphaC,CFL,itnormalisation)

    params = (;debug,γ,reinit_mod,os_γ_mult,tol,
               maxtrials,strict_trial,Gtol)
    new(problem,ls_evolver,vel_ext,projector,history,converged,has_oscillations,params,φ0)
  end
end

get_history(m::NullSpaceOptimiser) = m.history

function converged(m::NullSpaceOptimiser)
  return m.converged(m)
end

function default_ns_converged(
  m::NullSpaceOptimiser;
  J_tol = 0.2*maximum(get_min_dof_spacing(m.ls_evolver)),
  C_tol = 0.001,
  dx_tol = m.params.tol,
)
  h  = m.history
  it = get_last_iteration(h)
  it < 10 && return false

  Ji, Ci = h[:J,it], h[:C,it]
  J_prev = h[:J,it-5:it]
  A = all(J -> abs(Ji - J)/abs(Ji) < J_tol, J_prev)
  B = all(C -> abs(C) < C_tol, Ci)
  D = h[:normdx,it] < dx_tol
  return A && B && D
end

function default_has_oscillations(m::NullSpaceOptimiser,os_it;
    itlength=50,itstart=2itlength)
  h  = m.history
  it = get_last_iteration(h)
  if it < itstart || it < os_it + itlength + 1
    return false
  end
  J = h[:J]
  J_osc = ~isnan(_zerocrossing_period(J[it-itlength+1:it+1]));
  C = h[:C,it-itlength+1:it+1]
  C_osc = all(x->~isnan(_zerocrossing_period(x)),C);
  return J_osc && C_osc
end

# 0th iteration
function Base.iterate(m::NullSpaceOptimiser)
  history, params = m.history, m.params
  φh = m.φ0
  V_φ = get_ls_space(m.ls_evolver)
  U_reg = m.vel_ext.U_reg
  uhd = zero(V_φ)

  ## Reinitialise as SDF
  _, reinit_cache = reinit!(m.ls_evolver,φh)

  ## Compute FE problem and shape derivatives
  J, C, dJ, dC = Gridap.evaluate!(m.problem,φh)
  uh  = get_state(m.problem)
  vel = copy(get_free_dof_values(φh))
  φ_tmp = copy(vel)

  ## Hilbertian extension-regularisation
  project!(m.vel_ext,dJ,V_φ,uhd)
  project!(m.vel_ext,dC,V_φ,uhd)
  xiJ, xiC, muls = update_descent_direction!(m.projector,dJ,C,dC,m.vel_ext.K,V_φ)

  ## Step sizes & velocity
  normxiJ, normxiC, normdx, vel_buf = _compute_step_sizes_and_velocity!(m.projector,params.γ,0)
  interpolate!(FEFunction(U_reg,vel_buf),vel,V_φ)

  # Update history and build state
  push!(history,(J,C...,params.γ,normxiJ,normxiC,normdx,muls...))
  state = (;it=1,J,C,xiJ,xiC,muls,dJ,dC,uh,φh,uhd,vel,φ_tmp,params.γ,
           os_it=-1,reinit_cache,evo_cache=nothing)
  vars  = params.debug ? (0,uh,φh,state) : (0,uh,φh)
  return vars, state
end

# ith iteration
function Base.iterate(m::NullSpaceOptimiser,state)
  it,J,C,xiJ,xiC,muls,dJ,dC,uh,φh,uhd,vel,φ_tmp,γ,os_it,reinit_cache,evo_cache = state
  history,params = m.history,m.params

  ## Periodicially call GC
  iszero(it % 50) && GC.gc();

  ## Check stopping criteria
  if finished(m)
    return nothing
  end

  ## Oscillation detection
  if (γ > 0.001) && m.has_oscillations(m,os_it)
    os_it = it + 1
    γ    *= params.os_γ_mult
    print_msg(m.history,"   Oscillations detected, reducing γ to $(γ)\n",color=:yellow)
  end

  ## Halving line search (Python null-space style)
  V_φ = get_ls_space(m.ls_evolver); U_reg = m.vel_ext.U_reg
  J, C, dJ, dC, reinit_cache, evo_cache = _linesearch_nullspace!(
    m,it,J,C,φh,vel,φ_tmp,γ,reinit_cache,evo_cache)
  uh = get_state(m.problem)

  ## Hilbertian extension-regularisation & direction update
  project!(m.vel_ext,dJ,V_φ,uhd)
  project!(m.vel_ext,dC,V_φ,uhd)
  xiJ, xiC, muls = update_descent_direction!(m.projector,dJ,C,dC,m.vel_ext.K,V_φ)

  ## Step sizes & velocity
  normxiJ, normxiC, normdx, vel_buf = _compute_step_sizes_and_velocity!(m.projector,γ,it)
  interpolate!(FEFunction(U_reg,vel_buf),vel,V_φ)

  ## Update history and build state
  push!(history,(J,C...,γ,normxiJ,normxiC,normdx,muls...))
  state = (;it=it+1,J,C,xiJ,xiC,muls,dJ,dC,uh,φh,uhd,vel,φ_tmp,γ,
           os_it,reinit_cache,evo_cache)
  vars  = params.debug ? (it,uh,φh,state) : (it,uh,φh)
  return vars, state
end

# Halve dx (= γ * vel) until the trial step does not simultaneously worsen
# both objective and constraint norm. Mirrors nullspace.py:488-518.
function _linesearch_nullspace!(m::NullSpaceOptimiser,it,J,C,φh,vel,φ_tmp,γ,reinit_cache,evo_cache)
  params, history = m.params, m.history
  maxtrials, strict_trial, Gtol = params.maxtrials, params.strict_trial, params.Gtol
  reinit_mod = params.reinit_mod

  normC_old = isempty(C) ? 0.0 : norm(C)
  J_old = J
  φ = get_free_dof_values(φh); copy!(φ_tmp,φ)

  J_new, C_new, dJ_new, dC_new = J, C, nothing, nothing
  accepted = false
  for k in 0:maxtrials-1
    copy!(φ,φ_tmp)
    γ_eff = γ * 0.5^k
    _, evo_cache = evolve!(m.ls_evolver,φ,vel,γ_eff,evo_cache)
    if iszero(it % reinit_mod)
      _, reinit_cache = reinit!(m.ls_evolver,φh,reinit_cache)
    end

    J_new, C_new, dJ_new, dC_new = Gridap.evaluate!(m.problem,φh)
    normC_new = isempty(C_new) ? 0.0 : norm(C_new)

    reject = (J_new > J_old) && (normC_new >= normC_old)
    if strict_trial
      reject = reject || (normC_new > max(Gtol, normC_old))
    end

    if !reject
      accepted = true
      print_msg(history,"   Accepted linesearch trial $(k+1) (γ_eff = $(γ_eff))\n",color=:yellow)
      break
    else
      print_msg(history,"   Rejected linesearch trial $(k+1) (γ_eff = $(γ_eff))\n",color=:red)
    end
  end

  if !accepted
    print_msg(history,
      "   All $(maxtrials) linesearch trials failed; accepting smallest step\n",color=:red)
  end

  return J_new, C_new, dJ_new, dC_new, reinit_cache, evo_cache
end
