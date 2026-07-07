# ============================================================================
# Null Space Optimiser
#
# Level-set topology-optimisation port of the Null Space Optimizer of
# Feppon, Allaire & Dapogny ("Null space gradient flows for constrained
# optimization", ESAIM:COCV 2020; "Density based topology optimization with
# the Null Space Optimizer", SMO 2024). Reference Python implementation:
# the `nullspace_optimizer` package of F. Feppon.
#
# The design variable is the level-set function φ; the metric A is the
# Hilbertian inner product `vel_ext.K`; `project!` supplies the gradient
# A⁻¹·DF; `evolve!` plays the role of Feppon's `retract`. Constraints are
# handled with an ε-enlarged active set and a small dense dual QP solved by a
# self-contained active-set method (`_solve_bounded_qp`).
# ============================================================================

# ----------------------------------------------------------------------------
# Small dense linear algebra over the constraint set
# ----------------------------------------------------------------------------

# Robust solve for the small dense (symmetric) systems over the constraint set.
# Falls back to the Moore-Penrose pseudo-inverse when the constraint gradients
# are linearly dependent (degenerate/unqualified constraints).
function _safe_solve(A::AbstractMatrix{T},b::AbstractVector{T}) where T
  isempty(b) && return T[]
  x = try
    Symmetric(A) \ b
  catch
    fill(T(NaN),length(b))
  end
  all(isfinite,x) || (x = pinv(Matrix(A)) * b)
  return x
end

"""
    _solve_bounded_qp(S,b,p;tol,maxit) -> Λ

Solve the small dense convex QP

    min  ½ Λᵀ S Λ + bᵀ Λ    s.t.    Λ[p+1:end] ≥ 0

(the first `p` multipliers — those of equality constraints — are free) with a
primal active-set method (Nocedal & Wright, Alg. 16.3). `S` is the small
symmetric Gram matrix of the constraint gradients. Because the method is exact,
its solution already identifies the active set and supersedes the QP +
exact-solve refinement of the reference implementation.
"""
function _solve_bounded_qp(S::AbstractMatrix{T},b::AbstractVector{T},p::Integer;
    tol=1e-12,maxit=100) where T
  n = length(b)
  Λ = zeros(T,n)
  n == 0 && return Λ
  # Working set: bounded multipliers (index > p) currently fixed at their bound 0.
  # Λ = 0 is feasible, so start with every bound active.
  inW = falses(n)
  for i in (p+1):n
    inW[i] = true
  end
  for _ in 1:maxit
    free = .!inW
    # Minimiser of the equality-constrained subproblem with Λ[inW] fixed at 0.
    Λstar = zeros(T,n)
    if any(free)
      idx = findall(free)
      Λstar[idx] = _safe_solve(S[idx,idx],-b[idx])
    end
    d = Λstar .- Λ
    if maximum(abs,d;init=zero(T)) <= tol
      # At the equality-constrained minimiser: check bound-constraint multipliers.
      g = S * Λ .+ b
      rel = 0; relval = -tol
      for i in (p+1):n
        if inW[i] && g[i] < relval
          relval = g[i]; rel = i
        end
      end
      rel == 0 && return Λ              # KKT satisfied
      inW[rel] = false                  # releasing this bound decreases the objective
    else
      # Step toward Λstar, capped by the first bound that would be violated.
      α = one(T); block = 0
      for i in (p+1):n
        if free[i] && d[i] < -tol
          αi = -Λ[i] / d[i]
          if αi < α
            α = αi; block = i
          end
        end
      end
      Λ .+= α .* d
      block != 0 && (inW[block] = true)
    end
  end
  return Λ
end

# ----------------------------------------------------------------------------
# Null space projection map
# ----------------------------------------------------------------------------

"""
    struct NullSpaceMap

Computes the null-space descent direction for [`NullSpaceOptimiser`](@ref). Given
the (Hilbertian-extended) objective and constraint sensitivities it forms the
null-space direction `ξ_J`, the range-space (Gauss-Newton) direction `ξ_C`, and
combines them with Feppon's adaptive `A_J`/`A_C` scaling into a velocity field.

Constraints are reordered internally to `[equalities; inequalities]` to match
the reference implementation's `C = [G;H]` convention.
"""
struct NullSpaceMap{T}
  vel_ext :: AbstractVelocityExtension
  caches  :: T
  params  :: NamedTuple
  function NullSpaceMap(
      nC::Int,vel_ext::AbstractVelocityExtension;
      is_inequality::AbstractVector{Bool}=falses(nC),
      dt=0.1,αJ=1.0,αC=1.0,CFL=0.9,K=0.1,
      itnormalisation=1,normalize_tol=-1.0,tol_qp=1e-10,debug=false)
    gJ      = allocate_in_domain(vel_ext.K)
    gC      = [allocate_in_domain(vel_ext.K) for _ = 1:nC]
    Kg      = allocate_in_domain(vel_ext.K)
    xiJ     = allocate_in_domain(vel_ext.K)
    xiC     = allocate_in_domain(vel_ext.K)
    dx      = allocate_in_domain(vel_ext.K)
    uhd_reg = zero(vel_ext.U_reg)
    caches  = (gJ,gC,Kg,xiJ,xiC,dx,uhd_reg)
    # Constraint reordering: equalities first, then inequalities.
    p = count(!,is_inequality)
    q = count(is_inequality)
    perm = vcat(findall(!,is_inequality),findall(is_inequality))
    params = (;p,q,perm,dt,αJ,αC,CFL,K,itnormalisation,normalize_tol,tol_qp,debug)
    return new{typeof(caches)}(vel_ext,caches,params)
  end
end

"""
    update_descent_direction!(m::NullSpaceMap,dJ,C,dC,V_φ,it,normxiJ_save,prevmuls)

Interpolate the Hilbertian-extended sensitivities onto the regularisation space,
reorder the constraints, and compute the null-space velocity field.

Returns `(dx, normxiJ_save, muls)` where `dx` is the (unnormalised) velocity in
`U_reg` coordinates, `normxiJ_save` is the carried reference norm for the
adaptive scaling, and `muls` are the constraint Lagrange multipliers (internal
constraint order).
"""
function update_descent_direction!(m::NullSpaceMap,dJ,C,dC,V_φ,it,normxiJ_save,prevmuls)
  gJ,gC,Kg,xiJ,xiC,dx,uhd_reg = m.caches
  U_reg = m.vel_ext.U_reg
  perm = m.params.perm
  nC = m.params.p + m.params.q

  # Project the (already Hilbertian-extended) sensitivities from V_φ to U_reg coords.
  _interpolate_onto_rhs!(gJ,U_reg,uhd_reg,dJ,V_φ)
  for i = 1:nC
    _interpolate_onto_rhs!(gC[i],U_reg,uhd_reg,dC[perm[i]],V_φ)
  end
  # Constraint values reordered to [equalities; inequalities].
  Cint = Float64[C[perm[i]] for i = 1:nC]

  return _update_descent_direction!(m,Cint,it,normxiJ_save,prevmuls)
end

function _update_descent_direction!(m::NullSpaceMap,C,it,normxiJ_save,prevmuls)
  gJ,gC,Kg,xiJ,xiC,dx,uhd_reg = m.caches
  pr = m.params
  p,q = pr.p,pr.q
  nC = p + q
  Kmat = m.vel_ext.K
  T = eltype(dx)

  # Gram matrix S[i,j] = ⟨gCᵢ,gCⱼ⟩_K = DCᵢ A⁻¹ DCⱼ and coupling b[i] = ⟨gJ,gCᵢ⟩_K.
  S = zeros(T,nC,nC)
  b = zeros(T,nC)
  for j = 1:nC
    mul!(Kg,Kmat,gC[j])
    b[j] = dot(gJ,Kg)
    for i = j:nC
      S[i,j] = dot(gC[i],Kg)
      S[j,i] = S[i,j]
    end
  end

  # ε-enlarged active set: an inequality is "felt" within K·dt of saturation.
  # εⱼ uses the A-dual norm of DHⱼ, i.e. sqrt(S[j,j]) (Hilbertian analogue of
  # the reference `get_eps`).
  tildeEps = trues(nC)
  for j = (p+1):nC
    εj = sqrt(max(S[j,j],zero(T))) * pr.K * pr.dt
    tildeEps[j] = C[j] >= -εj
  end

  # Null-space direction ξ_J = gJ + Σ mulsᵢ gCᵢ. The multipliers solve the
  # bounded dual QP over the ε-active subset.
  muls = zeros(T,nC)
  act = findall(tildeEps)
  if !isempty(act)
    Λ = _solve_bounded_qp(S[act,act],b[act],p;tol=pr.tol_qp)
    for (k,i) in enumerate(act)
      muls[i] = Λ[k]
    end
  end
  copy!(xiJ,gJ)
  for i = 1:nC
    iszero(muls[i]) || (xiJ .+= muls[i] .* gC[i])
  end

  # Range-space (Gauss-Newton) direction ξ_C = Σ λᵢ gCᵢ over the violated
  # constraints plus the multiplier-active near-active ones.
  tilde = trues(nC)
  for j = (p+1):nC
    tilde[j] = C[j] >= zero(T)
  end
  indicesEps = copy(tilde)
  for j = (p+1):nC
    if tildeEps[j] && !tilde[j]
      indicesEps[j] = muls[j] > 30 * pr.tol_qp / 100
    end
  end
  fill!(xiC,zero(T))
  iact = findall(indicesEps)
  if !isempty(iact)
    λC = _safe_solve(S[iact,iact],T[C[i] for i in iact])
    for (k,i) in enumerate(iact)
      xiC .+= λC[k] .* gC[i]
    end
  end

  # Adaptive scaling of the two steps (reference `nlspace_solve`). The ξ_J step
  # is normalised exactly for the first `itnormalisation` iterations (or when
  # the active set changes), and bounded by αJ·dt afterwards.
  normxiJ = maximum(abs,xiJ)
  normxiC = maximum(abs,xiC)
  active_changed = false
  if pr.normalize_tol >= 0 && length(prevmuls) == nC
    for i = (p+1):nC
      if (muls[i] > pr.normalize_tol) != (prevmuls[i] > pr.normalize_tol)
        active_changed = true
        break
      end
    end
  end
  if it <= pr.itnormalisation || active_changed || !isfinite(normxiJ_save)
    normxiJ_save = normxiJ
    AJ = (pr.αJ * pr.dt) / (1e-9 + normxiJ)
  else
    AJ = pr.αJ * pr.dt / max(1e-9 + normxiJ,normxiJ_save)
  end
  AC = min(pr.CFL,pr.αC * pr.dt / max(normxiC,1e-9))

  # Combined velocity: dx = AJ·ξ_J + AC·ξ_C. Sign convention: `evolve!` descends
  # with the +∇J velocity (cf. AugmentedLagrangian / HilbertianProjection).
  copy!(dx,xiJ)
  dx .*= AJ
  dx .+= AC .* xiC

  return dx,normxiJ_save,muls
end

# ----------------------------------------------------------------------------
# Line-search helpers (reference acceptance test)
# ----------------------------------------------------------------------------

# Currently violated/active constraints: all equalities + inequalities with C ≥ 0.
function _ns_tilde(C,p)
  tilde = trues(length(C))
  for j = (p+1):length(C)
    tilde[j] = C[j] >= 0
  end
  return tilde
end

# 2-norm of C restricted to the subset selected by `mask`.
function _ns_subnorm(C,mask)
  s = zero(eltype(C))
  for i in eachindex(C)
    mask[i] && (s += C[i]^2)
  end
  return sqrt(s)
end

# Reference line-search acceptance test: reject only when the objective AND the
# violated-constraint norm both worsen. With `strict_trial` a stricter
# feasibility test is also applied. `C`/`newC` are in internal constraint order.
function _ns_accept(newJ,J,newC,C,tilde,p,strict_trial,Gtol,Htol)
  if newJ > J && _ns_subnorm(newC,tilde) >= _ns_subnorm(C,tilde)
    return false
  end
  if strict_trial
    nC = length(C)
    nG_new = sqrt(sum(i -> newC[i]^2,1:p;init=0.0))
    nG     = sqrt(sum(i -> C[i]^2,1:p;init=0.0))
    mH_new = p < nC ? maximum(@view newC[p+1:nC]) : 0.0
    mH     = p < nC ? maximum(@view C[p+1:nC]) : 0.0
    if nG_new > max(Gtol,nG) || mH_new > max(Htol,mH)
      return false
    end
  end
  return true
end

# Enable optimisations when not using auto diff (see `HilbertianProjection`).
# `WithAutoDiff` / `NoAutoDiff` are defined in HilbertianProjection.jl.

"""
    struct NullSpaceOptimiser{A} <: Optimiser

The Null Space Optimizer of Feppon, Allaire & Dapogny
([link](https://doi.org/10.1051/cocv/2020015)) for level-set topology
optimisation. The objective is decreased along a null-space direction `ξ_J`
that respects the active constraints, while a range-space direction `ξ_C`
restores feasibility. Inequality constraints are handled rigorously via an
ε-enlarged active set and a small dual QP.

# Parameters

- `problem::AbstractPDEConstrainedFunctionals`: The objective and constraint setup.
- `ls_evolver::AbstractLevelSetEvolution`: Solver for the evolution and reinitisation equations.
- `vel_ext::AbstractVelocityExtension`: The velocity-extension method; its inner
  product matrix `K` is used as the metric.
- `projector::NullSpaceMap`: Computes the null-space velocity field.
- `history::OptimiserHistory{Float64}`: Historical information for the optimisation problem.
- `converged::Function`: A function to check optimiser convergence.
- `params::NamedTuple`: Optimisation parameters.
"""
struct NullSpaceOptimiser{A} <: Optimiser
  problem    :: AbstractPDEConstrainedFunctionals
  ls_evolver :: AbstractLevelSetEvolution
  vel_ext    :: AbstractVelocityExtension
  projector  :: NullSpaceMap
  history    :: OptimiserHistory{Float64}
  converged  :: Function
  params     :: NamedTuple
  φ0 # TODO: Please remove me

  @doc """
      NullSpaceOptimiser(
        problem    :: AbstractPDEConstrainedFunctionals{N},
        ls_evolver :: AbstractLevelSetEvolution,
        vel_ext    :: AbstractVelocityExtension,
        φ0;
        is_inequality = falses(N),
        dt = 0.1, αJ = 1.0, αC = 1.0, CFL = 0.9, K = 0.1,
        itnormalisation = 1, normalize_tol = -1.0, maxtrials = 3, tol_qp = 1e-10,
        strict_trial = false, Gtol = 1e-7, Htol = 1e-3, tol_finite_diff = 0.15,
        reinit_mod = 1, γ_min = 0.001, γ_max = 1.0,
        maxiter = 1000, verbose = false, constraint_names = map(i -> Symbol("C_\$i"),1:N),
        converged::Function = default_ns_converged, debug = false
      ) where N

  Create an instance of `NullSpaceOptimiser`.

  # Required

  - `problem`, `ls_evolver`, `vel_ext` as above.
  - `φ0`: An initial level-set function as an `FEFunction` (or GridapDistributed equivalent).

  # Constraint handling

  - `is_inequality`: A `Bool` vector of length `N` marking which constraints are
    inequalities (`Cⱼ ≤ 0`). The default `falses(N)` treats every constraint as
    an equality (`Cⱼ = 0`), matching the other optimisers; with the default the
    method reduces to a projected-gradient scheme.

  # Step control (reference `dt`-stepping)

  - `dt = 0.1`: Pseudo-time step; the master step-size control. Because the
    derived `evolve!` coefficient `γ` must satisfy the CFL condition `γ ≤ 1`,
    `dt` must be chosen small enough for the mesh (a warning is printed if the
    derived `γ` saturates `γ_max`).
  - `αJ = 1.0`, `αC = 1.0`: Dimensionless scalings of the null-space and
    range-space steps.
  - `CFL = 0.9`: Upper bound on the range-space coefficient `A_C`.
  - `K = 0.1`: Distance (in units of `dt`) at which inequality constraints are
    "felt" — controls the ε-enlarged active set.
  - `itnormalisation = 1`: The null-space step is normalised exactly for this
    many initial iterations.
  - `normalize_tol = -1.0`: If `≥ 0`, also re-normalise when the active set
    changes.
  - `maxtrials = 3`: Number of step-halving line-search trials.
  - `tol_qp = 1e-10`: Tolerance for the active-set QP.
  - `strict_trial = false`, `Gtol = 1e-7`, `Htol = 1e-3`: Optional stricter
    line-search feasibility test.

  # Level-set / algorithm defaults

  - `reinit_mod = 1`: Reinitialise the level-set function every `reinit_mod` iterations.
  - `γ_min = 0.001`, `γ_max = 1.0`: Clamp on the derived `evolve!` coefficient
    `γ`. `γ_max` must be `≤ 1` for the `FiniteDifferenceEvolver` CFL condition.
  - `maxiter = 1000`: Maximum number of algorithm iterations.
  - `verbose = false`: Verbosity flag.
  - `constraint_names`: Constraint names for history output (user order).
  - `converged::Function = default_ns_converged`: Convergence criterion
    (`‖dx‖₂ < 1e-5·dt`).
  - `debug = false`: Debug flag.
  """
  function NullSpaceOptimiser(
      problem    :: AbstractPDEConstrainedFunctionals{N},
      ls_evolver :: AbstractLevelSetEvolution,
      vel_ext    :: AbstractVelocityExtension,
      φ0;
      is_inequality = falses(N),
      dt = 0.1, αJ = 1.0, αC = 1.0, CFL = 0.9, K = 0.1,
      itnormalisation = 1, normalize_tol = -1.0, maxtrials = 3, tol_qp = 1e-10,
      strict_trial = false, Gtol = 1e-7, Htol = 1e-3, tol_finite_diff = 0.15,
      reinit_mod = 1, γ_min = 0.001, γ_max = 1.0,
      maxiter = 1000, verbose = false,
      constraint_names = map(i -> Symbol("C_$i"),1:N),
      converged::Function = default_ns_converged, debug = false,
      γ_reinit = NaN
  ) where N

    @assert isnan(γ_reinit) "γ_reinit has been removed from all optimisers. Please set this
      in the corresponding reinitialiser (i.e., FiniteDifferenceReinitialiser)"
    @assert length(is_inequality) == N "is_inequality must have one Bool entry per constraint (N=$N)"
    @assert 0 < γ_min <= γ_max "We require 0 < γ_min ≤ γ_max"
    @assert γ_max <= 1 "γ_max must be ≤ 1 for the CFL condition of the FiniteDifferenceEvolver"
    @assert maxtrials >= 1 "We require maxtrials ≥ 1"

    _is_inequality = collect(Bool,is_inequality)
    constraint_names = map(Symbol,constraint_names)
    ns_keys = [:J,constraint_names...,:γ,:normdx]
    ns_bundles = Dict(:C => constraint_names)
    history = OptimiserHistory(Float64,ns_keys,ns_bundles,maxiter,verbose)

    projector = NullSpaceMap(N,vel_ext;is_inequality=_is_inequality,
      dt,αJ,αC,CFL,K,itnormalisation,normalize_tol,tol_qp,debug)

    # Optimisation when not using AD: derivatives can be recomputed cheaply
    # during the line search, avoiding a redundant adjoint solve afterwards.
    A = (hasproperty(problem,:analytic_dJ) && hasproperty(problem,:analytic_dC) &&
         (typeof(problem.analytic_dJ) <: Function) &&
         all(@. typeof(problem.analytic_dC) <: Function)) ? NoAutoDiff : WithAutoDiff

    params = (;debug,reinit_mod,γ_min,γ_max,maxtrials,
              strict_trial,Gtol,Htol,tol_finite_diff,dt)
    new{A}(problem,ls_evolver,vel_ext,projector,history,converged,params,φ0)
  end
end

get_history(m::NullSpaceOptimiser) = m.history

function converged(m::NullSpaceOptimiser)
  return m.converged(m)
end

"""
    default_ns_converged(m::NullSpaceOptimiser;tol)

Reference convergence criterion: the optimiser has converged once the (averaged
``L^2``) norm of the update `dx` drops below `tol = 1e-5·dt`.
"""
function default_ns_converged(m::NullSpaceOptimiser;tol = 1e-5 * m.params.dt)
  h = m.history
  it = get_last_iteration(h)
  it < 1 && return false
  return h[:normdx,it] < tol
end

# 0th iteration
function Base.iterate(m::NullSpaceOptimiser)
  history,params = m.history,m.params
  φh = m.φ0
  V_φ = get_ls_space(m.ls_evolver)
  uhd = zero(V_φ)
  nC = m.projector.params.p + m.projector.params.q

  ## Reinitialise as SDF
  _,reinit_cache = reinit!(m.ls_evolver,φh)

  ## Compute FE problem and shape derivatives
  J,C,dJ,dC = Gridap.evaluate!(m.problem,φh)
  uh    = get_state(m.problem)
  vel   = copy(get_free_dof_values(φh))
  φ_tmp = copy(vel)

  ## Hilbertian extension-regularisation
  project!(m.vel_ext,dJ,V_φ,uhd)
  project!(m.vel_ext,dC,V_φ,uhd)

  ## Update history and build state
  push!(history,(J,C...,NaN,NaN))
  state = (;it=1,J,C,dJ,dC,uh,φh,uhd,vel,φ_tmp,
           normxiJ_save=NaN,muls=zeros(nC),reinit_cache,evo_cache=nothing)
  vars = params.debug ? (0,uh,φh,state) : (0,uh,φh)
  return vars,state
end

# ith iteration
function Base.iterate(m::NullSpaceOptimiser,state)
  it,J,C,dJ,dC,uh,φh,uhd,vel,φ_tmp,normxiJ_save,muls,reinit_cache,evo_cache = state
  history,params = m.history,m.params

  ## Periodically call GC
  iszero(it % 50) && GC.gc()

  ## Check stopping criteria
  if finished(m)
    return nothing
  end

  V_φ   = get_ls_space(m.ls_evolver)
  U_reg = m.vel_ext.U_reg

  ## Null-space descent direction (reference adaptive dt-stepping)
  dx,normxiJ_save,muls = update_descent_direction!(
    m.projector,dJ,C,dC,V_φ,it,normxiJ_save,muls)
  interpolate!(FEFunction(U_reg,dx),vel,V_φ)

  ## Map the reference ‖dx‖∞ displacement onto the evolver coefficient γ. The
  ## evolver normalises the velocity internally, so γ carries the step size.
  Δmin   = minimum(get_min_dof_spacing(m.ls_evolver))
  msteps = get_max_steps(m.ls_evolver)
  normdx = sqrt(dot(vel,vel) / length(vel))
  γ_raw  = maximum(abs,vel) / (Δmin * msteps)
  γ      = clamp(γ_raw,params.γ_min,params.γ_max)
  if γ_raw > params.γ_max
    print_msg(m.history,
      "  Step exceeds CFL-stable size (γ_raw=$(γ_raw)); clamped to $(γ). Reduce dt.\n";
      color=:yellow)
  end

  ## Line search (reference acceptance test)
  J,C,dJ,dC,evo_cache,reinit_cache = _linesearch!(m,state,vel,γ)

  ## Hilbertian extension-regularisation of the new sensitivities
  project!(m.vel_ext,dJ,V_φ,uhd)
  project!(m.vel_ext,dC,V_φ,uhd)

  ## Update history and build state
  push!(history,(J,C...,γ,normdx))
  uh = get_state(m.problem)
  state = (;it=it+1,J,C,dJ,dC,uh,φh,uhd,vel,φ_tmp,normxiJ_save,muls,
           reinit_cache,evo_cache)
  vars = params.debug ? (it,uh,φh,state) : (it,uh,φh)
  return vars,state
end

function _linesearch!(m::NullSpaceOptimiser{WithAutoDiff},state,vel,γ)
  it,J,C,dJ,dC,uh,φh,uhd,_,φ_tmp,_,_,reinit_cache,evo_cache = state
  params,history = m.params,m.history
  maxtrials,reinit_mod = params.maxtrials,params.reinit_mod
  strict_trial,Gtol,Htol = params.strict_trial,params.Gtol,params.Htol
  perm,p = m.projector.params.perm,m.projector.params.p

  φ = get_free_dof_values(φh)
  copy!(φ_tmp,φ)
  Cint  = Float64[C[perm[i]] for i in eachindex(perm)]
  tilde = _ns_tilde(Cint,p)

  accepted = false
  for k = 0:(maxtrials-1)
    γk = γ * (0.5^k)
    copy!(φ,φ_tmp)
    _,evo_cache = evolve!(m.ls_evolver,φ,vel,γk,evo_cache)
    if iszero(it % reinit_mod)
      _,reinit_cache = reinit!(m.ls_evolver,φ,reinit_cache)
    end
    newJ,newC = evaluate_functionals!(m.problem,φh)
    newCint = Float64[newC[perm[i]] for i in eachindex(perm)]
    if _ns_accept(newJ,J,newCint,Cint,tilde,p,strict_trial,Gtol,Htol)
      accepted = true
      print_msg(history,"  Accepted iteration with γ = $(γk)\n";color=:yellow)
      break
    else
      print_msg(history,"  Rejected trial with γ = $(γk)\n";color=:red)
    end
  end
  accepted || print_msg(history,
    "  All trials rejected; accepting last with γ = $(γ * 0.5^(maxtrials-1))\n";color=:red)

  ## Recompute objective, constraints, and shape derivatives at the new level set
  J,C,dJ,dC = Gridap.evaluate!(m.problem,φh)
  return J,C,dJ,dC,evo_cache,reinit_cache
end

function _linesearch!(m::NullSpaceOptimiser{NoAutoDiff},state,vel,γ)
  it,J,C,dJ,dC,uh,φh,uhd,_,φ_tmp,_,_,reinit_cache,evo_cache = state
  params,history = m.params,m.history
  maxtrials,reinit_mod = params.maxtrials,params.reinit_mod
  strict_trial,Gtol,Htol = params.strict_trial,params.Gtol,params.Htol
  perm,p = m.projector.params.perm,m.projector.params.p

  φ = get_free_dof_values(φh)
  copy!(φ_tmp,φ)
  Cint  = Float64[C[perm[i]] for i in eachindex(perm)]
  tilde = _ns_tilde(Cint,p)

  accepted = false
  Jn,Cn,dJn,dCn = J,C,dJ,dC
  for k = 0:(maxtrials-1)
    γk = γ * (0.5^k)
    copy!(φ,φ_tmp)
    _,evo_cache = evolve!(m.ls_evolver,φ,vel,γk,evo_cache)
    if iszero(it % reinit_mod)
      _,reinit_cache = reinit!(m.ls_evolver,φ,reinit_cache)
    end
    # No AD: derivatives are cheap, so the full evaluation is done per trial.
    Jn,Cn,dJn,dCn = Gridap.evaluate!(m.problem,φh)
    newCint = Float64[Cn[perm[i]] for i in eachindex(perm)]
    if _ns_accept(Jn,J,newCint,Cint,tilde,p,strict_trial,Gtol,Htol)
      accepted = true
      print_msg(history,"  Accepted iteration with γ = $(γk)\n";color=:yellow)
      break
    else
      print_msg(history,"  Rejected trial with γ = $(γk)\n";color=:red)
    end
  end
  accepted || print_msg(history,"  All trials rejected; accepting last\n";color=:red)
  return Jn,Cn,dJn,dCn,evo_cache,reinit_cache
end
