module ThermalComplianceHPMTests
using Test

using Gridap, GridapTopOpt
using GridapTopOpt: WithAutoDiff, NoAutoDiff
"""
  (Serial) Minimum thermal compliance with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = ∫ κ*∇(u)⋅∇(u) dΩ
        Ω
    s.t., Vol(Ω) = vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ κ*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""
function main(;order,AD_case,α_min_decay_tol=0.0,check_converged=false)
  ## Parameters
  xmax = ymax = 1.0
  prop_Γ_N = 0.2
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax)
  el_size = (10,10)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5*order^2)/minimum(el_size)
  κ = 1
  vf = 0.4
  η_coeff = 2
  α_coeff = 4max_steps*γ

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() ||
      x[2] >= ymax-ymax*prop_Γ_D - eps()))
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <=
      ymax/2+ymax*prop_Γ_N/2 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
  dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
  reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ)
  pcfs = if AD_case == :no_ad
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ,analytic_dC=[dVol])
  elseif AD_case == :with_ad
    PDEConstrainedFunctionals(J,[Vol],state_map)
  elseif AD_case == :partial_ad1
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ)
  elseif AD_case == :partial_ad2
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])
  elseif AD_case == :custom_pcf
    objective = GridapTopOpt.StateParamMap(J,state_map)
    constraints = map(Ci -> GridapTopOpt.StateParamMap(Ci,state_map),[Vol])
    function φ_to_jc(φ)
      u = state_map(φ)
      j = objective(u,φ)
      c = map(constraint -> constraint(u,φ),constraints)
      [j,c...]
    end
    CustomPDEConstrainedFunctionals(φ_to_jc,length(constraints);state_map)
  elseif AD_case == :custom_pcf_analyticVol
    objective = GridapTopOpt.StateParamMap(J,state_map)
    constraints = map(Ci -> GridapTopOpt.StateParamMap(Ci,state_map),[Vol])
    function φ_to_jc2(φ)
      u = state_map(φ)
      j = objective(u,φ)
      c = map(constraint -> constraint(u,φ),constraints)
      [j,c...]
    end
    function analytic_dC1!(dC,φ)
      println("!!         analytic_dC1! called")
      φh = FEFunction(V_φ,φ)
      uh = get_state(state_map)
      _dC(q) = dVol(q,uh,φh)
      Gridap.FESpaces.assemble_vector!(_dC,dC,V_φ)
    end
    CustomPDEConstrainedFunctionals(φ_to_jc2,length(constraints);state_map,
      analytic_dC=[analytic_dC1!])
  else
    @error "AD case not defined"
  end

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;
    γ,verbose=true,constraint_names=[:Vol],α_min_decay_tol)
  @test optimiser.projector.params.α_min_decay_tol == α_min_decay_tol

  AD_case ∈ (:with_ad,:partial_ad1,:partial_ad2) && @test typeof(optimiser) <: HilbertianProjection{WithAutoDiff}
  AD_case ∈ (:no_ad,) && @test typeof(optimiser) <: HilbertianProjection{NoAutoDiff}

  # Do a few iterations
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  check_converged && converged_checks(optimiser)
  true
end

"""
  Checks the options `J_window` and `J_pair_mean` of `default_hp_converged` on a
  history with known values. The tolerances are given, J_tol = 0.002 and
  C_tol = 0.001, so the expected results follow from the values alone.
"""
function converged_checks(optimiser)
  h = optimiser.history
  # Write the objective J and the constraint C of iterations 0, 1, ... into the history
  function set_history!(J,C)
    GridapTopOpt.reset!(h)
    for k in eachindex(J)
      h.niter += 1
      h[:J,h.niter] = J[k]
      h[:Vol,h.niter] = C[k]
    end
  end
  converged(;kwargs...) = GridapTopOpt.default_hp_converged(optimiser;J_tol=0.002,C_tol=0.001,kwargs...)

  # Constant objective, constraint met: the defaults stop at iteration 11, not at 10
  set_history!(ones(11),zeros(11))
  @test !converged()
  set_history!(ones(12),zeros(12))
  @test converged()

  # Constraint not met in the last iteration
  C = zeros(12); C[end] = 0.002
  set_history!(ones(12),C)
  @test !converged()

  # Objective falls by 0.03 % per iteration: 0.15 % over 5 iterations, 0.6 % over 20
  J = [1 - 3e-4*k for k = 0:29]
  set_history!(J,zeros(30))
  @test converged()
  @test !converged(;J_window=20)

  # Objective alternates by ±0.3 % around 1: Jₖ never meets 0.2 %, the mean of a pair does
  J = [1 + 3e-3*(-1)^k for k = 0:29]
  set_history!(J,zeros(30))
  @test !converged()
  @test !converged(;J_window=20)
  @test converged(;J_pair_mean=true)
  @test converged(;J_pair_mean=true,J_window=20)

  # A window of 20 iterations stops at iteration 21 at the earliest
  set_history!(J[1:21],zeros(21))
  @test !converged(;J_pair_mean=true,J_window=20)
  set_history!(J[1:22],zeros(22))
  @test converged(;J_pair_mean=true,J_window=20)
end

# Test that these run successfully
@test main(;order=1,AD_case=:with_ad)
@test main(;order=1,AD_case=:partial_ad1)
@test main(;order=1,AD_case=:partial_ad2)
@test main(;order=1,AD_case=:no_ad,check_converged=true)
@test main(;order=1,AD_case=:custom_pcf)
@test main(;order=1,AD_case=:custom_pcf_analyticVol)
# The decay tolerance reaches compute_α through the constructor
@test main(;order=1,AD_case=:no_ad,α_min_decay_tol=0.01)

end # module