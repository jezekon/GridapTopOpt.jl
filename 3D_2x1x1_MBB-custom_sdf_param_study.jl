# ==============================================================================
# PARAMETRIC STUDY: 3D MBB Beam Level-Set Optimization
# ==============================================================================
# Full factorial study over: γ, max_steps
# α_coeff = 4 * max_steps * γ  (derived)
# Total combinations: 3 × 4 = 12
# ==============================================================================

using Gridap,
    Gridap.MultiField,
    GridapDistributed,
    GridapPETSc,
    GridapSolvers,
    PartitionedArrays,
    GridapTopOpt,
    SparseMatricesCSR,
    JLD2,
    WriteVTK,
    Dates

include("src/CustomModif/CustomModif.jl")

# ==============================================================================
# STUDY CONFIGURATION
# ==============================================================================
const TASK_NAME = "3D_2x1x1_MBB_01tol_r2.0"
const LEVELSET_SUFFIX = "_SDF_B-0.05.jld2"

# Parameter grid
const PARAM_γ = [0.06, 0.08, 0.1, 0.12]
const PARAM_MAX_STEPS = [2, 3, 4]

# Mesh settings
const EL_SIZE = (40, 20, 20)
const MESH_PARTITION = (1, 1, 1)

# ==============================================================================
# SINGLE RUN FUNCTION
# ==============================================================================
function run_single_optimization(
    distribute,
    γ::Float64,
    max_steps::Int,
    run_id::Int,
    total_runs::Int,
)
    # Derived parameter
    α_coeff = 4 * max_steps * γ

    # Create unique output directory
    write_dir = "./results/parametric_study_MBB_01tol/run$(lpad(run_id, 2, '0'))_γ$(γ)_maxsteps$(max_steps)_α$(α_coeff)/"

    ranks = distribute(LinearIndices((prod(MESH_PARTITION),)))
    i_am_main(ranks) && mkpath(write_dir)

    if i_am_main(ranks)
        println("\n" * "="^60)
        println("RUN $run_id / $total_runs")
        println("  γ = $γ, max_steps = $max_steps, α_coeff = $α_coeff")
        println("  Output: $write_dir")
        println("="^60)
        flush(stdout)
    end

    # ==========================================================================
    # PROBLEM PARAMETERS
    # ==========================================================================
    order = 1
    xmax, ymax, zmax = (2.0, 1.0, 1.0)
    dom = (0, xmax, 0, ymax, 0, zmax)

    γ_reinit = 0.5
    tol = 1 / (5 * order^2) / minimum(EL_SIZE)

    C = isotropic_elast_tensor(3, 1.0, 0.3)

    η_coeff = 2
    vf = 0.4
    iter_mod = 10

    # ==========================================================================
    # FE MODEL
    # ==========================================================================
    model = CartesianDiscreteModel(ranks, MESH_PARTITION, dom, EL_SIZE)
    el_Δ = get_el_Δ(model)
    elem_width_x = xmax / EL_SIZE[1]

    # Boundary conditions
    f_Γ_D1(x) = x[1] ≈ 0.0
    f_Γ_D2(x) = (x[2] ≈ 0.0) && (x[1] >= (xmax - (elem_width_x + eps())))
    force_radius = 0.1
    f_Γ_N(x) =
        (x[2] ≈ ymax) &&
        ((x[1])^2 + (x[3] - zmax / 2)^2 <= force_radius^2 + eps()) &&
        (x[1] >= -eps())

    update_labels!(1, model, f_Γ_D1, "Gamma_D1")
    update_labels!(2, model, f_Γ_D2, "Gamma_D2")
    update_labels!(3, model, f_Γ_N, "Gamma_N")

    # ==========================================================================
    # TRIANGULATION AND MEASURES
    # ==========================================================================
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model, tags = "Gamma_N")
    dΩ = Measure(Ω, 2 * order)
    dΓ_N = Measure(Γ_N, 2 * order)
    vol_D = sum(∫(1)dΩ)

    # Unit force
    A_N = sum(∫(1)dΓ_N)
    println("Area of Gamma_N: $A_N")
    g = VectorValue(0, -1/A_N, 0)

    # ==========================================================================
    # FE SPACES
    # ==========================================================================
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffe_scalar = ReferenceFE(lagrangian, Float64, order)

    V = TestFESpace(
        model,
        reffe;
        dirichlet_tags = ["Gamma_D1", "Gamma_D2"],
        dirichlet_masks = [(true, false, false), (false, true, false)],
    )
    U = TrialFESpace(V, [VectorValue(0.0, 0.0, 0.0), VectorValue(0.0, 0.0, 0.0)])
    V_φ = TestFESpace(model, reffe_scalar)
    V_reg = TestFESpace(model, reffe_scalar; dirichlet_tags = ["Gamma_N"])
    U_reg = TrialFESpace(V_reg, 0)

    # ==========================================================================
    # LEVEL-SET INITIALIZATION
    # ==========================================================================
    levelset_file = "data/3D_2x1x1_MBB/$(TASK_NAME)$(LEVELSET_SUFFIX)"
    φh =
        load_levelset_from_file(levelset_file, V_φ, dom, EL_SIZE[1], EL_SIZE[2], EL_SIZE[3])

    # ==========================================================================
    # ERSATZ MATERIAL
    # ==========================================================================
    interp = SmoothErsatzMaterialInterpolation(η = η_coeff * maximum(el_Δ))
    I, H, DH, ρ = interp.I, interp.H, interp.DH, interp.ρ

    # ==========================================================================
    # WEAK FORM AND FUNCTIONALS
    # ==========================================================================
    a(u, v, φ) = ∫((I ∘ φ) * (C ⊙ ε(u) ⊙ ε(v)))dΩ
    l(v, φ) = ∫(v ⋅ g)dΓ_N

    J(u, φ) = ∫((I ∘ φ) * (C ⊙ ε(u) ⊙ ε(u)))dΩ
    dJ(q, u, φ) = ∫((C ⊙ ε(u) ⊙ ε(u)) * q * (DH ∘ φ) * (norm ∘ ∇(φ)))dΩ
    Vol(u, φ) = ∫(((ρ ∘ φ) - vf) / vol_D)dΩ
    dVol(q, u, φ) = ∫(-1 / vol_D * q * (DH ∘ φ) * (norm ∘ ∇(φ)))dΩ

    # ==========================================================================
    # SOLVERS
    # ==========================================================================
    evo = FiniteDifferenceEvolver(FirstOrderStencil(3, Float64), model, V_φ; max_steps)
    reinit = FiniteDifferenceReinitialiser(
        FirstOrderStencil(3, Float64),
        model,
        V_φ;
        tol,
        γ_reinit,
    )
    ls_evo = LevelSetEvolution(evo, reinit)

    Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv = Vector{PetscScalar}
    solver = PETScLinearSolver()

    state_map = AffineFEStateMap(
        a,
        l,
        U,
        V,
        V_φ;
        assem_U = SparseMatrixAssembler(Tm, Tv, U, V),
        assem_adjoint = SparseMatrixAssembler(Tm, Tv, V, U),
        assem_deriv = SparseMatrixAssembler(Tm, Tv, V_φ, V_φ),
        ls = solver,
        adjoint_ls = solver,
    )

    pcfs = PDEConstrainedFunctionals(
        J,
        [Vol],
        state_map;
        analytic_dJ = dJ,
        analytic_dC = [dVol],
    )

    # ==========================================================================
    # HILBERT REGULARIZATION (α_coeff = 4 * max_steps * γ, derived)
    # ==========================================================================
    α = α_coeff * maximum(el_Δ)
    a_hilb(p, q) = ∫(α^2 * ∇(p) ⋅ ∇(q) + p * q)dΩ

    vel_ext = VelocityExtension(
        a_hilb,
        U_reg,
        V_reg;
        assem = SparseMatrixAssembler(Tm, Tv, U_reg, V_reg),
        ls = PETScLinearSolver(),
    )

    # ==========================================================================
    # OPTIMIZER
    # ==========================================================================
    opt_start_time = time()

    optimiser = AugmentedLagrangian(
        pcfs,
        ls_evo,
        vel_ext,
        φh;
        γ,
        maxiter = 150,
        verbose = i_am_main(ranks),
        constraint_names = [:Vol],
    )

    # ==========================================================================
    # OPTIMIZATION LOOP
    # ==========================================================================
    for (it, uh, φh) in optimiser
        if iszero(it % iter_mod)
            data = ["φ" => φh, "H(φ)" => (H ∘ φh), "uh" => uh]
            writevtk(Ω, write_dir * "out$it", cellfields = data)
        end

        write_history_energy(write_dir * "/history.txt", optimiser.history; ranks = ranks)
    end

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    opt_elapsed = time() - opt_start_time
    history = get_history(optimiser)
    it = history.niter
    uh = get_state(pcfs)

    # Final VTK
    writevtk(
        Ω,
        write_dir * "final",
        cellfields = ["φ" => φh, "H(φ)" => (H ∘ φh), "uh" => uh],
    )

    # Collect results
    final_J = 0.5*history[:J, it]
    final_C = history[:C, it]
    final_vol = final_C[1] * vol_D + vf
    converged_flag = optimiser.converged(optimiser)

    result = (
        run_id = run_id,
        γ = γ,
        max_steps = max_steps,
        α_coeff = α_coeff,
        iterations = it,
        time_s = opt_elapsed,
        converged = converged_flag,
        final_J = final_J,
        final_vol = final_vol,
    )

    # Save summary
    if i_am_main(ranks)
        summary = """
        RUN $run_id COMPLETE
        Parameters: γ=$γ, max_steps=$max_steps, α_coeff=$α_coeff
        Iterations: $it
        Time: $(round(opt_elapsed, digits=2)) s
        Converged: $converged_flag
        Final J: $final_J
        Final Vol: $final_vol
        """
        open(write_dir * "summary.txt", "w") do io
            print(io, summary)
        end
        println(summary)
    end

    return result
end

# ==============================================================================
# MAIN PARAMETRIC STUDY
# ==============================================================================
function run_parametric_study(start_run::Int, end_run::Int)
    # Generate all parameter combinations
    param_combinations =
        [(γ = γ, max_steps = ms) for γ in PARAM_γ for ms in PARAM_MAX_STEPS]
    total_runs = length(param_combinations)

    # Validate range
    start_run = clamp(start_run, 1, total_runs)
    end_run = clamp(end_run, start_run, total_runs)

    println("="^60)
    println("PARAMETRIC STUDY: 3D MBB Level-Set Optimization")
    println("="^60)
    println("Parameters:")
    println("  γ:         $PARAM_γ")
    println("  max_steps: $PARAM_MAX_STEPS")
    println("  α_coeff:   4 * max_steps * γ  (derived)")
    println("Total runs:  $total_runs")
    println("This batch:  $start_run → $end_run ($(end_run - start_run + 1) runs)")
    println("Mesh:        $EL_SIZE")
    println("Started:     $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("="^60)

    # Results storage
    all_results = []

    # PETSc options
    hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
      -ksp_converged_reason -ksp_rtol 1.0e-12"

    study_start = time()

    with_debug() do distribute
        GridapPETSc.with(args = split(hilb_solver_options)) do
            for (run_id, params) in enumerate(param_combinations)
                # Skip runs outside requested range
                if run_id < start_run || run_id > end_run
                    continue
                end

                result = run_single_optimization(
                    distribute,
                    params.γ,
                    params.max_steps,
                    run_id,
                    total_runs,
                )
                push!(all_results, result)

                # Save intermediate results after each run
                save_study_results(all_results, study_start, start_run, end_run)
            end
        end
    end

    # Final summary
    study_elapsed = time() - study_start
    print_final_summary(all_results, study_elapsed)
end

# ==============================================================================
# RESULTS EXPORT
# ==============================================================================
function save_study_results(results, study_start, start_run, end_run)
    results_dir = "./results/parametric_study_MBB_01tol/"
    mkpath(results_dir)

    # Batch-specific filename (safe for parallel execution)
    filename = "results_batch_$(start_run)-$(end_run).csv"

    # CSV export
    open(results_dir * filename, "w") do io
        println(
            io,
            "run_id,γ,max_steps,α_coeff,iterations,time_s,converged,final_J,final_vol",
        )
        for r in results
            println(
                io,
                "$(r.run_id),$(r.γ),$(r.max_steps),$(r.α_coeff),$(r.iterations)," *
                "$(round(r.time_s, digits=2)),$(r.converged),$(r.final_J),$(r.final_vol)",
            )
        end
    end

    # JLD2 backup
    @save results_dir * "results_backup_$(start_run)-$(end_run).jld2" results
end

function print_final_summary(results, total_time)
    println("\n" * "="^60)
    println("PARAMETRIC STUDY COMPLETE")
    println("="^60)
    println("Total time: $(round(total_time / 60, digits=2)) min")
    println("Runs completed: $(length(results))")

    # Find best result (lowest J among converged)
    converged = filter(r -> r.converged, results)
    if !isempty(converged)
        best = argmin(r -> r.final_J, converged)
        println("\nBEST CONVERGED RESULT:")
        println(
            "  Run $(best.run_id): γ=$(best.γ), max_steps=$(best.max_steps), α=$(best.α_coeff)",
        )
        println("  J = $(best.final_J), iterations = $(best.iterations)")
    end

    # Find fastest convergence
    if !isempty(converged)
        fastest = argmin(r -> r.iterations, converged)
        println("\nFASTEST CONVERGENCE:")
        println(
            "  Run $(fastest.run_id): γ=$(fastest.γ), max_steps=$(fastest.max_steps), α=$(fastest.α_coeff)",
        )
        println(
            "  iterations = $(fastest.iterations), time = $(round(fastest.time_s, digits=2)) s",
        )
    end

    println("="^60)
    println("Results saved to: ./results/parametric_study_MBB/results_batch_*.csv")
    println(
        "Merge with: cat results_batch_*.csv | head -1 > all.csv && tail -n +2 -q results_batch_*.csv >> all.csv",
    )
end

# ==============================================================================
# RUN
# ==============================================================================
# Usage:
#   julia parametric_study_MBB.jl           → runs 1-12 (default)
#   julia parametric_study_MBB.jl 10        → run 10 only
#   julia parametric_study_MBB.jl 10 12     → runs 10-12
#
#   JULIA_DEPOT_PATH="$(pwd)/.julia_depot" julia --project=. 3D_2x1x1_MBB-custom_sdf_param_study.jl 1 3

if length(ARGS) == 0
    run_parametric_study(1, 12)
elseif length(ARGS) == 1
    # Single run
    run_id = parse(Int, ARGS[1])
    run_parametric_study(run_id, run_id)
else
    # Range
    start_id = parse(Int, ARGS[1])
    end_id = parse(Int, ARGS[2])
    run_parametric_study(start_id, end_id)
end
