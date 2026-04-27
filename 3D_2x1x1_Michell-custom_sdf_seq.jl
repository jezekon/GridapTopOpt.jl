# ==============================================================================
# TOPOLOGICAL OPTIMIZATION: 3D Michell-type Beam
# ==============================================================================
# Usage: julia 3D_2x1x1_Michell-custom_sdf.jl <file_index>
#   file_index: 1-5 (selects levelset file from predefined list)
#
# Problem Visualization (side view, XY plane at z=0.5):
#
#        Y ↑
#          |
#      1.0 |████████████████████████████████████████████████
#          |█                                              █
#          |█         DESIGN DOMAIN                        █
#          |█         2.0 × 1.0 × 1.0                      █
#          |█                                              █
#       0  |████████████████████████████████████████████████
#          ▓▓                   ↓ F                      ▓▓
#       U1=U2=U3=0      circle r=0.1               U1=U2=U3=0
#       (corners)     at [1,0,0.5], [0,-1,0]          (corners)
#          └─────────────────────────────────────────────────→ X
#          0                   1.0                        2.0
#
#        (Z dimension: 0 to 1.0, perpendicular to page)
#        4 corner supports: 3×3 elements each at (x=0,z=0), (x=0,z=1),
#                           (x=2,z=0), (x=2,z=1)
#
# Boundary Conditions:
#   - Support: 4 corners on bottom face (y=0), 3×3 elements each - U1=U2=U3=0 (clamped)
#   - Point load: Circular region at [1,0,0.5], radius 0.1 - F = [0, -1, 0] N
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
# FILE CONFIGURATION - Select via CLI argument
# ==============================================================================
const LEVELSET_FILES = [
    "3D_2x1x1_Michell_00tol_r2.0",
    "3D_2x1x1_Michell_01tol_r2.0",
    "3D_2x1x1_Michell_02tol_r2.0",
    "3D_2x1x1_Michell_04tol_r2.0",
    "3D_2x1x1_Michell_08tol_r2.0",
    "3D_2x1x1_Michell_16tol_r2.0",
]
const LEVELSET_SUFFIX = "_SDF_B-0.05.jld2"
const DATA_DIR = "data/3D_2x1x1_Michell"

# Parse CLI argument
if length(ARGS) < 1
    println("Usage: julia 3D_2x1x1_Michell-custom_sdf.jl <file_index>")
    println("Available files:")
    for (i, f) in enumerate(LEVELSET_FILES)
        println("  $i: $f")
    end
    exit(1)
end

const FILE_INDEX = parse(Int, ARGS[1])
if FILE_INDEX < 1 || FILE_INDEX > length(LEVELSET_FILES)
    error("Invalid file index: $FILE_INDEX. Must be 1-$(length(LEVELSET_FILES))")
end

const TASK_NAME = LEVELSET_FILES[FILE_INDEX]
const LEVELSET_FILE = joinpath(DATA_DIR, TASK_NAME * LEVELSET_SUFFIX)

println("Selected task: $TASK_NAME")
println("Levelset file: $LEVELSET_FILE")

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
function main(mesh_partition, distribute, el_size, path, task_name)
    ranks = distribute(LinearIndices((prod(mesh_partition),)))

    # Problem parameters
    order = 1
    xmax, ymax, zmax = (2.0, 1.0, 1.0)
    dom = (0, xmax, 0, ymax, 0, zmax)

    # Level-set evolution parameters
    γ = 0.1
    γ_reinit = 0.5
    max_steps = floor(Int, order*minimum(el_size)/5)  # Max. steps per level-set evolution iteration
    # max_steps = 3.0
    tol = 1 / (5 * order^2) / minimum(el_size)

    # Material properties
    C = isotropic_elast_tensor(3, 1.0, 0.3)

    # Optimization parameters
    η_coeff = 2
    α_coeff = 4 * max_steps * γ
    vf = 0.4
    iter_mod = 500

    i_am_main(ranks) && mkpath(path)

    # FE Model
    println("Finite element model")
    model = CartesianDiscreteModel(ranks, mesh_partition, dom, el_size)
    el_Δ = get_el_Δ(model)
    elem_width_x = xmax / el_size[1]
    elem_width_z = zmax / el_size[3]

    # --------------------------------------------------------------------------
    # Boundary conditions - Michell-type with 4 corner supports (clamped)
    # --------------------------------------------------------------------------
    # Corner size: 3 elements in each direction (matching SIMP reference)
    corner_size_x = 3 * elem_width_x
    corner_size_z = 3 * elem_width_z

    # Support: 4 corners on bottom face (y=0), 3×3 elements each, clamped (U1=U2=U3=0)
    f_Γ_D1(x) =
        (x[2] ≈ 0.0) && (
            # Bottom-left-front corner (x≈0, z≈0)
            ((x[1] <= corner_size_x + eps()) && (x[3] <= corner_size_z + eps())) ||
            # Bottom-left-back corner (x≈0, z≈zmax)
            ((x[1] <= corner_size_x + eps()) && (x[3] >= zmax - corner_size_z - eps())) ||
            # Bottom-right-front corner (x≈xmax, z≈0)
            ((x[1] >= xmax - corner_size_x - eps()) && (x[3] <= corner_size_z + eps())) ||
            # Bottom-right-back corner (x≈xmax, z≈zmax)
            (
                (x[1] >= xmax - corner_size_x - eps()) &&
                (x[3] >= zmax - corner_size_z - eps())
            )
        )

    # Force: Circular region on bottom face (y=0), center at [1,0,0.5], dir [0,-1,0]
    force_radius = 0.1
    f_Γ_N(x) =
        (x[2] ≈ 0.0) && ((x[1] - 1.0)^2 + (x[3] - zmax / 2)^2 <= force_radius^2 + eps())

    update_labels!(2, model, f_Γ_D1, "Gamma_D1")
    update_labels!(3, model, f_Γ_N, "Gamma_N")

    visualize_boundary_conditions(model, f_Γ_D1, f_Γ_N, path)

    # Triangulation and measures
    println("Measurements")
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model, tags = "Gamma_N")
    dΩ = Measure(Ω, 2 * order)
    dΓ_N = Measure(Γ_N, 2 * order)
    vol_D = sum(∫(1)dΩ)

    # Unit force
    A_N = sum(∫(1)dΓ_N)
    println("Area of Gamma_N: $A_N")
    g = VectorValue(0, -1/A_N, 0)

    # FE Spaces - Michell-type: 4 corner supports clamped (U1=U2=U3=0)
    println("FE Space")
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    reffe_scalar = ReferenceFE(lagrangian, Float64, order)

    V = TestFESpace(
        model,
        reffe;
        dirichlet_tags = ["Gamma_D1"],
        dirichlet_masks = [(true, true, true)],
    )
    U = TrialFESpace(V, [VectorValue(0.0, 0.0, 0.0)])
    V_φ = TestFESpace(model, reffe_scalar)
    V_reg = TestFESpace(model, reffe_scalar; dirichlet_tags = ["Gamma_N"])
    U_reg = TrialFESpace(V_reg, 0)

    # Level-set initialization from file
    println("Level-set function initialization")
    println("Loading levelset from: $LEVELSET_FILE")
    φh =
        load_levelset_from_file(LEVELSET_FILE, V_φ, dom, el_size[1], el_size[2], el_size[3])
    i_am_main(ranks) && println("✓ load_levelset_from_file DONE!")

    # Ersatz material interpolation
    println("Ersatz model")
    interp = SmoothErsatzMaterialInterpolation(η = η_coeff * maximum(el_Δ))
    I, H, DH, ρ = interp.I, interp.H, interp.DH, interp.ρ

    # Weak formulation
    println("Weak formulation")
    a(u, v, φ) = ∫((I ∘ φ) * (C ⊙ ε(u) ⊙ ε(v)))dΩ
    l(v, φ) = ∫(v ⋅ g)dΓ_N

    # Optimization functionals
    J(u, φ) = ∫((I ∘ φ) * (C ⊙ ε(u) ⊙ ε(u)))dΩ
    dJ(q, u, φ) = ∫((C ⊙ ε(u) ⊙ ε(u)) * q * (DH ∘ φ) * (norm ∘ ∇(φ)))dΩ
    Vol(u, φ) = ∫(((ρ ∘ φ) - vf) / vol_D)dΩ
    dVol(q, u, φ) = ∫(-1 / vol_D * q * (DH ∘ φ) * (norm ∘ ∇(φ)))dΩ

    # Level-set evolution solvers
    println("Solvers for level-set evolution")
    evo = FiniteDifferenceEvolver(FirstOrderStencil(3, Float64), model, V_φ; max_steps)
    reinit = FiniteDifferenceReinitialiser(
        FirstOrderStencil(3, Float64),
        model,
        V_φ;
        tol,
        γ_reinit,
    )
    ls_evo = LevelSetEvolution(evo, reinit)

    # FE Operator setup
    println("Solver and fe operator")
    Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv = Vector{PetscScalar}

    solver = PETScLinearSolver()  # Generic solver for Michell BC
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

    # Hilbert regularization
    println("Hilbert regularization")
    α = α_coeff * maximum(el_Δ)
    a_hilb(p, q) = ∫(α^2 * ∇(p) ⋅ ∇(q) + p * q)dΩ
    vel_ext = VelocityExtension(
        a_hilb,
        U_reg,
        V_reg;
        assem = SparseMatrixAssembler(Tm, Tv, U_reg, V_reg),
        ls = PETScLinearSolver(),
    )

    # Optimizer
    println("Optimizer")
    opt_start_time = time()
    # optimiser = AugmentedLagrangian(
    #     pcfs,
    #     ls_evo,
    #     vel_ext,
    #     φh;
    #     γ,
    #     verbose = i_am_main(ranks),
    #     constraint_names = [:Vol],
    # )

    optimiser = HilbertianProjection(
        pcfs,
        ls_evo,
        vel_ext,
        φh;
        maxiter = 200,
        verbose = i_am_main(ranks),
        constraint_names = [:Vol],
        converged = m ->
            GridapTopOpt.default_hp_converged(m; J_tol = 0.002, C_tol = 0.001),
    )

    # Optimization loop
    println("Optimization")
    for (it, uh, φh) in optimiser
        data = ["φ" => φh, "H(φ)" => (H ∘ φh), "|∇(φ)|" => (norm ∘ ∇(φh)), "uh" => uh]

        if iszero(it % iter_mod)
            writevtk(Ω, path * "out$it", cellfields = data)

            φ_values = collect(get_free_dof_values(φh))
            nx, ny, nz = el_size .+ 1
            x = range(dom[1], dom[2], length = nx)
            y = range(dom[3], dom[4], length = ny)
            z = range(dom[5], dom[6], length = nz)

            vtk_file = vtk_grid(path * "clean_$it", x, y, z)
            vtk_point_data(vtk_file, reshape(φ_values, nx, ny, nz), "phi")
            vtk_save(vtk_file)
        end

        write_history_energy(path*"/history.txt", optimiser.history; ranks = ranks)
    end

    # Final output
    opt_elapsed = time() - opt_start_time
    history = get_history(optimiser)
    it = history.niter
    uh = get_state(pcfs)

    writevtk(
        Ω,
        path * "out$it",
        cellfields = [
            "φ" => φh,
            "H(φ)" => (H ∘ φh),
            "|∇(φ)|" => (norm ∘ ∇(φh)),
            "uh" => uh,
        ],
    )

    # Final VTI export
    φ_values = collect(get_free_dof_values(φh))
    nx, ny, nz = el_size .+ 1
    x = range(dom[1], dom[2], length = nx)
    y = range(dom[3], dom[4], length = ny)
    z = range(dom[5], dom[6], length = nz)

    vtk_file = vtk_grid(path * task_name * "_LS", x, y, z)
    vtk_point_data(vtk_file, reshape(φ_values, nx, ny, nz), "phi")
    vtk_save(vtk_file)

    if i_am_main(ranks)
        final_J = 0.5*history[:J, it]
        final_C = history[:C, it]
        final_vol = final_C[1] * vol_D + vf
        converged_flag = optimiser.converged(optimiser)

        summary = """
==================================================
LEVEL-SET TOPOLOGY OPTIMIZATION SUMMARY
==================================================
Task name:           $task_name
Iterations:          $it
Total time:          $(round(opt_elapsed, digits=2)) s
Converged:           $(converged_flag ? "Yes" : "No")
Final compliance:    $final_J
Final volume:        $final_vol
Generated:           $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
==================================================
"""
        open(path * task_name * "_LS-summary.txt", "w") do io
            print(io, summary)
        end
        println(summary)
    end
end

# ==============================================================================
# RUN
# ==============================================================================
function run_optimization(elx, ely, elz, task_name)
    write_dir = "./results/$(task_name)/"

    Px, Py, Pz = 1, 1, 1
    with_debug() do distribute
        mesh_partition = (Px, Py, Pz)
        el_size = (elx, ely, elz)

        hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
          -ksp_converged_reason -ksp_rtol 1.0e-12"

        GridapPETSc.with(args = split(hilb_solver_options)) do
            main(mesh_partition, distribute, el_size, write_dir, task_name)
        end
    end
end

run_optimization(40, 20, 20, TASK_NAME)

# JULIA_DEPOT_PATH="$(pwd)/.julia_depot" julia --project=. 3D_2x1x1_Michell-custom_sdf_seq.jl 1
