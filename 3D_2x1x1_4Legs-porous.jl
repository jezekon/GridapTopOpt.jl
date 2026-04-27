# ==============================================================================
# TOPOLOGICAL OPTIMIZATION: Minimum Elastic Compliance in 3D
# ==============================================================================
# This script solves a topological optimization problem using the level-set method
# and augmented Lagrangian approach. The goal is to find the optimal material
# distribution in a 3D beam to make it as stiff as possible (minimal deformation under load).
# ==============================================================================

using Gridap,                # Main library for FEM analysis
    Gridap.MultiField,       # Support for multi-physics problems
    GridapDistributed,       # Distributed computing (MPI)
    GridapPETSc,            # PETSc library solvers
    GridapSolvers,          # Advanced solvers
    PartitionedArrays,      # Distributed arrays for parallelization
    GridapTopOpt,           # Library for topological optimization
    SparseMatricesCSR,      # Efficient sparse matrix format
    WriteVTK,
    Dates

include("src/CustomModif/CustomModif.jl")

# ==============================================================================
# INPUT PARAMETERS FROM COMMAND LINE
# ==============================================================================
# Number of elements in each direction (x, y, z)
# global elx = parse(Int, ARGS[1])  # Number of elements in x direction
# global ely = parse(Int, ARGS[2])  # Number of elements in y direction
# global elz = parse(Int, ARGS[3])  # Number of elements in z direction

# MPI process distribution in 3D grid
# global Px = parse(Int, ARGS[4])   # Number of processes in x direction
# global Py = parse(Int, ARGS[5])   # Number of processes in y direction
# global Pz = parse(Int, ARGS[6])   # Number of processes in z direction

# Output directory for results
# global write_dir = ARGS[7]

# ==============================================================================
# MAIN FUNCTION - OPTIMIZATION PROBLEM DOCUMENTATION
# ==============================================================================
"""
  (MPI) Minimum elastic compliance with augmented Lagrangian method in 3D.

  Optimization problem:
      Min J(Ω) = ∫ C ⊙ ε(u) ⊙ ε(u) dΩ
        Ω
    s.t., Vol(Ω) = vf,                           # Constraint on material volume
          ⎡u∈V=H¹(Ω;u(Γ_D)=0)³,                  # Displacement in Sobolev space
          ⎣∫ C ⊙ ε(u) ⊙ ε(v) dΩ = ∫ v⋅g dΓ_N, ∀v∈V.  # Equilibrium (weak formulation)

  Where:
    - Ω is the optimized domain (beam)
    - u is the displacement field
    - C is the elastic tensor (material properties)
    - ε is the strain tensor
    - vf is the required volume fraction of material
    - Γ_D is the Dirichlet boundary (fixed support)
    - Γ_N is the Neumann boundary (load with force g)
"""
function main(mesh_partition, distribute, el_size, path)
    # Create distributed processes for MPI parallelization
    ranks = distribute(LinearIndices((prod(mesh_partition),)))

    # ==========================================================================
    # PROBLEM PARAMETERS
    # ==========================================================================
    println("Parameters")
    order = 1                          # Polynomial order for FE approximation (linear elements)
    xmax, ymax, zmax=(2.0, 1.0, 1.0)   # Beam dimensions [length x height x depth]
    prop_Γ_N = 0.2                     # Proportion of boundary length with applied force (20%)
    dom = (0, xmax, 0, ymax, 0, zmax)  # Problem domain

    # Level-set evolution parameters
    γ = 0.1                            # Time step for Hamilton-Jacobi equation
    γ_reinit = 0.5                     # Parameter for level-set function reinitialization
    max_steps = floor(Int, order*minimum(el_size)/5)  # Max. steps per level-set evolution iteration
    # max_steps = 3.0  # Max. steps per level-set evolution iteration
    tol = 1/(5order^2)/minimum(el_size)              # Tolerance for reinitialization

    # Material properties (isotropic material)
    C = isotropic_elast_tensor(3, 1.0, 0.3)  # E=1.0 (Young's modulus), ν=0.3 (Poisson's ratio)

    # Optimization parameters
    η_coeff = 2                        # Coefficient for "ersatz material" interpolation
    α_coeff = 4max_steps*γ             # Coefficient for Hilbert regularization of velocity field
    vf = 0.4                           # Target volume fraction (40% material)
    iter_mod = 500                     # Frequency of result writing (every 10 iterations)

    # Create output directory (main process only)
    i_am_main(ranks) && mkpath(path)

    # ==========================================================================
    # FINITE ELEMENT MODEL SETUP (FE SETUP)
    # ==========================================================================
    println("Finite element model")
    # Create regular Cartesian mesh distributed among MPI processes
    model = CartesianDiscreteModel(ranks, mesh_partition, dom, el_size);

    # Calculate element sizes (for regularization)
    el_Δ = get_el_Δ(model)

    # Define boundary conditions using functions
    # Size of each corner fixation
    fix_size = 0.3  # 0.3 x 0.3 squares at corners

    # Dirichlet BC: 4 corner fixations on x=0 face
    # Corners are at (0,0,0), (0,1,0), (0,0,1), (0,1,1) of the 1x1 face
    f_Γ_D(x) =
        (x[1] ≈ 0.0) && (
            # Bottom-left corner (y≈0, z≈0)
            (x[2] <= fix_size + eps() && x[3] <= fix_size + eps()) ||
            # Bottom-right corner (y≈ymax, z≈0)
            (x[2] >= ymax - fix_size - eps() && x[3] <= fix_size + eps()) ||
            # Top-left corner (y≈0, z≈zmax)
            (x[2] <= fix_size + eps() && x[3] >= zmax - fix_size - eps()) ||
            # Top-right corner (y≈ymax, z≈zmax)
            (x[2] >= ymax - fix_size - eps() && x[3] >= zmax - fix_size - eps())
        )

    # Neumann BC: circular region with radius 0.2 centered on x=xmax face
    load_radius = 0.1
    f_Γ_N(x) =
        (x[1] ≈ xmax) && (
            # Circle equation: (y - y_center)² + (z - z_center)² ≤ r²
            (x[2] - ymax/2)^2 + (x[3] - zmax/2)^2 <= load_radius^2 + eps()
        )

    # Label edges in model using defined functions
    update_labels!(1, model, f_Γ_D, "Gamma_D")
    update_labels!(2, model, f_Γ_N, "Gamma_N")
    visualize_boundary_conditions(model, f_Γ_D, f_Γ_N, path)

    # ==========================================================================
    # TRIANGULATION AND INTEGRATION MEASURES
    # ==========================================================================
    println("Measurements")
    Ω = Triangulation(model)                          # Entire domain
    Γ_N = BoundaryTriangulation(model, tags = "Gamma_N")  # Load boundary
    dΩ = Measure(Ω, 2*order)                         # Integration measure over volume (Gaussian quadrature)
    dΓ_N = Measure(Γ_N, 2*order)                      # Integration measure over boundary
    vol_D = sum(∫(1)dΩ)                               # Total domain volume

    # Unit force
    A_N = sum(∫(1)dΓ_N)
    println("Area of Gamma_N: $A_N")
    g = VectorValue(0, 0, -1/A_N)

    # ==========================================================================
    # DEFINITION OF FE SPACES
    # ==========================================================================
    println("FE Space")
    # Reference element for vector field (3D displacement)
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
    # Reference element for scalar field (level-set function)
    reffe_scalar = ReferenceFE(lagrangian, Float64, order)

    # Test space for displacement (with Dirichlet condition on Gamma_D)
    V = TestFESpace(model, reffe; dirichlet_tags = ["Gamma_D"])
    # Trial space for displacement (zero displacement on Gamma_D)
    U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))

    # Space for level-set function (no boundary conditions)
    V_φ = TestFESpace(model, reffe_scalar)

    # Space for velocity field regularization (with condition on Gamma_N)
    V_reg = TestFESpace(model, reffe_scalar; dirichlet_tags = ["Gamma_N"])
    U_reg = TrialFESpace(V_reg, 0)

    # ==========================================================================
    # LEVEL-SET FUNCTION INITIALIZATION
    # ==========================================================================
    println("Level-set function initialization")
    # Initial level-set function: sinusoidal pattern with 4 periods, offset 0.2
    # Level-set φ < 0 represents material, φ > 0 represents void
    φh = interpolate(initial_lsf(4, 0.2), V_φ)

    # ==========================================================================
    # MATERIAL INTERPOLATION ("ERSATZ MATERIAL")
    # ==========================================================================
    println("Ersatz model")
    # Smooth Ersatz Material Interpolation: smooth transition between material and void
    # η determines the thickness of transition zone
    interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
    I, H, DH, ρ = interp.I, interp.H, interp.DH, interp.ρ
    # I(φ)  - interpolation function for stiffness (0 for void, 1 for material)
    # H(φ)  - Heaviside function (0 or 1)
    # DH(φ) - derivative of Heaviside function (Dirac delta)
    # ρ(φ)  - density function for volume constraint

    # ==========================================================================
    # WEAK FORMULATION OF ELASTIC PROBLEM
    # ==========================================================================
    println("Weak formulation")
    # Bilinear form: stiffness × strain of test function
    a(u, v, φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    # Linear form: applied force on boundary
    l(v, φ) = ∫(v⋅g)dΓ_N

    # ==========================================================================
    # OPTIMIZATION FUNCTIONALS
    # ==========================================================================
    # Objective function: elastic energy (compliance - we want to minimize)
    J(u, φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(u)))dΩ

    # Derivative of objective function with respect to level-set (shape derivative)
    dJ(q, u, φ) = ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

    # Volume constraint: normalized deviation from target volume
    Vol(u, φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ

    # Derivative of volume constraint with respect to level-set
    dVol(q, u, φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

    # ==========================================================================
    # SOLVERS FOR LEVEL-SET EVOLUTION
    # ==========================================================================
    println("Solvers for level-set evolution")
    # Finite Difference Evolver: solves Hamilton-Jacobi equation (level-set evolution)
    evo = FiniteDifferenceEvolver(FirstOrderStencil(3, Float64), model, V_φ; max_steps)

    # Reinitializer: maintains level-set as a distance function (|∇φ| = 1)
    reinit = FiniteDifferenceReinitialiser(
        FirstOrderStencil(3, Float64),
        model,
        V_φ;
        tol,
        γ_reinit,
    )

    # Combined level-set evolution (evolution + reinitialization)
    ls_evo = LevelSetEvolution(evo, reinit)

    # ==========================================================================
    # SOLVER AND FE OPERATOR SETUP
    # ==========================================================================
    println("Solver and fe operator")
    # Matrix and vector types for PETSc
    Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv = Vector{PetscScalar}

    # Elasticity solver (solves equilibrium equation for given φ)
    solver = ElasticitySolver(V)

    # State map: mapping from level-set function φ to system state (displacement u)
    state_map = AffineFEStateMap(
        a,                # Bilinear form
        l,                # Linear form
        U,                # Trial space
        V,                # Test space
        V_φ;              # Space for level-set
        assem_U = SparseMatrixAssembler(Tm, Tv, U, V),           # Assembler for forward problem
        assem_adjoint = SparseMatrixAssembler(Tm, Tv, V, U),     # Assembler for adjoint problem
        assem_deriv = SparseMatrixAssembler(Tm, Tv, V_φ, V_φ),   # Assembler for derivatives
        ls = solver,            # Linear solver for forward problem
        adjoint_ls = solver,    # Linear solver for adjoint problem
    )

    # PDE Constrained Functionals: optimization functionals with PDE constraint
    pcfs = PDEConstrainedFunctionals(
        J,              # Objective function
        [Vol],          # List of constraints
        state_map;      # State mapping
        analytic_dJ = dJ,      # Analytic derivative of objective function
        analytic_dC = [dVol],  # Analytic derivative of constraints
    )

    # ==========================================================================
    # HILBERT REGULARIZATION OF VELOCITY FIELD
    # ==========================================================================
    println("Hilbert regularization")
    # Regularization parameter (dependent on element size)
    α = α_coeff*maximum(el_Δ)

    # Regularization bilinear form: H¹ norm of velocity field
    # Ensures smoothness and stability of evolution
    a_hilb(p, q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;

    # Velocity Extension: extension of velocity field from boundary to entire domain
    vel_ext = VelocityExtension(
        a_hilb,      # Regularization form
        U_reg,       # Trial space
        V_reg;       # Test space
        assem = SparseMatrixAssembler(Tm, Tv, U_reg, V_reg),
        ls = PETScLinearSolver(),  # PETSc solver for regularization problem
    )

    # ==========================================================================
    # OPTIMIZER - AUGMENTED LAGRANGIAN METHOD
    # ==========================================================================
    println("Optimizer")
    opt_start_time = time()

    optimiser = AugmentedLagrangian(
        pcfs,        # PDE-constrained functionals
        ls_evo,      # Level-set evolution
        vel_ext,     # Velocity regularization
        φh;          # Initial level-set function
        γ,           # Evolution time step
        # maxiter = 1,  # ← Přidej toto pro test
        verbose = i_am_main(ranks),  # Output only from main process
        constraint_names = [:Vol],   # Constraint names for output
    )

    # ==========================================================================
    # OPTIMIZATION LOOP
    # ==========================================================================
    # Iterate through optimizer (each iteration returns current state)
    println("Optimization")
    for (it, uh, φh) in optimiser
        data = ["φ"=>φh, "H(φ)" => (H ∘ φh), "|∇(φ)|" => (norm ∘ ∇(φh)), "uh" => uh]

        if iszero(it % iter_mod)
            writevtk(Ω, path*"out$it", cellfields = data)

            # Export do VTI
            φ_values = collect(get_free_dof_values(φh))
            nx, ny, nz = el_size .+ 1
            x = range(dom[1], dom[2], length = nx)
            y = range(dom[3], dom[4], length = ny)
            z = range(dom[5], dom[6], length = nz)

            vtk_file = vtk_grid(path*"clean_$it", x, y, z)
            vtk_point_data(vtk_file, reshape(φ_values, nx, ny, nz), "phi")
            vtk_save(vtk_file)
        end

        write_history_energy(path*"/history.txt", optimiser.history; ranks = ranks)
    end

    # ==========================================================================
    # FINAL OUTPUT
    # ==========================================================================
    opt_elapsed = time() - opt_start_time
    history = get_history(optimiser)
    it = history.niter
    uh = get_state(pcfs)

    writevtk(
        Ω,
        path*"out$it",
        cellfields = ["φ"=>φh, "H(φ)"=>(H ∘ φh), "|∇(φ)|"=>(norm ∘ ∇(φh)), "uh"=>uh],
    )

    # Final VTI export
    φ_values = collect(get_free_dof_values(φh))
    nx, ny, nz = el_size .+ 1
    x = range(dom[1], dom[2], length = nx)
    y = range(dom[3], dom[4], length = ny)
    z = range(dom[5], dom[6], length = nz)

    vtk_file = vtk_grid(path * "3D_2x1x1_4Legs_porous_LS", x, y, z)
    vtk_point_data(vtk_file, reshape(φ_values, nx, ny, nz), "phi")
    vtk_save(vtk_file)

    if i_am_main(ranks)
        final_J = 0.5*history[:J, it]
        final_C = history[:C, it]
        final_vol = final_C[1] * vol_D + vf
        # converged_flag = converged(optimiser)
        converged_flag = optimiser.converged(optimiser)

        task_name = basename(rstrip(path, '/'))
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
        open(path * "3D_2x1x1_4Legs_porous_LS-summary.txt", "w") do io
            print(io, summary)
        end
        println(summary)
    end
end

# ==============================================================================
# RUN WITH MPI
# ==============================================================================
function run_gridap_optimization_beam_3D(elx, ely, elz, write_dir)
    # Nastavení pro sériový běh
    Px, Py, Pz = 1, 1, 1
    # with_mpi() do distribute
    with_debug() do distribute
        # MPI process distribution in 3D grid
        mesh_partition = (Px, Py, Pz)

        # Number of elements in each direction
        el_size = (elx, ely, elz)

        # Solver settings for Hilbert regularization (algebraic multigrid)
        hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
          -ksp_converged_reason -ksp_rtol 1.0e-12"

        # Run with PETSc settings
        GridapPETSc.with(args = split(hilb_solver_options)) do
            main(mesh_partition, distribute, el_size, write_dir)
        end
    end
end

run_gridap_optimization_beam_3D(40, 20, 20, "./results/3D_2x1x1_4Legs-porous/")
# ==============================================================================
# END OF SCRIPT
# ==============================================================================
# Example execution:
# mpirun -np 8 julia elastic_compliance_ALM_3D.jl 40 20 20 2 2 2 ./results/
#
# Where:
#   40 20 20 - number of elements in x, y, z directions
#   2 2 2    - MPI process distribution (8 processes total)
#   ./results/ - output directory
# ============================================================================== 
#
# JULIA_DEPOT_PATH="$(pwd)/.julia_depot" julia --project=. 3D_2x1x1_4Legs-porous.jl
