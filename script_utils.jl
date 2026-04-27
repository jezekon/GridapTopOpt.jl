# ==============================================================================
# Local helpers shared by the 3D_2x1x1_* scripts.
# Boundary-condition VTK export and history writer (deformation energy form).
# ==============================================================================

using Printf
using GridapTopOpt: OptimiserHistory

"""
    visualize_boundary_conditions(model, f_Γ_D, f_Γ_N, output_path)

Export all boundaries to a single VTK file with a `bc_type` cell field:
0 = free, 1 = Dirichlet (Γ_D), 2 = Neumann (Γ_N).
"""
function visualize_boundary_conditions(
    model,
    f_Γ_D::Function,
    f_Γ_N::Function,
    output_path::String,
)
    Γ = BoundaryTriangulation(model)

    function bc_marker(x)
        if f_Γ_D(x)
            return 1.0
        elseif f_Γ_N(x)
            return 2.0
        else
            return 0.0
        end
    end

    bc_field = CellField(bc_marker, Γ)
    writevtk(Γ, output_path * "boundary_conditions", cellfields = ["bc_type" => bc_field])

    println("✓ Exported: $(output_path)boundary_conditions.vtu")
end

"""
    write_history_energy(path, h::OptimiserHistory; ranks=nothing)

Drop-in replacement for `write_history` that converts compliance `J`
to deformation energy (0.5*J).
"""
function write_history_energy(path::String, h::OptimiserHistory; ranks = nothing)
    if i_am_main(ranks)
        open(path, "w") do f
            ks = keys(first(h))
            content = join(ks, ", ")
            for s in h
                content *=
                    "\n" *
                    join([@sprintf("%.4e", k == :J ? 0.5 * s[k] : s[k]) for k in ks], ", ")
            end
            write(f, content)
        end
    end
end
