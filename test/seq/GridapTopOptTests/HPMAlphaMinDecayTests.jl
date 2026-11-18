module HPMAlphaMinDecayTests
# Checks the adaptive decay of the lower bound α_min in `compute_α` of the Hilbertian
# projection method, for one constraint. The rate λ is small, so the natural ∑α² lies
# below every bound and the lower bound is active. `compute_α` then sets ∑α² to the
# effective bound α_min*min(1,(|C|/α_min_decay_tol)^α_min_decay_power) itself, and
# α = sign(C)*√∑α². The expected values follow from this formula. Power 1 must give
# exactly the bound of the earlier patch, α_min*|C|/α_min_decay_tol.
using Test
using LinearAlgebra
using GridapTopOpt: compute_α

# One constraint with sensitivity v in the inner product K = I. For one vector the
# Gram-Schmidt orthogonalisation returns the vector itself.
function alpha_for(C; α_min=0.1, tol=0.01, power=1)
  K = Matrix(1.0I,3,3)
  v = [1.0,2.0,2.0]
  normsq = [dot(v,K*v)]
  P = zeros(3)
  λ = 1e-6
  α, ∑α², debug_flag = compute_α([C],[copy(v)],[v],normsq,K,P,λ,α_min,1.0,tol,power)
  return α[1], ∑α², debug_flag
end

@testset "Decay off (α_min_decay_tol = 0)" begin
  for power in (1,2)
    α, ∑α², flag = alpha_for(1e-4;tol=0.0,power)
    @test flag == 2
    @test ∑α² == 0.1
    @test α ≈ sqrt(0.1)
  end
end

@testset "Full bound for |C| ≥ α_min_decay_tol" begin
  for power in (1,2), C in (0.01,0.05,-0.05)
    α, ∑α², flag = alpha_for(C;power)
    @test flag == 2
    @test ∑α² == 0.1
    @test α ≈ sign(C)*sqrt(0.1)
  end
end

@testset "Power 1: bound of the earlier patch, α ∝ √|C|" begin
  α, ∑α², flag = alpha_for(2e-3;power=1)
  @test flag == 2
  @test ∑α² == 0.1*(2e-3/0.01)
  @test α ≈ sqrt(0.1*0.2)
  α_4, _, _ = alpha_for(4e-3;power=1)
  @test α_4/α ≈ sqrt(2)
end

@testset "Power 2: α linear in C" begin
  α, ∑α², flag = alpha_for(2e-3;power=2)
  @test flag == 2
  @test ∑α² ≈ 0.1*0.2^2
  @test α ≈ sqrt(0.1)*0.2
  α_4, _, _ = alpha_for(4e-3;power=2)
  @test α_4/α ≈ 2
  α_neg, _, _ = alpha_for(-2e-3;power=2)
  @test α_neg ≈ -α
end

end # module
