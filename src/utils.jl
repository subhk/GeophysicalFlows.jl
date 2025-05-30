"""
    lambdipole(U, R, grid::TwoDGrid; center=(mean(grid.x), mean(grid.y))

Return the two-dimensional vorticity field of the Lamb dipole with strength `U` and radius `R`,
on `grid`. The dipole is centered at `center = (xc, yc)`; default value for `center` is the center
of the domain.
"""
function lambdipole(U, R, grid::TwoDGrid{T, A}; center=(mean(grid.x), mean(grid.y))) where {T, A}
  firstzero = 3.8317059702075123156 # first zero of besselj1
  k = firstzero / R # dipole wavenumber for radius R in terms of first zero of besselj
  q₀ = -2U * k / besselj(0, k * R) # dipole amplitude for strength U and radius R
  x, y = gridpoints(grid)
  xc, yc = center
  r = @. sqrt((x - xc)^2 + (y - yc)^2)
  besseljorder1 = CUDA.@allowscalar A([besselj1(k * r[i, j]) for j=1:grid.ny, i=1:grid.nx])
  q = @. q₀ * besseljorder1 * (y - yc) / r
  @. q = ifelse(r ≥ R, 0, q)

  return q
end

"""
    peakedisotropicspectrum(grid, kpeak, E₀; mask = ones(size(grid.Krsq)), allones = false)

Generate a random two-dimensional relative vorticity field ``q(x, y)`` with Fourier spectrum
peaked around a central non-dimensional wavenumber `kpeak` and normalized so that its total
kinetic energy is `E₀`.
"""
function peakedisotropicspectrum(grid::TwoDGrid{T, A}, kpeak::Real, E₀::Real;
                                 mask = ones(size(grid.Krsq)), allones = false) where {T, A}

  grid.Lx !== grid.Ly && error("the domain is not square")

  k₀ = kpeak * 2π / grid.Lx
  modk = sqrt.(grid.Krsq)
  modψ = A(zeros(T, (grid.nk, grid.nl)))
  modψ = @. (modk^2 * (1 + (modk / k₀)^4))^(-0.5)
  CUDA.@allowscalar modψ[1, 1] = 0.0

  if allones
    ψh = modψ
  else
    phases = randn(Complex{T}, size(grid.Krsq))
    phases_real, phases_imag = real.(phases), imag.(phases)
    phases = A(phases_real) + im * A(phases_imag)
    ψh = @. phases * modψ
  end

  ψh .*= A(mask)
  energy_initial = sum(grid.Krsq .* abs2.(ψh)) / (grid.nx * grid.ny)^2
  ψh *= sqrt(E₀ / energy_initial)

  return A(irfft(- grid.Krsq .* ψh, grid.nx))
end
