# ##
# Surface Quasi-Geostrophy:
# This code solves an exmaple problem from Held et al JFM 1995
# Imposed condition is the 'surface buoyancy' given by a smooth elliptic vortex.
# This equations are Eq.(2) of the paper.

using FourierFlows, Plots, Statistics, Printf, Random

using FFTW: irfft, rfft
using Statistics: mean
using Random: seed!

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

using CUDA
#    Reexport

#nothingfunction(args...) = nothing


# parameters 
#
# Let's define a struct which contain the 
# main parameters of the problem
# struct Params{T} <: AbstractParams
#      ν :: T          # hyperviscosity coefficient
#     nν :: Int       # Order of the hyperviscosity
# end 
# nothing #hide

struct Params{T} <: AbstractParams
    ν :: T         # Hyperviscosity coefficient
   nν :: Int       # Order of the hyperviscous operator
end
nothing #hide

# variables
# conatins all the problem variables in physical 
# and Fourier (transformed) space :  Aphys, Atrans
struct Vars{Aphys, Atrans} <: AbstractVars
    b  :: Aphys
    u  :: Aphys
    v  :: Aphys
    bh :: Atrans
    uh :: Atrans
    vh :: Atrans
end
nothing #hide

# constructor creates the size of the variables
# based on the 'grids'
function Vars(::Dev, grid) where Dev
    T = eltype(grid)
    @devzeros Dev T (grid.nx, grid.ny) b u v
    @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh
    return Vars(b, u, v, bh, uh, vh) 
end
nothing #hide

# constructing the equation:
# here the linear part is described ny hyperviscosity part, 
# and the nonlinear part is computed by the 'calcN!'
# the equations are in Fourier space
function Equation(paramas::Params, grid::AbstractGrid)
    T = eltype(grid)
    L = @. -paramas.ν*grid.Krsq^paramas.nν
    CUDA.@allowscalar L[1,1] = 0
    return FourierFlows.Equation(L, calcN!, grid)
end
nothing #hide

# constructing the nonlinear part of the problem
# ofcourse in Fourier space.
function calcN!( N, sol, t, clock, vars, params, grid )
    @. vars.bh = sol
    @. vars.uh =   im * grid.l  * sqrt( grid.invKrsq ) * sol
    @. vars.vh = - im * grid.kr * sqrt( grid.invKrsq ) * sol

    # doing inverse-fft (I suppose!!)
    ldiv!( vars.u, grid.rfftplan, vars.uh )
    ldiv!( vars.v, grid.rfftplan, vars.vh )
    ldiv!( vars.b, grid.rfftplan, vars.bh )

    ub, ubh = vars.u, vars.uh
    vb, vbh = vars.v, vars.vh

    @. ub *= vars.b
    @. vb *= vars.b 

    # compute the advection term: u*bx + v*by
    mul!( ubh, grid.rfftplan, ub )
    mul!( vbh, grid.rfftplan, vb )

    @. N = - im * grid.kr * ubh - im * grid.l * vbh 
    return nothing
end 
nothing #hide

# All ready to go, my master!
# Hold on to the moment my apprentice, let me create some helpers to do thing smoothly.

# function which updates the variables in current time!
function updatevars!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol
    
    @. vars.bh = sol
    @. vars.uh =   im * grid.l  * sqrt( grid.invKrsq ) * sol
    @. vars.vh = - im * grid.kr * sqrt( grid.invKrsq ) * sol    

    # doing inverse-fft 
    ldiv!( vars.u, grid.rfftplan, deepcopy(vars.uh) )
    ldiv!( vars.v, grid.rfftplan, deepcopy(vars.vh) )
    ldiv!( vars.b, grid.rfftplan, deepcopy(vars.bh) )

    return nothing
end

# set buoyancy as the solution and upadate all other variables.
function set_b!(prob, b)
    mul!( prob.sol, prob.grid.rfftplan, b )
    CUDA.@allowscalar prob.sol[1,1] = 0  # domain average gives zero
    
    updatevars!(prob)

    return nothing
end

# calculating surface-averaged kinetic energy.
# In SQG formulation, this is exactly half of the buoyancy variance.
@inline function kinetic_energy(prob)
    
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    @. vars.uh =   im * grid.l  * sqrt( grid.invKrsq ) * sol
    @. vars.vh = - im * grid.kr * sqrt( grid.invKrsq ) * sol 
    
    ke_h = vars.uh

    @. ke_h = 0.5*( abs2(vars.uh) + abs2(vars.vh) )

    return 1/(grid.Lx * grid.Ly) * parsevalsum( ke_h, grid )
end

# ## Choosing a device: CPU or GPU
dev = CPU()    # Device (CPU/GPU)
nothing # hide

# numerical parameters
     nx = 512                           # grid resolution
     ny = 512  
stepper = "FilteredRK4"                 # time-stepping scheme
     dt = 0.01                          # timesep (in seconds)
     Tf = 60.                           # total time of the simulation
 nsteps = Int( Tf/dt )                  # total iteration of the simulation
  nplot = 10 #round( Int, nsteps/100 )      # step to save data
nothing # hide
  
# physical parameters
     Lx = 2π                # domain size 
     Ly = 2π
     nν = 4
      ν = 1.e-18
nothing # hide

# creating the grids struct
grid = TwoDGrid( dev, nx, Lx, ny, Ly; T=Float64)

# creating parameters struct
params = Params(ν, nν) 

vars = Vars(dev, grid)
equation = Equation(params, grid)
prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)

# Let's create some shortcuts.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = prob.grid.x, prob.grid.y
nothing # hide

# Here comes the initial ccondition, and set to fly.
X, Y = gridpoints(grid)
b0 = @. exp( -(X^2 + 4*Y^2) )

set_b!(prob, b0)
nothing #hide

#### plot the initial condition
heatmap(x, y, prob.vars.b',
    aspectratio = 1,
             c  = :deep, 
           clim = [0, 1],   
           xlim = ( -grid.Lx/2, grid.Lx/2 ), 
           ylim = ( -grid.Ly/2, grid.Ly/2 ),
         xticks = -3:3, 
         yticks = -3:3, 
         xlabel = "x", 
         ylabel = "y", 
          title = "Initial distributon of buoyancy", 
     framestyle = :box)

# creaing diganosis
KE = Diagnostic(kinetic_energy, prob; nsteps=nsteps)
diags = KE
nothing #

### ouput 
base_filename = string( "surfaceqg_Held_", nx )
datapath = "./"
plotpath = "./"

dataname = joinpath(datapath, base_filename)
plotname = joinpath(datapath, base_filename)
nothing #hide

if !isdir(plotpath); mkdir(plotpath); end
if !isdir(datapath); mkdir(datapath); end
nothing #hide

# get the output
get_sol(prob) = sol  # extract the fft-solution
get_u(prob) = irfft( im * grid.l  .* sqrt( grid.invKrsq ) .* sol, grid.nx )
out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide

### Visulation
function plot_output(prob)
    bₛ = prob.vars.b
    uₛ = prob.vars.u
    vₛ = prob.vars.v
  
    pbₛ = heatmap(x, y, bₛ',
         aspectratio = 1,
                   c = :deep,
                clim = (0, 1),
               xlims = (-grid.Lx/2, grid.Lx/2),
               ylims = (-grid.Ly/2, grid.Ly/2),
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "buoyancy bₛ",
          framestyle = :box)
  
    pKE = plot(1,
               label = "kinetic energy ∫½(uₛ²+vₛ²)dxdy/L²",
           linewidth = 2,
              legend = :bottomright,
               alpha = 0.7,
               xlims = (0, Tf),
               ylims = (0, 1e-2),
              xlabel = "t")
  
    # pb² = plot(1,
    #            label = "buoyancy variance ∫bₛ²dxdy/L²",
    #        linecolor = :red,
    #           legend = :bottomright,
    #        linewidth = 2,
    #            alpha = 0.7,
    #            xlims = (0, tf),
    #            ylims = (0, 2e-2),
    #           xlabel = "t")
  
    # layout = @layout [a{0.5w} Plots.grid(2, 1)]
    layout = @layout grid(2, 1)
    p = plot(pbₛ, pKE, layout=layout, size = (900, 500), dpi=150)
  
    return p
end
nothing # hide

# Time-stepping forward in time

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nplot)

    # CFL condition
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
  
    #if j%(500/nplot)==0
    if j%nplot==0
        log1 = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
            clock.step, clock.t, cfl, (time()-startwalltime)/60)
        
        log2 = @sprintf("kinetic energy: %.2e", KE.data[KE.i])

        println(log1)
        println(log2)
    end

    p[1][1][:z] = vars.b
    p[1][:title] = "buoyancy, t="*@sprintf("%.2f", clock.t)
    push!(p[2][1], KE.t[KE.i], KE.data[KE.i])
    #push!(p[3][1], B.t[B.i], B.data[B.i])
  
    stepforward!(prob, diags, nplot)
    updatevars!(prob)

end

mp4(anim, "SQG_Held1995.mp4", fps=14)