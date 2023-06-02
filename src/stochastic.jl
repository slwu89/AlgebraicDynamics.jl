module StochasticDynam
using ..UWDDynam

using Catlab
using Catlab.WiringDiagrams, Catlab.Programs
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.FinSets
using Catlab.Theories

using Catlab.WiringDiagrams.UndirectedWiringDiagrams: AbstractUWD
import Catlab.WiringDiagrams: oapply, ports

using OrdinaryDiffEq, DelayDiffEq
import OrdinaryDiffEq: ODEProblem, DiscreteProblem
import DelayDiffEq: DDEProblem
using Plots

using JumpProcesses

struct PoissonUndirectedSystem{T} <: UWDDynam.AbstractUndirectedSystem{T}
    nstates::Int
    intensity::Function
    portmap
end

struct PoissonUndirectedInterface{T} <: UWDDynam.AbstractUndirectedInterface{T}
    ports::Vector
    affects::Vector{T}
end

struct PoissonResourceSharer{T,I,S} <: UWDDynam.AbstractResourceSharer{T}
    interface::I
    system::S
end

function PoissonResourceSharer(nstates, intensity, affects)
    PoissonResourceSharer{Int,PoissonUndirectedInterface{Int},PoissonUndirectedSystem{Int}}(
        PoissonUndirectedInterface{Int}(1:nstates, affects),
        PoissonUndirectedSystem{Int}(nstates, intensity, 1:nstates)
    )
end

system(r::PoissonResourceSharer) = r.system 
interface(r::PoissonResourceSharer) = r.interface

UWDDynam.ports(r::PoissonResourceSharer) = ports(interface(r))
UWDDynam.nports(r::PoissonResourceSharer) = nports(interface(r))
UWDDynam.nstates(r::PoissonResourceSharer) = nstates(system(r)) 
UWDDynam.dynamics(r::PoissonResourceSharer) = dynamics(system(r))
UWDDynam.portmap(r::PoissonResourceSharer) = portmap(system(r)) 
UWDDynam.portfunction(r::PoissonResourceSharer) = portfunction(system(r))
UWDDynam.exposed_states(r::PoissonResourceSharer, u::AbstractVector) = exposed_states(system(r), u)

birth_rs = PoissonResourceSharer(1, (u,p,t) -> p.λ, [1])
death_rs = PoissonResourceSharer(1, (u,p,t) -> u[1]>0 ? p.μ : 0.0, [-1])

# the BD setup
bd_uwd = @relation (x,) begin
    birth(x)    
    death(x)
end

xs = [birth_rs, death_rs]

p = (λ=1.95,μ=2)

S′ = UWDDynam.induced_states(bd_uwd, xs)
S = coproduct((FinSet∘nstates).(xs))
states(b::Int) = legs(S)[b].func
state_map = legs(S′)[1]

# 1. for each box, we need to make the rate function and affects! function
# 2. generate the Jump and store it

# parts(bd_uwd, :Box)

# system(xs[1]).intensity
# interface(xs[1]).affects

# supertype(ConstantRateJump)

# states(2)
# state_map

# preimage(state_map, 1)
# state_map(states(1)) # which global states come from states in box 1?

jumps = Vector{JumpProcesses.AbstractJump}(undef, length(xs))

for b in parts(bd_uwd, :Box)
    states_b = state_map(states(b))
    rate(u,p,t) = begin
        system(xs[b]).intensity(u[states_b], p, t)
    end
    affect!(integrator) = begin
        for i in eachindex(states_b)
            integrator.u[states_b[i]] = +(integrator.u[states_b[i]], interface(xs[b]).affects[i])
        end
    end
    jumps[b] = ConstantRateJump(rate, affect!)
end

dprob = DiscreteProblem([10], (0.0,1000.0), p)
jprob = JumpProblem(dprob, Direct(), jumps...)

sol = solve(jprob, SSAStepper())

plot(sol)

end