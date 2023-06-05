using AlgebraicDynamics.UWDDynam

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
using Catlab.Graphics, Graphviz_jll

# --------------------------------------------------------------------------------
# resource sharers but with "affects" in the interface for stochastic systems

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

UWDDynam.system(r::PoissonResourceSharer) = r.system 
UWDDynam.interface(r::PoissonResourceSharer) = r.interface

UWDDynam.ports(r::PoissonResourceSharer) = ports(interface(r))
UWDDynam.nports(r::PoissonResourceSharer) = nports(interface(r))
UWDDynam.nstates(r::PoissonResourceSharer) = nstates(system(r)) 
UWDDynam.dynamics(r::PoissonResourceSharer) = dynamics(system(r))
UWDDynam.portmap(r::PoissonResourceSharer) = portmap(system(r)) 
UWDDynam.portfunction(r::PoissonResourceSharer) = portfunction(system(r))
UWDDynam.exposed_states(r::PoissonResourceSharer, u::AbstractVector) = exposed_states(system(r), u)

# --------------------------------------------------------------------------------
# the birth-death process model

bd_uwd = @relation (x,) begin
    birth(x)    
    death(x)
end

birth_rs = PoissonResourceSharer(1, (u,p,t) -> p.λ, [1])
death_rs = PoissonResourceSharer(1, (u,p,t) -> u[1]>0 ? p.μ : 0.0, [-1])

xs = [birth_rs, death_rs]

p = (λ=1.95,μ=2)

S′ = UWDDynam.induced_states(bd_uwd, xs)
S = coproduct((FinSet∘nstates).(xs))
states(b::Int) = legs(S)[b].func
state_map = legs(S′)[1]

# 1. for each box get the map from local states to global states (i.e. state_map(states(b)))
# 2. make the rate function and affects! function
# 3. generate the Jump and store it

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
to_graphviz(bd_uwd, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))

# --------------------------------------------------------------------------------
# the ubiquitous SIR model

sir_uwd = @relation (S,I,R) begin
    birth(S,I)  
    death(I,R)
end

to_graphviz(sir_uwd, box_labels = :name, junction_labels = :variable, edge_attrs=Dict(:len => ".75"))

infection_rs = PoissonResourceSharer(2, (u,p,t) -> all(u .≥ 1) ? p.β*u[1]*u[2] : 0.0, [-1,1])
recovery_rs = PoissonResourceSharer(2, (u,p,t) -> u[1] > 0 ? p.γ*u[1] : 0.0, [-1,1])

xs = [infection_rs, recovery_rs]

# parameters and initial conditions from sir-julia repo
tmax = 50.0
u0 = [990,10,0]
p = (β=0.5/sum(u0),γ=0.25)

# make the jump system
S′ = UWDDynam.induced_states(sir_uwd, xs)
S = coproduct((FinSet∘nstates).(xs))
states(b::Int) = legs(S)[b].func # box (b) -> states it has access to in coproduct (before identifying common states)
state_map = legs(S′)[1] # map from coproduct states to reduced states

# 1. for each box get the map from local states to global states (i.e. state_map(states(b)))
# 2. make the rate function and affects! function
# 3. generate the Jump and store it

jumps = Vector{JumpProcesses.AbstractJump}(undef, length(xs))

for b in parts(sir_uwd, :Box)
    states_b = state_map(states(b))
    rate(u,p,t) = begin
        system(xs[b]).intensity(u[states_b], p, t)
    end
    affect!(integrator) = begin
        for i in eachindex(states_b)
            integrator.u[states_b[i]] = integrator.u[states_b[i]] + interface(xs[b]).affects[i]
        end
    end
    jumps[b] = ConstantRateJump(rate, affect!)
end

dprob = DiscreteProblem(u0, (0.0,tmax), p)
jprob = JumpProblem(dprob, Direct(), jumps...)

sol = solve(jprob, SSAStepper())
plot(sol,label=["S" "I" "R"])

