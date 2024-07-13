module SARPOMDP

using POMDPs
using POMDPTools
using Distributions
import POMDPTools:Uniform as Uni
using Random
using StaticArrays
using Parameters
using LinearAlgebra
using Compose
using Compose:rectangle,stroke,circle
using BasicPOMCP
using DiscreteValueIteration
using Reel, Cairo, Fontconfig
using D3Trees
using ParticleFilters
using ColorSchemes
using StatsBase: sample, Weights
using JLD2



include("common.jl")

include(joinpath(@__DIR__,"batteryPOMDP","core.jl"))
export SAR_POMDP, SAR_State

include(joinpath(@__DIR__,"batteryPOMDP","functions.jl"))
include(joinpath(@__DIR__,"batteryPOMDP","simulate.jl"))
export simulateSARPOMDP, 
       SARPOMDPSimulator

include("visualize.jl")
export rendhist

end


