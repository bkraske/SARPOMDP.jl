struct SAR_State
    robot::SVector{2, Int}
    target::SVector{2, Int}
    battery::Int
end

mutable struct SAR_POMDP <: POMDP{SAR_State, Symbol, BitArray{1}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    robot_init::SVector{2, Int}
    tprob::Float64
    targetloc::SVector{2, Int}
    r_locs::Vector{SVector{2,Int}}
    r_vals::Vector{Float64}
    r_find::Float64
    reward::Matrix{Float64}
    maxbatt::Int
    auto_home::Bool
    terminate_on_find::Bool
    initial_state_dist::SparseCat{Vector{SAR_State},Vector{Float64}}
end

function SAR_POMDP(sinit::SAR_State; 
                        map_size=(10,10), 
                        rewarddist=Array{Float64}(undef, 0, 0), 
                        maxbatt=100,auto_home=true,terminate_on_find=true,
                        initial_state_dist=POMDPTools.Uniform(SAR_State(sinit.robot, SVector(x, y), maxbatt) for x in 1:map_size[1], y in 1:map_size[2]))

    @assert size(rewarddist) == map_size || size(rewarddist) == (0,0)

    obstacles = Set{SVector{2, Int}}()
    robot_init = sinit.robot
    tprob = 0.7
    targetloc = sinit.target

    r_locs = SVector{2,Int}[]
    r_vals = Float64[]
    for i in 1:size(rewarddist,1)
        for j in 1:size(rewarddist,2)
            r = rewarddist[i,j]
            if r != 0
                push!(r_locs,SVector(i,j))
                push!(r_vals,r)
            end
        end
    end

    if !isa(initial_state_dist,SparseCat{Vector{SAR_State},Vector{Float64}})
        supp = [support(initial_state_dist)...]
        cat_initial_state_dist = SparseCat(supp,[pdf(initial_state_dist,s) for s in supp])
    else
        cat_initial_state_dist = initial_state_dist
    end

    return SAR_POMDP(SVector(map_size), obstacles, robot_init, tprob, targetloc, r_locs, r_vals, 1000.0, rewarddist, Int(maxbatt), auto_home, terminate_on_find,cat_initial_state_dist)
end

function SAR_POMDP(;
    init_ro = [1,1],
    target = [4,4],
    maxbatt = 20,
    rew_locs = SVector{2,Int}.([[1,4],[4,1],[3,3]]),
    rew_vals = [2.0,2.0,1.0],
    r_find = 1000.0,
    map_size=(5,5),auto_home=true,terminate_on_find=true,
    initial_state_dist=POMDPTools.Uniform(SAR_State(init_ro, SVector(x, y), maxbatt) for x in 1:map_size[1], y in 1:map_size[2]))
    @assert size(rew_locs) == size(rew_vals)

    rewarddist = zeros(map_size)
    for (i,l) in enumerate(rew_locs)
        rewarddist[l...] = rew_vals[i]
    end

    sinit = SAR_State(init_ro,target,maxbatt)
    obstacles = Set{SVector{2, Int}}()
    robot_init = sinit.robot
    tprob = 0.7
    targetloc = sinit.target
    maxbatt = maxbatt

    if !isa(initial_state_dist,SparseCat{Vector{SAR_State},Vector{Float64}})
        supp = [support(initial_state_dist)...]
        cat_initial_state_dist = SparseCat(supp,[pdf(initial_state_dist,s) for s in supp])
    else
        cat_initial_state_dist = initial_state_dist
    end

    return SAR_POMDP(SVector(map_size), obstacles, robot_init, tprob, targetloc, rew_locs, rew_vals, r_find, rewarddist, Int(maxbatt), auto_home, terminate_on_find, cat_initial_state_dist)
end