struct SAR_State
    robot::SVector{2, Int}
    target::SVector{2, Int}
    visited::Tuple
    battery::Int
end

mutable struct SAR_POMDP <: POMDP{SAR_State, Symbol, BitArray{1}}
    size::SVector{2, Int}
    obstacles::Set{SVector{}}
    robot_init::SVector{2, Int}
    tprob::Float64
    targetloc::SVector{2, Int}
    r_locs::Vector{SVector{2,Int64}}
    r_vals::Vector{Float64}
    r_find::Float64
    reward::Matrix{Float64}
    maxbatt::Int
end

function SAR_POMDP(sinit::SAR_State; 
                        roi_points=Dict{SVector{2,Int64},Float64}(), 
                        m_size=(10,10), 
                        rewarddist=Array{Float64}(undef, 0, 0), 
                        maxbatt=100)
    obstacles = Set{SVector{2, Int}}()
    robot_init = sinit.robot
    tprob = 0.7
    targetloc = sinit.target
    if isempty(roi_points)
        r_locs = SVector{2,Int64}[]
        r_vals = Float64[]
        for i in 1:size(rewarddist)[1]
            for j in 1:size(rewarddist)[2]
                valu = rewarddist[i,j]
                if valu != 0
                    push!(r_vals,valu)
                    push!(r_locs,SVector{2}([i,j]))
                end
            end
        end
    else
        r_locs = keys(roi_points)
        r_vals = values(roi_points)
    end
    maxbatt = maxbatt
  
    return SAR_POMDP(m_size, obstacles, robot_init, tprob, targetloc, SVector{2,Int64}[r_locs...], Float64[r_vals...], 1000.0, rewarddist, maxbatt)
end

function SAR_POMDP(;
    init_ro = [1,1],
    target = [4,4],
    maxbatt = 20,
    rew_locs = SVector{3}([[1,4],[4,1],[3,3]]),
    rew_vals = SVector{3}([2.0,2.0,1.0]),
    r_find = 1000.0,
    size=(5,5))


    rewarddist = zeros(size)
    for (i,l) in enumerate(rew_locs)
        rewarddist[l...] = rew_vals[i]
    end

    sinit = SAR_State(init_ro,target,Tuple(Int.(zeros(length(rew_locs)))),maxbatt)
    obstacles = Set{SVector{2, Int}}()
    robot_init = sinit.robot
    tprob = 0.7
    targetloc = sinit.target
    maxbatt = maxbatt

    return SAR_POMDP(size, obstacles, robot_init, tprob, targetloc, rew_locs, rew_vals, r_find, rewarddist, maxbatt)
end