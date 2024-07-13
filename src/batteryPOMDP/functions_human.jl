function POMDPs.states(m::SAR_POMDP_human)
    nonterm = vec(collect(SAR_State_human(SVector(c[1],c[2]), 
            SVector(c[3],c[4]), d, e) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2]) 
            for d in 1:m.maxbatt for e in BitArray([0,1])))
    return push!(nonterm, SAR_State_human([-1,-1],[-1,-1],-1,false))
end

function POMDPs.stateindex(m::SAR_POMDP_human, s)
    if s.robot == SA[-1,-1]
        return m.size[1]^2 * m.size[2]^2 * m.maxbatt * 2 + 1
    else 
        return LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], 1:m.maxbatt, 1:2))[s.robot..., s.target..., s.battery, s.onpath+1]
    end
end

function POMDPs.initialstate(m::SAR_POMDP_human)
    return m.initial_state_dist
end

"""
    actions
"""

POMDPs.actions(m::SAR_POMDP_human) = actions(m.underlying_pomdp)

POMDPs.discount(m::SAR_POMDP_human) = discount(m.underlying_pomdp)


POMDPs.actionindex(m::SAR_POMDP_human, a) = actionindex(m.underlying_pomdp, a)


bounce(m::SAR_POMDP_human, pos, offset) = bounce(m.underlying_pomdp, pos, offset)

function POMDPs.transition(m::SAR_POMDP_human, s, a)
    states = SAR_State_human[]
    probs = Float64[]
    remaining_prob = 1.0
    horizon_idx = m.maxbatt-s.battery

    required_batt = dist(s.robot, m.robot_init)
    newrobot = bounce(m, s.robot, actiondir[a])

    if m.terminate_on_find && isequal(s.robot, s.target)
        return Deterministic(SAR_State_human([-1,-1], [-1,-1], -1, false))
    elseif m.auto_home && (s.battery - required_batt <= 1)
        return Deterministic(SAR_State_human([-1,-1], [-1,-1], -1, false))
    elseif !m.terminate_on_find && !m.auto_home
        if s != m.robot_init && newrobot == m.robot_init #THIS IS NOT STOCHASTIC SAFE...
            return Deterministic(SAR_State_human([-1,-1], [-1,-1], -1, false))
        end
    # elseif sp.battery == 1 #Handle empty battery
    #     return Deterministic(SAR_State_human([-1,-1], [-1,-1], -1))
    end

    push!(states, SAR_State_human(newrobot, s.target, s.battery-1, true))
    push!(states, SAR_State_human(newrobot, s.target, s.battery-1, false))

    # probability that the human is on the path
    if s.onpath && a == m.action_list[horizon_idx]
            #compute prob of observing the human observation at this timestep
            obsdist = observation(m, a, states[1])
            push!(probs, obsdist.probs[obsind[m.observation_list[horizon_idx]]], 1.0-probs[end])
    else
        push!(probs, 0.0, 1.0)
    end
    push!(states, SAR_State_human(newrobot, s.target, s.battery-1, true))
    push!(probs, remaining_prob)

    return SparseCat(states, probs)

end

"""
    observations(m::SAR_POMDP_human)

Retrieve observations in TargetSearch observation space

The the observations are ordered as follows:
    1: The target is not observed
    2: The target is in same grid cell as robot
    3: The target is to the left of the robot
    4: The target is to the right of the robot
    5: The target is below the robot
    6: The target is above the robot
"""
POMDPs.observations(m::SAR_POMDP_human) = OBSERVATIONS
POMDPs.obsindex(m::SAR_POMDP_human, o::BitVector) = obsind[o]

POMDPs.observation(m::SAR_POMDP_human, a::Symbol, sp::SAR_State_human) = observation(m.underlying_pomdp, a, sp.underlying_state)

POMDPs.reward(m::SAR_POMDP_human, s::SAR_State_human, a::Symbol, sp::SAR_State_human) = reward(m, s, a)

function POMDPs.reward(m::SAR_POMDP_human, s::SAR_State_human, a::Symbol)
    reward_running = 0.0 #-1.0
    reward_target = 0.0
    
    horizon_idx = m.maxbatt-s.battery

    required_batt = dist(s.robot, m.robot_init)
    if !m.auto_home && (s.battery - required_batt <= 1) && s != m.robot_init
        return -1e10
    end

    if isterminal(m, s) # IS THIS NECCESSARY?
        return 0.0
    end

    if isequal(s.robot, s.target) # if target is found
        reward_running = 0.0
        reward_target = m.r_find
        return reward_running + reward_target +  m.reward[s.robot...]
    end

    off_path_reward = typemin(Float64)

    if s.onpath && a == m.action_list[horizon_idx]
        return reward_running + reward_target + m.reward[s.robot...]
    else
        return reward_running + reward_target + off_path_reward + m.reward[s.robot...]
    end
end

#POMDPs.isterminal(m::SAR_POMDP_human, s::SAR_State_human) = s.robot == SA[-1,-1]

POMDPs.isterminal(m::SAR_POMDP_human, s::SAR_State_human) = isterminal(m.underlying_pomdp, s.underlying_state)
