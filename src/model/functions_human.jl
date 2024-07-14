function POMDPs.states(m::SAR_POMDP_human)
    nonterm = vec(collect(SAR_State_human(SAR_State(SVector(c[1],c[2]), 
            SVector(c[3],c[4]), d), e) for c in Iterators.product(1:m.underlying_pomdp.size[1], 1:m.underlying_pomdp.size[2], 1:m.underlying_pomdp.size[1], 1:m.underlying_pomdp.size[2]) 
            for d in 1:m.underlying_pomdp.maxbatt for e in BitArray([0,1])))
    return push!(nonterm, SAR_State_human(SAR_State([-1,-1],[-1,-1],-1),false))
end

function POMDPs.stateindex(m::SAR_POMDP_human, s)
    if s.underlying_state.robot == SA[-1,-1]
        return m.underlying_pomdp.size[1]^2 * m.underlying_pomdp.size[2]^2 * m.underlying_pomdp.maxbatt * 2 + 1
    else 
        return LinearIndices((1:m.underlying_pomdp.size[1], 
                              1:m.underlying_pomdp.size[2], 
                              1:m.underlying_pomdp.size[1], 
                              1:m.underlying_pomdp.size[2], 
                              1:m.underlying_pomdp.maxbatt, 
                              1:2))[s.underlying_state.robot..., s.underlying_state.target..., s.underlying_state.battery, s.onpath+1]
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
    horizon_idx = m.underlying_pomdp.maxbatt-s.underlying_state.battery + 1

    required_batt = dist(s.underlying_state.robot, m.underlying_pomdp.robot_init)
    newrobot = bounce(m, s.underlying_state.robot, actiondir[a])

    if m.underlying_pomdp.terminate_on_find && isequal(s.underlying_state.robot, s.underlying_state.target)
        return Deterministic(SAR_State_human(SAR_State([-1,-1], [-1,-1], -1), false))
    elseif m.underlying_pomdp.auto_home && (s.underlying_state.battery - required_batt <= 1)
        return Deterministic(SAR_State_human(SAR_State([-1,-1], [-1,-1], -1), false))
    elseif !m.underlying_pomdp.terminate_on_find && !m.underlying_pomdp.auto_home
        if s.underlying_state.robot != m.underlying_pomdp.robot_init && newrobot == m.underlying_pomdp.robot_init #THIS IS NOT STOCHASTIC SAFE...
            return Deterministic(SAR_State_human(SAR_State([-1,-1], [-1,-1], -1), false))
        end
    # elseif sp.battery == 1 #Handle empty battery
    #     return Deterministic(SAR_State_human([-1,-1], [-1,-1], -1))
    end

    push!(states, SAR_State_human(SAR_State(newrobot, s.underlying_state.target, s.underlying_state.battery-1), true),
                  SAR_State_human(SAR_State(newrobot, s.underlying_state.target, s.underlying_state.battery-1), false))

    #@info "Horizon index: $horizon_idx, observation_list: $(m.observation_list), action_list: $(m.action_list)"
    # probability that the human is on the path
    if horizon_idx <= length(m.action_list)
        if s.onpath && a == m.action_list[horizon_idx]
                #compute prob of observing the human observation at this timestep
                obsdist = observation(m, a, first(states))
                push!(probs, obsdist.probs[obsindex(m, m.observation_list[horizon_idx])])
                push!(probs, 1.0-probs[end])
        else
            push!(probs, 0.0, 1.0)
        end
    else
        push!(probs, 0.0, 1.0)
    end
    #push!(states, SAR_State_human(newrobot, s.target, s.battery-1, true))
    #push!(probs, remaining_prob)

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
    horizon_idx = m.underlying_pomdp.maxbatt-s.underlying_state.battery + 1
    rtot = reward(m.underlying_pomdp, s.underlying_state, a)
    
    if horizon_idx <= length(m.action_list)
        return s.onpath && a == m.action_list[horizon_idx] ? rtot : rtot + -10000
    else
        return rtot
    end
end

#POMDPs.isterminal(m::SAR_POMDP_human, s::SAR_State_human) = s.robot == SA[-1,-1]

POMDPs.isterminal(m::SAR_POMDP_human, s::SAR_State_human) = isterminal(m.underlying_pomdp, s.underlying_state)
