function POMDPs.states(m::SAR_POMDP) 
    nonterm = vec(collect(SAR_State(SVector(c[1],c[2]), SVector(c[3],c[4]), d) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2]) for d in 1:m.maxbatt))
    return push!(nonterm, SAR_State([-1,-1],[-1,-1],-1))
end

function POMDPs.stateindex(m::SAR_POMDP, s)
    if s.robot == SA[-1,-1]
        return m.size[1]^2 * m.size[2]^2 * m.maxbatt + 1
    else 
        return LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], 1:m.maxbatt))[s.robot..., s.target..., s.battery]
    end
end

function POMDPs.initialstate(m::SAR_POMDP)
    return m.initial_state_dist
end

"""
    actions
"""

POMDPs.actions(m::SAR_POMDP) = (:left, :right, :up, :down) #, :stay)

POMDPs.discount(m::SAR_POMDP) = 0.95


POMDPs.actionindex(m::SAR_POMDP, a) = actionind[a]


function bounce(m::SAR_POMDP, pos, offset)
    new = clamp.(pos + offset, SVector(1,1), m.size)
end

function POMDPs.transition(m::SAR_POMDP, s, a)
    states = SAR_State[]
    probs = Float64[]
    remaining_prob = 1.0

    required_batt = dist(s.robot, m.robot_init)
    newrobot = bounce(m, s.robot, actiondir[a])

    if m.terminate_on_find && isequal(s.robot, s.target)
        return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
    elseif m.auto_home && (s.battery - required_batt <= 1)
        return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
    elseif !m.terminate_on_find && !m.auto_home
        if s != m.robot_init && newrobot == m.robot_init #THIS IS NOT STOCHASTIC SAFE...
            return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
        end
    elseif newrobot == m.robot_init && s.battery < m.maxbatt
        return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
    # elseif sp.battery == 1 #Handle empty battery
    #     return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
    end

    push!(states, SAR_State(newrobot, s.target, s.battery-1))
    push!(probs, remaining_prob)

    return SparseCat(states, probs)

end

"""
    observations(m::SAR_POMDP)

Retrieve observations in TargetSearch observation space

The the observations are ordered as follows:
    1: The target is not observed
    2: The target is in same grid cell as robot
    3: The target is to the left of the robot
    4: The target is to the right of the robot
    5: The target is below the robot
    6: The target is above the robot
"""
POMDPs.observations(m::SAR_POMDP) = OBSERVATIONS
POMDPs.obsindex(m::SAR_POMDP, o::BitVector) = obsind[o]

function POMDPs.observation(m::SAR_POMDP, a::Symbol, sp::SAR_State)
    #obs = [BitVector([0,0,0,0,0]), BitVector([1,0,0,0,0]), BitVector([0,1,0,0,0]), BitVector([0,0,1,0,0]), BitVector([0,0,0,1,0]), BitVector([0,0,0,0,1])]

    if norm(sp.robot-sp.target) == 1.0 # target and robot within one grid cell of each other 
        targetloc = targetdir(sp)

        if targetloc == :left
            probs = [0.0, 0.0, 0.50, 0.0, 0.25, 0.25]
        elseif targetloc == :right
            probs = [0.0, 0.0, 0.0, 0.50, 0.25, 0.25]
        elseif targetloc == :up
            probs = [0.0, 0.0, 0.25, 0.25, 0.0, 0.50]
        elseif targetloc == :down
            probs = [0.0, 0.0, 0.25, 0.25, 0.50, 0.0]
        end

        return SparseCat(OBSERVATIONS, probs)
    end

    if sp.robot == sp.target # target and robot in same grid cell
        probs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        return SparseCat(OBSERVATIONS, probs)
    end

    probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return SparseCat(OBSERVATIONS, probs)

end

POMDPs.reward(m::SAR_POMDP, s::SAR_State, a::Symbol, sp::SAR_State) = reward(m, s, a)

function POMDPs.reward(m::SAR_POMDP, s::SAR_State, a::Symbol)
    reward_running = 0.0 #-1.0
    reward_target = 0.0
    
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

    return reward_running + reward_target + m.reward[s.robot...]
end


#POMDPs.isterminal(m::SAR_POMDP, s::SAR_State) = s.robot == SA[-1,-1]
function dist(curr, start)
    sum(abs.(curr-start))
end

function POMDPs.isterminal(m::SAR_POMDP, s::SAR_State)
    return s.robot == SA[-1,-1] || s.target == SA[-1,-1] || s.battery == -1
end