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
    # elseif sp.battery == 1 #Handle empty battery
    #     return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
    end

    push!(states, SAR_State(newrobot, s.target, s.battery-1))
    push!(probs, remaining_prob)

    return SparseCat(states, probs)

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

set_default_graphic_size(18cm,14cm)

function POMDPTools.ModelTools.render(m::SAR_POMDP, step)
    #set_default_graphic_size(14cm,14cm)
    nx, ny = m.size
    cells = []
    target_marginal = zeros(nx, ny)

    if haskey(step, :bp) && !ismissing(step[:bp])
        for sp in support(step[:bp])
            p = pdf(step[:bp], sp)
            if sp.target != [-1,-1] # TO-DO Fix this
                target_marginal[sp.target...] += p
            end
        end
    end
    #display(target_marginal)
    norm_top = normalize(target_marginal)
    #display(norm_top)
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x,y), m.size)
        t_op = norm_top[x,y]
        
        # TO-DO Fix This
        if t_op > 1.0
            if t_op < 1.001
                t_op = 0.999
            else
                @error("t_op > 1.001", t_op)
            end
        end
        opval = t_op
        if opval > 0.0 
           opval = clamp(t_op*2,0.05,1.0)
        end
        max_op = maximum(norm_top)
        min_op = minimum(norm_top)
        frac = (opval-min_op)/(max_op-min_op)
        clr = get(ColorSchemes.bamako, frac)
        
        target = compose(context(), rectangle(), fill(clr), stroke("gray"))
        #println("opval: ", t_op)
        compose!(cell, target)

        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.00000001mm), cells...)
    outline = compose(context(), linewidth(0.01mm), rectangle(), fill("white"), stroke("black"))

    if haskey(step, :sp)
        robot_ctx = cell_ctx(step[:sp].robot, m.size)
        robot = compose(robot_ctx, circle(0.5, 0.5, 0.5), fill("blue"))
        target_ctx = cell_ctx(step[:sp].target, m.size)
        target = compose(target_ctx, star(0.5,0.5,0.8,5,0.5), fill("orange"), stroke("black"))
    else
        robot = nothing
        target = nothing
    end 
    #img = read(joinpath(@__DIR__,"../..","drone.png"));
    #robot = compose(robot_ctx, bitmap("image/png",img, 0, 0, 1, 1))
    #person = read(joinpath(@__DIR__,"../..","missingperson.png"));
    #target = compose(target_ctx, bitmap("image/png",person, 0, 0, 1, 1))

    sz = min(w,h)
    
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), robot, target, grid, outline)
end

function normie(input, a)
    return (input-minimum(a))/(maximum(a)-minimum(a))
end

function rewardinds(m, pos::SVector{2, Int64})
    correct_ind = reverse(pos)
    xind = m.size[2]+1 - correct_ind[1]
    inds = [xind, correct_ind[2]]

    return pos
end


function POMDPTools.ModelTools.render(m::SAR_POMDP, step, plt_reward::Bool)
    nx, ny = m.size
    cells = []

    minr = minimum(m.reward)-1
    maxr = maximum(m.reward)

    if haskey(step, :hist)
        trajec = [(histstep[1].robot, histstep[2]) for histstep in step[:hist]]
        statehist = [s for (s,a) in trajec]
        actionhist = [a for (s,a) in trajec]
    end
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x,y), m.size)
        r = m.reward[rewardinds(m, SA[x,y])...]
        if iszero(r)
            target = compose(context(), rectangle(), fill("white"), stroke("gray"))
        else
            frac = (r-minr)/(maxr-minr)
            clr = get(ColorSchemes.turbo, frac)
            target = compose(context(), rectangle(), fill(clr), stroke("gray"), fillopacity(0.9))
        end

        if haskey(step, :hist)
            for (i, (xh, yh)) in enumerate(statehist)
                if x == xh && y == yh
                    if actionhist[i] == :left
                        spec = compose(context(), arrow(), stroke("black"), fill(nothing), linewidth(0.6mm), (context(), line([(0.5,0.5),(0.3,0.5)]), stroke("black")))
                        compose!(target, spec)
                    elseif actionhist[i] == :right
                        spec = compose(context(), arrow(), stroke("black"), fill(nothing), linewidth(0.6mm), (context(), line([(0.5,0.5),(0.7,0.5)]), stroke("black")))
                        compose!(target, spec)
                    elseif actionhist[i] == :up
                        spec = compose(context(), arrow(), stroke("black"), fill(nothing), linewidth(0.6mm), (context(), line([(0.5,0.5),(0.5,0.3)]), stroke("black")))
                        compose!(target, spec)
                    elseif actionhist[i] == :down
                        spec = compose(context(), arrow(), stroke("black"), fill(nothing), linewidth(0.6mm), (context(), line([(0.5,0.5),(0.5,0.7)]), stroke("black")))
                        compose!(target, spec)
                    end
                end
            end

            # if SA[x,y] in statehist
            #     if trajec == :left
            #         spec = compose(context(), line([(0.5,0.5),(0.1,0.5)]), stroke("black"))
            #         compose!(target, spec)
            #     elseif step[:a] == :right
            #         spec = compose(context(), line([(0.5,0.5),(0.9,0.5)]), stroke("black"))
            #         compose!(target, spec)
            #     elseif step[:a] == :up
            #         spec = compose(context(), line([(0.5,0.5),(0.5,0.9)]), stroke("black"))
            #         compose!(target, spec)
            #     elseif step[:a] == :down
            #         spec = compose(context(), line([(0.5,0.5),(0.5,0.1)]), stroke("black"))
            #         compose!(target, spec)
            #     end
            # #   spec = compose(context(), circle(0.5, 0.5, 0.1), fill("blue"), stroke("black"))
            # #   compose!(target, spec)
            # end
        end

        compose!(cell, target)
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(1mm), cells...)
    outline = compose(context(), linewidth(0.05mm), rectangle(), fill("black"), stroke("black"))

    if haskey(step, :sp)
        robot_ctx = cell_ctx(step[:sp].robot, m.size)
        robot = compose(robot_ctx, circle(0.5, 0.5, 0.3), fill("blue"))
        target_ctx = cell_ctx(step[:sp].target, m.size)
        target = compose(target_ctx, star(0.5,0.5,0.5,5,0.5), fill("orange"), stroke("black"))
    else
        robot = nothing
        target = nothing
    end
    sz = min(w,h)
    #return compose(context((w-sz)/2, (h-sz)/2, sz, (ny/nx)*sz), robot, target, grid, outline)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), robot, target, grid, outline)
end

#POMDPs.isterminal(m::SAR_POMDP, s::SAR_State) = s.robot == SA[-1,-1]
function dist(curr, start)
    sum(abs.(curr-start))
end

function POMDPs.isterminal(m::SAR_POMDP, s::SAR_State)
    return s.robot == SA[-1,-1] || s.target == SA[-1,-1] || s.battery == -1
end