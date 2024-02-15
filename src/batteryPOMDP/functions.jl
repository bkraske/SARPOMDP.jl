function POMDPs.states(m::SAR_POMDP) 
    nonterm = vec(collect(SAR_State(SVector(c[1],c[2]), SVector(c[3],c[4]), Tuple(c[5:end]), d) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], [0:1 for _ in 1:length(m.r_locs)]...) for d in 1:m.maxbatt))
    return push!(nonterm, SAR_State(SVector{2}([-1,-1]),SVector{2}([-1,-1]),Tuple([0 for _ in eachindex(m.r_locs)]),-1))
end

function POMDPs.stateindex(m::SAR_POMDP, s)
    r_len = length(m.r_locs)
    if s.robot == SA[-1,-1]
        return m.size[1]^2 * m.size[2]^2 * m.maxbatt * 2^r_len + 1
    else 
        return LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], [0:1 for _ in 1:r_len]..., 1:m.maxbatt))[s.robot..., s.target..., 1 .+ s.visited...,s.battery]
    end
end

function POMDPs.initialstate(m::SAR_POMDP)
    no_visit = Tuple(Int.(zeros(length(m.r_locs))))
    return POMDPTools.Uniform(SAR_State(m.robot_init, SVector(x, y), no_visit, m.maxbatt) for x in 1:m.size[1], y in 1:m.size[2])
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

    if isequal(s.robot, s.target) || (s.battery - required_batt <= 1)
        return Deterministic(SAR_State([-1,-1], [-1,-1], Tuple([0 for _ in eachindex(m.r_locs)]), -1))
    # elseif sp.battery == 1 #Handle empty battery
    #     return Deterministic(SAR_State([-1,-1], [-1,-1], -1))
    end

    newrobot = bounce(m, s.robot, actiondir[a])
    visit = [s.visited...]
    for i in eachindex(m.r_locs)
        if s == m.r_locs[i]
            visit[i] = 1
        end
    end
    new_visit = Tuple(visit)

    push!(states, SAR_State(newrobot, s.target, new_visit, s.battery-1))
    push!(probs, remaining_prob)

    return SparseCat(states, probs)

end

POMDPs.reward(m::SAR_POMDP, s::SAR_State, a::Symbol, sp::SAR_State) = reward(m, s, a)

function POMDPs.reward(m::SAR_POMDP, s::SAR_State, a::Symbol)
    reward_running = 0.0 #-1.0
    reward_target = 0.0

    if isterminal(m, s) # IS THIS NECCESSARY?
        return 0.0
    end

    if isequal(s.robot, s.target) # if target is found
        reward_running = 0.0
        reward_target = m.r_find
        return reward_running + reward_target
    end

    loc_val = m.reward[s.robot...]
    if loc_val != 0.0
        ind = findall(x->x==s,m.r_locs)
        if !isempty(ind) && s.visited[ind[1]] == 1
            loc_val = 0.0
        end
    end

    return reward_running + reward_target + loc_val
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

    return inds
end


function POMDPTools.ModelTools.render(m::SAR_POMDP, step, plt_reward::Bool)
    nx, ny = m.size
    cells = []
    
    minr = minimum(m.reward)-1
    maxr = maximum(m.reward)
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x,y), m.size)
        r = m.reward[rewardinds(m, SA[x,1+ny-y])...]
        if iszero(r)
            target = compose(context(), rectangle(), fill("black"), stroke("gray"))
        else
            frac = (r-minr)/(maxr-minr)
            clr = get(ColorSchemes.turbo, frac)
            target = compose(context(), rectangle(), fill(clr), stroke("gray"), fillopacity(0.5))
        end

        compose!(cell, target)
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.00000001mm), cells...)
    outline = compose(context(), linewidth(0.01mm), rectangle(), fill("white"), stroke("black"))

    if haskey(step, :sp)
        robot_ctx = cell_ctx(step[:sp].robot, m.size)
        robot = compose(robot_ctx, circle(0.5, 0.5, 0.5), fill("blue"))
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