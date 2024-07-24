mutable struct GridWorldEnv
    m::SAR_POMDP
    size::SVector{2, Int}
    rewards::Matrix{Float64}
    robotInit::SVector{2, Int}
    targetInit::SVector{2, Int}
end

function GridWorldEnv(m, rewards::Matrix{Float64}, targetInit; size=(10,10), robotInit=(1,1))
    return GridWorldEnv(m, SA[size[1], size[2]], rewards, robotInit, targetInit)
end


"""
    render(env::GridWorldEnv)
    render(env::GridWorldEnv, color=s->5.0, policy=s->SA[1,0])

Render a GridWorldEnv to a Compose.jl object that can be displayed in a Jupyter notebook or ElectronDisplay window.

# Keyword Arguments
- `color::Function`: A function that determines the color of each cell. Input is a state, output is either a Float64 between -10 and 10 that will produce a color ranging from red to green, or any color from Colors.jl.
- `policy::Function`: A function that allows showing an arrow in each cell to indicate the policy. Input is a state; output is an action.
"""
observations(env::GridWorldEnv) = [SA[x, y] for x in 1:env.size[1], y in 1:env.size[2]]

function renderMDP(env::GridWorldEnv; color::Function=s->get(env.rewards, s, -0.1), policy::Union{Function,Nothing}=nothing)
    nx, ny = env.size
    m = env.m
    cells = []
    for s in observations(env)
        r = env.rewards[rewardinds(m, SA[s...])...]
        clr = get(ColorSchemes.redgreensplit, (r+10.0)/20.0)
        cell = context((s[1]-1)/nx, (ny-s[2])/ny, 1/nx, 1/ny)
        if policy !== nothing
            a = policy(BasicState(s, env.targetInit))
            txt = compose(context(), Compose.text(0.5, 0.5, aarrow[a], hcenter, vcenter), stroke("black"))
            compose!(cell, txt)
        end
        clr = tocolor(r)
        compose!(cell, rectangle(), fill(clr), stroke("gray"))
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), cells...)
    outline = compose(context(), linewidth(1mm), rectangle(), stroke("gray"))

    s = env.robotInit
    agent_ctx = context((s[1]-1)/nx, (ny-s[2])/ny, 1/nx, 1/ny)
    agent = compose(agent_ctx, circle(0.5, 0.5, 0.4), fill("orange"))

    sz = min(w,h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), agent, grid, outline)
end


tocolor(x) = x
function tocolor(r::Float64)
    minr = -20.0
    maxr = 100.0
    frac = (r-minr)/(maxr-minr)
    return get(ColorSchemes.redgreensplit, frac)
end

const aarrow = Dict(:up=>'↑', :left=>'←', :down=>'↓', :right=>'→', :stay=>'⊙')


function renderVIPolicy(policy, mdp, s, rewarddist)
    gw = GridWorldEnv(mdp, rewarddist, s.target, size=mdp.size, robotInit=s.robot)
    vi_policy = s -> DiscreteValueIteration.action(policy, s)
    display(SARPOMDP.renderMDP(gw, policy = vi_policy))
end

function rendhist(hist, m; delay=0.1)
    for h ∈ hist
        remove_rewards(m, h.s.robot)
        display(render(m, h, true))
        sleep(delay)
    end
end 

function rendhist(hist, m, rewarddist; delay=0.1)
    m.reward = rewarddist
    for h ∈ hist
        remove_rewards(m, h.sp.robot)
        display(render(m, h, true))
        sleep(delay)
    end
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

function POMDPTools.ModelTools.render(m::SAR_POMDP_human, step)
    #set_default_graphic_size(14cm,14cm)
    m = m.underlying_pomdp

    nx, ny = m.size
    cells = []
    target_marginal = zeros(nx, ny)

    if haskey(step, :bp) && !ismissing(step[:bp])
        for sp in support(step[:bp])
            p = pdf(step[:bp], sp)
            if sp.underlying_state.target != [-1,-1] # TO-DO Fix this
                target_marginal[sp.underlying_state.target...] += p
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
        robot_ctx = cell_ctx(step[:sp].underlying_state.robot, m.size)
        robot = compose(robot_ctx, circle(0.5, 0.5, 0.5), fill("blue"))
        target_ctx = cell_ctx(step[:sp].underlying_state.target, m.size)
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

function POMDPTools.ModelTools.render(m::SAR_POMDP_human, step, plt_reward::Bool)
    nx, ny = m.underlying_pomdp.size
    cells = []

    minr = minimum(m.underlying_pomdp.reward)-1
    maxr = maximum(m.underlying_pomdp.reward)

    if haskey(step, :hist)
        trajec = [(histstep[1].underlying_state.robot, histstep[2]) for histstep in step[:hist]]
        statehist = [s for (s,a) in trajec]
        actionhist = [a for (s,a) in trajec]
    end
    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x,y), m.underlying_pomdp.size)
        r = m.underlying_pomdp.reward[rewardinds(m.underlying_pomdp, SA[x,y])...]
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
        end

        compose!(cell, target)
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(1mm), cells...)
    outline = compose(context(), linewidth(0.05mm), rectangle(), fill("black"), stroke("black"))

    if haskey(step, :sp)
        robot_ctx = cell_ctx(step[:sp].underlying_state.robot, m.underlying_pomdp.size)
        robot = compose(robot_ctx, circle(0.5, 0.5, 0.3), fill("blue"))
        target_ctx = cell_ctx(step[:sp].underlying_state.target, m.underlying_pomdp.size)
        target = compose(target_ctx, star(0.5,0.5,0.5,5,0.5), fill("orange"), stroke("black"))
    else
        robot = nothing
        target = nothing
    end
    sz = min(w,h)
    #return compose(context((w-sz)/2, (h-sz)/2, sz, (ny/nx)*sz), robot, target, grid, outline)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), robot, target, grid, outline)
end
#POMDPTools.ModelTools.render(m::SAR_POMDP_human, step) = POMDPTools.ModelTools.render(m.underlying_pomdp, step)
#POMDPTools.ModelTools.render(m::SAR_POMDP_human, step, plt_reward::Bool) = POMDPTools.ModelTools.render(m.underlying_pomdp, step, plt_reward)