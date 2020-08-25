struct Options

    n_trees::Int
    n_subfeat::Int
    n_thresholds::Int
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    random_forest::Bool

    function Options(n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, random_forest)

        @assert n_trees >= 1
        @assert n_subfeat >= 1
        @assert n_thresholds >= 1
        @assert min_samples_leaf >= 1
        @assert min_samples_split >= 1

        return new(n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, random_forest)

    end

end


struct Split

    cost::Float64
    feature::Int
    threshold::Float32

    n_left::Int
    n_right::Int

end

Split() = Split(Inf, 0, 0.0f0, 0, 0)


mutable struct Node

    id::Int

    depth::Int
    is_leaf::Bool
    probability::Float32

    left::Node
    right::Node
    split::Split

    samples::Vector{Int}
    features::Vector{Int}

    Node(depth, samples, features) = (
        node = new();
        node.depth = depth;
        node.is_leaf = true;
        node.samples = samples;
        node.features = features;
        node)

end


function entropy_loss(V, y, n_pos_samples, n_neg_samples, n_samples, feature, threshold)

    n_left_pos = 0
    n_left_neg = 0

    for ii in 1:n_samples
        if V[ii] < threshold
            if y[ii]
                n_left_pos += 1
            else
                n_left_neg += 1
            end
        end
    end

    n_left = n_left_pos + n_left_neg
    w_left = n_left / n_samples
    p_left_pos = n_left_pos / n_left
    p_left_neg = 1.0 - p_left_pos
    entropy_left = (p_left_neg == 0.0) ? 0.0 : -p_left_neg*log2(p_left_neg)
    entropy_left += (p_left_pos == 0.0) ? 0.0 : -p_left_pos*log2(p_left_pos)

    n_right_pos = n_pos_samples - n_left_pos
    n_right_neg = n_neg_samples - n_left_neg

    n_right = n_right_pos + n_right_neg
    w_right = n_right / n_samples
    p_right_pos = n_right_pos / n_right
    p_right_neg = 1.0 - p_right_pos
    entropy_right = (p_right_neg == 0.0) ? 0.0 : -p_right_neg*log2(p_right_neg)
    entropy_right += (p_right_pos == 0.0) ? 0.0 : -p_right_pos*log2(p_right_pos)

    cost = (w_left * entropy_left) + (w_right * entropy_right)

    return Split(cost, feature, threshold, n_left, n_right)

end


function free!(node, opt)

    if opt.random_forest
        node.probability = (node.probability > 0.5f0) ? 1.0f0 : 0.0f0
    end

    if isdefined(node, :features)
        node.features = Int64[]
    end

    if isdefined(node, :samples)
        node.samples = Int64[]
    end

    return

end


function split!(node, X, Y, opt)

    y = Y[node.samples]
    n_samples = length(node.samples)
    n_pos_samples = sum(y)
    n_neg_samples = n_samples - n_pos_samples
    node.probability = n_pos_samples / n_samples

    if node.depth == opt.max_depth ||
        node.probability == 1.0f0 ||
        node.probability == 0.0f0 ||
        n_samples < opt.min_samples_split ||
        n_samples == opt.min_samples_leaf
        free!(node, opt)
        return
    end

    best_split = Split()

    first_usable_feat = 1
    last_nonconst_feat = length(node.features)

    V = Vector{Float32}(undef, n_samples)

    mtry = 1
    while (mtry <= opt.n_subfeat) &&
        first_usable_feat <= last_nonconst_feat

        ff = rand(first_usable_feat:last_nonconst_feat)
        feature = node.features[ff]
        node.features[ff], node.features[first_usable_feat] =
            node.features[first_usable_feat], node.features[ff]
        first_usable_feat += 1

        minv, maxv = Inf32, -Inf32
        for (ii, ss) in enumerate(node.samples)
            V[ii] = X[ss, feature]
            if minv > V[ii]
                minv = V[ii]
            end
            if maxv < V[ii]
                maxv = V[ii]
            end
        end

        if minv == maxv
            first_usable_feat -= 1
            node.features[first_usable_feat], node.features[last_nonconst_feat] =
                node.features[last_nonconst_feat], node.features[first_usable_feat]
            last_nonconst_feat -= 1
            continue
        end

        thresholds = (opt.random_forest 
            ? collect(minv:((maxv - minv) / opt.n_thresholds):maxv)[2:end]
            : [minv + (maxv - minv) * rand(Float32) for rr in 1:opt.n_thresholds])

        for threshold in thresholds
            split = entropy_loss(V, y, n_pos_samples, n_neg_samples, n_samples, feature, threshold)
            if best_split.cost > split.cost
                best_split = split
            end
        end

        mtry += 1

    end

    if best_split.feature == 0 ||
        best_split.n_left < opt.min_samples_leaf ||
        best_split.n_right < opt.min_samples_leaf
        free!(node, opt)
        return
    end

    node.is_leaf = false
    node.split = best_split

    ll, rr = 1, 1
    left_samples = Vector{Int}(undef, best_split.n_left)
    right_samples = Vector{Int}(undef, best_split.n_right)
    for ss in node.samples
        if X[ss, best_split.feature] < best_split.threshold
            left_samples[ll] = ss
            ll += 1
        else
            right_samples[rr] = ss
            rr += 1
        end
    end

    node.left = Node(node.depth + 1, left_samples, node.features[1:last_nonconst_feat])
    node.right = Node(node.depth + 1, right_samples, node.features[1:last_nonconst_feat])

    free!(node, opt)

    return

end


function tree_builder(X::SharedArray{Float32,2}, Y::BitArray{1}, opt::Options)

    samples = opt.random_forest ? sort(rand(1:size(X, 1), size(X, 1))) : collect(1:size(X, 1))

    root = Node(1, samples, collect(1:size(X, 2)))

    stack = Node[root]
    while length(stack) > 0
        node = pop!(stack)
        split!(node, X, Y, opt)
        if !node.is_leaf
            push!(stack, node.left, node.right)
        end
    end

    return root

end


function tree_depth(node)

    return node.is_leaf ? 1 : 1 + max(tree_depth(node.left), tree_depth(node.right))

end


function tree_nodes(node)

    return node.is_leaf ? 1 : 1 + tree_nodes(node.left) + tree_nodes(node.right)

end


function tree_leaves(node)

    return node.is_leaf ? 1 : tree_leaves(node.left) + tree_leaves(node.right)

end
