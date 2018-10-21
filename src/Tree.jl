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


function free!(node)

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
        node.probability == 1.0 ||
        node.probability == 0.0 ||
        n_samples < opt.min_samples_split ||
        n_samples == opt.min_samples_leaf
        free!(node)
        return
    end

    best_split = Split()

    first_usable_feat = 1
    last_nonconst_feat = length(node.features)

    V = Vector{Float32}(n_samples)

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

        for tt in 1:opt.n_thresholds
            threshold = minv + (maxv - minv) * rand(Float32)
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
        free!(node)
        return
    end

    node.is_leaf = false
    node.split = best_split

    ll, rr = 1, 1
    left_samples = Vector{Int}(best_split.n_left)
    right_samples = Vector{Int}(best_split.n_right)
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

    free!(node)

    return

end


function tree_builder(X::SharedArray{Float32,2}, Y::BitArray{1}, opt::Options)

    root = Node(1, collect(1:size(X, 1)), collect(1:size(X, 2)))

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
