struct FlattenTree

    feature::Vector{Int}
    threshold::Vector{Float32}
    left::Vector{Int}
    right::Vector{Int}

end


function tree_assign_ids!(node, id=1)

    node.id = id

    if !node.is_leaf
        if node.split.n_left >= node.split.n_right
            id = tree_assign_ids!(node.left, id+1)
            id = tree_assign_ids!(node.right, id+1)
        else
            id = tree_assign_ids!(node.right, id+1)
            id = tree_assign_ids!(node.left, id+1)
        end
    end

    return id

end


function tree_converter(root)

    n = tree_assign_ids!(root)

    feature = Vector{Int}(undef, n)
    threshold = Vector{Float32}(undef, n)
    left = Vector{Int}(undef, n)
    right = Vector{Int}(undef, n)

    stack = Node[root]
    while length(stack) > 0
        node = pop!(stack)
        if node.is_leaf
            feature[node.id] = -1
            threshold[node.id] = node.probability
        else
            feature[node.id] = node.split.feature
            threshold[node.id] = node.split.threshold
            left[node.id] = node.left.id
            right[node.id] = node.right.id
            push!(stack, node.left, node.right)
        end
    end

    return FlattenTree(feature, threshold, left, right)

end


struct FlattenModel

    trees::Vector{FlattenTree}
    options::Options
    metadata::Metadata
    oob_samples::Vector{BitArray{1}}

    function FlattenModel(model::Model)

        return new([tree_converter(tree) for tree in model.trees], model.options, model.metadata, model.oob_samples)

    end

end


FlattenModel(X, Y;
    n_trees::Int=1,
    n_subfeat::Int=0,
    n_thresholds::Int=1,
    max_depth::Int=-1,
    min_samples_leaf::Int=1,
    min_samples_split::Int=2,
    rand_thresholds::Bool=true,
    soft_leaf_score::Bool=true,
    bagging::Float64=0.0,
    extra_pos_bagging::Float64=0.0,
    exclude_samples::Set{Int}=Set{Int}(),
    description::String="none"
    ) =
    FlattenModel(
        Model(X, Y,
            Options(
                n_trees,
                (1 <= n_subfeat <= size(X, 2)) ? n_subfeat : round(Int, sqrt(size(X, 2))),
                n_thresholds,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                rand_thresholds,
                soft_leaf_score,
                bagging,
                extra_pos_bagging,
                exclude_samples
                ),
            description
        )
    )


Base.show(io::IO, ::MIME"text/plain", model::FlattenModel) = print(io,
    "Model: ExtraTrees [flatten array-based representation]\n",
    "Trees: ", length(model.trees), "\n",
    "Data features: ", model.metadata.n_features, "\n",
    "Average depth: ", round(model.metadata.avg_tree_depth, digits=1), "\n",
    "Training time: ", round(model.metadata.trainingtime, digits=2), " sec\n",
    "Training pos samples: ", model.metadata.n_pos_samples, "\n",
    "Training neg samples: ", model.metadata.n_neg_samples, "\n",
    "Custom description: ", model.metadata.description, "\n"
    )


function (tree::FlattenTree)(sample::Vector{Float32})

    id = 1
    while tree.feature[id] > 0
        id = sample[tree.feature[id]] < tree.threshold[id] ? tree.left[id] : tree.right[id]
    end

    return tree.threshold[id]

end


function (model::FlattenModel)(sample::Vector{Float32})

    @assert model.metadata.n_features == length(sample)

    probability = 0.0f0

    for tree in model.trees
        probability += tree(sample)
    end

    return probability / length(model.trees)

end


function (model::FlattenModel)(X::Matrix{Float32})

    @assert model.metadata.n_features == size(X, 2)

    return [model(X[ss, :]) for ss in 1:size(X, 1)]

end


function (model::FlattenModel)(XS::SharedArray{Float32,2}, range::UnitRange{Int64})

    X = [XS[ss, :] for ss in range]

    probability = zeros(Float32, length(X))

    for tree in model.trees
        for ss in 1:length(X)
            probability[ss] += tree(X[ss])
        end
    end

    return probability ./ length(model.trees)

end


function oob_model(model::FlattenModel, XS::SharedArray{Float32,2}, range::UnitRange{Int64})

    X = [XS[ss, :] for ss in range]

    scores = zeros(Float32, length(X))
    tree_count = zeros(Int, length(X))

    for (tree, oob_samples) in zip(model.trees, model.oob_samples)
        for xx in 1:length(X)
            if oob_samples[range[xx]]
                scores[xx] += tree(X[xx])
                tree_count[xx] += 1
            end
        end
    end

    return scores ./ tree_count

end


function (model::FlattenModel)(XS::SharedArray{Float32,2}, range::UnitRange{Int64}, oob_samples::Bool)

    return oob_samples ? oob_model(model, XS, range) : model(XS, range)

end


function (model::FlattenModel)(XS::SharedArray{Float32,2}, oob_samples::Bool=false)

    @assert model.metadata.n_features == size(XS, 2)

    if oob_samples
        @assert length(model.oob_samples[1]) == size(XS, 1)
    end

    n_samples = size(XS, 1)
    n_workers = length(workers())
    step_size = ceil(Int, n_samples / n_workers)

    ranges = [ start:((stop=start+step_size-1) < n_samples ? stop : n_samples) for start in 1:step_size:n_samples ]

    results = pmap(range -> (first(range), model(XS, range, oob_samples)), ranges)

    return vcat(last.(sort(results, by=x->x[1]))...)

end
