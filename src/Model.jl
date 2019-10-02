struct Metadata

    n_features::Int
    n_pos_samples::Int
    n_neg_samples::Int
    description::String
    trainingtime::Float64
    avg_tree_depth::Float64

end


struct Model

    trees::Vector{Node}
    options::Options
    metadata::Metadata

    function Model(X::Matrix{Float32}, Y::BitArray{1}, opt::Options, description::String="none")

        @assert size(X, 1) == length(Y)

        XS = SharedArray(X)

        starttime = time_ns()
        trees = pmap((arg)->tree_builder(XS, Y, opt), 1:opt.n_trees)
        trainingtime = (time_ns() - starttime)/10^9 

        meta = Metadata(
            size(X, 2),
            sum(Y),
            sum(.!Y),
            description,
            trainingtime,
            mean([tree_depth(tree) for tree in trees])
            )

        return new(trees, opt, meta)

    end

end


Model(X, Y;
    n_trees::Int=1,
    n_subfeat::Int=0,
    n_thresholds::Int=1,
    max_depth::Int=-1,
    min_samples_leaf::Int=1,
    min_samples_split::Int=2,
    description::String="none",
    beta::Float64=0.5
    ) =
    Model(X, Y,
        Options(
            n_trees,
            (1 <= n_subfeat <= size(X, 2)) ? n_subfeat : round(Int, sqrt(size(X, 2))),
            n_thresholds,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            beta
            ),
        description
    )


Base.show(io::IO, ::MIME"text/plain", model::Model) = print(io,
    "Model: ExtraTrees\n",
    "Trees: ", length(model.trees), "\n",
    "Data features: ", model.metadata.n_features, "\n",
    "Average depth: ", round(model.metadata.avg_tree_depth, digits=1), "\n",
    "Training time: ", round(model.metadata.trainingtime, digits=2), " sec\n",
    "Training pos samples: ", model.metadata.n_pos_samples, "\n",
    "Training neg samples: ", model.metadata.n_neg_samples, "\n",
    "Custom description: ", model.metadata.description, "\n"
    )


function (node::Node)(sample::Vector{Float32})

    while !node.is_leaf
        node = sample[node.split.feature] < node.split.threshold ? node.left : node.right
    end

    return node.probability

end


function (model::Model)(sample::Vector{Float32})

    @assert model.metadata.n_features == length(sample)

    probability = 0.0f0

    for tree in model.trees
        probability += tree(sample)
    end

    return probability / length(model.trees)

end


function (model::Model)(X::Matrix{Float32})

    @assert model.metadata.n_features == size(X, 2)

    return [model(X[ss, :]) for ss in 1:size(X, 1)]

end
