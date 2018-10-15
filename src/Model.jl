immutable Model

    trees::Vector{Node}
    options::Options

    n_samples::Int
    n_features::Int
    n_pos_samples::Int
    n_neg_samples::Int
    trainingtime::Float64

    function Model(X::Matrix{Float32}, Y::BitArray{1}, opt::Options)

        @assert size(X, 1) == length(Y)

        XS = SharedArray(X)

        tic()

        return new(
            pmap((arg)->tree_builder(XS, Y, opt), 1:opt.n_trees),
            opt,
            size(X, 1),
            size(X, 2),
            sum(Y),
            sum(.!Y),
            toq())

    end

end


Model(X, Y;
    n_trees::Int=1,
    n_subfeat::Int=0,
    n_thresholds::Int=1,
    max_depth::Int=-1,
    min_samples_leaf::Int=1,
    min_samples_split::Int=2,
    description::String="none"
    ) =
    Model(X, Y, Options(
        n_trees,
        (1 <= n_subfeat <= size(X, 2)) ? n_subfeat : round(Int, sqrt(size(X, 2))),
        n_thresholds,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        description
    ))


Base.show(io::IO, ::MIME"text/plain", model::Model) = print(io,
    "Trees: ", length(model.trees), "\n",
    "Avg. depth: ", round(mean([tree_depth(tree) for tree in model.trees]), 1), "\n",
    "Training time: ", round(model.trainingtime, 2), " sec\n",
    "Training samples: ", model.n_samples, "\n",
    "Description: ", model.options.description, "\n"
    )


function (node::Node)(sample::Vector{Float32})

    while !node.is_leaf
        node = sample[node.split.feature] < node.split.threshold ? node.left : node.right
    end

    return node.probability

end


function (model::Model)(sample::Vector{Float32})

    probability = 0.0

    for tree in model.trees
        probability += tree(sample)
    end

    return probability / length(model.trees)

end


function (model::Model)(X::Matrix{Float32})

    @assert model.n_features == size(X, 2)

    return [model(X[ss, :]) for ss in 1:size(X, 1)]

end
