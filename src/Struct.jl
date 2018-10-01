immutable Split

    cost::Float64
    feature::Int
    threshold::Float32

    n_left::Int
    n_right::Int

end

Split() = Split(Inf, 0, 0.0f0, 0, 0)


mutable struct Node

    depth::Int
    is_leaf::Bool
    probability::Float64

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


immutable Options

    n_trees::Int
    n_subfeat::Int
    n_thresholds::Int
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    description::String

    function Options(n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, description)

        @assert n_trees >= 1
        @assert n_subfeat >= 1
        @assert n_thresholds >= 1
        @assert min_samples_leaf >= 1
        @assert min_samples_split >= 1

        return new(n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, description)

    end

end
