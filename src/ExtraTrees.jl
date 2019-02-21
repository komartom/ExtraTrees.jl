module ExtraTrees

using Statistics, Distributed, SharedArrays

export Model, FlattenModel

include("./Tree.jl")
include("./Model.jl")
include("./FlattenModel.jl")

end # module
