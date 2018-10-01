using ExtraTrees
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end


Xconst = ones(Float32, 1000, 10)
Yconst = rand(size(Xconst, 1)) .> 0.5

model = Model(Xconst, Yconst)

@test model.trees[1].is_leaf == true
@test model.trees[1].probability .== mean(Yconst)
@test ExtraTrees.tree_depth(model.trees[1]) == 1
@test ExtraTrees.tree_nodes(model.trees[1]) == 1
@test ExtraTrees.tree_leaves(model.trees[1]) == 1

################

Xrand = rand(Float32, 1000, 3)
Yrand = rand(size(Xrand, 1)) .> 0.5

model = Model(Xrand, Yrand, max_depth=2)

@test model.trees[1].is_leaf == false
@test ExtraTrees.tree_depth(model.trees[1]) == 2
@test ExtraTrees.tree_leaves(model.trees[1]) == 2
@test ExtraTrees.tree_nodes(model.trees[1]) == 1+2

################

model = Model(Xrand, Yrand, min_samples_leaf=length(Yrand))

@test model.trees[1].is_leaf == true
@test model.trees[1].probability .== mean(Yrand)
@test ExtraTrees.tree_depth(model.trees[1]) == 1

################

model = Model(Xrand, Yrand, min_samples_split=length(Yrand))

@test model.trees[1].is_leaf == false
@test ExtraTrees.tree_depth(model.trees[1]) == 2

################

Xoptim = rand(Float32, 1000, 10)
Yoptim = rand(size(Xoptim, 1)) .> 0.5
Xoptim = hcat(Xoptim, Yoptim)

model = Model(Xoptim, Yoptim, n_subfeat=size(Xoptim, 2))

@test model.trees[1].is_leaf == false
@test ExtraTrees.tree_depth(model.trees[1]) == 2
@test mean(Yoptim .== model(Xoptim)) == 1.0

################

D = readcsv("./digits-01.csv", Float32)

Ytrain = D[1:300, 1] .== 1.0
Xtrain = D[1:300, 2:end]

Ytest = D[301:end, 1] .== 1.0
Xtest = D[301:end, 2:end]

tree = Model(Xtrain, Ytrain)
forest = Model(Xtrain, Ytrain, n_trees=100)

acc_tree = mean(Ytest .== (tree(Xtest) .> 0.5))
acc_forest = mean(Ytest .== (forest(Xtest) .> 0.5))

@test acc_tree > 0.5
@test acc_forest > 0.98
@test acc_forest >= acc_tree

################

totally = Model(Xtrain, Ytrain, n_trees=100, n_subfeat=1)

@test mean([ExtraTrees.tree_depth(tree) for tree in totally.trees]) >=
    mean([ExtraTrees.tree_depth(tree) for tree in forest.trees])
