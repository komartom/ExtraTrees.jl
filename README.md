# ExtraTrees.jl
[![Build Status](https://travis-ci.org/komartom/ExtraTrees.jl.svg?branch=master)](https://travis-ci.org/komartom/ExtraTrees.jl) [![codecov.io](http://codecov.io/github/komartom/ExtraTrees.jl/coverage.svg?branch=master)](http://codecov.io/github/komartom/ExtraTrees.jl?branch=master)

Julia implementation of Extremely (Totally) Randomized Trees

* only binary classification is supported

[Pierre Geurts, Damien Ernst, and Louis Wehenkel. 2006. Extremely randomized trees. Mach. Learn. 63, 1 (April 2006), 3-42. ](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf)

## Installation
```julia
] add https://github.com/komartom/ExtraTrees.jl.git
```

## Simple example
5-times repeated 10-fold cross-validation on Ionosphere dataset
```julia
using ExtraTrees, DelimitedFiles, Statistics, Printf

download("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data", "ionosphere.csv")
D = readdlm("ionosphere.csv", ',')
X = convert(Matrix{Float32}, D[:, 1:end-1])
Y = D[:, end] .== "g"

const FOLDS = 10
const REPETITIONS = 5

accuracy = zeros(FOLDS, REPETITIONS)

for rr in 1:REPETITIONS

    ind = rand(1:FOLDS, length(Y))

    for ff in 1:FOLDS

        train = ind .!= ff
        test = ind .== ff

        # Training and testing ExtraTrees model
        model = Model(X[train,:], Y[train], n_trees=100)
        predictions = model(X[test,:]) .> 0.5

        accuracy[ff, rr] = mean(Y[test] .== predictions)

    end

end

print(@sprintf("Accuracy: %0.2f", mean(mean(accuracy, dims=1)))) #Accuracy: 0.94
```

## Options
```julia
methods(Model)
# ExtraTrees.Model(X, Y; n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, description)
```
