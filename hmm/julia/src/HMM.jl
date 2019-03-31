module HMM

import Random
using StatsBase

export TAGS
export StemCount, stemtrain, predict
export evaluate

include("types.jl")
include("utils.jl")

# These are the tags that we are interested in predicting
const TAGS = [
    "DET",
    "PROPN",
    "PUNCT",
    "ADJ",
    "SCONJ",
    "NOUN",
    "NUM",
    "X",
    "VERB",
    "ADV",
    "AUX",
    "PRON",
    "PART",
    "CCONJ",
    "ADP",
    "SYM",
    "INTJ"
]

"""
A random tag prediction that gives around 94% error
"""
function predict(x::Instance)::Prediction
    map(_ -> TAGS[Random.rand(1:end)], x)
end

"""
Plain unigram-ish count gives 18% error
"""
struct StemCount
    counts::Dict{String, Dict{String, Float64}}
end

function stemtrain(traindata::Array{Instance,1})::StemCount
    counts = Dict()
    stemindex = 3
    tagindex = 4

    for x in traindata
        for word in x
            stem = word[stemindex]
            tag = word[tagindex]

            if !(stem in keys(counts))
                counts[stem] = Dict()
            end

            counts[stem][tag] = get(counts[stem], tag, 0) + 1
        end
    end

    for stem in keys(counts)
        normalize!(counts[stem])
    end

    StemCount(counts)
end

function predict(model::StemCount, x::Instance)::Prediction
    output = []
    stemindex = 3

    for i in 1:length(x)
        conditionals = get(model.counts, x[i][stemindex], Dict())
        weights = ProbabilityWeights(map(t -> get(conditionals, t, 0), TAGS))

        push!(output, sample(TAGS, weights))
    end

    output
end

struct HMModel
    n::Int # Number of states
    m::Int # Number of observation states

    transition::Array{Float64,2}
    initial::Array{Float64,1}
    # TODO: Emissions should be something different
    emission::Array{Float64,2}
end

"""
Tell likelihood of observed values (words here). This is the
problem 1 from rabiner.
"""
function likelihood(model::HMModel, x::Instance)::Float64
    obindex = 3 # Choosing stem as the observation
    observations = [w[obindex] for w in x]

    # TODO: We need to go from observation to a certain index
    o2i(o) = 1

    # Converting to row vector for convenience
    statedist = model.initial'

    totalprob = 0
    for o in observations
        totalprob += sum(statedist .* emmission[:, o2i(o)])
        statedist = statedist * transition
    end

    totalprob
end

"""
Estimate parameters for the hmm
"""
function hmmtrain(traindata::Array{Instance,1})::HMModel
    # TODO
end

"""
Problem 2 from rabiner. NOTE: Only when the observed sequence is words
and hidden states are tags.
"""
function predict(model::HMModel, x::Instance)::Prediction
    # TODO
end

end
