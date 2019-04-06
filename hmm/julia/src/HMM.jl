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

    initial::Array{Float64,1}
    transition::Array{Float64,2}

    # Observation function
    ofn
    emission::Array{Float64,2}
end

"""
Return function for getting observation index from input word representation
"""
function observationfn(traindata::Array{Instance,1})
    # NOTE: This is very basic observation, will be changing it
    stemindex = 3

    traintokens = []
    for x in traindata
        for word in x
            push!(traintokens, lowercase(word[stemindex]))
        end
    end
    traintokens = unique(sort(traintokens))
    token2index = Dict(tk=>i for (i, tk) in enumerate(traintokens))

    function word2o(word)::Int
        if !(lowercase(word[stemindex]) in keys(token2index))
            # Last token represents oov
            return length(traintokens) + 1
        else
            token2index[lowercase(word[stemindex])]
        end
    end
end

"""
Tell likelihood of observed values (words here).
"""
function likelihood(model::HMModel, x::Instance)::Float64
    # TODO: Check why likelihood values are not aligning
    statedist = model.initial'
    statedist = statedist .* model.emission[:, model.ofn(x[1])]'

    for word in x[2:end]
        oi = model.ofn(word)
        statedist = statedist * model.transition
        statedist = statedist .* model.emission[:, oi]'
    end

    sum(statedist)
end

"""
Estimate parameters for the hmm for supervised training. Use the tags as
hidden states.
"""
function hmmtrain_supervised(traindata::Array{Instance,1})::HMModel
    ofn = observationfn(traindata)
    m = ofn("some gibberish that is not in vocab")

    initial = hmmtrain_initial(traindata)
    transition = hmmtrain_transition(traindata)
    emission = hmmtrain_emission(traindata)

    HMModel(length(TAGS), m, initial, transition, ofn, emission)
end

"""
Return initial state probabilities
"""
function hmmtrain_initial(traindata::Array{Instance,1})::Array{Float64,1}
    tag2i = Dict(t=>i for (i, t) in enumerate(TAGS))
    initial = zeros(length(TAGS))
    tagindex = 4

    for x in traindata
        initial[tag2i[x[1][tagindex]]] += 1
    end

    initial / sum(initial)
end

"""
Return state transition probabilities
"""
function hmmtrain_transition(traindata::Array{Instance,1})::Array{Float64,2}
    tag2i = Dict(t=>i for (i, t) in enumerate(TAGS))
    tagindex = 4
    transition = zeros(length(TAGS), length(TAGS))

    for x in traindata
        for i in 1:(length(x)-1)
            begtag = x[i][tagindex]
            endtag = x[i + 1][tagindex]

            transition[tag2i[begtag], tag2i[endtag]] += 1
        end
    end

    transition ./ sum(transition, dims=2)
end

"""
Return emission probabilities from n states to m observations
"""
function hmmtrain_emission(traindata::Array{Instance,1})::Array{Float64,2}
    tag2i = Dict(t=>i for (i, t) in enumerate(TAGS))
    ofn = observationfn(traindata)
    m = ofn("some gibberish that is not in vocab")
    stemindex = 3
    tagindex = 4

    # Start with 1 for smoothing
    emission = ones(length(TAGS), m)
    for x in traindata
        for word in x
            oi = ofn(word)
            ni = tag2i[word[tagindex]]
            emission[ni, oi] += 1
        end
    end

    emission ./ sum(emission, dims=2)
end

"""
Look at the observations (features from words) and predict internal
states (tags)
"""
function predict(model::HMModel, x::Instance)::Prediction
    stateprob = model.initial'

    # TODO: This is a basic decoding method, not the most correct one.
    tags = []
    for word in x
        oi = model.ofn(word)
        stateprob = stateprob .* model.emission[:, oi]'
        # NOTE: since we don't care about the value across time steps, we are
        #       normalizing to avoid underflow
        stateprob /= sum(stateprob)

        push!(tags, TAGS[argmax(stateprob')])
        stateprob = stateprob * model.transition
    end

    tags
end

end
