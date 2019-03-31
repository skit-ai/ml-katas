function normalize!(counts::Dict{Any,Any})
    total = sum(values(counts))

    for k in keys(counts)
        counts[k] /= total
    end
end


"""
Basic error count evaluator for universal tag prediction. We assume data
from MLDatasets and model from providing array of tags for each sentence.

Also note that this is not the right way to evaluate so we will be careful
when drawing conclusions.
"""
function evaluate(predictions::Array{Prediction,1}, testdata)::Float64
    @assert length(predictions) == length(testdata)

    errors = 0
    tagindex = 4

    for (truth, pred) in zip(testdata, predictions)
        for (tfields, ptag) in zip(truth, pred)
            if ptag != tfields[tagindex]
                errors += 1
            end
        end
    end

    errors / sum(map(length, testdata))
end
