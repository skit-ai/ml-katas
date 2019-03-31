function normalize!(counts::Dict{Any,Any})
    total = sum(values(counts))

    for k in keys(counts)
        counts[k] /= total
    end
end
