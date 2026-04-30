using JLD2

# Load the templates to see the distribution
println("Loading templates...")
templates = load("prebuilt_templates.jld2")
transient_template_info = templates["transient_template_info"]
transient_template_names = templates["transient_template_names"]

types = [transient_template_info["Type"][name] for name in transient_template_names]
unique_types = unique(types)
println("Found $(length(unique_types)) types: $unique_types")

for t in unique_types
    count = sum(types .== t)
    println("Type $t: $count templates")
end

# Check if they are contiguous
is_contiguous = true
current_type = types[1]
seen_types = Set([current_type])
for t in types
    global current_type, is_contiguous
    if t != current_type
        if t in seen_types
            is_contiguous = false
            break
        end
        push!(seen_types, t)
        current_type = t
    end
end
println("Are types contiguous? $is_contiguous")
