using CSV
using DataFrames
using DataStructures: OrderedDict
using JSON
using DelimitedFiles
using ProgressBars  
using FilePathsBase

# Parameters
survey_name = "dwayne_unit_tes"
iteration = "7"

SIREN_classes = ["Ia", "Ic", "Ib", "II"]
Survey_classes = ["Ia", "Ic", "Ib", "II"]

number_of_survey_class = zeros(Int, length(Survey_classes))
misclassifications = Dict{String, Array{String, 1}}()

# Initialize confusion matrix
confusion_matrix = OrderedDict{String, OrderedDict{String, Float64}}()
for SIREN_type in SIREN_classes
    confusion_matrix[SIREN_type] = OrderedDict{String, Float64}()
    for Survey_type in Survey_classes
        confusion_matrix[SIREN_type][Survey_type] = 0.0
    end
end

# File paths
classifications_metadata_path = joinpath("/Users/edcharleton/PhD/SIREN/classifications/dwayne_unit_tes_with_redshift", survey_name * "_classifications.json")
Survey_metadata_path = joinpath("/Users/edcharleton/PhD/SIREN/spectra_to_classify/Dwayne", "dwayne_metadata.csv")

# Load classifications JSON & True Classifications CSV
classifications_metadata = JSON.parsefile(classifications_metadata_path)
Survey_metadata = CSV.read(Survey_metadata_path, DataFrame)

# Check for duplicates in survey metadata
filenames = Survey_metadata[!, "Filename"]
if length(filenames) != length(unique(filenames))
    @warn "Duplicate filenames detected in Survey metadata CSV!"
end

match = 0
no_match = 0

classifications_metadata_keys = keys(classifications_metadata)

for key in ProgressBar(classifications_metadata_keys)
    # Extract predicted SIREN classification
    entry = classifications_metadata[key]


    
    # Extract SIREN classification based on JSON structure (Handles Dict or Array/Tuple format)
# Extract SIREN classification safely across multiple JSON formats
    SIREN_type = if entry isa AbstractString
        # Case 1: Entry is directly a string (e.g. "Ia")
        String(entry)
    elseif entry isa Dict
        # Case 2: Dict - check common key names (String or Symbol keys)
        possible_keys = ["type", "transient_type", "classification", "class", "label", "predicted_type", "best_match"]
        found_type = "Unknown"
        for k in possible_keys
            if haskey(entry, k) && !isnothing(entry[k])
                found_type = string(entry[k])
                break
            elseif haskey(entry, Symbol(k)) && !isnothing(entry[Symbol(k)])
                found_type = string(entry[Symbol(k)])
                break
            end
        end
        found_type
    elseif entry isa AbstractVector && !isempty(entry)
        # Case 3: Array - try second element if available, otherwise fallback to first
        length(entry) >= 2 ? string(entry[2]) : string(entry[1])
    else
        "Unknown"
    end


    # Normalize SIREN Supernova types
    if SIREN_type in ["91T-like", "91bg-like", "91bg", "91T", "SN91T-like", "SN91t-like", "91t-like", "Ia-91t", "Ia-91bg","Ia-91t-like", "Ia-91bg-like","SN 91-bg-like"]
        SIREN_type = "Ia"
    elseif SIREN_type in ["IIP", "IIL", "IIn", "IIb", "II-pec", "II-flash"]
        SIREN_type = "II"
    elseif SIREN_type in ["Ic-BL", "Ic-norm", "Ic-broad"]
        SIREN_type = "Ic"
    elseif SIREN_type in ["Ib-pec"]
        SIREN_type = "Ib"
    end

    # Fetch true classification from CSV metadata
    row_idx = findfirst(==(key), Survey_metadata[!, "Filename"])
    if isnothing(row_idx)
        @warn "Key not found in metadata CSV: $key"
        continue
    end

    Survey_type = Survey_metadata[row_idx, "Type"]
    if isnothing(Survey_type) || ismissing(Survey_type)
        continue
    end

    # Clean & Normalize True Survey classification strings
    Survey_type = replace(String(Survey_type), r"[\s\?]" => "")
    Survey_type = replace(Survey_type, "SN" => "")

    if Survey_type in ["Ia-91T-like", "Ia-91bg-like", "-Ia-91bg-like", "91-bg-like", "Ia-pec", "Ia", "Ca_Ia", "Ia02es-like", "Ia99aa-like", "Ia-02cxlike", "Ia-CSM", "Ia-CSM-(ambigious)", "Ia-rapid", "Ia-91t", "Ia-91bg"]
        Survey_type = "Ia"
    elseif Survey_type in ["Ibn", "Ib", "Ca-Ib", "Ib-pec"]
        Survey_type = "Ib"
    elseif Survey_type in ["IIn", "IIL", "IIP", "IIn-pec", "IIn-09ip", "II-pec", "IIb", "II", "II-flash"]
        Survey_type = "II"
    elseif Survey_type in ["Ic-BL", "Ic", "Ic-pec", "Ic-norm", "Ic-broad"]
        Survey_type = "Ic"
    elseif Survey_type in ["SLSN-I", "SLSN-II", "SL-I", "SLSN-Ib", "computed", "CV", "FBOT", "ILRT", "SLSN-IIn", "SLSN-IIb", "Imposter"]
        continue
    else
        @warn "Unrecognized Survey type for $key: $Survey_type"
        continue
    end

    # Handle ambiguous Ib/c classification
    if Survey_type == "Ib/c" && SIREN_type in ["Ib", "Ic"]
        Survey_type = SIREN_type
    end

    # Update Confusion Matrix Counters
    if SIREN_type in SIREN_classes && Survey_type in Survey_classes
        confusion_matrix[SIREN_type][Survey_type] += 1.0
        
        Survey_index = findfirst(==(Survey_type), Survey_classes)
        number_of_survey_class[Survey_index] += 1
    else
        @warn "Class out of matrix scope for $key: SIREN=$SIREN_type, True=$Survey_type"
        continue
    end

    # Check accuracy
    if Survey_type == SIREN_type
        global match += 1
    else
        misclassifications[key] = [SIREN_type, Survey_type]
        global no_match += 1
    end
end

# Normalize the confusion matrix column-wise
for (s_idx, Survey_type) in enumerate(Survey_classes)
    total_class_count = number_of_survey_class[s_idx]
    if total_class_count > 0
        for SIREN_type in SIREN_classes
            confusion_matrix[SIREN_type][Survey_type] /= total_class_count
        end
    end
end

# Output Results
println("\n--- Evaluation Summary ---")
println("Matched: ", match)
println("Not Matched: ", no_match)
println("Total Evaluated: ", match + no_match)
if (match + no_match) > 0
    println("Accuracy: ", round(match / (match + no_match) * 100, digits=2), "%")
end

# Export JSON files
output_dir = "/Users/edcharleton/PhD/SIREN/spectra_to_classify/Dwayne/testing"
mkpath(output_dir)

confusion_matrix_path = joinpath(output_dir, "$(survey_name)$(iteration)_confusion_matrix.json")
open(confusion_matrix_path, "w") do io
    JSON.print(io, confusion_matrix, 4)
end

misclassifications_path = joinpath(output_dir, "$(survey_name)$(iteration)_misclassifications.json")
open(misclassifications_path, "w") do io
    JSON.print(io, misclassifications, 4)
end