using CSV
using DataFrames
using DataStructures: OrderedDict
using JSON
using DelimitedFiles
using PyCall

# Import the two metadata files

# Read in the Survey metadata
survey_name = "SNLS"

SIREN_classes = ["Ia", "Ic", "Ib", "II"]
Survey_classes = ["Ia", "Ic", "Ib", "II"]

number_of_survey_class = zeros(Int, length(Survey_classes))

misclassifications = Dict{String, Array{String, 1}}()

# Initialize the confusion matrix as an OrderedDict
confusion_matrix = OrderedDict{String, OrderedDict{String, Float64}}()
for SIREN_type in SIREN_classes
    confusion_matrix[SIREN_type] = OrderedDict{String, Float64}()
    for Survey_type in Survey_classes
        confusion_matrix[SIREN_type][Survey_type] = 0.0
    end
end

classifications_metadata_path = joinpath("classifications_dir", survey_name * "_classifications.json")
Survey_metadata_path = joinpath("SIREN_test_sets", survey_name, survey_name * "_metadata.csv")

classifications_metadata = JSON.parsefile(classifications_metadata_path)
Survey_metadata = CSV.read(Survey_metadata_path, DataFrame, delim=',')

# Get the names of the dictionary keys

match = 0
no_match = 0

classifications_metadata_keys = keys(classifications_metadata)
for key in classifications_metadata_keys
    SIREN_type = classifications_metadata[key][2]
    if SIREN_type == "Unknown"
        continue
    end
    if SIREN_type == "91T-like" || SIREN_type == "91bg-like" || SIREN_type == "91bg" || SIREN_type == "91T" || SIREN_type == "SN91T-like" || SIREN_type == "SN91t-like" || SIREN_type == "91t-like"
        SIREN_type = "Ia"
    
    elseif SIREN_type == "IIP" || SIREN_type == "IIL" || SIREN_type == "IIn" || SIREN_type == "IIb"
        SIREN_type = "II"

    elseif SIREN_type == "Ic-BL"
        SIREN_type = "Ic"
    end
    Survey_type = Survey_metadata[!, "Obj. Type"][findfirst(x -> x == key, Survey_metadata[!, "Ascii file"])]

    # Strip any whitespace from the Survey type
    Survey_type = replace(Survey_type, " " => "")
    Survey_type = replace(Survey_type, "?" => "")
    Survey_type = replace(Survey_type, "SN" => "")

    if Survey_type in ["Ia-91T-like", "Ia-91bg-like"]
        Survey_type = "Ia"
    elseif Survey_type in ["Ibn"]
        Survey_type = "Ib"
    elseif Survey_type in ["IIn", "IIL", "IIP", "IIn-pec", "IIn-09ip", "II-pec", "IIb"]
        Survey_type = "II"
    elseif Survey_type == "Ic-BL"
        Survey_type = "Ic"
    elseif Survey_type in ["SLSN-I", "SLSN-II", "SL-I", ""]
        continue
    elseif isnothing(Survey_type)
        continue
    end

    # If Ib/c mark as correct if either Ib or Ic
    if Survey_type == "Ib/c" && (SIREN_type == "Ib" || SIREN_type == "Ic")
        Survey_type = SIREN_type
    end

    # Add to the confusion matrix
    if SIREN_type in SIREN_classes && Survey_type in Survey_classes
        confusion_matrix[SIREN_type][Survey_type] += 1
        Survey_index = findfirst(x -> x == Survey_type, Survey_classes)
        number_of_survey_class[Survey_index] += 1
    end

    if Survey_type == SIREN_type
        global match += 1
    else
        # Export the names of the misclassified objects, what it was classified as and what it should have been classified as
        misclassifications[key] = [SIREN_type, Survey_type]
        global no_match += 1
    end
end

# Normalize the confusion matrix
for SIREN_type in SIREN_classes
    for Survey_type in Survey_classes
        if number_of_survey_class[findfirst(x -> x == Survey_type, Survey_classes)] > 0
            confusion_matrix[SIREN_type][Survey_type] /= number_of_survey_class[findfirst(x -> x == Survey_type, Survey_classes)]
        end
    end
end

# Print the matched and unmatched counts
println("Matched: ", match)
println("Not Matched: ", no_match)

# And as a percentage
println("Matched: ", match / (match + no_match) * 100, "%")

println(confusion_matrix)

# Save the confusion matrix to a json file
confusion_matrix_path = joinpath("SIREN_test_sets/statistics", survey_name * "_confusion_matrix.json")
open(confusion_matrix_path, "w") do io
    JSON.print(io, confusion_matrix)
end

# Export the misclassifications
misclassifications_path = joinpath("SIREN_test_sets/statistics", survey_name * "_misclassifications.json")
open(misclassifications_path, "w") do io
    JSON.print(io, misclassifications)
end
