using CSV
using DataFrames
using DataStructures: OrderedDict
using JSON
using DelimitedFiles
using PyCall
using ProgressBars  
using FilePathsBase

# Import the two metadata files

# Read in the Survey metadata
survey_name = "Dwayne"
iteration = "7"

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

#look for duplicates in the survey metadata
for i in 1:length(Survey_metadata[!, "File Name"])
    for j in i+1:length(Survey_metadata[!, "File Name"])
        if Survey_metadata[!, "File Name"][i] == Survey_metadata[!, "File Name"][j]
            println("Duplicate: ", Survey_metadata[!, "File Name"][i])
        end
    end
end

# Get the names of the dictionary keys

match = 0
no_match = 0

classifications_metadata_keys = keys(classifications_metadata)
println(length(classifications_metadata_keys))
for key in ProgressBar(classifications_metadata_keys)
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

    elseif isnothing(SIREN_type)
        println(key)
        continue
    end

    try
        file_names = Survey_metadata[!, "File Name"]
        Survey_type = Survey_metadata[!, "Type"][findfirst(x -> x == key, file_names)]
    catch e
        @warn "Error finding survey type for $key: $e"
        continue
    end
        
    #Survey_type = Survey_metadata[!, "Type"][findfirst(x -> x == key, Survey_metadata[!, "File Name"])]

    # Strip any whitespace from the Survey type
    Survey_type = replace(Survey_type, " " => "")
    Survey_type = replace(Survey_type, "?" => "")
    Survey_type = replace(Survey_type, "SN" => "")

    if Survey_type in ["Ia-91T-like", "Ia-91bg-like", "-Ia-91bg-like", "91-bg-like", "Ia-pec", "Ia", "Ca_Ia", "Ia 02es-like", "Ia 99aa-like", "Ia-02cx like", "Ia-CSM", "Ia-CSM-(ambigious)", "Ia-pec", "Ia-rapid"]
        Survey_type = "Ia"
    elseif Survey_type in ["Ibn", "Ib", "Ca-Ib"]
        Survey_type = "Ib"
    elseif Survey_type in ["IIn", "IIL", "IIP", "IIn-pec", "IIn-09ip", "II-pec", "IIb", "II", "II-flash", "IIn"]
        Survey_type = "II"
    elseif Survey_type in ["Ib-pec", "Ib/c"]
        Survey_type = "Ib"
    elseif Survey_type in ["Ic-BL", "Ic", "Ic-pec"]
        Survey_type = "Ic"
    elseif Survey_type in ["SLSN-I", "SLSN-II", "SL-I", "SLSN-Ib", "computed", "CV", "FBOT", "ILRT", "SLSN-IIn", "SLSN-IIb", "SLSN-IIn", "SN - Imposter", ]
        continue
    elseif isnothing(Survey_type)
        println(key)
        continue
    else
        @warn "Error finding survey type for $key: $Survey_type"
    end§

    # If Ib/c mark as correct if either Ib or Ic
    if Survey_type == "Ib/c" && (SIREN_type == "Ib" || SIREN_type == "Ic")
        Survey_type = SIREN_type
    end

    # Add to the confusion matrix
    if SIREN_type in SIREN_classes && Survey_type in Survey_classes
        confusion_matrix[SIREN_type][Survey_type] += 1
        Survey_index = findfirst(x -> x == Survey_type, Survey_classes)
        number_of_survey_class[Survey_index] += 1
    else
        @warn "Error adding to confusion matrix for $key: $SIREN_type, $Survey_type"
    end

    if Survey_type == SIREN_type
        global match += 1
        # Move the plot to the folder called Dwayne_correct_fits
        plot_source_path = joinpath("/Users/edcharleton/PhD/SIREN/classifications_dir/plots/Dwayne", key * ".pdf")
        plot_destination_path = joinpath("/Users/edcharleton/PhD/SIREN/SIREN_test_sets/Dwayne_2_correct_fits", key * ".pdf")


        #try
            #cp(plot_source_path, plot_destination_path)
        #catch e
            #@warn "Error moving plot for $key: $e"
        #end
        
    else
        # Export the names of the misclassified objects, what it was classified as and what it should have been classified as
        misclassifications[key] = [SIREN_type, Survey_type]
        global no_match += 1
        plot_source_path = joinpath("/Users/edcharleton/PhD/SIREN/classifications_dir/plots/Dwayne", key * ".pdf")
        plot_destination_path = joinpath("/Users/edcharleton/PhD/SIREN/SIREN_test_sets/Dwayne_2_incorrect_fits", key * ".pdf")

        #try
           # cp(plot_source_path, plot_destination_path)
        #catch e
        #    @warn "Error moving plot for $key: $e"
        #end
    
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

println(match + no_match)

# And as a percentage
println("Matched: ", match / (match + no_match) * 100, "%")

# Save the confusion matrix to a json file
confusion_matrix_path = joinpath("SIREN_test_sets/statistics", survey_name * iteration * "_confusion_matrix.json")
open(confusion_matrix_path, "w") do io
    JSON.print(io, confusion_matrix)
end

# Export the misclassifications
misclassifications_path = joinpath("SIREN_test_sets/statistics", survey_name * iteration * "_misclassifications.json")
open(misclassifications_path, "w") do io
    JSON.print(io, misclassifications)
end
