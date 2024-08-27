using CSV
using DataFrames
using DataStructures: OrderedDict
using JSON
using DelimitedFiles

#import the two metadata files

#Read in the survey metadata


match = 0
no_match = 0

classes = ["Ia", "Ic", "Ib", "II"]
misclassifications = Dict{String, Array{String, 1}}()

confusion_matrix = zeros(Float64, length(classes), length(classes))


classifications_metadata_path = joinpath("classifications_dir/SNLS_classifications.json")
survey_metadata_path = joinpath("SIREN_test_sets/SNLS/SNLS_metadata.csv")


classifications_metadata= JSON.parsefile(classifications_metadata_path)
survey_metadata = CSV.read(survey_metadata_path, DataFrame, delim=',')

#get the names of the dictionary keys

classifications_metadata_keys = keys(classifications_metadata)
for key in classifications_metadata_keys
    SIREN_type = classifications_metadata[key][2]
    if SIREN_type == "Unknown"
        continue
    end
    if SIREN_type == "91T-like" || SIREN_type == "91bg-like" || SIREN_type == "91bg" || SIREN_type == "91T" || SIREN_type == "SN91T-like" ||SIREN_type == "SN91t-like" || SIREN_type == "91t-like"
        SIREN_type = "Ia"
    
    elseif SIREN_type == "IIP" || SIREN_type == "IIL" || SIREN_type == "IIn" || SIREN_type == "IIb"
        SIREN_type = "II"

    elseif SIREN_type == "Ic-BL"
        SIREN_type = "Ic"
    end
    survey_type = survey_metadata[!, "Obj. Type"][findfirst(x -> x == key, survey_metadata[!, "Ascii file"])]

    #Strip any whitespace from the survey type
    survey_type = replace(survey_type, " " => "")
    survey_type = replace(survey_type, "?" => "")

    survey_type = replace(survey_type, "SN" => "")

    if survey_type in ["Ia-91T-like", "Ia-91bg-like"]
        survey_type = "Ia"
    elseif survey_type in ["Ibn"]
        survey_type = "Ib"
    elseif survey_type in ["IIn", "IIL", "IIP", "IIn-pec", "IIn-09ip", "II-pec", "IIb"]
        survey_type = "II"
    elseif survey_type == "Ic-BL"
        survey_type = "Ic"
    elseif survey_type in ["SLSN-I", "SLSN-II", "SL-I", ""]
        continue
    elseif isnothing(survey_type)
        continue
    end

    #If Ib/c mark as correct if either Ib or Ic
    if survey_type == "Ib/c" && (SIREN_type == "Ib" || SIREN_type == "Ic")
        survey_type = SIREN_type
    end

    println("survey: ", survey_type, " SIREN: ", SIREN_type)


    #Add to the confusion matrix
    if survey_type in classes
        survey_index = findfirst(x -> x == survey_type, classes)
        SIREN_index = findfirst(x -> x == SIREN_type, classes)
        confusion_matrix[survey_index, SIREN_index] += 1
    end

    if survey_type == SIREN_type
        global match += 1
    else
        #export the names of the misclassified objects, what it was classified as and what it should have been classified as
        misclassifications[key] = [SIREN_type, survey_type]
        global no_match += 1
    end
end

println(length(misclassifications))


confusion_matrix = replace(confusion_matrix, NaN => 0)

#Normalise the confusion matrix
number_of_survey = sum(confusion_matrix, dims=1)

#Normalise the confusion matrix
confusion_matrix = confusion_matrix ./ number_of_survey

#print the matched and unmatched counts
println("Matched: ", match)
println("Not Matched: ", no_match)

#and as a percemtage
println("Matched: ", match / (match + no_match) * 100, "%")

#plot the confusion matrix
using Plots
#Flip the y axis
heatmap(classes, classes, confusion_matrix, aspect_ratio=1, c=:viridis, xlabel="SIREN Classification", ylabel="survey Classification", title="Confusion Matrix", fmt=:png, size=(800, 800), dpi=300, xrotation=45, yrotation=45)
#print the numbers on the heatmap
for i in 1:length(classes)
for j in 1:length(classes)
    annotate!(i-0.5, j-0.5, text(string(round(confusion_matrix[i, j], digits = 2)), :black, :center, 10))
end
end

# Display the plot
display(current())

println("Number of each survey classifications", number_of_survey)

#export the misclassifications
misclassifications_path = joinpath("SIREN_test_sets/misclassifications.json")
open(misclassifications_path, "w") do io
JSON.print(io, misclassifications)
end
