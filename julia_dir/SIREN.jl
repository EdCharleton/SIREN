#---------------------------------------------- IMPORT PACKAGES --------------------------------------------------------
include("SIREN_functions.jl")
using .ClassifierFunctions
using Pkg
using DataStructures
using Random
using DataFrames
using DelimitedFiles
using Statistics    
using Optim
using JSON
using BenchmarkTools
using ProgressBars
using Plots
using LsqFit
using PyCall
using PyPlot
using Conda
using LaTeXStrings
using CSV
using DataStructures: OrderedDict
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
rcParams = matplotlib[:rcParams]



#Set up threading for julia
ENV["JULIA_NUM_THREADS"] = "auto"
n_threads = Threads.nthreads()

#---------------------------------------------- LOAD IN TEMPLATE METADATA ----------------------------------------------
pure_sn_spectra_info = JSON.parsefile(joinpath("transient_template_dir", "metadata","pure_sn_spectra_data.json"), dicttype = OrderedDict)
pure_sn_spectra_file_names = open(joinpath("transient_template_dir", "metadata","pure_sn_spectra_names.txt")) do file
    readlines(file)
end

pure_hg_spectra_info = JSON.parsefile(joinpath("host_template_dir", "metadata","pure_hg_spectra_data.json"), dicttype = OrderedDict)
pure_hg_spectra_file_names = open(joinpath("host_template_dir", "metadata","pure_hg_spectra_names.txt")) do file
    readlines(file)
end

#----------------------------------------- READ IN TARGET AND TEMPLATE SPECTRA -----------------------------------------

show_plots = true
delete_plot_folder_contents = false

if delete_plot_folder_contents == true
    path = joinpath("classifications_dir", "plots" )
    delete_folder_contents(path)
end

println("----------------------------------------------")
println("Loading in Templates...")
println("----------------------------------------------")

#---------------------------------------Read in all the Transient Templates and metadata-----------------------------------------
sn_metadata_path = joinpath("SIREN_test_sets/demo_fits/demo_fits_metadata.csv")

sn_metadata = CSV.read(sn_metadata_path, DataFrame, delim=',')

all_target_names = sn_metadata[!, "Ascii file"]

telluric_mask = readdlm(joinpath("real_spectra_dir", "metadata", "tellurics.txt"))

sn_template_wavs, pure_sn_spectra_files = read_template_files(pure_sn_spectra_file_names, pure_sn_spectra_info, "Supernova Template")
hg_template_wavs, pure_hg_spectra_files = read_template_files(pure_hg_spectra_file_names, pure_hg_spectra_info, "Hg Template")


#read in target spectra
target_spectra_dict = read_target_spectra("SIREN_test_sets/demo_fits/demo_fits_ALL_SPECTRA", ',', 0)
target_spectra_names = collect(keys(target_spectra_dict))

#Initialise an empty dictionary to store the classifications
classifications = OrderedDict()

#-------------------------------------------------- RUN THE CLASSIFIER --------------------------------------------------
println("----------------------------------------------")
println("Classifying ", length(target_spectra_names), " spectra with ", n_threads, " threads...")
println("----------------------------------------------")


if length(target_spectra_names) > 10
    batch_mode = true
else
    batch_mode = false
end


for n in (batch_mode ? ProgressBar(target_spectra_names) : (target_spectra_names))
    target_wl, target_spectra, target_spectra_σ = target_spectra_dict[n]
    target_name = splitext(n)[1]
    target_index = findfirst(x -> x == target_name, all_target_names)

    if isnothing(target_index)
        target_name = n
        target_index = findfirst(x -> x == target_name, all_target_names)
    end

    #Find the closest matching name

    if isnothing(target_index)
        closest_match = find_closest_match(target_name, all_target_names)
        print("Target ", target_name, " not found in $sn_metadata_path", " Closest match is ")
        printstyled(closest_match, color = :red)
        continue
    end


    target_redshift = sn_metadata[target_index, "Redshift"]

    if ismissing(target_redshift)
        println("Target ", target_name, " has no redshift information")
        continue

    elseif target_redshift > 1.0 
        println("Target ", target_name, " has a redshift of ", target_redshift, " which is too high for SIREN")
    end 
    
    target_remove_telluric = "N"

    if batch_mode == false
        println("Classifying ", target_name, "...")
    end
    

    best_params, chi_red, path, z, type = chi_squared_classifier(
        pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files,
        sn_template_wavs, hg_template_wavs, target_spectra, target_wl, target_spectra_σ, target_name, target_redshift[1], telluric_mask, batch_mode, show_plots, target_remove_telluric)

    if batch_mode == false
        println("SN Name           : ", target_name)
        println("Redshift          : ", round(z, digits = 2))
        println("Path              : ", path)
        println("Type              : ", type)
        println("----------------------------------------------")
    end

    classifications[target_name] = (z, type, minimum(chi_red))
end



#Save the classifications as a JSON file
JSON.open(joinpath("classifications_dir", "demo_fits_classifications.json"), "w") do file
    JSON.print(file, classifications)
end

println("Classifications Finished")
println("----------------------------------------------")



