#---------------------------------------------- IMPORT PACKAGES --------------------------------------------------------
include("classifier_functions.jl")
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
sn_metadata = CSV.read(joinpath("real_spectra_dir", "metadata","sn_data.csv"), DataFrame, delim=',')
all_target_names = sn_metadata[!, "TRANSIENT_NAME"]

telluric_mask = readdlm(joinpath("real_spectra_dir", "metadata", "tellurics.txt"))

sn_template_wavs, pure_sn_spectra_files = read_template_files(pure_sn_spectra_file_names, pure_sn_spectra_info, "Supernova Template")
hg_template_wavs, pure_hg_spectra_files = read_template_files(pure_hg_spectra_file_names, pure_hg_spectra_info, "Hg Template")


#read in target spectra
target_spectra_dict = read_target_spectra("real_spectra_dir")
target_spectra_names = collect(keys(target_spectra_dict))

#-------------------------------------------------- RUN THE CLASSIFIER --------------------------------------------------
println("----------------------------------------------")
println("Classifying ", length(target_spectra_names), " spectra with ", n_threads, " threads...")
println("----------------------------------------------")


for n in (target_spectra_names)
    target_wl, target_spectra, target_spectra_σ = target_spectra_dict[n]
    target_name = splitext(n)[1]
    #When Working on the OzDES data
    target_index = findfirst(x -> x == target_name, all_target_names)
    target_redshift = sn_metadata[target_index, "REDSHIFT"]
    target_remove_telluric = sn_metadata[target_index, "REMOVE_TELLURIC"]


    println("Classifying ", target_name, "...")
    best_params, chi_red, path, z, type = chi_squared_classifier(
        pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files,
         sn_template_wavs, hg_template_wavs, target_spectra, target_wl, target_spectra_σ, target_name, target_redshift[1], telluric_mask, show_plots, target_remove_telluric)

    println("SN Name           : ", target_name)
    println("Redshift          : ", round(z, digits = 2))
    println("Path              : ", path)
    println("Type              : ", type)
    #println("SN fit params     : ", round(best_params[:, 1], digits = 4))
    #4println("HG fit params     : ", round(best_params[:, 2], digits = 4))
    #println("χ^2 Reduced       : ", round(chi_red[], digits = 4))
    println("----------------------------------------------")

end

println("Classifications Finished")
println("----------------------------------------------")



