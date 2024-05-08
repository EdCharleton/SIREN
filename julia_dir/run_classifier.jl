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
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
rcParams = matplotlib[:rcParams]

#Set up threading for julia
ENV["JULIA_NUM_THREADS"] = "auto"
n_threads = Threads.nthreads()

#---------------------------------------------- LOAD IN TEMPLATE METADATA ----------------------------------------------

four_most_wl = collect(3790:20:9510)

using DataStructures: OrderedDict
pure_sn_spectra_info = JSON.parsefile(joinpath("sn_template_dir", "metadata","pure_sn_spectra_data.json"), dicttype = OrderedDict)
pure_sn_spectra_file_names = open(joinpath("sn_template_dir", "metadata","pure_sn_spectra_names.txt")) do file
    readlines(file)
end

pure_hg_spectra_info = JSON.parsefile(joinpath("hg_template_dir", "metadata","pure_hg_spectra_data.json"), dicttype = OrderedDict)
pure_hg_spectra_file_names = open(joinpath("hg_template_dir", "metadata","pure_hg_spectra_names.txt")) do file
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
#---------------------------------------Temporary while working with OzDES data-----------------------------------------
OzDES_metadata = CSV.read(joinpath("real_spectra_dir", "metadata","snspect.csv"), DataFrame, delim=',')
all_target_names = OzDES_metadata[!, "TRANSIENT_NAME"]

pure_sn_spectra_files = read_template_files(pure_sn_spectra_file_names, pure_sn_spectra_info, "Supernova Template")
pure_hg_spectra_files = read_template_files(pure_hg_spectra_file_names, pure_hg_spectra_info, "Hg Template")


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
    target_index = findall(x -> occursin(x, target_name), all_target_names)
    target_redshift = OzDES_metadata[target_index, "Z_GAL"]
    if target_redshift[1] > 10.0 || target_redshift[1] < 0.0
        target_redshift[1] = Float16(OzDES_metadata[target_index, "Z_SN"][1])
    end

    println("Classifying ", target_name, "...")


    name_of_best_sn_match, name_of_best_hg_match, best_params, chi_red = chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, target_spectra, target_wl, target_spectra_σ, target_name, target_redshift[1], show_plots)

    println("SN Name           : ", target_name)
    println("SN fit params     : ", round(best_params[1], digits = 4))
    println("HG fit params     : ", round(best_params[2], digits = 4))
    println("χ^2 Reduced       : ", round(chi_red, digits = 4))
    println("----------------------------------------------")

end

println("Classifications Finished")
println("----------------------------------------------")



