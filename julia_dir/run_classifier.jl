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
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
rcParams = matplotlib[:rcParams]


# Load the data
four_most_wl = collect(3790:20:9510)

using DataStructures: OrderedDict
pure_sn_spectra_info = JSON.parsefile(joinpath("pure_sn_spectra_dir", "metadata","pure_sn_spectra_data.json"), dicttype = OrderedDict)
pure_sn_spectra_file_names = open(joinpath("pure_sn_spectra_dir", "metadata","pure_sn_spectra_names.txt")) do file
    readlines(file)
end

noisey_sn_spectra_info = JSON.parsefile(joinpath("noisey_sn_spectra_dir","metadata", "noisey_sn_spectra_data.json"), dicttype = OrderedDict) 

pure_hg_spectra_info = JSON.parsefile(joinpath("pure_hg_spectra_dir", "metadata","pure_hg_spectra_data.json"), dicttype = OrderedDict)
pure_hg_spectra_file_names = open(joinpath("pure_hg_spectra_dir", "metadata","pure_hg_spectra_names.txt")) do file
    readlines(file)
end

noisey_hg_spectra_info = JSON.parsefile(joinpath("noisey_hg_spectra_dir", "metadata","noisey_hg_spectra_data.json"), dicttype = OrderedDict)

convolved_spectra_info = JSON.parsefile(joinpath("convolved_spectra_dir", "metadata","convolved_spectra_data.json"), dicttype = OrderedDict)
convolved_spectra_file_names = open(joinpath("convolved_spectra_dir", "metadata","convolved_spectra_names.txt")) do file
    readlines(file)
end


#-------------------------------------------------- RUN THE CLASSIFIER --------------------------------------------------
show_plots = true

#Delete the plots if they exist
if show_plots == true
    path = joinpath("classifications_dir", "plots" )
    #delete_folder_contents(path)
end

ENV["JULIA_NUM_THREADS"] = "auto"
n_threads = Threads.nthreads()

println("----------------------------------------------")
println("Loading in Templates...")
println("----------------------------------------------")

pure_sn_spectra_files = read_template_files(pure_sn_spectra_file_names, pure_sn_spectra_info, "Supernova Template")
pure_hg_spectra_files = read_template_files(pure_hg_spectra_file_names, pure_hg_spectra_info, "Hg Template")

#read in target spectra
target_spectra_dict = read_target_spectra("real_spectra_dir")
target_spectra_names = collect(keys(target_spectra_dict))

println("----------------------------------------------")
println("Classifying ", length(target_spectra_names), " spectra with ", n_threads, " threads...")
println("----------------------------------------------")

for n in (target_spectra_names)
    target_wl, target_spectra = target_spectra_dict[n]
    target_name = splitext(n)[1]

    println("Classifying ", target_name, "...")
    name_of_best_sn_match, name_of_best_hg_match, best_params, chi_red = chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, target_spectra, target_name, show_plots)

    println("SN Name           : ", target_name)
    println("SN fit params     : ", round(best_params[1], digits = 4))
    println("HG fit params     : ", round(best_params[2], digits = 4))
    println("χ^2 Reduced       : ", round(chi_red, digits = 4))
    println("----------------------------------------------")

end

println("----------------------------------------------")
println("Classifications Finished")
println("----------------------------------------------")



