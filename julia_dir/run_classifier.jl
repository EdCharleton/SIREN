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
number_of_iterations = 1
success_rate_count = 0
show_plots = true

#Delete the plots if they exist
if show_plots == true
    path = joinpath("classifications_dir", "plots" )
    #delete_folder_contents(path)
end
ENV["JULIA_NUM_THREADS"] = "auto"
n_threads = Threads.nthreads()



println("----------------------------------------------")
println("Running the chi squared classifier ", number_of_iterations, " times with ", n_threads, " threads...")
println("----------------------------------------------")

for n in ProgressBar(1:number_of_iterations)

    # Randomly choose a spectra
    flattened_type = collect(Iterators.flatten(values(pure_sn_spectra_info["Type"])))
    number_of_types = length(unique(flattened_type))

    iterations_per_type = div(length(pure_sn_spectra_info["Type"]), number_of_types)

    all_sne = collect(1:1:(number_of_types*iterations_per_type))
    random_index = rand((all_sne))

    convolved_spectra_key = convolved_spectra_file_names[random_index]
    convolved_spectra_path = convolved_spectra_info["path"][convolved_spectra_key]
    convolved_spectra_hg_contamination = convolved_spectra_info["host galaxy contamination"][convolved_spectra_key]
    file_path = joinpath(convolved_spectra_path, string(convolved_spectra_key))
    convolved_spectra = readdlm(file_path)

    convolved_spectra = readdlm("real_spectra_dir/2011fe_host_galaxy_high_contam.dat")[:,2]


    name_of_best_sn_match, name_of_best_hg_match, best_params, success, chi_red = chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_info, pure_hg_spectra_file_names, convolved_spectra, random_index, n, show_plots)

    #Check if a success by comparison with the random index
    comparison_name = name_of_best_sn_match[1:end-4] * "_" * name_of_best_hg_match

    if success == true
        #println("Correct SN Found")
        global success_rate_count += 1
    else
        #println("Incorrect SN Found")
    end

    println("Fit Number        : ", n)
    println("best fit          : ", comparison_name)
    #println("redshift prior fit: ", redshift_prior_comparison)
    #println("real spectra      : ", convolved_spectra_key)
    #println("HG Contamination  : ", convolved_spectra_hg_contamination)
    #println("p0 sn             : ", p0_sn)
    #println("p0 hg             : ", p0_hg)
    println("SN fit params     : ", best_params[1])
    println("HG fit params     : ", best_params[2])
    println("Chi squared       : ", chi_red)
    #println("random index      : ", random_index)
    println("----------------------------------------------")

end


println(success_rate_count, " out of ", number_of_iterations, " were correctly identified.")



