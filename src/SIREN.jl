module SIREN
#---------------------------------------------- IMPORT PACKAGES --------------------------------------------------------
include("SIREN_functions.jl")
using .ClassifierFunctions
using Pkg
using DataStructures
using Random
using DataFrames
using DelimitedFiles
using Statistics    
using JSON
using BenchmarkTools
using ProgressBars
using LsqFit
using LaTeXStrings
using CSV
using DataStructures: OrderedDict

#The function used to call the main function and is necessary for PackageCompiler
function julia_main()::Cint
    try
        SIREN_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end
#Also see about doing host extinction in a different way, multiplying different combinations of the host and sn combination
    
function SIREN_main()
    #Get the command line arguments or the variables from the REPL if running in Julia
    if length(ARGS) > 0
        # Running in command-line mode
        parsed_args = parse_command_line_args()
        file_path = parsed_args["file_path"]
        target_redshift = parsed_args["redshift"]
        galactic_source = parsed_args["galactic_source"]
        show_plots = parsed_args["make_plot"]
        target_remove_telluric = parsed_args["remove_telluric"]
        batch_mode = parsed_args["batch_mode"]
        minimum_overlap = parsed_args["minimum_overlap"]
    else
        # Running in REPL mode, add your inputs to the below:
        file_path = "/Users/edcharleton/PhD/SIREN/spectra_to_classify/TiDES Spectra/TiDES-SN26aip.ascii"
        target_redshift = 0.08
        target_remove_telluric = "N"
        minimum_overlap = 75
        galactic_source = "N"
        show_plots = true
        batch_mode = false
    end

    #Set up threading for julia
    ENV["JULIA_NUM_THREADS"] = "auto"
    n_threads = Threads.nthreads()

    #---------------------------------------------- LOAD IN TEMPLATE METADATA ----------------------------------------------
    sn_template_metadata = JSON.parsefile(joinpath("transient_template_dir", "metadata","pure_sn_spectra_data.json"), dicttype = OrderedDict)
    sn_template_file_names = open(joinpath("transient_template_dir", "metadata","pure_sn_spectra_names.txt")) do file
        readlines(file)
    end

    hg_template_metadata = JSON.parsefile(joinpath("host_template_dir", "metadata","pure_hg_spectra_data.json"), dicttype = OrderedDict)
    hg_template_file_names = open(joinpath("host_template_dir", "metadata","pure_hg_spectra_names.txt")) do file
        readlines(file)
    end

    #----------------------------------------- READ IN TARGET AND TEMPLATE SPECTRA -----------------------------------------

    println("----------------------------------------------")
    println("Loading in Templates...")
    println("----------------------------------------------")

    #If in batch mode get the metadata of all the files
    if batch_mode == true
        sn_metadata_path = joinpath("$(file_path)/metadata.csv")
        sn_metadata = CSV.read(sn_metadata_path, DataFrame, delim=',')
        all_target_names = sn_metadata[!, "File Name"]
    end

    #Import the telluric mask where there is atmospheric absorption
    telluric_mask = readdlm(joinpath("src", "metadata", "tellurics.txt"))

    #Read in the templates of both host and supernova/transient wavlengths and spectra
    sn_template_wavs, sn_template_spectra = read_template_files(sn_template_file_names, sn_template_metadata, "Supernova Template")
    hg_template_wavs, hg_template_spectra, hg_template_mask = read_template_files(hg_template_file_names, hg_template_metadata, "Hg Template")

    #read in target spectra
    if batch_mode == true
        target_spectra_dict = read_target_spectra("$(file_path)/spectra", ',', 0)
    else
        target_spectra_dict = read_target_spectra("$file_path", ',', 0)
    end

    #Get the names of the target spectra
    target_spectra_names = collect(keys(target_spectra_dict))

    #Initialise an empty dictionary to store the classifications
    classifications = OrderedDict()

    #-------------------------------------------------- RUN THE CLASSIFIER --------------------------------------------------
    println("----------------------------------------------")
    println("Classifying ", length(target_spectra_names), " spectra with ", n_threads, " threads...")
    println("----------------------------------------------")

    #Loop through all input spectra, if in batch mode, display the progress bar
    for n in (batch_mode ? ProgressBar(target_spectra_names) : (target_spectra_names))
        target_wl, target_spectra, target_spectra_σ = target_spectra_dict[n]
        target_name = splitext(n)[1]

        if batch_mode == true
            target_index = findfirst(x -> x == target_name, all_target_names)
            if isnothing(target_index)
                target_name = n
                target_index = findfirst(x -> x == target_name, all_target_names)
            end
            target_redshift = sn_metadata[target_index, "Redshift"]
            target_remove_telluric = sn_metadata[target_index, "REMOVE_TELLURIC"]
        end

        if typeof(target_redshift) == Float64
            if target_redshift > 1.0
                println("Target ", target_name, " has a redshift of ", target_redshift, " which is too high for SIREN")
                continue
            end
        end 

        #print all the unique types in the template metadata to the console
        unique_types = unique(values(sn_template_metadata["Type"]))
        println(unique_types)

        if batch_mode == false
            println("Classifying ", target_name, "...")
        end


        best_params, chi_red, path, z, type = chi_squared_classifier(
            sn_template_metadata, sn_template_file_names, sn_template_spectra, hg_template_metadata, hg_template_file_names, hg_template_spectra, hg_template_mask,
            sn_template_wavs, hg_template_wavs, target_spectra, target_wl, target_spectra_σ, target_name, target_redshift, 
            telluric_mask, batch_mode, show_plots, target_remove_telluric, minimum_overlap)

        #Output the results of the best matching classification to the console as well as the location of the plot
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
    JSON.open(joinpath("$(file_path)_classifications.json"), "w") do file
        JSON.print(file, classifications)
    end

    println("Classifications Finished")
    println("----------------------------------------------")

end
end

#Call the main function to check its working - does not work in command line mode, so delete if turning into an exe file
SIREN.julia_main()