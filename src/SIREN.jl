module SIREN

# ── Imports ──────────────────────────────────────────────────────────────────
include("SIREN_functions.jl")
using .ClassifierFunctions
using Pkg
using DataStructures
using Random
using DataFrames
using DelimitedFiles
using Statistics
using JSON
using ProgressBars
using LsqFit
using LaTeXStrings
using CSV
using DataStructures: OrderedDict
using JLD2
using Plots


# ── julia_main ────────────────────────────────────────────────────────────────
# Entry point for the compiled binary. PackageCompiler (the binary compiler) requires this signature.
function julia_main()::Cint
    try
        SIREN_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end


# ── SIREN_main ────────────────────────────────────────────────────────────────
function SIREN_main()

    #Parse in inputs from the commandline
    parsed_args            = parse_command_line_args()
    file_path              = parsed_args["file_path"] #path to the target spectra or directory of spectra
    raw_z                  = parsed_args["redshift"] #redshift if known 
    local_z_min            = parsed_args["z_min"] #minimum redshift for grid search
    local_z_max            = parsed_args["z_max"] #maximum redshift for grid search
    local_z_step           = parsed_args["z_step"] #the step size
    target_redshift        = (raw_z == "None" || raw_z == "none") ? "None" : parse(Float64, raw_z) #check if there is a redshift provided
    local_type             = parsed_args["type"] #What is the object type (e.g type Ia type II etc)
    galactic_source        = parsed_args["galactic_source"] #Is it a cv (redshift set to 0)
    show_plots             = parsed_args["display_plot"] #use the interactive plotting tool
    target_remove_telluric = parsed_args["remove_telluric"] #Remove the wavelength regions with tellurics
    batch_mode             = parsed_args["batch"] #fit multiple spectra
    local_minimum_overlap  = parsed_args["minimum_overlap"] #minimum overlap between the target and the template
    use_gpu_flag           = parsed_args["use_gpu"] #Flag for whether or not to use a gpu
    savepath               = parsed_args["save_path"] #where to save
    meta_path              = parsed_args["metadata_path"] # where sn metadata is located


    ClassifierFunctions.use_gpu = use_gpu_flag
        
    # Supress the plotting window if its in batch mode or the plotting is set to off
    if batch_mode || !show_plots
        ENV["GKSwstype"] = "nul"
    end

    # ── Template metadata ─────────────────────────────────────────────────────
    sn_template_metadata   = JSON.parsefile(joinpath("templates", "transient_template_dir", "metadata", "pure_sn_spectra_data.json"),dicttype = OrderedDict)
    sn_template_file_names = readlines(joinpath("templates", "transient_template_dir", "metadata", "pure_sn_spectra_names.txt"))

    hg_template_metadata   = JSON.parsefile(joinpath("templates", "host_template_dir", "metadata", "pure_hg_spectra_data.json"), dicttype = OrderedDict)
    hg_template_file_names = readlines(joinpath("templates", "host_template_dir", "metadata", "pure_hg_spectra_names.txt"))

    println("----------------------------------------------")
    printstyled("Loading in Templates...\n", color=:green)
    println("----------------------------------------------")

    # ── Pre-built template matrices ───────────────────────────────────────────
    prebuilt_path = joinpath("templates/prebuilt_templates.jld2")

    println("Loading transient and host template matrices from $prebuilt_path ...")
    prebuilt    = load(prebuilt_path)

    #setup the master grid and the template matrices for SIREN
    master_grid = prebuilt["master_grid"]
    T_master    = ClassifierFunctions.to_gpu(prebuilt["T_master"])
    MT_master   = ClassifierFunctions.to_gpu(prebuilt["MT_master"])
    H_master    = ClassifierFunctions.to_gpu(prebuilt["H_master"])
    MH_master   = ClassifierFunctions.to_gpu(prebuilt["MH_master"])

    # Keep plain CPU copies for the interactive viewer (no GPU transfer needed)
    T_master_cpu  = prebuilt["T_master"]
    MT_master_cpu = prebuilt["MT_master"]
    H_master_cpu  = prebuilt["H_master"]
    MH_master_cpu = prebuilt["MH_master"]

    sn_template_wavs    = Vector{Vector{Float64}}()
    sn_template_spectra = Vector{Vector{Float64}}()
    hg_template_wavs    = Vector{Vector{Float64}}()
    hg_template_spectra = Vector{Vector{Float64}}()

    if batch_mode 
        println("----------------------------------------------")
        printstyled("Loading in Targets..\n", color=:green)
    else
        println("----------------------------------------------")
        printstyled("Loading in Target..\n", color=:green)
    end

    if batch_mode
        if meta_path == "None"
            meta_path = joinpath(file_path, "metadata.csv")
        end
        if isfile(meta_path)
            batch_metadata   = CSV.read(meta_path, DataFrame, delim = ',')
            println("Batch metadata loaded from $meta_path")
        else
            batch_metadata     = nothing
            printstyled("No metadata.csv found", color = :red)
            println(" at $meta_path — batch mode fitting with redshift as free parameter")
        end
    end

    #Load in the telluric mask, doesnt fit the wavelengths specified where a telluric is present
    telluric_mask = readdlm(joinpath("metadata", "tellurics.txt"))

    # In batch mode, scan the directory for all valid spectral files.
    # In single-file mode, read the one file given.
    target_spectra_dict = read_target_spectra(batch_mode ? file_path : file_path,',', 0;batch_mode = batch_mode)

    if isempty(target_spectra_dict)
        println("No target spectra found. Exiting.")
        return
    end

    #For saving the outputs (JSON and plots)
    if batch_mode
        target_names_for_saving = basename(file_path)
    else
        target_names_for_saving = splitext(basename(file_path))[1]
    end

    savepath = joinpath(savepath, target_names_for_saving)

    mkpath(savepath)   # ensure the plot/output directory exists

    #Showtime!
    target_spectra_names = collect(keys(target_spectra_dict))
    println("----------------------------------------------")
    if batch_mode
        printstyled("Classifying Targets...\n", color=:green)
    end

    #Lock for thread-safe writing to the classifications dict
    classifications = OrderedDict()
    class_lock      = ReentrantLock()

    #Load batch metadata once before the classifier loop
    global_params = Dict{String, Any}(
        "redshift"        => target_redshift,
        "z_min"           => local_z_min,
        "z_max"           => local_z_max,
        "z_step"          => local_z_step,
        "type"            => local_type,
        "remove_telluric" => target_remove_telluric,
        "galactic_source" => galactic_source,
        "minimum_overlap" => local_minimum_overlap,
    )

    batch_metadata = batch_mode ? read_batch_metadata(meta_path) : nothing

    ################################
    #         Show Time!           #
    ################################

    progress = ProgressBar(target_spectra_names)
    Threads.@threads for n in target_spectra_names

        target_wl, target_spectra, target_spectra_σ = target_spectra_dict[n]

        #Skips the fit if there are too few datapoints
        if length(target_wl) < 2
            printstyled("Skipping ", color=:yellow)
            println("$target_name: wavelength array has only $(length(target_wl)) point(s).")
            continue
        end

        # Display name, strip path and extension
        target_name = splitext(basename(n))[1]

        if !batch_mode
            printstyled("Classifying ", color=:green)
            println("$(target_name)...")
        end

        #Extract the parameters from the dictionary
        if batch_mode && !isnothing(batch_metadata)
            params       = get_target_params(target_name, batch_metadata, global_params)
            local_z      = params["redshift"] == "None" ? "None" : parse(Float64, params["redshift"])
            local_tell   = params["remove_telluric"]
            local_z_min  = params["z_min"]
            local_z_max  = params["z_max"]
            local_z_step = params["z_step"]
            local_type   = params["type"]
            local_minimum_overlap = params["minimum_overlap"]
        else
            local_z = "None"
            local_tell = "N"
        end

        # Skip implausible redshifts
        if typeof(local_z) == Float64 && local_z > 1.0
            println("Skipping $target_name: redshift $local_z > 1.0")
            continue
        end

        #The SIREN main function
        best_params, chi_red, z, type, top_fits = chi_squared_classifier(
        sn_template_metadata, sn_template_file_names, sn_template_spectra,
        hg_template_metadata, hg_template_file_names, hg_template_spectra,
        sn_template_wavs, hg_template_wavs,
        target_spectra, target_wl, target_spectra_σ,
        target_name, local_z, local_z_min, local_z_max, local_z_step, local_type,
        telluric_mask, batch_mode, show_plots,
        local_tell, local_minimum_overlap;
        master_grid = master_grid,
        T_master    = T_master,
        MT_master   = MT_master,
        H_master    = H_master,
        MH_master   = MH_master)

        # Store the best fitting infomation in the classifications dict 
        lock(class_lock) do
            classifications[target_name] = Dict(
                "redshift"  => z,
                "type"      => type,
                "chi_sq_red" => minimum(chi_red),
                "top_fits"  => top_fits)
        end

        #prints a short summary if not in batch mode
        if !batch_mode
            unique_types = unique(fit["transient_type"] for fit in top_fits)
            average_redshift = median([fit["redshift"] for fit in top_fits])

            #count the number of each unique type
            type_counts = Dict{String, Int}()
            for t in unique_types
                type_counts[t] = sum(fit["transient_type"] == t for fit in top_fits)
            end

            println("----------------------------------------------")
            printstyled("Classification Summary", color=:cyan)
            println(" (out of the top 100 templates):")
            #sort by type count
            type_counts = sort(collect(type_counts), by=x->x[2], rev=true)
            unique_types = [x[1] for x in type_counts]

            for (unique_type, count) in type_counts
                print("  $unique_type: $count,")
            end

            print("\n  Median z: $(round(average_redshift, digits=2)).")
            printstyled("\nTop 5 Fits:\n", color=:cyan)
            for fit in top_fits[1:min(5, length(top_fits))]
                println("  #$(fit["rank"])  $(fit["transient_type"]) + $(fit["host_type"])  z=$(round(fit["redshift"], digits=2)) | Transient Template = $(fit["transient_template"]),  χ²ᵣ=$(round(fit["chi_sq_red"], digits=3))")
            end
            println("----------------------------------------------")
        end
        batch_mode && update(progress)
    end
    
    if batch_mode
        best_types = Dict{String, String}()
        for target_name in target_spectra_names
            try
                fit = classifications[splitext(basename(target_name))[1]]
                type = fit["type"]
                best_types[target_name] = type
            catch
                continue
            end
        end
        unique_types = unique(values(best_types))
        println("----------------------------------------------")
        printstyled("Classification Summary", color=:cyan)
        println(":")
        for unique_type in unique_types
            count = sum(value == unique_type for value in values(best_types))
            println("  $unique_type: $count")
        end
        println("----------------------------------------------")
    end

    out_json = joinpath(savepath, "$(target_names_for_saving)_classifications.json")

    JSON.open(out_json, "w") do f
        JSON.print(f, classifications)
    end
    
    printstyled("Classifications saved to:", color=:green)
    println(" $out_json")
    println("----------------------------------------------")
    make_plots = true

    
    #Interactive fit viewer (single target mode only)
    if show_plots && !batch_mode && !isempty(target_spectra_names)
        n = first(target_spectra_names)
        target_name_viewer = splitext(basename(n))[1]
        wl_v, flux_v, flux_σ_v = target_spectra_dict[n]

        # Retrieve the top_fits for this spectrum from the classifications dict
        fits_for_viewer = get(get(classifications, target_name_viewer, Dict()), "top_fits", [])

        if !isempty(fits_for_viewer) && !isnothing(T_master_cpu)
            interactive_viewer(
                fits_for_viewer, target_name_viewer,
                flux_v, flux_σ_v, wl_v,
                sn_template_file_names, hg_template_file_names,
                T_master_cpu, MT_master_cpu, H_master_cpu, MH_master_cpu,
                master_grid, savepath)
        else
            println("Press Enter to exit.")
            readline()
        end
    
    elseif batch_mode && !isempty(target_spectra_names) && make_plots == true
        println("Generating plots for top fits...")
        #Plot and save the top fits for each target
        for target_name in target_spectra_names
            target_name_viewer = splitext(basename(target_name))[1]
            wl_v, flux_v, flux_σ_v = target_spectra_dict[target_name]

            fits_for_viewer = get(get(classifications, target_name_viewer, Dict()), "top_fits", [])

            if !isempty(fits_for_viewer) && !isnothing(T_master_cpu)
                p = build_fit_plot(
                    fits_for_viewer[1], target_name_viewer,
                    flux_v, flux_σ_v, wl_v,
                    sn_template_file_names, hg_template_file_names,
                    T_master_cpu, MT_master_cpu, H_master_cpu, MH_master_cpu,
                    master_grid; Rv = 3.1)

                out_plot = joinpath(savepath, "$(target_name_viewer)_top_fit.png")
                savefig(out_plot)
            else
                println("Could not reconstruct top fit for $target_name_viewer — skipping plot.")
            end
        end
    printstyled("Plots saved to:", color=:green)
    println(" $savepath")
    println("----------------------------------------------")
    end
end

end
SIREN.julia_main() #Needs commenting out when running through the package compiler and uncommented when run out of the Julia REPL 