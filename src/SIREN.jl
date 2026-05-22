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
using BenchmarkTools
using ProgressBars
using LsqFit
using LaTeXStrings
using CSV
using DataStructures: OrderedDict
using JLD2


# ── julia_main ────────────────────────────────────────────────────────────────
# Entry point for the compiled binary. PackageCompiler requires this signature.
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
    # ── Parse inputs ─────────────────────────────────────────────────────────
    if length(ARGS) > 0
        parsed_args            = parse_command_line_args()
        file_path              = parsed_args["file"]
        raw_z                  = parsed_args["redshift"]
        target_redshift        = (raw_z == "None" || raw_z == "none") ? "None" : parse(Float64, raw_z)
        galactic_source        = parsed_args["galactic_source"]
        show_plots             = parsed_args["show_plot"]
        target_remove_telluric = parsed_args["remove_telluric"]
        batch_mode             = parsed_args["batch"]
        minimum_overlap        = parsed_args["minimum_overlap"]
        use_gpu_flag           = parsed_args["use_gpu"]
        savepath               = parsed_args["savepath"]
    else 
        # REPL / development defaults — edit these for interactive runs
        file_path              = "/Users/edcharleton/PhD/SIREN/spectra"
        target_redshift        = "None"
        target_remove_telluric = "N"
        minimum_overlap        = 80
        galactic_source        = "N"
        show_plots             = true
        batch_mode             = isdir(file_path)   # auto-detect from path
        use_gpu_flag           = true
    end

    ClassifierFunctions.use_gpu = use_gpu_flag
        
    # ── Suppress GR display window in batch mode or when plots are off ────────
    # GKSwstype = "nul" routes GR to a null workstation — no window is created
    # or buffered, so nothing flashes up at exit even if plot objects are built.
    if batch_mode || !show_plots
        ENV["GKSwstype"] = "nul"
    end

    # ── Template metadata ─────────────────────────────────────────────────────
    sn_template_metadata   = JSON.parsefile(
        joinpath("templates", "transient_template_dir", "metadata", "pure_sn_spectra_data.json"),
        dicttype = OrderedDict)
    sn_template_file_names = readlines(
        joinpath("templates", "transient_template_dir", "metadata", "pure_sn_spectra_names.txt"))

    hg_template_metadata   = JSON.parsefile(
        joinpath("templates", "host_template_dir", "metadata", "pure_hg_spectra_data.json"),
        dicttype = OrderedDict)
    hg_template_file_names = readlines(
        joinpath("templates", "host_template_dir", "metadata", "pure_hg_spectra_names.txt"))

    println("----------------------------------------------")
    printstyled("Loading in Templates...\n", color=:green)
    println("----------------------------------------------")

    # ── Pre-built template matrices ───────────────────────────────────────────
    prebuilt_path = joinpath("templates/prebuilt_templates.jld2")
    if isfile(prebuilt_path)
        println("Loading pre-built template matrices from $prebuilt_path ...")
        prebuilt    = load(prebuilt_path)
        master_grid = prebuilt["master_grid"]
        T_master    = ClassifierFunctions.to_gpu(prebuilt["T_master"])
        MT_master   = ClassifierFunctions.to_gpu(prebuilt["MT_master"])
        H_master    = ClassifierFunctions.to_gpu(prebuilt["H_master"])
        MH_master   = ClassifierFunctions.to_gpu(prebuilt["MH_master"])

        sn_template_wavs    = Vector{Vector{Float64}}()
        sn_template_spectra = Vector{Vector{Float64}}()
        hg_template_wavs    = Vector{Vector{Float64}}()
        hg_template_spectra = Vector{Vector{Float64}}()
    else
        println("No pre-built templates found — using legacy loading.")
        master_grid = MT_master = T_master = H_master = MH_master = nothing
        sn_template_wavs,    sn_template_spectra = read_template_files(
            sn_template_file_names, sn_template_metadata, "Supernova Template")
        hg_template_wavs,    hg_template_spectra, _ = read_template_files(
            hg_template_file_names, hg_template_metadata, "Hg Template")
    end

    # ── Batch metadata (optional CSV with per-file redshift / telluric flag) ──
    # Expected columns: "File Name", "Redshift", "REMOVE_TELLURIC"
    # If the file doesn't exist, global CLI/REPL values are used for every spectrum.
    batch_metadata     = nothing
    batch_meta_names   = nothing

    if batch_mode 
        println("----------------------------------------------")
        printstyled("Loading in Targets..\n", color=:green)
    else
        println("----------------------------------------------")
        printstyled("Loading in Target..\n", color=:green)
    end

    if batch_mode
        meta_path = joinpath(file_path, "metadata.csv")
        if isfile(meta_path)
            batch_metadata   = CSV.read(meta_path, DataFrame, delim = ',')
            batch_meta_names = batch_metadata[!, "File Name"]
            println("Batch metadata loaded from $meta_path")
        else
            printstyled("No metadata.csv found", color = :red)
            println(" at $meta_path — batch mode fitting with redshift as free parameter")
        end
    end

    # ── Telluric mask ─────────────────────────────────────────────────────────
    telluric_mask = readdlm(joinpath("metadata", "tellurics.txt"))

    # ── Read target spectra ───────────────────────────────────────────────────
    # In batch mode, scan the directory for all valid spectral files.
    # In single-file mode, read the one file given.
    target_spectra_dict = read_target_spectra(batch_mode ? file_path : file_path,',', 0;batch_mode = batch_mode)

    if isempty(target_spectra_dict)
        println("No target spectra found. Exiting.")
        return
    end

    target_spectra_names = collect(keys(target_spectra_dict))
    println("----------------------------------------------")
    if batch_mode
        printstyled("Classifying Targets...\n", color=:green)
    else 
        printstyled("Classifying Target...\n", color=:green)
    end

    # ── Classify ──────────────────────────────────────────────────────────────
    classifications = OrderedDict()
    class_lock      = ReentrantLock()

    progress = ProgressBar(target_spectra_names)
    Threads.@threads for n in target_spectra_names

        batch_mode && update(progress)

        target_wl, target_spectra, target_spectra_σ = target_spectra_dict[n]

        # Display name: strip path and extension
        target_name = splitext(basename(n))[1]

        # Per-file redshift and telluric from metadata CSV, or fall back to global
        local_z         = target_redshift
        local_telluric  = target_remove_telluric

        if batch_mode && !isnothing(batch_metadata)
            # Match on filename (with or without extension)
            idx = findfirst(x -> splitext(x)[1] == target_name || x == target_name,
                            batch_meta_names)
            if !isnothing(idx)
                local_z        = batch_metadata[idx, "Redshift"]
                local_telluric = batch_metadata[idx, "REMOVE_TELLURIC"]
            else
                printstyled("Warning: ", color=:yellow)
                println("No redshift flag indicated in parsed arguments, using redshift = None")
            end
        end

        # Skip implausible redshifts
        if typeof(local_z) == Float64 && local_z > 1.0
            println("Skipping $target_name: redshift $local_z > 1.0")
            continue
        end

        batch_mode || println("Classifying $target_name...")

        best_params, chi_red, path, z, type = chi_squared_classifier(
            sn_template_metadata,   sn_template_file_names, sn_template_spectra,
            hg_template_metadata,   hg_template_file_names, hg_template_spectra,
            sn_template_wavs,       hg_template_wavs,
            target_spectra,         target_wl,              target_spectra_σ,
            target_name,            local_z,
            telluric_mask,          batch_mode,             show_plots,
            local_telluric,         minimum_overlap;
            master_grid = master_grid,
            T_master    = T_master,
            MT_master   = MT_master,
            H_master    = H_master,
            MH_master   = MH_master, savepath = savepath)

        lock(class_lock) do
            classifications[target_name] = (z, type, minimum(chi_red))
        end

        if !batch_mode
            println("SN Name  : $target_name")
            println("Redshift : $(round(z, digits=2))")
            println("Type     : $type")
            println("Plot     : $path")
            println("----------------------------------------------")
        end
    end

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = batch_mode ? joinpath(file_path, "classifications.json") :
                            "$(splitext(file_path)[1])_classifications.json"

    JSON.open(out_json, "w") do f
        JSON.print(f, classifications)
    end

    printstyled("Classifications saved to:", color=:green)
    println(" $out_json")
    println("----------------------------------------------")

    # ── Keep plot window open in single-file interactive mode ─────────────────
    if show_plots
        println("Press Enter to exit.")
        readline()
    end
end
end  # module SIREN

#SIREN.julia_main()