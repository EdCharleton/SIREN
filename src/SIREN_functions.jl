#------------------------------------------------------------
#Import Packages
#------------------------------------------------------------
#Define the code as a module
module ClassifierFunctions
    using ArgParse
    using Random
    using DataFrames
    using DelimitedFiles
    using Statistics    
    using Optim
    using JSON
    using ProgressBars
    using Pkg
    using DustExtinction    
    using DataStructures
    using Pkg
    using Interpolations: Linear, Gridded, interpolate
    using LsqFit
    using LaTeXStrings
    using Pkg
    using CSV
    using FITSIO
    using SavitzkyGolay
    using Plots
    using LinearAlgebra

    #------------------------------------------------------------
    # Hardware-Accelerated Math Setup
    #------------------------------------------------------------
    # We import at top-level (required by Julia) but check functionality in __init__
    try; import Metal; catch; end

    is_metal_loaded = false
    use_gpu = false

    function __init__()
        global is_metal_loaded
        


        try
            is_metal_loaded = Metal.functional()
        catch
        end

        #println("--- Hardware Setup ---")
        #println("Metal functional: ", is_metal_loaded)
        #println("Processing on: ", (is_metal_loaded ? "Apple Metal GPU" : "Native Multi-threaded CPU"))
        #println("----------------------")
    end

    function to_gpu(x)
        # If it's already on GPU, return as is
        if (is_metal_loaded && x isa Metal.MtlArray)
            return x
        end

        if use_gpu && is_metal_loaded
            # Metal requires Float32 for numerical performance, but masks must remain Bool.
            if eltype(x) == Bool
                return Metal.MtlArray(x)
            else
                return Metal.MtlArray(Float32.(x))
            end
        else
            return Array(x)
        end
    end

    # Return array to CPU safely
    function to_cpu(x)
        return Array(x)
    end
    #------------------------------------------------------------
    #Define Functions
    #------------------------------------------------------------

    #This section defines the functions used in the main script

    function parse_command_line_args()
        #Get the arguments when running in command line mode
        s = ArgParseSettings()
        @add_arg_table! s begin
            "--file_path", "-f"
                help = "Path to the input data file/directory (see --batch flag)"
                arg_type = String
                required = true
            
            "--redshift", "-z"
                help = "Redshift value (or 'None' for search)"
                arg_type = String
                default = "None"
            "--z_min"
                help = "Minimum redshift SIREN fits"
                arg_type = Float64
                default = 0.0
            
            "--z_max"
                help = "Maximum redshift SIREN fits"
                arg_type = Float64
                default = 0.8

            "--z_step"
                help = "Redshift step size for SIREN fits"
                arg_type = Float64
                default = 0.01
    
            "--batch", "-b"
                help = "Flag to run in batch mode (multiple at once)"
                arg_type = Bool
                default = false

            "--metadata_path"
                help = "Path to the metadata CSV file containing prior info for the spectra to fit such as redshifts"
                arg_type = String
                default = "None"

            "--type", "-t"
                help = "The type of transient to force only those fits"
                arg_type = String
                default = "None"  # Default is 'no'
            
            "--remove_telluric"
                help = "Flag to remove telluric absorption at telluric wavelengths"
                arg_type = String
                default = "N"  # Default is 'no'
    
            "--galactic_source", "-g"
                help = "Flag indicating whether the source is galactic"
                arg_type = String
                default = "N"  # Default is 'no'
    
            "--display_plot", "-d"
                help = "Flag to show the plot"
                arg_type = Bool
                default = true  # Default is true
    
            "--minimum_overlap", "-m"
                help = "Minimum overlap between the target and templates, units of %"
                arg_type = Float64
                default = 90 

            "--save_path", "-s"
                help = "Directory to save output plots and JSON file"
                arg_type = String
                default = "classifications"

            "--use_gpu"
                help = "Flag to enable/disable GPU acceleration"
                arg_type = Bool
                default = false

        end
    
        return parse_args(ARGS, s)
    end

    # ── Column name aliases ───────────────────────────────────────────────────────
    # Maps canonical parameter names to a list of accepted column name variants.
    # Case-insensitive matching is applied on top of these, so "Redshift" and
    # "REDSHIFT" both resolve without needing separate entries here.
    const COLUMN_ALIASES = Dict{String, Vector{String}}(
        "file_path"       => ["file", "file name", "file path", "path"],
        "redshift"        => ["redshift", "z", "red_shift", "z_spec", "zspec", "z_phot", "redshift_value"],
        "z_min"           => ["z_min", "zmin", "redshift_min", "min_z", "minimum_z"],
        "z_max"           => ["z_max", "zmax", "redshift_max", "max_z", "maximum_z"],
        "z_step"          => ["z_step", "zstep", "redshift_step", "step_z", "dz"],
        "remove_telluric" => ["remove_telluric", "telluric", "remove_tell", "tell", "telluric_removal"],
        "galactic_source" => ["galactic_source", "galactic", "gal_source", "is_galactic"],
        "minimum_overlap" => ["minimum_overlap", "min_overlap", "overlap", "min_overlap_pct"],
        "type" => ["type", "sn_type"],
    )

    # ── Default values (must match parse_command_line_args defaults) ──────────────
    const BATCH_DEFAULTS = Dict{String, Any}(
        "redshift"        => "None",
        "z_min"           => 0.0,
        "z_max"           => 0.8,
        "z_step"          => 0.01,
        "type"            => "None",
        "remove_telluric" => "N",
        "galactic_source" => "N",
        "minimum_overlap" => 90,
    )


    # ── resolve_column ─────────────────────────────────────────────────────────────
    # Returns the actual DataFrame column name that matches the canonical key,
    # or nothing if no match is found.
    function resolve_column(df_cols::Vector{String}, canonical::String)::Union{String, Nothing}
        aliases = get(COLUMN_ALIASES, canonical, [canonical])
        for alias in aliases
            for col in df_cols
                lowercase(strip(col)) == lowercase(strip(alias)) && return col
            end
        end
        return nothing
    end


    # ── read_batch_metadata ────────────────────────────────────────────────────────
    # Reads the metadata CSV and returns a Dict mapping target name → per-row params.
    # Missing columns fall back to BATCH_DEFAULTS. Empty cells also fall back.
    # Logs one warning per missing column at load time (not per-row).
    function read_batch_metadata(metadata_path::String)

        if !isfile(metadata_path)
            printstyled("Warning: ", color=:yellow)
            println("Metadata file not found at '$metadata_path' — using global defaults for all spectra.")
            return nothing
        end

        df = try
            CSV.read(metadata_path, DataFrame)
        catch e
            printstyled("Warning: ", color=:yellow)
            println("Could not read metadata CSV '$metadata_path': $e")
            return nothing
        end

        df_cols = string.(names(df))

        # ── Resolve column mapping once at load time ──────────────────────────────
        col_map = Dict{String, Union{String, Nothing}}()
        for canonical in keys(BATCH_DEFAULTS)
            resolved = resolve_column(df_cols, canonical)
            col_map[canonical] = resolved
            if isnothing(resolved)
                printstyled("  Note: ", color=:cyan)
                println("Metadata column '$canonical' not found — will use default ($(BATCH_DEFAULTS[canonical])) for all rows.")
            end
        end

        # ── File name column (required) ───────────────────────────────────────────
        name_col = resolve_column(df_cols, "file_path")
        if isnothing(name_col)
            # Try a broader search for anything that looks like a name column
            name_col = resolve_column(df_cols, "name")
            isnothing(name_col) && (name_col = resolve_column(df_cols, "filename"))
            isnothing(name_col) && (name_col = resolve_column(df_cols, "sn_name"))
            isnothing(name_col) && (name_col = resolve_column(df_cols, "object"))
            isnothing(name_col) && (name_col = resolve_column(df_cols, "id"))
        end

        if isnothing(name_col)
            printstyled("Warning: ", color=:yellow)
            println("No filename/name column found in metadata CSV. Expected one of: file, name, filename, sn_name, object, id.")
            return nothing
        end

        # ── Build lookup dict: stripped_name → param dict ─────────────────────────
        result = Dict{String, Dict{String, Any}}()
        for row in eachrow(df)
            raw_name = string(row[name_col])
            isempty(raw_name) || raw_name == "missing" && continue
            key = splitext(strip(raw_name))[1]   # strip extension so "SN2024a.fits" → "SN2024a"

            params = Dict{String, Any}()

            for (canonical, col) in col_map
                default_val = BATCH_DEFAULTS[canonical]

                if isnothing(col)
                    params[canonical] = default_val
                    continue
                end

                raw = row[col]

                # Treat missing, empty string, "none", "nan" as "use default"
                is_empty = ismissing(raw) ||
                        (raw isa AbstractString && lowercase(strip(string(raw))) in ("", "none", "nan", "n/a", "na"))

                if is_empty
                    params[canonical] = default_val
                    continue
                end

                # ── Type-coerce to match expected type ────────────────────────────
                params[canonical] = try
                    if default_val isa Float64
                        parse(Float64, string(raw))
                    elseif default_val isa Bool
                        v = lowercase(strip(string(raw)))
                        v in ("true", "yes", "y", "1")
                    else
                        # String params (redshift = "None" or a number, telluric = "Y"/"N")
                        str = strip(string(raw))
                        if canonical == "redshift"
                            lowercase(str) in ("none", "n/a", "na") ? "None" : string(parse(Float64, str))
                        else
                            str
                        end
                    end
                catch
                    # z_min and z_max are irrelevant when a specific redshift is
                    # already set for this row — suppress the warning in that case.
                    if canonical in ("z_min", "z_max", "z_step")
                        z_col = col_map["redshift"]
                        has_redshift = !isnothing(z_col) && begin
                            r = row[z_col]
                            !ismissing(r) &&
                            !(r isa AbstractString && lowercase(strip(string(r))) in ("", "none", "nan", "n/a", "na"))
                        end
                        has_redshift && (default_val; continue)
                    end

                    printstyled("  Warning: ", color=:yellow)
                    println("Could not parse '$raw' for column '$canonical' (row: $key) — using default $(default_val).")
                    default_val
                end
            end

            result[key] = params
        end

        println("Loaded metadata for $(length(result)) spectra from '$metadata_path'.")
        return result
    end


# ── get_target_params ──────────────────────────────────────────────────────────
# Returns the metadata dict for a given target name, falling back to global
# CLI/REPL values if the name isn't in the metadata or metadata wasn't loaded.
# global_params should be a Dict with the same canonical keys as BATCH_DEFAULTS.
function get_target_params(
        target_name::String,
        batch_metadata::Union{Dict, Nothing},
        global_params::Dict{String, Any})::Dict{String, Any}

    isnothing(batch_metadata) && return global_params

    # Try exact match first, then strip extension
    key = splitext(basename(target_name))[1]
    row = get(batch_metadata, key, nothing)
    if isnothing(row)
        row = get(batch_metadata, target_name, nothing)
    end

    if isnothing(row)
        printstyled("  Note: ", color=:cyan)
        println("'$target_name' not found in metadata — using global defaults.")
        return global_params
    end

    # Merge: row values take priority over global_params, global_params over BATCH_DEFAULTS
    merged = copy(global_params)
    for (k, v) in row
        merged[k] = v
    end
    return merged
end

    
    function find_closest_value(array, value)
        idx = argmin(abs.(array .- value))
        return array[idx]
    end

    function string_import(file_path::String, delimiter::Char, skip_header::Int)
        file_data = readdlm(file_path, delimiter, skipstart = skip_header)
    
        content = zeros(Float64, length(file_data), 3)
    
        failed = false
        for i in eachindex(file_data)
            row = file_data[i]
            vals = split(row)

            try
                content[i, 1] = parse(Float64, vals[1])
                content[i, 2] = parse(Float64, vals[2])
            catch
                failed = true
                break
            end
    
            if length(vals) == 2
                content[i, 3] = NaN
            else
                if vals[3] == "---" || vals[3] == "NaN" || vals[3] == ""
                    content[i, 3] = NaN
                else
                    content[i, 3] = parse(Float64, vals[3])
                end
            end
        end

        if failed == true
            imported = false
        else 
            imported = true
        end
    
        return content, imported
    end



    function calculate_spectral_error(spectra, spectra_λ, window_size::Int)
        #Use Savitsky Golay to smooth the spectrum, then calculate the residuals
        smoothed_spectra_sg = savitzky_golay(spectra, window_size, 2)
        smoothed_spectra = smoothed_spectra_sg.y
        residual = abs.(spectra .- smoothed_spectra)
        
        #Set the size of half the window, so counting from the middle point in the box car to the end or start to the end
        start_end_window = Int((window_size / 2) - 0.5)
        
        #Create new arrays to populate
        new_spectra = zeros(Float32, Int(length(spectra) - 2 * start_end_window))
        new_spectra_λ = zeros(Float32, Int(length(spectra_λ) - 2 * start_end_window))
        new_sigma = zeros(Float32, Int(length(residual) - 2 * start_end_window))
        new_smoothed_spectra = zeros(Float32, Int(length(smoothed_spectra) - 2 * start_end_window))
    
        for i in 1:length(new_spectra)
            old_array_index = i + start_end_window
            new_spectra[i] = spectra[old_array_index]
            new_spectra_λ[i] = spectra_λ[old_array_index]
            new_smoothed_spectra[i] = smoothed_spectra[old_array_index]
    
            # Compute Median Absolute Deviation (MAD) with scaling factor
            window_residuals = residual[old_array_index - start_end_window : old_array_index + start_end_window]
            MAD = median(abs.(window_residuals .- median(window_residuals)))
            new_sigma[i] = MAD
        end
    
        return new_smoothed_spectra, new_spectra_λ, new_sigma
    end
    
    
# ── Valid spectral file extensions ────────────────────────────────────────────
const VALID_EXTENSIONS = Set([
    ".dat", ".txt", ".ascii", ".csv", ".list",
    ".flm", ".asci", ".clean", ".asc", ".sed", ".fits"
])

# ── read_target_spectra ───────────────────────────────────────────────────────
# Single-file mode (batch_mode = false, default): reads the file at `directory`.
# Batch mode (batch_mode = true):  if `directory` is a directory path, scans it
# for every file whose extension is in VALID_EXTENSIONS and reads them all.
# Returns Dict{filename => (λ, flux, σ)}.
function read_target_spectra(directory::String, delimiter::Char = ',',
                             skip_header::Int = 0; batch_mode::Bool = false)

    files_dict = Dict{String, Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}}()

    # ── Resolve the list of files to process ─────────────────────────────────
    file_list = if batch_mode && isdir(directory)
        found = [joinpath(directory, f) for f in readdir(directory)
                 if isfile(joinpath(directory, f)) &&
                    splitext(f)[2] in VALID_EXTENSIONS]
        if isempty(found)
            @warn "Batch mode: no supported spectral files found in $directory"
        else
            printstyled("Batch mode:", color=:green)
            println(" found $(length(found)) spectra in $directory")
        end
        found
    else
        isdir(directory) && error("Single-file mode: expected a file path, got a directory. Pass batch_mode=true to scan a directory.")
        [directory]
    end

    isempty(file_list) && return files_dict

    # ── Per-file ingestion ────────────────────────────────────────────────────
    for file_path in file_list
        filename  = basename(file_path)
        ext       = splitext(filename)[2]

        λ = nothing; spectra = nothing; spectra_σ = nothing

        if ext in Set([".dat",".txt",".ascii",".csv",".list",
                       ".flm",".asci",".clean",".asc",".sed"])

            skip     = skip_header
            imported = false
            content  = nothing

            while !imported
                if ext == ".csv"
                    content  = CSV.read(file_path, DataFrame)
                    imported = true
                else
                    content = readdlm(file_path, delimiter, skipstart = skip)
                    try
                        _ = content[:, 2]
                    catch
                        content, imported = string_import(file_path, delimiter, skip)
                    end
                end

                if !imported
                    r1 = content[1, 1]
                    if typeof(r1) <: AbstractString || r1 == 0
                        skip += 1
                    else
                        imported = true
                    end
                end
            end

            λ       = Float32.(content[:, 1])
            spectra = Float32.(content[:, 2])

            spec_min = minimum(spectra)
            spectra .+= spec_min < 0 ? 2f0 * abs(spec_min) : 0f0

            no_errs = false
            try
                spectra_σ = Float32.(content[:, 3])
                no_errs   = isempty(spectra_σ) || any(isnan.(spectra_σ))
            catch
                no_errs = true
            end

            scaler  = maximum(filter(!isnan, spectra)) / 100f0
            spectra = spectra ./ scaler

            if !no_errs
                spectra_σ = spectra_σ ./ scaler
            else
                spectra, λ, spectra_σ = calculate_spectral_error(spectra, λ, 27)
            end

        elseif ext == ".fits"
            f           = FITS(file_path)
            spectra_hdr = read_header(f[1])
            spectra     = Float32.(read(f[1]))

            try
                spectra_σ = Float32.(sqrt.(read(f[2])))
            catch
                spectra_σ = spectra .* 0.1f0
            end

            try
                λ = Float32.(collect(
                    spectra_hdr["CRVAL1"] - spectra_hdr["CRPIX1"] * spectra_hdr["CDELT1"] :
                    spectra_hdr["CDELT1"] :
                    (spectra_hdr["NAXIS1"] - spectra_hdr["CRPIX1"] - 1) * spectra_hdr["CDELT1"] + spectra_hdr["CRVAL1"]
                ))
            catch
                λ = Float32.(collect(
                    spectra_hdr["CRVAL1"] - spectra_hdr["CRPIX1"] * spectra_hdr["CD1_1"] :
                    spectra_hdr["CD1_1"] :
                    (spectra_hdr["NAXIS1"] - spectra_hdr["CRPIX1"] - 1) * spectra_hdr["CD1_1"] + spectra_hdr["CRVAL1"]
                ))
            end

            valid     = .!isnan.(spectra)
            spectra   = spectra[valid]
            spectra_σ = spectra_σ[valid]
            λ         = λ[valid]

            spec_min = minimum(spectra)
            spectra .+= spec_min < 0 ? abs(spec_min) : 0f0

            scaler    = maximum(spectra)
            spectra   = (spectra ./ scaler) .* 100f0
            spectra_σ = (spectra_σ ./ scaler) .* 100f0

        else
            printstyled("ERROR: File type not supported - ", color=:red)
            println(filename)
            continue
        end

        files_dict[file_path] = (λ, spectra, spectra_σ)
    end

    return files_dict
end


    function read_template_files(template_file_names, template_metadata, label)
        println("Loading in $label Spectra...")
        #setup empty matricies to populate with templates
        flux = Vector{Vector{Float64}}(undef, length(template_file_names))
        λ = Vector{Vector{Float64}}(undef, length(template_file_names))
        #masks only present on host galaxy templates (SIREN doesnt fit any host templates with emission lines so they're masked out)
        if label == "Hg Template"
            mask = Vector{Vector{Float64}}(undef, length(template_file_names))
        end
        #Use multithreading to read in the files
        Threads.@threads for i in ProgressBar(eachindex(template_file_names))
            key = template_file_names[i]
            template_path = joinpath(template_metadata["path"][key], string(key))

            content = CSV.read(template_path, DataFrame, delim = ",", header = false)
            #Extract the wavelength and flux of the read in spectra
            λ[i] = content[:, 1]
            flux[i] = content[:, 2]
            if label == "Hg Template W Mask"
                mask[i] = content[:, 3]
            end
        end

        #Scale up the files to avoid dealing with small numbers 
        spec_means = [mean(flux[i]) for i in eachindex(flux)]
        flux = (flux ./ spec_means .* 100)

        #Temporarily set mask as an array filled with 1s
        mask = [ones(length(λ[i])) for i in eachindex(λ)]   

        if label == "Hg Template"
            return λ, flux, mask
        else
            return λ, flux
        end
    end

    function trim_nans(input_spectra, input_spectra_σ, input_spectra_λ)

        number_of_nans = sum(isnan.(input_spectra))

        valid_indices = findall(.!isnan.(input_spectra))
        if number_of_nans == 0 
            return input_spectra, input_spectra_σ, input_spectra_λ
        else
            # Trim arrays
            output_spectra = input_spectra[valid_indices]
            output_spectra_σ = input_spectra_σ[valid_indices]
            output_spectra_λ = input_spectra_λ[valid_indices]

            return output_spectra, output_spectra_σ, output_spectra_λ
        end
    end

    function make_bins(λ)
        edges = zeros(Float64, length(λ) + 1)
        widths = zeros(Float64, length(λ))
        edges[1] = λ[1] - (λ[2] - λ[1]) / 2
        widths[end] = (λ[end] - λ[end - 1])
        edges[end] = λ[end] + (λ[end] - λ[end - 1]) / 2
        edges[2:end-1] = (λ[2:end] .+ λ[1:end-1]) / 2
        widths[1:end-1] = edges[2:end-1] .- edges[1:end-2]
    
        return edges, widths
    end
    
    function spectral_resampler(new_λ, old_λ, old_fluxes, old_errs = nothing, fill = NaN, verbose = true)
        old_edges, old_widths = make_bins(old_λ)
        new_edges, new_widths = make_bins(new_λ)
    
        new_fluxes = zeros(size(old_fluxes)[1:end-1]..., size(new_λ)...)
    
        if old_errs !== nothing
            if size(old_errs) != size(old_fluxes)
                throw(error("If specified, spec_errs must be the same shape as spec_fluxes."))
            else
                new_errs = copy(new_fluxes)
            end
        end
    
        start = 0
        stop = 0
        warned = false
    
        for j in 1:size(new_λ, 1)
            if (new_edges[j] < old_edges[1]) || (new_edges[j+1] > old_edges[end])
                
                new_fluxes[j] = fill
    
                if old_errs !== nothing
                    new_errs[j] = fill
                end
    
                if (j == 1 || j == size(new_λ, 1)) && verbose && !warned
                    warned = true
                    #println("New_λ contains values outside the range in spec_λ, new_fluxes and new_errs will be filled with the value set in the 'fill' keyword argument. \n")
                end
                continue
            end
    
            while old_edges[start+1] <= new_edges[j]
                start += 1
            end
    
            while old_edges[stop+1] < new_edges[j+1]
                stop += 1
            end
            #println(start, " ", stop)
    
            if stop == start
                new_fluxes[j] = old_fluxes[start]
    
                if old_errs !== nothing
                    new_errs[j] = old_errs[start]
                end
            else
                start_factor = (old_edges[start+1] - new_edges[j]) / (old_edges[start+1] - old_edges[start])
                end_factor = (new_edges[j+1] - old_edges[stop]) / (old_edges[stop+1] - old_edges[stop])
    
                old_widths[start] *= start_factor
                old_widths[stop] *= end_factor
    
                f_widths = old_widths[start:stop] .* old_fluxes[start:stop]

                new_fluxes[j] = sum(f_widths) 
                new_fluxes[j] /= sum(old_widths[start:stop])

    
                if old_errs !== nothing
                    e_wid = old_widths[start:stop] .* old_errs[start:stop]
                    new_errs[j] = sqrt(sum(e_wid.^2)) ./ sum(old_widths[start:stop])
                end
    
                old_widths[start] /= start_factor
                old_widths[stop] /= end_factor
            end
        end
    
        if old_errs !== nothing
            return new_fluxes, new_errs
        else
            return new_fluxes
        end
    end

    #------------------- Remove Tellurics -------------------

    function remove_tellurics(telluric_λ, input_spectra_λ, input_spectra, input_spectra_σ)
        original_spectra = deepcopy(input_spectra)
        original_spectra_λ = deepcopy(input_spectra_λ)
        original_spectra_σ = deepcopy(input_spectra_σ)

        mask = trues(length(input_spectra))

        # Mask out the tellurics from the spectrum as well as the adjacent bins to account for any bins that are centered on the wavelength bin, flux conservation tings innit
        for i in 1:2:length(telluric_λ)
            start = telluric_λ[i]
            stop = telluric_λ[i+1]
            region_mask = ((input_spectra_λ .> start) .& (input_spectra_λ .< stop))


            region_indices = findall(region_mask)
            
            for idx in region_indices
                if idx > 1
                    mask[idx-1] = false
                end
                if idx < length(input_spectra)
                    mask[idx+1] = false
                end
            end
            mask .&= .!region_mask
        end

        tellurics = .!mask

        #input_spectra = input_spectra[mask]
        #input_spectra_σ = input_spectra_σ[mask]
        #input_spectra_λ = input_spectra_λ[mask]

        input_spectra[tellurics] .= NaN
        input_spectra_σ[tellurics] .= NaN
        input_spectra_λ[tellurics] .= NaN
    


        telluric_mask = original_spectra_λ[tellurics]
        plotting_telluric = original_spectra[tellurics]

        return input_spectra_λ, input_spectra, input_spectra_σ,  telluric_mask, plotting_telluric
    end
    #------------------- Remove Galaxy Features -------------------
    function remove_emission_lines(input_spectra_λ, input_spectra, input_spectra_σ, window_size, σ)

        original_spectra   = deepcopy(input_spectra)
        original_spectra_λ = deepcopy(input_spectra_λ)
        original_spectra_σ = deepcopy(input_spectra_σ)

        half = window_size ÷ 2
        n = length(input_spectra)
        flagged = fill(false, n)

        for iter in 1:10
            new_flags = false
            for i in 1:n
                if flagged[i]
                    continue
                end

                lo = max(1, i - half)
                hi = min(n, i + half)
                
                # Get window flux ignoring existing flagged items
                window_flux = Float32[]
                for j in lo:hi
                    if !flagged[j]
                        push!(window_flux, input_spectra[j])
                    end
                end
                
                if isempty(window_flux)
                    continue
                end

                win_median = median(window_flux)
                win_mad    = median(abs.(window_flux .- win_median))
                win_noise  = 1.4826 * win_mad

                if win_noise > 0 && input_spectra[i] > win_median + σ * win_noise
                    flagged[i] = true
                    new_flags = true
                end
            end
            !new_flags && break
        end

        missing_indices = findall(flagged)
        
        output_spectra   = deepcopy(input_spectra)
        output_spectra_σ = deepcopy(input_spectra_σ)
        output_spectra_λ = deepcopy(input_spectra_λ)

        # Downweight emission lines strictly via massive error
        output_spectra_σ[missing_indices] .= 1e3

        return input_spectra_λ[missing_indices], input_spectra[missing_indices],
            output_spectra_λ, output_spectra, output_spectra_σ
    end
    #------------------- Pre - Process Spectra -------------------

    function process_spectra(input_spectra_λ, input_spectra, input_spectra_σ, telluric_λ, remove_telluric)
        if remove_telluric == "Y" || remove_telluric == "y" || remove_telluric == "Yes" || remove_telluric == "yes" || remove_telluric == " Y" || remove_telluric == " y" || remove_telluric == " Yes" || remove_telluric == " yes"
            # Remove telluric featues from the spectra
            input_spectra_λ, input_spectra, input_spectra_σ, plotting_telluric_x, plotting_telluric_y = remove_tellurics(
                telluric_λ, 
                input_spectra_λ,
                input_spectra, 
                input_spectra_σ)
        else
            plotting_telluric_x = []
            plotting_telluric_y = []
        end

        # Remove any emission lines from the spectrum
        wavelength_window = 200 #Measured in Angstroms 
        #convert to indexes - find out many indices correspond to the above window
        indexes = findall(x -> x in (input_spectra_λ[1] - wavelength_window):(input_spectra_λ[1] + wavelength_window), input_spectra_λ)
        window_indexes = length(indexes)

        #Sigma clipping threshold for identifying emission lines
        σ_threshold = 3.2
        emission_line_λ, emission_line_flux,  output_spectra_λ, output_spectra, output_spectra_σ = remove_emission_lines(
            input_spectra_λ, 
            input_spectra, 
            input_spectra_σ, 
            window_indexes,
            σ_threshold)

        return output_spectra_λ, output_spectra, output_spectra_σ, plotting_telluric_x, plotting_telluric_y, emission_line_λ, emission_line_flux

    end
    

    function redshift_spectra(input_spectra_λ, input_spectra, input_spectra_σ, input_transient_templates, input_transient_λ, 
        transient_metadata, transient_file_names, input_host_templates, redshift,  input_host_λ, original_spectra_λ)

        # Redshift the target spectra to rest wavelength
        de_redshifted_λ = original_spectra_λ ./ (1 + redshift)

        #full_template_λ_range is the full range of the templates, when deredshifting the spectrum, you can probe the bluer template ends  up to 2315 Å 
        full_template_λ_range = collect(2315:15:9000)
        bottom_λ = 3800 / (1 + redshift)
        bottom_λ = find_closest_value(full_template_λ_range, bottom_λ)
    
        if bottom_λ < 2315
            bottom_λ = 2315
        end
    
        top_λ = 8000 / (1 + redshift)
        top_λ = find_closest_value(full_template_λ_range, top_λ)
        full_template_λ = collect(bottom_λ:15:top_λ)


    
        # Trim and rebin the spectra to fit to the wavelength range of the templates
        output_spectra, output_spectra_σ = spectral_resampler(
            full_template_λ, 
            de_redshifted_λ, 
            input_spectra, 
            input_spectra_σ
        )

        binned_spectra = copy(output_spectra)
        binned_spectra_σ = copy(output_spectra_σ)
        binned_spectra_λ = copy(full_template_λ)



        output_spectra, output_spectra_σ, output_spectra_λ= trim_nans(
            output_spectra, 
            output_spectra_σ, 
            full_template_λ 
        )

        # Trim the templates to the same wavelength range as the target spectra
        common_elements = intersect(output_spectra_λ, full_template_λ_range)
    
        # Find indices in full_template_λ_range
        template_indicies = findall(x -> x in common_elements, full_template_λ_range)
    
        # Find the standard and non-standard wavelength templates, standard templates are ones within a standard range that doesnt change
        standard_template_names = findall(x -> x == "Y", transient_metadata["Standard WL"])
        common_names = intersect(transient_file_names, standard_template_names)
        standard_template_indices = findall(x -> x in common_names, transient_file_names)
        
        non_standard_template_names = findall(x -> x == "N", transient_metadata["Standard WL"])
        common_names = intersect(transient_file_names, non_standard_template_names)
        non_standard_template_indices = findall(x -> x in common_names, transient_file_names)

        # Trim the templates to the same wavelength range as the target spectra
        # Trim standard templates to the common wavelength range.
        # Non-standard templates are left at their original wavelengths —
        # the classifier handles their trimming individually per target spectrum.
        output_transient_templates = Vector{Vector{Float64}}(undef, length(input_transient_templates))
        for i in eachindex(input_transient_templates)
            if i in standard_template_indices
                output_transient_templates[i] = input_transient_templates[i][template_indicies]
            else
                output_transient_templates[i] = input_transient_templates[i]
            end
        end
        
        output_host_templates = [input_host_templates[i][template_indicies] for i in eachindex(input_host_templates)]
    
        output_host_λ = [input_host_λ[i][template_indicies] for i in eachindex(input_host_λ)]

        output_transient_λ = Vector{Vector{Float64}}(undef, length(input_transient_λ))
        for λ in eachindex(input_transient_λ)
            if λ in standard_template_indices
                output_transient_λ[λ] = input_transient_λ[λ][template_indicies]
            else
                output_transient_λ[λ] = input_transient_λ[λ]
            end
        end

        return output_spectra, output_spectra_σ, output_spectra_λ, output_transient_templates, 
        output_host_templates, output_transient_λ, output_host_λ, standard_template_indices, non_standard_template_indices,
        binned_spectra, binned_spectra_σ, binned_spectra_λ
        
    end


    #------------------- Find the best fit parameters -----------------------------
    #These functions calculate the best fit parameters from the B and C parameters described in the SIREN paper
    #As of August 2025, this paper is my first year phd report lol
    function calculate_B(F_obs, F_SN, F_HG, sigma)
        numerator = sum(((F_SN .* F_obs)./sigma.^2) .- ((F_HG .* F_SN ./ sigma.^2).*(sum(F_HG .* F_obs ./ sigma.^2) / sum(F_HG.^2 ./ sigma.^2))))
        denominator = sum((F_SN.^2 ./ sigma.^2)  .- (F_HG.* F_SN ./ sigma.^2).* (sum(F_HG .* F_SN ./ sigma.^2) / sum(F_HG.^2 ./ sigma.^2)))
        B = numerator / denominator
        return B
    end

    function calculate_C(F_obs, F_SN, F_HG, sigma)
        numerator = sum((F_HG .* F_SN ./ sigma.^2) .* (sum(F_SN.*F_obs ./ sigma.^2)/sum(F_SN.^2 ./ sigma.^2)) .- (F_HG .* F_obs ./ sigma.^2))
        denominator = sum((F_HG .* F_SN ./ sigma.^2) .* (sum(F_SN .* F_HG ./ sigma.^2) / sum(F_SN.^2 ./ sigma.^2)) .- (F_HG.^2 ./ sigma.^2))
        C = numerator / denominator
        return C
    end
#------------------- Chi Squared Classifier -------------------
function chi_squared_classifier(
        transient_template_info, transient_template_names, transient_template_files,
        host_template_info,      host_template_names,      host_template_files,
        transient_template_λ,   host_template_λ,
        target_spectra,         target_λ,                  target_spectra_σ,
        SN_ID, target_redshift, z_min, z_max, z_step, input_type, telluric_λ,
        batch_mode, show_plot = false, remove_telluric = "N", minimum_overlap = 50;
        master_grid = nothing, T_master = nothing, MT_master = nothing,
        H_master = nothing, MH_master = nothing,
        Rv = 3.1,
        Av_min = -3.0, Av_max = 3.0, Av_step = 0.2,               # transient (SN) extinction grid
        Av_host_min = -3.0, Av_host_max = 3.0, Av_host_step = 0.2, # host-galaxy extinction grid — coarser by default
        savepath = "classifications",
        N_top = 100)

    use_prebuilt = !isnothing(T_master)

    # ── Upload static master matrices to GPU once ─────────────────────────────
    if use_prebuilt
        T_master_gpu  = to_gpu(T_master)
        MT_master_gpu = to_gpu(MT_master)
        H_master_gpu  = to_gpu(H_master)
        MH_master_gpu = to_gpu(MH_master)
    end

    n_transient_templates = length(transient_template_names)
    n_host_templates      = length(host_template_names)

    unique_types  = unique(values(transient_template_info["Type"]))
    n_types       = length(unique_types)
    type_to_index = Dict(t => i for (i, t) in enumerate(unique_types))

    chi_squared_arr     = fill(1e10, n_types)
    chi_squared_red_arr = fill(1e10, n_types)

    norm_arr      = zeros(Float32, n_types, 2)
    redshift_vals = zeros(Float32, n_types)
    best_indices  = zeros(Int,     n_types, 2)
    best_Av_transient_arr = zeros(Float32, n_types)   # ← was best_Av_arr; now split
    best_Av_host_arr      = zeros(Float32, n_types)   # ← new
    best_z_index  = zeros(Int,     n_types)
    contam_array  = Vector{Float32}(undef, n_types)

    top_fits = Vector{Dict{String,Any}}()
    sizehint!(top_fits, N_top)

    emission_lines_arr   = Vector{Vector{Float64}}(undef, n_types)
    emission_lines_λ_arr = Vector{Vector{Float64}}(undef, n_types)
    binned_spectra_arr   = Vector{Vector{Float64}}(undef, n_types)
    binned_λ_arr         = Vector{Vector{Float64}}(undef, n_types)
    binned_spectra_σ_arr = Vector{Vector{Float64}}(undef, n_types)
    best_fit_arr            = Vector{Vector{Float64}}(undef, n_types)
    best_fit_λ_arr          = Vector{Vector{Float64}}(undef, n_types)
    transient_component_arr = Vector{Vector{Float64}}(undef, n_types)
    host_component_arr      = Vector{Vector{Float64}}(undef, n_types)
    fitted_spectra_arr      = Vector{Vector{Float64}}(undef, n_types)
    fitted_spectra_σ_arr    = Vector{Vector{Float64}}(undef, n_types)
    fitted_spectra_λ_arr    = Vector{Vector{Float64}}(undef, n_types)

    working_flux   = deepcopy(target_spectra)
    working_flux_σ = deepcopy(target_spectra_σ)
    working_λ      = deepcopy(target_λ)
    original_λ     = deepcopy(target_λ)

    cleaned_λ, cleaned_flux, cleaned_flux_σ, telluric_λ, emission_line_λ, emission_line = process_spectra(
        working_λ, working_flux, working_flux_σ, telluric_λ, remove_telluric
    )

    original_target_λ      = deepcopy(working_λ)
    original_target_flux   = deepcopy(working_flux)
    original_target_flux_σ = deepcopy(working_flux_σ)

    # ── Extinction grids — transient and host now independent ─────────────────
    # Applying different amounts of dust reddening to the SN and its host is
    # Same Rv (dust law shape) is used for both for now 
    Av_grid_transient = collect(Av_min:Av_step:Av_max)
    Av_grid_host      = collect(Av_host_min:Av_host_step:Av_host_max)
    N_av_t = length(Av_grid_transient)
    N_av_h = length(Av_grid_host)
    ext_law = F99(Rv = Rv)

    # ── Pre-compute all extinction curves on the FIXED master_grid ────────────
    # NOTE ON COST: the Av search space now scales as N_av_t * N_av_h per
    # redshift step (previously just N_av). With the defaults above that's
    # 31 * 13 = 403 GEMM passes per z instead of 31 — the host grid defaults to
    # a coarser step specifically to keep this manageable. Tighten
    # Av_host_step if you need finer host-extinction resolution and can afford
    # the extra compute.
    ext_curves_transient_gpu = if use_prebuilt
        N_lam_master = length(master_grid)
        [to_gpu(Float32.(10.0 .^ (-0.4f0 .* Float32(Av) .* Float32.(ext_law.(master_grid)))))
         for Av in Av_grid_transient]
    else
        nothing
    end

    ext_curves_host_gpu = if use_prebuilt
        [to_gpu(Float32.(10.0 .^ (-0.4f0 .* Float32(Av) .* Float32.(ext_law.(master_grid)))))
         for Av in Av_grid_host]
    else
        nothing
    end

    redshift_range = collect(z_min:z_step:z_max)
    N_z = length(redshift_range)

    galactic_transients = Set(
        t_idx for t_idx in 1:n_transient_templates
        if transient_template_info["Galactic Source"][transient_template_names[t_idx]] == "Y"
    )

    N_T          = n_transient_templates
    N_H          = n_host_templates
    N_lam_master = use_prebuilt ? length(master_grid) : 0

    type_indices = [type_to_index[transient_template_info["Type"][transient_template_names[i]]] for i in 1:N_T]
    type_locks   = [ReentrantLock() for _ in 1:n_types]

    type_groups = [Int[] for _ in 1:n_types]
    for t_idx in 1:N_T
        push!(type_groups[type_indices[t_idx]], t_idx)
    end

    # ── Type constraint ───────────────────────────────────────────────────────
    # If input_type is set, restrict the search to only templates of that type.
    # Matching is case-insensitive. Falls back gracefully if no templates match.
    has_type_constraint = !ismissing(input_type) &&
                          input_type != "None"   &&
                          !isempty(strip(string(input_type)))

    allowed_t_indices = if has_type_constraint
        normalised = lowercase(strip(string(input_type)))
        allowed = Set{Int}(
            t_idx for t_idx in 1:N_T
            if lowercase(strip(transient_template_info["Type"][transient_template_names[t_idx]])) == normalised
        )
        if isempty(allowed)
            printstyled("Warning: ", color=:yellow)
            println("input_type='$input_type' matched no templates — ignoring type constraint.")
            Set{Int}(1:N_T)
        else
            println("Type constraint active: $(length(allowed)) template(s) of type '$input_type'.")
            allowed
        end
    else
        Set{Int}(1:N_T)
    end

    # ── Pre-compute all target resamples across the redshift grid ─────────────
    target_f_stack     = zeros(Float32, N_lam_master, N_z)
    target_w_stack     = zeros(Float32, N_lam_master, N_z)
    target_valid_stack = falses(N_lam_master, N_z)
    raw_flux_stack     = Vector{Vector{Float32}}(undef, N_z)
    raw_flux_σ_stack   = Vector{Vector{Float32}}(undef, N_z)

    Threads.@threads for iz in 1:N_z
        z_val    = redshift_range[iz]
        bottom_λ = 3800.0f0 / (1.0f0 + Float32(z_val))
        top_λ    = 8000.0f0 / (1.0f0 + Float32(z_val))

        de_redshifted_λ = original_λ ./ (1 + z_val)
        t_on_grid, t_σ_on_grid = spectral_resampler(
            master_grid, de_redshifted_λ, cleaned_flux, cleaned_flux_σ
        )

        raw_flux_stack[iz]   = copy(t_on_grid)
        raw_flux_σ_stack[iz] = copy(t_σ_on_grid)

        v_mask = .!isnan.(t_on_grid) .& .!isnan.(t_σ_on_grid) .&
                 (t_σ_on_grid .> 0) .& (master_grid .>= bottom_λ) .& (master_grid .<= top_λ)

        target_f_stack[v_mask, iz] .= Float32.(t_on_grid[v_mask])
        target_w_stack[v_mask, iz] .= 1.0f0 ./ Float32.(t_σ_on_grid[v_mask]).^2
        target_valid_stack[:, iz]  .= v_mask
    end

    # ── Pre-compute overlap matrix (template coverage vs target validity) ─────
    overlap_matrix = Float32.(to_cpu(MT_master_gpu))' * Float32.(target_valid_stack)
    N_pts_target_z = vec(sum(target_valid_stack, dims=1))

    # ── Pre-allocate GPU result tensors (reused every pass) ───────────────────
    if use_gpu && (is_metal_loaded)
        B_gpu          = similar(T_master_gpu, Float32, N_T, N_H)
        C_gpu          = similar(T_master_gpu, Float32, N_T, N_H)
        chi_sq_gpu     = similar(T_master_gpu, Float32, N_T, N_H)
        chi_sq_red_gpu = similar(T_master_gpu, Float32, N_T, N_H)
        T_ext_gpu      = similar(T_master_gpu)
        H_ext_gpu      = similar(H_master_gpu)
    else
        B_gpu          = zeros(Float32, N_T, N_H)
        C_gpu          = zeros(Float32, N_T, N_H)
        chi_sq_gpu     = zeros(Float32, N_T, N_H)
        chi_sq_red_gpu = zeros(Float32, N_T, N_H)
        T_ext_gpu      = zeros(Float32, size(T_master))
        H_ext_gpu      = zeros(Float32, size(H_master))
    end

    # ── Batch-upload target stack once ────────────────────────────────────────
    target_f_stack_gpu = to_gpu(target_f_stack)
    target_w_stack_gpu = to_gpu(target_w_stack)

    # ── Running-best trackers across all passes ───────────────────────────────
    best_chi_tracker           = fill(Float32(Inf), N_T * N_H, N_z)
    best_av_transient_tracker  = zeros(Float32,     N_T * N_H, N_z)   # ← was best_av_tracker
    best_av_host_tracker       = zeros(Float32,     N_T * N_H, N_z)   # ← new
    best_npts_tracker          = zeros(Float32,     N_T * N_H, N_z)

    # ── Outer z loop ──────────────────────────────────────────────────────────
    for (iz, z) in enumerate(batch_mode ? redshift_range : ProgressBar(redshift_range))

        if !ismissing(target_redshift) && target_redshift != "None"
            round(z, digits=2) != round(target_redshift, digits=2) && continue
        end

        if use_prebuilt
            Y_gpu = target_f_stack_gpu[:, iz]
            W_gpu = target_w_stack_gpu[:, iz]
            WT    = W_gpu .* MT_master_gpu
        else
            # ── Legacy path ──────────────────────────────────────────────────
            redshifted_flux, redshifted_flux_σ, redshifted_λ,
            transient_templates, host_templates,
            transient_template_λ_z, host_template_λ_z,
            standard_template_indices, non_standard_template_indices,
            binned_flux, binned_flux_σ, binned_λ = redshift_spectra(
                cleaned_λ, cleaned_flux, cleaned_flux_σ,
                transient_template_files, transient_template_λ,
                transient_template_info, transient_template_names,
                host_template_files, z, host_template_λ, original_λ
            )

            frozen_flux   = redshifted_flux
            frozen_flux_σ = redshifted_flux_σ
            frozen_λ      = redshifted_λ
            frozen_λ_set  = Set(frozen_λ)

            N_lam = length(frozen_flux)
            N_T_l = length(transient_templates)
            N_H_l = length(host_templates)

            target_w_l   = zeros(Float32, N_lam)
            target_f_l   = zeros(Float32, N_lam)
            target_valid = .!isnan.(frozen_flux) .& .!isnan.(frozen_flux_σ) .& (frozen_flux_σ .> 0)
            target_w_l[target_valid] .= 1.0f0 ./ (frozen_flux_σ[target_valid].^2)
            target_f_l[target_valid] .= frozen_flux[target_valid]

            T_mat_base  = zeros(Float32, N_lam, N_T_l)
            MT_mat      = zeros(Float32, N_lam, N_T_l)
            H_mat_base  = zeros(Float32, N_lam, N_H_l)
            MH_mat      = zeros(Float32, N_lam, N_H_l)

            for h_idx in 1:N_H_l
                if h_idx in non_standard_template_indices
                    common_set   = intersect(Set(host_template_λ_z[h_idx]), frozen_λ_set)
                    common_idx_h = findall(in(common_set), host_template_λ_z[h_idx])
                    common_idx_z = findall(in(common_set), frozen_λ)
                    H_mat_base[common_idx_z, h_idx] .= host_templates[h_idx][common_idx_h]
                    MH_mat[common_idx_z, h_idx]     .= 1.0f0
                else
                    H_mat_base[:, h_idx] .= host_templates[h_idx]
                    MH_mat[:, h_idx]     .= 1.0f0
                end
                nan_mask = isnan.(H_mat_base[:, h_idx])
                if any(nan_mask)
                    H_mat_base[nan_mask, h_idx] .= 0.0f0
                    MH_mat[nan_mask, h_idx]     .= 0.0f0
                end
            end

            for t_idx in 1:N_T_l
                z != 0.0 && t_idx in galactic_transients && continue
                if t_idx in non_standard_template_indices
                    common_set = intersect(Set(transient_template_λ_z[t_idx]), frozen_λ_set)
                    (length(common_set) / N_lam) < (minimum_overlap / 100) && continue
                    common_idx_t = findall(in(common_set), transient_template_λ_z[t_idx])
                    common_idx_z = findall(in(common_set), frozen_λ)
                    T_mat_base[common_idx_z, t_idx] .= transient_templates[t_idx][common_idx_t]
                    MT_mat[common_idx_z, t_idx]     .= 1.0f0
                else
                    T_mat_base[:, t_idx] .= transient_templates[t_idx]
                    MT_mat[:, t_idx]     .= 1.0f0
                end
                nan_mask = isnan.(T_mat_base[:, t_idx])
                if any(nan_mask)
                    T_mat_base[nan_mask, t_idx] .= 0.0f0
                    MT_mat[nan_mask, t_idx]     .= 0.0f0
                end
            end

            Y_gpu = to_gpu(target_f_l)
            W_gpu = to_gpu(target_w_l)
            MT_g  = to_gpu(MT_mat)
            MH_g  = to_gpu(MH_mat)
            WT    = W_gpu .* MT_g

            ext_law_base = Float32.(ext_law.(frozen_λ))
        end

        # ── Inner Av loop — transient extinction is the outer loop since it
        # touches T_mat/T_ext_gpu, which we want to recompute as few times as
        # possible; host extinction is the inner loop, recomputed per combo. ──
        for (av_t_idx, Av_t) in enumerate(Av_grid_transient)

            if use_prebuilt
                T_ext_gpu .= T_master_gpu .* ext_curves_transient_gpu[av_t_idx]
            else
                ext_curve_t_cpu = Float32.(10.0 .^ (-0.4f0 .* Float32(Av_t) .* ext_law_base))
                T_mat = T_mat_base .* ext_curve_t_cpu
                T_g   = to_gpu(T_mat)
            end

            for (av_h_idx, Av_h) in enumerate(Av_grid_host)

                if use_prebuilt
                    H_ext_gpu .= H_master_gpu .* ext_curves_host_gpu[av_h_idx]

                    WT_T    = WT .* T_ext_gpu
                    WT_T2   = WT_T .* T_ext_gpu
                    H_clean = H_ext_gpu .* MH_master_gpu

                    S_sh_gpu  = transpose(WT_T) * H_clean
                    S_os_gpu  = transpose(WT_T .* Y_gpu) * MH_master_gpu
                    S_oh_gpu  = transpose(WT) * (H_clean .* Y_gpu)
                    S_ss_gpu  = transpose(WT_T2) * MH_master_gpu
                    S_hh_gpu  = transpose(WT) * (H_clean .* H_clean)
                    S_oo_gpu  = transpose(WT .* (Y_gpu.^2)) * MH_master_gpu
                    N_pts_gpu = transpose(MT_master_gpu) * MH_master_gpu
                else
                    ext_curve_h_cpu = Float32.(10.0 .^ (-0.4f0 .* Float32(Av_h) .* ext_law_base))
                    H_mat = H_mat_base .* ext_curve_h_cpu
                    H_g   = to_gpu(H_mat)

                    WT_T    = WT .* T_g
                    WT_T2   = WT_T .* T_g
                    H_clean = H_g .* MH_g

                    S_sh_gpu  = transpose(WT_T) * H_clean
                    S_os_gpu  = transpose(WT_T .* Y_gpu) * MH_g
                    S_oh_gpu  = transpose(WT) * (H_clean .* Y_gpu)
                    S_ss_gpu  = transpose(WT_T2) * MH_g
                    S_hh_gpu  = transpose(WT) * (H_clean .* H_clean)
                    S_oo_gpu  = transpose(WT .* (Y_gpu.^2)) * MH_g
                    N_pts_gpu = transpose(MT_g) * MH_g
                end

                # ── Analytical solve ──────────────────────────────────────────
                det_gpu = max.(S_ss_gpu .* S_hh_gpu .- S_sh_gpu.^2, 1f-10)

                B_gpu .= (S_os_gpu .* S_hh_gpu .- S_oh_gpu .* S_sh_gpu) ./ det_gpu
                C_gpu .= (S_oh_gpu .* S_ss_gpu .- S_os_gpu .* S_sh_gpu) ./ det_gpu

                chi_sq_gpu .= S_oo_gpu .+ B_gpu.^2 .* S_ss_gpu .+ C_gpu.^2 .* S_hh_gpu .-
                              2.0f0 .* B_gpu .* S_os_gpu .- 2.0f0 .* C_gpu .* S_oh_gpu .+
                              2.0f0 .* B_gpu .* C_gpu .* S_sh_gpu

                invalid_coeffs = (B_gpu .< 0.0f0) .| (C_gpu .< 0.0f0) .| (N_pts_gpu .<= 0.0f0)
                chi_sq_gpu[invalid_coeffs] .= Float32(Inf)

                chi_sq_red_gpu .= chi_sq_gpu ./ max.(N_pts_gpu, 1.0f0)

                chi_pass_vec  = vec(to_cpu(chi_sq_red_gpu))
                npts_pass_vec = vec(to_cpu(N_pts_gpu))

                chi_pass_vec[isnan.(chi_pass_vec)] .= Float32(Inf)

                current_best = @view best_chi_tracker[:, iz]
                improved     = chi_pass_vec .< current_best

                best_chi_tracker[improved, iz]          .= chi_pass_vec[improved]
                best_npts_tracker[improved, iz]         .= npts_pass_vec[improved]
                best_av_transient_tracker[improved, iz] .= Float32(Av_t)
                best_av_host_tracker[improved, iz]      .= Float32(Av_h)

            end  # ── end host Av loop
        end  # ── end transient Av loop
    end  # ── end z loop

    # ── Reshape best trackers for type-level reduction ────────────────────────
    results_3d      = reshape(best_chi_tracker,          N_T, N_H, N_z)
    npts_3d         = reshape(best_npts_tracker,         N_T, N_H, N_z)
    av_transient_3d = reshape(best_av_transient_tracker, N_T, N_H, N_z)
    av_host_3d      = reshape(best_av_host_tracker,      N_T, N_H, N_z)

    # ── Parallel type-level reduction ─────────────────────────────────────────
    Threads.@threads for t_type_index in 1:n_types
        t_indices_for_type = type_groups[t_type_index]

        best_val = Inf32
        best_t   = 0; best_h = 0; best_iz = 0

        for iz in 1:N_z
            z       = redshift_range[iz]
            N_pts_z = N_pts_target_z[iz]

            for t_idx in t_indices_for_type
                t_idx in galactic_transients && z != 0.0 && continue
                t_idx in allowed_t_indices || continue   # ← type constraint

                if overlap_matrix[t_idx, iz] / max(N_pts_z, 1.0f0) < (minimum_overlap / 100.0f0)
                    continue
                end

                for h_idx in 1:N_H
                    c_val = results_3d[t_idx, h_idx, iz]
                    if c_val < best_val
                        best_val = c_val
                        best_t   = t_idx
                        best_h   = h_idx
                        best_iz  = iz
                    end
                end
            end
        end

        if best_val < chi_squared_red_arr[t_type_index] && best_t > 0
            chi_squared_arr[t_type_index]         = best_val * npts_3d[best_t, best_h, best_iz]
            chi_squared_red_arr[t_type_index]     = best_val
            best_indices[t_type_index, 1]         = best_t
            best_indices[t_type_index, 2]         = best_h
            best_Av_transient_arr[t_type_index]   = av_transient_3d[best_t, best_h, best_iz]
            best_Av_host_arr[t_type_index]        = av_host_3d[best_t, best_h, best_iz]
            redshift_vals[t_type_index]            = redshift_range[best_iz]
            best_z_index[t_type_index]             = best_iz
        end
    end

    # ── Extract top N_top fits globally ──────────────────────────────────────
    let
        flat_chi   = vec(best_chi_tracker)
        flat_av_t  = vec(best_av_transient_tracker)
        flat_av_h  = vec(best_av_host_tracker)
        flat_npts  = vec(best_npts_tracker)

        valid_flat = falses(N_T * N_H * N_z)
        for iz in 1:N_z
            z_val   = redshift_range[iz]
            N_pts_z = N_pts_target_z[iz]
            for t_idx in 1:N_T
                t_idx in galactic_transients && z_val != 0.0 && continue
                t_idx in allowed_t_indices || continue   # ← type constraint
                overlap_matrix[t_idx, iz] / max(N_pts_z, 1.0f0) < (minimum_overlap / 100.0f0) && continue
                for h_idx in 1:N_H
                    th_flat  = (h_idx - 1) * N_T + t_idx
                    flat_idx = th_flat + (iz - 1) * (N_T * N_H)
                    valid_flat[flat_idx] = true
                end
            end
        end

        candidate_idxs  = findall(valid_flat .& isfinite.(flat_chi))
        seen_transients = Set{Tuple{String, Float64}}()

        if !isempty(candidate_idxs)
            n_cand     = min(N_top * 10, length(candidate_idxs))
            cand_order = partialsortperm(flat_chi[candidate_idxs], 1:n_cand)

            global_rank = 0
            for ci in cand_order
                global_rank >= N_top && break

                flat_idx = candidate_idxs[ci]
                th_flat  = mod1(flat_idx, N_T * N_H)
                iz       = (flat_idx - 1) ÷ (N_T * N_H) + 1
                t_idx    = mod1(th_flat, N_T)
                h_idx    = (th_flat - 1) ÷ N_T + 1

                z_val    = redshift_range[iz]
                Av_t_val = Float64(flat_av_t[flat_idx])
                Av_h_val = Float64(flat_av_h[flat_idx])

                transient_key = (transient_template_names[t_idx], Float64(z_val))
                transient_key in seen_transients && continue
                push!(seen_transients, transient_key)

                global_rank += 1
                push!(top_fits, Dict{String,Any}(
                    "rank"               => global_rank,
                    "chi_sq_red"         => Float64(flat_chi[flat_idx]),
                    "transient_type"     => transient_template_info["Type"][transient_template_names[t_idx]],
                    "transient_template" => transient_template_names[t_idx],
                    "host_type"          => host_template_info["HG Type"][host_template_names[h_idx]],
                    "host_template"      => host_template_names[h_idx],
                    "redshift"           => Float64(z_val),
                    "Av_transient"       => Av_t_val,          # ← was "Av"
                    "Av_host"            => Av_h_val,          # ← new
                    "E_B_V_transient"    => Av_t_val / Float64(Rv),  # ← was "E_B_V"
                    "E_B_V_host"         => Av_h_val / Float64(Rv),  # ← new
                    "phase"              => get(transient_template_info["tau"], transient_template_names[t_idx], "N/A")
                ))
            end
        end
    end

    # ── Deferred reconstruction ───────────────────────────────────────────────
    for t_type_index in 1:n_types
        t_idx = best_indices[t_type_index, 1]
        h_idx = best_indices[t_type_index, 2]
        t_idx == 0 && continue

        iz    = best_z_index[t_type_index]
        z_val = redshift_vals[t_type_index]
        Av_t  = best_Av_transient_arr[t_type_index]
        Av_h  = best_Av_host_arr[t_type_index]

        if use_prebuilt
            bottom_λ  = 3800.0f0 / (1.0f0 + Float32(z_val))
            top_λ     = 8000.0f0 / (1.0f0 + Float32(z_val))
            row_start = max(1, round(Int, (bottom_λ - 2315.0f0) / 15.0f0) + 1)
            row_end   = min(N_lam_master, round(Int, (top_λ - 2315.0f0) / 15.0f0) + 1)

            target_f_win      = target_f_stack[row_start:row_end, iz]
            target_w_win      = target_w_stack[row_start:row_end, iz]
            frozen_flux_σ_win = raw_flux_σ_stack[iz][row_start:row_end]
            frozen_λ_win      = master_grid[row_start:row_end]

            MT_col = to_cpu(MT_master_gpu[row_start:row_end, t_idx])
            MH_col = to_cpu(MH_master_gpu[row_start:row_end, h_idx])
            T_col  = to_cpu(T_master_gpu[row_start:row_end, t_idx])
            H_col  = to_cpu(H_master_gpu[row_start:row_end, h_idx])

            # Apply the separately-fitted transient and host extinction
            ext_curve_t_win = Float32.(10.0 .^ (-0.4f0 .* Float32(Av_t) .* Float32.(ext_law.(frozen_λ_win))))
            ext_curve_h_win = Float32.(10.0 .^ (-0.4f0 .* Float32(Av_h) .* Float32.(ext_law.(frozen_λ_win))))
            T_col .= T_col .* ext_curve_t_win
            H_col .= H_col .* ext_curve_h_win

            z_val != 0.0 && t_idx in galactic_transients && (MT_col .= 0.0f0)
            invalid_win = .!target_valid_stack[row_start:row_end, iz]
            MT_col[invalid_win] .= 0.0f0
            MH_col[invalid_win] .= 0.0f0

            valid_mask = (target_w_win .> 0) .& (MT_col .> 0) .& (MH_col .> 0)

            Y_v = target_f_win[valid_mask]
            W_v = target_w_win[valid_mask]
            T_v = T_col[valid_mask]
            H_v = H_col[valid_mask]

            s_ss = sum(W_v .* T_v.^2)
            s_hh = sum(W_v .* H_v.^2)
            s_sh = sum(W_v .* T_v .* H_v)
            s_os = sum(W_v .* Y_v .* T_v)
            s_oh = sum(W_v .* Y_v .* H_v)

            det_win        = s_ss * s_hh - s_sh^2
            norm_transient = (s_os * s_hh - s_oh * s_sh) / det_win
            norm_host      = (s_oh * s_ss - s_os * s_sh) / det_win

            norm_arr[t_type_index, 1] = norm_transient
            norm_arr[t_type_index, 2] = norm_host

            binned_flux_win   = raw_flux_stack[iz][row_start:row_end]
            binned_flux_σ_win = frozen_flux_σ_win
            binned_λ_win      = master_grid[row_start:row_end]
        else
            continue
        end

        norm_transient = norm_arr[t_type_index, 1]
        norm_host      = norm_arr[t_type_index, 2]

        b_target_v    = target_f_win[valid_mask]
        b_target_σ_v  = frozen_flux_σ_win[valid_mask]
        b_transient_v = T_col[valid_mask]
        b_host_v      = H_col[valid_mask]
        b_host_λ      = frozen_λ_win[valid_mask]

        best_f = (b_transient_v .* norm_transient) .+ (b_host_v .* norm_host)

        host_flux_sum = sum(b_host_v .* norm_host)
        total_flux    = sum(best_f)
        contam_array[t_type_index] = (host_flux_sum / total_flux) * 100

        emission_lines_arr[t_type_index]   = collect(Float64, emission_line)
        emission_lines_λ_arr[t_type_index] = collect(Float64, emission_line_λ)

        binned_spectra_arr[t_type_index]   = collect(Float64, binned_flux_win)
        binned_λ_arr[t_type_index]         = collect(Float64, binned_λ_win)
        binned_spectra_σ_arr[t_type_index] = collect(Float64, binned_flux_σ_win)

        best_fit_arr[t_type_index]            = collect(Float64, best_f)
        best_fit_λ_arr[t_type_index]          = collect(Float64, b_host_λ)
        fitted_spectra_arr[t_type_index]      = collect(Float64, b_target_v)
        fitted_spectra_σ_arr[t_type_index]    = collect(Float64, b_target_σ_v)
        fitted_spectra_λ_arr[t_type_index]    = collect(Float64, b_host_λ)
        transient_component_arr[t_type_index] = collect(Float64, b_transient_v .* norm_transient)
        host_component_arr[t_type_index]      = collect(Float64, b_host_v .* norm_host)
    end

    # ── Collect results ───────────────────────────────────────────────────────
    fitted_type_indices = findall(x -> x != 0, best_indices[:, 1])

    top_fit_types           = [transient_template_info["Type"][transient_template_names[i]]  for i in best_indices[fitted_type_indices, 1]]
    top_fit_hg_types        = [host_template_info["HG Type"][host_template_names[i]]         for i in best_indices[fitted_type_indices, 2]]
    top_fit_av_transient    = [best_Av_transient_arr[i]                                       for i in fitted_type_indices]  # ← was top_fit_av
    top_fit_av_host         = [best_Av_host_arr[i]                                            for i in fitted_type_indices]  # ← new
    top_fit_ebv_transient   = [av / Rv for av in top_fit_av_transient]                                                       # ← was top_fit_eb_v
    top_fit_ebv_host        = [av / Rv for av in top_fit_av_host]                                                            # ← new
    top_fit_phase           = [transient_template_info["tau"][transient_template_names[i]]   for i in best_indices[fitted_type_indices, 1]]

    best_type_full_idx = argmin(chi_squared_red_arr)
    best_type_idx      = findfirst(==(best_type_full_idx), fitted_type_indices)
    best_redshift      = redshift_vals[best_type_full_idx]

    if isempty(top_fit_types) || isnothing(best_type_idx)
        best_type = "Unknown"
        println("--------------------")
        printstyled("ERROR: ", color=:red)
        println("Fitting $SN_ID - No fitting results found.")
        println("--------------------")
    else
        best_type = top_fit_types[best_type_idx]
    end

    return norm_arr, chi_squared_red_arr, best_redshift, best_type, top_fits
end
# ── build_fit_plot ────────────────────────────────────────────────────────
# Reconstructs and returns a Plots.jl figure for a single entry from top_fits.
# Used by the interactive viewer in SIREN.jl to cycle through top results.
# T/MT/H/MH_master_cpu must be plain CPU Arrays (not GPU arrays).
function build_fit_plot(
        fit::Dict{String,Any},
        target_name::String,
        original_flux, original_flux_σ, original_λ,
        sn_template_names,
        hg_template_names,
        T_master_cpu, MT_master_cpu,
        H_master_cpu, MH_master_cpu,
        master_grid;
        Rv::Float64 = 3.1, z_step::Float64 = 0.01)

    z_val   = Float32(fit["redshift"])
    Av_val  = Float32(fit["Av"])
    t_name  = fit["transient_template"]
    h_name  = fit["host_template"]
    t_type  = fit["transient_type"]
    h_type  = fit["host_type"]
    phase   = fit["phase"]
    rank    = fit["rank"]
    chi_red = round(fit["chi_sq_red"], digits=3)

    t_idx = findfirst(==(t_name), sn_template_names)
    h_idx = findfirst(==(h_name), hg_template_names)
    (isnothing(t_idx) || isnothing(h_idx)) && return nothing

    # Wavelength window in rest frame for this z
    N_lam_master = length(master_grid)
    bottom_λ  = 3800.0f0 / (1.0f0 + z_val)
    top_λ     = 8000.0f0 / (1.0f0 + z_val)
    row_start = max(1, round(Int, (bottom_λ - 2315.0f0) / 15.0f0) + 1)
    row_end   = min(N_lam_master, round(Int, (top_λ  - 2315.0f0) / 15.0f0) + 1)
    frozen_λ_win = master_grid[row_start:row_end]

    # Apply extinction to template columns
    ext_law   = F99(Rv = Float32(Rv))
    ext_curve = Float32.(10.0 .^ (-0.4f0 .* Av_val .* Float32.(ext_law.(frozen_λ_win))))

    T_col  = T_master_cpu[row_start:row_end, t_idx]  .* ext_curve
    H_col  = H_master_cpu[row_start:row_end, h_idx]  .* ext_curve
    MT_col = copy(MT_master_cpu[row_start:row_end, t_idx])
    MH_col = copy(MH_master_cpu[row_start:row_end, h_idx])

    # Resample observed spectrum to the rest-frame master grid at this z
    de_redshifted_λ = Float64.(original_λ) ./ (1.0 + z_val)
    t_on_grid, t_σ_on_grid = spectral_resampler(
        Float64.(master_grid), de_redshifted_λ,
        Float64.(original_flux), Float64.(original_flux_σ))

    target_f_win = Float32.(t_on_grid[row_start:row_end])
    target_σ_win = Float32.(t_σ_on_grid[row_start:row_end])
    target_w_win = zeros(Float32, length(frozen_λ_win))

    valid = .!isnan.(target_f_win) .& .!isnan.(target_σ_win) .& (target_σ_win .> 0) .&
            (frozen_λ_win .>= bottom_λ) .& (frozen_λ_win .<= top_λ)

    target_w_win[valid] .= 1.0f0 ./ target_σ_win[valid].^2
    MT_col[.!valid] .= 0.0f0
    MH_col[.!valid] .= 0.0f0

    valid_mask = (target_w_win .> 0) .& (MT_col .> 0) .& (MH_col .> 0)
    sum(valid_mask) < 5 && return nothing

    Y_v = target_f_win[valid_mask]
    W_v = target_w_win[valid_mask]
    T_v = T_col[valid_mask]
    H_v = H_col[valid_mask]

    s_ss = sum(W_v .* T_v.^2);  s_hh = sum(W_v .* H_v.^2)
    s_sh = sum(W_v .* T_v .* H_v)
    s_os = sum(W_v .* Y_v .* T_v);  s_oh = sum(W_v .* Y_v .* H_v)

    det_win = s_ss * s_hh - s_sh^2
    abs(det_win) < 1e-10 && return nothing

    B = (s_os * s_hh - s_oh * s_sh) / det_win   # transient scale
    C = (s_oh * s_ss - s_os * s_sh) / det_win   # host scale
    (B < 0 || C < 0) && return nothing

    # Observer-frame wavelengths for plotting
    plot_λ         = Float64.(frozen_λ_win[valid_mask]) .* (1.0 + z_val)
    transient_comp = Float64.(T_col[valid_mask]) .* B
    host_comp      = Float64.(H_col[valid_mask]) .* C
    best_f         = transient_comp .+ host_comp

    # Raw observed spectrum (NaN-filtered)
    obs_λ_raw    = collect(Float64, original_λ)
    obs_flux_raw = collect(Float64, original_flux)
    obs_err_raw  = collect(Float64, original_flux_σ)
    obs_valid    = isfinite.(obs_flux_raw) .& isfinite.(obs_err_raw) .& (obs_err_raw .> 0)
    obs_λ    = obs_λ_raw[obs_valid]
    obs_flux = obs_flux_raw[obs_valid]
    obs_err  = obs_err_raw[obs_valid]

    # Binned observed (valid pixels only, observer frame)
    binned_flux_raw = Float64.(target_f_win[valid_mask])
    binned_err_raw  = Float64.(target_σ_win[valid_mask])
    binned_valid    = isfinite.(binned_flux_raw) .& isfinite.(binned_err_raw)
    binned_λ    = plot_λ[binned_valid]
    binned_flux = binned_flux_raw[binned_valid]
    binned_err  = binned_err_raw[binned_valid]

    display_name = splitext(basename(target_name))[1]
    z_precision  = round(Int, -log10(z_step)) +1 #Changes with the z_step used in fitting.
    z_str        = round(Float64(z_val), digits=z_precision)
    av_str       = round(Float64(Av_val), digits=1)

    finite_obs   = filter(isfinite, obs_flux)
    host_pos     = host_comp[host_comp .> 0]
    trans_pos    = transient_comp[transient_comp .> 0]
    template_min = (isempty(host_pos) || isempty(trans_pos)) ? minimum(finite_obs) :
                    minimum([host_pos; trans_pos])
    y_lo = min(minimum(finite_obs), template_min) * 0.7
    y_hi = quantile(finite_obs, 0.99) * 1.3

    ribbon_obs    = clamp.(min.(obs_err,    (y_hi-y_lo)*0.1), 0.0, max.(y_hi .- obs_flux,    0.0))
    ribbon_binned = clamp.(min.(binned_err, (y_hi-y_lo)*0.1), 0.0, max.(y_hi .- binned_flux, 0.0))

    p = redirect_stderr(devnull) do
        _p = plot(
            size=(1600, 900), dpi=300, grid=false,
            guidefontsize=22, tickfontsize=16, titlefont=22, legendfontsize=16,
            left_margin=9*Plots.mm, bottom_margin=10*Plots.mm,
            top_margin=10*Plots.mm, right_margin=4*Plots.mm)

        plot!(_p, obs_λ, obs_flux,
            ribbon=ribbon_obs, fillalpha=0.15, fillcolor="#C8C8C8",
            linecolor="#909090", linewidth=0.8,
            label="Observed ($display_name)",
            xlabel="Wavelength (Å)", ylabel="Flux (arb. units)",
            legend=:topright, ylims=(y_lo, y_hi))

        plot!(_p, binned_λ, binned_flux,
            ribbon=ribbon_binned, fillalpha=0.10, fillcolor=:black,
            linecolor=:black, linewidth=1.4, label="Binned + Lines Removed")

        plot!(_p, plot_λ, best_f,
            linecolor=:red, linewidth=1.8,
            label="Fit #$rank  ($t_type + $h_type)  χ²ᵣ=$chi_red")

        plot!(_p, plot_λ, transient_comp,
            linecolor=:royalblue, linewidth=1.2, linestyle=:dash,
            label="Transient ($t_type)")

        plot!(_p, plot_λ, host_comp,
            linecolor=:darkorange, linewidth=1.2, linestyle=:dash,
            label="Host Galaxy ($h_type)")

        title!(_p, "$display_name  |  z=$z_str  |  type=$t_type  |  Phase=$phase days | Rank #$rank")
        _p
    end
    return p
end
# ── read_single_key ───────────────────────────────────────────────────────────
# Reads one keypress from stdin without requiring Enter.
# Puts the terminal in raw mode (via Julia's C runtime), reads the key(s),
# then restores cooked mode. Arrow keys send a 3-byte ESC sequence:
#   ESC (0x1b) → '[' (0x5b) → A/B/C/D
# Only Base Julia is used — fully PackageCompiler-safe.
function read_single_key()
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, 1)
    key = :unknown
    try
        b = read(stdin, UInt8)

        if b == 0x1b                                # ESC — could be arrow key
            sleep(0.04)                             # let the rest of the sequence arrive
            if bytesavailable(stdin) >= 2
                b2 = read(stdin, UInt8)
                b3 = read(stdin, UInt8)
                if b2 == UInt8('[')
                    if     b3 == UInt8('C'); key = :right   # →
                    elseif b3 == UInt8('D'); key = :left    # ←
                    elseif b3 == UInt8('A'); key = :up      # ↑  (treat as next)
                    elseif b3 == UInt8('B'); key = :down    # ↓  (treat as prev)
                    end
                end
            else
                key = :quit                         # bare ESC = quit
            end

        elseif b == UInt8('q') || b == UInt8('Q'); key = :quit
        elseif b == UInt8('s') || b == UInt8('S'); key = :save
        elseif b == UInt8('g') || b == UInt8('G'); key = :goto
        elseif b == 13 || b == 10;                  key = :right   # Enter = next
        end
    finally
        ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, 0)
    end
    return key
end


# ── bring_plot_to_front ───────────────────────────────────────────────────────
# On macOS, GR spawns a separate GKSTerm process for its window.
# Clicking the terminal causes that window to lose focus.
# This function uses a one-line AppleScript to re-raise it.
# On non-macOS systems (or if the call fails) it silently does nothing.
function bring_plot_to_front()
    Sys.isapple() || return
    try
        run(`osascript -e 'tell application "System Events" to set frontmost of first process whose name is "GKSTerm" to true'`,
            wait = true)
    catch
        # Window management is best-effort — silently ignore
    end
end


function interactive_viewer(
        top_fits, target_name,
        original_flux, original_flux_σ, original_λ,
        sn_template_names, hg_template_names,
        T_master_cpu, MT_master_cpu,
        H_master_cpu, MH_master_cpu,
        master_grid, savepath;
        Rv::Float64 = 3.1)

    isempty(top_fits) && (println("No fits to display."); return)

    n_fits       = length(top_fits)
    current      = Ref(1)
    message      = Ref("")          # one-shot message shown on the status line
    first_render = Ref(true)

    # Status block is always exactly 2 lines:
    #   line 1 — fit info
    #   line 2 — message / empty
    STATUS_LINES = 2

    function render_status(idx)
        if !first_render[]
            # Move cursor up to the top of the status block and wipe everything below
            print("\e[$(STATUS_LINES)A\e[0J")
        end
        first_render[] = false

        fit = top_fits[idx]
        printstyled("  Fit $idx / $n_fits", color=:cyan)
        print("  —  $(fit["transient_type"]) + $(fit["host_type"])")
        print("  z=$(round(fit["redshift"], digits=2))")
        print("  χ²ᵣ=$(round(fit["chi_sq_red"], digits=3))")
        println("  Av=$(fit["Av"])")

        if !isempty(message[])
            printstyled("  $(message[])\n", color=:yellow)
            message[] = ""          # clear after showing once
        else
            println()               # keep block height constant
        end
    end

    # ── Static header — printed once ─────────────────────────────────────────
    println("\n──────────────────────────────────────────────")
    printstyled("Interactive Fit Viewer\n", color=:cyan)
    println("  ← / →   previous / next fit")
    println("  s        save current plot as PNG")
    println("  g        jump to fit number")
    println("  q / ESC  quit")
    println("──────────────────────────────────────────────")

    plot_cache = Dict{Int, Any}()

    function get_plot(idx)
        haskey(plot_cache, idx) && return plot_cache[idx]
        p = build_fit_plot(
            top_fits[idx], target_name,
            original_flux, original_flux_σ, original_λ,
            sn_template_names, hg_template_names,
            T_master_cpu, MT_master_cpu, H_master_cpu, MH_master_cpu,
            master_grid; Rv = Rv)
        plot_cache[idx] = p
        return p
    end

    while true
        idx = current[]

        render_status(idx)

        p = get_plot(idx)
        isnothing(p) ? (message[] = "Could not reconstruct this fit.") : display(p)

        key = read_single_key()

        if key == :right || key == :up
            if idx < n_fits
                current[] = idx + 1
            else
                message[] = "Already at last fit."
            end

        elseif key == :left || key == :down
            if idx > 1
                current[] = idx - 1
            else
                message[] = "Already at first fit."
            end

        elseif key == :goto
            # The goto prompt expands below the status block temporarily.
            # Reset first_render so the next cycle doesn't try to scroll back
            # through the prompt text.
            print("  Jump to fit number (1–$n_fits): ")
            raw = strip(readline())
            first_render[] = true
            target_idx = tryparse(Int, raw)
            if isnothing(target_idx) || target_idx < 1 || target_idx > n_fits
                message[] = "Invalid number — enter a value between 1 and $n_fits."
            else
                current[] = target_idx
            end

        elseif key == :save
            if isnothing(p)
                message[] = "Nothing to save — fit could not be reconstructed."
            else
                display_name = splitext(basename(target_name))[1]
                outpath = joinpath(savepath,
                    "$(display_name)_fit_$(lpad(idx, 3, '0')).png")
                redirect_stderr(devnull) do
                    savefig(p, outpath)
                end
                message[] = "Saved: $outpath"
            end

        elseif key == :quit
            print("\e[$(STATUS_LINES)A\e[0J")   # clear status on exit
            println("  Viewer closed.")
            break
        end
    end
end


    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_target_spectra, read_template_files, trim_nans, 
    calculate_B, calculate_C, calculate_spectral_error, parse_command_line_args, build_fit_plot, interactive_viewer, read_batch_metadata, get_target_params
end