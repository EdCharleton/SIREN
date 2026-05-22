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
    using BenchmarkTools
    using DataStructures
    using Pkg
    using Interpolations: Linear, Gridded, interpolate
    using LsqFit
    using LaTeXStrings
    using ProfileView
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
    use_gpu = true

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
            "--file", "-f"
                help = "Path to the input data file/directory (see --batch flag)"
                arg_type = String
                required = true
            
            "--redshift", "-z"
                help = "Redshift value (or 'None' for search)"
                arg_type = String
                default = "None"
    
            "--batch", "-b"
                help = "Flag to run in batch mode (multiple at once)"
                arg_type = Bool
                default = false
            
            "--remove_telluric"
                help = "Flag to remove telluric absorption at telluric wavelengths"
                arg_type = String
                default = "N"  # Default is 'no'
    
            "--galactic_source"
                help = "Flag indicating whether the source is galactic"
                arg_type = String
                default = "N"  # Default is 'no'
    
            "--show_plot"
                help = "Flag to show the plot"
                arg_type = Bool
                default = false  # Default is 'no'
    
            "--minimum_overlap"
                help = "Minimum overlap between the target and templates, units of %"
                arg_type = Float64
                default = 75  # Default is '0.0'

            "--use_gpu"
                help = "Flag to enable/disable GPU acceleration"
                arg_type = Bool
                default = false
            "--savepath"
                help = "Directory to save output plots"
                arg_type = String
                default = "classifications"
        end
    
        return parse_args(ARGS, s)
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

function plot_best_fit(SN_ID, z,
    best_indices, transient_template_info, transient_template_names,
    host_template_info, host_template_names,
    original_flux, original_flux_σ, original_λ,
    best_fit_arr, best_fit_λ_arr, transient_component_arr, host_component_arr,
    fitted_spectra_arr, fitted_spectra_σ_arr, fitted_spectra_λ_arr,
    chi_squared_red_arr, redshift_vals,
    emission_lines_arr, emission_lines_λ_arr,
    savepath = "classifications", show_plot = false)

    best = argmin(chi_squared_red_arr)

    if best_indices[best, 1] == 0
        println("--------------------")
        printstyled("ERROR: ", color=:red)
        println("plot_best_fit: No valid fit found for $SN_ID")
        println("--------------------")
        return nothing
    end

    t_idx = best_indices[best, 1]
    h_idx = best_indices[best, 2]

    transient_type = transient_template_info["Type"][transient_template_names[t_idx]]
    host_type      = host_template_info["HG Type"][host_template_names[h_idx]]
    phase          = get(transient_template_info["tau"], transient_template_names[t_idx], "N/A")
    chi_red        = round(chi_squared_red_arr[best], digits=3)
    z_best         = round(Float64(redshift_vals[best]), digits=3)

    # ── Data ─────────────────────────────────────────────────────────────────
    obs_λ_raw    = collect(Float64, original_λ)
    obs_flux_raw = collect(Float64, original_flux)
    obs_err_raw  = collect(Float64, original_flux_σ)

    # Filter NaNs and infs from observed spectrum before plotting.
    # Without this, Plots.jl draws the ribbon polygon across gaps,
    # producing diagonal lines across the entire figure.
    obs_valid    = isfinite.(obs_flux_raw) .& isfinite.(obs_err_raw) .& (obs_err_raw .> 0)
    obs_λ        = obs_λ_raw[obs_valid]
    obs_flux     = obs_flux_raw[obs_valid]
    obs_err      = obs_err_raw[obs_valid]

    # Binned spectrum — filter NaNs for the same reason
    binned_λ_raw    = collect(Float64, fitted_spectra_λ_arr[best]) .* (1 + z_best)
    binned_flux_raw = collect(Float64, fitted_spectra_arr[best])
    binned_err_raw  = collect(Float64, fitted_spectra_σ_arr[best])

    binned_valid = isfinite.(binned_flux_raw) .& isfinite.(binned_err_raw)
    binned_λ     = binned_λ_raw[binned_valid]
    binned_flux  = binned_flux_raw[binned_valid]
    binned_err   = binned_err_raw[binned_valid]

    # Best fit and components
    fit_λ          = collect(Float64, best_fit_λ_arr[best]) .* (1 + z_best)
    best_fit       = collect(Float64, best_fit_arr[best])
    transient_comp = collect(Float64, transient_component_arr[best])
    host_comp      = collect(Float64, host_component_arr[best])

    em_lines   = emission_lines_arr[best]
    em_lines_λ = emission_lines_λ_arr[best]

    display_name = splitext(splitpath(SN_ID)[end])[1]

    # ── Y-axis limits ─────────────────────────────────────────────────────────
    finite_flux = filter(isfinite, obs_flux)
    y_lo = quantile(finite_flux, 0.01)
    y_hi = quantile(finite_flux, 0.99) * 1.3

    # ── Clamp ribbon heights to prevent spike artifacts ───────────────────────
    # Large error values near spectral spikes cause the ribbon to extend far
    # beyond the y-axis clip, producing additional diagonal fill artifacts.
    ribbon_obs    = clamp.(min.(obs_err,    (y_hi - y_lo) * 0.1), 0.0, max.(y_hi .- obs_flux,    0.0))
    ribbon_binned = clamp.(min.(binned_err, (y_hi - y_lo) * 0.1), 0.0, max.(y_hi .- binned_flux, 0.0))

    # ── Plot — all rendering errors suppressed via redirect_stderr ────────────
    p = redirect_stderr(devnull) do
        _p = plot(
            size           = (1600, 900),
            dpi            = 300,
            grid           = false,
            guidefontsize  = 22,
            tickfontsize   = 16,
            titlefont      = 22,
            legendfontsize = 16,
            left_margin    = 9Plots.mm,
            bottom_margin  = 10Plots.mm,
            top_margin     = 10Plots.mm,
            right_margin   = 4Plots.mm
        )

        # 1. Observed spectrum (NaN-filtered) + clamped ribbon
        plot!(_p,
            obs_λ, obs_flux,
            ribbon    = ribbon_obs,
            fillalpha = 0.15,
            fillcolor = "#C8C8C8",
            linecolor = "#909090",
            linewidth = 0.8,
            label     = "Observed ($display_name)",
            xlabel    = "Wavelength (Å)",
            ylabel    = "Flux (arb. units)",
            legend    = :topright,
            ylims     = (y_lo, y_hi)
        )

        # 2. Binned + emission-line-removed spectrum
        plot!(_p,
            binned_λ, binned_flux,
            ribbon    = ribbon_binned,
            fillalpha = 0.10,
            fillcolor = :black,
            linecolor = :black,
            linewidth = 1.4,
            label     = "Binned + Lines Removed"
        )

        # 3. Best fit
        plot!(_p,
            fit_λ, best_fit,
            linecolor = :red,
            linewidth = 1.8,
            label     = "Best Fit ($transient_type + $host_type)"
        )

        # 4. Transient component
        plot!(_p,
            fit_λ, transient_comp,
            linecolor = :royalblue,
            linewidth = 1.2,
            linestyle = :dash,
            label     = "Transient ($transient_type)"
        )

        # 5. Host component
        plot!(_p, fit_λ, host_comp, linecolor = :darkorange, linewidth = 1.2, linestyle = :dash, label = "Host Galaxy ($host_type)")

        # 6. Emission line markers
        if !isempty(em_lines_λ)
            for (λ_em, name) in zip(em_lines_λ, em_lines)
                vline!(_p, [λ_em],
                    linecolor = :green,
                    linewidth = 0.8,
                    linestyle = :dot,
                    label     = ""
                )
                annotate!(_p, λ_em, y_hi * 0.92,
                    text(string(name), :green, :center, 7))
            end
        end

        title!(_p, "$display_name  |  z = $z_best  | type = $transient_type |  Phase = $phase days")
        _p
    end

    if show_plot
        display(p)
    end

    if !isnothing(savepath)
        outpath = joinpath(savepath, "$(display_name)_best_fit.png")
        redirect_stderr(devnull) do
            savefig(p, outpath)
        end
        return outpath
    end
    return p
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
        println("Loading in Target Spectra...")
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
        print(window_indexes)

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
        SN_ID, target_redshift, telluric_λ,
        batch_mode, show_plot = false, remove_telluric = "N", minimum_overlap = 50;
        master_grid = nothing, T_master = nothing, MT_master = nothing,
        H_master = nothing, MH_master = nothing,
        Rv = 3.1, Av_min = -3.0, Av_max = 3.0, Av_step = 0.5, savepath = "classifications")

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
    best_Av_arr   = zeros(Float32, n_types)
    best_z_index  = zeros(Int,     n_types)
    contam_array  = Vector{Float32}(undef, n_types)

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

    # ── Extinction grid ───────────────────────────────────────────────────────
    Av_grid = collect(Av_min:Av_step:Av_max)
    N_av    = length(Av_grid)
    ext_law = F99(Rv = Rv)

    # ── Pre-compute all extinction curves on the FIXED master_grid ────────────
    ext_curves_gpu = if use_prebuilt
        N_lam_master = length(master_grid)
        [to_gpu(Float32.(10.0 .^ (-0.4f0 .* Float32(Av) .* Float32.(ext_law.(master_grid)))))
         for Av in Av_grid]
    else
        nothing
    end

    # ── Redshift grid ─────────────────────────────────────────────────────────
    redshift_max = if ismissing(target_redshift) || target_redshift == "None"
        0.8
    else
        target_redshift > 0.8 ? target_redshift : 0.8
    end
    redshift_range = collect(0.00:0.01:redshift_max)
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
    best_chi_tracker  = fill(Float32(Inf), N_T * N_H, N_z)
    best_av_tracker   = zeros(Float32,     N_T * N_H, N_z)
    best_npts_tracker = zeros(Float32,     N_T * N_H, N_z)

    # ── Outer z loop ──────────────────────────────────────────────────────────
    for (iz, z) in enumerate(batch_mode ? redshift_range : ProgressBar(redshift_range))

        if !ismissing(target_redshift) && target_redshift != "None"
            round(z, digits=2) != round(target_redshift, digits=2) && continue
        end

        # Pre-compute/extract z-specific variables ONCE per z to save time
        if use_prebuilt
            Y_gpu = target_f_stack_gpu[:, iz]
            W_gpu = target_w_stack_gpu[:, iz]
            # WT is independent of Av, compute it once for this z
            WT = W_gpu .* MT_master_gpu
        else
            # ── Legacy path: Redshifting done ONCE per z ────────────────────
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

            # We create base matrices so we don't double-apply extinction in the Av loop
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

            # Push statics to GPU once per z
            Y_gpu = to_gpu(target_f_l)
            W_gpu = to_gpu(target_w_l)
            MT_g  = to_gpu(MT_mat)
            MH_g  = to_gpu(MH_mat)
            WT    = W_gpu .* MT_g
            
            ext_law_base = Float32.(ext_law.(frozen_λ))
        end

        # ── Inner Av loop ─────────────────────────────────────────────────────
        for (av_idx, Av) in enumerate(Av_grid)

            if use_prebuilt
                T_ext_gpu .= T_master_gpu .* ext_curves_gpu[av_idx]
                H_ext_gpu .= H_master_gpu .* ext_curves_gpu[av_idx]

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
                # Apply extinction to the clean base matrices
                ext_curve_cpu = Float32.(10.0 .^ (-0.4f0 .* Float32(Av) .* ext_law_base))
                T_mat = T_mat_base .* ext_curve_cpu
                H_mat = H_mat_base .* ext_curve_cpu

                T_g = to_gpu(T_mat)
                H_g = to_gpu(H_mat)

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

            # ── Analytical solve ──────────────────────────────────────────────
            det_gpu = max.(S_ss_gpu .* S_hh_gpu .- S_sh_gpu.^2, 1f-10)

            B_gpu .= (S_os_gpu .* S_hh_gpu .- S_oh_gpu .* S_sh_gpu) ./ det_gpu
            C_gpu .= (S_oh_gpu .* S_ss_gpu .- S_os_gpu .* S_sh_gpu) ./ det_gpu

            chi_sq_gpu .= S_oo_gpu .+ B_gpu.^2 .* S_ss_gpu .+ C_gpu.^2 .* S_hh_gpu .-
                          2.0f0 .* B_gpu .* S_os_gpu .- 2.0f0 .* C_gpu .* S_oh_gpu .+
                          2.0f0 .* B_gpu .* C_gpu .* S_sh_gpu

            invalid_coeffs = (B_gpu .< 0.0f0) .| (C_gpu .< 0.0f0) .| (N_pts_gpu .<= 0.0f0)
            chi_sq_gpu[invalid_coeffs] .= Float32(Inf)

            chi_sq_red_gpu .= chi_sq_gpu ./ max.(N_pts_gpu, 1.0f0)

            # ── Immediate CPU tracker update for this specific z slice ────────
            chi_pass_vec  = vec(to_cpu(chi_sq_red_gpu))
            npts_pass_vec = vec(to_cpu(N_pts_gpu))

            chi_pass_vec[isnan.(chi_pass_vec)] .= Float32(Inf)

            current_best = @view best_chi_tracker[:, iz]
            improved     = chi_pass_vec .< current_best

            best_chi_tracker[improved, iz]  .= chi_pass_vec[improved]
            best_npts_tracker[improved, iz] .= npts_pass_vec[improved]
            best_av_tracker[improved, iz]   .= Float32(Av)

        end  # ── end Av loop
    end  # ── end z loop

    # ── Reshape best trackers for type-level reduction ────────────────────────
    results_3d  = reshape(best_chi_tracker,  N_T, N_H, N_z)
    npts_3d     = reshape(best_npts_tracker, N_T, N_H, N_z)
    av_3d       = reshape(best_av_tracker,   N_T, N_H, N_z)

    # ── Parallel type-level reduction ─────────────────────────────────────────
    Threads.@threads for t_type_index in 1:n_types
        t_indices_for_type = type_groups[t_type_index]

        best_val = Inf32
        best_t   = 0; best_h = 0; best_iz = 0

        for iz in 1:N_z
            z          = redshift_range[iz]
            N_pts_z    = N_pts_target_z[iz]

            for t_idx in t_indices_for_type
                t_idx in galactic_transients && z != 0.0 && continue

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
            chi_squared_arr[t_type_index]     = best_val * npts_3d[best_t, best_h, best_iz]
            chi_squared_red_arr[t_type_index] = best_val
            best_indices[t_type_index, 1]     = best_t
            best_indices[t_type_index, 2]     = best_h
            best_Av_arr[t_type_index]         = av_3d[best_t, best_h, best_iz]
            redshift_vals[t_type_index]        = redshift_range[best_iz]
            best_z_index[t_type_index]         = best_iz
        end
    end

    # ── Deferred reconstruction ───────────────────────────────────────────────
    for t_type_index in 1:n_types
        t_idx = best_indices[t_type_index, 1]
        h_idx = best_indices[t_type_index, 2]
        t_idx == 0 && continue

        iz    = best_z_index[t_type_index]
        z_val = redshift_vals[t_type_index]
        Av    = best_Av_arr[t_type_index]

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

            ext_curve_win = Float32.(10.0 .^ (-0.4f0 .* Float32(Av) .* Float32.(ext_law.(frozen_λ_win))))
            T_col .= T_col .* ext_curve_win
            H_col .= H_col .* ext_curve_win

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

    top_fit_types    = [transient_template_info["Type"][transient_template_names[i]]  for i in best_indices[fitted_type_indices, 1]]
    top_fit_hg_types = [host_template_info["HG Type"][host_template_names[i]]         for i in best_indices[fitted_type_indices, 2]]
    top_fit_av       = [best_Av_arr[i]                                                 for i in fitted_type_indices]
    top_fit_eb_v     = [av / Rv for av in top_fit_av]
    top_fit_phase    = [transient_template_info["tau"][transient_template_names[i]]   for i in best_indices[fitted_type_indices, 1]]

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

    #If the savepath directory doesn't exist, create it
    if !isdir(savepath)
        mkpath(savepath)
    end

    plotpath = plot_best_fit(
        SN_ID, best_redshift,
        best_indices, transient_template_info, transient_template_names,
        host_template_info, host_template_names,
        original_target_flux, original_target_flux_σ, original_target_λ,
        best_fit_arr, best_fit_λ_arr, transient_component_arr, host_component_arr,
        fitted_spectra_arr, fitted_spectra_σ_arr, fitted_spectra_λ_arr,
        chi_squared_red_arr, redshift_vals,
        emission_lines_arr, emission_lines_λ_arr,
        savepath, show_plot)

    return norm_arr, chi_squared_red_arr, plotpath, best_redshift, best_type
end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_target_spectra, read_template_files, trim_nans, calculate_B, calculate_C, calculate_spectral_error, parse_command_line_args
end