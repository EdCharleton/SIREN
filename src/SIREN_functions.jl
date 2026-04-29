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

    #------------------------------------------------------------
    # Hardware-Accelerated Math Setup
    #------------------------------------------------------------
    # We import at top-level (required by Julia) but check functionality in __init__
    try; import CUDA; catch; end
    try; import Metal; catch; end

    is_cuda_loaded = false
    is_metal_loaded = false
    use_gpu = true

    function __init__()
        global is_cuda_loaded, is_metal_loaded
        
        # Check functionality at runtime
        try
            is_cuda_loaded = CUDA.functional()
        catch
        end

        try
            is_metal_loaded = Metal.functional()
        catch
        end

        println("--- Hardware Setup ---")
        println("CUDA functional: ", is_cuda_loaded)
        println("Metal functional: ", is_metal_loaded)
        println("Processing on: ", is_cuda_loaded ? "NVIDIA GPU" : (is_metal_loaded ? "Apple Metal GPU" : "Native Multi-threaded CPU"))
        println("----------------------")
    end

    function to_gpu(x)
        if use_gpu && is_cuda_loaded
            return CUDA.CuArray(x)
        elseif use_gpu && is_metal_loaded
            # Note: Metal requires Float32. If using Float64, this may fall back to CPU or error.
            return Metal.MtlArray(x)
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
                help = "Path to the input data file"
                arg_type = String
                required = true
            
            "--redshift", "-z"
                help = "Redshift value (or 'None' for search)"
                arg_type = String
                required = true
    
            "--batch", "-b"
                help = "Flag to run in batch mode"
                arg_type = Bool
                default = false
            
            "--remove_telluric"
                help = "Flag to remove telluric absorption"
                arg_type = String
                default = "N"  # Default is 'no'
    
            "--galactic_source"
                help = "Flag indicating whether the source is galactic"
                arg_type = String
                default = "N"  # Default is 'no'
    
            "--make_plot"
                help = "Flag to generate a plot"
                arg_type = Bool
                default = true  # Default is 'yes'
    
            "--minimum_overlap"
                help = "Minimum overlap between the target and templates, units of %"
                arg_type = Float64
                default = 75  # Default is '0.0'

            "--use_gpu"
                help = "Flag to enable/disable GPU acceleration"
                arg_type = Bool
                default = true
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
    savepath = nothing)

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
    # Original unbinned spectrum (observer frame)
    obs_λ    = collect(Float64, original_λ)
    obs_flux = collect(Float64, original_flux)
    obs_err  = collect(Float64, original_flux_σ)

    # Binned + processed spectrum (rest frame → shift back to observer frame)
    binned_λ    = collect(Float64, fitted_spectra_λ_arr[best]) .* (1 + z_best)
    binned_flux = collect(Float64, fitted_spectra_arr[best])
    binned_err  = collect(Float64, fitted_spectra_σ_arr[best])

    # Best fit and components (rest frame → observer frame)
    fit_λ          = collect(Float64, best_fit_λ_arr[best])     .* (1 + z_best)
    best_fit       = collect(Float64, best_fit_arr[best])
    transient_comp = collect(Float64, transient_component_arr[best])
    host_comp      = collect(Float64, host_component_arr[best])

    em_lines   = emission_lines_arr[best]
    em_lines_λ = emission_lines_λ_arr[best]

    display_name = splitext(splitpath(SN_ID)[end])[1]

    # ── Y-axis limits from the unbinned spectrum, ignoring outlier spikes ────
    finite_flux = filter(isfinite, obs_flux)
    y_lo = quantile(finite_flux, 0.01)
    y_hi = quantile(finite_flux, 0.99) * 1.3

    # ── Plot ─────────────────────────────────────────────────────────────────
    p = plot(
        size          = (1600, 900),
        dpi           = 300,
        grid          = false,
        guidefontsize = 22,
        tickfontsize  = 16,
        titlefont     = 22,
        legendfontsize = 16,
        left_margin   = 9Plots.mm,
        bottom_margin = 10Plots.mm,
        top_margin    = 10Plots.mm,
        right_margin  = 4Plots.mm
    )

    # 1. Original unbinned spectrum + error ribbon — light grey fill, mid-grey line
    plot!(p,
        obs_λ, obs_flux,
        ribbon    = min.(obs_err, (y_hi - y_lo) * 0.1),
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

    # 2. Binned + emission-line-removed spectrum — solid black
    plot!(p,
        binned_λ, binned_flux,
        ribbon    = min.(binned_err, (y_hi - y_lo) * 0.1),
        fillalpha = 0.10,
        fillcolor = :black,
        linecolor = :black,
        linewidth = 1.4,
        label     = "Binned + Lines Removed"
    )

    # 3. Best fit — solid red
    plot!(p,
        fit_λ, best_fit,
        linecolor = :red,
        linewidth = 1.8,
        label     = "Best Fit ($transient_type + $host_type)"
    )

    # 4. Transient component — dashed blue
    plot!(p,
        fit_λ, transient_comp,
        linecolor  = :royalblue,
        linewidth  = 1.2,
        linestyle  = :dash,
        label      = "Transient ($transient_type)"
    )

    # 5. Host component — dashed orange
    plot!(p,
        fit_λ, host_comp,
        linecolor  = :darkorange,
        linewidth  = 1.2,
        linestyle  = :dash,
        label      = "Host Galaxy ($host_type)"
    )

    # 6. Emission line markers
    if !isempty(em_lines_λ)
        for (λ_em, name) in zip(em_lines_λ, em_lines)
            vline!(p, [λ_em],
                linecolor = :green,
                linewidth = 0.8,
                linestyle = :dot,
                label     = ""
            )
            annotate!(p, λ_em, y_hi * 0.92,
                text(string(name), :green, :center, 7))
        end
    end

    title!(p, "$display_name  |  z = $z_best  | type = $transient_type |  Phase = $phase days")

    if !isnothing(savepath)
        outpath = joinpath(savepath, "$(display_name)_best_fit.png")
        savefig(p, outpath)
        println("Plot saved to $outpath")
        return outpath
    end

    display(p)
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
    
    function read_target_spectra(directory::String, delimiter::Char = ',', skip_header::Int = 0)
        files_dict = Dict{String, Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}}()
        println("Loading in Target Spectra...")

        skip = skip_header
        filename = directory

        if endswith(filename, ".dat") || endswith(filename, ".txt") || endswith(filename, ".ascii") || endswith(filename, ".csv") || endswith(filename, ".list") || endswith(filename, ".flm") || endswith(filename, ".asci") || endswith(filename, ".clean") || endswith(filename, ".asc") || endswith(filename, ".sed")
            file_path = joinpath(directory)

            imported = false
            while !imported

                if endswith(filename, ".csv")
                    content = CSV.read(file_path, DataFrame)
                else 
                    content = readdlm(file_path, delimiter, skipstart = skip)
                    try
                        spectra = content[:, 2] 
                    catch
                        content, imported = string_import(file_path, delimiter, skip) #Check if the rows are being imported as one big string instead of columns
                    end
                end

                if typeof(content[1,1]) == String || content[1, 1] == 0 || typeof(content[1,1]) == SubString{String}
                    imported = false
                    skip += 1
                else
                    imported = true
                end
                global content
            end
            λ = content[:, 1]
            spectra = content[:, 2]

            spec_min = minimum(spectra)
            if spec_min < 0
                spectra = spectra .+ (2*abs(spec_min))
            end

            no_errs = false

            try
                spectra_σ = content[:, 3]
                if isempty(spectra_σ) || any(isnan.(spectra_σ))
                    no_errs = true
                end
            catch
                no_errs = true
            end

            scaler = maximum(filter(!isnan, spectra)) / 100
            spectra = (spectra ./ scaler)

            if !no_errs
                spectra_σ = (spectra_σ ./ scaler)
            else
                spectra, λ, spectra_σ = calculate_spectral_error(spectra, λ, 27)
            end

            files_dict[filename] = (λ, spectra, spectra_σ)

        elseif endswith(filename, ".fits")
            file_path = joinpath(directory, filename)
            
            f = FITS(file_path)
            spectra_hdr = read_header(f[1])

            spectra = read(f[1])

            try
                spectra_σ = read(f[2])
            catch
                spectra_σ = spectra .* 0.1
            end

            spectra_σ = sqrt.(spectra_σ)
            try
                λ= collect(spectra_hdr["CRVAL1"] - (spectra_hdr["CRPIX1"] * spectra_hdr["CDELT1"]) : spectra_hdr["CDELT1"] : (spectra_hdr["NAXIS1"] - spectra_hdr["CRPIX1"] - 1) * spectra_hdr["CDELT1"] + spectra_hdr["CRVAL1"])
            catch
                λ= collect(spectra_hdr["CRVAL1"] - (spectra_hdr["CRPIX1"] * spectra_hdr["CD1_1"]) : spectra_hdr["CD1_1"] : (spectra_hdr["NAXIS1"] - spectra_hdr["CRPIX1"] - 1) * spectra_hdr["CD1_1"] + spectra_hdr["CRVAL1"])
            end

            
            #remove any nans from the spectra and the corresponding λ
            nans_indices = findall(.!isnan.(spectra))
            spectra = spectra[nans_indices]
            spectra_σ = spectra_σ[nans_indices]
            λ = λ[nans_indices]

            spec_min = minimum(spectra)
            if spec_min < 0
                spectra = spectra .+ abs(spec_min)
            end

            scaler = maximum(spectra) 

            spectra = (spectra ./ scaler) .* 100
            spectra_σ = (spectra_σ ./ scaler) .* 100

            files_dict[filename] = (λ, spectra, spectra_σ)

        else
            println(" ")
            printstyled("ERROR: File type not supported - ", color=:red)
            println("$filename")
            println(" ")
        end

        #end
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
        H_master = nothing, MH_master = nothing)

    use_prebuilt = !isnothing(T_master)

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

    working_flux    = deepcopy(target_spectra)
    working_flux_σ  = deepcopy(target_spectra_σ)
    working_λ       = deepcopy(target_λ)
    original_λ      = deepcopy(target_λ)

    cleaned_λ, cleaned_flux, cleaned_flux_σ, telluric_λ, emission_line_λ, emission_line = process_spectra(
        working_λ, working_flux, working_flux_σ, telluric_λ, remove_telluric
    )

    original_target_λ      = deepcopy(working_λ)
    original_target_flux   = deepcopy(working_flux)
    original_target_flux_σ = deepcopy(working_flux_σ)

    redshift_max = if ismissing(target_redshift) || target_redshift == "None"
        0.6
    else
        target_redshift > 0.6 ? target_redshift : 0.6
    end
    redshift_range = collect(0.00:0.01:redshift_max)

    # Pre-compute galactic transient indices (constant across z)
    galactic_transients = Set(
        t_idx for t_idx in 1:n_transient_templates
        if transient_template_info["Galactic Source"][transient_template_names[t_idx]] == "Y"
    )

    for z in (batch_mode ? redshift_range : ProgressBar(redshift_range))

        if !ismissing(target_redshift) && target_redshift != "None"
            round(z, digits=2) != round(target_redshift, digits=2) && continue
        end

        if use_prebuilt
            # ── FAST PATH: slice pre-built master matrices ────────────────────
            bottom_λ = 3800.0f0 / (1.0f0 + Float32(z))
            top_λ    = 8000.0f0 / (1.0f0 + Float32(z))
            row_start = max(1, round(Int, (bottom_λ - 2315.0f0) / 15.0f0) + 1)
            row_end   = min(length(master_grid), round(Int, (top_λ - 2315.0f0) / 15.0f0) + 1)
            active_grid = master_grid[row_start:row_end]
            N_lam = length(active_grid)

            de_redshifted_λ = original_λ ./ (1 + z)
            target_on_grid, target_σ_on_grid = spectral_resampler(
                active_grid, de_redshifted_λ, cleaned_flux, cleaned_flux_σ
            )

            binned_flux   = copy(target_on_grid)
            binned_flux_σ = copy(target_σ_on_grid)
            binned_λ      = copy(active_grid)

            N_T = size(T_master, 2)
            N_H = size(H_master, 2)
            target_w = zeros(Float32, N_lam)
            target_f = zeros(Float32, N_lam)
            target_valid = .!isnan.(target_on_grid) .& .!isnan.(target_σ_on_grid) .& (target_σ_on_grid .> 0)
            target_w[target_valid] .= 1.0f0 ./ Float32.(target_σ_on_grid[target_valid]).^2
            target_f[target_valid] .= Float32.(target_on_grid[target_valid])

            T_mat  = T_master[row_start:row_end, :]
            MT_mat = MT_master[row_start:row_end, :]
            H_mat  = H_master[row_start:row_end, :]
            MH_mat = MH_master[row_start:row_end, :]

            # Zero out galactic transients at z != 0
            if z != 0.0
                for t_idx in galactic_transients
                    T_mat[:, t_idx]  .= 0.0f0
                    MT_mat[:, t_idx] .= 0.0f0
                end
            end

            # Zero out template rows where the target has no valid data.
            # This matches the legacy path where trim_nans removed NaN rows,
            # ensuring N_pts only counts wavelengths with valid target + template data.
            invalid_target = .!target_valid
            T_mat[invalid_target, :]  .= 0.0f0
            MT_mat[invalid_target, :] .= 0.0f0
            H_mat[invalid_target, :]  .= 0.0f0
            MH_mat[invalid_target, :] .= 0.0f0

            # Compute overlap and exclude templates with insufficient coverage
            # This matches legacy behavior where templates with < 50% overlap were skipped.
            # Denominator should be the number of valid target points.
            N_target_pts = sum(target_valid)
            overlap_counts = sum(MT_mat, dims=1) # Count of valid pixels per template
            insufficient_overlap = (overlap_counts ./ max(N_target_pts, 1)) .< (minimum_overlap / 100)
            
            # We'll apply this mask to T_mat later or just set chi_sq to Inf
            # Actually, let's pass it to the GPU part via MT_mat
            MT_mat[:, vec(insufficient_overlap)] .= 0.0f0

            frozen_flux   = target_f # Use the sanitized vectors
            frozen_flux_σ = target_σ_on_grid # Still has NaNs but we use target_w
            frozen_λ      = active_grid

        else
            # ── LEGACY PATH: build matrices from template arrays ──────────────
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
            N_T = length(transient_templates)
            N_H = length(host_templates)

            target_w = zeros(Float32, N_lam)
            target_f = zeros(Float32, N_lam)
            target_valid = .!isnan.(frozen_flux) .& .!isnan.(frozen_flux_σ) .& (frozen_flux_σ .> 0)
            target_w[target_valid] .= 1.0f0 ./ (frozen_flux_σ[target_valid].^2)
            target_f[target_valid] .= frozen_flux[target_valid]

            T_mat = zeros(Float32, N_lam, N_T)
            MT_mat = zeros(Float32, N_lam, N_T)
            H_mat = zeros(Float32, N_lam, N_H)
            MH_mat = zeros(Float32, N_lam, N_H)

            for h_idx in 1:N_H
                if h_idx in non_standard_template_indices
                    common_set = intersect(Set(host_template_λ_z[h_idx]), frozen_λ_set)
                    common_idx_h = findall(in(common_set), host_template_λ_z[h_idx])
                    common_idx_z = findall(in(common_set), frozen_λ)
                    H_mat[common_idx_z, h_idx] .= host_templates[h_idx][common_idx_h]
                    MH_mat[common_idx_z, h_idx] .= 1.0f0
                else
                    H_mat[:, h_idx] .= host_templates[h_idx]
                    MH_mat[:, h_idx] .= 1.0f0
                end
                nan_mask = isnan.(H_mat[:, h_idx])
                if any(nan_mask)
                    H_mat[nan_mask, h_idx] .= 0.0f0
                    MH_mat[nan_mask, h_idx] .= 0.0f0
                end
            end

            for t_idx in 1:N_T
                if z != 0.0 && t_idx in galactic_transients
                    continue
                end
                if t_idx in non_standard_template_indices
                    common_set = intersect(Set(transient_template_λ_z[t_idx]), frozen_λ_set)
                    if (length(common_set) / N_lam) < (minimum_overlap / 100)
                        continue
                    end
                    common_idx_t = findall(in(common_set), transient_template_λ_z[t_idx])
                    common_idx_z = findall(in(common_set), frozen_λ)
                    T_mat[common_idx_z, t_idx] .= transient_templates[t_idx][common_idx_t]
                    MT_mat[common_idx_z, t_idx] .= 1.0f0
                else
                    T_mat[:, t_idx] .= transient_templates[t_idx]
                    MT_mat[:, t_idx] .= 1.0f0
                end
                nan_mask = isnan.(T_mat[:, t_idx])
                if any(nan_mask)
                    T_mat[nan_mask, t_idx] .= 0.0f0
                    MT_mat[nan_mask, t_idx] .= 0.0f0
                end
            end
        end
        
        # Optional offload to GPU if functional (otherwise native dense matrix runs locally)
        Y_gpu  = to_gpu(target_f)
        W_gpu  = to_gpu(target_w)
        T_gpu  = to_gpu(T_mat)
        MT_gpu = to_gpu(MT_mat)
        H_gpu  = to_gpu(H_mat)
        MH_gpu = to_gpu(MH_mat)
        
        # Prepare components analytically
        WT = W_gpu .* MT_gpu                  
        WT_Y = WT .* Y_gpu                    
        WT_T = WT .* T_gpu                    
        WT_T2 = WT_T .* T_gpu                 
        H_clean = H_gpu .* MH_gpu             

        # Executing primary Matrix operations via BLAS GEMM directly inside hardware backend!
        S_sh = transpose(WT_T) * H_clean      
        S_os = transpose(WT_T .* Y_gpu) * MH_gpu       
        S_oh = transpose(WT) * (H_clean .* Y_gpu) 
        S_ss = transpose(WT_T2) * MH_gpu      
        S_hh = transpose(WT) * (H_clean .* H_clean) 
        S_oo = transpose(WT .* (Y_gpu.^2)) * MH_gpu   
        N_pts = transpose(MT_gpu) * MH_gpu    

        # Mathematical Determinants (protecting from div-by-zero singularities gracefully)
        det = max.(S_ss .* S_hh .- S_sh.^2, 1f-10) 
        
        B = (S_os .* S_hh .- S_oh .* S_sh) ./ det
        C = (S_oh .* S_ss .- S_os .* S_sh) ./ det

        chi_sq = S_oo .+ B.^2 .* S_ss .+ C.^2 .* S_hh .- 2.0f0 .* B .* S_os .- 2.0f0 .* C .* S_oh .+ 2.0f0 .* B .* C .* S_sh
        chi_sq_red = chi_sq ./ max.(N_pts, 1.0f0)
        
        # Return exact coefficients matching matrix arrays locally 
        B_cpu = to_cpu(B)
        C_cpu = to_cpu(C)
        chi_sq_red_cpu = to_cpu(chi_sq_red)
        N_pts_cpu = to_cpu(N_pts)
        
        valid_BC = (B_cpu .>= 0.0) .& (C_cpu .>= 0.0) .& (N_pts_cpu .> 0.0)
        chi_sq_red_cpu[.!valid_BC] .= Inf

        type_indices = [type_to_index[transient_template_info["Type"][transient_template_names[i]]] for i in 1:N_T]
        type_locks = [ReentrantLock() for _ in 1:n_types]

        # Thread-safe assignments since matrix math operates over all variables jointly
        Threads.@threads for t_idx in 1:N_T
            t_idx in galactic_transients && z != 0.0 && continue
            
            t_type_index = type_indices[t_idx]
            
            for h_idx in 1:N_H
                c_val = chi_sq_red_cpu[t_idx, h_idx]
                
                # B and C failed bounds (set to inf) or was worse model fit
                if c_val >= chi_squared_red_arr[t_type_index]
                    continue
                end
                
                lock(type_locks[t_type_index]) do
                    if c_val < chi_squared_red_arr[t_type_index]
                        
                        norm_transient = B_cpu[t_idx, h_idx]
                        norm_host = C_cpu[t_idx, h_idx]
                        
                        chi_squared_arr[t_type_index]     = c_val * N_pts_cpu[t_idx, h_idx]
                        chi_squared_red_arr[t_type_index] = c_val
                        norm_arr[t_type_index, 1]         = norm_transient
                        norm_arr[t_type_index, 2]         = norm_host
                        best_indices[t_type_index, 1]     = t_idx
                        best_indices[t_type_index, 2]     = h_idx
                        redshift_vals[t_type_index]       = z

                        # Reconstruct precisely to output metrics
                        valid_mask = (target_w .> 0) .& (MT_mat[:, t_idx] .> 0) .& (MH_mat[:, h_idx] .> 0)
                        
                        b_target_v = target_f[valid_mask]
                        b_target_σ_v = frozen_flux_σ[valid_mask]
                        b_transient_v = T_mat[valid_mask, t_idx]
                        b_host_v = H_mat[valid_mask, h_idx]
                        b_host_λ = frozen_λ[valid_mask]
                        
                        best_f = (b_transient_v .* norm_transient) .+ (b_host_v .* norm_host)
                        
                        host_flux_sum = sum(b_host_v .* norm_host)
                        total_flux = sum(best_f)
                        contam_array[t_type_index] = (host_flux_sum / total_flux) * 100

                        emission_lines_arr[t_type_index]   = collect(Float64, emission_line)
                        emission_lines_λ_arr[t_type_index] = collect(Float64, emission_line_λ)

                        binned_spectra_arr[t_type_index]   = collect(Float64, binned_flux)
                        binned_λ_arr[t_type_index]         = collect(Float64, binned_λ)
                        binned_spectra_σ_arr[t_type_index] = collect(Float64, binned_flux_σ)

                        best_fit_arr[t_type_index]            = collect(Float64, best_f)
                        best_fit_λ_arr[t_type_index]          = collect(Float64, b_host_λ)
                        fitted_spectra_arr[t_type_index]      = collect(Float64, b_target_v)
                        fitted_spectra_σ_arr[t_type_index]    = collect(Float64, b_target_σ_v)
                        fitted_spectra_λ_arr[t_type_index]    = collect(Float64, b_host_λ)
                        transient_component_arr[t_type_index] = collect(Float64, b_transient_v .* norm_transient)
                        host_component_arr[t_type_index]      = collect(Float64, b_host_v .* norm_host)
                    end
                end
            end
        end
    end

    # ── Collect results ───────────────────────────────────────────────────────
    fitted_type_indices = findall(x -> x != 0, best_indices[:, 1])

    top_fit_types    = [transient_template_info["Type"][transient_template_names[i]]    for i in best_indices[fitted_type_indices, 1]]
    top_fit_hg_types = [host_template_info["HG Type"][host_template_names[i]]          for i in best_indices[fitted_type_indices, 2]]
    top_fit_av       = [transient_template_info["SN A_v"][transient_template_names[i]] for i in best_indices[fitted_type_indices, 1]]
    top_fit_eb_v     = [av / 3.1 for av in top_fit_av]
    top_fit_phase    = [transient_template_info["tau"][transient_template_names[i]]    for i in best_indices[fitted_type_indices, 1]]

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

    savepath = "/Users/chris/Documents/siren_plots"

    plotpath = show_plot ? plot_best_fit(
        SN_ID,                          # e.g. "2024abc.fits" or full path
        best_redshift,                  # Float from redshift_vals[best_type_full_idx]

        # Template index array and metadata
        best_indices,                   # n_types × 2 Int array of [t_idx, h_idx]
        transient_template_info,        # OrderedDict with "Type", "tau", "SN A_v", etc.
        transient_template_names,       # Vector of template filenames
        host_template_info,             # OrderedDict with "HG Type" etc.
        host_template_names,            # Vector of host filenames

        # Original unbinned target spectrum (before any processing)
        original_target_flux,           # Vector{Float32} — raw observed flux
        original_target_flux_σ,         # Vector{Float32} — raw observed uncertainty
        original_target_λ,              # Vector{Float32} — raw observed wavelengths

        # Fit results stored per best type
        best_fit_arr,                   # Vector{Array} — best fit flux per type
        best_fit_λ_arr,                 # Vector{Array} — wavelengths of best fit
        transient_component_arr,        # Vector{Array} — transient-only contribution
        host_component_arr,             # Vector{Array} — host-only contribution

        # Binned + processed spectrum (what was actually fit)
        fitted_spectra_arr,             # Vector{Array} — binned flux in fit region
        fitted_spectra_σ_arr,           # Vector{Array} — binned uncertainty in fit region
        fitted_spectra_λ_arr,           # Vector{Array} — rest-frame λ of fit region

        # Diagnostics
        chi_squared_red_arr,            # Vector{Float64} — reduced χ² per type
        redshift_vals,                  # Vector{Float32} — best redshift per type

        # Emission lines that were masked out
        emission_lines_arr,             # Vector{Array} — masked flux values
        emission_lines_λ_arr,           # Vector{Array} — masked wavelengths

        savepath                        # String path to save dir, or nothing to display
    ) : "No Plot Saved"

    return norm_arr, chi_squared_red_arr, plotpath, best_redshift, best_type
end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_target_spectra, read_template_files, trim_nans, calculate_B, calculate_C, calculate_spectral_error, parse_command_line_args
end