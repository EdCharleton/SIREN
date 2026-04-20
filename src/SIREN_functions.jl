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
    #Define Functions
    #------------------------------------------------------------

    #This section defines the functions used in the main script

    function parse_command_line_args()
        #Get the arguments when running in command line mode
        s = ArgParseSettings()
        @add_arg_table! s begin
            "file_path"
                help = "Path to the input data file"
                arg_type = String
                required = true
            
            "redshift"
                help = "Redshift value"
                arg_type = Float64
                required = true
    
            "batch_mode"
                help = "Flag to run in batch mode"
                arg_type = Bool
                required = true
                default = false  # Default is 'no'
            
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
    indexes, pure_sn_spectra_info, pure_sn_spectra_file_names,
    pure_hg_spectra_info, pure_hg_spectra_file_names,
    original_spectrum, original_spectrum_σ, original_spectrum_λ,
    best_fit_arr, best_fit_λ_arr, sn_component_arr, hg_component_arr,
    binned_spectra_arr, binned_spectra_error, binned_λ_arr,
    chi_squared_red_arr, redshift_vals,
    emission_lines, emission_lines_λ,
    savepath = nothing)

        best = argmin(chi_squared_red_arr)

        if indexes[best, 1] == 0
            println("--------------------")
            printstyled("ERROR: ", color=:red)
            println("plot_best_fit: No valid fit found for $SN_ID")
            println("--------------------")
            return nothing
        end

        sn_idx = indexes[best, 1]
        hg_idx = indexes[best, 2]

        sn_type = pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[sn_idx]]
        hg_type = pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[hg_idx]]
        phase   = get(pure_sn_spectra_info["tau"], pure_sn_spectra_file_names[sn_idx], "N/A")
        chi_red = round(chi_squared_red_arr[best], digits = 3)
        z       = round(Float64(redshift_vals[best]), digits = 3)

        obs_flux    = collect(Float64, original_spectrum)
        obs_err     = collect(Float64, original_spectrum_σ)
        obs_λ       = collect(Float64, original_spectrum_λ)
        best_fit    = collect(Float64, best_fit_arr[best])
        fit_λ       = collect(Float64, best_fit_λ_arr[best])   # own λ axis
        sn_comp     = collect(Float64, sn_component_arr[best])
        hg_comp     = collect(Float64, hg_component_arr[best])
        binned_flux = collect(Float64, binned_spectra_arr[best])
        binned_λ    = collect(Float64, binned_λ_arr[best])
        em_lines    = emission_lines[best]
        em_lines_λ  = emission_lines_λ[best]

        display_name = splitext(splitpath(SN_ID)[end])[1]

        # ── Sensible y-axis limits ignoring outlier spikes ────────────────────────
        finite_flux = filter(isfinite, obs_flux)
        y_lo = quantile(finite_flux, 0.01)
        y_hi = quantile(finite_flux, 0.99) * 1.3

        p = plot(
            layout = (1, 1),
            size = (1600, 900),
            dpi = 300,
            grid = false,
            guidefontsize = 22,
            tickfontsize = 16,
            titlefont = 22,
            legendfontsize = 16,
            left_margin = 9Plots.mm,
            bottom_margin = 10Plots.mm,
            top_margin = 10Plots.mm,
            right_margin = 4Plots.mm
        )

        # Observed spectrum — clipped ribbon so spikes don't dominate
        plot!(p[1],
            obs_λ, obs_flux,
            ribbon = min.(obs_err, (y_hi - y_lo) * 0.1),   # cap ribbon height
            fillalpha = 0.2,
            fillcolor = :grey,
            linecolor = "#8E8E8E",
            linewidth = 1.2,
            label = "Observed ($display_name)",
            xlabel = "Wavelength (Å)",
            ylabel = "Flux (arb. units)",
            legend = :topright,
            ylims = (y_lo, y_hi)
        )

        # Binned spectrum
        plot!(p[1],
            binned_λ .* (1+z), binned_flux,
            linewidth = 1,
            linecolor = :black,
            label = "Binned + Processed"
        )

        # Best fit — uses its own λ axis so it covers exactly where it was computed
        plot!(p[1],
            fit_λ .* (1+z), best_fit,
            linecolor = :red,
            linewidth = 1.5,
            label = "Best Fit ($sn_type + $hg_type)"
        )

        plot!(p[1],
            fit_λ .* (1+z), sn_comp,
            linecolor = :blue,
            linewidth = 1.0,
            linestyle = :dash,
            label = "SN ($sn_type)"
        )

        plot!(p[1],
            fit_λ .* (1+z), hg_comp,
            linecolor = :orange,
            linewidth = 1.0,
            linestyle = :dash,
            label = "Host Galaxy ($hg_type)"
        )

        if !isempty(em_lines_λ)
            for (λ_em, name) in zip(em_lines_λ, em_lines)
                vline!(p[1], [λ_em],
                    linecolor = :green,
                    linewidth = 0.8,
                    linestyle = :dot,
                    label = ""
                )
                annotate!(p[1], λ_em, y_hi * 0.92,
                    text(string(name), :green, :center, 7))
            end
        end

        title!(p[1], "$display_name  |  z = $z  |  Phase = $phase days  |  χ²ᵣ = $chi_red")

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
        flux = Vector{Array{Any}}(undef, length(template_file_names))
        λ = Vector{Array{Any}}(undef, length(template_file_names))
        #masks only present on host galaxy templates (SIREN doesnt fit any host templates with emission lines so they're masked out)
        if label == "Hg Template"
            mask = Vector{Array{Any}}(undef, length(template_file_names))
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

        for iter in 1:10
            n       = length(input_spectra)
            flagged = fill(false, n)

            for i in 1:n
                lo = max(1, i - half)
                hi = min(n, i + half)
                window_flux = input_spectra[lo:hi]

                win_median = median(window_flux)
                win_mad    = median(abs.(window_flux .- win_median))
                win_noise  = 1.4826 * win_mad

                if win_noise > 0 && input_spectra[i] > win_median + σ * win_noise
                    flagged[i] = true
                end
            end

            keep_index = findall(.!flagged)
            sum(flagged) == 0 && break

            input_spectra   = input_spectra[keep_index]
            input_spectra_σ = input_spectra_σ[keep_index]
            input_spectra_λ = input_spectra_λ[keep_index]
        end

        missing_λ       = setdiff(original_spectra_λ, input_spectra_λ)
        missing_indices = findall(x -> x in missing_λ, original_spectra_λ)

        output_spectra   = original_spectra
        output_spectra_σ = original_spectra_σ
        output_spectra_λ = original_spectra_λ

        output_spectra[missing_indices]   .= NaN
        output_spectra_σ[missing_indices] .= NaN
        output_spectra_λ[missing_indices] .= NaN

        return original_spectra_λ[missing_indices], original_spectra[missing_indices],
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
    

    function redshift_spectra(input_spectra_λ, input_spectra, input_spectra_σ, input_transient_templates, input_transient_λ, transient_metadata, transient_file_names, input_host_templates, redshift,  input_host_λ, input_host_mask, original_spectra_λ)

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

        binned_spectra = deepcopy(output_spectra)
        binned_spectra_σ = deepcopy(output_spectra_σ)
        binned_spectra_λ = deepcopy(full_template_λ)

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
    
        output_transient_templates = [input_transient_templates[i][template_indicies] for i in standard_template_indices]
        output_host_templates = [input_host_templates[i][template_indicies] for i in eachindex(input_host_templates)]
    
        output_host_λ = [input_host_λ[i][template_indicies] for i in eachindex(input_host_λ)]
        output_host_mask = [input_host_mask[i][template_indicies] for i in eachindex(input_host_mask)]

        output_transient_λ = Array{Any}(undef, length(input_transient_λ))
        for λ in eachindex(input_transient_λ)
            if λ in standard_template_indices
                output_transient_λ[λ] = input_transient_λ[λ][template_indicies]
            else
                output_transient_λ[λ] = input_transient_λ[λ]
            end
        end

        return output_spectra, output_spectra_σ, output_spectra_λ, output_transient_templates, 
        output_host_templates, output_transient_λ, output_host_λ, standard_template_indices, non_standard_template_indices,
        binned_spectra, binned_spectra_σ, binned_spectra_λ, output_host_mask
        
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
function chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, 
    pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, hg_mask, sn_template_λ, 
    hg_template_λ, altered_spectra, target_λ, altered_spectra_σ, SN_ID, target_redshift, telluric_λ, 
    batch_mode, show_plot = false, remove_telluric = "N", minimum_overlap = 50)



    # ---- Begin fit_spectrum logic ----
    num_sn = length(pure_sn_spectra_files)
    num_hg = length(pure_hg_spectra_files)

    number_of_fits = length(unique(values(pure_sn_spectra_info["Type"])))
    unique_types = unique(values(pure_sn_spectra_info["Type"]))

    #Create an empty array to populate with chi-squared vals for each fit, set to a high number to replace
    chi_squared_arr = collect(1:1:number_of_fits) .* 10.0^10
    chi_squared_red_arr = collect(1:1:number_of_fits) .* 10.0^10

    norm_arr = zeros(Float32, number_of_fits, 2)
    redshift_vals = zeros(Float32, number_of_fits)
    indexes = zeros(Int, number_of_fits, 2)
    emission_lines = Vector{Array}(undef, number_of_fits)
    emission_lines_λ = Vector{Array}(undef, number_of_fits)

    binned_spectra_arr = Vector{Array}(undef, number_of_fits)
    binned_λ_arr = Vector{Array}(undef, number_of_fits)
    binned_spectra_σ_arr = Vector{Array}(undef, number_of_fits)
    best_fit_arr = Vector{Array}(undef, number_of_fits)
    best_fit_λ_arr = Vector{Array}(undef, number_of_fits)
    sn_component_arr = Vector{Array}(undef, number_of_fits)
    hg_component_arr = Vector{Array}(undef, number_of_fits)
    spectra_fit_region = Vector{Array}(undef, number_of_fits)
    spectra_σ_fit_region = Vector{Array}(undef, number_of_fits)
    spectra_λ_fit_region = Vector{Array}(undef, number_of_fits)
    contam_array = Vector{Float32}(undef, number_of_fits)

    spectra_to_fit = deepcopy(altered_spectra)
    spectra_to_fit_σ = deepcopy(altered_spectra_σ)
    spectra_λ = deepcopy(target_λ)
    original_λ = deepcopy(target_λ)

    println("Pre-Processing λ")
    println(spectra_λ[1], " ",spectra_λ[end])
    println(length(spectra_λ))

    #Removes any emission lines from the spectrum and teluric features 
    processed_spectra_λ, processed_spectra, processed_spectra_σ, telluric_λ, emission_line_λ, emission_line = process_spectra(
        spectra_λ,
        spectra_to_fit,
        spectra_to_fit_σ,
        telluric_λ,
        remove_telluric
    )

    #Create a temp plot in diagnostics
    plot(spectra_λ, spectra_to_fit, label = "Original Spectra", color = :red, linewidth = 1.5, dpi=300, ylims = (0,15), grid = false)
    plot!(processed_spectra_λ, processed_spectra, label = "Processed Spectra", color = :black, linewidth = 1.5)
    savefig("Development/diagnostic_plots/emission_line_removal.png")

    original_specta_to_fit_λ = deepcopy(spectra_λ)
    original_spectra_to_fit = deepcopy(spectra_to_fit)
    original_spectra_to_fit_σ = deepcopy(spectra_to_fit_σ)


    println("Post-Processed λ")
    println(processed_spectra_λ[1], " ", processed_spectra_λ[end])
    println(length(processed_spectra_λ))

    redshift_max = 0.8
    if ismissing(target_redshift) || target_redshift == "None"
        redshift_max = 0.6
    else
        redshift_max = target_redshift > 0.6 ? target_redshift : 0.6
    end

    redshift_range = collect(0.00:0.01:redshift_max)

    for z in (batch_mode ? redshift_range : ProgressBar(redshift_range))

        spectra_to_fit = deepcopy(processed_spectra)
        spectra_to_fit_σ = deepcopy(processed_spectra_σ)
        spectra_to_fit_λ = deepcopy(processed_spectra_λ)

        if !ismissing(target_redshift) && target_redshift != "None"
            round(z, digits=2) != round(target_redshift, digits=2) && continue
        end

        

        spectra_to_fit, spectra_to_fit_σ, spectra_to_fit_λ, pure_sn_spectra, pure_hg_spectra, sn_λ, hg_λ,
        standard_template_indices, non_standard_template_indices, binned_spectra, binned_spectra_σ, binned_spectra_λ, hg_mask = redshift_spectra(
            spectra_to_fit_λ,
            spectra_to_fit,
            spectra_to_fit_σ,
            pure_sn_spectra_files,
            sn_template_λ,
            pure_sn_spectra_info,
            pure_sn_spectra_file_names,
            pure_hg_spectra_files,
            z,
            hg_template_λ, hg_mask, original_λ)

        plot((original_specta_to_fit_λ ./ (1+z)), original_spectra_to_fit, label = "Original Spectra", color = :grey, linewidth = 0.8, dpi=300, ylims = (0,15), grid = false)
        plot!(spectra_to_fit_λ, spectra_to_fit, label = "Redshifted + Binned + Lines Removed", color = :black, linewidth = 1.5)
        plot!(binned_spectra_λ, binned_spectra, label = "Nans Removed", color = :orange, linewidth = 1)
        savefig("Development/diagnostic_plots/redshifted_spectra.png")

        println("Redshifted λ")
        println(spectra_to_fit_λ[1], " ", spectra_to_fit_λ[end])
        println(length(spectra_to_fit_λ))
        println("END")

        

        deep_copy_spectra = deepcopy(spectra_to_fit)
        deep_copy_spectra_σ = deepcopy(spectra_to_fit_σ)
        deep_copy_λ = deepcopy(spectra_to_fit_λ)

        # ── Precompute outside both loops ────────────────────────────────────────────

        # Avoid dict lookup + findall on every iteration
        type_to_index = Dict(t => i for (i, t) in enumerate(unique_types))

        # Convert template wavelength to a Set once for O(1) lookup
        full_template_λ_set = Set(deep_copy_λ)

        # Precompute valid_indices per hg template — these don't depend on sn_i at all
        hg_valid_indices = [findall(==(1), hg_mask[hg_i_idx]) for hg_i_idx in 1:num_hg]

        # Precompute which sn templates are galactic sources to avoid dict lookup in loop
        galactic_sn = Set(
            sn_i_idx for sn_i_idx in 1:num_sn
            if pure_sn_spectra_info["Galactic Source"][pure_sn_spectra_file_names[sn_i_idx]] == "Y"
        )


        # ── Inner loop replacement ────────────────────────────────────────────────────

        for sn_i_idx in ProgressBar(1:num_sn)

            if sn_i_idx in galactic_sn && z != 0.0
                continue
            end
            type = pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[sn_i_idx]]

            #Trim the main spectra if there are any non standard templates

            if sn_i_idx in non_standard_template_indices
                sn_i_λ_full = sn_λ[sn_i_idx]

                # Set-based intersection: O(n) instead of O(n²)
                sn_i_λ_set = Set(sn_i_λ_full)
                common_set = intersect(sn_i_λ_set, full_template_λ_set)

                if (length(common_set)/length(deep_copy_spectra)) < (minimum_overlap/100)
                    continue
                end

                common_indices_sn       = findall(in(common_set), sn_i_λ_full)
                common_indices_template = findall(in(common_set), deep_copy_λ)

                sn_i = pure_sn_spectra_files[sn_i_idx][common_indices_sn]
                sn_i_λ  = sn_i_λ_full[common_indices_sn]

                spectra_to_fit  = deep_copy_spectra[common_indices_template]
                spectra_to_fit_σ = deep_copy_spectra_σ[common_indices_template]

                # Precompute sn_i_λ as Set once for hg intersection below
                sn_i_λ_set_trimmed = Set(sn_i_λ)
            else
                sn_i            = pure_sn_spectra[sn_i_idx]
                sn_i_λ          = sn_λ[sn_i_idx]
                spectra_to_fit   = deep_copy_spectra
                spectra_to_fit_σ = deep_copy_spectra_σ
                sn_i_λ_set_trimmed = nothing
            end

            # Precompute type index once per sn template
            type  = pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[sn_i_idx]]
            index = type_to_index[type]

            is_non_standard = sn_i_idx in non_standard_template_indices

            for hg_i_idx in 1:num_hg
                valid_idx = hg_valid_indices[hg_i_idx]

                if is_non_standard
                    hg_i_λ_full = hg_λ[hg_i_idx]

                    # O(n) Set intersection instead of O(n²)
                    common_indices_hg = findall(in(sn_i_λ_set_trimmed), hg_i_λ_full)

                    if (length(common_indices_hg)/length(deep_copy_spectra)) < (minimum_overlap/100)
                        continue
                    end

                    hg_i     = pure_hg_spectra[hg_i_idx][common_indices_hg]
                    hg_i_λ   = hg_i_λ_full[common_indices_hg]
                    hg_i_mask = hg_mask[hg_i_idx][common_indices_hg]

                    # Recompute valid_indices for this trimmed mask
                    valid_idx = findall(==(1), hg_i_mask)
                else
                    hg_i     = pure_hg_spectra[hg_i_idx]
                    hg_i_λ   = hg_λ[hg_i_idx]
                end

                spec_v   = view(spectra_to_fit,   valid_idx)
                spec_σ_v = view(spectra_to_fit_σ, valid_idx)
                λ_region = view(hg_i_λ,           valid_idx)
                sn_v     = view(sn_i,             valid_idx)
                hg_v     = view(hg_i,             valid_idx)


                norm_sn = calculate_B(spec_v, sn_v, hg_v, spec_σ_v)
                norm_hg = calculate_C(spec_v, sn_v, hg_v, spec_σ_v)

                template_best_fit = (sn_v .* norm_sn) .+ (hg_v .* norm_hg)

                chi_squared     = sum((spec_v .- template_best_fit).^2 ./ spec_σ_v.^2)
                if norm_sn < 0 || norm_hg < 0
                    continue
                end
                chi_squared_red = chi_squared / length(spec_v)

                
                index = findall(x -> x == type, unique_types)[1]

                if chi_squared_red < chi_squared_red_arr[index]
                    chi_squared_arr[index] = chi_squared
                    chi_squared_red_arr[index] = chi_squared_red
                    #Store the A and B normalisation parameters
                    norm_arr[index, 1] = norm_sn
                    norm_arr[index, 2] = norm_hg
                    indexes[index, 1] = sn_i_idx
                    indexes[index, 2] = hg_i_idx
                    redshift_vals[index] = z

                    contamination = ((sum(hg_i .* norm_hg)) / (sum((sn_i .* norm_sn) .+ (hg_i .* norm_hg)))) * 100
                    contam_array[index] = contamination

                    #Any masked out emission lines
                    emission_lines[index] = emission_line
                    emission_lines_λ[index] = emission_line_λ

                    binned_spectra_arr[index] = binned_spectra
                    binned_λ_arr[index] = binned_spectra_λ
                    binned_spectra_σ_arr[index] = binned_spectra_σ

                    best_fit_arr[index]       = collect(template_best_fit)
                    best_fit_λ_arr[index]     = collect(λ_region)
                    spectra_fit_region[index] = collect(spec_v)
                    spectra_σ_fit_region[index] = collect(spec_σ_v)
                    spectra_λ_fit_region[index] = collect((λ_region))
                    sn_component_arr[index]   = collect(sn_v .* norm_sn)
                    hg_component_arr[index]   = collect(hg_v .* norm_hg)


                end
            end
        end
    end

    # ---- End fit_spectrum logic ----
    fits = findall(x -> x != 0, indexes[:, 1])

    top_fits_sn_type = [pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[i]] for i in indexes[fits, 1]]
    top_fits_hg_type = [pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[i]] for i in indexes[fits, 2]]
    top_fits_sn_a_v  = [pure_sn_spectra_info["SN A_v"][pure_sn_spectra_file_names[i]] for i in indexes[fits, 1]]
    top_fits_Eb_v    = [i / 3.1 for i in top_fits_sn_a_v]
    top_fits_phase   = [pure_sn_spectra_info["tau"][pure_sn_spectra_file_names[i]] for i in indexes[fits, 1]]

    # best_chi from argmin is an index into the full-length chi_squared_red_arr.
    # Find where that sits within the `fits` subset for the type arrays.
    best_chi_full  = argmin(chi_squared_red_arr)
    best_chi       = findfirst(==(best_chi_full), fits)  # index into top_fits_* arrays
    best_redshift  = redshift_vals[best_chi]

    if isempty(top_fits_sn_type) || isnothing(best_chi)
        best_type = "Unknown"
        println("--------------------")
        printstyled("ERROR: ", color=:red)
        println("Fitting $SN_ID - No fitting results found.")
        println("--------------------")
    else
        best_type = top_fits_sn_type[best_chi]
    end

    savepath = "/Users/edcharleton/PhD/SIREN/classifications_dir/plots"

    plotpath = show_plot ? plot_best_fit(
        SN_ID, best_redshift,
        indexes, pure_sn_spectra_info, pure_sn_spectra_file_names, #template info
        pure_hg_spectra_info, pure_hg_spectra_file_names, #Template info
        original_spectra_to_fit, original_spectra_to_fit_σ, original_specta_to_fit_λ, #Original spectra with emission lines and telurics
        best_fit_arr, best_fit_λ_arr, sn_component_arr, hg_component_arr,
        spectra_fit_region, spectra_σ_fit_region, spectra_λ_fit_region, #Region of the spectra that was actually fit
        chi_squared_red_arr, redshift_vals,
        emission_lines, emission_lines_λ,
        savepath
    ) : "No Plot Saved"


    return norm_arr, chi_squared_red_arr, plotpath, best_redshift, best_type
end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_target_spectra, read_template_files, trim_nans, calculate_B, calculate_C, calculate_spectral_error, parse_command_line_args
end