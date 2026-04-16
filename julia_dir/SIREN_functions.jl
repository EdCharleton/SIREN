#------------------------------------------------------------
#Import Packages
#------------------------------------------------------------
#Define the code as a module
module ClassifierFunctions

    using Random
    using DataFrames
    using DelimitedFiles
    using Statistics    
    using JSON
    using ProgressBars
    using Pkg
    using PyPlot
    using DustExtinction    
    using BenchmarkTools
    using DataStructures
    using Pkg
    using Interpolations: Linear, Gridded, interpolate
    using LsqFit
    using LaTeXStrings
    using LinearAlgebra
    using ProfileView
    using Pkg
    using Conda
    using CSV
    using FITSIO
    using Distributed
    using SavitzkyGolay
    

    #------------------------------------------------------------
    #Define Functions
    #------------------------------------------------------------

    #This section defines the functions used in the main script

    #------------------ Delete Folder Contents ------------------
    """
        delete_folder_contents(folder_path)

    Function that deletes the contents of a folder.

    # Parameters
    - `folder_path::AbstractString`: Path to the folder.

    # Returns
    - `Nothing`: Nothing is returned.
    """
    function delete_folder_contents(folder_path::AbstractString)
        # Iterate over all items in the folder
        for item in readdir(folder_path)
            item_path = joinpath(folder_path, item)

            # Check if it's a file or a directory
            if isfile(item_path)
                # If it's a file, delete it
                rm(item_path)
            elseif isdir(item_path)
                # If it's a directory, delete it recursively
                rm(item_path, recursive=true)
            end
        end
    end

    #--------------------- Adaptive Binning ---------------------
    """
        adapative_binning(altered_spectra, pure_spectra)
    TBW
    """
    function adaptive_binning(altered_spectra, pure_spectra)
        bin_mask = altered_spectra .< 15

        binned_altered_spectra = [sum(altered_spectra[i:i+1]) for i in 1:2:length(altered_spectra)-1 if i == length(altered_spectra) || !bin_mask[i]]
        binned_pure_spectra = [sum(pure_spectra[i:i+1]) for i in 1:2:length(pure_spectra)-1 if i == length(pure_spectra) || !bin_mask[i]]
        return binned_altered_spectra, binned_pure_spectra
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
        smoothed_spectra_sg = savitzky_golay(spectra, window_size, 2)
        smoothed_spectra = smoothed_spectra_sg.y
        residual = abs.(spectra .- smoothed_spectra)
    
        start_end_window = Int((window_size / 2) - 0.5)
    
        new_spectra = zeros(Float16, Int(length(spectra) - 2 * start_end_window))
        new_spectra_λ = zeros(Float16, Int(length(spectra_λ) - 2 * start_end_window))
        new_sigma = zeros(Float16, Int(length(residual) - 2 * start_end_window))
        new_smoothed_spectra = zeros(Float16, Int(length(smoothed_spectra) - 2 * start_end_window))
    
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

        #Using PyPlot display the spectra and error
        PyPlot.figure(dpi = 400)
        plot(spectra_λ, spectra, label = "Spectra", color = "black", zorder=1, alpha = 0.5)
        plot(spectra_λ, smoothed_spectra, label = "Smoothed Spectra", color = "red")
        fill_between(new_spectra_λ, new_smoothed_spectra .- new_sigma.*2, new_smoothed_spectra .+ new_sigma.*2, alpha = 0.5, label = "Error", color = "red")
        savefig("spectra_error.png")
        close()
    
        return new_smoothed_spectra, new_spectra_λ, new_sigma
    end
    
    function read_target_spectra(directory::String, delimiter::Char = ',', skip_header::Int = 0)
        files_dict = Dict{String, Tuple{Vector{Float16}, Vector{Float16}, Vector{Float16}}}()
        println("Loading in Target Spectra...")

        for filename in ProgressBar(readdir(directory))
            skip = skip_header
            #Loading in .txt type files
            if endswith(filename, ".dat") || endswith(filename, ".txt") || endswith(filename, ".ascii") || endswith(filename, ".csv") || endswith(filename, ".list") || endswith(filename, ".flm") || endswith(filename, ".asci") || endswith(filename, ".clean") || endswith(filename, ".asc") || endswith(filename, ".sed")
                #println(filename, " ", skip)
                file_path = joinpath(directory, filename)

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

                scaler = maximum(spectra) / 100
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
    
        end
        return files_dict
    end

    function read_template_files(spectra_file_names, spectra_info, label)
        println("Loading in $label Spectra...")
        spectra_files = Vector{Array{Any}}(undef, length(spectra_file_names))
        λ = Vector{Array{Any}}(undef, length(spectra_file_names))
        if label == "Hg Template"
            mask = Vector{Array{Any}}(undef, length(spectra_file_names))
        end

        Threads.@threads for i in ProgressBar(eachindex(spectra_file_names))
            key = spectra_file_names[i]
            spectra_path = joinpath(spectra_info["path"][key], string(key))

            content = CSV.read(spectra_path, DataFrame, delim = ",", header = false)
            λ[i] = content[:, 1]
            spectra_files[i] = content[:, 2]
            if label == "Hg Template"
                mask[i] = content[:, 3]
            end
        end

        #Scale up the files to avoid dealing with small numbers 
        spec_means = [mean(spectra_files[i]) for i in eachindex(spectra_files)]
        spectra_files = (spectra_files ./ spec_means .* 100)

        if label == "Hg Template"
            return λ, spectra_files, mask
        else
            return λ, spectra_files
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
        original_spectra = deepcopy(input_spectra)
        original_spectra_λ = deepcopy(input_spectra_λ)
        original_spectra_σ = deepcopy(input_spectra_σ)

        for i in 1:5

            emission_line = fill(NaN, length(input_spectra))
            emission_line_λ= fill(NaN, length(input_spectra))

            rolling_average = [mean(input_spectra[i:i+window_size]) for i in 1:(length(input_spectra)-window_size)]
            rolling_sigma = [std(input_spectra[i:i+window_size]) for i in 1:(length(input_spectra)-window_size)]

            for i in 1:length(input_spectra)-window_size
                if input_spectra[i] > (rolling_average[i] + (σ * rolling_sigma[i]))
                    emission_line[i] = input_spectra[i]
                    emission_line_λ[i] = input_spectra_λ[i]
                end
            end

            line_index = findall(isnan.(emission_line))

            if isempty(line_index)
                break
            end

            input_spectra = input_spectra[line_index]
            input_spectra_σ = input_spectra_σ[line_index]
            input_spectra_λ = input_spectra_λ[line_index]
        end

        #Find the missing wavelengths between the original_spectra_λ and input_spectra_λ
        missing_λ = setdiff(original_spectra_λ, input_spectra_λ)
        missing_indices = findall(x -> x in missing_λ, original_spectra_λ)

        #TEST
        input_spectra = original_spectra
        input_spectra_σ = original_spectra_σ
        input_spectra_λ = original_spectra_λ

        input_spectra[missing_indices] .= NaN
        input_spectra_σ[missing_indices] .= NaN
        input_spectra_λ[missing_indices] .= NaN

        emission_line_λ_arr = original_spectra_λ[missing_indices]
        emission_line_arr = original_spectra[missing_indices]

        return emission_line_λ_arr, emission_line_arr,  input_spectra_λ, input_spectra, input_spectra_σ
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
        window_indexes = 8
        σ_threshold = 2.6
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

    #-------------------  fit_spectrum -------------------

    """
        fit_spectrum(spectra_to_fit, spectra_λ, filter_curve, filter_λ, pure_sn_spectra, sn_file_names, pure_hg_spectra, hg_file_names)

    TBW
    """
    function fit_spectrum(input_spectra, input_spectra_σ, input_spectra_λ, sn_templates, sn_file_names, 
        sn_info, hg_templates, hg_file_names, hg_info, sn_wavelength, hg_wavelength, hg_mask, target_redshift, telluric_λ, remove_telluric, batch_mode)
        #Generate empty matricies to store the different parameters returned from the fit
        num_sn = length(sn_templates)
        num_hg = length(hg_templates)

        number_of_fits = length(unique(values(sn_info["Type"])))
        unique_types = unique(values(sn_info["Type"]))

        #Generate an empy array of length number_of_fits to store the top number_of_fits χ2 values
        chi_squared_arr = collect(1:1:number_of_fits).*10.0^10
        chi_squared_red_arr = collect(1:1:number_of_fits).*10.0^10

        #Empty arrays to store the fitting info
        norm_arr = zeros(Float16, number_of_fits, 2)
        redshift_vals = zeros(Float16, number_of_fits)
        indexes = zeros(Int, number_of_fits, 2)
        emission_lines = Vector{Array}(undef, number_of_fits)
        emission_lines_λ= Vector{Array}(undef, number_of_fits)

        #The empty arrays to store the binned spectra
        binned_spectra_arr = Vector{Array}(undef, number_of_fits)
        binned_λ_arr = Vector{Array}(undef, number_of_fits)
        binned_spectra_σ_arr = Vector{Array}(undef, number_of_fits)
        best_fit_arr = Vector{Array}(undef, number_of_fits)
        sn_component_arr = Vector{Array}(undef, number_of_fits)
        hg_component_arr = Vector{Array}(undef, number_of_fits)
        spectra_fit_region = Vector{Array}(undef, number_of_fits)
        spectra_σ_fit_region = Vector{Array}(undef, number_of_fits)
        spectra_λ_fit_region = Vector{Array}(undef, number_of_fits)
        contam_array = Vector{Float16}(undef, number_of_fits)

        spectra_to_fit = deepcopy(input_spectra)
        spectra_to_fit_σ = deepcopy(input_spectra_σ)
        spectra_λ = deepcopy(input_spectra_λ)
        original_λ = deepcopy(input_spectra_λ)
        

        #pre process the spectra
        processed_spectra_λ, processed_spectra, processed_spectra_σ, telluric_λ, emission_line_λ, emission_line= process_spectra(
            spectra_λ, 
            spectra_to_fit, 
            spectra_to_fit_σ, 
            telluric_λ,
            remove_telluric
        )

        #If redshift is a free parameter, set the range
        redshift_max = 0.8

        if ismissing(target_redshift)
            redshift_max = 0.6
        else
            if target_redshift > 0.6
                redshift_max = target_redshift
            else
                redshift_max = 0.6
            end
        end

        #Define the redshift steps
        redshift_range = collect(0.00:0.01:redshift_max)
        
        for z in (batch_mode ? redshift_range : ProgressBar(redshift_range))

            spectra_to_fit = deepcopy(processed_spectra)
            spectra_to_fit_σ = deepcopy(processed_spectra_σ)
            spectra_to_fit_λ = deepcopy(processed_spectra_λ)

            #Check if there is a provided redshift value
            if ismissing(target_redshift)
                
            else
                if round(z, digits = 2) != round(target_redshift, digits = 2)
                    continue
                end
            end

            #Redshift the spectra
            spectra_to_fit, spectra_to_fit_σ, spectra_to_fit_λ, pure_sn_spectra, pure_hg_spectra, sn_λ, hg_λ, 
            standard_template_indices, non_standard_template_indices, binned_spectra, binned_spectra_σ, binned_spectra_λ, hg_mask = redshift_spectra(
                spectra_to_fit_λ, 
                spectra_to_fit, 
                spectra_to_fit_σ, 
                sn_templates, 
                sn_wavelength, 
                sn_info, 
                sn_file_names, 
                hg_templates, 
                z, 
                hg_wavelength, hg_mask, original_λ)

            deep_copy_spectra = deepcopy(spectra_to_fit)
            deep_copy_spectra_σ = deepcopy(spectra_to_fit_σ)
            deep_copy_λ = deepcopy(spectra_to_fit_λ)

            for sn_i_idx in ProgressBar(1:num_sn)
                λ_region = Array{Any}(undef)
                spectra_to_fit = copy(deep_copy_spectra)   
                spectra_to_fit_σ = copy(deep_copy_spectra_σ)
                full_template_λ = copy(deep_copy_λ)
            
                if sn_info["Galactic Source"][sn_file_names[sn_i_idx]] == "Y" && z != 0.0
                    continue
                end
            
                #Make sure the templates are the same length as the spectra
                if sn_i_idx in non_standard_template_indices
                    sn_i = sn_templates[sn_i_idx]
                    sn_i_λ = sn_λ[sn_i_idx]
            
                    common_wavelengths = intersect(sn_i_λ, full_template_λ)
                    common_indices_sn = findall(in(common_wavelengths), sn_i_λ)
                    common_indices_template = findall(in(common_wavelengths), full_template_λ)
            
                    sn_i = sn_i[common_indices_sn]
                    sn_i_λ = sn_i_λ[common_indices_sn]
                    spectra_to_fit = spectra_to_fit[common_indices_template]
                    spectra_to_fit_σ = spectra_to_fit_σ[common_indices_template]
                else
                    sn_i = pure_sn_spectra[sn_i_idx] 
                    sn_i_λ = sn_λ[sn_i_idx]
                end
            
                ready_to_fit_sn_i = copy(sn_i)
                ready_to_fit_spectra = copy(spectra_to_fit)
                ready_to_fit_spectra_σ = copy(spectra_to_fit_σ)
            
                for hg_i_idx in 1:num_hg
                    spectra_to_fit = ready_to_fit_spectra  
                    spectra_to_fit_σ = ready_to_fit_spectra_σ
                    sn_i = ready_to_fit_sn_i
            
                    if sn_i_idx in non_standard_template_indices
                        hg_i = pure_hg_spectra[hg_i_idx]
                        hg_i_λ = hg_λ[hg_i_idx]
                        hg_i_mask = hg_mask[hg_i_idx]
            
                        common_wavelengths = intersect(sn_i_λ, hg_i_λ)
                        common_indices_hg = findall(in(common_wavelengths), hg_i_λ)
            
                        hg_i = hg_i[common_indices_hg]
                        hg_i_λ = hg_i_λ[common_indices_hg]
                        hg_i_mask = hg_i_mask[common_indices_hg]
                        λ_region = hg_i_λ
                    else
                        hg_i = pure_hg_spectra[hg_i_idx] 
                        hg_i_λ = hg_λ[hg_i_idx]
                        hg_i_mask = hg_mask[hg_i_idx]
                        λ_region = hg_i_λ
                    end
            
                    valid_indices = findall(==(1), hg_i_mask)
                    spectra_to_fit = view(spectra_to_fit, valid_indices)
                    spectra_to_fit_σ = view(spectra_to_fit_σ, valid_indices)
                    λ_region = view(λ_region, valid_indices)
                    sn_i = view(sn_i, valid_indices)
                    hg_i = view(hg_i, valid_indices)

                    norm_sn = calculate_B(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)
                    norm_hg = calculate_C(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)

                    template_best_fit =  (sn_i * norm_sn) .+ (hg_i * norm_hg)

                    chi_squared = sum((spectra_to_fit .- template_best_fit).^2 ./ spectra_to_fit_σ.^2)
                    #No_of_Free_Params= 2
                    #chi_squared_red = chi_squared / ((length(spectra_to_fit) - No_of_Free_Params) )

                    fractional_overlap = length(spectra_to_fit) / length(processed_spectra)
                    chi_squared_red = chi_squared / fractional_overlap

                    #Find the corresponding index in the chi_squared_arr from the type
                    type = sn_info["Type"][sn_file_names[sn_i_idx]]
                    index = findall(x -> x == type, unique_types)[1]

                    if chi_squared_red < chi_squared_red_arr[index]

                        chi_squared_arr[index] = chi_squared
                        chi_squared_red_arr[index] = chi_squared_red
                        norm_arr[index, 1] = norm_sn
                        norm_arr[index, 2] = norm_hg
                        indexes[index, 1] = sn_i_idx
                        indexes[index, 2] = hg_i_idx
                        redshift_vals[index] = z

                        contamination = ((sum(hg_i .* norm_hg)) / ((sum((sn_i .* norm_sn) .+ (hg_i .* norm_hg))))) * 100
                        contam_array[index] = contamination

                        emission_lines[index] = emission_line
                        emission_lines_λ[index] = emission_line_λ

                        binned_spectra_arr[index] = binned_spectra
                        binned_λ_arr[index] = binned_spectra_λ
                        binned_spectra_σ_arr[index] = binned_spectra_σ

                        best_fit_arr[index] = template_best_fit
                        sn_component_arr[index] = sn_i * norm_sn
                        hg_component_arr[index] = hg_i * norm_hg
                        spectra_fit_region[index] = spectra_to_fit
                        spectra_σ_fit_region[index] = spectra_to_fit_σ
                        spectra_λ_fit_region[index] = λ_region
                    end
                end
            end
        end
        return chi_squared_arr, chi_squared_red_arr, norm_arr, indexes, redshift_vals, best_fit_arr, sn_component_arr, hg_component_arr, spectra_fit_region, spectra_σ_fit_region, spectra_λ_fit_region, emission_lines_λ, binned_spectra_arr, binned_λ_arr, binned_spectra_σ_arr, telluric_λ, contam_array
    end

    #------------------- Chi Squared Classifier -------------------
    function chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, hg_mask,sn_template_λ, hg_template_λ, altered_spectra, target_λ, altered_spectra_σ, SN_ID, target_redshift, telluric_λ, batch_mode, show_plot = false, remove_telluric = "N")

        #Save the original target spectra
        original_altered_spectra = deepcopy(altered_spectra)
        original_altered_spectra_σ = deepcopy(altered_spectra_σ)
        original_wavelength = deepcopy(target_λ)

        top_fits_chi, top_fits_chi_red, top_fits_norm_params, top_fits_indices, top_fits_sn_redshift, top_fits_best_fits, top_fits_sn_component, top_fits_hg_component, top_fits_spectra_fit_region, top_fits_spectra_σ_fit_region, top_fits_λ_fit_region, emission_line_λ_arr, binned_plotting_spectra_arr, binned_plotting_λ_arr, binned_plotting_spectra_σ_arr, masked_tellurics, contam_frac = fit_spectrum(
            altered_spectra, 
            altered_spectra_σ, 
            target_λ, 
            pure_sn_spectra_files, 
            pure_sn_spectra_file_names, 
            pure_sn_spectra_info, 
            pure_hg_spectra_files, 
            pure_hg_spectra_file_names, 
            pure_hg_spectra_info, 
            sn_template_λ, 
            hg_template_λ, 
            hg_mask,
            target_redshift,
            telluric_λ, remove_telluric, batch_mode)

        fits = findall(x -> x != 0, top_fits_indices[:, 1])

        top_fits_sn_type = [pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[i]] for i in top_fits_indices[fits, 1]]
        top_fits_hg_type = [pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[i]] for i in top_fits_indices[fits, 2]]
        top_fits_sn_a_v = [pure_sn_spectra_info["SN A_v"][pure_sn_spectra_file_names[i]] for i in top_fits_indices[fits, 1]]
        top_fits_Eb_v = [i / 3.1 for i in top_fits_sn_a_v]
        top_fits_phase = [pure_sn_spectra_info["tau"][pure_sn_spectra_file_names[i]] for i in top_fits_indices[fits, 1]]
        
        plotpath = "No Plot Saved"
        best_chi = argmin(top_fits_chi_red)

        if isempty(top_fits_sn_type)
            show_plot = false
        end

        if show_plot == true
            # Adjust rcParams
            PyPlot.rc("axes", linewidth=1.75)
            PyPlot.rc("font", size=16)
            PyPlot.rc("axes", labelsize=16)
            PyPlot.rc("xtick", labelsize=14)
            PyPlot.rc("ytick", labelsize=14)
            PyPlot.rc("legend", fontsize=14)
        
            number_of_plots = length(fits)
        
            for i in 1:1
                i = best_chi  # Placeholder for testing
        
                phase_display = if top_fits_phase[i] == "N/A"
                    top_fits_phase[i] = pure_sn_spectra_info["t0"][pure_sn_spectra_file_names[top_fits_indices[i, 1]]]
                    L", t_0 = " * string(top_fits_phase[i]) * " days"
                else
                    L", \tau = " * string(top_fits_phase[i]) * " days"
                end
        
                emission_line_λ = emission_line_λ_arr[i]
        
                binned_plotting_spectra = binned_plotting_spectra_arr[i]
                binned_plotting_λ = binned_plotting_λ_arr[i]
                binned_plotting_spectra_σ = binned_plotting_spectra_σ_arr[i]
        
                common_λ = intersect(masked_tellurics, binned_plotting_λ)
                telluric_indices = findall(x -> x in common_λ, binned_plotting_λ)
                telluric_fluxes = binned_plotting_spectra[telluric_indices]
                telluric_λ = binned_plotting_λ[telluric_indices]
        
                plotting_original_wavelength = original_wavelength ./ (1 + top_fits_sn_redshift[i])
                plotting_telluric_λ = telluric_λ ./ (1 + top_fits_sn_redshift[i])
        
                # Set A4 size in inches
                a4_width, a4_height = 8.27, 11.69
        
                fig, axs = PyPlot.subplots(nrows=2, ncols=1, figsize=(a4_width, a4_height), dpi=400)
                fig.subplots_adjust(hspace=0, top=0.85, bottom=0.15)
        
                axs[1].tick_params(axis="both", which="major", width=1, length=6, direction="in")
                axs[1].tick_params(axis="both", which="minor", width=1, length=4, direction="in")
                axs[1].minorticks_on()
        
                host_frac = try
                    Int(round(contam_frac[i]))
                catch
                    999
                end
        
                figure_title = split(SN_ID, "_")[1]
                
                if top_fits_hg_type[i] == "Arp220.txt" || top_fits_hg_type[i] == "I20551.txt" || top_fits_hg_type[i] == "I22491.txt" || top_fits_hg_type[i] == "Mrk231.txt" || top_fits_hg_type[i] == "M82.txt" || top_fits_hg_type[i] == "NGC6240.txt" || top_fits_hg_type[i] == "NGC6090.txt"
                    host_template_title = "Starburst"

                elseif top_fits_hg_type[i] == "Ell2.txt" || top_fits_hg_type[i] == "Ell5.txt" || top_fits_hg_type[i] == "Ell13.txt"
                    host_template_title = "Elliptical"

                elseif top_fits_hg_type[i] == "S0.txt" || top_fits_hg_type[i] == "Sa.txt" || top_fits_hg_type[i] == "Sb.txt" || top_fits_hg_type[i] == "Sc.txt" || top_fits_hg_type[i] == "Sd.txt" || top_fits_hg_type[i] == "Sdm.txt" || top_fits_hg_type[i] == "Spi4.txt"
                    host_template_title = "Spiral"
                else
                    host_template_title = top_fits_hg_type[i]
                end
        
                fig.suptitle("$figure_title: \n χ²_ν = $(round(top_fits_chi_red[i], sigdigits=2)), SN = Type $(top_fits_sn_type[i]), Host = $host_template_title, \n z = $(round(top_fits_sn_redshift[i], sigdigits=2)), HG Contam = $host_frac% $phase_display", fontsize=18)
        
                # First subplot (original spectrum)
                axs[1].fill_between(plotting_original_wavelength, original_altered_spectra .- original_altered_spectra_σ, original_altered_spectra .+ original_altered_spectra_σ, color="black", alpha=0.12, label="Original Spectrum σ")
                axs[1].fill_between(binned_plotting_λ, binned_plotting_spectra .- binned_plotting_spectra_σ, binned_plotting_spectra .+ binned_plotting_spectra_σ, color="black", alpha=0.5, label="Binned Spectrum σ")
                axs[1].plot(plotting_original_wavelength, original_altered_spectra, label="Original Spectrum", color="black", alpha=0.2)
                axs[1].plot(binned_plotting_λ, binned_plotting_spectra, label="Binned Spectrum", color="black", alpha=0.7)
                axs[1].plot(top_fits_λ_fit_region[i], top_fits_best_fits[i], label="Best fit", color="red")
        
                axs[1].set_xlabel("Rest Wavelength [Å]")
                axs[1].set_ylabel("Flux [AU]")
        
                plotting_nans = isnan.(binned_plotting_spectra)
                max_array = binned_plotting_spectra[.!plotting_nans]
                max_σ_array = binned_plotting_spectra_σ[.!plotting_nans]
        
                try
                    axis_min, axis_max = minimum(max_array .- max_σ_array), maximum(max_array .+ max_σ_array)
                    if axis_min > 0
                        axis_min = 0
                    end

                    if axis_max < maximum(top_fits_best_fits[i])
                        axis_max = maximum(top_fits_best_fits[i]) * 1.05
                    end
                    axs[1].set_ylim(axis_min, axis_max)
                catch
                end
        
                # Second axis for observed wavelength
                x_axis = collect(minimum(plotting_original_wavelength) - 100:1:maximum(plotting_original_wavelength) + 100)
                rest_wavelength = x_axis .* (1 + top_fits_sn_redshift[i])
        
                axs2 = axs[1].twiny()
                axs2.tick_params(axis="both", which="major", width=1, length=6, direction="in", top=true)
                axs2.tick_params(axis="both", which="minor", width=1, length=4, direction="in", top=true)
                axs2.minorticks_on()
                axs2.set_xlabel("Observed Wavelength [Å]")
        
                min_rest_wavelength, max_rest_wavelength = minimum(rest_wavelength), maximum(rest_wavelength)
                tick_positions = round.(collect(ceil(min_rest_wavelength / 1000) * 1000:1000:floor(max_rest_wavelength / 1000) * 1000), digits=0)
                axs2.set_xticks(tick_positions)
                axs2.set_xlim(min_rest_wavelength, max_rest_wavelength)
        
                supernova_residual = top_fits_spectra_fit_region[i] .- top_fits_hg_component[i]
                sn_component = top_fits_sn_component[i]
        
                # Second subplot (supernova component)
                axs[2].tick_params(axis="both", which="major", width=1, length=6, direction="in")
                axs[2].tick_params(axis="both", which="minor", width=1, length=4, direction="in")
                axs[2].minorticks_on()
        
                axs[2].plot(top_fits_λ_fit_region[i], supernova_residual, label="Supernova Residual", color="#797A9E", alpha=0.6)
                axs[2].plot(top_fits_λ_fit_region[i], sn_component, label="Best Fit (SN)", color="magenta")
        
                axs[2].set_xlabel("Rest Wavelength [Å]")
                axs[2].set_ylabel("Flux [AU]")
                axs[2].set_xlim(axs[1].get_xlim())
        
                axs3 = axs[2].twiny()
                axs3.tick_params(axis="both", which="major", width=1, length=6, direction="in", top=true)
                axs3.tick_params(axis="both", which="minor", width=1, length=4, direction="in", top=true)
                axs3.minorticks_on()
                axs3.set_xticks(tick_positions)
                axs3.set_xlim(min_rest_wavelength, max_rest_wavelength)
                axs3.set_xticklabels([])
        
                handles_1, labels_1 = axs[1].get_legend_handles_labels()
                handles_2, labels_2 = axs[2].get_legend_handles_labels()
        
                fig.legend(vcat(handles_1, handles_2), vcat(labels_1, labels_2), loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon = false)
        
                plotpath = joinpath("classifications_dir", "plots", "Dwayne")
                if !isdir(plotpath)
                    mkdir(plotpath)
                end
        
                savefig(joinpath(plotpath, string(SN_ID) * ".pdf"))
                close(fig)
            end
        end
        

        best_chi = argmin(top_fits_chi_red)
        best_redshift = top_fits_sn_redshift[best_chi]

        if isempty(top_fits_sn_type)
            best_type = "Unknown"
            println("--------------------")
            printstyled("ERROR: ", color=:red)
            println("Fitting $SN_ID - No fitting results found.")
            println("--------------------")
        else
            best_type = top_fits_sn_type[best_chi]
        end
        

        return  top_fits_norm_params, top_fits_chi_red, plotpath, best_redshift, best_type
    end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_target_spectra, read_template_files, trim_nans, calculate_B, calculate_C
end