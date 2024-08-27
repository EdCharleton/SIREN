#------------------------------------------------------------
#Import Packages
#------------------------------------------------------------


#Define the code as a module
module ClassifierFunctions

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
    using Plots
    using Pkg
    using Interpolations: Linear, Gridded, interpolate
    using LsqFit
    using PyCall
    using LaTeXStrings
    using LinearAlgebra
    using ProfileView
    using Pkg
    using Conda
    using CSV
    using FITSIO
    using Distributed

    #using DuckDB

    matplotlib = pyimport("matplotlib")
    plt = pyimport("matplotlib.pyplot")
    numpy = pyimport("numpy")

    rcParams = matplotlib[:rcParams]
    

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

    #------------------------- arg_n_smallest_values --------------------------
    """
        arg_n_smallest_values(A::AbstractArray{T,N}, n::Integer) where {T,N}
    TBW
    """
    function arg_n_smallest_values(A::AbstractArray{T,N}, n::Integer) where {T,N}
        perm = sortperm(vec(A))
        ci = CartesianIndices(A)
        return ci[perm[1:n]]
    end

    #------------------------- find closest value --------------------------
    
    function find_closest_value(array, value)
        idx = argmin(abs.(array .- value))
        return array[idx]
    end

    #--------------------- Read in files ---------------------
    """
        read_file(key, spectra_dict)

        TBW
    """
    function read_file(key, spectra_dict)
        return readdlm(joinpath(spectra_dict["path"][key], string(key)))
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
    

    #------------------- Read in target spectra -------------------
    function read_target_spectra(directory::String, delimiter::Char = ',', skip_header::Int = 0)
        files_dict = Dict{String, Tuple{Vector{Float16}, Vector{Float16}, Vector{Float16}}}()
        println("Loading in Target Spectra...")
        skip = skip_header
        for filename in ProgressBar(readdir(directory))
            if endswith(filename, ".dat") || endswith(filename, ".txt") || endswith(filename, ".asci") || endswith(filename, ".csv") || endswith(filename, ".list")
                
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
                    spectra_σ = sqrt.(spectra) 
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
            end
    
        end
        return files_dict
    end

    #------------------- Load in Template Files -------------------
    function read_template_files(spectra_file_names, spectra_info, label)
        println("Loading in $label Spectra...")
        spectra_files = Vector{Array{Any}}(undef, length(spectra_file_names))
        λ = Vector{Array{Any}}(undef, length(spectra_file_names))

        Threads.@threads for i in ProgressBar(eachindex(spectra_file_names))
            key = spectra_file_names[i]
            spectra_path = joinpath(spectra_info["path"][key], string(key))

            content = CSV.read(spectra_path, DataFrame, delim = ",", header = false)
            λ[i] = content[:, 1]
            spectra_files[i] = content[:, 2]
        end

        #Scale up the files to avoid dealing with small numbers 
        spec_means = [mean(spectra_files[i]) for i in eachindex(spectra_files)]
        spectra_files = (spectra_files ./ spec_means .* 100)

        return λ, spectra_files
    end

    #------------------- Trim NaNs -------------------

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

    #------------------- Spectral Resampler -------------------

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
        σ_threshold = 2.5
        emission_line_λ, emission_line_flux,  output_spectra_λ, output_spectra, output_spectra_σ = remove_emission_lines(
            input_spectra_λ, 
            input_spectra, 
            input_spectra_σ, 
            window_indexes,
            σ_threshold)

        return output_spectra_λ, output_spectra, output_spectra_σ, plotting_telluric_x, plotting_telluric_y, emission_line_λ, emission_line_flux

    end
    

    function redshift_spectra(input_spectra_λ, input_spectra, input_spectra_σ, input_transient_templates, input_transient_λ, transient_metadata, transient_file_names, input_host_templates, redshift,  input_host_λ, original_spectra_λ)

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
    
        # Find the standard and non-standard wavelength templates
        standard_template_names = findall(x -> x == "Y", transient_metadata["Standard WL"])
        common_names = intersect(transient_file_names, standard_template_names)
        standard_template_indices = findall(x -> x in common_names, transient_file_names)
        
        non_standard_template_names = findall(x -> x == "N", transient_metadata["Standard WL"])
        common_names = intersect(transient_file_names, non_standard_template_names)
        non_standard_template_indices = findall(x -> x in common_names, transient_file_names)
    
        output_transient_templates = [input_transient_templates[i][template_indicies] for i in standard_template_indices]
        output_host_templates = [input_host_templates[i][template_indicies] for i in eachindex(input_host_templates)]
    
        output_host_λ = [input_host_λ[i][template_indicies] for i in eachindex(input_host_λ)]
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
        binned_spectra, binned_spectra_σ, binned_spectra_λ
        
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
        sn_info, hg_templates, hg_file_names, hg_info, sn_wavelength, hg_wavelength, target_redshift, telluric_λ, remove_telluric, batch_mode)
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

        redshift_max = 0.6

        if target_redshift > redshift_max
            redshift_max = round(target_redshift, digits = 2)
        end
        #Loop through the redshift range of z=0.0 to z=0.6
        redshift_range = collect(0.00:0.01:redshift_max)
        
        for z in (batch_mode ? redshift_range : ProgressBar(redshift_range))

            spectra_to_fit = deepcopy(processed_spectra)
            spectra_to_fit_σ = deepcopy(processed_spectra_σ)
            spectra_to_fit_λ = deepcopy(processed_spectra_λ)

            #Check if there is a provided redshift value

            if round(target_redshift, digits = 2) == 999
        
            else
                if round(z, digits = 2) != round(target_redshift, digits = 2)
                    continue
                end
            end

            #Redshift the spectra
            spectra_to_fit, spectra_to_fit_σ, spectra_to_fit_λ, pure_sn_spectra, pure_hg_spectra, sn_λ, hg_λ, 
            standard_template_indices, non_standard_template_indices, binned_spectra, binned_spectra_σ, binned_spectra_λ = redshift_spectra(
                spectra_to_fit_λ, 
                spectra_to_fit, 
                spectra_to_fit_σ, 
                sn_templates, 
                sn_wavelength, 
                sn_info, 
                sn_file_names, 
                hg_templates, 
                z, 
                hg_wavelength, original_λ)
            

            deep_copy_spectra = deepcopy(spectra_to_fit)
            deep_copy_spectra_σ = deepcopy(spectra_to_fit_σ)
            deep_copy_λ = deepcopy(spectra_to_fit_λ)

            for sn_i_idx in (1:num_sn)
                λ_region = Array{Any}(undef)
                spectra_to_fit = deepcopy(deep_copy_spectra)
                spectra_to_fit_σ = deepcopy(deep_copy_spectra_σ)
                full_template_λ= deepcopy(deep_copy_λ)

                if sn_info["Galactic Source"][sn_file_names[sn_i_idx]] == "Y"
                    if z != 0.0
                        continue
                    end
                end


                if sn_i_idx in non_standard_template_indices
                    sn_i = sn_templates[sn_i_idx]
                    sn_i_λ= sn_λ[sn_i_idx]


                    common_wavelengths = intersect(sn_i_λ, full_template_λ)

                    sn_i = sn_i[findall(x -> x in common_wavelengths, sn_i_λ)]
                    sn_i_λ= sn_i_λ[findall(x -> x in common_wavelengths, sn_i_λ)]

                    spectra_to_fit = spectra_to_fit[findall(x -> x in common_wavelengths, full_template_λ)]
                    spectra_to_fit_σ = spectra_to_fit_σ[findall(x -> x in common_wavelengths, full_template_λ)]

                else
                    sn_i = pure_sn_spectra[sn_i_idx] 
                    sn_i_λ= sn_λ[sn_i_idx]
                end

                for hg_i_idx in 1:num_hg

                    if sn_i_idx in non_standard_template_indices
                        hg_i = pure_hg_spectra[hg_i_idx]
                        hg_i_λ= hg_λ[hg_i_idx]

                        common_wavelengths = intersect(sn_i_λ, hg_i_λ)

                        hg_i = hg_i[findall(x -> x in common_wavelengths, hg_i_λ)]
                        hg_i_λ= hg_i_λ[findall(x -> x in common_wavelengths, hg_i_λ)]
                        λ_region = hg_i_λ

                    else
                        hg_i = pure_hg_spectra[hg_i_idx] 
                        hg_i_λ= hg_λ[hg_i_idx]
                        λ_region = hg_i_λ
                    end

                    norm_sn = calculate_B(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)
                    norm_hg = calculate_C(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)

                    convolved_pure_spectra =  (sn_i * norm_sn) .+ (hg_i * norm_hg)

                    chi_squared = sum((spectra_to_fit .- convolved_pure_spectra).^2 ./ spectra_to_fit_σ.^2)
                    DoF = 2
                    #chi_squared_red = chi_squared / ((length(spectra_to_fit) - DoF) * length(spectra_to_fit))
                    chi_squared_red = chi_squared / ((length(spectra_to_fit) - DoF) )

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

                        emission_lines[index] = emission_line
                        emission_lines_λ[index] = emission_line_λ

                        binned_spectra_arr[index] = binned_spectra
                        binned_λ_arr[index] = binned_spectra_λ
                        binned_spectra_σ_arr[index] = binned_spectra_σ

                        best_fit_arr[index] = convolved_pure_spectra
                        sn_component_arr[index] = sn_i * norm_sn
                        hg_component_arr[index] = hg_i * norm_hg
                        spectra_fit_region[index] = spectra_to_fit
                        spectra_σ_fit_region[index] = spectra_to_fit_σ
                        spectra_λ_fit_region[index] = λ_region
                    end
                end
            end
        end
        return chi_squared_arr, chi_squared_red_arr, norm_arr, indexes, redshift_vals, best_fit_arr, sn_component_arr, hg_component_arr, spectra_fit_region, spectra_σ_fit_region, spectra_λ_fit_region, emission_lines_λ, binned_spectra_arr, binned_λ_arr, binned_spectra_σ_arr, telluric_λ
    end

    #------------------- Chi Squared Classifier -------------------
    function chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, sn_template_λ, hg_template_λ, altered_spectra, target_λ, altered_spectra_σ, SN_ID, target_redshift, telluric_λ, batch_mode, show_plot = false, remove_telluric = "N")

        #Save the original target spectra
        original_altered_spectra = deepcopy(altered_spectra)
        original_altered_spectra_σ = deepcopy(altered_spectra_σ)
        original_wavelenth = deepcopy(target_λ)

        top_fits_chi, top_fits_chi_red, top_fits_norm_params, top_fits_indices, top_fits_sn_redshift, top_fits_best_fits, top_fits_sn_component, top_fits_hg_component, top_fits_spectra_fit_region, top_fits_spectra_σ_fit_region, top_fits_λ_fit_region, emission_line_λ_arr, binned_plotting_spectra_arr, binned_plotting_λ_arr, binned_plotting_spectra_σ_arr, masked_tellurics = fit_spectrum(
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
            target_redshift,
            telluric_λ, remove_telluric, batch_mode)

        fits = findall(x -> x != 0, top_fits_indices[:, 1])

        top_fits_sn_type = [pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[i]] for i in top_fits_indices[fits, 1]]
        top_fits_hg_type = [pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[i]] for i in top_fits_indices[fits, 2]]
        top_fits_sn_a_v = [pure_sn_spectra_info["SN A_v"][pure_sn_spectra_file_names[i]] for i in top_fits_indices[fits, 1]]
        top_fits_Eb_v = [i / 3.1 for i in top_fits_sn_a_v[fits]]
        top_fits_phase = [pure_sn_spectra_info["tau"][pure_sn_spectra_file_names[i]] for i in top_fits_indices[fits, 1]]
        
        plotpath = "No Plot Saved"

        if show_plot == true
            # Adjust rcparams
            rcParams = matplotlib["rcParams"]
            rcParams["axes.linewidth"] = 3

            number_of_plots = length(fits)
            #Create multiple new plots of the top 5 fits ----------------------------------------------------------------
            for i in 1:number_of_plots

                #Defines whether to set the phase as τ or t0
                if top_fits_phase[i] == "N/A"
                    top_fits_phase[i] = pure_sn_spectra_info["t0"][pure_sn_spectra_file_names[top_fits_indices[i, 1]]]
                    phase_display = L", $t_0$ = " * string(top_fits_phase[i]) * " days"
                else
                    phase_display = L", $\tau$ = " * string(top_fits_phase[i]) * " days"
                end

                #Extract the plotting data
                emission_line_λ= emission_line_λ_arr[i]
   
                binned_plotting_spectra = binned_plotting_spectra_arr[i]
                binned_plotting_λ= binned_plotting_λ_arr[i]
                binned_plotting_spectra_σ = binned_plotting_spectra_σ_arr[i]

                common_λ = intersect(masked_tellurics, binned_plotting_λ)
                telluric_indices = findall(x -> x in common_λ, binned_plotting_λ)
                telluric_fluxes = binned_plotting_spectra[telluric_indices]
                telluric_λ= binned_plotting_λ[telluric_indices]


                #Generate lines for plotting
                plotting_emission_line = [binned_plotting_spectra[i-1:i+1] for i in 1:length(binned_plotting_spectra)-1 if binned_plotting_λ[i] in emission_line_λ]
                plotting_emission_line_λ= [binned_plotting_λ[i-1:i+1] for i in 1:length(binned_plotting_spectra)-1 if binned_plotting_λ[i] in emission_line_λ]

                plotting_telluric = [binned_plotting_spectra[i-1:i+1] for i in 1:length(binned_plotting_spectra)-1 if binned_plotting_λ[i] in telluric_λ]
                plotting_telluric_λ= [binned_plotting_λ[i-1:i+1] for i in 1:length(binned_plotting_spectra)-1 if binned_plotting_λ[i] in telluric_λ]

                plotting_original_wavelength = original_wavelenth ./ (1 + top_fits_sn_redshift[i])
                plotting_telluric_λ= plotting_telluric_λ./ (1 + top_fits_sn_redshift[i])

                #----------------------------------------------------------------------------------------------------------------

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)
                plt.subplots_adjust(top = 0.85, bottom = 0.15)

                axs.tick_params(axis="both", which="major", width=1, length=6, direction="in", bottom = true, top = true, left = true, right = true)
                axs.tick_params(axis="both", which="minor", width=1, length=4, direction="in", bottom = true, top = true, left = true, right = true)
                axs.minorticks_on()

                fig.suptitle(string(SN_ID) * L": \chi^2_\nu / Å = " * string(round(top_fits_chi_red[i], digits=2))  * ", SN = Type " * string(top_fits_sn_type[i])  * ", Host = " * string(top_fits_hg_type[i])  * ", Z = " *string(round(top_fits_sn_redshift[i], digits = 2)) * "\n" * "HG Contam = " * string(round(top_fits_norm_params[i, 2]/(top_fits_norm_params[i, 1] + top_fits_norm_params[i, 2]) * 100, digits = 3)) * "%" * phase_display * ", E(B-V) = " * string(round(top_fits_Eb_v[i], digits = 2)), fontsize = 18)


                #axs.set_title("Source, SNe and Best Fit Spectrum")
                axs.fill_between(plotting_original_wavelength, (original_altered_spectra .- original_altered_spectra_σ)[:, 1], (original_altered_spectra .+ original_altered_spectra_σ)[:, 1], color="black", alpha=0.12)

                axs.fill_between(binned_plotting_λ, binned_plotting_spectra .- binned_plotting_spectra_σ, binned_plotting_spectra .+ binned_plotting_spectra_σ, color="black", alpha=0.5)
                axs.plot(plotting_original_wavelength, original_altered_spectra, label="Original Spectrum", color="black", alpha = 0.2)
                axs.plot(binned_plotting_λ, binned_plotting_spectra, label="Binned Spectrum", color="black", alpha = 0.7)
                axs.plot(top_fits_λ_fit_region[i], top_fits_best_fits[i] , label="Best fit", color="red")


                #for i in eachindex(plotting_emission_line)
                #    axs.plot(plotting_emission_line_λ[i], plotting_emission_line[i], color="orange", linestyle = ":")
                #end

                #for i in eachindex(plotting_telluric)
                #    axs.plot(plotting_telluric_λ[i], plotting_telluric[i], color="orange")
                #end

                axs.set_xlabel("Rest Wavelength " * "[" * L"\AA" * "]")
                axs.set_ylabel("Flux [AU]")
                plotting_nans = isnan.(binned_plotting_spectra)
                max_array = (binned_plotting_spectra[.!plotting_nans])
                max_σ_array = (binned_plotting_spectra_σ[.!plotting_nans])
                axis_min = minimum(max_array .- max_σ_array) * 1.1
                axis_max = maximum(max_array .+ max_σ_array) * 1.1
                #axs.set_ylim(axis_min, axis_max)

                offset = 30

                supernova_residual = top_fits_spectra_fit_region[i] .- top_fits_hg_component[i]
                supernova_residual = supernova_residual .- offset
                sn_component = top_fits_sn_component[i] .- offset

                axs.plot(top_fits_λ_fit_region[i], supernova_residual, label="Supernova (Spectrum - Host)", color="black")
                axs.plot(top_fits_λ_fit_region[i], sn_component, label="Pure SN", color="#1289A7")
                axs.set_xlim(minimum(plotting_original_wavelength) -100, maximum(plotting_original_wavelength) + 100)
                #axs.set_ylim(minimum(supernova_residual), 1.1 * maximum(binned_plotting_spectra))
                axs.legend(loc="center", bbox_to_anchor=(0.5, -.15), ncol = 4)

                #Add a second axis to display the rest wavelength
                x_axis = collect(minimum(plotting_original_wavelength)-100:1:maximum(plotting_original_wavelength)+100)

                rest_wavelength = x_axis .* (1 + top_fits_sn_redshift[i])

                axs2 = axs.twiny()
                axs2.tick_params(axis="both", which="major", width=1, length=6, direction="in",  top = true)
                axs2.tick_params(axis="both", which="minor", width=1, length=4, direction="in",  top = true)
                axs2.minorticks_on()
                axs2.set_xlabel("Observed Wavelength " * "[" * L"\AA" * "]")
        
                # Calculate appropriate tick positions for rest wavelengths
                # Calculate the minimum and maximum rest wavelengths
                min_rest_wavelength = (minimum(rest_wavelength))
                max_rest_wavelength = (maximum(rest_wavelength)) 

                # Generate tick positions at intervals of 1000 Angstroms
                tick_min = ceil(min_rest_wavelength / 1000) * 1000
                tick_max = floor(max_rest_wavelength / 1000) * 1000

                tick_positions = collect(tick_min:1000:tick_max)
                tick_positions = round.(tick_positions, digits=0)

                # Set tick labels for the second x-axis
                axs2.set_xticks(tick_positions)
                #axs2.set_xticklabels(["$(tick)" for tick in tick_positions])
                axs2.set_xlim(min_rest_wavelength, max_rest_wavelength)

                plotpath = joinpath("classifications_dir", "plots", string(SN_ID))
                if !isdir(plotpath)
                    mkdir(plotpath)
                end

                plt.savefig(plotpath * "/" * string(SN_ID) * "_fit_" * string(i) * ".png")
                plt.close()
                plt.clf()   
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

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_file, read_target_spectra, read_template_files, trim_nans, calculate_B, calculate_C
end