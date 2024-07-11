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
    spectres = pyimport("spectres")
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

    #------------------- Read in target spectra -------------------
    function read_target_spectra(directory::String)
        files_dict = Dict{String, Tuple{Vector{Float16}, Vector{Float16}, Vector{Float16}}}()
        for filename in readdir(directory)
            if endswith(filename, ".dat") || endswith(filename, ".txt") || endswith(filename, ".asci") || endswith(filename, ".csv")
                file_path = joinpath(directory, filename)

                if endswith(filename, ".csv")
                    content = CSV.read(file_path, DataFrame, delim = ",")
                else 
                    content = readdlm(file_path)
                end
                wl = content[:, 1]
                spectra = content[:, 2] 
                spec_min = minimum(spectra)
                if spec_min < 0
                    spectra = spectra .+ abs(spec_min)
                end
                spectra = spectra ./ maximum(spectra) .* 100

                try
                    spectra_σ = content[:, 3]
                catch
                    spectra_σ = sqrt.(spectra)
                end
                #if no error provided, spectra_σ is set to root of the spectra
                spectra_σ = sqrt.(spectra_σ)


                files_dict[filename] = (wl, spectra, spectra_σ)

            elseif endswith(filename, ".fits")
                file_path = joinpath(directory, filename)
                
                f = FITS(file_path)
                spectra_hdr = read_header(f[1])

                spectra = read(f[1])

                try
                    spectra_σ = read(f[2])
                catch
                    spectra_σ = sqrt.(spectra)
                end

                spectra_σ = sqrt.(spectra_σ)
                try
                    wl = collect(spectra_hdr["CRVAL1"] - (spectra_hdr["CRPIX1"] * spectra_hdr["CDELT1"]) : spectra_hdr["CDELT1"] : (spectra_hdr["NAXIS1"] - spectra_hdr["CRPIX1"] - 1) * spectra_hdr["CDELT1"] + spectra_hdr["CRVAL1"])
                catch
                    wl = collect(spectra_hdr["CRVAL1"] - (spectra_hdr["CRPIX1"] * spectra_hdr["CD1_1"]) : spectra_hdr["CD1_1"] : (spectra_hdr["NAXIS1"] - spectra_hdr["CRPIX1"] - 1) * spectra_hdr["CD1_1"] + spectra_hdr["CRVAL1"])
                end

                
                #remove any nans from the spectra and the corresponding wl
                nans_indices = findall(.!isnan.(spectra))
                spectra = spectra[nans_indices]
                spectra_σ = spectra_σ[nans_indices]
                wl = wl[nans_indices]

                spec_min = minimum(spectra)
                if spec_min < 0
                    spectra = spectra .+ abs(spec_min)
                end

                scaler = maximum(spectra) 

                spectra = (spectra ./ scaler) .* 100
                spectra_σ = (spectra_σ ./ scaler) .* 100
    
                files_dict[filename] = (wl, spectra, spectra_σ)
            end
    
        end
        return files_dict
    end

    #------------------- Load in Template Files -------------------
    function read_template_files(spectra_file_names, spectra_info, label)
        println("Loading in $label Spectra...")
        spectra_files = Vector{Array{Any}}(undef, length(spectra_file_names))
        wavs = Vector{Array{Any}}(undef, length(spectra_file_names))

        Threads.@threads for i in ProgressBar(eachindex(spectra_file_names))
            key = spectra_file_names[i]
            spectra_path = joinpath(spectra_info["path"][key], string(key))

            content = CSV.read(spectra_path, DataFrame, delim = ",", header = false)
            wavs[i] = content[:, 1]
            spectra_files[i] = content[:, 2]
        end

        #Scale up the files to avoid dealing with small numbers 
        spec_means = [mean(spectra_files[i]) for i in eachindex(spectra_files)]
        spectra_files = (spectra_files ./ spec_means .* 100)

        return wavs, spectra_files
    end

    #------------------- Trim NaNs -------------------

    function trim_nans(target_spectra, target_spectra_σ, template_wl, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl)

        number_of_nans = sum(isnan.(target_spectra))

        valid_indices = findall(.!isnan.(target_spectra))
        if number_of_nans == 0 
            return target_spectra, target_spectra_σ, template_wl, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl
        else
            # Trim arrays
            target_spectra = target_spectra[valid_indices]
            target_spectra_σ = target_spectra_σ[valid_indices]
            template_wl = template_wl[valid_indices]

            pure_sn_spectra_files = [pure_sn_spectra_files[i][valid_indices] for i in eachindex(pure_sn_spectra_files)]
            pure_hg_spectra_files = [pure_hg_spectra_files[i][valid_indices] for i in eachindex(pure_hg_spectra_files)]

            supernova_wl = [supernova_wl[i][valid_indices] for i in eachindex(supernova_wl)]
            host_wl = [host_wl[i][valid_indices] for i in eachindex(host_wl)]
            
            return target_spectra,target_spectra_σ, template_wl, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl
        end
    end

    #------------------- Remove Galaxy Features -------------------

    function remove_galaxy_features(target_wl, target_spectra, target_spectra_σ)

        window = 15 #corresponds to bin size of 300 angstroms
        emission_line = fill(NaN, length(target_spectra))
        emission_line_wl = fill(NaN, length(target_spectra))

        emission_line = Float16[]
        emission_line_wl = Float16[]
    
        for i in eachindex(target_spectra)
            # Step up in bins 120, find the average and look for peaks above 3 sigma
            if i + window < length(target_spectra)
                average = mean(target_spectra[i:i+window])
                sigma = std(target_spectra[i:i+window])
                if target_spectra[i] > average + (2 * sigma)
                    #emission_line[i] = target_spectra[i]
                    #emission_line_wl[i] = target_wl[i]

                    push!(emission_line, target_spectra[i])
                    push!(emission_line_wl, target_wl[i])
                end
    
            else
                average = mean(target_spectra[i-window:i])
                sigma = std(target_spectra[i-window:i])
                if target_spectra[i] > average + (2 * sigma)
                    ##emission_line[i] = target_spectra[i]
                    #emission_line_wl[i] = target_wl[i]

                    push!(emission_line, target_spectra[i])
                    push!(emission_line_wl, target_wl[i])
                end
            end
        end
        return emission_line, emission_line_wl
    end

    function rolling_average_emission_removal(target_wl, target_spectra, target_spectra_σ, window_size, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl)
        global_emission_line = fill(NaN, length(target_spectra))
        global_emission_line_wl = fill(NaN, length(target_spectra))

        for i in 1:3

            emission_line = fill(NaN, length(target_spectra))
            emission_line_wl = fill(NaN, length(target_spectra))

            rolling_average = [mean(target_spectra[i:i+window_size]) for i in 1:(length(target_spectra)-window_size)]
            rolling_sigma = [std(target_spectra[i:i+window_size]) for i in 1:(length(target_spectra)-window_size)]

            for i in 1:length(target_spectra)-window_size
                if target_spectra[i] > (rolling_average[i] + (2.6 * rolling_sigma[i]))
                    emission_line[i] = target_spectra[i]
                    emission_line_wl[i] = target_wl[i]
                end
            end

            line_index = findall(.!isnan.(emission_line))
            emission_line = emission_line[line_index]
            emission_line_wl = emission_line_wl[line_index]

            #Set the global emission line indexs at the corresponding wavelength to the emission line found
            global_indexes = findall(x -> x in emission_line_wl, target_wl)
            global_emission_line[global_indexes] .= emission_line
            global_emission_line_wl[global_indexes] .= emission_line_wl

            #mask out the emission lines from the target spectra and the pure spectra
            target_spectra[line_index] .= NaN

            target_spectra, target_spectra_σ, target_wl, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl = trim_nans(target_spectra, target_spectra_σ, target_wl, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl)

        end

        line_index = findall(.!isnan.(global_emission_line))
        global_emission_line = global_emission_line[line_index]
        global_emission_line_wl = global_emission_line_wl[line_index]

        return global_emission_line, global_emission_line_wl, target_spectra, target_spectra_σ, target_wl, pure_sn_spectra_files, pure_hg_spectra_files, supernova_wl, host_wl
    end
    

    function process_spectra(sn_templates, hg_templates, hg_wavelength, sn_wavelength, z, sn_info, sn_file_names, spectra_wl, input_spectra, input_spectra_σ)
        full_sn_spectra = deepcopy(sn_templates)
        full_hg_spectra = deepcopy(hg_templates)
        full_hg_wl = deepcopy(hg_wavelength)
        full_sn_wl = deepcopy(sn_wavelength)
        
        full_template_wl_range = collect(2315:15:9000)
    
        bottom_wl = 3800 / (1 + z)
        bottom_wl = find_closest_value(full_template_wl_range, bottom_wl)
    
        if bottom_wl < 2315
            bottom_wl = 2315
        end
    
        top_wl = 8000 / (1 + z)
        top_wl = find_closest_value(full_template_wl_range, top_wl)
    
        full_template_wl = collect(bottom_wl:15:top_wl)
    
        # Trim the templates to the same wavelength range as the target spectra
        common_elements = intersect(full_template_wl, full_template_wl_range)

    
        # Find indices in full_template_wl_range
        template_indicies = findall(x -> x in common_elements, full_template_wl_range)
    
        # Find the standard and non-standard wavelength templates
        standard_template_names = findall(x -> x == "Y", sn_info["Standard WL"])
        common_names = intersect(sn_file_names, standard_template_names)
        standard_template_indices = findall(x -> x in common_names, sn_file_names)
        
        non_standard_template_names = findall(x -> x == "N", sn_info["Standard WL"])
        common_names = intersect(sn_file_names, non_standard_template_names)
        non_standard_template_indices = findall(x -> x in common_names, sn_file_names)
    
        pure_sn_spectra = [full_sn_spectra[i][template_indicies] for i in standard_template_indices]
        pure_hg_spectra = [full_hg_spectra[i][template_indicies] for i in eachindex(full_hg_spectra)]
    
        hg_wl = [full_hg_wl[i][template_indicies] for i in eachindex(full_hg_wl)]

        sn_wl = Array{Any}(undef, length(full_sn_wl))
        for wl in eachindex(full_sn_wl)
            if wl in standard_template_indices
                sn_wl[wl] = full_sn_wl[wl][template_indicies]
            else
                sn_wl[wl] = full_sn_wl[wl]
            end
        end
    
        # Redshift the target spectra to rest wavelength
        de_redshifted_wl = spectra_wl ./ (1 + z)
    
        # Trim and rebin the spectra to fit to the wavelength range of the templates
        spectra_to_fit, spectra_to_fit_σ = spectres.spectres(
            numpy.array(full_template_wl), 
            numpy.array(de_redshifted_wl), 
            numpy.array(input_spectra), 
            numpy.array(input_spectra_σ), 
            verbose = false
        )
        
        spectra_to_fit, spectra_to_fit_σ, full_template_wl, pure_sn_spectra, pure_hg_spectra, sn_wl, hg_wl = trim_nans(
            spectra_to_fit, 
            spectra_to_fit_σ, 
            full_template_wl, 
            pure_sn_spectra, 
            pure_hg_spectra, 
            sn_wl, 
            hg_wl
        )
    
        # Copied purely for plotting
        binned_spectra = deepcopy(spectra_to_fit)
        binned_spectra_σ = deepcopy(spectra_to_fit_σ)
        binned_wl = deepcopy(full_template_wl)
    
        # Remove any emission lines from the spectrum
        emission_line, emission_line_wl, spectra_to_fit, spectra_to_fit_σ, full_template_wl, pure_sn_spectra, pure_hg_spectra, sn_wl, hg_wl = rolling_average_emission_removal(
            full_template_wl, 
            spectra_to_fit, 
            spectra_to_fit_σ, 
            12, 
            pure_sn_spectra, 
            pure_hg_spectra, 
            sn_wl, 
            hg_wl
        )
    
        deep_copy_spectra = deepcopy(spectra_to_fit)
        deep_copy_spectra_σ = deepcopy(spectra_to_fit_σ)
        deep_copy_wl = deepcopy(full_template_wl)
    
        return (
            spectra_to_fit, spectra_to_fit_σ, full_template_wl, 
            pure_sn_spectra, pure_hg_spectra, sn_wl, hg_wl, 
            binned_spectra, binned_spectra_σ, binned_wl, 
            deep_copy_spectra, deep_copy_spectra_σ, deep_copy_wl,
            non_standard_template_indices, emission_line, emission_line_wl,
            full_sn_spectra, full_hg_spectra, full_hg_wl, full_sn_wl
        )
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
        fit_spectrum(spectra_to_fit, spectra_wl, filter_curve, filter_wl, pure_sn_spectra, sn_file_names, pure_hg_spectra, hg_file_names)

    TBW
    """
    function fit_spectrum(input_spectra, input_spectra_σ, spectra_wl, sn_templates, sn_file_names, sn_info, hg_templates, hg_file_names, hg_info, sn_wavelength, hg_wavelength, target_redshift)
        #Generate empty matricies to store the different parameters returned from the fit
        num_sn = length(sn_templates)
        num_hg = length(hg_templates)

        #Generate an empy array of length 5 to store the top 5 χ2 values
        chi_squared_arr = [10.0^10, 11.0^10, 12.0^10, 13.0^10, 14.0^10]
        chi_squared_red_arr = [10.0^10, 11.0^10, 12.0^10, 13.0^10, 14.0^10]

        #Empty arrays to store the fitting info
        norm_arr = zeros(Float16, 5, 2)
        redshift_vals = zeros(Float16, 5)
        indexes = zeros(Int, 5, 2)
        emission_lines = Vector{Array}(undef, 5)
        emission_lines_wl = Vector{Array}(undef, 5)

        #The empty arrays to store the binned spectra
        binned_spectra_arr = Vector{Array}(undef, 5)
        binned_wl_arr = Vector{Array}(undef, 5)
        binned_spectra_σ_arr = Vector{Array}(undef, 5)
        best_fit_arr = Vector{Array}(undef, 5)
        sn_component_arr = Vector{Array}(undef, 5)
        hg_component_arr = Vector{Array}(undef, 5)
        spectra_fit_region = Vector{Array}(undef, 5)
        spectra_σ_fit_region = Vector{Array}(undef, 5)
        spectra_wl_fit_region = Vector{Array}(undef, 5)

        #Loop through the redshift range of z=0.0 to z=0.6
        redshift_range = collect(0.01:0.01:0.6)
        for z in ProgressBar(redshift_range)

            #Check if there is a provided redshift value
            if round(target_redshift, digits = 2) == 999
        
            else
                if round(z, digits = 2) != round(target_redshift, digits = 2)
                    continue
                end
            end

            #Process the spectra
            spectra_to_fit, spectra_to_fit_σ, full_template_wl, pure_sn_spectra, pure_hg_spectra, sn_wl, hg_wl, binned_spectra, binned_spectra_σ, binned_wl, deep_copy_spectra, deep_copy_spectra_σ, deep_copy_wl, non_standard_template_indices, emission_line, emission_line_wl, full_sn_spectra, full_hg_spectra, full_hg_wl, full_sn_wl = process_spectra(
                sn_templates, 
                hg_templates, 
                hg_wavelength, 
                sn_wavelength, 
                z, 
                sn_info, 
                sn_file_names, 
                spectra_wl, 
                input_spectra, 
                input_spectra_σ
            )

            for sn_i_idx in (1:num_sn)
                λ_region = Array{Any}(undef)
                spectra_to_fit = deepcopy(deep_copy_spectra)
                spectra_to_fit_σ = deepcopy(deep_copy_spectra_σ)
                full_template_wl = deepcopy(deep_copy_wl)


                if sn_i_idx in non_standard_template_indices
                    sn_i = full_sn_spectra[sn_i_idx]
                    sn_i_wl = sn_wl[sn_i_idx]


                    common_wavelengths = intersect(sn_i_wl, full_template_wl)

                    sn_i = sn_i[findall(x -> x in common_wavelengths, sn_i_wl)]
                    sn_i_wl = sn_i_wl[findall(x -> x in common_wavelengths, sn_i_wl)]

                    spectra_to_fit = spectra_to_fit[findall(x -> x in common_wavelengths, full_template_wl)]
                    spectra_to_fit_σ = spectra_to_fit_σ[findall(x -> x in common_wavelengths, full_template_wl)]

                else
                    sn_i = pure_sn_spectra[sn_i_idx] 
                    sn_i_wl = sn_wl[sn_i_idx]
                    #if z == round(.56, digits = 2)
                        #plt.plot(full_template_wl, spectra_to_fit, label = "Target Spectra", color = "black")
                        #plt.plot(sn_i_wl, sn_i, label = "Template Spectra", color = "red")
                        #plt.show()
                    #end
                    
                end

                for hg_i_idx in 1:num_hg

                    if sn_i_idx in non_standard_template_indices
                        hg_i = pure_hg_spectra[hg_i_idx]
                        hg_i_wl = hg_wl[hg_i_idx]

                        common_wavelengths = intersect(sn_i_wl, hg_i_wl)
                        λ_region = hg_i_wl

                        hg_i = hg_i[findall(x -> x in common_wavelengths, hg_i_wl)]
                        hg_i_wl = hg_i_wl[findall(x -> x in common_wavelengths, hg_i_wl)]
                        λ_region = hg_i_wl

                    else
                        hg_i = pure_hg_spectra[hg_i_idx] 
                        hg_i_wl = hg_wl[hg_i_idx]
                        λ_region = hg_i_wl

                        #println(length(sn_i), " ", length(sn_i_wl), " ",length(hg_i), " ",length(hg_i_wl))
                    end


                    norm_sn = calculate_B(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)
                    norm_hg = calculate_C(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)

                    convolved_pure_spectra =  (sn_i * norm_sn) .+ (hg_i * norm_hg)

                    chi_squared = sum((spectra_to_fit .- convolved_pure_spectra).^2 ./ spectra_to_fit_σ.^2)
                    DoF = 2
                    chi_squared_red = chi_squared / ((length(spectra_to_fit) - DoF) * length(spectra_to_fit))

                    #if the value for chi-squared is less than the current minimum value, replace the minimum value with the new value
                    if chi_squared_red < maximum(chi_squared_red_arr)
                        max_index = argmax(chi_squared_red_arr)
                        chi_squared_arr[max_index] = chi_squared
                        chi_squared_red_arr[max_index] = chi_squared_red
                        norm_arr[max_index, 1] = norm_sn
                        norm_arr[max_index, 2] = norm_hg
                        indexes[max_index, 1] = sn_i_idx
                        indexes[max_index, 2] = hg_i_idx
                        redshift_vals[max_index] = z
                        emission_lines[max_index] = emission_line
                        emission_lines_wl[max_index] = emission_line_wl
                        binned_spectra_arr[max_index] = binned_spectra
                        binned_wl_arr[max_index] = binned_wl
                        binned_spectra_σ_arr[max_index] = binned_spectra_σ

                        best_fit_arr[max_index] = convolved_pure_spectra
                        sn_component_arr[max_index] = sn_i*norm_sn
                        hg_component_arr[max_index] = hg_i*norm_hg
                        spectra_fit_region[max_index] = spectra_to_fit
                        spectra_σ_fit_region[max_index] = spectra_to_fit_σ
                        spectra_wl_fit_region[max_index] = λ_region
                        
                        #println(round(length(spectra_to_fit), digits = 2), " ", round(chi_squared, digits = 2), " ", chi_squared_red, " ", z)
                        
                    end
                end
            end
        end
        return chi_squared_arr, chi_squared_red_arr, norm_arr, indexes, redshift_vals, best_fit_arr, sn_component_arr, hg_component_arr, spectra_fit_region, spectra_σ_fit_region, spectra_wl_fit_region, emission_lines_wl, binned_spectra_arr, binned_wl_arr, binned_spectra_σ_arr
    end

    #------------------- Chi Squared Classifier -------------------
        """
        chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_info, pure_hg_spectra_file_names, altered_spectra, altered_spectra_info, altered_spectra_file_names, random_index, SN_ID, show_plot = false)

    TBW
    """
    function chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, sn_template_wl, hg_template_wl, altered_spectra, target_wl, altered_spectra_σ, SN_ID, target_redshift, show_plot = false)

        #Save the original target spectra
        original_altered_spectra = deepcopy(altered_spectra)
        original_altered_spectra_σ = deepcopy(altered_spectra_σ)
        original_wavelenth = deepcopy(target_wl)

        top_5_chi, top_5_chi_red, top_5_norm_params, top_5_indices, top_5_sn_redshift, top_5_best_fits, top_5_sn_component, top_5_hg_component, top_5_spectra_fit_region, top_5_spectra_σ_fit_region, top_5_λ_fit_region, emission_line_wl_arr, binned_plotting_spectra_arr, binned_plotting_wl_arr, binned_plotting_spectra_σ_arr = fit_spectrum(
            altered_spectra, 
            altered_spectra_σ, 
            target_wl, 
            pure_sn_spectra_files, 
            pure_sn_spectra_file_names, 
            pure_sn_spectra_info, 
            pure_hg_spectra_files, 
            pure_hg_spectra_file_names, 
            pure_hg_spectra_info, 
            sn_template_wl, 
            hg_template_wl, 
            target_redshift)

        #Extract the best fit parameters
        top_5_sn_type = [pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]
        top_5_hg_type = [pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[i]] for i in top_5_indices[:, 2]]
        top_5_sn_a_v = [pure_sn_spectra_info["SN A_v"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]
        top_5_Eb_v = [i / 3.1 for i in top_5_sn_a_v]
        top_5_phase = [pure_sn_spectra_info["tau"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]
        
        plotpath = "No Plot Saved"

        if show_plot == true
            # Adjust rcparams
            rcParams = matplotlib["rcParams"]
            rcParams["axes.linewidth"] = 3

            #Create multiple new plots of the top 5 fits ----------------------------------------------------------------
            for i in 1:5

                #Defines whether to set the phase as τ or t0
                if top_5_phase[i] == "N/A"
                    top_5_phase[i] = pure_sn_spectra_info["t0"][pure_sn_spectra_file_names[top_5_indices[i, 1]]]
                    phase_display = L", $t_0$ = " * string(top_5_phase[i]) * " days"
                else
                    phase_display = L", $\tau$ = " * string(top_5_phase[i]) * " days"
                end

                #Extract the plotting data
                emission_line_wl = emission_line_wl_arr[i]
   
                binned_plotting_spectra = binned_plotting_spectra_arr[i]
                binned_plotting_wl = binned_plotting_wl_arr[i]
                binned_plotting_spectra_σ = binned_plotting_spectra_σ_arr[i]

                #Generate lines for plotting
                plotting_emission_line = [binned_plotting_spectra[i:i+1] for i in 1:length(binned_plotting_spectra)-1 if binned_plotting_wl[i] in emission_line_wl]
                plotting_emission_line_wl = [binned_plotting_wl[i:i+1] for i in 1:length(binned_plotting_spectra)-1 if binned_plotting_wl[i] in emission_line_wl]

                plotting_original_wavelength = original_wavelenth ./ (1 + top_5_sn_redshift[i])

                #----------------------------------------------------------------------------------------------------------------

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)
                plt.subplots_adjust(top = 0.85, bottom = 0.15)

                axs.tick_params(axis="both", which="major", width=1, length=6, direction="in", bottom = true, top = true, left = true, right = true)
                axs.tick_params(axis="both", which="minor", width=1, length=4, direction="in", bottom = true, top = true, left = true, right = true)
                axs.minorticks_on()

                fig.suptitle(string(SN_ID) * L": \chi^2_\nu = " * string(round(top_5_chi_red[i], digits=2))  * ", SN = Type " * string(top_5_sn_type[i])  * ", Host = " * string(top_5_hg_type[i])  * ", Z = " *string(round(top_5_sn_redshift[i], digits = 2)) * "\n" * "HG Contam = " * string(round(top_5_norm_params[i, 2]/(top_5_norm_params[i, 1] + top_5_norm_params[i, 2]) * 100, digits = 3)) * "%" * phase_display * ", E(B-V) = " * string(round(top_5_Eb_v[i], digits = 2)), fontsize = 18)


                #axs.set_title("Source, SNe and Best Fit Spectrum")
                axs.fill_between(plotting_original_wavelength, (original_altered_spectra .- original_altered_spectra_σ)[:, 1], (original_altered_spectra .+ original_altered_spectra_σ)[:, 1], color="black", alpha=0.12)

                axs.fill_between(top_5_λ_fit_region[i], top_5_spectra_fit_region[i] .- top_5_spectra_σ_fit_region[i], top_5_spectra_fit_region[i] .+ top_5_spectra_σ_fit_region[i], color="black", alpha=0.5)
                axs.plot(plotting_original_wavelength, original_altered_spectra, label="Original Spectrum", color="black", alpha = 0.2)
                axs.plot(top_5_λ_fit_region[i], top_5_spectra_fit_region[i], label="Binned Spectrum", color="black", alpha = 0.7)
                axs.plot(top_5_λ_fit_region[i], top_5_best_fits[i] , label="Best fit", color="red")

                for i in eachindex(plotting_emission_line)
                    axs.plot(plotting_emission_line_wl[i], plotting_emission_line[i], color="orange", linestyle = ":")
                end

                axs.set_xlabel("Rest Wavelength " * "[" * L"\AA" * "]")
                axs.set_ylabel("Flux [AU]")
                #axs.set_ylim(0, 1.1 * maximum(binned_plotting_spectra[i]))

                offset = 30

                supernova_residual = top_5_spectra_fit_region[i] .- top_5_hg_component[i]
                supernova_residual = supernova_residual .- offset
                sn_component = top_5_sn_component[i] .- offset

                axs.plot(top_5_λ_fit_region[i], supernova_residual, label="Supernova (Spectrum - Host)", color="black")
                axs.plot(top_5_λ_fit_region[i], sn_component, label="Pure SN", color="#1289A7")
                axs.set_xlim(minimum(plotting_original_wavelength) -100, maximum(plotting_original_wavelength) + 100)
                #axs.set_ylim(minimum(supernova_residual), 1.1 * maximum(binned_plotting_spectra))
                axs.legend(loc="center", bbox_to_anchor=(0.5, -.15), ncol = 4)

                #Add a second axis to display the rest wavelength
                x_axis = collect(minimum(plotting_original_wavelength)-100:1:maximum(plotting_original_wavelength)+100)

                rest_wavelength = x_axis .* (1 + top_5_sn_redshift[i])

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

        best_chi = argmin(top_5_chi_red)
        best_redshift = top_5_sn_redshift[best_chi]
        best_type = top_5_sn_type[best_chi]

        return  top_5_norm_params, top_5_chi_red, plotpath, best_redshift, best_type
    end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_file, read_target_spectra, read_template_files, trim_nans, calculate_B, calculate_C
end