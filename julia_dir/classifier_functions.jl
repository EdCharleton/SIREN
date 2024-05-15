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


    #------------------------- fitting_func --------------------------
    """
        fitting_func(x, p)
    Function that normalises the pure sn and hg spectra to the altered spectra.

    # Parameters
    - `x::Array{Float64,1}`: The pure spectra.
    - `p::Array{Float64,1}`: The normalisation parameters.

    # Returns
    - `normalised_x::Array{Float64,1}`: The normalised pure spectra.
    """
    function fitting_func(x, params)
        a, b = params
        fit = (x[:, 1] .* a) .+ (x[:, 2] .* b)
        return fit
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
                content = readdlm(file_path)
                wl = content[:, 1]
                spectra = content[:, 2] 
                spec_min = minimum(spectra)
                if spec_min < 0
                    spectra = spectra .+ abs(spec_min)
                end
                spectra = spectra ./ maximum(spectra) .* 100

                #if no error provided, spectra_σ is set to root of the spectra
                spectra_σ = sqrt.(spectra)

                files_dict[filename] = (wl, spectra, spectra_σ)

            elseif endswith(filename, ".fits")
                file_path = joinpath(directory, filename)
                
                f = FITS(file_path)
                spectra_hdr = read_header(f[1])

                spectra = read(f[1])
                spectra_σ = read(f[2])
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
        Threads.@threads for i in ProgressBar(eachindex(spectra_file_names))
            key = spectra_file_names[i]
            spectra_path = joinpath(spectra_info["path"][key], string(key))

            #con = DBInterface.connect(DuckDB.DB)
            #results = DBInterface.execute(con,  """SELECT * FROM read_parquet('$spectra_path')""")
            #df = DataFrame((results))
            #array = collect(df.spectra)
            #spectra_files[i] = array
            spectra_files[i] = readdlm(spectra_path)
        end

        #Scale up the files to avoid dealing with small numbers 
        spec_means = [mean(spectra_files[i]) for i in eachindex(spectra_files)]
        spectra_files = (spectra_files ./ spec_means .* 100)

        return spectra_files
    end


    #------------------- Check target size-------------------

    function check_target_size(target_wl, template_wl, target_spectra, target_spectra_σ)
        #Function that checks if the target spectra Wavelength range is the same as the pure spectra, if it isnt, rebin the pure spectra to that of the target spectra
        if target_wl[1] > template_wl[1] || target_wl[end] < template_wl[end]
            println("Target spectra is not the same size as the template spectra, target is being rebinned to the template spectra")

            new_spectrum, new_spectrum_σ = spectres.spectres(numpy.array(template_wl), numpy.array(target_wl), numpy.array(target_spectra), numpy.array(target_spectra_σ), verbose = false)
        end        

        return template_wl, new_spectrum, new_spectrum_σ

    end

    #------------------- Trim NaNs -------------------

    function trim_nans(target_spectra, target_spectra_σ, template_wl, pure_sn_spectra_files, pure_hg_spectra_files)

        valid_indices = findall(.!isnan.(target_spectra))
        
        # Trim arrays
        target_spectra = target_spectra[valid_indices]
        target_spectra_σ = target_spectra_σ[valid_indices]
        template_wl = template_wl[valid_indices]

        pure_sn_spectra_files = [pure_sn_spectra_files[i][valid_indices] for i in eachindex(pure_sn_spectra_files)]
        pure_hg_spectra_files = [pure_hg_spectra_files[i][valid_indices] for i in eachindex(pure_hg_spectra_files)]
        
        return target_spectra,target_spectra_σ, template_wl, pure_sn_spectra_files, pure_hg_spectra_files
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

    function rolling_average_emission_removal(target_wl, target_spectra, target_spectra_σ, window_size, pure_sn_spectra_files, pure_hg_spectra_files)
        global_emission_line = fill(NaN, length(target_spectra))
        global_emission_line_wl = fill(NaN, length(target_spectra))


        for i in 1:3
            emission_line = fill(NaN, length(target_spectra))
            emission_line_wl = fill(NaN, length(target_spectra))

            rolling_average = [mean(target_spectra[i:i+window_size]) for i in 1:(length(target_spectra)-window_size)]
            rolling_sigma = [std(target_spectra[i:i+window_size]) for i in 1:(length(target_spectra)-window_size)]

            for i in 1:length(target_spectra)-window_size
                if target_spectra[i] > (rolling_average[i] + (2.5 * rolling_sigma[i]))
                    #println("Line Found at Wavelength: ", target_wl[i])
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
            
            #Trim the Emission lines from the target spectra
            target_spectra, target_spectra_σ, target_wl, pure_sn_spectra_files, pure_hg_spectra_files = trim_nans(target_spectra, target_spectra_σ, target_wl, pure_sn_spectra_files, pure_hg_spectra_files)

        end
        line_index = findall(.!isnan.(global_emission_line))
        global_emission_line = global_emission_line[line_index]
        global_emission_line_wl = global_emission_line_wl[line_index]

        return global_emission_line, global_emission_line_wl, target_spectra, target_spectra_σ, target_wl, pure_sn_spectra_files, pure_hg_spectra_files
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
    function fit_spectrum(spectra_to_fit, spectra_to_fit_σ, spectra_wl, pure_sn_spectra, sn_file_names, sn_info, pure_hg_spectra, hg_file_names, hg_info, target_redshift)
        #Generate empty matricies to store the different parameters returned from the fit
        num_sn = length(pure_sn_spectra)
        num_hg = length(pure_hg_spectra)

        #Generate an empy array of length 5 to store the top 5 χ2 values
        chi_squared_arr = [10.0^10, 11.0^10, 12.0^10, 13.0^10, 14.0^10]
        chi_squared_red_arr = [10.0^10, 11.0^10, 12.0^10, 13.0^10, 14.0^10]
        norm_arr = zeros(Float16, 5, 2)
        indexes = zeros(Int, 5, 2)


        Threads.@threads for sn_i_idx in ProgressBar(1:num_sn)

            #For testing purposes. If the redshift is not zero, skip the supernova. Also includes OzDES parameters for no recorded redshifts
            if sn_info["redshift"][sn_file_names[sn_i_idx]] != round(target_redshift, digits = 2) && round(target_redshift, digits = 2) < 10.0 && round(target_redshift, digits = 2) > 0.0
               continue
            end

            if sn_info["Type"][sn_file_names[sn_i_idx]][1] != "Ia"
                continue
            end

            sn_i = pure_sn_spectra[sn_i_idx] 

            sn_redshift = sn_info["redshift"][sn_file_names[sn_i_idx]]
            hg_indexes = findall(x -> hg_info["redshift"][x] == sn_redshift, hg_file_names)
            matching_z_hg_spectra = pure_hg_spectra[hg_indexes]

            
            #Iterate through all possible combinations of supernova (SN) and Host Galaxy (HG) spectra
            for hg_i_idx in eachindex(hg_indexes)

                hg_i = matching_z_hg_spectra[hg_i_idx] 

                norm_sn = calculate_B(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)
                norm_hg = calculate_C(spectra_to_fit, sn_i, hg_i, spectra_to_fit_σ)

                convolved_pure_spectra =  (sn_i * norm_sn) .+ (hg_i * norm_hg)
                chi_squared = sum((spectra_to_fit .- convolved_pure_spectra).^2 ./ spectra_to_fit_σ.^2)
                DoF = 2
                chi_squared_red = chi_squared / (length(spectra_to_fit) - DoF)

                #if the value for chi-squared is less than the current minimum value, replace the minimum value with the new value
                if chi_squared < maximum(chi_squared_arr)
                    max_index = argmax(chi_squared_red_arr)
                    chi_squared_arr[max_index] = chi_squared
                    chi_squared_red_arr[max_index] = chi_squared_red
                    norm_arr[max_index, 1] = norm_sn
                    norm_arr[max_index, 2] = norm_hg
                    indexes[max_index, 1] = sn_i_idx
                    indexes[max_index, 2] = hg_indexes[hg_i_idx]
                end

            end
        end
        return chi_squared_arr, chi_squared_red_arr, norm_arr, indexes
    end




    #------------------- Chi Squared Classifier -------------------
        """
        chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_info, pure_hg_spectra_file_names, altered_spectra, altered_spectra_info, altered_spectra_file_names, random_index, SN_ID, show_plot = false)

    TBW
    """
    function chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_sn_spectra_files, pure_hg_spectra_info, pure_hg_spectra_file_names, pure_hg_spectra_files, altered_spectra, target_wl, altered_spectra_σ, SN_ID, target_redshift, show_plot = false)

        four_most_wl = collect(3800:15:9500)
        #Save the original target spectra
        original_altered_spectra = deepcopy(altered_spectra)
        original_altered_spectra_σ = deepcopy(altered_spectra_σ)
        original_wavelength = deepcopy(target_wl)
        
        #Check if the target spectra is the same size as the template spectra
        four_most_wl, altered_spectra, altered_spectra_σ = check_target_size(target_wl, four_most_wl, altered_spectra, altered_spectra_σ)

        #Trim NaNs
        altered_spectra, altered_spectra_σ, four_most_wl, pure_sn_spectra_files, pure_hg_spectra_files = trim_nans(altered_spectra, altered_spectra_σ, four_most_wl, pure_sn_spectra_files, pure_hg_spectra_files)
        
        binned_plotting_spectra = deepcopy(altered_spectra)
        binned_plotting_spectra_σ = deepcopy(altered_spectra_σ)
        binned_plotting_wl = deepcopy(four_most_wl)

        #Check for any emission lines and deweight them in the fit by setting them to NaNs
        emission_line, emission_line_wl, altered_spectra, altered_spectra_σ, four_most_wl, pure_sn_spectra_files, pure_hg_spectra_files = rolling_average_emission_removal(four_most_wl, altered_spectra, altered_spectra_σ, 10, pure_sn_spectra_files, pure_hg_spectra_files)

        #chi_squared,  chi_squared_red, norm_vals= fit_sn_hg(pure_sn_spectra_files, pure_hg_spectra_files, pure_sn_spectra_info, pure_hg_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_file_names, altered_spectra, (normalisation_guess_sn ), (normalisation_guess_hg))
        top_5_chi, top_5_chi_red, top_5_norm_params, top_5_indices = fit_spectrum(altered_spectra, altered_spectra_σ, four_most_wl, pure_sn_spectra_files, pure_sn_spectra_file_names, pure_sn_spectra_info,pure_hg_spectra_files, pure_hg_spectra_file_names, pure_hg_spectra_info, target_redshift)
    
        #Extract the best fit parameters
        top_5_sn_redshift = [pure_sn_spectra_info["redshift"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]
        top_5_sn_type = [pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]
        top_5_hg_type = [pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[i]] for i in top_5_indices[:, 2]]
        top_5_sn_a_v = [pure_sn_spectra_info["SN A_v"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]
        top_5_Eb_v = [i / 3.1 for i in top_5_sn_a_v]
        top_5_phase = [pure_sn_spectra_info["tau"][pure_sn_spectra_file_names[i]] for i in top_5_indices[:, 1]]


        if show_plot == true
            # Adjust rcparams
            rcParams = matplotlib["rcParams"]
            rcParams["axes.linewidth"] = 3

            #Create multiple new plots of the top 5 fits ----------------------------------------------------------------
            for i in 1:5
                # Create a single subplot
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)

                #Adjusting tick parameters for both subplots
                axs.tick_params(axis="both", which="major", width=1, length=6, direction="in", bottom = true, top = true, left = true, right = true)
                axs.tick_params(axis="both", which="minor", width=1, length=4, direction="in", bottom = true, top = true, left = true, right = true)
                axs.minorticks_on()

                plot_color = "red"
                #Generate the best fit (made and the respective host galaxy and supernova components) for plotting
                sn_component = (top_5_norm_params[i, 1] * pure_sn_spectra_files[top_5_indices[i, 1]])
                hg_component = (top_5_norm_params[i, 2] * pure_hg_spectra_files[top_5_indices[i, 2]])

                best_fit = sn_component .+ hg_component

                # Plotting the data
                axs.set_title(string(SN_ID) * L": \chi^2_\nu = " * string(round(top_5_chi_red[i], digits=2))  * ", SN = Type " * string(top_5_sn_type[i][1])  * ", Host = " * string(top_5_hg_type[i])  * ", Z = " *string(round(top_5_sn_redshift[i], digits = 2)) * ", HG Contam = " * string(round(top_5_norm_params[i, 2]/(top_5_norm_params[i, 1] + top_5_norm_params[i, 2]) * 100, digits = 3)) * "%" * ", Phase = " * string(top_5_phase[i]) * " days" * ", E(B-V) = " * string(round(top_5_Eb_v[i], digits = 2)))
                axs.fill_between(original_wavelength, (original_altered_spectra .- original_altered_spectra_σ)[:, 1], (original_altered_spectra .+ original_altered_spectra_σ)[:, 1], color="black", alpha=0.12)
                axs.fill_between(four_most_wl, (altered_spectra .- altered_spectra_σ)[:, 1], (altered_spectra .+ altered_spectra_σ)[:, 1], color="black", alpha=0.5)
                axs.plot(original_wavelength, original_altered_spectra, label="Target Spectra", color="black", alpha = 0.2)
                axs.plot(binned_plotting_wl, binned_plotting_spectra, label="Target Spectra", color="black", alpha = 0.7)
                axs.plot(four_most_wl, best_fit, label="Best fit", color=plot_color)

                plotting_emission_line = [binned_plotting_spectra[i-1:i+1] for i in 1:length(binned_plotting_spectra) if binned_plotting_wl[i] in emission_line_wl]
                plotting_emission_line_wl = [binned_plotting_wl[i-1:i+1] for i in 1:length(binned_plotting_spectra) if binned_plotting_wl[i] in emission_line_wl]

                for i in 1:length(plotting_emission_line)
                    axs.plot(plotting_emission_line_wl[i], plotting_emission_line[i], color="orange", linestyle = ":")
                end

                axs.plot(four_most_wl, hg_component, label="Pure HG", color="#D980FA", linestyle = "--")
                axs.plot(four_most_wl, sn_component, label="Pure SN", color="#1289A7", linestyle = ":")
                axs.set_xlabel("Wavelength " * "[" * L"\AA" * "]")
                axs.set_ylabel("Flux [AU]")
                axs.set_ylim(0, 1.1 * maximum(binned_plotting_spectra))
                axs.legend(loc="center", bbox_to_anchor=(0.5, -0.1), ncol = 4)

                #Add a second axis to display the rest wavelength
                x_axis = axs.get_xticks()
                rest_wavelength = x_axis ./ (1 + target_redshift)
                plot_rest_wavelength = collect(rest_wavelength[1]:200:rest_wavelength[end])

                axs2 = axs.twiny()
                axs2.tick_params(axis="both", which="major", width=1, length=6, direction="in",  top = true)
                axs2.tick_params(axis="both", which="minor", width=1, length=4, direction="in",  top = true)
                axs2.minorticks_on()
                axs2.set_xlabel("Wavelength " * "[" * L"\AA" * "]")
        
                # Calculate appropriate tick positions for rest wavelengths
                # Calculate the minimum and maximum rest wavelengths
                min_rest_wavelength = floor(minimum(rest_wavelength))
                max_rest_wavelength = ceil(maximum(rest_wavelength)) 

                # Generate tick positions at intervals of 200 Angstroms
                tick_positions = collect(min_rest_wavelength:1000:max_rest_wavelength)

                # Set tick labels for the second x-axis
                axs2.set_xticks(tick_positions)
                axs2.set_xticklabels(["$(tick)" for tick in tick_positions])
                axs2.set_xlim(min_rest_wavelength, max_rest_wavelength)

                plotpath = joinpath("classifications_dir", "plots", string(SN_ID))
                if !isdir(plotpath)
                    mkdir(plotpath)
                end
                plt.savefig(plotpath * "/" * string(SN_ID) * "_fit_" * string(i) * ".png")
                plt.close()

                #Plot just the supernova

                supernova_residual = altered_spectra .- hg_component

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)  # figsize adjusted for two plots
                axs.tick_params(axis="both", which="major", width=1, length=6, direction="in", bottom = true, top = true, left = true, right = true)
                axs.tick_params(axis="both", which="minor", width=1, length=4, direction="in", bottom = true, top = true, left = true, right = true)
                axs.minorticks_on()

                axs.set_title(string(SN_ID) * L": \chi^2_\nu = " * string(round(top_5_chi_red[i], digits=2))  * ", SN = Type " * string(top_5_sn_type[i][1])  * ", Host = " * string(top_5_hg_type[i])  * ", Z = " *string(round(top_5_sn_redshift[i], digits = 2)) * ", HG Contam = " * string(round(top_5_norm_params[i, 2]/(top_5_norm_params[i, 1] + top_5_norm_params[i, 2]) * 100, digits = 3)) * "%" * ", Phase = " * string(top_5_phase[i]) * " days" * ", E(B-V) = " * string(round(top_5_Eb_v[i], digits = 2)))
                axs.plot(four_most_wl, supernova_residual, label="Supernova (Spectrum - Host)", color="black")
                axs.set_xlabel("Wavelength " * "[" * L"\AA" * "]")
                axs.set_ylabel("Flux [AU]")
                axs.legend(loc="center", bbox_to_anchor=(0.5, -0.1), ncol = 1)
                plt.savefig(plotpath * "/" * string(SN_ID) * "_fit_" * string(i) * "_supernova.png")
                plt.close()


            end
            

        end

        #Export the best fit and the target spectra
        #best_fit_path = joinpath("classifications_dir" ,"best_fit", "best_fit_" * string(SN_ID) * ".txt")
        #target_spectra_path = joinpath("classifications_dir" , "target_spectra", "target_spectra_" * string(SN_ID) * ".txt")
        #writedlm(best_fit_path, best_fit)
        #writedlm(target_spectra_path, altered_spectra)

        # Export the best fit parameters as metadata
        #best_fit_metadata_path = joinpath("classifications_dir", "metadata", "best_fit_" * string(SN_ID) * "_metadata.json")
        #best_fit_metadata = OrderedDict("SN Name" => pure_sn_spectra_file_names[best_indices[1]], "HG Name" => pure_hg_spectra_file_names[best_indices[2]], "SN Norm" => best_norm_params[1], "HG Norm" => best_norm_params[2], "Chi Squared" => best_chi, "Chi Squared Reduced" => best_chi_red)
        #JSON.write(best_fit_metadata_path, JSON.json(best_fit_metadata))

        #sn_best_fit_name = pure_sn_spectra_file_names[best_indices[1]]
        #hg_best_fit_name = pure_hg_spectra_file_names[best_indices[2]]

        return  top_5_norm_params, top_5_chi_red
    end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, fit_spectrum, read_file, read_target_spectra, read_template_files, check_target_size, trim_nans, calculate_B, calculate_C
end