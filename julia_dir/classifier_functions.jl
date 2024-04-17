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
    matplotlib = pyimport("matplotlib")
    plt = pyimport("matplotlib.pyplot")
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
        bin_mask = altered_spectra .< 20

        binned_altered_spectra = [sum(altered_spectra[i:i+1]) for i in 1:2:length(altered_spectra)-1 if i == length(altered_spectra) || !bin_mask[i]]
        binned_pure_spectra = [sum(pure_spectra[i:i+1]) for i in 1:2:length(pure_spectra)-1 if i == length(pure_spectra) || !bin_mask[i]]
        return binned_altered_spectra, binned_pure_spectra
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

    
    #------------------- Fit The SNe and HG spectra -------------------

    """
    fit_sn_hg(pure_sn_spectra, pure_hg
    _spectra, sn_info, hg_info, sn_file_names, hg_file_names, target_spectra, p0_sn, p0_hg)

    Function that combines and fits all possible combinations of supernova and host galaxy spectra to the target spectra

    # Parameters
    - `pure_sn_spectra::Array{Array{Float64,2}}`: The pure sn spectra.
    - `pure_hg_spectra::Array{Array{Float64,2}}`: The pure hg spectra.
    - `sn_info::Dict{String,Any}`: The sn info.
    - `hg_info::Dict{String,Any}`: The hg info.
    - `sn_file_names::Array{String,1}`: The sn file names.
    - `hg_file_names::Array{String,1}`: The hg file names.
    - `target_spectra::Array{Float64,2}`: The spectra to be fit to
    - `p0_sn::Array{Float64,1}`: The initial normalisation parameters for the sn spectra.
    - `p0_hg::Array{Float64,1}`: The initial normalisation parameters for the hg spectra.

    # Returns
    - `chi_squared_arr::Array{Float64,2}`: The chi squared array.
    - `chi_squared_red_arr::Array{Float64,2}`: The reduced chi squared array.
    - `norm_arr::Array{Float64,3}`: The array containing normalisation parameters of both SN and HG.
    - `norm_err_arr::Array{Float64,3}`: The Associated Errors
"""
function fit_sn_hg(pure_sn_spectra, pure_hg_spectra, sn_info, hg_info, sn_file_names, hg_file_names, target_spectra, p0_sn, p0_hg)
    #Generate empty matricies to store the different parameters returned from the fit
        num_sn = length(pure_sn_spectra)
        num_hg = length(pure_hg_spectra)

        println("number of templates =" * string(num_sn))
        #generate empty arrays to populate with the results of the fits
        chi_squared_arr = fill(Inf, num_sn, num_hg)
        chi_squared_red_arr = fill(Inf, num_sn, num_hg)
        norm_arr = zeros(Float64, num_sn, num_hg, 2)
    
        max_target_spec_index = argmax(target_spectra)
        max_target_spec = maximum(target_spectra)

        Threads.@threads for sn_i_idx in ProgressBar(1:num_sn)

            sn_i = pure_sn_spectra[sn_i_idx]
            sn_redshift = sn_info["redshift"][sn_file_names[sn_i_idx]]
            hg_indexes = findall(x -> hg_info["redshift"][x] == sn_redshift, hg_file_names)
            matching_z_hg_spectra = pure_hg_spectra[hg_indexes]

            
            #Iterate through all possible combinations of supernova (SN) and Host Galaxy (HG) spectra
            for (hg_i_idx, hg_i) in enumerate(matching_z_hg_spectra)
                p0 = [p0_sn[sn_i_idx], p0_hg[hg_i_idx]]
                hg_sn_spec = hcat(sn_i, hg_i)
                
                #Set bounds, the normalisation parameters should be positive, and the extinction parameters should be between -2 and 1
                lower_bound = [0.0, 0.0]
                upper_bound = [max_target_spec ./ sn_i[max_target_spec_index]*1.2, max_target_spec ./ hg_i[max_target_spec_index]*1.2]

                fit = curve_fit.(fitting_func, Ref(hg_sn_spec), Ref(target_spectra[:, 1]), Ref(p0); lower=lower_bound, upper=upper_bound)
    
                DoF = 2
                norm_sn = fit.param[1]
                norm_hg = fit.param[2]

                convolved_pure_spectra =  (sn_i * norm_sn) .+ (hg_i * norm_hg)
                chi_squared = sum((target_spectra .- convolved_pure_spectra).^2 ./ target_spectra)
    
                chi_squared_arr[sn_i_idx, hg_indexes[hg_i_idx]] = chi_squared
                chi_squared_red_arr[sn_i_idx, hg_indexes[hg_i_idx]] = chi_squared / (length(target_spectra) - DoF)
                norm_arr[sn_i_idx, hg_indexes[hg_i_idx], 1] = norm_sn
                norm_arr[sn_i_idx, hg_indexes[hg_i_idx], 2] = norm_hg

            end
        end
    
        return chi_squared_arr, chi_squared_red_arr, norm_arr
    end
    
#------------------- B finding func -----------------------------

#Attempt with no 1-B = C condition

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


#------------------- Filter Flux Calculations -------------------

#function filter_flux(spectra_wl, spectra_flux, filter_curve, filter_wl, sn_spectra, hg_spectra)

"""
    filter_flux(spectra_to_fit, spectra_wl, filter_curve, filter_wl, pure_sn_spectra, sn_file_names, pure_hg_spectra, hg_file_names)

TBW
"""
function filter_flux(spectra_to_fit, spectra_wl, filter_curve, filter_wl, pure_sn_spectra, sn_file_names, sn_info, pure_hg_spectra, hg_file_names, hg_info)
    # Trim the spectra to the same range as the filter
    #indexes = findall(x -> x >= minimum(filter_wl) && x <= maximum(filter_wl), spectra_wl)
    #spectra_wl = spectra_wl[indexes]
    #filter_spectra_to_fit = spectra_to_fit[indexes]

    # Interpolate the filter to have the same resolution as the spectra
    #filter_curve_interp = interpolate((filter_wl,), filter_curve, Gridded(Linear()))

    # Evaluate the interpolated filter at the wavelengths of the spectra
    #interpolated_filter_values = filter_curve_interp(spectra_wl)

    # Calculate the flux inside the filter of the spectrum and the individual components
    #spectra_flux = sum(filter_spectra_to_fit .* interpolated_filter_values) 

    #Generate empty matricies to store the different parameters returned from the fit
    num_sn = length(pure_sn_spectra)
    num_hg = length(pure_hg_spectra)

    #generate empty arrays to populate with the results of the fits
    chi_squared_arr = fill(Inf, num_sn, num_hg)
    chi_squared_red_arr = fill(Inf, num_sn, num_hg)
    norm_arr = zeros(Float64, num_sn, num_hg, 2)

    #only use spectra at redshift 0 for testing:


    Threads.@threads for sn_i_idx in ProgressBar(1:num_sn)

        #If the redshift of the supernova isnt 0, skip it
        if sn_info["redshift"][sn_file_names[sn_i_idx]] != 0
            continue
        end

        #sn_flux = sum(pure_sn_spectra[sn_i_idx][indexes] .* interpolated_filter_values)
        #sn_flux_val = sn_flux / spectra_flux
        sn_i = pure_sn_spectra[sn_i_idx] 

        sn_redshift = sn_info["redshift"][sn_file_names[sn_i_idx]]
        hg_indexes = findall(x -> hg_info["redshift"][x] == sn_redshift, hg_file_names)
        matching_z_hg_spectra = pure_hg_spectra[hg_indexes]

        
        #Iterate through all possible combinations of supernova (SN) and Host Galaxy (HG) spectra
        for hg_i_idx in eachindex(hg_indexes)
            #hg_flux = sum(matching_z_hg_spectra[hg_i_idx][indexes] .* interpolated_filter_values)
            #hg_flux_val = hg_flux / spectra_flux

            hg_i = matching_z_hg_spectra[hg_i_idx] 

            norm_sn = calculate_B(spectra_to_fit, sn_i, hg_i, sqrt.(spectra_to_fit))
            norm_hg = calculate_C(spectra_to_fit, sn_i, hg_i, sqrt.(spectra_to_fit))

            convolved_pure_spectra =  (sn_i * norm_sn) .+ (hg_i * norm_hg)
            chi_squared = sum((spectra_to_fit .- convolved_pure_spectra).^2 ./ spectra_to_fit)

            DoF = 2

            chi_squared_arr[sn_i_idx, hg_indexes[hg_i_idx]] = chi_squared
            chi_squared_red_arr[sn_i_idx, hg_indexes[hg_i_idx]] = chi_squared / (length(spectra_to_fit) - DoF)
            norm_arr[sn_i_idx, hg_indexes[hg_i_idx], 1] = norm_sn 
            norm_arr[sn_i_idx, hg_indexes[hg_i_idx], 2] = norm_hg 

        end
    end

    return chi_squared_arr, chi_squared_red_arr, norm_arr
end


#------------------- Analytic Minimisation -------------------
"""
    analytic_minimisation(pure_sn_spectra_templates, spectra_to_fit)

TBW
"""
function analytic_minimisation(pure_sn_spectra_templates, sn_template_info, sn_file_names, spectra_to_fit)
    n_templates = length(pure_sn_spectra_templates)
    chi_squared = fill(Inf, n_templates)
    A_vals = fill(Inf, n_templates)

    Threads.@threads for i in ProgressBar(1:n_templates)
        A = (sum((pure_sn_spectra_templates[i] .* spectra_to_fit) ./ sqrt.(spectra_to_fit))) / (sum((pure_sn_spectra_templates[i].^2) ./ sqrt.(spectra_to_fit)))
        chi_squared[i] = sum((spectra_to_fit .- (A .* pure_sn_spectra_templates[i])).^2 ./ spectra_to_fit)
        A_vals[i] = A
    end

    best_index = argmin(chi_squared)
    best_A = A_vals[best_index]
    best_chi = chi_squared[best_index]
    best_A_v = sn_template_info["SN A_v"][sn_file_names[best_index]]
    best_Z = sn_template_info["redshift"][sn_file_names[best_index]]
    best_phase = sn_template_info["tau"][sn_file_names[best_index]]

    #Plotting - delete when done
    four_most_wl = collect(3790:20:9510)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)  # figsize adjusted for two plots           
    axs.tick_params(axis="both", which="major", width=1, length=6, direction="in", bottom = true, top = true, left = true, right = true)
    axs.tick_params(axis="both", which="minor", width=1, length=4, direction="in", bottom = true, top = true, left = true, right = true)
    axs.minorticks_on()
    axs.plot(four_most_wl, spectra_to_fit, label="Target Spectra", color="black", alpha = 0.5)
    axs.plot(four_most_wl, best_A .* pure_sn_spectra_templates[best_index], label="Best fit", color="green")
    axs.fill_between(four_most_wl, (spectra_to_fit .- spectra_to_fit.^0.5)[:, 1], (spectra_to_fit .+ spectra_to_fit.^0.5)[:, 1], color="black", alpha=0.1)
    axs.text(0.8, 0.9, L"\chi^2_\nu = " * string(round(best_chi/(length(spectra_to_fit)-1), digits=2)) * "\n" * "A = " * string(round(best_A, digits = 2)) * "\n" * L"A_v = " * string(best_A_v) * "\n" * "Z = " * string(best_Z) * "\n" * L"\tau = " * string(best_phase), horizontalalignment="center", verticalalignment="center", transform=axs.transAxes, color="black", fontsize=12, bbox=Dict("facecolor"=>"white", "edgecolor"=>"black", "boxstyle"=>"round,pad=0.5"))
    axs.set_xlabel("Wavelength " * "[" * L"\AA" * "]")
    axs.set_ylabel("Flux [Counts]")
    axs.set_ylim(0, 1.1 * maximum([maximum(spectra_to_fit), maximum(best_A .* pure_sn_spectra_templates[best_index])]))
    axs.legend(loc="center", bbox_to_anchor=(0.5, -0.1), ncol = 4)
    plotpath = joinpath("classifications_dir", "plots" , "analytic_fit_" * string(1) * ".png")
    plt.savefig(plotpath)
    plt.close()


    return best_A, best_chi
end


    #------------------- Chi Squared Classifier -------------------
    """
    chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_info, pure_hg_spectra_file_names, altered_spectra, altered_spectra_info, altered_spectra_file_names, random_index, SN_ID, show_plot = false)

TBW
"""
function chi_squared_classifier(pure_sn_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_info, pure_hg_spectra_file_names, altered_spectra, random_index, SN_ID, show_plot = false)

        println("Loading in templates...")
        #Read in the pure sn and hg spectra
        @time begin
            pure_sn_spectra_files = Vector{Any}(undef, length(pure_sn_spectra_file_names))
            Threads.@threads for i in eachindex(pure_sn_spectra_file_names)
                key = pure_sn_spectra_file_names[i]
                pure_sn_spectra_files[i] = read_file(key, pure_sn_spectra_info)
            end

            pure_hg_spectra_files = Vector{Any}(undef, length(pure_hg_spectra_file_names))
            Threads.@threads for i in eachindex(pure_hg_spectra_file_names)
                key = pure_hg_spectra_file_names[i]
                pure_hg_spectra_files[i] = read_file(key, pure_hg_spectra_info)
            end
        end

        #To avoid dealing with small numbers, immediately scale up the spectra to be around 100
        sn_spec_means = [mean(pure_sn_spectra_files[i]) for i in eachindex(pure_sn_spectra_files)]
        hg_spec_means = [mean(pure_hg_spectra_files[i]) for i in eachindex(pure_hg_spectra_files)]
        pure_sn_spectra_files = pure_sn_spectra_files ./ sn_spec_means .* 100
        pure_hg_spectra_files = pure_hg_spectra_files ./ hg_spec_means .* 100

        #Generate a guess at what the normalisation parameters should be. Half max for both components
        normalisation_guess_sn = (1 ./ maximum.(pure_sn_spectra_files)) .* maximum(altered_spectra) .* 1.0
        normalisation_guess_hg = (1 ./ maximum.(pure_hg_spectra_files)) .* maximum(altered_spectra) .* 1.0
        
        # Fit the spectra and return chi-squared and normalisation parameters
        four_most_wl = collect(3790:20:9510)

        bessel_v_dat = readdlm("filters_dir/Generic_Bessell.V.dat")
        bessel_v_wl = bessel_v_dat[:, 1]
        bessel_v_curve = bessel_v_dat[:, 2]
        

        alt_chi, alt_A = analytic_minimisation(pure_sn_spectra_files, pure_sn_spectra_info, pure_sn_spectra_file_names, altered_spectra)

        @time begin
            chi_squared,  chi_squared_red, norm_vals = filter_flux(altered_spectra, four_most_wl, bessel_v_curve, bessel_v_wl, pure_sn_spectra_files, pure_sn_spectra_file_names, pure_sn_spectra_info,pure_hg_spectra_files, pure_hg_spectra_file_names, pure_hg_spectra_info)
        end
        

        #chi_squared,  chi_squared_red, norm_vals= fit_sn_hg(pure_sn_spectra_files, pure_hg_spectra_files, pure_sn_spectra_info, pure_hg_spectra_info, pure_sn_spectra_file_names, pure_hg_spectra_file_names, altered_spectra, (normalisation_guess_sn ), (normalisation_guess_hg))
        best_indices = argmin(chi_squared)

        best_chi = chi_squared[best_indices[1], best_indices[2], :]
        best_chi_red = chi_squared_red[best_indices[1], best_indices[2]]
        best_norm_params = [norm_vals[best_indices[1], best_indices[2], 1], norm_vals[best_indices[1], best_indices[2], 2]]
        
        #Extract the best fit parameters
        sn_redshift = pure_sn_spectra_info["redshift"][pure_sn_spectra_file_names[best_indices[1]]]
        sn_type = pure_sn_spectra_info["Type"][pure_sn_spectra_file_names[best_indices[1]]]
        hg_type = pure_hg_spectra_info["HG Type"][pure_hg_spectra_file_names[best_indices[2]]]
        sn_a_v = pure_sn_spectra_info["SN A_v"][pure_sn_spectra_file_names[best_indices[1]]]
        phase = pure_sn_spectra_info["tau"][pure_sn_spectra_file_names[best_indices[1]]]

        #Extract the actual parameters
        #target_type = altered_spectra_info["Type"][altered_spectra_file_names[random_index]]
        #target_redshift = altered_spectra_info["redshift"][altered_spectra_file_names[random_index]]
        #true_hg_type = altered_spectra_info["HG Type"][altered_spectra_file_names[random_index]]
        #true_sn_a_v = altered_spectra_info["SN A_v"][altered_spectra_file_names[random_index]]
        
        #Define identified as a value of redshift within ± 0.03 of the true redshift and the correct type
        #identified = (abs(sn_redshift - target_redshift) < 0.03) && (sn_type == target_type)
        #identified_type = sn_type == target_type

        identified = true
        identified_type = true

        if show_plot == true
            #Define the 4MOST wavelength range

            # Adjust rcparams
            rcParams = matplotlib["rcParams"]
            rcParams["axes.linewidth"] = 3
            
            # Create a single subplot
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=300)  # figsize adjusted for two plots
            
            #Adjusting tick parameters for both subplots
            axs.tick_params(axis="both", which="major", width=1, length=6, direction="in", bottom = true, top = true, left = true, right = true)
            axs.tick_params(axis="both", which="minor", width=1, length=4, direction="in", bottom = true, top = true, left = true, right = true)
            axs.minorticks_on()

            #Set the plot color, if the type and redshift is Identified, then its green, it its just the type, then orange. Wrong classfications are red
            """
            if identified == true
                plot_color = "green"
            elseif identified_type == true
                plot_color = "#EE5A24"
            else
                plot_color = "#EA2027"
            end
            """

            plot_color = "green"
            #Generate the best fit (made and the respective host galaxy and supernova components) for plotting
            sn_component = (best_norm_params[1] * pure_sn_spectra_files[best_indices[1]])
            hg_component = (best_norm_params[2] * pure_hg_spectra_files[best_indices[2]])

            best_fit = sn_component .+ hg_component

            # Plotting the data
            axs.set_title("SN2011fe With Large Sa Host Galaxy Contamination")
            axs.plot(four_most_wl, altered_spectra, label="Target Spectra", color="black", alpha = 0.5)
            axs.plot(four_most_wl, best_fit, label="Best fit", color=plot_color)
            axs.fill_between(four_most_wl, (altered_spectra .- altered_spectra.^0.5)[:, 1], (altered_spectra .+ altered_spectra.^0.5)[:, 1], color="black", alpha=0.1)
            axs.plot(four_most_wl, hg_component, label="Pure HG", color="#D980FA", linestyle = "--")
            axs.plot(four_most_wl, sn_component, label="Pure SN", color="#1289A7", linestyle = ":")
            axs.set_xlabel("Wavelength " * "[" * L"\AA" * "]")
            axs.set_ylabel("Flux [AU]")
            axs.text(0.8, 0.8, L"\chi^2_\nu = " * string(round(best_chi_red, digits=2)) * "\n" * "SN = Type " * string(sn_type[1]) * "\n" * "Host = " * string(hg_type) * "\n" * "Z = " *string(round(sn_redshift, digits = 2)) * "\n" * "HG Contamination = " * string(round(best_norm_params[2]/(best_norm_params[1] + best_norm_params[2]) * 100, digits = 3)) * "%" * '\n' * "Phase = " * string(phase) * " days", horizontalalignment="center", verticalalignment="center", transform=axs.transAxes, color="black", fontsize=12, bbox=Dict("facecolor"=>"white", "edgecolor"=>"black", "boxstyle"=>"round,pad=0.5"))
            axs.set_ylim(0, 1.1 * maximum([maximum(altered_spectra), maximum(best_norm_params[1] .*pure_sn_spectra_files[best_indices[1]]), maximum(best_norm_params[2] .*pure_hg_spectra_files[best_indices[2]])]))
            axs.legend(loc="center", bbox_to_anchor=(0.5, -0.1), ncol = 4)
            #axs.text(0.9, 0.9, "True Type = " * string(target_type[1]) * "\n" * "True Z = " * string(target_redshift) * "\n" * "True HG = " * true_hg_type * "\n" * "True " * L"A_v = " * string(true_sn_a_v), horizontalalignment="center", verticalalignment="center", transform=axs.transAxes, color="black", fontsize=12, bbox=Dict("facecolor"=>"white", "edgecolor"=>"black", "boxstyle"=>"round,pad=0.5"))

            # Save the plot
            plotpath = joinpath("classifications_dir", "plots" , "sn_2011fe_with_large_sa_hg" * ".png")
            plt.savefig(plotpath)
            plt.close()
        end

        #Export the best fit and the target spectra
        best_fit_path = joinpath("classifications_dir" ,"best_fit", "best_fit_" * string(SN_ID) * ".txt")
        target_spectra_path = joinpath("classifications_dir" , "target_spectra", "target_spectra_" * string(SN_ID) * ".txt")
        writedlm(best_fit_path, best_fit)
        writedlm(target_spectra_path, altered_spectra)

        # Export the best fit parameters as metadata
        best_fit_metadata_path = joinpath("classifications_dir", "metadata", "best_fit_" * string(SN_ID) * "_metadata.json")
        best_fit_metadata = OrderedDict("SN Name" => pure_sn_spectra_file_names[best_indices[1]], "HG Name" => pure_hg_spectra_file_names[best_indices[2]], "SN Norm" => best_norm_params[1], "HG Norm" => best_norm_params[2], "Chi Squared" => best_chi, "Chi Squared Reduced" => best_chi_red, "Identified" => identified, "Identified Type" => identified_type)
        JSON.write(best_fit_metadata_path, JSON.json(best_fit_metadata))

        sn_best_fit_name = pure_sn_spectra_file_names[best_indices[1]]
        hg_best_fit_name = pure_hg_spectra_file_names[best_indices[2]]

        return sn_best_fit_name, hg_best_fit_name, best_norm_params, identified, best_chi_red
    end

    export delete_folder_contents, adapative_binning, chi_squared_classifier, fit_sn_hg, fitting_func, filter_flux
end