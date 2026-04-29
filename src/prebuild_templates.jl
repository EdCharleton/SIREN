#!/usr/bin/env julia
# prebuild_templates.jl — Build master template matrices and save as JLD2
# Run this script once whenever templates are added or modified.
# Usage: julia --project=. src/prebuild_templates.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DelimitedFiles, CSV, DataFrames, JSON, Statistics, JLD2, ProgressBars
using DataStructures: OrderedDict

println("============================================")
println("  SIREN+ Template Pre-Builder")
println("============================================")

# ── Master wavelength grid (must match the template spacing) ──
const MASTER_GRID = collect(Float32, 2315.0:15.0:9005.0)
const N_GRID = length(MASTER_GRID)
const OUTPUT_PATH = joinpath(@__DIR__, "..", "prebuilt_templates.jld2")

# ── Load metadata ──
println("Loading metadata...")
sn_meta = JSON.parsefile(
    joinpath("transient_template_dir", "metadata", "pure_sn_spectra_data.json"),
    dicttype = OrderedDict)
sn_names = open(readlines,
    joinpath("transient_template_dir", "metadata", "pure_sn_spectra_names.txt"))

hg_meta = JSON.parsefile(
    joinpath("host_template_dir", "metadata", "pure_hg_spectra_data.json"),
    dicttype = OrderedDict)
hg_names = open(readlines,
    joinpath("host_template_dir", "metadata", "pure_hg_spectra_names.txt"))

N_T = length(sn_names)
N_H = length(hg_names)
println("  Transient templates : $N_T")
println("  Host templates      : $N_H")
println("  Master grid points  : $N_GRID  ($(MASTER_GRID[1]) – $(MASTER_GRID[end]) Å)")

# ── Helper: place a template onto the master grid ──
function place_on_grid!(matrix, mask, col_idx, λ_vec, flux_vec)
    for j in eachindex(λ_vec)
        wl = λ_vec[j]
        fl = flux_vec[j]
        # Compute grid index arithmetically (all templates are on 15 Å spacing)
        grid_idx = round(Int, (wl - 2315.0) / 15.0) + 1
        if 1 <= grid_idx <= N_GRID && isfinite(fl)
            matrix[grid_idx, col_idx] = Float32(fl)
            mask[grid_idx, col_idx] = 1.0f0
        end
    end
end

# ── Build transient master matrices ──
println("\nBuilding transient master matrix...")
T_master  = zeros(Float32, N_GRID, N_T)
MT_master = zeros(Float32, N_GRID, N_T)

# Read and scale all transient templates
sn_flux_raw = Vector{Vector{Float64}}(undef, N_T)
sn_λ_raw    = Vector{Vector{Float64}}(undef, N_T)

Threads.@threads for i in ProgressBar(1:N_T)
    key = sn_names[i]
    path = joinpath(sn_meta["path"][key], key)
    content = CSV.read(path, DataFrame, delim=",", header=false)
    sn_λ_raw[i]    = Vector{Float64}(content[:, 1])
    sn_flux_raw[i] = Vector{Float64}(content[:, 2])
end

# Scale templates (same as read_template_files: divide by mean, multiply by 100)
println("Scaling transient templates...")
for i in 1:N_T
    finite_flux = filter(isfinite, sn_flux_raw[i])
    m = isempty(finite_flux) ? 0.0 : mean(finite_flux)
    if m != 0
        sn_flux_raw[i] .= sn_flux_raw[i] ./ m .* 100
    end
end

# Place onto master grid
println("Placing transient templates onto master grid...")
for i in ProgressBar(1:N_T)
    place_on_grid!(T_master, MT_master, i, sn_λ_raw[i], sn_flux_raw[i])
end

# ── Build host master matrices ──
println("\nBuilding host master matrix...")
H_master  = zeros(Float32, N_GRID, N_H)
MH_master = zeros(Float32, N_GRID, N_H)

hg_flux_raw = Vector{Vector{Float64}}(undef, N_H)
hg_λ_raw    = Vector{Vector{Float64}}(undef, N_H)

for i in ProgressBar(1:N_H)
    key = hg_names[i]
    path = joinpath(hg_meta["path"][key], key)
    content = CSV.read(path, DataFrame, delim=",", header=false)
    hg_λ_raw[i]    = Vector{Float64}(content[:, 1])
    hg_flux_raw[i] = Vector{Float64}(content[:, 2])
end

# Scale host templates
println("Scaling host templates...")
for i in 1:N_H
    finite_flux = filter(isfinite, hg_flux_raw[i])
    m = isempty(finite_flux) ? 0.0 : mean(finite_flux)
    if m != 0
        hg_flux_raw[i] .= hg_flux_raw[i] ./ m .* 100
    end
end

# Place onto master grid
println("Placing host templates onto master grid...")
for i in ProgressBar(1:N_H)
    place_on_grid!(H_master, MH_master, i, hg_λ_raw[i], hg_flux_raw[i])
end

# ── Summary statistics ──
sn_coverage = [count(MT_master[:, i] .> 0) for i in 1:N_T]
hg_coverage = [count(MH_master[:, i] .> 0) for i in 1:N_H]
println("\n── Coverage Statistics ──")
println("  SN templates: min=$(minimum(sn_coverage)), median=$(Int(median(sn_coverage))), max=$(maximum(sn_coverage)) grid points")
println("  HG templates: min=$(minimum(hg_coverage)), median=$(Int(median(hg_coverage))), max=$(maximum(hg_coverage)) grid points")

# ── Save ──
println("\nSaving to $OUTPUT_PATH ...")
jldsave(OUTPUT_PATH;
    master_grid  = MASTER_GRID,
    T_master     = T_master,
    MT_master    = MT_master,
    H_master     = H_master,
    MH_master    = MH_master,
    sn_names     = sn_names,
    hg_names     = hg_names,
)

filesize_mb = round(filesize(OUTPUT_PATH) / 1024^2, digits=1)
println("Done! Saved $filesize_mb MB to $OUTPUT_PATH")
println("============================================")
