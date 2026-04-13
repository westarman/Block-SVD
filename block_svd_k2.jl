using LinearAlgebra, Images, Plots, Statistics, FileIO, Printf

# ─ helpers ─

function load_grayscale(path::String)::Matrix{Float64}
    img = load(path)
    return Float64.(Gray.(img))
end

function save_grayscale(path::String, mat::Matrix{Float64})
    save(path, Gray.(clamp.(mat, 0.0, 1.0)))
end

# ─ core compression ─
"""
    split_blocks(A, k) → (blocks, (m_crop, n_crop))

Slice a matrix into non-overlapping k×k tiles (edge-crops if not divisible).
"""
function split_blocks(A::Matrix{Float64}, k::Int)
    m, n   = size(A)
    mc, nc = m - mod(m, k), n - mod(n, k)
    A_crop = A[1:mc, 1:nc]
    blocks = [A_crop[i:i+k-1, j:j+k-1]
              for i in 1:k:mc for j in 1:k:nc]
    return blocks, (mc, nc)
end

"""
    reconstruct(blocks, dims, k) → Matrix

Stitch tiles back into a full image.
"""
function reconstruct(blocks::Vector{Matrix{Float64}},
                     dims::Tuple{Int,Int}, k::Int)
    m, n = dims
    out  = zeros(m, n)
    idx  = 1
    for i in 1:k:m, j in 1:k:n
        out[i:i+k-1, j:j+k-1] = blocks[idx]
        idx += 1
    end
    return out
end

"""
    compress_block(block, r) → Matrix

Low-rank rank-r SVD approximation of a single k×k block.
"""
function compress_block(block::Matrix{Float64}, r::Int)
    F   = svd(block)
    r   = min(r, length(F.S))          # guard against r > min(k,k)
    U_r = F.U[:, 1:r]
    Σ_r = Diagonal(F.S[1:r])
    V_r = F.V[:, 1:r]
    return U_r * Σ_r * V_r'
end

"""
    compress_image(img, k, r) → Matrix

Block-wise rank-r SVD compression of the full image.
"""
function compress_image(img::Matrix{Float64}, k::Int, r::Int)
    blocks, dims = split_blocks(img, k)
    recon_blocks = [compress_block(b, r) for b in blocks]
    return reconstruct(recon_blocks, dims, k)
end

# ─ metrics ─

function psnr(orig::Matrix{Float64}, recon::Matrix{Float64})
    # crop orig to recon size (due to edge-crop)
    oh, ow = size(recon)
    orig_c = orig[1:oh, 1:ow]
    mse    = mean((orig_c .- recon) .^ 2)
    mse == 0 && return Inf
    return 20 * log10(1.0 / sqrt(mse))
end

function singular_value_energy(block::Matrix{Float64})
    F = svd(block)
    energy = F.S .^ 2
    return energy / sum(energy)
end

function plot_singular_energy(block)
    energy = singular_value_energy(block)
    plot(energy,
        marker=:circle,
        xlabel="Index",
        ylabel="Energy contribution",
        title="Singular value energy distribution",
        legend=false)
end

"""
Estimated compression ratio (bytes):
  original: 1 byte/pixel (8-bit grayscale)
  compressed: for each block, store U (k×r), Σ (r,), V (k×r) as Float64
"""
function compression_ratio(img_size::Tuple{Int,Int}, k::Int, r::Int)
    m, n        = img_size
    mc, nc      = m - mod(m, k), n - mod(n, k)
    n_blocks    = (mc ÷ k) * (nc ÷ k)
    orig_bytes = 8 * mc * nc
    # U: k*r, Σ: r, V: k*r  →  (2k*r + r) Float64s = 8*(2kr+r) bytes
    comp_bytes  = n_blocks * 8 * (2*k*r + r)
    return orig_bytes / comp_bytes
end

# ─ experiment ─

"""
    run_experiment(img, k_list, r_fracs) → DataFrame-like NamedTuple vector

For every (k, r) combination (where r goes from 1 to k÷2), compute
compression ratio and PSNR and return results as a vector of named tuples.
"""
function run_experiment(img::Matrix{Float64},
                        k_list::Vector{Int})
    results = NamedTuple[]
    for k in k_list
        r_max = k ÷ 2
        for r in 1:r_max
            recon   = compress_image(img, k, r)
            cr      = compression_ratio(size(img), k, r)
            p       = psnr(img, recon)
            push!(results, (k=k, r=r, cr=cr, psnr=p))
        end
    end
    return results
end

# ─ plotting  ─
"""
Plot PSNR vs. compression ratio, one curve per k value.
"""
function plot_psnr_vs_cr(results, output_path="psnr_vs_cr.png")
    k_vals = sort(unique(r.k for r in results))
    plt    = plot(xlabel="Compression ratio",
                  ylabel="PSNR (dB)",
                  title="PSNR vs compression ratio",
                  legend=:bottomright,
                  size=(700, 450))
    for k in k_vals
        sub  = filter(r -> r.k == k, results)
        crs  = [r.cr   for r in sub]
        psnrs = [r.psnr for r in sub]
        # sort by CR so lines don't cross themselves
        ord  = sortperm(crs)
        plot!(plt, crs[ord], psnrs[ord],
              marker=:circle, label="k=$k")
    end
    savefig(plt, output_path)
    println("Saved: $output_path")
    return plt
end

"""
Plot PSNR vs. rank r, one curve per k value.
"""
function plot_psnr_vs_rank(results, output_path="psnr_vs_rank.png")
    k_vals = sort(unique(r.k for r in results))
    plt    = plot(xlabel="Rank r",
                  ylabel="PSNR (dB)",
                  title="PSNR vs rank r",
                  legend=:bottomright,
                  size=(700, 450))
    for k in k_vals
        sub   = sort(filter(r -> r.k == k, results), by=r -> r.r)
        rs    = [r.r    for r in sub]
        psnrs = [r.psnr for r in sub]
        plot!(plt, rs, psnrs, marker=:circle, label="k=$k")
    end
    savefig(plt, output_path)
    println("Saved: $output_path")
    return plt
end

"""
Visual grid: rows = k values, columns = selected r values.
"""
function plot_reconstruction_grid(img::Matrix{Float64},
                                  k_list::Vector{Int},
                                  output_path="reconstruction_grid.png")
    n_cols = 5   # original + 4 reconstructions per row
    n_rows = length(k_list)
    panels = []

    for k in k_list
        r_max = k ÷ 2
        rs = unique(clamp.(round.(Int, [1, k÷8, k÷4, k÷2]), 1, k÷2))
        # first column: original
        push!(panels, heatmap(reverse(img, dims=1),
                              title="k=$k  Original",
                              c=:grays, aspect_ratio=:equal,
                              axis=nothing, colorbar=false,
                              titlefontsize=7))

        # next columns: reconstructions for selected r values
        for r in rs
            recon = compress_image(img, k, r)
            cr    = compression_ratio(size(img), k, r)
            p     = psnr(img, recon)
            push!(panels, heatmap(reverse(recon, dims=1),
                                  title="k=$k r=$r\nCR=$(round(cr,digits=1))× $(round(p,digits=0))dB",
                                  c=:grays, aspect_ratio=:equal,
                                  axis=nothing, colorbar=false,
                                  titlefontsize=7))
        end

        # pad with blank panels if fewer than 4 r values were generated
        while length(panels) % n_cols != 0
            push!(panels, plot(grid=false, axis=nothing, border=:none))
        end
    end

    plt = plot(panels...,
               layout=(n_rows, n_cols),
               size=(300*n_cols, 300*n_rows))
    savefig(plt, output_path)
    println("Saved: $output_path")
    return plt
end

function main()
    # ─ config ─
    # change to your image path
    img_path = "/home/milicazd/Documents/milica/2. letnik/mm/mm_project/images/my_image.png"
    k_list   = [8, 16, 32, 64]
    out_dir  = "output"
    mkpath(out_dir)

    # ─ load ─
    println("Loading $img_path ...")
    img = load_grayscale(img_path)
    println("Image size: $(size(img))")

    # Analyze singular value energy on one sample block
    k = 8
    sample_block = img[1:k, 1:k]
    energy = singular_value_energy(sample_block)

    println("\nSingular value energy (first 8x8 block):")
    println(round.(energy, digits=4))

    display(plot_singular_energy(sample_block))

    # ─ run ─
    println("\nRunning experiment (k = $k_list, r = 1 … k÷2) …")
    results = run_experiment(img, k_list)

    # ─ print table ─
    println("\n  k    r     CR      PSNR (dB)")
    println("  ", "-"^36)
    for res in results
        @printf("  %-4d %-5d %-7.2f %.2f\n", res.k, res.r, res.cr, res.psnr)
    end

    # ─ plots ─
    println()
    plot_psnr_vs_cr(results,   joinpath(out_dir, "psnr_vs_cr.png"))
    plot_psnr_vs_rank(results, joinpath(out_dir, "psnr_vs_rank.png"))
    plot_reconstruction_grid(img, [8, 16, 32],
                             joinpath(out_dir, "reconstruction_grid.png"))

    println("\nDone. Results saved to $(out_dir)/")
end

main()