using LinearAlgebra, Images, Statistics, Plots, FileIO
using ImageQualityIndexes
using ProgressMeter
using CSV, DataFrames

include("prepare_image.jl")

function load_grayscale_safe(path::String)::Matrix{Float64}
    """Load image with proper error handling"""
    if !isfile(path)
        error("File not found: $path")
    end
    
    try
        img = load(path)
        return Float64.(Gray.(img))
    catch e
        error("Failed to load image $path: $e")
    end
end

# split image into non-overlapping k*k blocks
function split_blocks(A::Matrix{Float64}, k::Int)
    m, n = size(A)
    m_crop = m - mod(m, k)
    n_crop = n - mod(n, k)
    
    if m_crop == 0 || n_crop == 0
        error("Image too small for block size k=$k. Image size: $(size(A))")
    end

    A_crop = A[1:m_crop, 1:n_crop]
    blocks = Matrix{Float64}[]

    for i in 1:k:m_crop
        for j in 1:k:n_crop
            push!(blocks, A_crop[i:i+k-1, j:j+k-1])
        end
    end

    return blocks, (m_crop, n_crop)
end

function reconstruct(blocks, dims, k)
    m, n = dims
    recon = zeros(m, n)
    idx = 1
    for i in 1:k:m
        for j in 1:k:n
            recon[i:i+k-1, j:j+k-1] = blocks[idx]
            idx += 1
        end
    end
    return recon
end

# adaptive block compression (choose r based on q)
# we can try multiple block sizes and then pick best compression 

function compress_adaptive(block::Matrix{Float64}, q::Float64)
    """Adaptive rank selection with numerical stability"""
    
    # Validate q
    if q <= 0 || q > 1
        @warn "q=$q out of range (0,1], clamping to 0.99"
        q = min(max(q, 0.01), 0.99)
    end
    
    # Handle constant blocks
    if all(block .≈ block[1,1])
        # Constant block - rank 1 approximation is perfect
        try
            F = svd(block)
            return (F.U[:, 1:1], F.S[1:1], F.V[:, 1:1]), 1
        catch
            # Fallback for problematic constant blocks
            k = size(block, 1)
            U = ones(k, 1) ./ sqrt(k)
            Σ = [sum(block) * sqrt(k)]
            V = ones(k, 1) ./ sqrt(k)
            return (U, Σ, V), 1
        end
    end
    
    try
        F = svd(block)
        total = sum(F.S .^ 2)
        
        if total == 0
            # Zero block
            k = size(block, 1)
            return (zeros(k, 1), [0.0], zeros(k, 1)), 1
        end
        
        # Find rank needed to preserve q fraction of energy
        energy_ratio = cumsum(F.S .^ 2) / total
        r = findfirst(energy_ratio .>= q)
        
        if isnothing(r)
            r = length(F.S)
        end
        
        # Ensure r is at least 1 and not greater than matrix dimensions
        r = max(1, min(r, length(F.S)))
        
        return (F.U[:, 1:r], F.S[1:r], F.V[:, 1:r]), r
        
    catch e
        @warn "SVD failed for block: $e, using rank 1 approximation"
        # Fallback to rank 1 approximation using first row/column
        k = size(block, 1)
        U = block[:, 1:1] ./ norm(block[:, 1])
        Σ = [norm(block[:, 1])]
        V = ones(k, 1) ./ sqrt(k)
        return (U, Σ, V), 1
    end
end

function decompress_block(U_r, Σ_r, V_r)
    return U_r * diagm(Σ_r) * V_r'
end

# whole image compression / decompression

function compress_image(img::Matrix{Float64}, k::Int, q::Float64)
    blocks, dims = split_blocks(img, k)
    comp_blocks = Vector{Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}}}() #maybe we should consider something different
    ranks = Int[]
    for blk in blocks
        comp, r = compress_adaptive(blk, q)
        push!(comp_blocks, comp)
        push!(ranks, r)
    end

    return (comp_blocks, ranks, dims, k)
    
end

function decompress_image(compressed)
    comp_blocks, ranks, dims, k = compressed
    recon_blocks = Matrix{Float64}[]
    for (comp, r) in zip(comp_blocks, ranks)
        U, Σ, V = comp
        push!(recon_blocks, U * diagm(Σ) * V')
    end

    return reconstruct(recon_blocks, dims, k)
end

# compression ratio (bytes)

function compression_ratio(compressed, original_size)
    m, n = original_size
    comp_blocks, ranks, dims, k = compressed
    mc, nc = dims
    
    # Original stored as uint8 (1 byte per pixel)
    orig_bytes = mc * nc * 1  
    
    # SVD factors stored as float64 (8 bytes per value)
    # To simulate a fairer comparison, use float16 (2 bytes)
    bytes_per_value = 2  # float16
    total_bytes = sum(r -> bytes_per_value * (2*k*r + r), ranks)
    total_bytes += length(ranks)  # rank overhead
    
    return orig_bytes / total_bytes
end

# quality metrics

function psnr(orig::Matrix{Float64}, recon::Matrix{Float64})
    """Compute PSNR after cropping to common dimensions"""
    # Crop to common dimensions
    h1, w1 = size(orig)
    h2, w2 = size(recon)
    min_h = min(h1, h2)
    min_w = min(w1, w2)
    
    orig_cropped = orig[1:min_h, 1:min_w]
    recon_cropped = recon[1:min_h, 1:min_w]
    
    mse = mean((orig_cropped .- recon_cropped) .^ 2)
    if mse == 0
        return Inf
    end
    return 20 * log10(1.0 / sqrt(mse))
end

function compute_ssim_safe(img::Matrix{Float64}, recon::Matrix{Float64})
    """Compute SSIM after cropping both images to same dimensions"""
    h1, w1 = size(img)
    h2, w2 = size(recon)
    min_h = min(h1, h2)
    min_w = min(w1, w2)
    
    img_cropped = img[1:min_h, 1:min_w]
    recon_cropped = recon[1:min_h, 1:min_w]
    
    try
        return assess_ssim(img_cropped, recon_cropped)
    catch e
        @warn "SSIM computation failed: $e"
        return NaN
    end
end

function run_experiment(img_path, k_list, q_list)
    """Run experiments with comprehensive error handling"""
    
    # Check if image exists
    if !isfile(img_path)
        error("Image file not found: $img_path")
    end
    
    # Load image with error handling
    try
        img = load_grayscale_safe(img_path)
        println("✓ Image loaded successfully. Size: $(size(img))")
    catch e
        error("Failed to load image: $e")
    end
    
    img = load_grayscale_safe(img_path)
    results = []
    
    # Progress bar for long experiments
    @showprogress for k in k_list
        for q in q_list
            println("\nRunning k=$k, q=$q")
            
            try
                comp = compress_image(img, k, q)
                recon = decompress_image(comp)
                
                cr = compression_ratio(comp, size(img))
                psnr_val = psnr(img, recon)
                ssim_val = compute_ssim_safe(img, recon)
                avg_rank = mean(comp[2])
                
                push!(results, (
                    k=k, 
                    q=q, 
                    cr=cr, 
                    psnr=psnr_val, 
                    ssim=ssim_val, 
                    avg_rank=avg_rank
                ))
                
                println("  ✓ CR=$(round(cr, digits=2)), PSNR=$(round(psnr_val, digits=2))dB, SSIM=$(round(ssim_val, digits=4)), Avg Rank=$(round(avg_rank, digits=2))")
                
            catch e
                @warn "Failed for k=$k, q=$q: $e"
                push!(results, (k=k, q=q, cr=NaN, psnr=NaN, ssim=NaN, avg_rank=NaN))
            end
        end
    end
    
    return results, img
end

# plot results

function plot_results(results)
    """Plot PSNR and SSIM vs compression ratio"""
    k_vals = unique([r.k for r in results if !isnan(r.cr)])
    
    p1 = plot(xlabel="Compression ratio", ylabel="PSNR (dB)", 
              title="PSNR vs CR", legend=:bottomright)
    p2 = plot(xlabel="Compression ratio", ylabel="SSIM", 
              title="SSIM vs CR", legend=:bottomright)
    
    for k in k_vals
        kres = filter(r -> r.k == k && !isnan(r.cr), results)
        if !isempty(kres)
            # Sort by CR for clean lines
            sorted = sort(kres, by=r -> r.cr)
            crs = [r.cr for r in sorted]
            psnrs = [r.psnr for r in sorted]
            ssims = [r.ssim for r in sorted]
            
            plot!(p1, crs, psnrs, marker=:circle, label="k=$k", linewidth=2)
            plot!(p2, crs, ssims, marker=:circle, label="k=$k", linewidth=2)
        end
    end
    
    plot(p1, p2, layout=(2,1), size=(600,800))
end

function plot_rank_distribution(ranks, k, q)
    """Plot histogram of rank distribution"""
    histogram(ranks, bins=1:k,
        xlabel="Rank r",
        ylabel="Frequency",
        title="Rank distribution (k=$k, q=$q)",
        legend=false,
        color=:steelblue,
        alpha=0.7)
end


function rank_heatmap(ranks, dims, k, q)
    """Create heatmap showing rank per block"""
    m, n = dims
    rows = div(m, k)
    cols = div(n, k)
    
    # Reshape ranks into grid
    R = reshape(ranks, cols, rows)'
    
    heatmap(R,
        title="Rank map (k=$k, q=$q)",
        xlabel="Block column",
        ylabel="Block row",
        clims=(0, k),
        colorbar_title="Rank r",
        aspect_ratio=:equal)
end

# show reconstructed image for one configuration

function show_reconstructed(img_path, k, q)
    """Display reconstructed image with rank distribution"""
    try
        img = load_grayscale_safe(img_path)
        comp = compress_image(img, k, q)
        comp_blocks, ranks, dims, _ = comp
        recon = decompress_image(comp)
        cr = compression_ratio(comp, size(img))
        psnr_val = psnr(img, recon)
        
        println("\n=== Reconstruction Results ===")
        println("k=$k, q=$q")
        println("CR = $(round(cr, digits=2))×")
        println("PSNR = $(round(psnr_val, digits=2)) dB")
        println("Avg rank = $(round(mean(ranks), digits=2))")
        println("Min rank = $(minimum(ranks)), Max rank = $(maximum(ranks))")
        
        # Create visualization
        p1 = heatmap(reverse(img, dims=1), 
                     title="Original", 
                     c=:grays, aspect_ratio=:equal, 
                     axis=nothing, colorbar=false)
        
        p2 = heatmap(reverse(recon, dims=1), 
                     title="Reconstructed (k=$k, q=$q)", 
                     c=:grays, aspect_ratio=:equal, 
                     axis=nothing, colorbar=false)
        
        plot(p1, p2, layout=(1,2), size=(800,400))
        
        # Show rank distribution
        display(plot_rank_distribution(ranks, k, q))
        display(rank_heatmap(ranks, dims, k, q))
        
    catch e
        @error "Failed to show reconstruction: $e"
    end
end

# funciton that plots a grid of reconstructed images for different k and q combinations

function show_quality_grid(img_path, k_list, q_list)
    """Create grid of reconstructed images for different parameters"""
    img = load_grayscale_safe(img_path)
    plots = []
    
    for k in k_list
        for q in q_list
            try
                comp = compress_image(img, k, q)
                recon = decompress_image(comp)
                cr = compression_ratio(comp, size(img))
                psnr_val = psnr(img, recon)
                
                p = heatmap(reverse(recon, dims=1), 
                    title="k=$k, q=$q\nCR=$(round(cr,digits=2))× PSNR=$(round(psnr_val,digits=1))dB",
                    c=:grays, aspect_ratio=:equal, axis=nothing, colorbar=false,
                    titlefontsize=7)
                push!(plots, p)
            catch e
                @warn "Failed for k=$k, q=$q: $e"
                # Add blank plot for failed case
                p = plot(title="Failed: k=$k, q=$q", grid=false, axis=nothing)
                push!(plots, p)
            end
        end
    end
    
    # Add original in top-left
    p_orig = heatmap(reverse(img, dims=1), title="Original",
        c=:grays, aspect_ratio=:equal, axis=nothing, colorbar=false,
        titlefontsize=7)
    pushfirst!(plots, p_orig)
    
    n_cols = length(q_list) + 1
    n_rows = length(k_list)
    plot(plots..., layout=(n_rows, n_cols), size=(300*n_cols, 300*n_rows))
end

function save_results_to_csv(results, output_dir)
    """Save experiment results to CSV file"""
    try
        df = DataFrame(results)
        csv_path = joinpath(output_dir, "experiment_results.csv")
        CSV.write(csv_path, df)
        println("✓ Results saved to $csv_path")
        return true
    catch e
        @warn "Could not save CSV: $e"
        return false
    end
end

function main()
    println("=== Adaptive SVD Image Compression ===\n")
    
    output_dir = "output"
    mkpath(output_dir)
    
    img_path = "images/my_image.png"
    k_list = [8, 16, 32]  # Removed 64 for speed, can add back
    q_list = [0.7, 0.85, 0.9, 0.95, 0.99]  # Reduced list for faster testing
    
    # Validate inputs
    if !isfile(img_path)
        println("ERROR: Image not found at '$img_path'")
        println("Please update img_path to point to your image file.")
        return
    end
    
    println("Image path: $img_path")
    println("Block sizes: $k_list")
    println("Quality factors: $q_list\n")
    
    try
        # Run experiment
        results, original_img = run_experiment(img_path, k_list, q_list)
        
        # Save results
        save_results_to_csv(results, output_dir)
        
        # Print summary table
        println("\n=== Summary Results ===")
        println(" k    q     CR     PSNR(dB)  SSIM    AvgRank")
        println("-" ^ 55)
        for r in results
            if !isnan(r.cr)
                println(lpad(r.k,3), " ", lpad(r.q,4), " ", 
                        lpad(round(r.cr, digits=2),5), " ",
                        lpad(round(r.psnr, digits=2),8), " ", 
                        lpad(round(r.ssim, digits=4),6), " ",
                        lpad(round(r.avg_rank, digits=2),6))
            else
                println(lpad(r.k,3), " ", lpad(r.q,4), "     FAILED")
            end
        end
        
        # Generate plots
        println("\n=== Generating Plots ===")
        
        p = plot_results(results)
        savefig(p, joinpath(output_dir, "rate_distortion.png"))
        println("✓ Saved: rate_distortion.png")
        
        show_reconstructed(img_path, 16, 0.95)
        savefig(joinpath(output_dir, "reconstruction_example.png"))
        println("✓ Saved: reconstruction_example.png")
        
        grid_plot = show_quality_grid(img_path, [8, 32], [0.85, 0.95, 0.99])
        savefig(joinpath(output_dir, "quality_grid.png"))
        println("✓ Saved: quality_grid.png")
        
        println("\n=== Done! ===")
        println("All results saved to '$output_dir/' directory")
        
    catch e
        println("\nERROR in main(): $e")
        println(stacktrace(catch_backtrace()))
    end
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end