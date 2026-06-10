#include "gs_renderer.h"

#include <fstream>
#include <memory>
#include <csignal>
#include <cctype>
#include <filesystem>
#include <iterator>
#include <optional>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace {

std::optional<uint32_t> readJsonUint(const std::string& json, const std::string& key) {
    std::string token = "\"" + key + "\"";
    size_t pos = json.find(token);
    if (pos == std::string::npos)
        return std::nullopt;
    pos = json.find(':', pos + token.size());
    if (pos == std::string::npos)
        return std::nullopt;
    ++pos;
    while (pos < json.size() && std::isspace((unsigned char)json[pos]))
        ++pos;
    if (pos >= json.size() || !std::isdigit((unsigned char)json[pos]))
        return std::nullopt;
    uint32_t value = 0;
    while (pos < json.size() && std::isdigit((unsigned char)json[pos])) {
        value = value * 10 + (uint32_t)(json[pos] - '0');
        ++pos;
    }
    return value;
}

void validateShaderConfigManifest(const std::map<std::string, std::string>& spirv_paths) {
    auto projection = spirv_paths.find("projection_forward");
    if (projection == spirv_paths.end())
        return;

    std::filesystem::path manifest_path =
        std::filesystem::path(projection->second).parent_path() / "shader_config.json";
    std::ifstream manifest(manifest_path);
    if (!manifest)
        return;

    std::string json(
        (std::istreambuf_iterator<char>(manifest)),
        std::istreambuf_iterator<char>()
    );

    auto check = [&](const std::string& key, uint32_t expected) {
        std::optional<uint32_t> actual = readJsonUint(json, key);
        if (!actual)
            throw std::runtime_error("Shader config manifest is missing `" + key + "`: " + manifest_path.string());
        if (*actual != expected) {
            throw std::runtime_error(
                "Shader config mismatch for `" + key + "`: C++=" +
                std::to_string(expected) + ", shader=" + std::to_string(*actual) +
                ". Regenerate shaders from vksplat_config.h."
            );
        }
    };

    check("subgroup_size", VKSPLAT_SUBGROUP_SIZE);
    check("tile_height", VKSPLAT_TILE_HEIGHT);
    check("tile_width", VKSPLAT_TILE_WIDTH);
    check("tile_size", VKSPLAT_TILE_SIZE);
    check("sh_reorder_size", VKSPLAT_SH_REORDER_SIZE);
    check("use_emulated_int64", VKSPLAT_USE_EMULATED_INT64);
    check("use_emulated_f32_atomic", VKSPLAT_USE_EMULATED_F32_ATOMIC);
    check("sorting_key_bits", VKSPLAT_SORTING_KEY_BITS);
    check("rect_tile_space_words", VKSPLAT_RECT_TILE_SPACE_WORDS);
    check("shader_requires_int64", vksplat_config::kShaderRequiresInt64 ? 1u : 0u);
    check("radix_sort_radix", VKSPLAT_RADIX_SORT_RADIX);
    check("radix_workgroup_size", VKSPLAT_RADIX_WORKGROUP_SIZE);
    check("radix_partition_division", VKSPLAT_RADIX_PARTITION_DIVISION);
    check("radix_partition_size", VKSPLAT_RADIX_PARTITION_SIZE);
    check("tensor_backward_config_count", VKSPLAT_TENSOR_BWD_CONFIG_COUNT);
}

}  // namespace


VulkanGSRenderer::VulkanGSRenderer()
    : VulkanGSPipeline() {
}

VulkanGSRenderer::~VulkanGSRenderer() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}

void VulkanGSRenderer::cleanup() {
    VulkanGSPipeline::cleanup();
}


VulkanGSPipeline::DeviceRequirement VulkanGSRenderer::getDeviceRequirement() {
    const uint32_t minSharedMemory = ([=]() -> uint32_t {
        constexpr uint32_t tile_size = VKSPLAT_TILE_SIZE;
        uint32_t raster_backward_0 = tile_size*(9+1+1)*sizeof(float);  // per pixel
        uint32_t raster_backward_1 = tile_size*(4+4+1)*sizeof(float);  // per splat
        uint32_t ssim_shared_x = VKSPLAT_SSIM_BLOCK_X + 2 * VKSPLAT_SSIM_HALO;
        uint32_t ssim_shared_y = VKSPLAT_SSIM_BLOCK_Y + 2 * VKSPLAT_SSIM_HALO;
        uint32_t ssim = (
            ssim_shared_x * ssim_shared_y * (2+3) +
            VKSPLAT_TILE_SIZE * (5+3)
        ) * sizeof(float);
        uint32_t radix_sort_rts =
            VKSPLAT_RADIX_PARTITION_SIZE * (VKSPLAT_SORTING_KEY_BITS / 32) * sizeof(uint32_t) +
            VKSPLAT_RADIX_SORT_RADIX * sizeof(uint32_t);
        uint32_t tensor_bwd = 0;
        for (const auto& config : vksplat_config::kTensorBwdConfigs)
            tensor_bwd = std::max(tensor_bwd, config.min_shared_mem);
        return std::max({
            raster_backward_0, raster_backward_1, tensor_bwd, ssim, radix_sort_rts
        });
    })();
    return DeviceRequirement{
        { 12*16777216/VKSPLAT_TILE_SIZE, 4096/std::min(VKSPLAT_TILE_HEIGHT,VKSPLAT_TILE_WIDTH), 1 },
        { std::max(VKSPLAT_CUMSUM_BLOCK_SIZE, VKSPLAT_MORTON_APPLY_THREADS), 1*std::max(VKSPLAT_TILE_HEIGHT,VKSPLAT_TILE_WIDTH), 1 },
        minSharedMemory
    };
}


void VulkanGSRenderer::initialize(const std::map<std::string, std::string> &spirv_paths, int device_id) {

    VulkanGSPipeline::initialize(device_id);
    validateShaderConfigManifest(spirv_paths);
    
    createComputePipeline(pipeline_projection_forward, spirv_paths.at("projection_forward"));
    createComputePipeline(pipeline_generate_keys, spirv_paths.at("generate_keys"));
    for (int i = 0; i < 2; ++i) {
        createComputePipeline(pipeline_compute_tile_ranges[i], spirv_paths.at("compute_tile_ranges"));
        createComputePipeline(pipeline_rasterize_forward[i], spirv_paths.at("rasterize_forward"));
        createComputePipeline(pipeline_rasterize_backward[0][i], spirv_paths.at("rasterize_backward_0"));
        createComputePipeline(pipeline_rasterize_backward[1][i], spirv_paths.at("rasterize_backward_1"));
        for (size_t tensor_idx = 0; tensor_idx < vksplat_config::kTensorBwdConfigs.size(); ++tensor_idx) {
            size_t impl_idx = tensor_idx + 2;
            createComputePipeline(
                pipeline_rasterize_backward[impl_idx][i],
                spirv_paths.at("rasterize_backward_" + std::to_string(impl_idx)),
                vksplat_config::kTensorBwdConfigs[tensor_idx].min_shared_mem
            );
        }
    }
    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"));
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"));
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"));
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"));
    createComputePipeline(pipeline_sorting_1.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_1.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_1.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sorting_2.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_2.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_2.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sum, spirv_paths.at("sum"));
    createComputePipeline(pipeline_where, spirv_paths.at("where"));

}


void VulkanGSRenderer::executeProjectionForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t alloc_reserve
) {
    PerfTimer::Timer<PerfTimer::ProjectionForward> timer(this);
    DEVICE_GUARD;

    size_t num_splats = buffers.num_splats;

    bufferMemoryBarrier({
        { buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.sh_coeffs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
        { buffers.scales_opacs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    size_t alloc_size = std::max(num_splats, alloc_reserve);
    executeCompute(
        {{num_splats, VKSPLAT_SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_projection_forward,
        {
            // inputs
            buffers.xyz_ws.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.scales_opacs.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.tiles_touched, alloc_size),
            resizeDeviceBuffer(buffers.rect_tile_space, VKSPLAT_RECT_TILE_SPACE_WORDS * alloc_size),
            resizeDeviceBuffer(buffers.radii, alloc_size),
            resizeDeviceBuffer(buffers.xy_vs, 2*alloc_size),
            resizeDeviceBuffer(buffers.depths, alloc_size),
            resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4*alloc_size),
            resizeDeviceBuffer(buffers.rgb, 3*alloc_size),
        }
    );

}

void VulkanGSRenderer::executeGenerateKeys(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    PerfTimer::Timer<PerfTimer::GenerateKeys> timer(this);
    DEVICE_GUARD;

    size_t num_elements = buffers.num_splats;
    size_t num_indices = buffers.num_indices;

    // barrier shouldn't be needed as this is after cumsum and read element
    #if 0
    bufferMemoryBarrier({
        { buffers.xy_vs.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.inv_cov_vs_opacity.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rect_tile_space.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);
    #endif

    executeCompute(
        {{num_elements, VKSPLAT_TILE_SHADER_GENERATE_KEYS_BLOCK_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_generate_keys,
        {
            // inputs
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.depths.deviceBuffer,
            buffers.rect_tile_space.deviceBuffer,
            buffers.index_buffer_offset.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.unsorted_keys(), num_indices),
            resizeDeviceBuffer(buffers.unsorted_gauss_idx(), num_indices),
        }
    );

}

void VulkanGSRenderer::executeComputeTileRanges(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    PerfTimer::Timer<PerfTimer::ComputeTileRanges> timer(this);
    DEVICE_GUARD;

    size_t num_indices = buffers.num_indices;
    size_t num_tiles = (size_t)(uniforms.grid_height*uniforms.grid_width);

    bufferMemoryBarrier({
        { buffers.sorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    VulkanGSRendererUniforms uniforms_1 = uniforms;
    uniforms_1.active_sh = (uint32_t)num_indices;  // alias memory

    executeCompute(
        {{num_indices+1, VKSPLAT_TILE_SHADER_TILE_RANGES_THREADS}},
        &uniforms_1, sizeof(uniforms),
        pipeline_compute_tile_ranges[buffers.is_unsorted_1],
        {
            // inputs
            buffers.sorted_keys().deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.tile_ranges, num_tiles+1),
        }
    );
}

void VulkanGSRenderer::executeRasterizeForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    if (buffers.num_indices == 0)
        return;

    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);
    DEVICE_GUARD;

    size_t num_pixels = uniforms.image_height * uniforms.image_width;

    bufferMemoryBarrier({
        { buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.tile_ranges.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    executeCompute(
        {{uniforms.image_width, VKSPLAT_TILE_WIDTH}, {uniforms.image_height, VKSPLAT_TILE_HEIGHT}},
        &uniforms, sizeof(uniforms),
        pipeline_rasterize_forward[buffers.is_unsorted_1],
        std::vector<_VulkanBuffer>({
            // inputs
            buffers.sorted_gauss_idx().deviceBuffer,
            buffers.tile_ranges.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.pixel_state, 4*num_pixels),
            resizeDeviceBuffer(buffers.n_contributors, num_pixels),
        })
    );

    // _displayImage(buffers.pixel_state, uniforms.image_width, true);
    // exit(0);
}


void VulkanGSRenderer::initRasterizationBackwardScheduler() {
    if (!rasterizeBackwardScheduler.empty())
        return;

    // empirically tuned per vendor to filter out slow ones
    if (deviceInfo.vendor == DeviceVendor::Intel_R_)
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerPixel);
    else if (deviceInfo.vendor == DeviceVendor::NVIDIA ||
             deviceInfo.vendor == DeviceVendor::AMD) {
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerSplat);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_0_8_8);
    }
    else {
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerPixel);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::PerSplat);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_0_8_0);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_0_8_8);
        rasterizeBackwardAlternatives.push_back(RasterBackwardImpl::Tensor_1_16_0);
    }

    for (size_t i = 0; i < rasterizeBackwardAlternatives.size(); ++i) {
        int scheduled_impl = (int)rasterizeBackwardAlternatives[i];
        for (int j = 0; j < 2; ++j)
            if (pipeline_rasterize_backward[scheduled_impl][j].shader == VK_NULL_HANDLE) {
                printf("WARNING: a backward implementation is disabled due to hardware limitation.\n");
                rasterizeBackwardAlternatives.erase(rasterizeBackwardAlternatives.begin()+i);
                --i;
                break;
            }
    }
    if (rasterizeBackwardAlternatives.empty())
        throw std::runtime_error("No rasterization backward implementation is available");

    rasterizeBackwardScheduler = ThompsonSamplingScheduler(
        rasterizeBackwardAlternatives.size(),
        1.0,  // initial_noise_std, in seconds
        100,  // max_no_update
        1000,  // adapt_tau
        100  // warmup_tau
    );

    auto timerCallback = [&](const std::vector<std::pair<size_t, double>>& time_updates) {
        for (size_t impl = 0; impl < (size_t)RasterBackwardImpl::size; impl++) {
            auto [count, dt] = time_updates[PerfTimer::_RasterizeBackwardScheduling_PerPixel + impl];
            if (count == 0)
                continue;
            auto alternative = std::find(
                rasterizeBackwardAlternatives.begin(), rasterizeBackwardAlternatives.end(),
                (RasterBackwardImpl)impl);
            if (alternative != rasterizeBackwardAlternatives.end())
                rasterizeBackwardScheduler.update(
                    alternative - rasterizeBackwardAlternatives.begin(), dt);
        }
    };
    timerCallbacks.push_back(timerCallback);
}

void VulkanGSRenderer::executeRasterizeBackward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers
) {
    std::shared_ptr<void> timer = nullptr;
    RasterBackwardImpl scheduled_impl = RasterBackwardImpl::Default;
    if (RASTERIZE_BACKWARD_USE_SCHEDULING) {
        using namespace PerfTimer;
        initRasterizationBackwardScheduler();
        scheduled_impl = rasterizeBackwardAlternatives[rasterizeBackwardScheduler.sample()];
        // scheduled_impl = RasterBackwardImpl::PerSplat;
        if (scheduled_impl == RasterBackwardImpl::PerPixel)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_PerPixel>>(this);
        else if (scheduled_impl == RasterBackwardImpl::PerSplat)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_PerSplat>>(this);
        else if (scheduled_impl == RasterBackwardImpl::Tensor_0_8_0)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_Tensor_0_8_0>>(this);
        else if (scheduled_impl == RasterBackwardImpl::Tensor_0_8_8)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_Tensor_0_8_8>>(this);
        else if (scheduled_impl == RasterBackwardImpl::Tensor_1_16_0)
            timer = std::make_shared<Timer<_RasterizeBackwardScheduling_Tensor_1_16_0>>(this);
    }
    else {
        timer = std::make_shared<PerfTimer::Timer<PerfTimer::RasterizeBackward>>(this);
    }

    size_t num_elements = buffers.num_splats;
    
    DEVICE_GUARD;

    clearDeviceBuffer(buffers.v_xy_vs, 2*num_elements);
    clearDeviceBuffer(buffers.v_inv_cov_vs_opacity, 4*num_elements);
    clearDeviceBuffer(buffers.v_rgb, 3*num_elements);
    if (buffers.num_indices == 0)
        return;
    bufferMemoryBarrier({
        { buffers.n_contributors.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.v_pixel_state.deviceBuffer, COMPUTE_SHADER_WRITE },
        { buffers.v_xy_vs.deviceBuffer, TRANSFER_WRITE },
        { buffers.v_inv_cov_vs_opacity.deviceBuffer, TRANSFER_WRITE },
        { buffers.v_rgb.deviceBuffer, TRANSFER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    std::vector<std::pair<size_t, size_t>> dims;
    if (scheduled_impl == RasterBackwardImpl::PerPixel)
        dims = {{(int)uniforms.image_width, VKSPLAT_TILE_WIDTH}, {(int)uniforms.image_height, VKSPLAT_TILE_HEIGHT}};
    else
        dims = {{uniforms.grid_width*uniforms.grid_height, 1}};

    executeCompute(
        dims,
        &uniforms, sizeof(uniforms),
        RASTERIZE_BACKWARD_USE_SCHEDULING ? pipeline_rasterize_backward[(size_t)scheduled_impl][buffers.is_unsorted_1] :
            pipeline_rasterize_backward[1][buffers.is_unsorted_1],  // per-splat backward
        {
            // inputs
            buffers.sorted_gauss_idx().deviceBuffer,
            buffers.tile_ranges.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            buffers.pixel_state.deviceBuffer,
            buffers.n_contributors.deviceBuffer,
            buffers.v_pixel_state.deviceBuffer,
            // outputs
            buffers.v_xy_vs.deviceBuffer,
            buffers.v_inv_cov_vs_opacity.deviceBuffer,
            buffers.v_rgb.deviceBuffer,
        }
    );

}


void VulkanGSRenderer::executeCumsum(
    VulkanGSPipelineBuffers &buffers,
    Buffer<int32_t> &input_buffer,
    Buffer<int32_t> &output_buffer
) {
    PerfTimer::Timer<PerfTimer::_Cumsum> timer(this);
    DEVICE_GUARD;

    size_t num_elements = input_buffer.deviceSize();
    const size_t block_0 = VKSPLAT_CUMSUM_BLOCK_SIZE;
    const size_t block_limit = VKSPLAT_SUBGROUP_SIZE*VKSPLAT_SUBGROUP_SIZE*VKSPLAT_SUBGROUP_SIZE;
    const size_t block = std::min(block_0, block_limit);

    uint32_t uniforms[2] = {
        (uint32_t)num_elements, 1
    };
    // int uniform_size = 2*sizeof(uint32_t);
    int uniform_size = 1*sizeof(uint32_t);

    bufferMemoryBarrier({
        { input_buffer.deviceBuffer, COMPUTE_SHADER_WRITE },
    }, COMPUTE_SHADER_READ);

    resizeDeviceBuffer(output_buffer, num_elements);

    if (num_elements <= block_0) {
        executeCompute(
            {{num_elements, block_0}},
            uniforms, uniform_size,
            pipeline_cumsum.single_pass,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
            }
        );
    }

    else if (num_elements <= block*block) {
        resizeDeviceBuffer(buffers._cumsum_blockSums, _CEIL_DIV(num_elements, block), true);

        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE },
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );
    }

    else if (num_elements <= block*block*block) {
        size_t num_elements_1 = _CEIL_DIV(num_elements, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_elements_1, true);
        resizeDeviceBuffer(buffers._cumsum_blockSums2, _CEIL_DIV(num_elements_1,block), true);

        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements_1/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE },
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );
    }

    // can't reasonably expect more than 1G splats
    // although there may be more than 1G sorting indices
    else {
        _THROW_ERROR("Too many numbers for cumsum");
    }

}

void VulkanGSRenderer::executeCalculateIndexBufferOffset(
    VulkanGSPipelineBuffers& buffers
) {
    PerfTimer::Timer<PerfTimer::CalculateIndexBufferOffset> timer(this);

    size_t num_elements = buffers.num_splats;

    executeCumsum(
        buffers,
        buffers.tiles_touched,
        buffers.index_buffer_offset
    );

    if (commandBatchInProgress) bufferMemoryBarrier({
        { buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);
    int num_indices = readElement<int32_t>(buffers.index_buffer_offset.deviceBuffer, num_elements-1);
    buffers.num_indices = (size_t)num_indices;
    // printf("num_splats=%d num_indices=%d\n", (int)num_elements, (int)num_indices);
}

void VulkanGSRenderer::executeSort(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits
) {
    PerfTimer::Timer<PerfTimer::SortRTS> timer(this);

    size_t num_elements = buffers.unsorted_keys().deviceSize();
    if (num_elements != buffers.unsorted_gauss_idx().deviceSize())
        _THROW_ERROR("number of elements don't match in executeSort");

    const int RADIX = VKSPLAT_RADIX_SORT_RADIX;
    const int PARTITION_SIZE = VKSPLAT_RADIX_PARTITION_SIZE;

    auto& globalHistogram = buffers._sorting_histogram;
    auto& partitionHistogram = buffers._sorting_histogram_cumsum;

    const size_t num_parts = _CEIL_DIV(num_elements, PARTITION_SIZE);

    int max_nonzero_bit = VKSPLAT_SORTING_KEY_BITS;
    if (num_bits == -1 && VKSPLAT_SORTING_KEY_BITS == 64) {
        int32_t num_tiles = (int32_t)(uniforms.grid_height*uniforms.grid_width);
        max_nonzero_bit = 23;  // float fraction bits
        int32_t temp = num_tiles;
        while (temp)
            temp >>= 1, max_nonzero_bit++;
    }
    else if (num_bits >= 0)
        max_nonzero_bit = num_bits;
    int num_passes = _CEIL_DIV(max_nonzero_bit, VKSPLAT_RADIX_BITS_PER_PASS);

    resizeDeviceBuffer(partitionHistogram, num_parts*RADIX);
    resizeDeviceBuffer(buffers.sorted_keys(), num_elements);
    resizeDeviceBuffer(buffers.sorted_gauss_idx(), num_elements);

    DEVICE_GUARD;
    clearDeviceBuffer(globalHistogram, num_passes * RADIX);
    bufferMemoryBarrier({
        { globalHistogram.deviceBuffer, TRANSFER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    for (int pass = 0; VKSPLAT_RADIX_BITS_PER_PASS*pass < max_nonzero_bit; pass++) {

        auto& pipeline_sorting = buffers.is_unsorted_1 ? pipeline_sorting_1 : pipeline_sorting_2;

        uint32_t uniforms[2];
        uniforms[0] = (uint32_t)pass;
        uniforms[1] = (uint32_t)num_elements;

        if (pass)
        bufferMemoryBarrier({
            { buffers.unsorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE },
            { buffers.unsorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_parts, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.upsweep,
            {
                buffers.unsorted_keys().deviceBuffer,
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{RADIX, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.spine,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ);
        executeCompute(
            {{num_parts, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.downsweep,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
                buffers.unsorted_keys().deviceBuffer,
                buffers.unsorted_gauss_idx().deviceBuffer,
                buffers.sorted_keys().deviceBuffer,
                buffers.sorted_gauss_idx().deviceBuffer,
            }
        );

        buffers.is_unsorted_1 = !buffers.is_unsorted_1;
    }
    buffers.is_unsorted_1 = !buffers.is_unsorted_1;

}


int32_t VulkanGSRenderer::executeSum(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t> &input_buffer
) {
    PerfTimer::Timer<PerfTimer::_Sum> timer(this);
    // DEVICE_GUARD;

    size_t num_elements = input_buffer.deviceSize();

    clearDeviceBuffer(buffers._temp_sum, 1);
    bufferMemoryBarrier({
        { buffers._temp_sum.deviceBuffer, TRANSFER_WRITE },
    }, COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{num_elements, VKSPLAT_SUM_BLOCK_SIZE}},
        &num_elements, sizeof(uint32_t),
        pipeline_sum,
        {
            input_buffer.deviceBuffer,
            buffers._temp_sum.deviceBuffer,
        }
    );

    bufferMemoryBarrier({
        { buffers._temp_sum.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);

    copyFromDevice(buffers._temp_sum);
    return buffers._temp_sum[0];
}

void VulkanGSRenderer::executeWhere(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t> &input_buffer,
    Buffer<int32_t> &output_buffer
) {
    PerfTimer::Timer<PerfTimer::_Where> timer(this);
    DEVICE_GUARD;

    size_t num_elements = input_buffer.deviceSize();

    executeCumsum(
        buffers,
        input_buffer,
        buffers._temp_cumsum
    );

    bufferMemoryBarrier({
        { buffers._temp_cumsum.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);
    int32_t num_nonzero = readElement<int32_t>(buffers._temp_cumsum.deviceBuffer, num_elements-1);

    executeCompute(
        {{num_elements, VKSPLAT_WHERE_BLOCK_SIZE}},
        (uint32_t*)&num_elements, sizeof(uint32_t),
        pipeline_where,
        {
            input_buffer.deviceBuffer,
            buffers._temp_cumsum.deviceBuffer,
            resizeDeviceBuffer(output_buffer, num_nonzero),
        }
    );
}
