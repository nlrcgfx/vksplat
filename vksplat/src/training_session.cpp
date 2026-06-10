#include "training_session.h"

#include <algorithm>
#include <stdexcept>


TrainerConfig make_default_trainer_config() {
    TrainerConfig config;

    config.output_dir = "/mnt/d/gs/outputs";
    config.output_ply = "splat.ply";

    config.dataset_dir = "/mnt/d/gs/data/360_v2/bicycle";
    config.image_dir = "images_4";
    config.mask_dir = "";
    config.sparse_dir = "sparse/0";
    config.eval_interval = 8;

    config.image_cache_device = TrainerConfig::CacheImage::CPU;

    config.global_scale = 1.0f;
    config.init_scale = 1.0f;
    config.init_opacity = 0.1f;

    config.strategy = TrainerConfig::Strategy::Default;

    config.max_steps = 30000;
    config.ssim_lambda = 0.2f;
    config.means_lr = 1.6e-4f;
    config.means_lr_final = 1.6e-6f;
    config.features_dc_lr = 0.0025f;
    config.features_rest_lr = 0.0025f / 20.0f;
    config.opacities_lr = 0.05f;
    config.scales_lr = 0.005f;
    config.quats_lr = 0.001f;
    config.scale_reg = 0.0f;
    config.opacity_reg = 0.0f;

    config.refine_start_iter = 500;
    config.refine_stop_iter = 15000;
    config.refine_every = 100;

    config.prune_opa = 0.005f;
    config.grow_grad2d = 0.0002f;
    config.grow_scale3d = 0.01f;
    config.grow_scale2d = 0.05f;
    config.prune_scale3d = 0.1f;
    config.prune_scale2d = 0.15f;
    config.refine_scale2d_stop_iter = 0;
    config.reset_every = 3000;
    config.stop_reset_at = -1;
    config.pause_refine_after_reset = 0;

    config.noise_lr = 5e5f;
    config.min_opacity = 0.005f;
    config.grow_factor = 1.05f;
    config.cap_max = 1000000;

    return config;
}

TrainerConfig make_mcmc_trainer_config() {
    TrainerConfig config = make_default_trainer_config();

    config.strategy = TrainerConfig::Strategy::MCMC;
    config.init_scale = 0.1f;
    config.init_opacity = 0.5f;
    config.opacities_lr = 0.05f;
    config.scale_reg = 0.01f;
    config.opacity_reg = 0.01f;
    config.refine_stop_iter = 25000;

    return config;
}

std::string trainer_strategy_to_string(TrainerConfig::Strategy strategy) {
    switch (strategy) {
        case TrainerConfig::Strategy::Default:
            return "default";
        case TrainerConfig::Strategy::MCMC:
            return "mcmc";
    }
    throw std::runtime_error("Unknown trainer strategy");
}

TrainerConfig::Strategy trainer_strategy_from_string(const std::string& value) {
    if (value == "default") {
        return TrainerConfig::Strategy::Default;
    }
    if (value == "mcmc") {
        return TrainerConfig::Strategy::MCMC;
    }
    throw std::runtime_error("Unknown strategy: " + value);
}

std::string trainer_cache_image_to_string(TrainerConfig::CacheImage cache_image) {
    switch (cache_image) {
        case TrainerConfig::CacheImage::CPU:
            return "cpu";
        case TrainerConfig::CacheImage::GPU:
            return "gpu";
    }
    throw std::runtime_error("Unknown image cache device");
}

TrainerConfig::CacheImage trainer_cache_image_from_string(const std::string& value) {
    if (value == "cpu") {
        return TrainerConfig::CacheImage::CPU;
    }
    if (value == "gpu") {
        return TrainerConfig::CacheImage::GPU;
    }
    throw std::runtime_error("Unknown image_cache_device: " + value);
}

VkSplatTrainingSession::VkSplatTrainingSession()
    : trainer(),
      uniforms(),
      buffers(),
      config(make_default_trainer_config()) {
}

void VkSplatTrainingSession::initialize(std::string spirv_dir, int device_id) {
    buffers.num_splats = 0;

    std::vector<std::string> spirv_paths = {
        "projection_forward",
        "generate_keys",
        "compute_tile_ranges",
        "rasterize_forward",
        "rasterize_backward_0",
        "rasterize_backward_1",
        "rasterize_backward_2",
        "rasterize_backward_3",
        "rasterize_backward_4",
        "cumsum_single_pass",
        "cumsum_block_scan",
        "cumsum_scan_block_sums",
        "cumsum_add_block_offsets",
        "radix_sort/upsweep",
        "radix_sort/spine",
        "radix_sort/downsweep",
        "ssim_forward",
        "ssim_backward",
        "fused_projection_backward_optimizer",
        "sum",
        "where",
        "default_update_state",
        "default_compute_grow_mask",
        "default_duplicate",
        "default_split",
        "default_compute_prune_mask",
        "default_prune",
        "default_prune_mean",
        "default_prune_sh",
        "default_reset_opa",
        "mcmc_inject_noise",
        "mcmc_compute_probs",
        "mcmc_compute_relocation_index_map",
        "mcmc_compute_relocation",
        "mcmc_update_relocation",
        "mcmc_compute_add_index_map",
        "mcmc_compute_add",
        "mcmc_update_add",
        "morton_sort_compute_stats",
        "morton_sort_generate_keys",
        "morton_sort_apply_indices",
        "morton_sort_apply_indices_sh",
        "morton_sort_update_buffer",
        "morton_sort_update_buffer_sh",
    };

    std::map<std::string, std::string> spirv_paths_dict;
    for (const std::string& name : spirv_paths) {
        spirv_paths_dict[name] =
            (name.find('/') == std::string::npos) ?
            spirv_dir + "generated/" + name + ".spv" :
            spirv_dir + name + ".spv";
    }

    trainer.initialize(spirv_paths_dict, device_id);

    PerfTimer::reset();
}

VkSplatTrainingSession::TrainMetadata VkSplatTrainingSession::set_train_config(
    const TrainerConfig& train_config
) {
    config = train_config;

    trainer.load_colmap_dataset(config, buffers);
    trainer.get_train_camera(0, uniforms);
    uniforms.active_sh = 3;

    TrainMetadata metadata;
    metadata.dataparser_transform = trainer.get_dataparser_transform();
    metadata.device = trainer.get_device_info();
    return metadata;
}

size_t VkSplatTrainingSession::num_train() const {
    return trainer.num_train();
}

size_t VkSplatTrainingSession::num_val() const {
    return trainer.num_val();
}

VkSplatTrainingSession::ImagePath VkSplatTrainingSession::get_train_image_path(size_t idx) {
    auto image = trainer.get_train_image(idx);
    return {image.image_path, image.mask_path};
}

VkSplatTrainingSession::ImagePath VkSplatTrainingSession::get_val_image_path(size_t idx) {
    auto image = trainer.get_val_image(idx);
    return {image.image_path, image.mask_path};
}

void VkSplatTrainingSession::set_uniforms(
    uint32_t active_sh,
    const float* world_view_transform,
    uint32_t image_height,
    uint32_t image_width,
    float fx,
    float fy,
    float cx,
    float cy,
    bool is_fisheye
) {
    uniforms.active_sh = active_sh;
    uniforms.image_height = image_height;
    uniforms.image_width = image_width;
    uniforms.camera_model = is_fisheye ? 2 : 0;
    uniforms.fx = fx;
    uniforms.fy = fy;
    uniforms.cx = cx;
    uniforms.cy = cy;
    uniforms.grid_height = _CEIL_DIV(image_height, VKSPLAT_TILE_HEIGHT);
    uniforms.grid_width = _CEIL_DIV(image_width, VKSPLAT_TILE_WIDTH);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uniforms.world_view_transform[4 * i + j] = world_view_transform[4 * j + i];
        }
    }

    uniforms.step = 0;
}

void VkSplatTrainingSession::set_gauss_params(
    const float* xyz_ws,
    size_t xyz_count,
    const float* rotations,
    size_t rotation_count,
    const float* scales,
    size_t scale_count,
    const float* opacities,
    size_t opacity_count,
    const float* sh_coeffs,
    size_t sh_count
) {
    if (xyz_count % 3 != 0) {
        throw std::runtime_error("xyz_ws must be (N, 3)");
    }
    const size_t n = xyz_count / 3;
    if (rotation_count != 4 * n || scale_count != 3 * n ||
        opacity_count != n || sh_count != 16 * 3 * n) {
        throw std::runtime_error("All Gaussian parameter buffers must have the same N");
    }

    buffers.num_splats = n;
    buffers.xyz_ws.assign(xyz_ws, xyz_ws + xyz_count);
    buffers.rotations.assign(rotations, rotations + rotation_count);
    buffers.assignScalesOpacs(buffers.scales_opacs, buffers.num_splats, scales, opacities);
    buffers.sh_coeffs.assign(sh_coeffs, sh_coeffs + sh_count);
    buffers.reorderSH(buffers.sh_coeffs);

    if (config.strategy == TrainerConfig::Strategy::MCMC) {
        trainer.resizeDeviceBuffer(buffers.xyz_ws, 3 * config.cap_max);
        trainer.resizeDeviceBuffer(buffers.sh_coeffs, 12 * 4 * config.cap_max);
        trainer.resizeDeviceBuffer(buffers.rotations, 4 * config.cap_max);
        trainer.resizeDeviceBuffer(buffers.scales_opacs, 4 * config.cap_max);
    }
    trainer.copyToDevice(buffers.xyz_ws);
    trainer.copyToDevice(buffers.sh_coeffs);
    trainer.copyToDevice(buffers.rotations);
    trainer.copyToDevice(buffers.scales_opacs);
}

void VkSplatTrainingSession::set_pixel_state(
    const float* pixel_state,
    size_t pixel_state_count,
    const int32_t* n_contributors,
    size_t n_contributors_count
) {
    const size_t pixel_count = static_cast<size_t>(uniforms.image_height) * uniforms.image_width;
    if (pixel_state_count != 4 * pixel_count) {
        throw std::runtime_error("pixel_state must be (H, W, 4)");
    }
    if (n_contributors_count != pixel_count) {
        throw std::runtime_error("n_contributors must be (H, W, 1)");
    }

    buffers.pixel_state.assign(pixel_state, pixel_state + pixel_state_count);
    trainer.copyToDevice(buffers.pixel_state);

    buffers.n_contributors.assign(n_contributors, n_contributors + n_contributors_count);
    trainer.copyToDevice(buffers.n_contributors);
}

void VkSplatTrainingSession::set_pixel_state_grad(
    const float* v_pixel_state,
    size_t v_pixel_state_count
) {
    const size_t pixel_count = static_cast<size_t>(uniforms.image_height) * uniforms.image_width;
    if (v_pixel_state_count != 4 * pixel_count) {
        throw std::runtime_error("v_pixel_state must be (H, W, 4)");
    }

    buffers.v_pixel_state.assign(v_pixel_state, v_pixel_state + v_pixel_state_count);
    trainer.copyToDevice(buffers.v_pixel_state);
}

void VkSplatTrainingSession::projection_forward() {
    uniforms.num_splats = static_cast<uint32_t>(buffers.num_splats);
    try {
        size_t alloc_reserve = config.strategy == TrainerConfig::Strategy::MCMC ?
            config.cap_max : 0;
        trainer.executeProjectionForward(uniforms, buffers, alloc_reserve);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeProjectionForward failed");
    }
}

void VkSplatTrainingSession::process_tiles() {
    auto deviceGuard = DeviceGuard(&trainer);

    try {
        trainer.executeCalculateIndexBufferOffset(buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeCalculateIndexBufferOffset failed");
    }
    if (buffers.num_indices == 0) {
        return;
    }

    try {
        trainer.executeGenerateKeys(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeGenerateKeys failed");
    }

    try {
        trainer.executeSort(uniforms, buffers, -1);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeSort failed");
    }

    try {
        trainer.executeComputeTileRanges(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeComputeTileRanges failed");
    }
}

void VkSplatTrainingSession::rasterize_forward() {
    try {
        trainer.executeRasterizeForward(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
    }
}

void VkSplatTrainingSession::rasterize_backward() {
    try {
        trainer.executeRasterizeBackward(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeRasterizeBackward failed");
    }
}

void VkSplatTrainingSession::forward() {
    auto deviceGuard = DeviceGuard(&trainer);
    projection_forward();
    process_tiles();
    rasterize_forward();
}

void VkSplatTrainingSession::backward_optimize(int step) {
    auto deviceGuard = DeviceGuard(&trainer);

    rasterize_backward();

    try {
        trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step + 1);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeFusedProjectionBackwardOptimizerStep failed");
    }
}

void VkSplatTrainingSession::compute_pixel_loss_grad(size_t train_idx) {
    if (buffers.num_indices == 0) {
        return;
    }
    try {
        trainer.executeComputeSSIMGradient(config, uniforms, buffers, train_idx);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeComputeSSIMGradient failed");
    }
}

void VkSplatTrainingSession::post_backward_step(int step) {
    if (config.strategy == TrainerConfig::Strategy::Default) {
        try {
            trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
        } catch (const std::runtime_error& err) {
            throw std::runtime_error(
                std::string(err.what()) + ". trainer.executeDefaultPostBackward failed");
        }
    } else if (config.strategy == TrainerConfig::Strategy::MCMC) {
        try {
            trainer.executeMCMCPostBackward(config, uniforms, buffers, step);
        } catch (const std::runtime_error& err) {
            throw std::runtime_error(
                std::string(err.what()) + ". trainer.executeMCMCPostBackward failed");
        }
    }
}

void VkSplatTrainingSession::set_train_image(size_t train_idx) {
    trainer.get_train_camera(train_idx, uniforms);
}

void VkSplatTrainingSession::train_step(size_t train_idx, int step) {
    set_train_image(train_idx);
    const int kShDegreeInterval = 1000;
    uniforms.active_sh = std::min(step / kShDegreeInterval, 3);
    uniforms.step = step;

    auto deviceGuard = DeviceGuard(&trainer);

    forward();
    compute_pixel_loss_grad(train_idx);
    backward_optimize(step);
    post_backward_step(step);
}

void VkSplatTrainingSession::render_train(size_t idx) {
    trainer.get_train_camera(idx, uniforms);
    forward();
}

void VkSplatTrainingSession::render_val(size_t idx) {
    trainer.get_val_camera(idx, uniforms);
    forward();
}

size_t VkSplatTrainingSession::get_vram_usage() {
    return buffers.getTotalAllocSize();
}

size_t VkSplatTrainingSession::get_peak_vram_usage() {
    return trainer.getPeakAllocSize();
}

std::map<std::string, std::tuple<size_t, double>>
VkSplatTrainingSession::get_timing_breakdown() {
    return PerfTimer::get_summary();
}

std::map<std::string, size_t> VkSplatTrainingSession::get_vram_breakdown() {
    return buffers.getVramBreakdown();
}

const Buffer<float>& VkSplatTrainingSession::copy_pixel_state_from_device() {
    trainer.copyFromDevice(buffers.pixel_state);
    return buffers.pixel_state;
}

void VkSplatTrainingSession::write_ply(std::string filename) {
    trainer.writePLY(filename, buffers);
}

void VkSplatTrainingSession::cleanup() {
    trainer.cleanupBuffers(buffers);
    trainer.cleanup();
}
