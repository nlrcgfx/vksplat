#pragma once

#include "gs_trainer.h"

#include <array>
#include <map>
#include <string>
#include <tuple>
#include <variant>
#include <vector>


TrainerConfig make_default_trainer_config();
TrainerConfig make_mcmc_trainer_config();

std::string trainer_strategy_to_string(TrainerConfig::Strategy strategy);
TrainerConfig::Strategy trainer_strategy_from_string(const std::string& value);
std::string trainer_cache_image_to_string(TrainerConfig::CacheImage cache_image);
TrainerConfig::CacheImage trainer_cache_image_from_string(const std::string& value);

class VkSplatTrainingSession {
public:
    using DeviceInfo = std::map<
        std::string,
        std::variant<uint32_t, std::vector<uint32_t>, bool, std::string>>;

    struct ImagePath {
        std::string image_path;
        std::string mask_path;
    };

    struct TrainMetadata {
        glm::mat4 dataparser_transform;
        DeviceInfo device;
    };

    VulkanGSTrainer trainer;
    VulkanGSRendererUniforms uniforms;
    VulkanGSPipelineBuffers buffers;
    TrainerConfig config;

    VkSplatTrainingSession();

    void initialize(std::string spirv_dir, int device_id);
    TrainMetadata set_train_config(const TrainerConfig& train_config);

    size_t num_train() const;
    size_t num_val() const;

    ImagePath get_train_image_path(size_t idx);
    ImagePath get_val_image_path(size_t idx);

    void set_uniforms(
        uint32_t active_sh,
        const float* world_view_transform,
        uint32_t image_height,
        uint32_t image_width,
        float fx,
        float fy,
        float cx,
        float cy,
        bool is_fisheye);

    void set_gauss_params(
        const float* xyz_ws,
        size_t xyz_count,
        const float* rotations,
        size_t rotation_count,
        const float* scales,
        size_t scale_count,
        const float* opacities,
        size_t opacity_count,
        const float* sh_coeffs,
        size_t sh_count);

    void set_pixel_state(
        const float* pixel_state,
        size_t pixel_state_count,
        const int32_t* n_contributors,
        size_t n_contributors_count);

    void set_pixel_state_grad(const float* v_pixel_state, size_t v_pixel_state_count);

    void projection_forward();
    void process_tiles();
    void rasterize_forward();
    void rasterize_backward();
    void forward();
    void backward_optimize(int step);
    void compute_pixel_loss_grad(size_t train_idx);
    void post_backward_step(int step);

    void set_train_image(size_t train_idx);
    void train_step(size_t train_idx, int step);
    void render_train(size_t idx);
    void render_val(size_t idx);

    size_t get_vram_usage();
    size_t get_peak_vram_usage();
    std::map<std::string, std::tuple<size_t, double>> get_timing_breakdown();
    std::map<std::string, size_t> get_vram_breakdown();

    const Buffer<float>& copy_pixel_state_from_device();

    void write_ply(std::string filename);
    void cleanup();
};
