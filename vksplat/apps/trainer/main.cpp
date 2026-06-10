#include "training_session.h"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct RunOptions {
    int train_steps = 30000;
    int device_id = -1;
    std::string shader_dir = VKSPLAT_DEFAULT_SHADER_DIR;
    bool save_train_renders = false;
    bool save_val_renders = true;
};

std::string ensure_trailing_slash(std::string path) {
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path += '/';
    }
    return path;
}

std::string path_string(const fs::path& path) {
    return path.lexically_normal().generic_string();
}

std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif

    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return stream.str();
}

std::string format_bytes(size_t bytes) {
    std::ostringstream stream;
    if (bytes >= 1024ull * 1024ull * 1024ull) {
        stream << std::fixed << std::setprecision(2)
               << static_cast<double>(bytes) / static_cast<double>(1024ull * 1024ull * 1024ull)
               << " GiB";
    } else if (bytes >= 1024ull * 1024ull) {
        stream << std::fixed << std::setprecision(1)
               << static_cast<double>(bytes) / static_cast<double>(1024ull * 1024ull)
               << " MiB";
    } else {
        stream << std::fixed << std::setprecision(1)
               << static_cast<double>(bytes) / static_cast<double>(1024ull)
               << " KiB";
    }
    return stream.str();
}

json device_info_to_json(const VkSplatTrainingSession::DeviceInfo& info) {
    json result = json::object();
    for (const auto& [key, value] : info) {
        std::visit([&](const auto& item) {
            result[key] = item;
        }, value);
    }
    return result;
}

json matrix_to_json(const glm::mat4& matrix) {
    json result = json::array();
    for (int row = 0; row < 4; row++) {
        json row_json = json::array();
        for (int col = 0; col < 4; col++) {
            row_json.push_back(matrix[col][row]);
        }
        result.push_back(row_json);
    }
    return result;
}

json image_path_to_json(const VkSplatTrainingSession::ImagePath& image) {
    json result;
    result["image_path"] = image.image_path;
    if (!image.mask_path.empty()) {
        result["mask_path"] = image.mask_path;
    }
    return result;
}

json trainer_config_to_json(const TrainerConfig& config, const RunOptions& run_options) {
    json result;

    result["enable_viewer"] = false;
    result["viewer_port"] = 7007;
    result["output_dir"] = config.output_dir;
    result["output_ply"] = config.output_ply;
    result["train_steps"] = run_options.train_steps;
    result["save_train_renders"] = run_options.save_train_renders;

    result["dataset_dir"] = config.dataset_dir;
    result["image_dir"] = config.image_dir;
    result["mask_dir"] = config.mask_dir;
    result["sparse_dir"] = config.sparse_dir;
    result["eval_interval"] = config.eval_interval;

    result["image_cache_device"] = trainer_cache_image_to_string(config.image_cache_device);

    result["global_scale"] = config.global_scale;
    result["init_scale"] = config.init_scale;
    result["init_opacity"] = config.init_opacity;
    result["strategy"] = trainer_strategy_to_string(config.strategy);

    result["max_steps"] = config.max_steps;
    result["ssim_lambda"] = config.ssim_lambda;
    result["means_lr"] = config.means_lr;
    result["means_lr_final"] = config.means_lr_final;
    result["features_dc_lr"] = config.features_dc_lr;
    result["features_rest_lr"] = config.features_rest_lr;
    result["opacities_lr"] = config.opacities_lr;
    result["scales_lr"] = config.scales_lr;
    result["quats_lr"] = config.quats_lr;
    result["scale_reg"] = config.scale_reg;
    result["opacity_reg"] = config.opacity_reg;

    result["refine_start_iter"] = config.refine_start_iter;
    result["refine_stop_iter"] = config.refine_stop_iter;
    result["refine_every"] = config.refine_every;

    result["prune_opa"] = config.prune_opa;
    result["grow_grad2d"] = config.grow_grad2d;
    result["grow_scale3d"] = config.grow_scale3d;
    result["grow_scale2d"] = config.grow_scale2d;
    result["prune_scale3d"] = config.prune_scale3d;
    result["prune_scale2d"] = config.prune_scale2d;
    result["refine_scale2d_stop_iter"] = config.refine_scale2d_stop_iter;
    result["reset_every"] = config.reset_every;
    result["stop_reset_at"] = config.stop_reset_at;
    result["pause_refine_after_reset"] = config.pause_refine_after_reset;

    result["noise_lr"] = config.noise_lr;
    result["min_opacity"] = config.min_opacity;
    result["grow_factor"] = config.grow_factor;
    result["cap_max"] = config.cap_max;

    return result;
}

json metadata_to_json(const VkSplatTrainingSession::TrainMetadata& metadata) {
    json result;
    result["dataparser_transform"] = matrix_to_json(metadata.dataparser_transform);
    result["device"] = device_info_to_json(metadata.device);
    return result;
}

void write_json_file(const fs::path& path, const json& value) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open JSON output: " + path_string(path));
    }
    stream << value.dump(4) << '\n';
}

uint8_t to_byte(float value) {
    if (!std::isfinite(value)) {
        return 0;
    }
    value = std::max(0.0f, std::min(1.0f, value));
    return static_cast<uint8_t>(std::lround(255.0f * value));
}

void save_pixel_state_png(VkSplatTrainingSession& session, const fs::path& path) {
    const Buffer<float>& pixels = session.copy_pixel_state_from_device();
    const int width = static_cast<int>(session.uniforms.image_width);
    const int height = static_cast<int>(session.uniforms.image_height);
    const size_t expected_count = static_cast<size_t>(width) * height * 4;

    if (width <= 0 || height <= 0 || pixels.size() < expected_count) {
        throw std::runtime_error("Rendered pixel buffer has invalid dimensions");
    }

    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; i++) {
        const float* src = pixels.data() + 4 * i;
        uint8_t* dst = rgb.data() + 3 * i;
        dst[0] = to_byte(src[2]);
        dst[1] = to_byte(src[1]);
        dst[2] = to_byte(src[0]);
    }

    if (!stbi_write_png(path_string(path).c_str(), width, height, 3, rgb.data(), width * 3)) {
        throw std::runtime_error("Failed to write render PNG: " + path_string(path));
    }
}

json timing_breakdown_to_json(
    const std::map<std::string, std::tuple<size_t, double>>& breakdown
) {
    json result = json::object();
    for (const auto& [key, value] : breakdown) {
        result[key] = json::array({std::get<0>(value), std::get<1>(value)});
    }
    return result;
}

json vram_breakdown_to_json(const std::map<std::string, size_t>& breakdown) {
    json result = json::object();
    for (const auto& [key, value] : breakdown) {
        result[key] = value;
    }
    return result;
}

json image_paths_json(VkSplatTrainingSession& session, bool validation) {
    json result = json::array();
    const size_t count = validation ? session.num_val() : session.num_train();
    for (size_t i = 0; i < count; i++) {
        result.push_back(validation ? image_path_to_json(session.get_val_image_path(i)) :
                                      image_path_to_json(session.get_train_image_path(i)));
    }
    return result;
}

void print_timing_breakdown(
    const std::map<std::string, std::tuple<size_t, double>>& breakdown,
    double elapsed_seconds
) {
    std::vector<std::tuple<std::string, size_t, double>> rows;
    for (const auto& [item, value] : breakdown) {
        const size_t count = std::get<0>(value);
        const double seconds = std::get<1>(value);
        if (count != 0) {
            rows.emplace_back(item, count, seconds);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        return std::get<2>(lhs) > std::get<2>(rhs);
    });

    std::cout << "\nTiming breakdown:\n";
    double total = 0.0;
    for (const auto& [item, count, seconds] : rows) {
        std::cout << item << " - " << count << ", "
                  << std::fixed << std::setprecision(3) << seconds << " secs\n";
        if (!item.empty() && item[0] != '_') {
            total += seconds;
        }
    }
    std::cout << "Total - " << std::fixed << std::setprecision(3) << total
              << " / " << elapsed_seconds << " secs\n";
}

void print_vram_breakdown(
    const std::map<std::string, size_t>& breakdown,
    size_t total_queried,
    size_t peak_queried
) {
    std::vector<std::pair<std::string, size_t>> rows;
    for (const auto& [item, bytes] : breakdown) {
        if (bytes != 0) {
            rows.emplace_back(item, bytes);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    std::cout << "\nVRAM breakdown:\n";
    size_t total = 0;
    for (const auto& [item, bytes] : rows) {
        total += bytes;
        std::cout << item << " - " << format_bytes(bytes) << '\n';
    }
    std::cout << "Total - " << format_bytes(total) << '\n';
    std::cout << "Total (queried) - " << format_bytes(total_queried) << '\n';
    std::cout << "Peak (queried) - " << format_bytes(peak_queried) << '\n';
}

template<typename T>
void apply_if_set(CLI::Option* option, T& field, const T& value) {
    if (option->count() > 0) {
        field = value;
    }
}

void apply_config_paths(TrainerConfig& config, const std::string& output_ply_name) {
    const fs::path dataset_dir(config.dataset_dir);
    const std::string dataset_name = dataset_dir.filename().empty() ?
        "dataset" : dataset_dir.filename().string();
    const fs::path work_dir = fs::path(config.output_dir) / (timestamp_now() + "_" + dataset_name);

    fs::create_directories(work_dir);

    config.output_dir = ensure_trailing_slash(path_string(work_dir));
    config.image_dir = ensure_trailing_slash(path_string(dataset_dir / fs::path(config.image_dir)));
    if (!config.mask_dir.empty()) {
        config.mask_dir = ensure_trailing_slash(path_string(dataset_dir / fs::path(config.mask_dir)));
    }
    config.sparse_dir = ensure_trailing_slash(path_string(dataset_dir / fs::path(config.sparse_dir)));
    config.output_ply = path_string(work_dir / fs::path(output_ply_name));
}

}  // namespace

int main(int argc, char** argv) {
    RunOptions run_options;

    TrainerConfig cli_defaults = make_mcmc_trainer_config();
    std::string strategy = trainer_strategy_to_string(cli_defaults.strategy);
    std::string output_ply_name = cli_defaults.output_ply;
    std::string image_cache_device = trainer_cache_image_to_string(cli_defaults.image_cache_device);

    std::string output_dir = cli_defaults.output_dir;
    std::string dataset_dir = cli_defaults.dataset_dir;
    std::string image_dir = cli_defaults.image_dir;
    std::string mask_dir = cli_defaults.mask_dir;
    std::string sparse_dir = cli_defaults.sparse_dir;
    int eval_interval = cli_defaults.eval_interval;
    float global_scale = cli_defaults.global_scale;
    float init_scale = cli_defaults.init_scale;
    float init_opacity = cli_defaults.init_opacity;
    int max_steps = cli_defaults.max_steps;
    float ssim_lambda = cli_defaults.ssim_lambda;
    float means_lr = cli_defaults.means_lr;
    float means_lr_final = cli_defaults.means_lr_final;
    float features_dc_lr = cli_defaults.features_dc_lr;
    float features_rest_lr = cli_defaults.features_rest_lr;
    float opacities_lr = cli_defaults.opacities_lr;
    float scales_lr = cli_defaults.scales_lr;
    float quats_lr = cli_defaults.quats_lr;
    float scale_reg = cli_defaults.scale_reg;
    float opacity_reg = cli_defaults.opacity_reg;
    int refine_start_iter = cli_defaults.refine_start_iter;
    int refine_stop_iter = cli_defaults.refine_stop_iter;
    int refine_every = cli_defaults.refine_every;
    float prune_opa = cli_defaults.prune_opa;
    float grow_grad2d = cli_defaults.grow_grad2d;
    float grow_scale3d = cli_defaults.grow_scale3d;
    float grow_scale2d = cli_defaults.grow_scale2d;
    float prune_scale3d = cli_defaults.prune_scale3d;
    float prune_scale2d = cli_defaults.prune_scale2d;
    int refine_scale2d_stop_iter = cli_defaults.refine_scale2d_stop_iter;
    int reset_every = cli_defaults.reset_every;
    int stop_reset_at = cli_defaults.stop_reset_at;
    int pause_refine_after_reset = cli_defaults.pause_refine_after_reset;
    float noise_lr = cli_defaults.noise_lr;
    float min_opacity = cli_defaults.min_opacity;
    float grow_factor = cli_defaults.grow_factor;
    int cap_max = cli_defaults.cap_max;
    bool no_val_renders = false;

    CLI::App app{"Train a VkSplat 3D Gaussian Splatting model from COLMAP data."};

    app.add_option("--strategy", strategy, "Training preset")
        ->check(CLI::IsMember({"default", "mcmc"}))
        ->capture_default_str();
    auto output_dir_opt = app.add_option("--output-dir", output_dir, "Output root directory")
        ->capture_default_str();
    auto output_ply_opt = app.add_option("--output-ply", output_ply_name, "Output PLY filename")
        ->capture_default_str();
    app.add_option("--train-steps", run_options.train_steps, "Number of training steps")
        ->capture_default_str();
    app.add_option("--device", run_options.device_id, "Vulkan device id; -1 selects automatically")
        ->capture_default_str();
    app.add_option("--shader-dir", run_options.shader_dir, "Directory containing compiled shaders")
        ->capture_default_str();
    app.add_flag(
        "--save-train-renders",
        run_options.save_train_renders,
        "Render and save train images after training");
    app.add_flag(
        "--no-val-renders",
        no_val_renders,
        "Skip validation render PNG export");

    auto dataset_dir_opt = app.add_option("--dataset-dir", dataset_dir, "COLMAP dataset directory")
        ->capture_default_str();
    auto image_dir_opt = app.add_option("--image-dir", image_dir, "Image directory relative to dataset")
        ->capture_default_str();
    auto mask_dir_opt = app.add_option("--mask-dir", mask_dir, "Optional mask directory relative to dataset")
        ->capture_default_str();
    auto sparse_dir_opt = app.add_option("--sparse-dir", sparse_dir, "Sparse COLMAP directory relative to dataset")
        ->capture_default_str();
    auto eval_interval_opt = app.add_option("--eval-interval", eval_interval, "Validation image interval")
        ->capture_default_str();
    auto image_cache_device_opt = app.add_option(
        "--image-cache-device",
        image_cache_device,
        "Where to cache loaded training images")
        ->check(CLI::IsMember({"cpu", "gpu"}))
        ->capture_default_str();
    auto global_scale_opt = app.add_option("--global-scale", global_scale, "Scene global scale")
        ->capture_default_str();
    auto init_scale_opt = app.add_option("--init-scale", init_scale, "Initial Gaussian scale multiplier")
        ->capture_default_str();
    auto init_opacity_opt = app.add_option("--init-opacity", init_opacity, "Initial Gaussian opacity")
        ->capture_default_str();

    auto max_steps_opt = app.add_option("--max-steps", max_steps, "LR schedule max steps")
        ->capture_default_str();
    auto ssim_lambda_opt = app.add_option("--ssim-lambda", ssim_lambda, "SSIM loss weight")
        ->capture_default_str();
    auto means_lr_opt = app.add_option("--means-lr", means_lr, "Initial means learning rate")
        ->capture_default_str();
    auto means_lr_final_opt = app.add_option("--means-lr-final", means_lr_final, "Final means learning rate")
        ->capture_default_str();
    auto features_dc_lr_opt = app.add_option("--features-dc-lr", features_dc_lr, "DC SH learning rate")
        ->capture_default_str();
    auto features_rest_lr_opt = app.add_option("--features-rest-lr", features_rest_lr, "Rest SH learning rate")
        ->capture_default_str();
    auto opacities_lr_opt = app.add_option("--opacities-lr", opacities_lr, "Opacity learning rate")
        ->capture_default_str();
    auto scales_lr_opt = app.add_option("--scales-lr", scales_lr, "Scale learning rate")
        ->capture_default_str();
    auto quats_lr_opt = app.add_option("--quats-lr", quats_lr, "Quaternion learning rate")
        ->capture_default_str();
    auto scale_reg_opt = app.add_option("--scale-reg", scale_reg, "Scale regularization")
        ->capture_default_str();
    auto opacity_reg_opt = app.add_option("--opacity-reg", opacity_reg, "Opacity regularization")
        ->capture_default_str();

    auto refine_start_iter_opt = app.add_option("--refine-start-iter", refine_start_iter, "First refinement step")
        ->capture_default_str();
    auto refine_stop_iter_opt = app.add_option("--refine-stop-iter", refine_stop_iter, "Last refinement step")
        ->capture_default_str();
    auto refine_every_opt = app.add_option("--refine-every", refine_every, "Refinement interval")
        ->capture_default_str();
    auto prune_opa_opt = app.add_option("--prune-opa", prune_opa, "Default strategy opacity prune threshold")
        ->capture_default_str();
    auto grow_grad2d_opt = app.add_option("--grow-grad2d", grow_grad2d, "Default strategy 2D grow threshold")
        ->capture_default_str();
    auto grow_scale3d_opt = app.add_option("--grow-scale3d", grow_scale3d, "Default strategy 3D grow threshold")
        ->capture_default_str();
    auto grow_scale2d_opt = app.add_option("--grow-scale2d", grow_scale2d, "Default strategy 2D scale grow threshold")
        ->capture_default_str();
    auto prune_scale3d_opt = app.add_option("--prune-scale3d", prune_scale3d, "Default strategy 3D scale prune threshold")
        ->capture_default_str();
    auto prune_scale2d_opt = app.add_option("--prune-scale2d", prune_scale2d, "Default strategy 2D scale prune threshold")
        ->capture_default_str();
    auto refine_scale2d_stop_iter_opt = app.add_option(
        "--refine-scale2d-stop-iter",
        refine_scale2d_stop_iter,
        "Stop step for 2D scale refinement")
        ->capture_default_str();
    auto reset_every_opt = app.add_option("--reset-every", reset_every, "Opacity reset interval")
        ->capture_default_str();
    auto stop_reset_at_opt = app.add_option("--stop-reset-at", stop_reset_at, "Last opacity reset step; -1 means no limit")
        ->capture_default_str();
    auto pause_refine_after_reset_opt = app.add_option(
        "--pause-refine-after-reset",
        pause_refine_after_reset,
        "Refinement pause after opacity reset")
        ->capture_default_str();

    auto noise_lr_opt = app.add_option("--noise-lr", noise_lr, "MCMC noise learning-rate multiplier")
        ->capture_default_str();
    auto min_opacity_opt = app.add_option("--min-opacity", min_opacity, "MCMC minimum opacity")
        ->capture_default_str();
    auto grow_factor_opt = app.add_option("--grow-factor", grow_factor, "MCMC grow factor")
        ->capture_default_str();
    auto cap_max_opt = app.add_option("--cap-max", cap_max, "MCMC max Gaussian count")
        ->capture_default_str();

    try {
        app.parse(argc, argv);

        TrainerConfig config = strategy == "mcmc" ? make_mcmc_trainer_config() :
                                                    make_default_trainer_config();
        config.strategy = trainer_strategy_from_string(strategy);

        apply_if_set(output_dir_opt, config.output_dir, output_dir);
        apply_if_set(output_ply_opt, config.output_ply, output_ply_name);
        apply_if_set(dataset_dir_opt, config.dataset_dir, dataset_dir);
        apply_if_set(image_dir_opt, config.image_dir, image_dir);
        apply_if_set(mask_dir_opt, config.mask_dir, mask_dir);
        apply_if_set(sparse_dir_opt, config.sparse_dir, sparse_dir);
        apply_if_set(eval_interval_opt, config.eval_interval, eval_interval);
        apply_if_set(image_cache_device_opt, config.image_cache_device,
                     trainer_cache_image_from_string(image_cache_device));
        apply_if_set(global_scale_opt, config.global_scale, global_scale);
        apply_if_set(init_scale_opt, config.init_scale, init_scale);
        apply_if_set(init_opacity_opt, config.init_opacity, init_opacity);
        apply_if_set(max_steps_opt, config.max_steps, max_steps);
        apply_if_set(ssim_lambda_opt, config.ssim_lambda, ssim_lambda);
        apply_if_set(means_lr_opt, config.means_lr, means_lr);
        apply_if_set(means_lr_final_opt, config.means_lr_final, means_lr_final);
        apply_if_set(features_dc_lr_opt, config.features_dc_lr, features_dc_lr);
        apply_if_set(features_rest_lr_opt, config.features_rest_lr, features_rest_lr);
        apply_if_set(opacities_lr_opt, config.opacities_lr, opacities_lr);
        apply_if_set(scales_lr_opt, config.scales_lr, scales_lr);
        apply_if_set(quats_lr_opt, config.quats_lr, quats_lr);
        apply_if_set(scale_reg_opt, config.scale_reg, scale_reg);
        apply_if_set(opacity_reg_opt, config.opacity_reg, opacity_reg);
        apply_if_set(refine_start_iter_opt, config.refine_start_iter, refine_start_iter);
        apply_if_set(refine_stop_iter_opt, config.refine_stop_iter, refine_stop_iter);
        apply_if_set(refine_every_opt, config.refine_every, refine_every);
        apply_if_set(prune_opa_opt, config.prune_opa, prune_opa);
        apply_if_set(grow_grad2d_opt, config.grow_grad2d, grow_grad2d);
        apply_if_set(grow_scale3d_opt, config.grow_scale3d, grow_scale3d);
        apply_if_set(grow_scale2d_opt, config.grow_scale2d, grow_scale2d);
        apply_if_set(prune_scale3d_opt, config.prune_scale3d, prune_scale3d);
        apply_if_set(prune_scale2d_opt, config.prune_scale2d, prune_scale2d);
        apply_if_set(refine_scale2d_stop_iter_opt, config.refine_scale2d_stop_iter,
                     refine_scale2d_stop_iter);
        apply_if_set(reset_every_opt, config.reset_every, reset_every);
        apply_if_set(stop_reset_at_opt, config.stop_reset_at, stop_reset_at);
        apply_if_set(pause_refine_after_reset_opt, config.pause_refine_after_reset,
                     pause_refine_after_reset);
        apply_if_set(noise_lr_opt, config.noise_lr, noise_lr);
        apply_if_set(min_opacity_opt, config.min_opacity, min_opacity);
        apply_if_set(grow_factor_opt, config.grow_factor, grow_factor);
        apply_if_set(cap_max_opt, config.cap_max, cap_max);

        run_options.save_val_renders = !no_val_renders;

        output_ply_name = config.output_ply;
        apply_config_paths(config, output_ply_name);
        run_options.shader_dir = ensure_trailing_slash(run_options.shader_dir);

        std::cout << "Work dir: " << config.output_dir << "\n\n";

        VkSplatTrainingSession session;
        session.initialize(run_options.shader_dir, run_options.device_id);
        auto metadata = session.set_train_config(config);

        const fs::path output_dir_path(config.output_dir);
        write_json_file(output_dir_path / "config.json", trainer_config_to_json(config, run_options));
        write_json_file(output_dir_path / "train.json", metadata_to_json(metadata));

        if (session.num_train() == 0) {
            throw std::runtime_error("No training images were loaded");
        }

        std::vector<size_t> shuffle_idx(session.num_train());
        std::iota(shuffle_idx.begin(), shuffle_idx.end(), 0);
        std::mt19937 rng(std::random_device{}());

        auto t0 = std::chrono::high_resolution_clock::now();
        const int progress_interval = std::max(1, run_options.train_steps / 100);
        for (int step = 0; step < run_options.train_steps; step++) {
            if (step > 0 && step % static_cast<int>(shuffle_idx.size()) == 0) {
                std::shuffle(shuffle_idx.begin(), shuffle_idx.end(), rng);
            }
            const size_t image_idx = shuffle_idx[step % shuffle_idx.size()];
            session.train_step(image_idx, step);

            if (step == 0 || (step + 1) % progress_interval == 0 ||
                step + 1 == run_options.train_steps) {
                std::cout << "\rTraining " << (step + 1) << "/"
                          << run_options.train_steps << std::flush;
            }
        }
        std::cout << '\n';
        auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_seconds =
            std::chrono::duration<double>(t1 - t0).count();

        const size_t num_splats = session.buffers.num_splats;
        const size_t vram_usage = session.get_vram_usage();
        const size_t peak_vram_usage = session.get_peak_vram_usage();
        const auto timing_breakdown = session.get_timing_breakdown();
        const auto vram_breakdown = session.get_vram_breakdown();

        std::cout << "Time elapsed: " << std::fixed << std::setprecision(1)
                  << elapsed_seconds << " seconds\n";
        std::cout << "VRAM usage: "
                  << static_cast<double>(vram_usage) /
                         static_cast<double>(1024ull * 1024ull * 1024ull)
                  << " GiB\n";
        std::cout << "Num splats: " << num_splats << '\n';

        json train_json = metadata_to_json(metadata);
        train_json["num_splats"] = num_splats;
        train_json["time_elapsed"] = elapsed_seconds;
        train_json["vram"] = vram_usage;
        train_json["peak_vram"] = peak_vram_usage;
        train_json["breakdown"] = timing_breakdown_to_json(timing_breakdown);
        train_json["vram_breakdown"] = vram_breakdown_to_json(vram_breakdown);
        train_json["train_images"] = image_paths_json(session, false);
        train_json["val_images"] = image_paths_json(session, true);
        write_json_file(output_dir_path / "train.json", train_json);

        print_timing_breakdown(timing_breakdown, elapsed_seconds);
        print_vram_breakdown(vram_breakdown, vram_usage, peak_vram_usage);
        std::cout << '\n';

        std::cout << "Writing PLY\n";
        session.write_ply(config.output_ply);
        std::cout << '\n';

        if (run_options.save_train_renders) {
            for (size_t i = 0; i < session.num_train(); i++) {
                session.render_train(i);
                std::ostringstream filename;
                filename << "train_" << std::setw(5) << std::setfill('0') << i << ".png";
                save_pixel_state_png(session, output_dir_path / filename.str());
                std::cout << "\rRendering train images " << (i + 1) << "/"
                          << session.num_train() << std::flush;
            }
            std::cout << "\n\n";
        }

        if (run_options.save_val_renders) {
            for (size_t i = 0; i < session.num_val(); i++) {
                session.render_val(i);
                std::ostringstream filename;
                filename << "val_" << std::setw(5) << std::setfill('0') << i << ".png";
                save_pixel_state_png(session, output_dir_path / filename.str());
                std::cout << "\rRendering val images " << (i + 1) << "/"
                          << session.num_val() << std::flush;
            }
            std::cout << "\n\n";
        }

        session.cleanup();
    } catch (const CLI::ParseError& err) {
        return app.exit(err);
    } catch (const std::exception& err) {
        std::cerr << "Error: " << err.what() << '\n';
        return 1;
    }

    return 0;
}
