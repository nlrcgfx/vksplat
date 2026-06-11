#include "training_session.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

#if VKSPLAT_ENABLE_BUFFER_DUMPS
namespace fs = std::filesystem;

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    out << ch;
                }
                break;
        }
    }
    return out.str();
}

std::string json_string(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string path_json_string(const fs::path& path) {
    return json_string(path.lexically_normal().generic_string());
}

std::string size_array_json(const std::vector<size_t>& values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

std::string device_info_json(const VkSplatTrainingSession::DeviceInfo& info) {
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (const auto& pair : info) {
        if (!first)
            out << ',';
        first = false;
        out << json_string(pair.first) << ':';
        std::visit([&](const auto& item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, std::string>) {
                out << json_string(item);
            } else if constexpr (std::is_same_v<T, bool>) {
                out << (item ? "true" : "false");
            } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
                out << '[';
                for (size_t i = 0; i < item.size(); ++i) {
                    if (i)
                        out << ',';
                    out << item[i];
                }
                out << ']';
            } else {
                out << item;
            }
        }, pair.second);
    }
    out << '}';
    return out.str();
}

std::string read_text_file_if_exists(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream)
        return "";
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    );
}

std::string step_directory_name(int step) {
    std::ostringstream out;
    out << "step_" << std::setw(6) << std::setfill('0') << step;
    return out.str();
}

std::string buffer_dump_filename(const std::string& name, bool capacity) {
    return capacity ? name + ".capacity.vkbd" : name + ".vkbd";
}

struct BufferDumpSpec {
    std::string name;
    std::string category;
    std::string dtype;
    size_t element_size = 1;
    std::vector<size_t> shape;
    const _VulkanBuffer* buffer = nullptr;
    bool default_group = true;
};

template<typename T>
std::vector<size_t> buffer_payload_shape(const Buffer<T>& buffer, std::initializer_list<size_t> trailing_dims = {}) {
    size_t elements = buffer.deviceBuffer.size / sizeof(T);
    size_t trailing_product = 1;
    for (size_t dim : trailing_dims)
        trailing_product *= dim;
    if (trailing_product == 0 || trailing_dims.size() == 0)
        return {elements};
    if (elements % trailing_product != 0)
        return {elements};

    std::vector<size_t> shape;
    shape.reserve(trailing_dims.size() + 1);
    shape.push_back(elements / trailing_product);
    for (size_t dim : trailing_dims)
        shape.push_back(dim);
    return shape;
}

template<typename T>
void add_buffer_spec(
    std::vector<BufferDumpSpec>& specs,
    const std::string& name,
    const std::string& category,
    const std::string& dtype,
    const Buffer<T>& buffer,
    std::vector<size_t> shape,
    bool default_group = true
) {
    specs.push_back({
        name,
        category,
        dtype,
        sizeof(T),
        std::move(shape),
        &buffer.deviceBuffer,
        default_group
    });
}

void add_raw_buffer_spec(
    std::vector<BufferDumpSpec>& specs,
    const std::string& name,
    const std::string& category,
    const std::string& dtype,
    size_t element_size,
    const _VulkanBuffer* buffer,
    std::vector<size_t> shape,
    bool default_group = true
) {
    specs.push_back({
        name,
        category,
        dtype,
        element_size,
        std::move(shape),
        buffer,
        default_group
    });
}

std::vector<size_t> fallback_shape(const BufferDumpSpec& spec) {
    if (!spec.shape.empty())
        return spec.shape;
    if (!spec.buffer || spec.element_size == 0)
        return {};
    return { spec.buffer->size / spec.element_size };
}

void write_text_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("Failed to open text output: " + path.lexically_normal().generic_string());
    stream << content;
    if (!stream)
        throw std::runtime_error("Failed to write text output: " + path.lexically_normal().generic_string());
}
#endif

}  // namespace


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
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_device_info = trainer.get_device_info();
    dump_shader_config_json = read_text_file_if_exists(
        std::filesystem::path(spirv_paths_dict["projection_forward"]).parent_path() /
        "shader_config.json"
    );
#endif

    PerfTimer::reset();
}

#if VKSPLAT_ENABLE_BUFFER_DUMPS
void VkSplatTrainingSession::configure_buffer_dumps(const BufferDumpConfig& config_in) {
    dump_config = config_in;
    if (!dump_config.enabled)
        return;
    if (dump_config.root_dir.empty())
        throw std::runtime_error("Buffer dump root directory is empty");
    if (dump_config.run_id.empty())
        dump_config.run_id = "vksplat";

    fs::create_directories(dump_config.root_dir);
    dump_device_info = trainer.get_device_info();
    trainer.set_buffer_dump_callback(
        [this](const std::string& directory, const std::string& stage, const std::string& substage) {
            fs::path step_dir = step_directory_name(dump_current_step);
            this->dump_checkpoint(
                (step_dir / directory).generic_string(),
                stage,
                substage,
                dump_current_step,
                dump_current_train_idx
            );
        }
    );

    std::ostringstream manifest;
    manifest << "{\n"
             << "  \"format\": \"vkbd\",\n"
             << "  \"version\": 1,\n"
             << "  \"run_id\": " << json_string(dump_config.run_id) << ",\n"
             << "  \"dump_all_buffers\": " << (dump_config.dump_all_buffers ? "true" : "false") << ",\n"
             << "  \"dump_capacity\": " << (dump_config.dump_capacity ? "true" : "false") << ",\n"
             << "  \"root_dir\": " << path_json_string(dump_config.root_dir) << ",\n"
             << "  \"stage_manifest_pattern\": \"init/**/manifest.json and step_*/**/manifest.json\",\n"
             << "  \"device\": " << device_info_json(dump_device_info) << ",\n"
             << "  \"shader_config\": "
             << (dump_shader_config_json.empty() ? "null" : dump_shader_config_json) << "\n"
             << "}\n";
    write_text_file(fs::path(dump_config.root_dir) / "manifest.json", manifest.str());
}

void VkSplatTrainingSession::finalize_buffer_dumps() {
    if (dump_config.enabled)
        trainer.set_buffer_dump_callback(nullptr);
}

void VkSplatTrainingSession::dump_after_dataset_load() {
    dump_checkpoint("init/after_dataset_load", "init", "after_dataset_load", -1, static_cast<size_t>(-1));
}

void VkSplatTrainingSession::dump_checkpoint(
    const std::string& directory,
    const std::string& stage,
    const std::string& substage,
    int step,
    size_t train_idx
) {
    if (!dump_config.enabled)
        return;
    if (step < 0 && directory.rfind("init/", 0) != 0)
        return;

    std::vector<BufferDumpSpec> specs;
    specs.reserve(64);
    std::string train_image_path;
    if (train_idx != static_cast<size_t>(-1) && train_idx < trainer.num_train())
        train_image_path = trainer.get_train_image(train_idx).image_path;

    add_buffer_spec(specs, "xyz_ws", "gaussian", "float32", buffers.xyz_ws, buffer_payload_shape(buffers.xyz_ws, {3}));
    add_buffer_spec(specs, "sh_coeffs", "gaussian", "float32", buffers.sh_coeffs, buffer_payload_shape(buffers.sh_coeffs, {12, 4}));
    add_buffer_spec(specs, "rotations", "gaussian", "float32", buffers.rotations, buffer_payload_shape(buffers.rotations, {4}));
    add_buffer_spec(specs, "scales_opacs", "gaussian", "float32", buffers.scales_opacs, buffer_payload_shape(buffers.scales_opacs, {4}));

    add_buffer_spec(specs, "tiles_touched", "projection", "int32", buffers.tiles_touched, buffer_payload_shape(buffers.tiles_touched));
    add_buffer_spec(specs, "rect_tile_space", "projection", VKSPLAT_USE_EMULATED_INT64 ? "int32" : "int64", buffers.rect_tile_space, buffer_payload_shape(buffers.rect_tile_space, {VKSPLAT_RECT_TILE_SPACE_WORDS}));
    add_buffer_spec(specs, "radii", "projection", "int32", buffers.radii, buffer_payload_shape(buffers.radii));
    add_buffer_spec(specs, "xy_vs", "projection", "float32", buffers.xy_vs, buffer_payload_shape(buffers.xy_vs, {2}));
    add_buffer_spec(specs, "depths", "projection", "float32", buffers.depths, buffer_payload_shape(buffers.depths));
    add_buffer_spec(specs, "inv_cov_vs_opacity", "projection", "float32", buffers.inv_cov_vs_opacity, buffer_payload_shape(buffers.inv_cov_vs_opacity, {4}));
    add_buffer_spec(specs, "rgb", "projection", "float32", buffers.rgb, buffer_payload_shape(buffers.rgb, {3}));

    add_buffer_spec(specs, "index_buffer_offset", "tile_sort", "int32", buffers.index_buffer_offset, buffer_payload_shape(buffers.index_buffer_offset));
    add_buffer_spec(specs, "sorting_keys_1", "tile_sort", VKSPLAT_SORTING_KEY_BITS == 64 ? "uint64" : "uint32", buffers.sorting_keys_1, buffer_payload_shape(buffers.sorting_keys_1));
    add_buffer_spec(specs, "sorting_keys_2", "tile_sort", VKSPLAT_SORTING_KEY_BITS == 64 ? "uint64" : "uint32", buffers.sorting_keys_2, buffer_payload_shape(buffers.sorting_keys_2));
    add_buffer_spec(specs, "sorting_gauss_idx_1", "tile_sort", "int32", buffers.sorting_gauss_idx_1, buffer_payload_shape(buffers.sorting_gauss_idx_1));
    add_buffer_spec(specs, "sorting_gauss_idx_2", "tile_sort", "int32", buffers.sorting_gauss_idx_2, buffer_payload_shape(buffers.sorting_gauss_idx_2));
    add_buffer_spec(specs, "tile_ranges", "tile_sort", "int32", buffers.tile_ranges, buffer_payload_shape(buffers.tile_ranges));

    add_buffer_spec(specs, "pixel_state", "raster_loss", "float32", buffers.pixel_state, {uniforms.image_height, uniforms.image_width, 4});
    add_buffer_spec(specs, "n_contributors", "raster_loss", "int32", buffers.n_contributors, {uniforms.image_height, uniforms.image_width});
    add_buffer_spec(specs, "v_pixel_state", "raster_loss", "float32", buffers.v_pixel_state, {uniforms.image_height, uniforms.image_width, 4});

    if (train_idx != static_cast<size_t>(-1) && train_idx < trainer.num_train()) {
        const _VulkanBuffer* ref_buffer = &trainer.get_train_image(train_idx).buffer.deviceBuffer;
        if (ref_buffer->buffer == VK_NULL_HANDLE)
            ref_buffer = &buffers.ref_image.deviceBuffer;
        add_raw_buffer_spec(
            specs,
            "ref_image",
            "raster_loss",
            "uint8",
            sizeof(uint8_t),
            ref_buffer,
            {uniforms.image_height, uniforms.image_width, 4}
        );
    } else {
        add_buffer_spec(specs, "ref_image", "raster_loss", "uint8", buffers.ref_image, {uniforms.image_height, uniforms.image_width, 4});
    }

    if (stage == "loss")
        add_buffer_spec(specs, "ssim_map", "raster_loss", "float32", buffers._temp_gauss_attr, {uniforms.image_height, uniforms.image_width, 12});

    add_buffer_spec(specs, "v_xy_vs", "backward", "float32", buffers.v_xy_vs, buffer_payload_shape(buffers.v_xy_vs, {2}));
    add_buffer_spec(specs, "v_depths", "backward", "float32", buffers.v_depths, buffer_payload_shape(buffers.v_depths));
    add_buffer_spec(specs, "v_inv_cov_vs_opacity", "backward", "float32", buffers.v_inv_cov_vs_opacity, buffer_payload_shape(buffers.v_inv_cov_vs_opacity, {4}));
    add_buffer_spec(specs, "v_rgb", "backward", "float32", buffers.v_rgb, buffer_payload_shape(buffers.v_rgb, {3}));
    add_buffer_spec(specs, "g_xyz_ws", "optimizer", "float32", buffers.g_xyz_ws, buffer_payload_shape(buffers.g_xyz_ws, {2, 3}));
    add_buffer_spec(specs, "g_sh_coeffs_1", "optimizer", "float32", buffers.g_sh_coeffs_1, buffer_payload_shape(buffers.g_sh_coeffs_1, {12, 4}));
    add_buffer_spec(specs, "g_sh_coeffs_2", "optimizer", "float32", buffers.g_sh_coeffs_2, buffer_payload_shape(buffers.g_sh_coeffs_2, {12, 4}));
    add_buffer_spec(specs, "g_rotations", "optimizer", "float32", buffers.g_rotations, buffer_payload_shape(buffers.g_rotations, {2, 4}));
    add_buffer_spec(specs, "g_scales_opacs", "optimizer", "float32", buffers.g_scales_opacs, buffer_payload_shape(buffers.g_scales_opacs, {2, 4}));

    add_buffer_spec(specs, "default_grad", "strategy_default", "float32", buffers.default_grad, buffer_payload_shape(buffers.default_grad, {2}));
    add_buffer_spec(specs, "default_radii", "strategy_default", "float32", buffers.default_radii, buffer_payload_shape(buffers.default_radii));
    add_buffer_spec(specs, "default_dupli_mask", "strategy_default", "int32", buffers.default_dupli_mask, buffer_payload_shape(buffers.default_dupli_mask));
    add_buffer_spec(specs, "default_split_mask", "strategy_default", "int32", buffers.default_split_mask, buffer_payload_shape(buffers.default_split_mask));
    add_buffer_spec(specs, "default_keep_mask", "strategy_default", "int32", buffers.default_keep_mask, buffer_payload_shape(buffers.default_keep_mask));

    add_buffer_spec(specs, "mcmc_sample_probs", "strategy_mcmc", "int32", buffers.mcmc_sample_probs, buffer_payload_shape(buffers.mcmc_sample_probs));
    add_buffer_spec(specs, "mcmc_sample_probs_cumsum", "strategy_mcmc", "int32", buffers.mcmc_sample_probs_cumsum, buffer_payload_shape(buffers.mcmc_sample_probs_cumsum));
    add_buffer_spec(specs, "mcmc_index_map", "strategy_mcmc", "int32", buffers.mcmc_index_map, {});
    add_buffer_spec(specs, "mcmc_n_idx_buffer", "strategy_mcmc", "int32", buffers.mcmc_n_idx_buffer, buffer_payload_shape(buffers.mcmc_n_idx_buffer));

    add_buffer_spec(specs, "_temp_gauss_attr", "scratch", "float32", buffers._temp_gauss_attr, {}, false);
    add_buffer_spec(specs, "_temp_indices", "scratch", "int32", buffers._temp_indices, {}, false);
    add_buffer_spec(specs, "_temp_sum", "scratch", "int32", buffers._temp_sum, {}, false);
    add_buffer_spec(specs, "_temp_cumsum", "scratch", "int32", buffers._temp_cumsum, {}, false);
    add_buffer_spec(specs, "_cumsum_blockSums", "scratch", "int32", buffers._cumsum_blockSums, {}, false);
    add_buffer_spec(specs, "_cumsum_blockSums2", "scratch", "int32", buffers._cumsum_blockSums2, {}, false);
    add_buffer_spec(specs, "_sorting_histogram", "scratch", "int32", buffers._sorting_histogram, {}, false);
    add_buffer_spec(specs, "_sorting_histogram_cumsum", "scratch", "int32", buffers._sorting_histogram_cumsum, {}, false);

    const fs::path checkpoint_dir = fs::path(dump_config.root_dir) / fs::path(directory);
    fs::create_directories(checkpoint_dir);

    std::ostringstream manifest;
    manifest << "{\n"
             << "  \"run_id\": " << json_string(dump_config.run_id) << ",\n"
             << "  \"step\": " << step << ",\n"
             << "  \"train_idx\": ";
    if (train_idx == static_cast<size_t>(-1))
        manifest << "null";
    else
        manifest << train_idx;
    manifest << ",\n"
             << "  \"stage\": " << json_string(stage) << ",\n"
             << "  \"substage\": " << json_string(substage) << ",\n"
             << "  \"train_image_path\": "
             << (train_image_path.empty() ? "null" : json_string(train_image_path)) << ",\n"
             << "  \"directory\": " << path_json_string(directory) << ",\n"
             << "  \"num_splats\": " << buffers.num_splats << ",\n"
             << "  \"num_indices\": " << buffers.num_indices << ",\n"
             << "  \"is_unsorted_1\": " << (buffers.is_unsorted_1 ? "true" : "false") << ",\n"
             << "  \"raster_backward_variant\": " << json_string(dump_last_raster_backward_variant) << ",\n"
             << "  \"entries\": [\n";

    bool first_entry = true;
    for (const BufferDumpSpec& spec : specs) {
        if (!dump_config.dump_all_buffers && !spec.default_group)
            continue;

        auto write_manifest_entry_prefix = [&]() {
            if (!first_entry)
                manifest << ",\n";
            first_entry = false;
            manifest << "    {\"buffer_name\": " << json_string(spec.name)
                     << ", \"category\": " << json_string(spec.category);
        };

        if (!spec.buffer || spec.buffer->buffer == VK_NULL_HANDLE) {
            write_manifest_entry_prefix();
            manifest << ", \"status\": \"skipped\", \"reason\": \"absent\"}";
            continue;
        }

        const std::vector<size_t> shape = fallback_shape(spec);
        const size_t logical_bytes = spec.buffer->size;
        const size_t alloc_bytes = spec.buffer->allocSize;
        const size_t logical_elements = spec.element_size == 0 ? 0 : logical_bytes / spec.element_size;
        const fs::path logical_path = checkpoint_dir / buffer_dump_filename(spec.name, false);
        std::vector<uint8_t> payload;
        trainer.copyRawBytesFromDevice(*spec.buffer, logical_bytes, payload);

        auto make_metadata = [&](const std::string& payload_kind, size_t payload_bytes) {
            std::ostringstream meta;
            meta << "{"
                 << "\"run_id\":" << json_string(dump_config.run_id)
                 << ",\"step\":" << step
                 << ",\"train_idx\":";
            if (train_idx == static_cast<size_t>(-1))
                meta << "null";
            else
                meta << train_idx;
            meta << ",\"stage\":" << json_string(stage)
                 << ",\"substage\":" << json_string(substage)
                 << ",\"train_image_path\":"
                 << (train_image_path.empty() ? "null" : json_string(train_image_path))
                 << ",\"buffer_name\":" << json_string(spec.name)
                 << ",\"category\":" << json_string(spec.category)
                 << ",\"dtype\":" << json_string(spec.dtype)
                 << ",\"element_size\":" << spec.element_size
                 << ",\"shape\":" << size_array_json(shape)
                 << ",\"logical_elements\":" << logical_elements
                 << ",\"logical_bytes\":" << logical_bytes
                 << ",\"alloc_bytes\":" << alloc_bytes
                 << ",\"payload_bytes\":" << payload_bytes
                 << ",\"payload_kind\":" << json_string(payload_kind)
                 << ",\"num_splats\":" << buffers.num_splats
                 << ",\"num_indices\":" << buffers.num_indices
                 << ",\"image_width\":" << uniforms.image_width
                 << ",\"image_height\":" << uniforms.image_height
                 << ",\"grid_width\":" << uniforms.grid_width
                 << ",\"grid_height\":" << uniforms.grid_height
                 << ",\"is_unsorted_1\":" << (buffers.is_unsorted_1 ? "true" : "false")
                 << ",\"device\":" << device_info_json(dump_device_info)
                 << ",\"shader_config\":" << (dump_shader_config_json.empty() ? "null" : dump_shader_config_json)
                 << ",\"raster_backward_variant\":" << json_string(dump_last_raster_backward_variant)
                 << "}";
            return meta.str();
        };

        VulkanBufferDumpFileInfo file_info;
        file_info.filename = logical_path.lexically_normal().generic_string();
        file_info.metadata_json = make_metadata("logical", logical_bytes);
        file_info.logical_bytes = logical_bytes;
        file_info.alloc_bytes = alloc_bytes;
        file_info.payload_bytes = logical_bytes;
        file_info.flags = 0;
        VulkanGSPipeline::writeStructuredBufferDump(file_info, payload);

        write_manifest_entry_prefix();
        manifest << ", \"status\": \"dumped\", \"file\": "
                 << path_json_string(logical_path.filename())
                 << ", \"dtype\": " << json_string(spec.dtype)
                 << ", \"shape\": " << size_array_json(shape)
                 << ", \"logical_bytes\": " << logical_bytes
                 << ", \"alloc_bytes\": " << alloc_bytes;

        if (dump_config.dump_capacity && alloc_bytes >= logical_bytes) {
            const fs::path capacity_path = checkpoint_dir / buffer_dump_filename(spec.name, true);
            trainer.copyRawBytesFromDevice(*spec.buffer, alloc_bytes, payload);

            VulkanBufferDumpFileInfo capacity_info;
            capacity_info.filename = capacity_path.lexically_normal().generic_string();
            capacity_info.metadata_json = make_metadata("capacity", alloc_bytes);
            capacity_info.logical_bytes = logical_bytes;
            capacity_info.alloc_bytes = alloc_bytes;
            capacity_info.payload_bytes = alloc_bytes;
            capacity_info.flags = 1;
            VulkanGSPipeline::writeStructuredBufferDump(capacity_info, payload);
            manifest << ", \"capacity_file\": " << path_json_string(capacity_path.filename());
        }
        manifest << "}";
    }

    manifest << "\n  ]\n}\n";
    write_text_file(checkpoint_dir / "manifest.json", manifest.str());
}
#endif

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
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_device_info = metadata.device;
    dump_after_dataset_load();
#endif
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
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/01_projection_forward",
        "projection_forward",
        "after_projection_forward",
        dump_current_step,
        dump_current_train_idx
    );
#endif
}

void VkSplatTrainingSession::process_tiles() {
    auto deviceGuard = DeviceGuard(&trainer);

    try {
        trainer.executeCalculateIndexBufferOffset(buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeCalculateIndexBufferOffset failed");
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/02_process_tiles/after_index_offset",
        "process_tiles",
        "after_index_offset",
        dump_current_step,
        dump_current_train_idx
    );
#endif
    if (buffers.num_indices == 0) {
        return;
    }

    try {
        trainer.executeGenerateKeys(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeGenerateKeys failed");
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/02_process_tiles/after_generate_keys",
        "process_tiles",
        "after_generate_keys",
        dump_current_step,
        dump_current_train_idx
    );
#endif

    try {
        trainer.executeSort(uniforms, buffers, -1);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeSort failed");
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/02_process_tiles/after_sort",
        "process_tiles",
        "after_sort",
        dump_current_step,
        dump_current_train_idx
    );
#endif

    try {
        trainer.executeComputeTileRanges(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeComputeTileRanges failed");
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/02_process_tiles/after_tile_ranges",
        "process_tiles",
        "after_tile_ranges",
        dump_current_step,
        dump_current_train_idx
    );
#endif
}

void VkSplatTrainingSession::rasterize_forward() {
    try {
        trainer.executeRasterizeForward(uniforms, buffers);
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(std::string(err.what()) + ". trainer.executeRasterizeForward failed");
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/03_rasterize_forward",
        "rasterize_forward",
        "after_rasterize_forward",
        dump_current_step,
        dump_current_train_idx
    );
#endif
}

void VkSplatTrainingSession::rasterize_backward() {
    try {
        trainer.executeRasterizeBackward(uniforms, buffers);
#if VKSPLAT_ENABLE_BUFFER_DUMPS
        dump_last_raster_backward_variant = trainer.get_last_raster_backward_variant();
#endif
    } catch (const std::runtime_error& err) {
        throw std::runtime_error(
            std::string(err.what()) + ". trainer.executeRasterizeBackward failed");
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/05_backward/after_rasterize_backward",
        "backward",
        "after_rasterize_backward",
        dump_current_step,
        dump_current_train_idx
    );
#endif
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
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/05_backward/after_fused_optimizer",
        "backward",
        "after_fused_optimizer",
        dump_current_step,
        dump_current_train_idx
    );
#endif
}

void VkSplatTrainingSession::compute_pixel_loss_grad(size_t train_idx) {
    if (buffers.num_indices == 0) {
        return;
    }
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_current_train_idx = train_idx;
#endif
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
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_checkpoint(
        step_directory_name(dump_current_step) + "/06_post_backward/after_post_backward",
        "post_backward",
        "after_post_backward",
        dump_current_step,
        dump_current_train_idx
    );
#endif
}

void VkSplatTrainingSession::set_train_image(size_t train_idx) {
    trainer.get_train_camera(train_idx, uniforms);
}

void VkSplatTrainingSession::train_step(size_t train_idx, int step) {
    set_train_image(train_idx);
    const int kShDegreeInterval = 1000;
    uniforms.active_sh = std::min(step / kShDegreeInterval, 3);
    uniforms.step = step;

#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_current_step = step;
    dump_current_train_idx = train_idx;
    dump_last_raster_backward_variant = "not_run";
    dump_checkpoint(
        step_directory_name(step) + "/00_step_start",
        "step",
        "step_start",
        step,
        train_idx
    );
#endif

    auto deviceGuard = DeviceGuard(&trainer);

    forward();
    compute_pixel_loss_grad(train_idx);
    backward_optimize(step);
    post_backward_step(step);
}

void VkSplatTrainingSession::render_train(size_t idx) {
    trainer.get_train_camera(idx, uniforms);
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_current_step = -1;
    dump_current_train_idx = idx;
#endif
    forward();
}

void VkSplatTrainingSession::render_val(size_t idx) {
    trainer.get_val_camera(idx, uniforms);
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    dump_current_step = -1;
    dump_current_train_idx = static_cast<size_t>(-1);
#endif
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
#if VKSPLAT_ENABLE_BUFFER_DUMPS
    finalize_buffer_dumps();
#endif
    trainer.cleanupBuffers(buffers);
    trainer.cleanup();
}
