#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "training_session.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

class PyVkSplat : public VkSplatTrainingSession {
public:
    template<typename Ti, typename To, bool on_device = true>
    py::array_t<To> buffer_to_array(
        Buffer<Ti>& buffer,
        size_t n1 = static_cast<size_t>(-1),
        size_t n2 = static_cast<size_t>(-1),
        std::string reorder = "",
        size_t n1_actual = static_cast<size_t>(-1),
        size_t n1_offset = 0
    ) {
        if (on_device) {
            trainer.copyFromDevice(buffer);
        }
        if (reorder == "sh") {
            buffers.undoReorderSH(buffer, buffers.num_splats);
        }

        size_t n0 = buffer.size();
        if (n1 != static_cast<size_t>(-1)) {
            n0 /= n1;
        }
        if (n2 != static_cast<size_t>(-1)) {
            n0 /= n2;
        }

        std::vector<To> buffer_float;
        auto toFloat = [&](Ti* buffer_half) -> To* {
            size_t n = buffer.size();
            buffer_float.resize(n);
            for (size_t i = 0; i < n; i++) {
                buffer_float[i] = static_cast<To>(halfToFloat(static_cast<uint16_t>(buffer_half[i])));
            }
            return buffer_float.data();
        };

        if (n1 == static_cast<size_t>(-1) && n2 == static_cast<size_t>(-1)) {
            using dim = std::array<py::ssize_t, 1>;
            return py::array_t<To>(
                dim({static_cast<py::ssize_t>(n0)}),
                dim({static_cast<py::ssize_t>(sizeof(To))}),
                sizeof(Ti) == sizeof(To) ? reinterpret_cast<To*>(buffer.data()) :
                                            toFloat(buffer.data()));
        }
        if (n2 == static_cast<size_t>(-1)) {
            using dim2 = std::array<py::ssize_t, 2>;
            if (n1_actual == static_cast<size_t>(-1)) {
                n1_actual = n1;
            }
            return py::array_t<To>(
                dim2({static_cast<py::ssize_t>(n0), static_cast<py::ssize_t>(n1_actual)}),
                dim2({static_cast<py::ssize_t>(n1 * sizeof(To)),
                      static_cast<py::ssize_t>(sizeof(To))}),
                sizeof(Ti) == sizeof(To) ? n1_offset + reinterpret_cast<To*>(buffer.data()) :
                                            toFloat(n1_offset + buffer.data()));
        }

        using dim3 = std::array<py::ssize_t, 3>;
        return py::array_t<To>(
            dim3({static_cast<py::ssize_t>(n0),
                  static_cast<py::ssize_t>(n1),
                  static_cast<py::ssize_t>(n2)}),
            dim3({static_cast<py::ssize_t>(n1 * n2 * sizeof(To)),
                  static_cast<py::ssize_t>(n2 * sizeof(To)),
                  static_cast<py::ssize_t>(sizeof(To))}),
            sizeof(Ti) == sizeof(To) ? reinterpret_cast<To*>(buffer.data()) :
                                        toFloat(buffer.data()));
    }

    template<typename T>
    void array_to_buffer(py::array_t<T> array, Buffer<T>& buffer) {
        auto array_buf = array.request();
        buffer.assign(
            static_cast<T*>(array_buf.ptr),
            static_cast<T*>(array_buf.ptr) + array_buf.size);
        trainer.copyToDevice(buffer);
    }

    py::dict set_train_config(py::dict train_config) {
        TrainerConfig cfg;

        cfg.output_dir = train_config["output_dir"].cast<std::string>();
        cfg.output_ply = train_config["output_ply"].cast<std::string>();

        cfg.dataset_dir = train_config["dataset_dir"].cast<std::string>();
        cfg.image_dir = train_config["image_dir"].cast<std::string>();
        cfg.mask_dir = train_config["mask_dir"].cast<std::string>();
        cfg.sparse_dir = train_config["sparse_dir"].cast<std::string>();
        cfg.eval_interval = train_config["eval_interval"].cast<int>();

        cfg.image_cache_device =
            trainer_cache_image_from_string(train_config["image_cache_device"].cast<std::string>());

        cfg.global_scale = train_config["global_scale"].cast<float>();
        cfg.init_scale = train_config["init_scale"].cast<float>();
        cfg.init_opacity = train_config["init_opacity"].cast<float>();

        cfg.strategy = trainer_strategy_from_string(train_config["strategy"].cast<std::string>());

        cfg.max_steps = train_config["max_steps"].cast<int>();
        cfg.ssim_lambda = train_config["ssim_lambda"].cast<float>();
        cfg.means_lr = train_config["means_lr"].cast<float>();
        cfg.means_lr_final = train_config["means_lr_final"].cast<float>();
        cfg.features_dc_lr = train_config["features_dc_lr"].cast<float>();
        cfg.features_rest_lr = train_config["features_rest_lr"].cast<float>();
        cfg.opacities_lr = train_config["opacities_lr"].cast<float>();
        cfg.scales_lr = train_config["scales_lr"].cast<float>();
        cfg.quats_lr = train_config["quats_lr"].cast<float>();
        cfg.scale_reg = train_config["scale_reg"].cast<float>();
        cfg.opacity_reg = train_config["opacity_reg"].cast<float>();

        cfg.refine_start_iter = train_config["refine_start_iter"].cast<int>();
        cfg.refine_stop_iter = train_config["refine_stop_iter"].cast<int>();
        cfg.refine_every = train_config["refine_every"].cast<int>();

        cfg.prune_opa = train_config["prune_opa"].cast<float>();
        cfg.grow_grad2d = train_config["grow_grad2d"].cast<float>();
        cfg.grow_scale3d = train_config["grow_scale3d"].cast<float>();
        cfg.grow_scale2d = train_config["grow_scale2d"].cast<float>();
        cfg.prune_scale3d = train_config["prune_scale3d"].cast<float>();
        cfg.prune_scale2d = train_config["prune_scale2d"].cast<float>();
        cfg.refine_scale2d_stop_iter = train_config["refine_scale2d_stop_iter"].cast<int>();
        cfg.reset_every = train_config["reset_every"].cast<int>();
        cfg.stop_reset_at = train_config["stop_reset_at"].cast<int>();
        cfg.pause_refine_after_reset = train_config["pause_refine_after_reset"].cast<int>();

        cfg.noise_lr = train_config["noise_lr"].cast<float>();
        cfg.min_opacity = train_config["min_opacity"].cast<float>();
        cfg.grow_factor = train_config["grow_factor"].cast<float>();
        cfg.cap_max = train_config["cap_max"].cast<int>();

        auto metadata = VkSplatTrainingSession::set_train_config(cfg);

        py::dict train_meta;
        train_meta["dataparser_transform"] = py::array_t<float>(
            {4, 4},
            {sizeof(float), 4 * sizeof(float)},
            reinterpret_cast<float*>(&metadata.dataparser_transform));
        train_meta["device"] = metadata.device;
        return train_meta;
    }

    py::array_t<uint8_t> get_train_image(size_t idx) {
        auto& im = trainer.get_train_image(idx);
        return buffer_to_array<uint8_t, uint8_t, false>(im.buffer, im.camera.w, 4);
    }

    py::array_t<uint8_t> get_val_image(size_t idx) {
        auto& im = trainer.get_val_image(idx);
        return buffer_to_array<uint8_t, uint8_t, false>(im.buffer, im.camera.w, 4);
    }

    void set_uniforms(
        uint32_t active_sh,
        py::array_t<float> world_view_transform,
        uint32_t image_height,
        uint32_t image_width,
        float fx,
        float fy,
        float cx,
        float cy,
        bool is_fisheye
    ) {
        auto transform_buf = world_view_transform.request();
        if (transform_buf.ndim != 2 || transform_buf.shape[0] != 4 ||
            transform_buf.shape[1] != 4) {
            throw std::runtime_error("world_view_transform must be (4, 4) array");
        }

        VkSplatTrainingSession::set_uniforms(
            active_sh,
            static_cast<float*>(transform_buf.ptr),
            image_height,
            image_width,
            fx,
            fy,
            cx,
            cy,
            is_fisheye);
    }

    py::dict get_uniforms() {
        py::dict result;

        result["active_sh"] = uniforms.active_sh + 0;
        result["image_height"] = uniforms.image_height + 0;
        result["image_width"] = uniforms.image_width + 0;
        result["fx"] = uniforms.fx * 1.0f;
        result["fy"] = uniforms.fy * 1.0f;
        result["cx"] = uniforms.cx * 1.0f;
        result["cy"] = uniforms.cy * 1.0f;

        auto world_view_array = py::array_t<float>({4, 4});
        auto world_view_buf = world_view_array.request();
        float* world_view_ptr = static_cast<float*>(world_view_buf.ptr);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                world_view_ptr[4 * i + j] = uniforms.world_view_transform[4 * j + i];
            }
        }
        result["world_view_transform"] = world_view_array;

        auto K_array = py::array_t<float>({3, 3});
        auto K_buf = K_array.request();
        float* K_ptr = static_cast<float*>(K_buf.ptr);
        std::fill(K_ptr, K_ptr + 9, 0.0f);
        K_ptr[0] = uniforms.fx;
        K_ptr[4] = uniforms.fy;
        K_ptr[2] = uniforms.cx;
        K_ptr[5] = uniforms.cy;
        K_ptr[8] = 1.0f;
        result["K"] = K_array;

        return result;
    }

    void set_gauss_params(
        py::array_t<float> xyz_ws,
        py::array_t<float> rotations,
        py::array_t<float> scales,
        py::array_t<float> opacities,
        py::array_t<float> sh_coeffs
    ) {
        auto xyz_buf = xyz_ws.request();
        auto sh_buf = sh_coeffs.request();
        auto rot_buf = rotations.request();
        auto scale_buf = scales.request();
        auto opacity_buf = opacities.request();

        if (xyz_buf.ndim != 2 || xyz_buf.shape[1] != 3) {
            throw std::runtime_error("xyz_ws must be (N, 3) array");
        }
        if (sh_buf.ndim != 3 || sh_buf.shape[1] != 16 || sh_buf.shape[2] != 3) {
            throw std::runtime_error("sh_coeffs must be (N, 16, 3) array");
        }
        if (rot_buf.ndim != 2 || rot_buf.shape[1] != 4) {
            throw std::runtime_error("rotations must be (N, 4) array");
        }
        if (scale_buf.ndim != 2 || scale_buf.shape[1] != 3) {
            throw std::runtime_error("scales must be (N, 3) array");
        }
        if (opacity_buf.ndim != 2 || opacity_buf.shape[1] != 1) {
            throw std::runtime_error("opacities must be (N, 1) array");
        }

        VkSplatTrainingSession::set_gauss_params(
            static_cast<float*>(xyz_buf.ptr),
            static_cast<size_t>(xyz_buf.size),
            static_cast<float*>(rot_buf.ptr),
            static_cast<size_t>(rot_buf.size),
            static_cast<float*>(scale_buf.ptr),
            static_cast<size_t>(scale_buf.size),
            static_cast<float*>(opacity_buf.ptr),
            static_cast<size_t>(opacity_buf.size),
            static_cast<float*>(sh_buf.ptr),
            static_cast<size_t>(sh_buf.size));
    }

    void set_pixel_state(py::array_t<float> pixel_state, py::array_t<int> n_contributors) {
        auto pixel_state_buf = pixel_state.request();
        if (pixel_state_buf.ndim != 3 ||
            pixel_state_buf.shape[0] != uniforms.image_height ||
            pixel_state_buf.shape[1] != uniforms.image_width ||
            pixel_state_buf.shape[2] != 4) {
            throw std::runtime_error("pixel_state must be (H, W, 4) array");
        }

        auto n_contributors_buf = n_contributors.request();
        if (n_contributors_buf.ndim != 3 ||
            n_contributors_buf.shape[0] != uniforms.image_height ||
            n_contributors_buf.shape[1] != uniforms.image_width ||
            n_contributors_buf.shape[2] != 1) {
            throw std::runtime_error("n_contributors must be (H, W, 1) array");
        }

        VkSplatTrainingSession::set_pixel_state(
            static_cast<float*>(pixel_state_buf.ptr),
            static_cast<size_t>(pixel_state_buf.size),
            reinterpret_cast<int32_t*>(n_contributors_buf.ptr),
            static_cast<size_t>(n_contributors_buf.size));
    }

    void set_pixel_state_grad(py::array_t<float> v_pixel_state) {
        auto v_pixel_state_buf = v_pixel_state.request();
        if (v_pixel_state_buf.ndim != 3 ||
            v_pixel_state_buf.shape[0] != uniforms.image_height ||
            v_pixel_state_buf.shape[1] != uniforms.image_width ||
            v_pixel_state_buf.shape[2] != 4) {
            throw std::runtime_error("v_pixel_state must be (H, W, 4) array");
        }

        VkSplatTrainingSession::set_pixel_state_grad(
            static_cast<float*>(v_pixel_state_buf.ptr),
            static_cast<size_t>(v_pixel_state_buf.size));
    }

    py::dict get_train_image_path(size_t idx) {
        auto image = VkSplatTrainingSession::get_train_image_path(idx);
        py::dict result;
        result["image_path"] = image.image_path;
        if (!image.mask_path.empty()) {
            result["mask_path"] = image.mask_path;
        }
        return result;
    }

    py::dict get_val_image_path(size_t idx) {
        auto image = VkSplatTrainingSession::get_val_image_path(idx);
        py::dict result;
        result["image_path"] = image.image_path;
        if (!image.mask_path.empty()) {
            result["mask_path"] = image.mask_path;
        }
        return result;
    }
};

#define DEF_BUFFER_0(key, buffer_name, dtype, ...) \
    def_property_readonly(#key, [](PyVkSplat& self) { \
        return self.buffer_to_array<dtype, dtype>(self.buffers.buffer_name, __VA_ARGS__); \
    })

#define DEF_BUFFER(key, dtype, ...) \
    DEF_BUFFER_0(key, key, dtype, __VA_ARGS__)

#define DEF_ARRAY(key, dtype) \
    def("_set_" #key, [](PyVkSplat& self, py::array_t<dtype> array) { \
        self.array_to_buffer(array, self.buffers.key); \
    }, "Directly set `" #key "` buffer (use at own risk)", py::arg(#key))

#define DEF_BUFFER_ARRAY(key, dtype, ...) \
    DEF_BUFFER(key, dtype, __VA_ARGS__) \
    .DEF_ARRAY(key, dtype)

PYBIND11_MODULE(vksplat, m) {
    m.doc() = "Vulkan Gaussian Rasterization Python Bindings";

    py::class_<PyVkSplat>(m, "VkSplat")
        .def(py::init<>())
        .def("initialize", &PyVkSplat::initialize, "Initialize the trainer with SPIRV code")
        .def("set_train_config", &PyVkSplat::set_train_config, "Set training configuration")
        .def_property_readonly("num_train", &PyVkSplat::num_train)
        .def_property_readonly("num_val", &PyVkSplat::num_val)
        .def("get_train_image_path", &PyVkSplat::get_train_image_path)
        .def("get_val_image_path", &PyVkSplat::get_val_image_path)
        .def("get_train_image", &PyVkSplat::get_train_image)
        .def("get_val_image", &PyVkSplat::get_val_image)
        .def("set_uniforms", &PyVkSplat::set_uniforms, "Set trainer uniforms (image/camera info)",
             py::arg("active_sh"), py::arg("world_view_transform"),
             py::arg("image_height"), py::arg("image_width"),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
             py::arg("is_fisheye"))
        .def("get_uniforms", &PyVkSplat::get_uniforms, "Get trainer uniforms as a dict")
        .def("set_gauss_params", &PyVkSplat::set_gauss_params, "Set Gaussian parameters",
             py::arg("xyz_ws"), py::arg("sh_coeffs"), py::arg("rotations"),
             py::arg("scales"), py::arg("opacities"))
        .def("set_pixel_state", &PyVkSplat::set_pixel_state,
             "Set output pixels and n_contributors",
             py::arg("pixel_state"), py::arg("n_contributors"))
        .def("set_pixel_state_grad", &PyVkSplat::set_pixel_state_grad,
             "Set gradient of output pixels",
             py::arg("v_pixel_state"))
        .def("projection_forward", &PyVkSplat::projection_forward,
             "Execute the forward projection trainer")
        .def("process_tiles", &PyVkSplat::process_tiles, "Process tiles")
        .def("rasterize_forward", &PyVkSplat::rasterize_forward,
             "Execute the forward rasterization trainer")
        .def("rasterize_backward", &PyVkSplat::rasterize_backward,
             "Execute the backward rasterization trainer")
        .def("forward", &PyVkSplat::forward, "Execute forward all the way")
        .def("compute_pixel_loss_grad", &PyVkSplat::compute_pixel_loss_grad,
             "Execute compute gradient of pixel loss")
        .def("set_train_image", &PyVkSplat::set_train_image,
             "Train on the give image for one step.")
        .def("train_step", &PyVkSplat::train_step,
             "Train on the give image for one step.")
        .def("render_train", &PyVkSplat::render_train,
             "Render the train camera pose to buffer.")
        .def("render_val", &PyVkSplat::render_val,
             "Render the eval camera pose to buffer.")
        .def("post_backward_step", &PyVkSplat::post_backward_step,
             "Run this after each step for densification.")
        .def("get_vram_usage", &PyVkSplat::get_vram_usage, "Get VRAM Usage")
        .def("get_peak_vram_usage", &PyVkSplat::get_peak_vram_usage, "Get peak VRAM Usage")
        .def("get_timing_breakdown", &PyVkSplat::get_timing_breakdown,
             "Get timing for each step since start")
        .def("get_vram_breakdown", &PyVkSplat::get_vram_breakdown,
             "Get VRAM for each buffer at the current moment")
        .def("write_ply", &PyVkSplat::write_ply, "Save PLY file")
        .def("cleanup", &PyVkSplat::cleanup, "Cleanup Vulkan resources")
        .DEF_BUFFER_ARRAY(xyz_ws, float, 3, static_cast<size_t>(-1), "mean")
        .DEF_BUFFER_ARRAY(sh_coeffs, float, 16, 3, "sh")
        .DEF_BUFFER_ARRAY(rotations, float, 4)
        .DEF_BUFFER_0(scales, scales_opacs, float, 4, static_cast<size_t>(-1), "", 3, 0)
        .DEF_BUFFER_0(opacities, scales_opacs, float, 4, static_cast<size_t>(-1), "", 1, 3)
        .DEF_BUFFER_ARRAY(tiles_touched, int32_t, static_cast<size_t>(-1))
#if VKSPLAT_USE_EMULATED_INT64
        .DEF_BUFFER_ARRAY(rect_tile_space, rectTileSpace_t, VKSPLAT_RECT_TILE_SPACE_WORDS)
#else
        .DEF_BUFFER_ARRAY(rect_tile_space, rectTileSpace_t, static_cast<size_t>(-1))
#endif
        .DEF_BUFFER_ARRAY(radii, int32_t, static_cast<size_t>(-1))
        .DEF_BUFFER_ARRAY(xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(depths, float, 1)
        .DEF_BUFFER_ARRAY(inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(rgb, float, 3)
        .DEF_BUFFER_ARRAY(index_buffer_offset, int32_t, static_cast<size_t>(-1))
        .DEF_BUFFER_ARRAY(tile_ranges, int32_t, 2)
        .DEF_BUFFER_ARRAY(pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(n_contributors, int32_t, self.uniforms.image_width, 1)
        .DEF_BUFFER_ARRAY(ssim_map, float, self.uniforms.image_width, 12)
        .DEF_BUFFER_ARRAY(v_pixel_state, float, self.uniforms.image_width, 4)
        .DEF_BUFFER_ARRAY(v_xy_vs, float, 2)
        .DEF_BUFFER_ARRAY(v_depths, float, 1)
        .DEF_BUFFER_ARRAY(v_inv_cov_vs_opacity, float, 4)
        .DEF_BUFFER_ARRAY(v_rgb, float, 3)
        .def("_set_num_splats", [](PyVkSplat& self, size_t n) {
            self.buffers.num_splats = n;
        }, "Directly set buffers.num_splats (use at own risk)", py::arg("num_splats"));

    py::register_exception<std::runtime_error>(m, "RuntimeError");
}
