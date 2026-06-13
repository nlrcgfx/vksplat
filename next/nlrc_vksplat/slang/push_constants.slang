namespace PushConstants {

struct ElementCount {
  uint32_t numElements;
}

struct Renderer {
    uint image_height;
    uint image_width;
    uint grid_height;
    uint grid_width;
    uint num_splats;
    uint active_sh;
    uint step;
    uint camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float4x4 world_view_transform;
}

}
