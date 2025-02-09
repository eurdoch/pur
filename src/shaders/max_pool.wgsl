// Buffer bindings
@group(0) @binding(0) var<storage, read> input: array<f32>;          // Input feature maps
@group(0) @binding(1) var<storage, read_write> output: array<f32>;   // Output feature maps
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;  // Max indices (pairs of h,w)

struct PoolParams {
    channels: u32,
    input_height: u32,
    input_width: u32,
    pool_height: u32,
    pool_width: u32,
    stride: u32,
}
@group(0) @binding(3) var<uniform> params: PoolParams;

// Helper function to get input index
fn get_input_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * (params.channels * params.input_height * params.input_width) +
           c * (params.input_height * params.input_width) +
           h * params.input_width + w;
}

// Helper function to get output index
fn get_output_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let output_height = (params.input_height - params.pool_height) / params.stride + 1u;
    let output_width = (params.input_width - params.pool_width) / params.stride + 1u;
    return n * (params.channels * output_height * output_width) +
           c * (output_height * output_width) +
           h * output_width + w;
}

// Helper function to get indices buffer index (stores two u32s per position)
fn get_indices_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return get_output_idx(n, c, h, w) * 2u;
}

@compute @workgroup_size(8, 8, 1)
fn forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_height = (params.input_height - params.pool_height) / params.stride + 1u;
    let output_width = (params.input_width - params.pool_width) / params.stride + 1u;
    
    let c = global_id.z;  // channel
    let oh = global_id.x; // output height position
    let ow = global_id.y; // output width position
    
    // Bounds check
    if (c >= params.channels || oh >= output_height || ow >= output_width) {
        return;
    }
    
    // Input position corresponds to output position * stride
    let h_start = oh * params.stride;
    let w_start = ow * params.stride;
    
    // Find maximum in pooling window
    var max_val = -3.402823466e+38f;  // Float minimum
    var max_h = 0u;
    var max_w = 0u;
    
    for (var ph = 0u; ph < params.pool_height; ph = ph + 1u) {
        for (var pw = 0u; pw < params.pool_width; pw = pw + 1u) {
            let input_idx = get_input_idx(0u, c, h_start + ph, w_start + pw);
            let val = input[input_idx];
            
            if (val > max_val) {
                max_val = val;
                max_h = ph;
                max_w = pw;
            }
        }
    }
    
    // Store output
    let output_idx = get_output_idx(0u, c, oh, ow);
    output[output_idx] = max_val;
    
    // Store indices
    let indices_idx = get_indices_idx(0u, c, oh, ow);
    indices[indices_idx] = max_h;
    indices[indices_idx + 1u] = max_w;
}
