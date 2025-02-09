// Buffer bindings - same as forward pass
@group(0) @binding(0) var<storage, read> input: array<f32>;          // Input gradients
@group(0) @binding(1) var<storage, read_write> output: array<f32>;   // Output gradients
@group(0) @binding(2) var<storage, read> indices: array<u32>;        // Max indices (pairs of h,w)

struct PoolParams {
    channels: u32,
    input_height: u32,
    input_width: u32,
    pool_height: u32,
    pool_width: u32,
    stride: u32,
}
@group(0) @binding(3) var<uniform> params: PoolParams;

// Helper functions
fn get_input_grad_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * (params.channels * params.input_height * params.input_width) +
           c * (params.input_height * params.input_width) +
           h * params.input_width + w;
}

fn get_output_grad_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let output_height = (params.input_height - params.pool_height) / params.stride + 1u;
    let output_width = (params.input_width - params.pool_width) / params.stride + 1u;
    return n * (params.channels * output_height * output_width) +
           c * (output_height * output_width) +
           h * output_width + w;
}

fn get_indices_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return get_output_grad_idx(n, c, h, w) * 2u;
}

@compute @workgroup_size(8, 8, 1)
fn backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.z;  // channel
    let h = global_id.x;  // height
    let w = global_id.y;  // width
    
    let output_height = (params.input_height - params.pool_height) / params.stride + 1u;
    let output_width = (params.input_width - params.pool_width) / params.stride + 1u;
    
    // Bounds check
    if (c >= params.channels || h >= output_height || w >= output_width) {
        return;
    }
    
    // Get output gradient and max indices
    let output_idx = get_output_grad_idx(0u, c, h, w);
    let indices_base = get_indices_idx(0u, c, h, w);
    let max_h = indices[indices_base];
    let max_w = indices[indices_base + 1u];
    
    // Calculate input position
    let h_start = h * params.stride;
    let w_start = w * params.stride;
    
    // Propagate gradient to max element position
    let input_idx = get_input_grad_idx(0u, c, h_start + max_h, w_start + max_w);
    output[input_idx] = input[output_idx];
}
