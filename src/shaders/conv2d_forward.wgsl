// First bind group - Input related bindings
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> padded_input: array<f32>;
@group(0) @binding(3) var<uniform> params: ConvParams;

// Second bind group - Output related bindings
@group(1) @binding(0) var<storage, read_write> activation: array<f32>;
@group(1) @binding(1) var<storage, read_write> preactivation: array<f32>;

struct ConvParams {
    in_channels: u32,
    out_channels: u32,
    input_height: u32,
    input_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride: u32,
    padding: u32,
}

// Helper function to get weight index in the flattened array
fn get_weight_idx(oc: u32, ic: u32, kh: u32, kw: u32) -> u32 {
    let kernel_size = params.kernel_height * params.kernel_width;
    return oc * (params.in_channels * kernel_size) + ic * kernel_size + kh * params.kernel_width + kw;
}

// Helper function to get padded input index
fn get_padded_input_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let padded_height = params.input_height + 2u * params.padding;
    let padded_width = params.input_width + 2u * params.padding;
    return n * (params.in_channels * padded_height * padded_width) + 
           c * (padded_height * padded_width) + 
           h * padded_width + w;
}

// Helper function to get output index
fn get_output_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let output_height = (params.input_height + 2u * params.padding - params.kernel_height) / params.stride + 1u;
    let output_width = (params.input_width + 2u * params.padding - params.kernel_width) / params.stride + 1u;
    return n * (params.out_channels * output_height * output_width) +
           c * (output_height * output_width) +
           h * output_width + w;
}

// Activation functions
fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

@compute @workgroup_size(8, 8, 1)
fn forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_height = (params.input_height + 2u * params.padding - params.kernel_height) / params.stride + 1u;
    let output_width = (params.input_width + 2u * params.padding - params.kernel_width) / params.stride + 1u;
    
    // Each thread processes one output pixel for one output channel
    let oc = global_id.z;  // output channel
    let oh = global_id.x;  // output height position
    let ow = global_id.y;  // output width position
    
    // Bounds check
    if (oc >= params.out_channels || oh >= output_height || ow >= output_width) {
        return;
    }
    
    // Compute convolution for this output pixel
    var sum = 0.0;
    
    // Input position corresponds to output position * stride
    let h_start = oh * params.stride;
    let w_start = ow * params.stride;
    
    // Iterate over input channels and kernel
    for (var ic = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh = 0u; kh < params.kernel_height; kh = kh + 1u) {
            for (var kw = 0u; kw < params.kernel_width; kw = kw + 1u) {
                let input_idx = get_padded_input_idx(0u, ic, h_start + kh, w_start + kw);
                let weight_idx = get_weight_idx(oc, ic, kh, kw);
                sum = sum + padded_input[input_idx] * weights[weight_idx];
            }
        }
    }
    
    // Add bias
    sum = sum + bias[oc];
    
    // Get output index for this thread
    let output_idx = get_output_idx(0u, oc, oh, ow);
    
    // Store preactivation
    preactivation[output_idx] = sum;
    
    // Apply activation function (ReLU)
    activation[output_idx] = relu(sum);
}
