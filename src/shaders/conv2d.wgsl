// Buffer bindings
@group(0) @binding(0) var<storage, read> weights: array<f32>;        // Flattened weights
@group(0) @binding(1) var<storage, read> bias: array<f32>;          // Output channel biases
@group(0) @binding(2) var<storage, read_write> weight_grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias_grads: array<f32>;
@group(0) @binding(4) var<storage, read_write> padded_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> activation: array<f32>;
@group(0) @binding(6) var<storage, read_write> preactivation: array<f32>;

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
@group(0) @binding(7) var<uniform> params: ConvParams;

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

// Activation functions
fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
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
    
    // Calculate output index
    let output_idx = oc * (output_height * output_width) + oh * output_width + ow;
    
    // Store preactivation
    preactivation[output_idx] = sum;
    
    // Apply activation function (assuming ReLU for now)
    activation[output_idx] = relu(sum);
}
