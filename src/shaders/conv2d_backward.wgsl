// Define the ConvParams struct to match the Rust struct
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

// Buffer bindings - same as forward pass
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight_grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias_grads: array<f32>;
@group(0) @binding(4) var<storage, read_write> input_grads: array<f32>;  // Changed from padded_input to input_grads and made read_write
@group(0) @binding(5) var<storage, read> activation: array<f32>;  // output gradients
@group(0) @binding(6) var<storage, read> preactivation: array<f32>;
@group(0) @binding(7) var<uniform> params: ConvParams;

// Helper functions from forward pass
fn get_weight_idx(oc: u32, ic: u32, kh: u32, kw: u32) -> u32 {
    let kernel_size = params.kernel_height * params.kernel_width;
    return oc * (params.in_channels * kernel_size) + ic * kernel_size + kh * params.kernel_width + kw;
}

fn get_padded_input_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let padded_height = params.input_height + 2u * params.padding;
    let padded_width = params.input_width + 2u * params.padding;
    return n * (params.in_channels * padded_height * padded_width) +
           c * (padded_height * padded_width) +
           h * padded_width + w;
}

fn get_output_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    let output_height = (params.input_height + 2u * params.padding - params.kernel_height) / params.stride + 1u;
    let output_width = (params.input_width + 2u * params.padding - params.kernel_width) / params.stride + 1u;
    return n * (params.out_channels * output_height * output_width) +
           c * (output_height * output_width) +
           h * output_width + w;
}

// Compute weight and bias gradients
@compute @workgroup_size(8, 8, 1)
fn backward_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let oc = global_id.x;
    let ic = global_id.y;
    let k_idx = global_id.z;

    // Convert k_idx to kernel height and width indices
    let kh = k_idx / params.kernel_width;
    let kw = k_idx % params.kernel_width;

    // Bounds check
    if (oc >= params.out_channels || ic >= params.in_channels ||
        kh >= params.kernel_height || kw >= params.kernel_width) {
        return;
    }

    let output_height = (params.input_height + 2u * params.padding - params.kernel_height) / params.stride + 1u;
    let output_width = (params.input_width + 2u * params.padding - params.kernel_width) / params.stride + 1u;

    var grad_sum = 0.0;

    // Accumulate gradients over all output positions
    for (var oh = 0u; oh < output_height; oh = oh + 1u) {
        for (var ow = 0u; ow < output_width; ow = ow + 1u) {
            let h_start = oh * params.stride;
            let w_start = ow * params.stride;

            let output_grad = activation[get_output_idx(0u, oc, oh, ow)];
            let input_val = input_grads[get_padded_input_idx(0u, ic, h_start + kh, w_start + kw)];

            grad_sum += input_val * output_grad;
        }
    }

    // Store weight gradient
    let weight_idx = get_weight_idx(oc, ic, kh, kw);
    weight_grads[weight_idx] = grad_sum;
}

// Compute bias gradients
@compute @workgroup_size(64, 1, 1)
fn backward_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let oc = global_id.x;

    if (oc >= params.out_channels) {
        return;
    }

    let output_height = (params.input_height + 2u * params.padding - params.kernel_height) / params.stride + 1u;
    let output_width = (params.input_width + 2u * params.padding - params.kernel_width) / params.stride + 1u;

    var grad_sum = 0.0;

    // Sum gradients over spatial dimensions
    for (var oh = 0u; oh < output_height; oh = oh + 1u) {
        for (var ow = 0u; ow < output_width; ow = ow + 1u) {
            grad_sum += activation[get_output_idx(0u, oc, oh, ow)];
        }
    }

    bias_grads[oc] = grad_sum;
}

// Compute input gradients
@compute @workgroup_size(8, 8, 1)
fn backward_input(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ic = global_id.z;
    let ih = global_id.x;
    let iw = global_id.y;

    if (ic >= params.in_channels || ih >= params.input_height || iw >= params.input_width) {
        return;
    }

    let output_height = (params.input_height + 2u * params.padding - params.kernel_height) / params.stride + 1u;
    let output_width = (params.input_width + 2u * params.padding - params.kernel_width) / params.stride + 1u;

    var grad_sum = 0.0;

    // Compute range of output pixels that depend on this input pixel
    let ih_signed = i32(ih);
    let iw_signed = i32(iw);
    let padding_signed = i32(params.padding);
    let kernel_height_signed = i32(params.kernel_height);
    let kernel_width_signed = i32(params.kernel_width);
    let stride_signed = i32(params.stride);

    let oh_start = u32(max(0, 
        (ih_signed + padding_signed - kernel_height_signed + stride_signed) / stride_signed
    ));
    let oh_end = min(output_height, (ih + params.padding) / params.stride + 1u);
    let ow_start = u32(max(0, 
        (iw_signed + padding_signed - kernel_width_signed + stride_signed) / stride_signed
    ));
    let ow_end = min(output_width, (iw + params.padding) / params.stride + 1u);

    // Accumulate gradients
    for (var oc = 0u; oc < params.out_channels; oc = oc + 1u) {
        for (var oh = oh_start; oh < oh_end; oh = oh + 1u) {
            for (var ow = ow_start; ow < ow_end; ow = ow + 1u) {
                let h_start = oh * params.stride;
                let w_start = ow * params.stride;

                let kh = ih + params.padding - h_start;
                let kw = iw + params.padding - w_start;

                let weight = weights[get_weight_idx(oc, ic, kh, kw)];
                let output_grad = activation[get_output_idx(0u, oc, oh, ow)];

                grad_sum += weight * output_grad;
            }
        }
    }

    // Store input gradient
    let input_idx = get_padded_input_idx(0u, ic, ih + params.padding, iw + params.padding);
    input_grads[input_idx] = grad_sum;
}
