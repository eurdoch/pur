struct ConvParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_height: u32,
    input_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride: u32,
    padding: u32,
    output_height: u32,
    output_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvParams;

fn get_input_idx(b: u32, c: u32, h: u32, w: u32) -> u32 {
    return ((b * params.in_channels + c) * params.input_height + h) * params.input_width + w;
}

fn get_weight_idx(oc: u32, ic: u32, kh: u32, kw: u32) -> u32 {
    return ((oc * params.in_channels + ic) * params.kernel_height + kh) * params.kernel_width + kw;
}

fn get_output_idx(b: u32, c: u32, h: u32, w: u32) -> u32 {
    return ((b * params.out_channels + c) * params.output_height + h) * params.output_width + w;
}

@compute @workgroup_size(8, 8, 1)
fn conv2d_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    let out_channel = workgroup_id.z;
    
    // Check if we're within output bounds
    if (x >= params.output_width || y >= params.output_height) {
        return;
    }

    // For each batch
    for (var batch = 0u; batch < params.batch_size; batch = batch + 1u) {
        var sum = 0.0;
        
        // For each input channel
        for (var ic = 0u; ic < params.in_channels; ic = ic + 1u) {
            // For each kernel position
            for (var kh = 0u; kh < params.kernel_height; kh = kh + 1u) {
                for (var kw = 0u; kw < params.kernel_width; kw = kw + 1u) {
                    let h_pos = i32(y * params.stride + kh) - i32(params.padding);
                    let w_pos = i32(x * params.stride + kw) - i32(params.padding);
                    
                    // Check if we're within input bounds
                    if (h_pos >= 0 && 
                        w_pos >= 0 && 
                        h_pos < i32(params.input_height) && 
                        w_pos < i32(params.input_width)) {
                        
                        let input_val = input[get_input_idx(
                            batch,
                            ic,
                            u32(h_pos),
                            u32(w_pos)
                        )];
                        
                        let weight_val = weights[get_weight_idx(
                            out_channel,
                            ic,
                            kh,
                            kw
                        )];
                        
                        sum = sum + input_val * weight_val;
                    }
                }
            }
        }
        
        // Add bias and store result
        let output_idx = get_output_idx(batch, out_channel, y, x);
        output[output_idx] = sum + bias[out_channel];
    }
}
