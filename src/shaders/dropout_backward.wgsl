// Buffer bindings
@group(0) @binding(0) var<storage, read> input: array<f32>;        // Input gradients
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // Output gradients
@group(0) @binding(2) var<storage, read> mask: array<f32>;         // Dropout mask

struct DropoutParams {
    dropout_rate: f32,
    scale: f32,
    is_training: u32,
    size: u32,
}
@group(0) @binding(3) var<uniform> params: DropoutParams;

@compute @workgroup_size(256)
fn backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.size) {
        return;
    }
    
    if (params.is_training != 0u) {
        // During training, multiply gradients by the mask
        output[idx] = input[idx] * mask[idx];
    } else {
        // During inference, just pass through the gradients
        output[idx] = input[idx];
    }
}
