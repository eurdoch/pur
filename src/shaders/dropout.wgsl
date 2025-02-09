// Buffer bindings
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;

struct DropoutParams {
    dropout_rate: f32,
    scale: f32,
    is_training: u32,
    size: u32,
}
@group(0) @binding(3) var<uniform> params: DropoutParams;

// Random number generation (xoshiro128** algorithm)
struct Rng {
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
}

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn next_u32(rng: ptr<private, Rng>) -> u32 {
    let result = rotl((*rng).s1 * 5u, 7u) * 9u;
    let t = (*rng).s1 << 17u;
    
    (*rng).s2 ^= (*rng).s0;
    (*rng).s3 ^= (*rng).s1;
    (*rng).s1 ^= (*rng).s2;
    (*rng).s0 ^= (*rng).s3;
    
    (*rng).s2 ^= t;
    (*rng).s3 = rotl((*rng).s3, 45u);
    
    return result;
}

// Convert u32 to float in [0, 1)
fn random_float(rng: ptr<private, Rng>) -> f32 {
    return f32(next_u32(rng)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if (idx >= params.size) {
        return;
    }
    
    if (params.is_training != 0u) {
        // Initialize RNG with a different seed for each thread
        var rng: Rng;
        rng.s0 = 0x9E3779B9u + idx;
        rng.s1 = 0x243F6A88u ^ idx;
        rng.s2 = 0xB7E15162u + idx;
        rng.s3 = 0x93C63F9Au ^ idx;
        
        // Generate random mask
        let rand = random_float(&rng);
        let mask_val = select(0.0, params.scale, rand > params.dropout_rate);
        mask[idx] = mask_val;
        
        // Apply mask to input
        output[idx] = input[idx] * mask_val;
    } else {
        // During inference, just pass through the input
        output[idx] = input[idx];
    }
}
