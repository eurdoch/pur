// Buffer bindings match the bind group layout
@group(0) @binding(0) var<storage, read> weights: array<f32>; // neurons x inputs matrix
@group(0) @binding(1) var<storage, read> bias: array<f32>;    // neurons vector
@group(0) @binding(2) var<storage, read_write> weight_grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias_grads: array<f32>;
@group(0) @binding(4) var<storage, read_write> activation: array<f32>;     // output
@group(0) @binding(5) var<storage, read_write> preactivation: array<f32>;  // pre-activation output

struct Params {
    num_inputs: u32,
    num_neurons: u32,
    activation_type: u32, // 0=ReLU, 1=Sigmoid, 2=Tanh, 3=Softmax
}
@group(0) @binding(6) var<uniform> params: Params;

// Activation functions
fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn tanh_custom(x: f32) -> f32 {
    return tanh(x);
}

@compute @workgroup_size(256)
fn forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let neuron_idx = global_id.x;
    
    // Check if this thread should process a neuron
    if (neuron_idx >= params.num_neurons) {
        return;
    }

    // Compute matrix multiplication for this neuron
    var sum = 0.0;
    for (var i = 0u; i < params.num_inputs; i = i + 1u) {
        let weight_idx = neuron_idx * params.num_inputs + i;
        sum = sum + weights[weight_idx] * activation[i];  // input is in activation buffer
    }
    
    // Add bias
    sum = sum + bias[neuron_idx];
    
    // Store preactivation
    preactivation[neuron_idx] = sum;
    
    // Apply activation function
    var result = sum;
    switch params.activation_type {
        case 0u: { // ReLU
            result = relu(sum);
        }
        case 1u: { // Sigmoid
            result = sigmoid(sum);
        }
        case 2u: { // Tanh
            result = tanh_custom(sum);
        }
        default: { // Linear (no activation)
            result = sum;
        }
    }
    
    // Store result
    activation[neuron_idx] = result;
}
