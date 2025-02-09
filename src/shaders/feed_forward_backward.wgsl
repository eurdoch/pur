// Buffer bindings
@group(0) @binding(0) var<storage, read> weights: array<f32>;        // (neurons x inputs)
@group(0) @binding(1) var<storage, read> bias: array<f32>;          // (neurons)
@group(0) @binding(2) var<storage, read_write> weight_grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> bias_grads: array<f32>;
@group(0) @binding(4) var<storage, read> activation: array<f32>;     // Output gradients
@group(0) @binding(5) var<storage, read> preactivation: array<f32>;  // Pre-activation values

struct Params {
    num_inputs: u32,
    num_neurons: u32,
    activation_type: u32, // 0=ReLU, 1=Sigmoid, 2=Tanh, 3=Softmax
}
@group(0) @binding(6) var<uniform> params: Params;

// Activation derivatives
fn relu_derivative(x: f32) -> f32 {
    return select(0.0, 1.0, x > 0.0);
}

fn sigmoid_derivative(x: f32) -> f32 {
    let s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 - s);
}

fn tanh_derivative(x: f32) -> f32 {
    let t = tanh(x);
    return 1.0 - t * t;
}

// Get activation derivative based on type
fn get_activation_derivative(x: f32, activation_type: u32) -> f32 {
    switch activation_type {
        case 0u: { return relu_derivative(x); }
        case 1u: { return sigmoid_derivative(x); }
        case 2u: { return tanh_derivative(x); }
        default: { return 1.0; } // Linear or Softmax (handled separately)
    }
}

// Compute gradients with respect to weights and biases
@compute @workgroup_size(16, 16, 1)
fn backward_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let neuron_idx = global_id.x;
    let input_idx = global_id.y;
    
    // Bounds check
    if (neuron_idx >= params.num_neurons || input_idx >= params.num_inputs) {
        return;
    }
    
    // Get output gradient and multiply by activation derivative
    let output_grad = activation[neuron_idx];
    let preact_val = preactivation[neuron_idx];
    
    let grad = if (params.activation_type == 3u) {
        // Special case for softmax
        output_grad
    } else {
        output_grad * get_activation_derivative(preact_val, params.activation_type)
    };
    
    // Update weight gradient
    let weight_idx = neuron_idx * params.num_inputs + input_idx;
    let input_val = activation[input_idx]; // Previous layer's activation
    weight_grads[weight_idx] = grad * input_val;
    
    // Update bias gradient (only when input_idx is 0 to avoid multiple updates)
    if (input_idx == 0u) {
        bias_grads[neuron_idx] = grad;
    }
}

// Compute gradients with respect to inputs
@compute @workgroup_size(256)
fn backward_inputs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_idx = global_id.x;
    
    if (input_idx >= params.num_inputs) {
        return;
    }
    
    var input_grad = 0.0;
    
    // Accumulate gradients from all neurons
    for (var n = 0u; n < params.num_neurons; n = n + 1u) {
        let output_grad = activation[n];
        let preact_val = preactivation[n];
        
        let grad = if (params.activation_type == 3u) {
            // Special case for softmax
            output_grad
        } else {
            output_grad * get_activation_derivative(preact_val, params.activation_type)
        };
        
        let weight_idx = n * params.num_inputs + input_idx;
        input_grad += grad * weights[weight_idx];
    }
    
    // Store input gradient in activation buffer for next layer
    activation[input_idx] = input_grad;
}
