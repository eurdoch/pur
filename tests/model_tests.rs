use pur::{
    Model, 
    ActivationType, 
    WeightInitStrategy
};

#[test]
fn test_parameter_count_accuracy() {
    // Test parameter count for different model configurations
    let test_cases = vec![
        // Simple 2-layer model: (10 inputs -> 5 neurons) + (5 inputs -> 3 neurons)
        // Weights: 10 * 5 + 5 * 3 = 50 + 15 = 65
        // Biases: 5 + 3 = 8
        // Total: 50 + 15 + 5 + 3 = 73
        (vec![(10, 5), (5, 3)], WeightInitStrategy::Xavier, 73),
        
        // Deeper model: (20 -> 16), (16 -> 12), (12 -> 8), (8 -> 4)
        // Weights: (20*16) + (16*12) + (12*8) + (8*4) = 320 + 192 + 96 + 32 = 640
        // Biases: 16 + 12 + 8 + 4 = 40
        // Total: 640 + 40 = 680
        (vec![(20, 16), (16, 12), (12, 8), (8, 4)], WeightInitStrategy::HeNormal, 680),
        
        // Another configuration: (15 -> 10), (10 -> 7), (7 -> 5)
        // Weights: (15*10) + (10*7) + (7*5) = 150 + 70 + 35 = 255
        // Biases: 10 + 7 + 5 = 22
        // Total: 150 + 70 + 35 + 10 + 7 + 5 = 277
        (vec![(15, 10), (10, 7), (7, 5)], WeightInitStrategy::Random, 277)
    ];

    for (layer_configs, weight_init, expected_params) in test_cases {
        let model = Model::new(
            &layer_configs, 
            ActivationType::ReLU, 
            weight_init,
            false
        );
        
        // Detailed parameter breakdown
        let layer_details: Vec<_> = model.layers.iter()
            .map(|layer| (layer.inputs, layer.neurons, 
                layer.inputs * layer.neurons, 
                layer.neurons))
            .collect();
        
        let actual_params = model.parameter_count();
        
        println!("Layer Configurations: {:?}", layer_configs);
        println!("Layer Details (inputs, neurons, weights, biases): {:?}", layer_details);
        println!("Expected Params: {}", expected_params);
        println!("Actual Params: {}", actual_params);
        
        assert_eq!(
            actual_params, 
            expected_params, 
            "Parameter count mismatch for model with {:?}", 
            layer_configs
        );
    }
}

#[test]
fn test_inference() {
    // Create a simple model: 2 inputs -> 3 hidden neurons -> 1 output
    let model = Model::new(
        &[(2, 3), (3, 1)], 
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier,
        false
    );

    // Test inference with a valid input
    let input = vec![1.0, 2.0];
    let output = model.inference(&input);

    // Verify output size matches the last layer's neuron count
    assert_eq!(output.len(), 1);

    // Ensure output is not exactly the same as input (some transformation occurred)
    assert_ne!(output[0], input[0]);
    assert_ne!(output[0], input[1]);
}

#[test]
#[should_panic(expected = "Input size")]
fn test_inference_invalid_input_size() {
    // Create a model with 2 inputs
    let model = Model::new(
        &[(2, 3), (3, 1)], 
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier,
        false
    );

    // Try to run inference with incorrect input size
    let invalid_input = vec![1.0, 2.0, 3.0];
    model.inference(&invalid_input);
}

#[test]
fn test_forward_pass() {
    // Create a multi-layer model for detailed forward pass testing
    let model = Model::new(
        &[(3, 4), (4, 2)], 
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier,
        false
    );

    // Test input
    let input = vec![1.0, 2.0, 3.0];
    let activations = model.forward(input);

    // Verify activations structure
    assert_eq!(activations.len(), 3);  // Input + 2 layers
    
    // Check first layer (input layer)
    assert_eq!(activations[0], vec![1.0, 2.0, 3.0]);
    
    // Verify sizes of layer activations
    assert_eq!(activations[1].len(), 4);  // First layer neurons
    assert_eq!(activations[2].len(), 2);  // Output layer neurons
}

#[test]
fn test_reinitialize_weights() {
    let mut model = Model::new(
        &[(3, 4), (4, 2)], 
        ActivationType::ReLU, 
        WeightInitStrategy::Random,
        false
    );

    // Store initial weights
    let initial_weights: Vec<Vec<f32>> = model.layers.iter()
        .map(|layer| layer.weights.clone())
        .collect();

    // Reinitialize with a different strategy
    model.reinitialize_weights(WeightInitStrategy::Xavier);

    // Check that weights have changed
    for (i, layer) in model.layers.iter().enumerate() {
        assert_ne!(
            layer.weights, 
            initial_weights[i], 
            "Weights for layer {} should be different after reinitialization", 
            i
        );
    }
}

#[test]
fn test_training_basic() {
    // Create a simple model for training test
    let mut model = Model::new(
        &[(2, 3), (3, 1)], 
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier,
        false
    );

    // Simple training scenario with a fixed input and target
    let input = vec![0.5, 1.5];
    let target = vec![1.0];
    let learning_rate = 0.01;

    // Perform training and check loss
    let loss = model.train(&input, &target, learning_rate);

    // Loss should be a finite number between 0 and some reasonable upper bound
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
    // We don't check for a specific loss value as it depends on initialization
}

#[test]
fn test_compute_loss() {
    // Test case 1: Model with softmax last layer (multiclass classification)
    let model_softmax = Model::new(
        &[(2, 3), (3, 3)],  // 3 output classes
        ActivationType::ReLU,
        WeightInitStrategy::Xavier,
        true  // Enable softmax for last layer
    );

    // Test softmax cross-entropy loss
    let predicted_softmax = vec![0.7, 0.2, 0.1];  // Softmax outputs (sum to 1)
    let target_softmax = vec![1.0, 0.0, 0.0];     // One-hot encoded target
    let loss_softmax = model_softmax.calculate_loss(&predicted_softmax, &target_softmax);
    assert!(loss_softmax > 0.0, "Softmax loss should be positive");
    assert!(loss_softmax.is_finite(), "Softmax loss should be finite");

    // Test case 2: Model with sigmoid output (binary classification)
    let model_sigmoid = Model::new(
        &[(2, 3), (3, 1)],  // Single output
        ActivationType::Sigmoid,
        WeightInitStrategy::Xavier,
        false
    );

    // Test binary cross-entropy loss
    let test_cases_binary = vec![
        (vec![0.7], vec![1.0]),    // Relatively good prediction
        (vec![0.1], vec![0.0]),    // Good prediction for negative class
        (vec![0.5], vec![1.0]),    // Uncertain prediction
    ];

    for (predicted, target) in test_cases_binary {
        let loss = model_sigmoid.calculate_loss(&predicted, &target);
        assert!(loss >= 0.0, "Binary cross-entropy loss should be positive");
        assert!(loss.is_finite(), "Binary cross-entropy loss should be finite");
    }

    // Test case 3: Model with regular output (regression)
    let model_regular = Model::new(
        &[(2, 3), (3, 2)],  // Multiple outputs
        ActivationType::ReLU,
        WeightInitStrategy::Xavier,
        false
    );

    // Test MSE loss
    let test_cases_mse = vec![
        (vec![0.5, 0.5], vec![1.0, 1.0]),    // Some error
        (vec![0.0, 0.0], vec![0.0, 0.0]),    // Perfect prediction
        (vec![2.0, -1.0], vec![0.0, 0.0]),   // Large error
    ];

    for (predicted, target) in test_cases_mse {
        let loss = model_regular.calculate_loss(&predicted, &target);
        assert!(loss >= 0.0, "MSE loss should be non-negative");
        assert!(loss.is_finite(), "MSE loss should be finite");
    }
}