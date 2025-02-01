use pur::{
    Layer,
    ActivationType,
    WeightInitStrategy
};

#[test]
fn test_layer_initialization() {
    let layer = Layer::new(
        3,  // inputs 
        4,  // neurons
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier
    );

    // Check layer configuration
    assert_eq!(layer.inputs, 3);
    assert_eq!(layer.neurons, 4);
    
    // Verify weights dimensions
    assert_eq!(layer.weights.len(), 3 * 4);
    
    // Verify biases dimensions
    assert_eq!(layer.biases.len(), 4);
}

#[test]
fn test_forward_propagate() {
    let layer = Layer::new(
        3,  // inputs 
        2,  // neurons
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier
    );

    // Prepare input vector 
    let input = vec![1.0, 2.0, 3.0];
    
    // Perform forward propagation
    let output = layer.forward_propagate(&input);

    // Verify output dimensions
    assert_eq!(output.len(), 2);
    
    // Each output should be a transformed value
    assert!(output[0] >= 0.0);  // ReLU ensures non-negative
    assert!(output[1] >= 0.0);
}

#[test]
#[should_panic(expected = "Input size does not match layer's input size")]
fn test_forward_propagate_invalid_input_size() {
    let layer = Layer::new(
        3,  // inputs 
        2,  // neurons
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier
    );

    // Try to forward propagate with incorrect input size
    let invalid_input = vec![1.0, 2.0];
    layer.forward_propagate(&invalid_input);
}

#[test]
fn test_parameter_count() {
    let layer = Layer::new(
        3,  // inputs 
        4,  // neurons
        ActivationType::ReLU, 
        WeightInitStrategy::Xavier
    );

    // Parameter count should be weights + biases
    assert_eq!(layer.parameter_count(), 3 * 4 + 4);
}

#[test]
fn test_weight_initialization_strategies() {
    let strategies = vec![
        WeightInitStrategy::Random, 
        WeightInitStrategy::Xavier, 
        WeightInitStrategy::HeNormal
    ];

    for strategy in strategies {
        let layer = Layer::new(
            3,  // inputs 
            4,  // neurons
            ActivationType::ReLU, 
            strategy
        );

        // Check randomization
        assert_ne!(
            layer.weights[0], 0.0, 
            "Failed with strategy: {:?}", strategy
        );
    }
}

#[test]
fn test_reinitialize_weights() {
    let mut layer = Layer::new(
        3,  // inputs 
        4,  // neurons
        ActivationType::ReLU, 
        WeightInitStrategy::Random
    );

    // Store initial weights 
    let initial_weights = layer.weights.clone();

    // Reinitialize with a different strategy
    layer.initialize_weights(WeightInitStrategy::Xavier);

    // Check that weights have changed
    assert_ne!(
        layer.weights, 
        initial_weights, 
        "Weights should be different after reinitialization"
    );
}