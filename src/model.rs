use crate::layer::{Layer, WeightInitStrategy};
use crate::activation::ActivationType;
use crate::hyperparameters::ModelHyperparameters;

/// Represents a neural network model
#[derive(Debug, Clone)]
pub struct Model {
    /// Layers of the neural network
    pub layers: Vec<Layer>,
    
    /// Hyperparameters for the model
    pub hyperparameters: ModelHyperparameters,
}

impl Model {
    /// Create a new neural network model with specified layer configurations
    ///
    /// # Arguments
    ///
    /// * `layer_configs` - A vector of tuples specifying (inputs, neurons)
    /// * `activation` - Default activation function for all layers
    /// * `weight_init` - Weight initialization strategy
    pub fn new(
        layer_configs: &[(usize, usize)], 
        activation: ActivationType,
        weight_init: WeightInitStrategy
    ) -> Self {
        // Validate layer configurations
        if layer_configs.len() < 2 {
            panic!("At least two layers (input and output) are required");
        }

        // Create layers
        let mut layers = Vec::new();
        
        // Iterate through all layer configurations
        for &(inputs, neurons) in layer_configs.iter() {
            // Create layer with specified configuration
            let layer = Layer::new(
                inputs,   // Number of inputs from previous layer
                neurons,  // Number of neurons in current layer
                activation,
                weight_init
            );
            
            layers.push(layer);
        }
        
        Model {
            layers,
            hyperparameters: ModelHyperparameters::default(),
        }
    }
    
    /// Get total number of parameters in the model
    pub fn parameter_count(&self) -> usize {
        self.layers.iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }
    
    /// Reinitialize all layer weights
    pub fn reinitialize_weights(&mut self, weight_init: WeightInitStrategy) {
        for layer in &mut self.layers {
            layer.initialize_weights(weight_init);
        }
    }

    /// Perform inference (forward propagation) on input data
    ///
    /// # Arguments
    /// 
    /// * `input` - Input data vector
    ///
    /// # Returns
    ///
    /// Final layer output after propagating through all layers
    pub fn inference(&self, mut input: Vec<f64>) -> Vec<f64> {
        // Validate input size matches first layer's input size
        if input.len() != self.layers[0].inputs {
            panic!(
                "Input size {} does not match first layer's input size {}", 
                input.len(), 
                self.layers[0].inputs
            );
        }

        // Propagate through each layer
        for layer in &self.layers {
            // Compute layer's output based on current input
            input = layer.forward_propagate(&input);
        }

        // Return final layer output
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                weight_init
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
            WeightInitStrategy::Xavier
        );

        // Test inference with a valid input
        let input = vec![1.0, 2.0];
        let output = model.inference(input.clone());

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
            WeightInitStrategy::Xavier
        );

        // Try to run inference with incorrect input size
        let invalid_input = vec![1.0, 2.0, 3.0];
        model.inference(invalid_input);
    }
}
