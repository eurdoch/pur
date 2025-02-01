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

    /// Perform a forward pass through the network and return all activations
    pub fn forward(&self, input: Vec<f64>) -> Vec<Vec<f64>> {
        // Placate all layer outputs
        let mut activations = vec![input];
        
        for layer in &self.layers {
            let output = layer.forward_propagate(activations.last().unwrap());
            activations.push(output);
        }

        activations
    }

    /// Compute Mean Squared Error loss
    fn compute_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        predicted.iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / predicted.len() as f64
    }

    /// Perform backward pass and return weight and bias gradients
    fn backward(&self, activations: Vec<Vec<f64>>, target: Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Initialize gradient storage with the correct sizes for each layer
        let mut weight_grads = self.layers.iter()
            .map(|layer| vec![0.0; layer.weights.len()])
            .collect::<Vec<_>>();
        let mut bias_grads = self.layers.iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect::<Vec<_>>();

        // Compute error for output layer
        let mut errors = activations.last().unwrap().iter()
            .zip(target.iter())
            .map(|(a, t)| a - t)
            .collect::<Vec<f64>>();

        for (i, layer) in self.layers.iter().enumerate().rev() {
            // Compute gradients for weights and biases
            let activation_derivatives: Vec<f64> = activations[i + 1].iter()
                .map(|&x| layer.activation.derivative(x as f32) as f64)
                .collect();
            
            for (neuron_idx, &error) in errors.iter().enumerate() {
                for (input_idx, &prev_activation) in activations[i].iter().enumerate() {
                    weight_grads[i][neuron_idx * layer.inputs + input_idx] += error * activation_derivatives[neuron_idx] * prev_activation;
                }
                bias_grads[i][neuron_idx] += error * activation_derivatives[neuron_idx];
            }

            // Propagate error to previous layer
            if i > 0 {
                let mut new_errors = vec![0.0; layer.inputs];
                
                for (neuron_idx, &error) in errors.iter().enumerate() {
                    for (input_idx, _) in activations[i].iter().enumerate() {
                        new_errors[input_idx] += layer.weights[neuron_idx * layer.inputs + input_idx] as f64 * error;
                    }
                }

                errors = new_errors;
            }
        }

        (weight_grads, bias_grads)
    }

    /// Update model weights based on computed gradients
    fn update_weights(&mut self, weight_grads: Vec<Vec<f64>>, bias_grads: Vec<Vec<f64>>, learning_rate: f64) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for j in 0..layer.weights.len() {
                layer.weights[j] -= (weight_grads[i][j] * learning_rate) as f32;
            }

            for j in 0..layer.biases.len() {
                layer.biases[j] -= (bias_grads[i][j] * learning_rate) as f32;
            }
        }
    }

    /// Train the model for one iteration using a single data point
    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>, learning_rate: f64) -> f64 {
        // Forward pass to get activations
        let activations = self.forward(input);

        // Compute loss
        let loss = self.compute_loss(activations.last().unwrap(), &target);

        // Backward pass to compute gradients
        let (weight_grads, bias_grads) = self.backward(activations, target);

        // Update weights using computed gradients
        self.update_weights(weight_grads, bias_grads, learning_rate);

        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

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

    #[test]
    fn test_forward_pass() {
        // Create a multi-layer model for detailed forward pass testing
        let model = Model::new(
            &[(3, 4), (4, 2)], 
            ActivationType::ReLU, 
            WeightInitStrategy::Xavier
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
            WeightInitStrategy::Random
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
            WeightInitStrategy::Xavier
        );

        // Simple training scenario with a fixed input and target
        let input = vec![0.5, 1.5];
        let target = vec![1.0];
        let learning_rate = 0.01;

        // Perform training and check loss
        let loss = model.train(input, target, learning_rate);

        // Loss should be a finite number between 0 and some reasonable upper bound
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        // We don't check for a specific loss value as it depends on initialization
    }

    #[test]
    fn test_compute_loss() {
        let model = Model::new(
            &[(2, 3), (3, 1)], 
            ActivationType::ReLU, 
            WeightInitStrategy::Xavier
        );

        // Test cases for loss computation
        let test_cases = vec![
            (vec![1.0], vec![1.0], 0.0),  // Perfect prediction
            (vec![0.5], vec![1.0], 0.25),  // Moderate error
            (vec![0.0], vec![1.0], 1.0),  // Large error
        ];

        for (predicted, target, expected_loss) in test_cases {
            let computed_loss = model.compute_loss(&predicted, &target);
            assert!((computed_loss - expected_loss).abs() < 1e-6, 
                "Loss computation failed. Got {}, expected {}", 
                computed_loss, expected_loss);
        }
    }
}