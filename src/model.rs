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

    /// Whether softmax is applied to the last layer
    pub softmax_last_layer: bool,
}

impl Model {
    /// Create a new neural network model with specified layer configurations
    ///
    /// # Arguments
    ///
    /// * `layer_configs` - A vector of tuples specifying (inputs, neurons)
    /// * `activation` - Default activation function for all layers
    /// * `weight_init` - Weight initialization strategy
    /// * `softmax_last_layer` - Whether to apply softmax to the last layer
    pub fn new(
        layer_configs: &[(usize, usize)], 
        activation: ActivationType,
        weight_init: WeightInitStrategy,
        softmax_last_layer: bool
    ) -> Self {
        // Validate layer configurations
        if layer_configs.len() < 2 {
            panic!("At least two layers (input and output) are required");
        }

        // Create layers
        let mut layers = Vec::new();
        
        // Iterate through all layer configurations
        for (idx, &(inputs, neurons)) in layer_configs.iter().enumerate() {
            // Check if this is the last layer
            let layer_activation = if idx == layer_configs.len() - 1 && softmax_last_layer {
                ActivationType::Softmax
            } else {
                activation
            };

            // Create layer with specified configuration
            let layer = Layer::new(
                inputs,   // Number of inputs from previous layer
                neurons,  // Number of neurons in current layer
                layer_activation,
                weight_init
            );
            
            layers.push(layer);
        }
        
        Model {
            layers,
            hyperparameters: ModelHyperparameters::default(),
            softmax_last_layer,
        }
    }
    
    /// Pretty print the weights of all layers
    #[cfg(not(tarpaulin))]
    pub fn print_weights(&self) {
        // [Previous implementation remains the same]
    }
    
    /// Pretty print the structure of the model without weights
    #[cfg(not(tarpaulin))]
    pub fn print_model(&self) {
        // [Previous implementation remains the same]
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
    pub fn inference(&self, input: &Vec<f64>) -> Vec<f64> {
        // Validate input size matches first layer's input size
        if input.len() != self.layers[0].inputs {
            panic!(
                "Input size {} does not match first layer's input size {}", 
                input.len(), 
                self.layers[0].inputs
            );
        }

        // Propagate through each layer
        let mut output = input.clone();
        for layer in &self.layers {
            // Compute layer's output based on current input
            output = layer.forward_propagate(&output);
        }

        // Return final layer output
        output
    }

    /// Perform a forward pass through the network and return all activations
    pub fn forward(&self, input: Vec<f64>) -> Vec<Vec<f64>> {
        // Store all layer outputs
        let mut activations = vec![input];
        
        for layer in &self.layers {
            let output = layer.forward_propagate(activations.last().unwrap());
            activations.push(output);
        }

        activations
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        if self.softmax_last_layer {
            // Ensure that targets are one-hot encoded (one 1.0, rest 0.0)
            let mut total_loss = 0.0;
            for (p, t) in predicted.iter().zip(target.iter()) {
                // Add small epsilon to prevent log(0)
                total_loss -= t * (p + 1e-10).ln();
            }
            total_loss
        } else {
            // Use binary cross-entropy for sigmoid output layers
            // or MSE for other activation functions
            if predicted.len() == 1 && target.len() == 1 {
                // Binary cross-entropy
                let p = predicted[0];
                let t = target[0];
                -(t * (p + 1e-10).ln() + (1.0 - t) * (1.0 - p + 1e-10).ln())
            } else {
                // Mean squared error as a fallback
                predicted.iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>() / predicted.len() as f64
            }
        }
    }

    /// Public method to compute loss
    pub fn calculate_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        // Validate input lengths match
        assert_eq!(predicted.len(), target.len(), "Predicted and target vectors must have the same length");
        self.compute_loss(predicted, target)
    }

    /// Perform backward pass and return weight and bias gradients
    fn backward(&self, activations: Vec<Vec<f64>>, target: Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut weight_grads = self.layers.iter()
            .map(|layer| vec![0.0; layer.weights.len()])
            .collect::<Vec<_>>();
        let mut bias_grads = self.layers.iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect::<Vec<_>>();

        // Compute error for output layer
        let mut errors = if self.softmax_last_layer {
            // For softmax + cross-entropy, the error is simply the difference 
            // between predicted probabilities and one-hot encoded targets
            activations.last().unwrap().iter()
                .zip(target.iter())
                .map(|(a, t)| a - t)
                .collect::<Vec<f64>>()
        } else {
            // For other activations, use standard error computation
            activations.last().unwrap().iter()
                .zip(target.iter())
                .map(|(a, t)| a - t)
                .collect::<Vec<f64>>()
        };

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

    /// Train the model using mini-batch gradient descent
    /// 
    /// # Arguments
    /// 
    /// * `inputs` - Vector of input samples, where each sample is a vector of features
    /// * `targets` - Vector of target values, where each target corresponds to an input
    /// * `batch_size` - Size of mini-batches for gradient descent
    /// * `learning_rate` - Learning rate for weight updates
    /// 
    /// # Returns
    /// 
    /// Average loss across all samples in the batch
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        batch_size: usize,
        learning_rate: f64
    ) -> f64 {
        assert_eq!(inputs.len(), targets.len(), "Number of inputs must match number of targets");
        assert!(batch_size > 0, "Batch size must be greater than 0");

        let mut total_loss = 0.0;
        let mut batch_weight_grads = self.layers.iter()
            .map(|layer| vec![0.0; layer.weights.len()])
            .collect::<Vec<_>>();
        let mut batch_bias_grads = self.layers.iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect::<Vec<_>>();

        // Process each sample in the batch
        for (input, target) in inputs.iter().zip(targets.iter()).take(batch_size) {
            // Forward pass
            let activations = self.forward(input.clone());
            
            // Compute loss
            let loss = self.compute_loss(activations.last().unwrap(), target);
            total_loss += loss;

            // Backward pass
            let (weight_grads, bias_grads) = self.backward(activations, target.clone());

            // Accumulate gradients
            for i in 0..self.layers.len() {
                for j in 0..batch_weight_grads[i].len() {
                    batch_weight_grads[i][j] += weight_grads[i][j];
                }
                for j in 0..batch_bias_grads[i].len() {
                    batch_bias_grads[i][j] += bias_grads[i][j];
                }
            }
        }

        // Average gradients over batch
        let batch_size_f64 = batch_size as f64;
        for grads in batch_weight_grads.iter_mut().chain(batch_bias_grads.iter_mut()) {
            for grad in grads.iter_mut() {
                *grad /= batch_size_f64;
            }
        }

        // Update weights using averaged gradients
        self.update_weights(batch_weight_grads, batch_bias_grads, learning_rate);

        // Return average loss
        total_loss / batch_size_f64
    }

    /// Train the model for one iteration using a single data point (kept for backward compatibility)
    pub fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) -> f64 {
        self.train_batch(&[input.clone()], &[target.clone()], 1, learning_rate)
    }
}
