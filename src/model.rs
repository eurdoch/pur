use ndarray::Array1;

use crate::layer::Layer;
use crate::activation::ActivationType;

#[derive(Debug, Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
}

pub struct LayerConfig {
    pub neurons: usize,
    pub inputs: usize,
    pub activation: ActivationType,
}

impl Model {
    /// Create a new neural network model with specified layer configurations
    ///
    /// # Arguments
    ///
    /// * `layer_configs` - A vector of tuples specifying (inputs, neurons, activation)
    pub fn new(
        layer_configs: Vec<LayerConfig>, 
    ) -> Self {
        if layer_configs.len() < 2 {
            panic!("At least two layers (input and output) are required");
        }

        let mut layers = Vec::new();
        for config in layer_configs {
            let layer = Layer::new(
                config.inputs,   // Number of input from previous layer  
                config.neurons,  // Number of neurons in current layer
                config.activation,
            );
            
            layers.push(layer);
        }
        
        Model {
            layers,
        }
    }

    pub fn forward(
        &self,
        input: &Array1<f32>,
    ) -> Array1<f32> {
        let mut current_input = input.clone();
        // TODO why does htis need ot be cloned?
        for layer in self.layers.clone() {
            current_input = layer.forward(&current_input);
        }
        current_input
    }
    
    ///// Train the model using mini-batch gradient descent
    ///// 
    ///// # Arguments
    ///// 
    ///// * `inputs` - Vector of input samples, where each sample is a vector of features
    ///// * `targets` - Vector of target values, where each target corresponds to an input
    ///// * `batch_size` - Size of mini-batches for gradient descent
    ///// * `learning_rate` - Learning rate for weight updates
    ///// 
    ///// # Returns
    ///// 
    ///// Average loss across all samples in the batch
    //pub fn train_batch(
    //    &mut self,
    //    inputs: &[Vec<f64>],
    //    targets: &[Vec<f64>],
    //    batch_size: usize,
    //    learning_rate: f64
    //) -> f64 {
    //    assert_eq!(inputs.len(), targets.len(), "Number of inputs must match number of targets");
    //    assert!(batch_size > 0, "Batch size must be greater than 0");

    //    let mut total_loss = 0.0;
    //    let mut batch_weight_grads = self.layers.iter()
    //        .map(|layer| vec![0.0; layer.weights.len()])
    //        .collect::<Vec<_>>();
    //    let mut batch_bias_grads = self.layers.iter()
    //        .map(|layer| vec![0.0; layer.biases.len()])
    //        .collect::<Vec<_>>();

    //    // Process each sample in the batch
    //    for (input, target) in inputs.iter().zip(targets.iter()).take(batch_size) {
    //        // Forward pass
    //        let activations = self.forward(input.clone());
    //        
    //        // Compute loss
    //        let loss = self.compute_loss(activations.last().unwrap(), target);
    //        total_loss += loss;

    //        // Backward pass
    //        let (weight_grads, bias_grads) = self.backward(activations, target.clone());

    //        // Accumulate gradients
    //        for i in 0..self.layers.len() {
    //            for j in 0..batch_weight_grads[i].len() {
    //                batch_weight_grads[i][j] += weight_grads[i][j];
    //            }
    //            for j in 0..batch_bias_grads[i].len() {
    //                batch_bias_grads[i][j] += bias_grads[i][j];
    //            }
    //        }
    //    }

    //    // Average gradients over batch
    //    let batch_size_f64 = batch_size as f64;
    //    for grads in batch_weight_grads.iter_mut().chain(batch_bias_grads.iter_mut()) {
    //        for grad in grads.iter_mut() {
    //            *grad /= batch_size_f64;
    //        }
    //    }

    //    // Update weights using averaged gradients
    //    self.update_weights(batch_weight_grads, batch_bias_grads, learning_rate);

    //    // Return average loss
    //    total_loss / batch_size_f64
    //}

    ///// Train the model for one iteration using a single data point (kept for backward compatibility)
    //pub fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) -> f64 {
    //    self.train_batch(&[input.clone()], &[target.clone()], 1, learning_rate)
    //}
}
