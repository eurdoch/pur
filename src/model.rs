use ndarray::Array1;
use crate::activation::ActivationType;
use crate::layers::max_pool::MaxPoolLayer;
use crate::optimizer::Optimizer;
use crate::Loss;
use crate::layers::{Conv2DLayer, FeedForwardLayer, Layer, Regularizer};

#[derive(Debug, Clone)]
pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    pub loss: Loss,
    optimizer: Optimizer,
}

pub enum LayerType {
    FeedForward,
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    },
    MaxPool {
        in_channels: usize,
        input_height: usize,
        input_width: usize,
        pool_size: (usize, usize),
        stride: usize,
    }
}

pub struct LayerConfig {
    pub layer_type: LayerType,
    pub neurons: usize,
    pub inputs: usize,
    pub activation: ActivationType,
    pub regularizer: Option<Regularizer>, // Added regularizer to config
}

impl Model {
    /// Create a new neural network model with specified layer configurations
    ///
    /// # Arguments
    ///
    /// * `layer_configs` - A vector of tuples specifying (inputs, neurons, activation)
    pub fn new(
        layer_configs: Vec<LayerConfig>, 
        loss: Loss,
        optimizer: Optimizer,
    ) -> Self {
        if layer_configs.len() < 2 {
            panic!("At least two layers (input and output) are required");
        }

        let mut layers = Vec::new();
        for config in layer_configs {
            match config.layer_type {
                LayerType::FeedForward => {
                    layers.push(Box::new(FeedForwardLayer::new(
                        config.inputs,
                        config.neurons,
                        config.activation,
                        config.regularizer,
                    )) as Box<dyn Layer>);
                },
                LayerType::Conv2D { 
                    in_channels, 
                    out_channels, 
                    kernel_size,
                    stride,
                    padding 
                } => {
                    // Calculate input dimensions
                    let side_length = (config.inputs / in_channels).isqrt();
                    layers.push(Box::new(Conv2DLayer::new(
                        in_channels,
                        out_channels,
                        side_length, // height
                        side_length, // width
                        kernel_size,
                        stride,
                        padding,
                        config.activation,
                        config.regularizer,
                    )) as Box<dyn Layer>);
                },
                LayerType::MaxPool {
                    in_channels,
                    input_height,
                    input_width,
                    pool_size,
                    stride,
                } => {
                    layers.push(Box::new(MaxPoolLayer::new(
                        in_channels,
                        input_height,
                        input_width,
                        pool_size,
                        stride,
                    )) as Box<dyn Layer>);
                }
            }
        }
                    
        Model {
            layers,
            loss,
            optimizer
        }
    }

    /// Returns the total number of trainable parameters in the model
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    // TODO create separate function for inference or pass bool to disable grads
    pub fn forward(
        &mut self,
        input: &Array1<f32>,
    ) -> Array1<f32> {
        let mut current_input = input.clone();
        for layer in &mut self.layers {  // Use &mut reference instead
            current_input = layer.forward(&current_input);
        }
        current_input
    }

    pub fn train_batch(
        &mut self,
        inputs: Vec<Array1<f32>>,
        targets: Vec<Array1<f32>>,
        batch_size: usize,
    ) -> f32 {
        self.zero_gradients();
        
        let mut total_loss: f32 = 0.0;
        
        // Calculate regularization loss from all layers that have regularization enabled
        let regularization_loss: f32 = self.layers
            .iter()
            .map(|layer| layer.regularization_loss())
            .sum();
        
        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass with caching
            let mut layer_inputs = vec![input.clone()];
            let mut current_input = input.clone();
            
            for layer in &mut self.layers {
                current_input = layer.forward(&current_input);
                layer_inputs.push(current_input.clone());
            }
            
            let output = layer_inputs.last().unwrap();
            total_loss += self.loss.calculate(output, target);
            
            // Initial gradient based on loss function
            let mut grad_output = match self.loss {
                Loss::CrossEntropyLoss => {
                    output - target
                }
            };
            
            // Backward pass through each layer
            for i in (0..self.layers.len()).rev() {
                let prev_cache = if i > 0 {
                    Some(&layer_inputs[i])
                } else {
                    None
                };
                
                grad_output = self.layers[i].backward(
                    &layer_inputs[i],
                    &grad_output,
                    prev_cache
                );
                
                // Apply regularization gradients if regularization is enabled for this layer
                self.layers[i].apply_regularization_gradients();
            }
        }
        
        // Average gradients over batch size
        let batch_size = batch_size as f32;
        for layer in &mut self.layers {
            layer.set_weight_grads(&layer.params().weight_grads / batch_size);
            layer.set_bias_grads(&layer.params().bias_grads / batch_size);
        }
        
        self.update_parameters();
        
        // Return average loss including regularization
        (total_loss + regularization_loss) / batch_size
    }

    pub fn train(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        epochs: usize,
        batch_size: usize,
    ) {
        let total_samples = inputs.len();
        
        for epoch in 0..epochs {
            println!("Epoch {}", epoch);
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            
            // Process data in batches
            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_samples);
                let current_batch_size = batch_end - batch_start;
                
                // Extract batch
                let batch_inputs: Vec<Array1<f32>> = inputs[batch_start..batch_end].to_vec();
                let batch_targets: Vec<Array1<f32>> = targets[batch_start..batch_end].to_vec();
                
                // Train on batch
                let batch_loss = self.train_batch(batch_inputs, batch_targets, current_batch_size);
                total_loss += batch_loss;
                batch_count += 1;
                
                // Print progress every 100 batches
                if batch_count % 100 == 0 {
                    println!(
                        "Batch {} / {}, Average Loss: {:.4}", 
                        batch_count, 
                        (total_samples + batch_size - 1) / batch_size,
                        total_loss / batch_count as f32
                    );
                }
            }
            
            println!(
                "Epoch {} complete. Average loss: {:.4}", 
                epoch, 
                total_loss / batch_count as f32
            );
        }
    }

    pub fn update_parameters(&mut self) {
        for layer in &mut self.layers {
            layer.params_mut().weights = &layer.params().weights - 
                self.optimizer.learning_rate * &layer.params().weight_grads;
            layer.params_mut().bias = &layer.params().bias - 
                self.optimizer.learning_rate * &layer.params().bias_grads;
        }
    }

    pub fn zero_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.params_mut().weight_grads.fill(0.0);
            layer.params_mut().bias_grads.fill(0.0);
        }
    }
}
