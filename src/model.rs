use ndarray::Array1;
use crate::activation::ActivationType;
use crate::optimizer::Optimizer;
use crate::Loss;
use crate::layers::{Conv2DLayer, FeedForwardLayer, Layer};

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
    }
}

pub struct LayerConfig {
    pub layer_type: LayerType,
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
                    // For cross-entropy loss with softmax, the gradient is (output - target)
                    output - target
                }
            };
            
            // Backward pass
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
            }
        }
        
        // Average gradients over batch
        let batch_size = batch_size as f32;
        for layer in &mut self.layers {
            layer.set_weight_grads(&layer.params().weight_grads / batch_size);
            layer.set_bias_grads(&layer.params().bias_grads / batch_size);
        }
        
        self.update_parameters();
        total_loss / batch_size
    }

    // TODO optimize with ndarray views
    pub fn train(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        epochs: usize,
        batch_size: usize,
    ) {
        let combined_data: Vec<_> = inputs.iter().cloned().zip(targets.iter().cloned()).collect();

        for i in 0..epochs {
            println!("Epoch {}", i);
            let chunks = combined_data.chunks(batch_size);
            let chunks_length = chunks.len();
            for (chunk_idx, chunk) in chunks.enumerate() {
                let (inputs, labels): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
                let images_view = inputs.iter().map(|v| Array1::from_vec(v.to_vec())).collect::<Vec<_>>();
                let labels_view = labels.iter().map(|v| Array1::from_vec(v.to_vec())).collect::<Vec<_>>();

                let loss = self.train_batch(images_view, labels_view, batch_size);
                if chunk_idx % 100 == 0 {
                    println!("Batch {} / {}, Loss: {}", chunk_idx, chunks_length, loss);
                }
            }
        }
    }

    pub fn update_parameters(
        &mut self,
    ) {
        for layer in &mut self.layers {
            layer.params_mut().weights = &layer.params().weights - self.optimizer.learning_rate * &layer.params().weight_grads;
            layer.params_mut().bias = &layer.params().bias - self.optimizer.learning_rate * &layer.params().bias_grads;
        }
    }

    pub fn zero_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.params_mut().weight_grads.fill(0.0);
            layer.params_mut().bias_grads.fill(0.0);
        }
    }
}


