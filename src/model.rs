use ndarray::{Array1, Array2};

use crate::layer::Layer;
use crate::activation::ActivationType;
use crate::Loss;

#[derive(Debug, Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
    pub loss: Loss,
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
        loss: Loss,
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
            loss
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

    // TODO optimize this
    pub fn backward(
        &mut self,
        input: &Array1<f32>,
        activations: &Array1<f32>,
        target: &Array1<f32>,
    ) {
        let doutputs: Array1<f32> = activations - target;

        let layer_depth = self.layers.len();
        let mut dlayers: Vec<Array1<f32>> = vec![Array1::zeros(0); layer_depth];

        for i in (0..layer_depth).rev() {
            let layer = &self.layers[i];

            if i == layer_depth-1 {
                dlayers[i] = doutputs.clone();
                let op = outer_product(&self.layers[i-1].activation_cache, &doutputs);
                self.layers[i].weight_grads = &self.layers[i].weight_grads + op;
            } else {
                let upper_layer = &self.layers[i+1];
                let dlayer = &upper_layer.weights.dot(&dlayers[i+1]) * 
                    &layer.preactivation_cache.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                dlayers[i] = dlayer.clone();
                self.layers[i].bias_grads = &self.layers[i].bias_grads + dlayer.clone();
                if i == 0 {
                    let weight_grads: Array2<f32> = &self.layers[i].weight_grads + outer_product(&input, &dlayer);
                    self.layers[i].weight_grads = weight_grads;
                } else {
                    let op = outer_product(&self.layers[i-1].activation_cache, &dlayer);
                    let weight_grads: Array2<f32> = &self.layers[i].weight_grads + op;
                    self.layers[i].weight_grads = weight_grads;
                }
            }
        }
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
            let output = self.forward(&input);
            total_loss = total_loss + self.loss.calculate(&output, target);
            self.backward(&input, &output, &target);
        }

        for layer in &mut self.layers {
            layer.weight_grads = &layer.weight_grads / 32.0;
            layer.bias_grads = &layer.bias_grads / 32.0;
        }

        self.update_parameters();
        total_loss / batch_size as f32
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

                let loss = self.train_batch(images_view, labels_view, 32);
                if chunk_idx % 100 == 0 {
                    println!("Batch {} / {}, Loss: {}", chunk_idx, chunks_length, loss);
                }
            }
        }
    }

    pub fn update_parameters(&mut self) {
        for layer in &mut self.layers {
            layer.weights = &layer.weights - 0.001 * &layer.weight_grads;
            layer.bias = &layer.bias - 0.001 * &layer.bias_grads;
        }
    }

    pub fn zero_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.weight_grads.fill(0.0);
            layer.bias_grads.fill(0.0);
        }
    }
}

fn outer_product(dlayer: &Array1<f32>, activation_cache: &Array1<f32>) -> Array2<f32> {
    let a = dlayer.view().into_shape_with_order((dlayer.len(), 1)).unwrap();
    let b = activation_cache.view().into_shape_with_order((1, activation_cache.len())).unwrap();
    
    a.dot(&b)
}
