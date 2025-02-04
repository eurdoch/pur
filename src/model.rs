use ndarray::{Array1, Array2};

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
        for mut layer in self.layers.clone() {
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
        //let loss = -(target * activations.mapv(f32::log10)).sum();
        let doutputs: Array1<f32> = -(target / activations);

        let layer_depth = self.layers.len();
        let mut dlayers: Vec<Array1<f32>> = vec![Array1::zeros(0); layer_depth];

        for i in (0..layer_depth).rev() {
            println!("Current index: {}", &i);
            let layer = &self.layers[i];

            if i == layer_depth-1 {
                let dlayer = &doutputs * &target.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                dlayers[i] = dlayer;
            } else {
                let upper_layer = &self.layers[i+1];
                let dlayer = &upper_layer.weights.dot(&dlayers[i+1]) * 
                    &layer.preactivation_cache.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                dlayers[i] = dlayer.clone();
                self.layers[i].bias_grads = dlayer.clone();
                if i == 0 {
                    let weight_grads: Array2<f32> = outer_product(&dlayer, input);
                    self.layers[i].weight_grads = weight_grads;
                } else {
                    let weight_grads: Array2<f32> = outer_product(&dlayer, &self.layers[i-1].activation_cache);
                    self.layers[i].weight_grads = weight_grads;
                }
            }
        }
    }
}

fn outer_product(dlayer: &Array1<f32>, activation_cache: &Array1<f32>) -> Array2<f32> {
    let a = dlayer.view().into_shape_with_order((dlayer.len(), 1)).unwrap();
    let b = activation_cache.view().into_shape_with_order((1, activation_cache.len())).unwrap();
    
    a.dot(&b)
}
