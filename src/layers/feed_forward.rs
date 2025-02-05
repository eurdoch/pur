use crate::{activation::ActivationType, utils::outer_product};
use crate::layers::Layer;
use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};

use super::LayerParams;

#[derive(Debug, Clone)]
pub struct FeedForwardLayer {
    pub params: LayerParams,
}

impl FeedForwardLayer {
    pub fn new(inputs: usize, neurons: usize, activation: ActivationType) -> Self {
        // Assume He normalization
        let std_dev = (2.0 / inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        let weights: Array2<f32> = Array2::from_shape_fn((inputs, neurons), |_| normal_dist.sample(&mut rand::rng()));
        let bias: Array1<f32> = Array1::zeros(neurons);
        let weight_grads: Array2<f32> = Array2::zeros((inputs, neurons));
        let bias_grads: Array1<f32> = Array1::zeros(neurons);
        let activation_cache: Array1<f32> = Array1::zeros(neurons);
        let preactivation_cache: Array1<f32> = Array1::zeros(neurons);

        let params = LayerParams {
            neurons,
            inputs,
            weights,
            bias,
            activation,
            weight_grads,
            bias_grads,
            activation_cache,
            preactivation_cache,
        };

        FeedForwardLayer {
            params
        }
    }
}

impl Layer for FeedForwardLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.params.inputs, "Input size does not match layer's input size");

        let output = input.dot(&self.params.weights) + &self.params.bias;
        self.params.preactivation_cache = output.clone();
        let activated_output = self.params.activation.forward(output);
        self.params.activation_cache = activated_output.clone();
        activated_output
    }

    fn backward(&mut self, 
        input: &Array1<f32>,
        grad_output: &Array1<f32>,
        prev_layer_cache: Option<&Array1<f32>>
    ) -> Array1<f32> {
        // Get activation derivative with respect to preactivation
        let activation_derivative = self.params.preactivation_cache
            .mapv(|x| self.params.activation.derivative(x));
            
        // Compute gradient with respect to preactivation
        let dlayer = match self.params.activation {
            ActivationType::Softmax => grad_output.clone(), // Special case for softmax
            _ => grad_output * &activation_derivative,
        };
        
        // Add to bias gradients
        self.add_to_bias_grads(dlayer.clone());
        
        // Compute weight gradients
        let activation_input = prev_layer_cache.unwrap_or(input);
        let weight_grads = outer_product(activation_input, &dlayer);
        self.add_to_weight_grads(weight_grads);
        
        // Compute gradient for previous layer
        self.params.weights.t().dot(&dlayer)
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn params(&self) -> &LayerParams {
        &self.params
    }

    fn params_mut(&mut self) -> &mut LayerParams {
        &mut self.params
    }

    fn set_weight_grads(&mut self, grads: Array2<f32>) {
        self.params.weight_grads = grads;
    }

    fn set_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads = grads;
    }

    fn add_to_weight_grads(&mut self, grads: Array2<f32>) {
        self.params.weight_grads = &self.params.weight_grads + grads;
    }

    fn add_to_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads = &self.params.bias_grads + grads;
    }
}
