use crate::activation::ActivationType;
use crate::layers::Layer;
use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};

use super::{LayerParams, Regularizer};

#[derive(Debug, Clone)]
pub struct FeedForwardLayer {
    pub params: LayerParams,
}

impl FeedForwardLayer {
    pub fn new(
        inputs: usize, 
        neurons: usize, 
        activation: ActivationType,
        regularizer: Option<Regularizer>,
    ) -> Self {
        // Assume He normalization
        let std_dev = (2.0 / inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        // Initialize weights as (neurons × inputs) for correct matrix multiplication
        let weights: Array2<f32> = Array2::from_shape_fn((neurons, inputs), |_| normal_dist.sample(&mut rand::rng()));
        let bias: Array1<f32> = Array1::zeros(neurons);
        let weight_grads: Array2<f32> = Array2::zeros((neurons, inputs));
        let bias_grads: Array1<f32> = Array1::zeros(neurons);
        let activation_cache: Array1<f32> = Array1::zeros(neurons);
        let preactivation_cache: Array1<f32> = Array1::zeros(neurons);

        let params = LayerParams {
            neurons,
            inputs,
            weights,
            bias,
            activation,
            regularizer,
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

        // weights is (neurons × inputs), input is (inputs), result is (neurons)
        let output = self.params.weights.dot(input) + &self.params.bias;
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
        
        // Add to bias gradients (dlayer is already correctly shaped as (neurons))
        self.add_to_bias_grads(dlayer.clone());
        
        // Compute weight gradients
        // dlayer is (neurons), input is (inputs), result should be (neurons × inputs)
        let activation_input = prev_layer_cache.unwrap_or(input).clone();
        let dlayer_2d = dlayer.clone().insert_axis(ndarray::Axis(1));
        let input_2d = activation_input.insert_axis(ndarray::Axis(0));
        let weight_grads = dlayer_2d.dot(&input_2d);
        self.add_to_weight_grads(weight_grads);
        
        // Compute gradient for previous layer
        // weights is (neurons × inputs), dlayer is (neurons), result should be (inputs)
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
        self.params.weight_grads = &self.params.weight_grads + &grads;
    }

    fn add_to_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads = &self.params.bias_grads + &grads;
    }
}
