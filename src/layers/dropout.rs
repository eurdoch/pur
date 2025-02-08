use std::any::Any;

use ndarray::{Array1, Array2};
use rand::Rng;
use super::{Layer, LayerParams};
use crate::activation::ActivationType;

#[derive(Debug)]
pub struct DropoutLayer {
    params: LayerParams,
    dropout_rate: f32,
    scale: f32,
    mask: Option<Array1<f32>>,
    is_training: bool,
}

impl DropoutLayer {
    pub fn new(size: usize, dropout_rate: f32) -> Self {
        assert!(dropout_rate >= 0.0 && dropout_rate < 1.0, "Dropout rate must be between 0 and 1");
        
        let params = LayerParams {
            neurons: size,
            inputs: size,
            weights: Array2::zeros((1, 1)),  // Dropout doesn't use weights
            bias: Array1::zeros(1),          // or bias
            activation: ActivationType::Linear,
            regularizer: None,
            weight_grads: Array2::zeros((1, 1)),
            bias_grads: Array1::zeros(1),
            activation_cache: Array1::zeros(size),
            preactivation_cache: Array1::zeros(size),
        };

        DropoutLayer {
            params,
            dropout_rate,
            scale: 1.0 / (1.0 - dropout_rate), // Scale factor for training
            mask: None,
            is_training: true,
        }
    }

    pub fn set_training(&mut self, is_training: bool) {
        self.is_training = is_training;
    }
}

impl Layer for DropoutLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        if self.is_training {
            let mut rng = rand::rng();
            let mask: Array1<f32> = Array1::from_shape_fn(input.len(), |_| {
                if rng.random::<f32>() > self.dropout_rate { self.scale } else { 0.0 }
            });
            
            let output = input * &mask;
            self.mask = Some(mask);
            output
        } else {
            input.clone() // During inference, just pass through
        }
    }

    fn backward(
        &mut self,
        _input: &Array1<f32>,
        grad_output: &Array1<f32>,
        _prev_layer_cache: Option<&Array1<f32>>,
    ) -> Array1<f32> {
        // During backprop, we multiply gradients by the same mask
        if let Some(ref mask) = self.mask {
            grad_output * mask
        } else {
            grad_output.clone()
        }
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

    // Implement remaining Layer trait methods...
    fn set_weight_grads(&mut self, _grads: Array2<f32>) {}
    fn set_bias_grads(&mut self, _grads: Array1<f32>) {}
    fn add_to_weight_grads(&mut self, _grads: Array2<f32>) {}
    fn add_to_bias_grads(&mut self, _grads: Array1<f32>) {}

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Clone for DropoutLayer {
    fn clone(&self) -> Self {
        DropoutLayer {
            params: self.params.clone(),
            dropout_rate: self.dropout_rate,
            scale: self.scale,
            mask: self.mask.clone(),
            is_training: self.is_training,
        }
    }
}
