pub mod feed_forward;
pub mod conv2d;

use std::fmt::Debug;
use ndarray::{Array1, Array2};
use crate::activation::ActivationType;

#[derive(Debug, Clone)]
pub struct LayerParams {
    pub neurons: usize,
    pub inputs: usize,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: ActivationType,
    pub weight_grads: Array2<f32>,
    pub bias_grads: Array1<f32>,
    pub activation_cache: Array1<f32>,
    pub preactivation_cache: Array1<f32>,
}

pub trait Layer: Debug {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32>;
    fn backward(&mut self, 
        input: &Array1<f32>,
        grad_output: &Array1<f32>,
        prev_layer_cache: Option<&Array1<f32>>
    ) -> Array1<f32>;

    fn clone_box(&self) -> Box<dyn Layer>;

    fn params(&self) -> &LayerParams;
    fn params_mut(&mut self) -> &mut LayerParams;

    fn set_weight_grads(&mut self, grads: Array2<f32>);
    fn set_bias_grads(&mut self, grads: Array1<f32>);
    fn add_to_weight_grads(&mut self, grads: Array2<f32>);
    fn add_to_bias_grads(&mut self, grads: Array1<f32>);
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub use feed_forward::FeedForwardLayer;
pub use conv2d::Conv2DLayer;
