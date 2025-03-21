pub mod feed_forward;
pub mod conv2d;
pub mod max_pool;
pub mod dropout;

pub use dropout::DropoutLayer;

use std::{any::Any, fmt::Debug};
use ndarray::{Array1, Array2};
use crate::activation::ActivationType;

#[derive(Debug, Clone, Copy)]
pub enum Regularizer {
    L1 { lambda: f32 },
    L2 { lambda: f32 },
}

impl Regularizer {
    pub fn compute_gradient(&self, weights: &Array2<f32>) -> Array2<f32> {
        match *self {
            Regularizer::L1 { lambda } => {
                // L1 gradient is sign(w) * lambda
                let signs = weights.mapv(|w| w.signum());
                signs.mapv(|s| s * lambda)
            },
            Regularizer::L2 { lambda } => {
                // L2 gradient is lambda * w
                weights.mapv(|w| w * lambda)
            }
        }
    }

    pub fn compute_loss(&self, weights: &Array2<f32>) -> f32 {
        match *self {
            Regularizer::L1 { lambda } => {
                // L1 loss is lambda * sum(|w|)
                lambda * weights.iter().map(|w| w.abs()).sum::<f32>()
            },
            Regularizer::L2 { lambda } => {
                // L2 loss is (lambda/2) * sum(w^2)
                (lambda / 2.0) * weights.iter().map(|w| w * w).sum::<f32>()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerParams {
    pub neurons: usize,
    pub inputs: usize,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: ActivationType,
    pub regularizer: Option<Regularizer>,
    pub weight_grads: Array2<f32>,
    pub bias_grads: Array1<f32>,
    pub activation_cache: Array1<f32>,
    pub preactivation_cache: Array1<f32>,
}

#[derive(Debug)]
pub struct GpuLayerParams {
    pub weights_buffer: wgpu::Buffer,
    pub bias_buffer: wgpu::Buffer,
    pub weight_grads_buffer: wgpu::Buffer,
    pub bias_grads_buffer: wgpu::Buffer,
    pub activation_buffer: wgpu::Buffer,
    pub preactivation_buffer: wgpu::Buffer,
    pub padded_input_buffer: Option<wgpu::Buffer>,
    pub conv_params_buffer: Option<wgpu::Buffer>,
    pub indices_buffer: Option<wgpu::Buffer>,
    pub pool_params_buffer: Option<wgpu::Buffer>,
    pub dropout_mask_buffer: Option<wgpu::Buffer>,
    pub dropout_params_buffer: Option<wgpu::Buffer>,
    
    // Forward pass bind groups and layouts
    pub forward_bind_group_1: wgpu::BindGroup,
    pub forward_bind_group_2: wgpu::BindGroup,
    pub forward_bind_group_layout_1: wgpu::BindGroupLayout,
    pub forward_bind_group_layout_2: wgpu::BindGroupLayout,
    
    // Backward pass bind group and layout
    pub backward_bind_group: Option<wgpu::BindGroup>,
    pub backward_bind_group_layout: Option<wgpu::BindGroupLayout>,
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
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn parameter_count(&self) -> usize {
        let params = self.params();
        params.weights.len() + params.bias.len()
    }

    fn regularization_loss(&self) -> f32 {
        match self.params().regularizer {
            Some(reg) => reg.compute_loss(&self.params().weights),
            None => 0.0
        }
    }

    fn apply_regularization_gradients(&mut self) {
        if let Some(reg) = self.params().regularizer {
            let reg_grads = reg.compute_gradient(&self.params().weights);
            self.add_to_weight_grads(reg_grads);
        }
    }

    fn create_gpu_buffers(&self, device: &wgpu::Device) -> GpuLayerParams;
    fn update_gpu_buffers(&self, queue: &wgpu::Queue, params: &GpuLayerParams);
    fn read_gpu_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: &GpuLayerParams);
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub use feed_forward::FeedForwardLayer;
pub use conv2d::Conv2DLayer;
