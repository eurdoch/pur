use std::any::Any;

use ndarray::{Array1, Array2, Array4};
use super::{Layer, LayerParams};
use crate::activation::ActivationType;

#[derive(Debug)]
pub struct MaxPoolLayer {
    params: LayerParams,
    input_shape: (usize, usize, usize),  // (channels, height, width)
    output_shape: (usize, usize, usize), // Added output shape
    pool_size: (usize, usize),
    stride: usize,
    cached_input: Option<Array4<f32>>,
    max_indices: Option<Array4<(usize, usize)>>,
}

impl MaxPoolLayer {
    pub fn new(
        in_channels: usize,
        input_height: usize,
        input_width: usize,
        pool_size: (usize, usize),
        stride: usize,
    ) -> Self {
        let input_shape = (in_channels, input_height, input_width);
        
        // Calculate output dimensions
        let output_height = ((input_height - pool_size.0) / stride) + 1;
        let output_width = ((input_width - pool_size.1) / stride) + 1;
        let output_shape = (in_channels, output_height, output_width);
        let total_outputs = in_channels * output_height * output_width;
        
        let params = LayerParams {
            neurons: total_outputs,
            inputs: in_channels * input_height * input_width,
            weights: Array2::zeros((1, 1)),
            bias: Array1::zeros(1),
            activation: ActivationType::Linear,
            regularizer: None,
            weight_grads: Array2::zeros((1, 1)),
            bias_grads: Array1::zeros(1),
            activation_cache: Array1::zeros(total_outputs),
            preactivation_cache: Array1::zeros(total_outputs),
        };

        MaxPoolLayer {
            params,
            input_shape,
            output_shape,  // Store output shape
            pool_size,
            stride,
            cached_input: None,
            max_indices: None,
        }
    }

    fn input_to_4d(&self, input: &Array1<f32>) -> Array4<f32> {
        let (c, h, w) = self.input_shape;
        Array4::from_shape_vec(
            (1, c, h, w),
            input.to_vec()
        ).unwrap()
    }

    fn grad_to_4d(&self, grad: &Array1<f32>) -> Array4<f32> {
        let (c, h, w) = self.output_shape;
        Array4::from_shape_vec(
            (1, c, h, w),
            grad.to_vec()
        ).unwrap()
    }
    
    fn output_to_1d(&self, output: &Array4<f32>) -> Array1<f32> {
        let (_, channels, output_height, output_width) = output.dim();
        let flat_len = channels * output_height * output_width;
        let mut result = Array1::zeros(flat_len);
        
        let mut idx = 0;
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    result[idx] = output[[0, c, h, w]];
                    idx += 1;
                }
            }
        }
        result
    }
}

impl Layer for MaxPoolLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let input_4d = self.input_to_4d(input);
        self.cached_input = Some(input_4d.clone());
        
        let (_, channels, height, width) = input_4d.dim();
        let output_height = ((height - self.pool_size.0) / self.stride) + 1;
        let output_width = ((width - self.pool_size.1) / self.stride) + 1;
        
        let mut output = Array4::<f32>::zeros((1, channels, output_height, output_width));
        let mut max_indices = Array4::<(usize, usize)>::from_elem(
            (1, channels, output_height, output_width),
            (0, 0)
        );
        
        // Perform max pooling
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    let h_start = h * self.stride;
                    let w_start = w * self.stride;
                    
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_h = 0;
                    let mut max_w = 0;
                    
                    // Find maximum in pooling window
                    for ph in 0..self.pool_size.0 {
                        for pw in 0..self.pool_size.1 {
                            let val = input_4d[[0, c, h_start + ph, w_start + pw]];
                            if val > max_val {
                                max_val = val;
                                max_h = ph;
                                max_w = pw;
                            }
                        }
                    }
                    
                    output[[0, c, h, w]] = max_val;
                    max_indices[[0, c, h, w]] = (max_h, max_w);
                }
            }
        }
        
        self.max_indices = Some(max_indices);
        let output_1d = self.output_to_1d(&output);
        self.params.activation_cache = output_1d.clone();
        output_1d
    }

    fn backward(
        &mut self,
        _input: &Array1<f32>,
        grad_output: &Array1<f32>,
        _prev_layer_cache: Option<&Array1<f32>>,
    ) -> Array1<f32> {
        let input_4d = self.cached_input.as_ref().unwrap();
        let max_indices = self.max_indices.as_ref().unwrap();
        
        let (_, channels, height, width) = input_4d.dim();
        let mut input_gradient = Array4::<f32>::zeros((1, channels, height, width));
        
        // Use grad_to_4d instead of input_to_4d for gradient
        let grad_output_4d = self.grad_to_4d(grad_output);
        let (_, _, output_height, output_width) = grad_output_4d.dim();
        
        // Rest of backward implementation remains the same
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    let h_start = h * self.stride;
                    let w_start = w * self.stride;
                    let (max_h, max_w) = max_indices[[0, c, h, w]];
                    
                    input_gradient[[0, c, h_start + max_h, w_start + max_w]] +=
                        grad_output_4d[[0, c, h, w]];
                }
            }
        }
        
        self.output_to_1d(&input_gradient)
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
        self.params.weight_grads += &grads;
    }

    fn add_to_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads += &grads;
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Clone for MaxPoolLayer {
    fn clone(&self) -> Self {
        MaxPoolLayer {
            params: self.params.clone(),
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            pool_size: self.pool_size,
            stride: self.stride,
            cached_input: self.cached_input.clone(),
            max_indices: self.max_indices.clone(),
        }
    }
}
