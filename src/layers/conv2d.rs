use ndarray::{Array1, Array2, Array4};
use crate::activation::ActivationType;
use super::{Layer, LayerParams};
use rand_distr::{Normal, Distribution};

#[derive(Debug)]
pub struct Conv2DLayer {
    params: LayerParams,
    input_shape: (usize, usize, usize), // (channels, height, width)
    kernel_size: (usize, usize),
    stride: usize,
    padding: usize,
    cached_input: Option<Array4<f32>>,
    cached_padded_input: Option<Array4<f32>>,
}

impl Conv2DLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        input_height: usize,
        input_width: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
        activation: ActivationType,
    ) -> Self {
        let input_shape = (in_channels, input_height, input_width);
        let total_inputs = in_channels * kernel_size.0 * kernel_size.1;
        
        // He initialization for ReLU
        let std_dev = (2.0 / total_inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();
        
        // Initialize weights with He initialization
        let weights = Array2::from_shape_fn(
            (out_channels, total_inputs),
            |_| normal_dist.sample(&mut rand::rng())
        );
        
        let bias = Array1::zeros(out_channels);
        
        let params = LayerParams {
            neurons: out_channels,
            inputs: total_inputs,
            weights,
            bias,
            activation,
            weight_grads: Array2::zeros((out_channels, total_inputs)),
            bias_grads: Array1::zeros(out_channels),
            activation_cache: Array1::zeros(out_channels),
            preactivation_cache: Array1::zeros(out_channels),
        };

        Conv2DLayer {
            params,
            input_shape,
            kernel_size,
            stride,
            padding,
            cached_input: None,
            cached_padded_input: None,
        }
    }

    fn weights_to_4d(&self) -> Array4<f32> {
        let (out_channels, _) = self.params.weights.dim();
        let (in_channels, kh, kw) = (
            self.input_shape.0,
            self.kernel_size.0,
            self.kernel_size.1
        );
        
        let mut weights_4d = Array4::zeros((out_channels, in_channels, kh, kw));
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let flat_idx = ic * (kh * kw) + kh_idx * kw + kw_idx;
                        weights_4d[[oc, ic, kh_idx, kw_idx]] = self.params.weights[[oc, flat_idx]];
                    }
                }
            }
        }
        weights_4d
    }

    fn input_to_4d(&self, input: &Array1<f32>) -> Array4<f32> {
        let (c, h, w) = self.input_shape;
        Array4::from_shape_vec(
            (1, c, h, w),
            input.to_vec()
        ).unwrap()
    }

    fn output_to_1d(&self, output: &Array4<f32>) -> Array1<f32> {
        let (_, out_channels, output_height, output_width) = output.dim();
        let flat_len = out_channels * output_height * output_width;
        let mut result = Array1::zeros(flat_len);
        
        let mut idx = 0;
        for oc in 0..out_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    result[idx] = output[[0, oc, oh, ow]];
                    idx += 1;
                }
            }
        }
        result
    }
}

impl Layer for Conv2DLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // Convert input to 4D
        let input_4d = self.input_to_4d(input);
        self.cached_input = Some(input_4d.clone());
        
        let weights_4d = self.weights_to_4d();
        let (_, _, input_height, input_width) = input_4d.dim();
        let (out_channels, _, kernel_height, kernel_width) = weights_4d.dim();
        
        // Calculate output dimensions
        let output_height = ((input_height + 2 * self.padding - kernel_height) / self.stride) + 1;
        let output_width = ((input_width + 2 * self.padding - kernel_width) / self.stride) + 1;
        
        // Initialize output tensor
        let mut output = Array4::<f32>::zeros((1, out_channels, output_height, output_width));
        
        // Apply padding if needed
        let padded_input = if self.padding > 0 {
            let mut padded = Array4::<f32>::zeros((
                1,
                self.input_shape.0,
                input_height + 2 * self.padding,
                input_width + 2 * self.padding,
            ));
            
            let pad = self.padding as i32;
            padded.slice_mut(ndarray::s![
                ..,
                ..,
                pad..(pad + input_height as i32),
                pad..(pad + input_width as i32)
            ]).assign(&input_4d);
            
            padded
        } else {
            input_4d.clone()
        };
        
        self.cached_padded_input = Some(padded_input.clone());
        
        // Perform convolution
        for oc in 0..out_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let h_start = oh * self.stride;
                    let w_start = ow * self.stride;
                    
                    let mut sum = 0.0;
                    for ic in 0..self.input_shape.0 {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                sum += padded_input[[0, ic, h_start + kh, w_start + kw]] 
                                    * weights_4d[[oc, ic, kh, kw]];
                            }
                        }
                    }
                    
                    output[[0, oc, oh, ow]] = sum + self.params.bias[oc];
                }
            }
        }

        // Store preactivation values
        let output_1d = self.output_to_1d(&output);
        self.params.preactivation_cache = output_1d.clone();
        
        // Apply activation function
        self.params.activation_cache = self.params.activation.forward(output_1d);
        self.params.activation_cache.clone()
    }

    fn backward(
        &mut self,
        _input: &Array1<f32>,
        grad_output: &Array1<f32>,
        _prev_layer_cache: Option<&Array1<f32>>,
    ) -> Array1<f32> {
        let input_4d = self.cached_input.as_ref().unwrap();
        let padded_input = self.cached_padded_input.as_ref().unwrap();
        let weights_4d = self.weights_to_4d();
        
        // Convert grad_output to 4D
        let (_, _, input_height, input_width) = input_4d.dim();
        let (out_channels, in_channels, kernel_height, kernel_width) = weights_4d.dim();
        let output_height = ((input_height + 2 * self.padding - kernel_height) / self.stride) + 1;
        let output_width = ((input_width + 2 * self.padding - kernel_width) / self.stride) + 1;
        let grad_output_4d = Array4::from_shape_vec(
            (1, out_channels, output_height, output_width),
            grad_output.to_vec()
        ).unwrap();

        // Initialize gradients
        let mut input_gradient = Array4::<f32>::zeros(input_4d.dim());
        let mut weight_grads = Array2::zeros(self.params.weights.dim());
        let mut bias_grads = Array1::zeros(self.params.bias.len());
        
        // Calculate gradients
        for oc in 0..out_channels {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let h_start = oh * self.stride;
                    let w_start = ow * self.stride;
                    let output_grad = grad_output_4d[[0, oc, oh, ow]];
                    
                    // Update bias gradients
                    bias_grads[oc] += output_grad;
                    
                    // Update weight gradients
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let input_val = padded_input[[0, ic, h_start + kh, w_start + kw]];
                                let flat_idx = ic * (kernel_height * kernel_width) + kh * kernel_width + kw;
                                weight_grads[[oc, flat_idx]] += input_val * output_grad;
                            }
                        }
                    }
                    
                    // Update input gradients
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let flat_idx = ic * (kernel_height * kernel_width) + kh * kernel_width + kw;
                                let weight = self.params.weights[[oc, flat_idx]];
                                // Handle padding offset with explicit bounds checking
                                let h_idx = (h_start as i32) + (kh as i32) - (self.padding as i32);
                                let w_idx = (w_start as i32) + (kw as i32) - (self.padding as i32);
                                
                                if h_idx >= 0 && w_idx >= 0 && 
                                   (h_idx as usize) < input_4d.dim().2 && 
                                   (w_idx as usize) < input_4d.dim().3 {
                                    input_gradient[[0, ic, h_idx as usize, w_idx as usize]] += weight * output_grad;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        self.params.weight_grads = weight_grads;
        self.params.bias_grads = bias_grads;
        
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
}

impl Clone for Conv2DLayer {
    fn clone(&self) -> Self {
        Conv2DLayer {
            params: self.params.clone(),
            input_shape: self.input_shape,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            cached_input: self.cached_input.clone(),
            cached_padded_input: self.cached_padded_input.clone(),
        }
    }
}
