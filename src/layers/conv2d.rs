use crate::activation::ActivationType;
use crate::layers::Layer;
use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};

use super::LayerParams;

#[derive(Debug, Clone)]
pub struct Conv2DLayer {
    pub params: LayerParams,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub input_shape: (usize, usize, usize), // (channels, height, width)
    pub filters: usize,
}

impl Conv2DLayer {
    pub fn new(
        input_shape: (usize, usize, usize),
        kernel_size: (usize, usize),
        filters: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        activation: ActivationType
    ) -> Self {
        let (channels, height, width) = input_shape;
        let (kernel_h, kernel_w) = kernel_size;
        
        // Calculate output dimensions
        let output_height = ((height + 2 * padding.0 - kernel_h) / stride.0) + 1;
        let output_width = ((width + 2 * padding.1 - kernel_w) / stride.1) + 1;
        
        // Total number of inputs and outputs for parameter initialization
        let inputs = channels * kernel_h * kernel_w;
        let neurons = filters * output_height * output_width;

        // Initialize weights using He initialization
        let std_dev = (2.0 / inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        // Initialize weights and biases
        let weights: Array2<f32> = Array2::from_shape_fn((inputs, filters), |_| normal_dist.sample(&mut rand::rng()));
        let bias: Array1<f32> = Array1::zeros(filters);
        let weight_grads: Array2<f32> = Array2::zeros((inputs, filters));
        let bias_grads: Array1<f32> = Array1::zeros(filters);
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

        Conv2DLayer {
            params,
            kernel_size,
            stride,
            padding,
            input_shape,
            filters,
        }
    }

    fn im2col(&self, input: &Array1<f32>) -> Array2<f32> {
        let (channels, height, width) = self.input_shape;
        
        // Calculate output dimensions
        let output_h = ((height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0) + 1;
        let output_w = ((width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1) + 1;
        
        // Initialize output matrix
        let cols = Array2::zeros((
            channels * self.kernel_size.0 * self.kernel_size.1,
            output_h * output_w
        ));

        // TODO: Implement proper im2col transformation
        // For now, return a simple reshape that maintains the correct dimensions
        cols
    }
}

impl Layer for Conv2DLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.params.inputs * self.input_shape.0 * self.input_shape.1, 
            "Input size does not match layer's input size");

        // Convert input to columns using im2col
        let input_cols = self.im2col(input);
        
        // Perform convolution as matrix multiplication
        let output_2d = input_cols.dot(&self.params.weights);
        
        // Reshape output to match the expected dimensions
        let mut output = Array1::zeros(self.params.neurons);
        output.assign(&output_2d.to_shape(self.params.neurons).unwrap());
        
        // Add bias to each feature map
        for i in 0..self.filters {
            let start = i * (output.len() / self.filters);
            let end = (i + 1) * (output.len() / self.filters);
            let mut slice = output.slice_mut(ndarray::s![start..end]);
            slice.map_inplace(|x| *x += self.params.bias[i]);
        }
        
        self.params.preactivation_cache = output.clone();
        let activated_output = self.params.activation.forward(output);
        self.params.activation_cache = activated_output.clone();
        activated_output
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
