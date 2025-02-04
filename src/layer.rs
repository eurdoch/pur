use crate::activation::ActivationType;
use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};

#[derive(Debug, Clone)]
pub struct Layer {
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

impl Layer {
    /// Constructs a new layer with specified configuration
    ///
    /// # Arguments
    ///
    /// * `inputs` - Number of inputs to this layer
    /// * `neurons` - Number of neurons in this layer
    /// * `activation` - Activation function type for the layer
    pub fn new(
        inputs: usize, 
        neurons: usize, 
        activation: ActivationType, 
    ) -> Self {
        // Assume He normalization
        let std_dev = (2.0 / inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        let weights: Array2<f32> = Array2::from_shape_fn((inputs, neurons), |_| normal_dist.sample(&mut rand::rng()));
        let bias: Array1<f32> = Array1::zeros(neurons);
        let weight_grads: Array2<f32> = Array2::zeros((inputs, neurons));
        let bias_grads: Array1<f32> = Array1::zeros(neurons);
        let activation_cache: Array1<f32> = Array1::zeros(neurons);
        let preactivation_cache: Array1<f32> = Array1::zeros(neurons);

        Layer {
            neurons,
            inputs,
            weights,
            bias,
            activation,
            weight_grads,
            bias_grads,
            activation_cache,
            preactivation_cache,
        }
    }
    
    /// Forward propagation through the layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector to the layer
    ///
    /// # Returns
    ///
    /// Transformed output vector after applying weights, biases, and activation
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.inputs, "Input size does not match layer's input size");

        let output = input.dot(&self.weights) + &self.bias;
        self.preactivation_cache = output.clone();
        let activated_output = self.activation.forward(output);
        self.activation_cache = activated_output.clone();
        activated_output
    }
}
