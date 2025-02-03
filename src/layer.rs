use crate::activation::ActivationType;
use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};

/// Represents a layer in the neural network
#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: usize,
    pub inputs: usize,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub activation: ActivationType,
}

impl Layer {
    /// Constructs a new layer with specified configuration
    ///
    /// # Arguments
    ///
    /// * `inputs` - Number of inputs to this layer
    /// * `neurons` - Number of neurons in this layer
    /// * `activation` - Activation function type for the layer
    /// * `weight_init` - Weight initialization strategy
    pub fn new(
        inputs: usize, 
        neurons: usize, 
        activation: ActivationType, 
    ) -> Self {
        // Assume He normalization
        let std_dev = (2.0 / inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        // TODO replace deprecated function thread_rng
        let weights: Array2<f32> = Array2::from_shape_fn((784, 128), |_| normal_dist.sample(&mut rand::thread_rng()));
        let bias: Array1<f32> = Array1::zeros(128);

        Layer {
            neurons,
            inputs,
            weights,
            bias,
            activation,
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
    pub fn forward_propagate(&self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.inputs, "Input size does not match layer's input size");

        let output = input.dot(&self.weights) + &self.bias;

        match self.activation {
            ActivationType::Softmax => {
                let sum_of_exponentials = output.mapv(f32::exp).sum();
                output.mapv(|x| x.exp() / sum_of_exponentials)
            },
            ActivationType::ReLU => output.mapv(|x| x.max(0.0)),
            ActivationType::Sigmoid => output.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::Tanh => output.mapv(|x| x.tanh()),
        }
    }
    
    // TODO implement with ndarray
    //pub fn initialize_weights(&mut self, strategy: WeightInitStrategy) {
    //    match strategy {
    //        WeightInitStrategy::Random => {
    //            // Random initialization between -1 and 1
    //            self.weights = (0..self.weights.len())
    //                .map(|_| fastrand::f32() * 2.0 - 1.0)
    //                .collect();
    //            
    //            self.biases = (0..self.biases.len())
    //                .map(|_| fastrand::f32() * 2.0 - 1.0)
    //                .collect();
    //        },
    //        WeightInitStrategy::Xavier => {
    //            // Xavier/Glorot initialization
    //            let scale = (6.0 / (self.inputs as f32 + self.neurons as f32)).sqrt();
    //            
    //            self.weights = (0..self.weights.len())
    //                .map(|_| (fastrand::f32() * 2.0 - 1.0) * scale)
    //                .collect();
    //            
    //            self.biases = vec![0.0; self.neurons]; // Biases typically initialized to zero
    //        },
    //        WeightInitStrategy::HeNormal => {
    //            // He initialization for ReLU networks
    //            let std_dev = (2.0 / self.inputs as f32).sqrt();
    //            
    //            self.weights = (0..self.weights.len())
    //                .map(|_| fastrand::f32() * std_dev)
    //                .collect();
    //            
    //            self.biases = vec![0.0; self.neurons];
    //        },
    //    }
    //}
}

/// Weight initialization strategies
#[derive(Debug, Clone, Copy)]
pub enum WeightInitStrategy {
    /// Uniform random initialization between -1 and 1
    Random,
    
    /// Xavier/Glorot initialization
    Xavier,
    
    /// He initialization (good for ReLU networks)
    HeNormal,
}
