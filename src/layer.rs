use crate::activation::ActivationType;
use rand::random;

/// Represents a layer in the neural network
#[derive(Debug, Clone)]
pub struct Layer {
    /// Number of neurons in the layer
    pub neurons: usize,
    
    /// Number of inputs to this layer
    pub inputs: usize,
    
    /// Weights of connections between neurons
    pub weights: Vec<f32>,
    
    /// Bias values for neurons
    pub biases: Vec<f32>,
    
    /// Activation function type for the layer
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
        weight_init: WeightInitStrategy
    ) -> Self {
        let mut layer = Layer {
            neurons,
            inputs,
            weights: vec![0.0; inputs * neurons],
            biases: vec![0.0; neurons],
            activation,
        };
        
        // Initialize weights based on the specified strategy
        layer.initialize_weights(weight_init);
        
        layer
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
    pub fn forward_propagate(&self, input: &Vec<f64>) -> Vec<f64> {
        // Validate input size
        assert_eq!(input.len(), self.inputs, "Input size does not match layer's input size");

        // Compute output for each neuron
        let mut output = vec![0.0; self.neurons];
        
        for neuron in 0..self.neurons {
            // Compute weighted sum for this neuron
            let mut neuron_output = 0.0;
            
            // Dot product of inputs and weights for this neuron
            for input_idx in 0..self.inputs {
                let weight_idx = neuron * self.inputs + input_idx;
                neuron_output += input[input_idx] * self.weights[weight_idx] as f64;
            }
            
            // Add bias
            neuron_output += self.biases[neuron] as f64;
            
            // Apply activation function
            output[neuron] = match self.activation {
                ActivationType::ReLU => neuron_output.max(0.0),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-neuron_output).exp()),
                ActivationType::Tanh => neuron_output.tanh(),
                ActivationType::Linear => neuron_output, // Linear means no transformation
            };
        }
        
        output
    }
    
    /// Initialize weights using a specific strategy
    pub fn initialize_weights(&mut self, strategy: WeightInitStrategy) {
        match strategy {
            WeightInitStrategy::Random => {
                // Random initialization between -1 and 1
                self.weights = (0..self.weights.len())
                    .map(|_| random::<f32>() * 2.0 - 1.0)
                    .collect();
                
                self.biases = (0..self.biases.len())
                    .map(|_| random::<f32>() * 2.0 - 1.0)
                    .collect();
            },
            WeightInitStrategy::Xavier => {
                // Xavier/Glorot initialization
                let scale = (6.0 / (self.inputs as f32 + self.neurons as f32)).sqrt();
                
                self.weights = (0..self.weights.len())
                    .map(|_| (random::<f32>() * 2.0 - 1.0) * scale)
                    .collect();
                
                self.biases = vec![0.0; self.neurons]; // Biases typically initialized to zero
            },
            WeightInitStrategy::HeNormal => {
                // He initialization for ReLU networks
                let std_dev = (2.0 / self.inputs as f32).sqrt();
                
                self.weights = (0..self.weights.len())
                    .map(|_| random::<f32>() * std_dev)
                    .collect();
                
                self.biases = vec![0.0; self.neurons];
            },
        }
    }
    
    /// Get the total number of parameters in this layer
    pub fn parameter_count(&self) -> usize {
        // Weights + Biases
        self.weights.len() + self.biases.len()
    }
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