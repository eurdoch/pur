use crate::activation::ActivationType;

/// Represents a layer in the neural network
#[derive(Debug, Clone)]
pub struct Layer {
    /// Number of neurons in the layer
    pub neurons: usize,
    
    /// Weights of connections between neurons
    pub weights: Vec<f32>,
    
    /// Bias values for neurons
    pub biases: Vec<f32>,
    
    /// Activation function type for the layer
    pub activation: ActivationType,
}

impl Layer {
    // Methods for layer-specific operations can be added here
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ActivationType;

    #[test]
    fn test_layer_creation() {
        let layer = Layer {
            neurons: 5,
            weights: vec![0.0; 10 * 5],
            biases: vec![0.0; 5],
            activation: ActivationType::ReLU,
        };

        assert_eq!(layer.neurons, 5);
        assert_eq!(layer.weights.len(), 50);
        assert_eq!(layer.biases.len(), 5);
    }
}
