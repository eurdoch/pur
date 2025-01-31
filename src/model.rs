use rand::random;
use crate::layer::Layer;
use crate::activation::ActivationType;
use crate::hyperparameters::ModelHyperparameters;

/// Represents a neural network model
#[derive(Debug, Clone)]
pub struct Model {
    /// Layers of the neural network
    pub layers: Vec<Layer>,
    
    /// Hyperparameters for the model
    pub hyperparameters: ModelHyperparameters,
}

impl Model {
    /// Create a new neural network model
    pub fn new(layer_sizes: Vec<usize>, activation: ActivationType) -> Self {
        let layers = layer_sizes.windows(2)
            .map(|window| Layer {
                neurons: window[1],
                weights: vec![0.0; window[0] * window[1]], // Weights between previous and current layer
                biases: vec![0.0; window[1]],
                activation,
            })
            .collect();
        
        Model {
            layers,
            hyperparameters: ModelHyperparameters::default(),
        }
    }
    
    /// Initialize weights randomly
    pub fn initialize_weights(&mut self) {
        for layer in &mut self.layers {
            layer.weights = layer.weights.iter()
                .map(|_| random::<f32>() * 2.0 - 1.0) // Random weights between -1 and 1
                .collect();
            
            layer.biases = layer.biases.iter()
                .map(|_| random::<f32>() * 2.0 - 1.0) // Random biases between -1 and 1
                .collect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let mut model = Model::new(vec![10, 5, 3], ActivationType::ReLU);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.layers[0].neurons, 5);
        assert_eq!(model.layers[1].neurons, 3);
        
        model.initialize_weights();
        // Additional tests can be added
    }
}
