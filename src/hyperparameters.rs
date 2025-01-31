/// Hyperparameters for the neural network model
#[derive(Debug, Clone)]
pub struct ModelHyperparameters {
    /// Learning rate for training
    pub learning_rate: f32,
    
    /// Number of training epochs
    pub epochs: usize,
    
    /// Batch size for training
    pub batch_size: usize,
    
    /// Regularization parameter
    pub regularization: f32,
}

impl Default for ModelHyperparameters {
    fn default() -> Self {
        ModelHyperparameters {
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
            regularization: 0.001,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_hyperparameters() {
        let hp = ModelHyperparameters::default();
        
        assert_eq!(hp.learning_rate, 0.01);
        assert_eq!(hp.epochs, 100);
        assert_eq!(hp.batch_size, 32);
        assert_eq!(hp.regularization, 0.001);
    }
}