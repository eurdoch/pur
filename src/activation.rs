/// Enum representing different activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    Linear,
}

impl ActivationType {
    /// Applies the activation function to a given input
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Linear => x,
        }
    }

    /// Computes the derivative of the activation function
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            ActivationType::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            },
            ActivationType::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationType::Tanh => 1.0 - x.tanh().powi(2),
            ActivationType::Linear => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::EPSILON;

    #[test]
    fn test_activation_functions() {
        // Sigmoid tests
        assert!((ActivationType::Sigmoid.apply(0.0) - 0.5).abs() < EPSILON);
        
        // ReLU tests
        assert_eq!(ActivationType::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationType::ReLU.apply(2.0), 2.0);
        
        // Tanh tests
        assert!((ActivationType::Tanh.apply(0.0)).abs() < EPSILON);
        
        // Linear tests
        assert_eq!(ActivationType::Linear.apply(5.0), 5.0);
    }

    #[test]
    fn test_activation_derivatives() {
        // Sigmoid derivative
        assert!((ActivationType::Sigmoid.derivative(0.0) - 0.25).abs() < EPSILON);
        
        // ReLU derivative
        assert_eq!(ActivationType::ReLU.derivative(-1.0), 0.0);
        assert_eq!(ActivationType::ReLU.derivative(2.0), 1.0);
        
        // Tanh derivative
        assert!((ActivationType::Tanh.derivative(0.0) - 1.0).abs() < EPSILON);
        
        // Linear derivative
        assert_eq!(ActivationType::Linear.derivative(5.0), 1.0);
    }
}
