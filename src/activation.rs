/// Enum representing different activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
}

impl ActivationType {
    /// Computes the derivative of the activation function
    ///
    /// Note: For Softmax, this derivative is context-dependent 
    /// and is typically computed during backpropagation in the Layer struct.
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            ActivationType::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            },
            ActivationType::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationType::Tanh => 1.0 - x.tanh().powi(2),
            ActivationType::Softmax => {
                // For individual softmax neurons, the derivative 
                // depends on the full vector output and is context-dependent
                // This is a placeholder implementation
                x.exp()
            }
        }
    }
}
