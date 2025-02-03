use ndarray::Array1;

/// Enum representing different activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    Softmax,
}

impl ActivationType {
    pub fn forward(&self, input: Array1<f32>) -> Array1<f32> {
        match self {
            ActivationType::Softmax => {
                let sum_of_exponentials = input.mapv(f32::exp).sum();
                input.mapv(|x| x.exp() / sum_of_exponentials)
            },
            ActivationType::ReLU => input.mapv(|x| x.max(0.0)),
            ActivationType::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::Tanh => input.mapv(|x| x.tanh()),
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
            ActivationType::Softmax => {
                let softmax_output = self.forward(Array1::from_elem(1, x));
                softmax_output[0] * (1.0 - softmax_output[0])
            }
        }
    }

}

