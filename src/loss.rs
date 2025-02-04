use ndarray::Array1;

#[derive(Debug, Clone)]
pub enum Loss {
    CrossEntropyLoss,
}

impl Loss {
    pub fn calculate(&self, prediction: &Array1<f32>, target: &Array1<f32>) -> f32 {
        match self {
            Loss::CrossEntropyLoss => {
                // Cross entropy loss: -Σ(target * log(prediction))
                // Also add small epsilon to avoid log(0)
                let epsilon = 1e-15;
                let safe_pred = prediction.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                -(target * safe_pred.mapv(|x| x.ln())).sum()
            }
        }
    }
}
