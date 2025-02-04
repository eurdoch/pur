#[derive(Debug, Clone)]
pub struct Optimizer {
    pub learning_rate: f32,
}

impl Optimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

