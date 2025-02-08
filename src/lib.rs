mod model;
mod activation;
mod loss;
mod optimizer;
mod layers;
mod gpu_layers;

pub use model::Model;
pub use model::{LayerConfig, LayerType};
pub use activation::ActivationType;
pub use loss::Loss;
pub use optimizer::Optimizer;
pub use layers::feed_forward;
pub use layers::Regularizer;

// GPU 
pub use gpu_layers::Conv2DGPU;
pub use gpu_layers::ConvParams;
