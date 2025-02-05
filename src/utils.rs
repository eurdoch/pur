use ndarray::{Array1, Array2};

pub fn outer_product(dlayer: &Array1<f32>, activation_cache: &Array1<f32>) -> Array2<f32> {
    let a = dlayer.view().into_shape_with_order((dlayer.len(), 1)).unwrap();
    let b = activation_cache.view().into_shape_with_order((1, activation_cache.len())).unwrap();
    
    a.dot(&b)
}
