use std::any::Any;
use bytemuck;

use crate::activation::ActivationType;
use crate::layers::Layer;
use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};

use super::{GpuLayerParams, LayerParams, Regularizer};

#[derive(Debug, Clone)]
pub struct FeedForwardLayer {
    pub params: LayerParams,
}

impl FeedForwardLayer {
    pub fn new(
        inputs: usize, 
        neurons: usize, 
        activation: ActivationType,
        regularizer: Option<Regularizer>,
    ) -> Self {
        // Assume He normalization
        let std_dev = (2.0 / inputs as f32).sqrt();
        let normal_dist = Normal::new(0.0, std_dev).unwrap();

        // Initialize weights as (neurons × inputs) for correct matrix multiplication
        let weights: Array2<f32> = Array2::from_shape_fn((neurons, inputs), |_| normal_dist.sample(&mut rand::rng()));
        let bias: Array1<f32> = Array1::zeros(neurons);
        let weight_grads: Array2<f32> = Array2::zeros((neurons, inputs));
        let bias_grads: Array1<f32> = Array1::zeros(neurons);
        let activation_cache: Array1<f32> = Array1::zeros(neurons);
        let preactivation_cache: Array1<f32> = Array1::zeros(neurons);

        let params = LayerParams {
            neurons,
            inputs,
            weights,
            bias,
            activation,
            regularizer,
            weight_grads,
            bias_grads,
            activation_cache,
            preactivation_cache,
        };

        FeedForwardLayer {
            params
        }
    }
}

impl Layer for FeedForwardLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.params.inputs, "Input size does not match layer's input size");

        // weights is (neurons × inputs), input is (inputs), result is (neurons)
        let output = self.params.weights.dot(input) + &self.params.bias;
        self.params.preactivation_cache = output.clone();
        let activated_output = self.params.activation.forward(output);
        self.params.activation_cache = activated_output.clone();
        activated_output
    }

    fn backward(&mut self, 
        input: &Array1<f32>,
        grad_output: &Array1<f32>,
        prev_layer_cache: Option<&Array1<f32>>
    ) -> Array1<f32> {
        // Get activation derivative with respect to preactivation
        let activation_derivative = self.params.preactivation_cache
            .mapv(|x| self.params.activation.derivative(x));
            
        // Compute gradient with respect to preactivation
        let dlayer = match self.params.activation {
            ActivationType::Softmax => grad_output.clone(), // Special case for softmax
            _ => grad_output * &activation_derivative,
        };
        
        // Add to bias gradients (dlayer is already correctly shaped as (neurons))
        self.add_to_bias_grads(dlayer.clone());
        
        // Compute weight gradients
        // dlayer is (neurons), input is (inputs), result should be (neurons × inputs)
        let activation_input = prev_layer_cache.unwrap_or(input).clone();
        let dlayer_2d = dlayer.clone().insert_axis(ndarray::Axis(1));
        let input_2d = activation_input.insert_axis(ndarray::Axis(0));
        let weight_grads = dlayer_2d.dot(&input_2d);
        self.add_to_weight_grads(weight_grads);
        
        // Compute gradient for previous layer
        // weights is (neurons × inputs), dlayer is (neurons), result should be (inputs)
        self.params.weights.t().dot(&dlayer)
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn params(&self) -> &LayerParams {
        &self.params
    }

    fn params_mut(&mut self) -> &mut LayerParams {
        &mut self.params
    }

    fn set_weight_grads(&mut self, grads: Array2<f32>) {
        self.params.weight_grads = grads;
    }

    fn set_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads = grads;
    }

    fn add_to_weight_grads(&mut self, grads: Array2<f32>) {
        self.params.weight_grads = &self.params.weight_grads + &grads;
    }

    fn add_to_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads = &self.params.bias_grads + &grads;
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn create_gpu_buffers(&self, device: &wgpu::Device) -> GpuLayerParams {
        // Create buffer for weights
        let weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feed Forward Weights"),
            size: (self.params.weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for biases
        let bias_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feed Forward Biases"),
            size: (self.params.bias.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for weight gradients
        let weight_grads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feed Forward Weight Gradients"),
            size: (self.params.weight_grads.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for bias gradients
        let bias_grads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feed Forward Bias Gradients"),
            size: (self.params.bias_grads.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for activations
        let activation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feed Forward Activations"),
            size: (self.params.activation_cache.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for preactivations
        let preactivation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feed Forward Preactivations"),
            size: (self.params.preactivation_cache.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Feed Forward Bind Group Layout"),
            entries: &[
                // weights buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // bias buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // weight gradients buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // bias gradients buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // activation buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // preactivation buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feed Forward Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_grads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias_grads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: preactivation_buffer.as_entire_binding(),
                },
            ],
        });

        GpuLayerParams {
            weights_buffer,
            bias_buffer,
            weight_grads_buffer,
            bias_grads_buffer,
            activation_buffer,
            preactivation_buffer,
            bind_group,
            padded_input_buffer: None,
            conv_params_buffer: None,
            indices_buffer: None,
            pool_params_buffer: None,
            dropout_mask_buffer: None,
            dropout_params_buffer: None,
            bind_group_layout,
        }
    }

    fn update_gpu_buffers(&self, queue: &wgpu::Queue, params: &GpuLayerParams) {
        // Write weights to GPU
        queue.write_buffer(
            &params.weights_buffer,
            0,
            bytemuck::cast_slice(self.params.weights.as_slice().unwrap())
        );

        // Write biases to GPU
        queue.write_buffer(
            &params.bias_buffer,
            0,
            bytemuck::cast_slice(self.params.bias.as_slice().unwrap())
        );

        // Write gradients if needed
        queue.write_buffer(
            &params.weight_grads_buffer,
            0,
            bytemuck::cast_slice(self.params.weight_grads.as_slice().unwrap())
        );

        queue.write_buffer(
            &params.bias_grads_buffer,
            0,
            bytemuck::cast_slice(self.params.bias_grads.as_slice().unwrap())
        );
    }

    fn read_gpu_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: &GpuLayerParams) {
        // Create staging buffer for reading weights
        let weight_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weight Staging Buffer"),
            size: (self.params.weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading biases
        let bias_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bias Staging Buffer"),
            size: (self.params.bias.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy buffers
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Buffer Encoder"),
        });

        // Copy from GPU buffers to staging buffers
        encoder.copy_buffer_to_buffer(
            &params.weights_buffer,
            0,
            &weight_staging_buffer,
            0,
            weight_staging_buffer.size()
        );

        encoder.copy_buffer_to_buffer(
            &params.bias_buffer,
            0,
            &bias_staging_buffer,
            0,
            bias_staging_buffer.size()
        );

        // Submit commands
        queue.submit(Some(encoder.finish()));

        // Map staging buffers and read data
        let weight_slice = weight_staging_buffer.slice(..).get_mapped_range();
        let bias_slice = bias_staging_buffer.slice(..).get_mapped_range();

        // Copy data back to CPU arrays
        self.params.weights.as_slice_mut().unwrap().copy_from_slice(
            bytemuck::cast_slice(&weight_slice)
        );
        self.params.bias.as_slice_mut().unwrap().copy_from_slice(
            bytemuck::cast_slice(&bias_slice)
        );

        // Clean up
        drop(weight_slice);
        drop(bias_slice);
        weight_staging_buffer.unmap();
        bias_staging_buffer.unmap();
    }
}
