use std::any::Any;

use ndarray::{Array1, Array2};
use rand::Rng;
use super::{GpuLayerParams, Layer, LayerParams};
use crate::activation::ActivationType;

#[derive(Debug)]
pub struct DropoutLayer {
    params: LayerParams,
    dropout_rate: f32,
    scale: f32,
    mask: Option<Array1<f32>>,
    is_training: bool,
}

impl DropoutLayer {
    pub fn new(size: usize, dropout_rate: f32) -> Self {
        assert!(dropout_rate >= 0.0 && dropout_rate < 1.0, "Dropout rate must be between 0 and 1");
        
        let params = LayerParams {
            neurons: size,
            inputs: size,
            weights: Array2::zeros((1, 1)),  // Dropout doesn't use weights
            bias: Array1::zeros(1),          // or bias
            activation: ActivationType::Linear,
            regularizer: None,
            weight_grads: Array2::zeros((1, 1)),
            bias_grads: Array1::zeros(1),
            activation_cache: Array1::zeros(size),
            preactivation_cache: Array1::zeros(size),
        };

        DropoutLayer {
            params,
            dropout_rate,
            scale: 1.0 / (1.0 - dropout_rate), // Scale factor for training
            mask: None,
            is_training: true,
        }
    }

    pub fn set_training(&mut self, is_training: bool) {
        self.is_training = is_training;
    }
}

impl Layer for DropoutLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        if self.is_training {
            let mut rng = rand::rng();
            let mask: Array1<f32> = Array1::from_shape_fn(input.len(), |_| {
                if rng.random::<f32>() > self.dropout_rate { self.scale } else { 0.0 }
            });
            
            let output = input * &mask;
            self.mask = Some(mask);
            output
        } else {
            input.clone() // During inference, just pass through
        }
    }

    fn backward(
        &mut self,
        _input: &Array1<f32>,
        grad_output: &Array1<f32>,
        _prev_layer_cache: Option<&Array1<f32>>,
    ) -> Array1<f32> {
        // During backprop, we multiply gradients by the same mask
        if let Some(ref mask) = self.mask {
            grad_output * mask
        } else {
            grad_output.clone()
        }
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

    // Implement remaining Layer trait methods...
    fn set_weight_grads(&mut self, _grads: Array2<f32>) {}
    fn set_bias_grads(&mut self, _grads: Array1<f32>) {}
    fn add_to_weight_grads(&mut self, _grads: Array2<f32>) {}
    fn add_to_bias_grads(&mut self, _grads: Array1<f32>) {}

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn create_gpu_buffers(&self, device: &wgpu::Device) -> GpuLayerParams {
        let size = self.params.neurons;

        // Create buffer for input values
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dropout Input"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for output values
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dropout Output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for mask values
        let mask_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dropout Mask"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for dropout parameters
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dropout Parameters"),
            size: std::mem::size_of::<DropoutParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create minimal buffers for unused parameters
        let empty_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dropout Empty Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dropout Bind Group Layout"),
            entries: &[
                // Input buffer
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
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Mask buffer
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
                // Parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dropout Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        GpuLayerParams {
            weights_buffer: input_buffer,
            bias_buffer: empty_buffer.clone(),
            weight_grads_buffer: empty_buffer.clone(),
            bias_grads_buffer: empty_buffer.clone(),
            activation_buffer: output_buffer,
            preactivation_buffer: empty_buffer,
            bind_group,
            dropout_mask_buffer: Some(mask_buffer),
            dropout_params_buffer: Some(params_buffer),
            conv_params_buffer: None,
            indices_buffer: None,
            padded_input_buffer: None,
            pool_params_buffer: None,
            bind_group_layout,
            backward_bind_group: None,
            backward_bind_group_layout: None,
        }
    }

    fn update_gpu_buffers(&self, queue: &wgpu::Queue, params: &GpuLayerParams) {
        // Write dropout parameters
        let dropout_params = DropoutParams {
            dropout_rate: self.dropout_rate,
            scale: self.scale,
            is_training: self.is_training as u32,
            size: self.params.neurons as u32,
        };

        queue.write_buffer(
            params.dropout_params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&dropout_params)
        );

        // Write mask if it exists and we're in training mode
        if let Some(mask) = &self.mask {
            queue.write_buffer(
                params.dropout_mask_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(mask.as_slice().unwrap())
            );
        }
    }

    fn read_gpu_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: &GpuLayerParams) {
        // Create staging buffer for reading mask
        let mask_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dropout Mask Staging Buffer"),
            size: (self.params.neurons * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Dropout Read Buffer Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            params.dropout_mask_buffer.as_ref().unwrap(),
            0,
            &mask_staging_buffer,
            0,
            mask_staging_buffer.size()
        );

        // Submit commands
        queue.submit(Some(encoder.finish()));

        // Map buffer and read data
        let mask_slice = mask_staging_buffer.slice(..).get_mapped_range();

        // Update mask
        let mut mask = Array1::zeros(self.params.neurons);
        mask.as_slice_mut().unwrap().copy_from_slice(
            bytemuck::cast_slice(&mask_slice)
        );
        self.mask = Some(mask);

        // Clean up
        drop(mask_slice);
        mask_staging_buffer.unmap();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DropoutParams {
    dropout_rate: f32,
    scale: f32,
    is_training: u32,
    size: u32,
}

impl Clone for DropoutLayer {
    fn clone(&self) -> Self {
        DropoutLayer {
            params: self.params.clone(),
            dropout_rate: self.dropout_rate,
            scale: self.scale,
            mask: self.mask.clone(),
            is_training: self.is_training,
        }
    }
}
