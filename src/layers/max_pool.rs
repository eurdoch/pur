use std::any::Any;

use ndarray::{Array1, Array2, Array4};
use super::{GpuLayerParams, Layer, LayerParams};
use crate::activation::ActivationType;

#[derive(Debug)]
pub struct MaxPoolLayer {
    pub params: LayerParams,
    pub input_shape: (usize, usize, usize),  // (channels, height, width)
    pub output_shape: (usize, usize, usize), // Added output shape
    pub pool_size: (usize, usize),
    pub stride: usize,
    pub cached_input: Option<Array4<f32>>,
    pub max_indices: Option<Array4<(usize, usize)>>,
}

impl MaxPoolLayer {
    pub fn new(
        in_channels: usize,
        input_height: usize,
        input_width: usize,
        pool_size: (usize, usize),
        stride: usize,
    ) -> Self {
        let input_shape = (in_channels, input_height, input_width);
        
        // Calculate output dimensions
        let output_height = ((input_height - pool_size.0) / stride) + 1;
        let output_width = ((input_width - pool_size.1) / stride) + 1;
        let output_shape = (in_channels, output_height, output_width);
        let total_outputs = in_channels * output_height * output_width;
        
        let params = LayerParams {
            neurons: total_outputs,
            inputs: in_channels * input_height * input_width,
            weights: Array2::zeros((1, 1)),
            bias: Array1::zeros(1),
            activation: ActivationType::Linear,
            regularizer: None,
            weight_grads: Array2::zeros((1, 1)),
            bias_grads: Array1::zeros(1),
            activation_cache: Array1::zeros(total_outputs),
            preactivation_cache: Array1::zeros(total_outputs),
        };

        MaxPoolLayer {
            params,
            input_shape,
            output_shape,  // Store output shape
            pool_size,
            stride,
            cached_input: None,
            max_indices: None,
        }
    }

    fn input_to_4d(&self, input: &Array1<f32>) -> Array4<f32> {
        let (c, h, w) = self.input_shape;
        Array4::from_shape_vec(
            (1, c, h, w),
            input.to_vec()
        ).unwrap()
    }

    fn grad_to_4d(&self, grad: &Array1<f32>) -> Array4<f32> {
        let (c, h, w) = self.output_shape;
        Array4::from_shape_vec(
            (1, c, h, w),
            grad.to_vec()
        ).unwrap()
    }
    
    fn output_to_1d(&self, output: &Array4<f32>) -> Array1<f32> {
        let (_, channels, output_height, output_width) = output.dim();
        let flat_len = channels * output_height * output_width;
        let mut result = Array1::zeros(flat_len);
        
        let mut idx = 0;
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    result[idx] = output[[0, c, h, w]];
                    idx += 1;
                }
            }
        }
        result
    }
}

impl Layer for MaxPoolLayer {
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let input_4d = self.input_to_4d(input);
        self.cached_input = Some(input_4d.clone());
        
        let (_, channels, height, width) = input_4d.dim();
        let output_height = ((height - self.pool_size.0) / self.stride) + 1;
        let output_width = ((width - self.pool_size.1) / self.stride) + 1;
        
        let mut output = Array4::<f32>::zeros((1, channels, output_height, output_width));
        let mut max_indices = Array4::<(usize, usize)>::from_elem(
            (1, channels, output_height, output_width),
            (0, 0)
        );
        
        // Perform max pooling
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    let h_start = h * self.stride;
                    let w_start = w * self.stride;
                    
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_h = 0;
                    let mut max_w = 0;
                    
                    // Find maximum in pooling window
                    for ph in 0..self.pool_size.0 {
                        for pw in 0..self.pool_size.1 {
                            let val = input_4d[[0, c, h_start + ph, w_start + pw]];
                            if val > max_val {
                                max_val = val;
                                max_h = ph;
                                max_w = pw;
                            }
                        }
                    }
                    
                    output[[0, c, h, w]] = max_val;
                    max_indices[[0, c, h, w]] = (max_h, max_w);
                }
            }
        }
        
        self.max_indices = Some(max_indices);
        let output_1d = self.output_to_1d(&output);
        self.params.activation_cache = output_1d.clone();
        output_1d
    }

    fn backward(
        &mut self,
        _input: &Array1<f32>,
        grad_output: &Array1<f32>,
        _prev_layer_cache: Option<&Array1<f32>>,
    ) -> Array1<f32> {
        let input_4d = self.cached_input.as_ref().unwrap();
        let max_indices = self.max_indices.as_ref().unwrap();
        
        let (_, channels, height, width) = input_4d.dim();
        let mut input_gradient = Array4::<f32>::zeros((1, channels, height, width));
        
        // Use grad_to_4d instead of input_to_4d for gradient
        let grad_output_4d = self.grad_to_4d(grad_output);
        let (_, _, output_height, output_width) = grad_output_4d.dim();
        
        // Rest of backward implementation remains the same
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    let h_start = h * self.stride;
                    let w_start = w * self.stride;
                    let (max_h, max_w) = max_indices[[0, c, h, w]];
                    
                    input_gradient[[0, c, h_start + max_h, w_start + max_w]] +=
                        grad_output_4d[[0, c, h, w]];
                }
            }
        }
        
        self.output_to_1d(&input_gradient)
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
        self.params.weight_grads += &grads;
    }

    fn add_to_bias_grads(&mut self, grads: Array1<f32>) {
        self.params.bias_grads += &grads;
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn create_gpu_buffers(&self, device: &wgpu::Device) -> GpuLayerParams {
        let (in_channels, input_height, input_width) = self.input_shape;
        let (_, output_height, output_width) = self.output_shape;

        // Create buffer for input feature maps
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool Input"),
            size: (in_channels * input_height * input_width * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for output feature maps
        let activation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool Output"),
            size: (in_channels * output_height * output_width * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create buffer for max indices (two u32s per element for h and w indices)
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool Indices"),
            size: (in_channels * output_height * output_width * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for pooling parameters
        let pool_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool Parameters"),
            size: std::mem::size_of::<PoolParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MaxPool Bind Group Layout"),
            entries: &[
                // input buffer
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
                // output (activation) buffer
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
                // indices buffer
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
                // pooling parameters
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
            label: Some("MaxPool Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pool_params_buffer.as_entire_binding(),
                },
            ],
        });

        GpuLayerParams {
            weights_buffer: input_buffer,      // Reuse weights_buffer field for input
            bias_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaxPool Empty Bias"),
                size: 4,  // Minimum size buffer
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            weight_grads_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaxPool Empty Weight Grads"),
                size: 4,  // Minimum size buffer
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bias_grads_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaxPool Empty Bias Grads"),
                size: 4,  // Minimum size buffer
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            activation_buffer,
            preactivation_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MaxPool Empty Preactivation"),
                size: 4,  // Minimum size buffer
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bind_group,
            indices_buffer: Some(indices_buffer),
            pool_params_buffer: Some(pool_params_buffer),
            padded_input_buffer: None,
            conv_params_buffer: None,
            dropout_mask_buffer: None,
            dropout_params_buffer: None,
            bind_group_layout,
            backward_bind_group: None,
            backward_bind_group_layout: None,
        }
    }

    fn update_gpu_buffers(&self, queue: &wgpu::Queue, params: &GpuLayerParams) {
        if let Some(input) = &self.cached_input {
            // Write input data to GPU
            queue.write_buffer(
                &params.weights_buffer, // Using weights_buffer for input
                0,
                bytemuck::cast_slice(input.as_slice().unwrap())
            );
        }

        // Write pooling parameters
        let pool_params = PoolParams {
            channels: self.input_shape.0 as u32,
            input_height: self.input_shape.1 as u32,
            input_width: self.input_shape.2 as u32,
            pool_height: self.pool_size.0 as u32,
            pool_width: self.pool_size.1 as u32,
            stride: self.stride as u32,
        };

        queue.write_buffer(
            params.pool_params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&pool_params)
        );
    }

    fn read_gpu_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: &GpuLayerParams) {
        // Create staging buffer for reading output
        let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool Output Staging Buffer"),
            size: (self.params.neurons * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading indices
        let indices_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool Indices Staging Buffer"),
            size: (self.params.neurons * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MaxPool Read Buffer Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &params.activation_buffer,
            0,
            &output_staging_buffer,
            0,
            output_staging_buffer.size()
        );

        encoder.copy_buffer_to_buffer(
            params.indices_buffer.as_ref().unwrap(),
            0,
            &indices_staging_buffer,
            0,
            indices_staging_buffer.size()
        );

        // Submit commands
        queue.submit(Some(encoder.finish()));

        // Map buffers and read data
        let output_slice = output_staging_buffer.slice(..).get_mapped_range();
        let indices_slice = indices_staging_buffer.slice(..).get_mapped_range();

        // Update activation cache
        self.params.activation_cache.as_slice_mut().unwrap().copy_from_slice(
            bytemuck::cast_slice(&output_slice)
        );

        // Update max indices
        let indices: Vec<(usize, usize)> = indices_slice
            .chunks(8) // 2 u32s per index pair
            .map(|chunk| {
                let indices = bytemuck::cast_slice::<u8, u32>(chunk);
                (indices[0] as usize, indices[1] as usize)
            })
            .collect();

        self.max_indices = Some(Array4::from_shape_vec(
            (1, self.output_shape.0, self.output_shape.1, self.output_shape.2),
            indices
        ).unwrap());

        // Clean up
        drop(output_slice);
        drop(indices_slice);
        output_staging_buffer.unmap();
        indices_staging_buffer.unmap();
    }
}

impl Clone for MaxPoolLayer {
    fn clone(&self) -> Self {
        MaxPoolLayer {
            params: self.params.clone(),
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            pool_size: self.pool_size,
            stride: self.stride,
            cached_input: self.cached_input.clone(),
            max_indices: self.max_indices.clone(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PoolParams {
    channels: u32,
    input_height: u32,
    input_width: u32,
    pool_height: u32,
    pool_width: u32,
    stride: u32,
}
