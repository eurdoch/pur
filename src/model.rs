use std::any::TypeId;

use ndarray::{Array1, Array2};
use crate::activation::ActivationType;
use crate::layers::max_pool::MaxPoolLayer;
use crate::optimizer::Optimizer;
use crate::Loss;
use crate::layers::{Conv2DLayer, DropoutLayer, FeedForwardLayer, GpuLayerParams, Layer, Regularizer};

#[derive(Debug)]
pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
    pub loss: Loss,
    optimizer: Optimizer,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    gpu_layer_params: Option<Vec<GpuLayerParams>>,
    shader_modules: Option<Vec<(wgpu::ShaderModule, wgpu::ShaderModule)>>,
    forward_pipelines: Option<Vec<wgpu::ComputePipeline>>,
    backward_pipelines: Option<Vec<wgpu::ComputePipeline>>,
}

#[derive(Debug)]
pub enum LayerType {
    FeedForward,
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    },
    MaxPool {
        in_channels: usize,
        input_height: usize,
        input_width: usize,
        pool_size: (usize, usize),
        stride: usize,
    },
    Dropout {
        size: usize,
        dropout_rate: f32,
    }
}

#[derive(Debug)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub neurons: Option<usize>,       // Optional: needed for FeedForward
    pub inputs: Option<usize>,        // Optional: needed for FeedForward
    pub activation: Option<ActivationType>, // Optional: needed for FeedForward and Conv2D
    pub regularizer: Option<Regularizer>,  // Optional: for regularization
}

impl Model {
    /// Create a new neural network model with specified layer configurations
    ///
    /// # Arguments
    ///
    /// * `layer_configs` - A vector of tuples specifying (inputs, neurons, activation)
    pub async fn new(
        layer_configs: Vec<LayerConfig>, 
        loss: Loss,
        optimizer: Optimizer,
        mode: &str,
    ) -> Self {
        if layer_configs.len() < 2 {
            panic!("At least two layers (input and output) are required");
        }

        let (device, queue) = match mode {
            "gpu" => {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::METAL,
                    ..Default::default()
                });
                
                // Request an adapter for Metal backend
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    })
                    .await
                    .unwrap();

                let (device, queue) = adapter
                    .request_device(
                        &wgpu::DeviceDescriptor {
                            label: Some("Convolution Device"),
                            required_features: wgpu::Features::empty(),
                            required_limits: wgpu::Limits::downlevel_defaults(),
                            memory_hints: Default::default(),
                        },
                        None,
                    )
                    .await
                    .unwrap();

                (Some(device), Some(queue))
            },
            _ => (None, None)
        };

        let mut layers = Vec::new();
        for config in layer_configs {
            match config.layer_type {
                LayerType::FeedForward => {
                    let neurons = config.neurons.expect("FeedForward layer requires neurons");
                    let inputs = config.inputs.expect("FeedForward layer requires inputs");
                    let activation = config.activation.expect("FeedForward layer requires activation");
                    
                    layers.push(Box::new(FeedForwardLayer::new(
                        inputs,
                        neurons,
                        activation,
                        config.regularizer,
                    )) as Box<dyn Layer>);
                },
                LayerType::Conv2D { 
                    in_channels, 
                    out_channels, 
                    kernel_size,
                    stride,
                    padding 
                } => {
                    let inputs = config.inputs.expect("Conv2D layer requires inputs");
                    let activation = config.activation.expect("Conv2D layer requires activation");
                    
                    // Calculate input dimensions
                    let side_length = (inputs / in_channels).isqrt();
                    layers.push(Box::new(Conv2DLayer::new(
                        in_channels,
                        out_channels,
                        side_length, // height
                        side_length, // width
                        kernel_size,
                        stride,
                        padding,
                        activation,
                        config.regularizer,
                    )) as Box<dyn Layer>);
                },
                LayerType::MaxPool {
                    in_channels,
                    input_height,
                    input_width,
                    pool_size,
                    stride,
                } => {
                    layers.push(Box::new(MaxPoolLayer::new(
                        in_channels,
                        input_height,
                        input_width,
                        pool_size,
                        stride,
                    )) as Box<dyn Layer>);
                },
                LayerType::Dropout {
                    size,
                    dropout_rate,
                } => {
                    layers.push(Box::new(DropoutLayer::new(
                        size,
                        dropout_rate,
                    )) as Box<dyn Layer>);
                }
            }
        }

        let mut model = Model {
            layers,
            loss,
            optimizer,
            device,
            queue,
            gpu_layer_params: None,
            shader_modules: None,
            forward_pipelines: None,
            backward_pipelines: None,
        };

        if mode == "gpu" {
            model.init_gpu_resources();
        }

        model
    }

    /// Returns the total number of trainable parameters in the model
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    pub fn set_training(&mut self, is_training: bool) {
        for layer in &mut self.layers {
            if let Some(dropout) = layer.as_any_mut().downcast_mut::<DropoutLayer>() {
                dropout.set_training(is_training);
            }
        }
    }

    // TODO create separate function for inference or pass bool to disable grads
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut current_input = input.clone();
        for layer in &mut self.layers {
            current_input = layer.forward(&current_input);
        }
        current_input
    }

    pub async fn train_batch(
        &mut self,
        inputs: Vec<Array1<f32>>,
        targets: Vec<Array1<f32>>,
        batch_size: usize,
    ) -> f32 {
        self.zero_gradients();
        let mut total_loss: f32 = 0.0;
        match (&self.device, &self.queue) {
            (Some(device), Some(queue)) => {
                // Calculate regularization loss
                let regularization_loss: f32 = self.layers
                    .iter()
                    .map(|layer| layer.regularization_loss())
                    .sum();

                for (input, target) in inputs.iter().zip(targets.iter()) {
                    // Forward pass
                    let mut current_input = input.clone();
                    let mut layer_inputs = vec![current_input.clone()];

                    // Forward pass through each layer
                    for i in 0..self.layers.len() {
                        let params = &self.gpu_layer_params.as_ref().unwrap()[i];
                        let pipeline = &self.forward_pipelines.as_ref().unwrap()[i];

                        // Write input to GPU
                        queue.write_buffer(
                            &params.weights_buffer,
                            0,
                            bytemuck::cast_slice(current_input.as_slice().unwrap())
                        );

                        // Create command encoder
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Forward Pass Encoder"),
                        });

                        // Create compute pass
                        {
                            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                            compute_pass.set_pipeline(pipeline);
                            
                            // Use appropriate bind group based on layer type
                            if self.layers[i].as_any_mut().type_id() == TypeId::of::<Conv2DLayer>() {
                                compute_pass.set_bind_group(0, &params.bind_group, &[]); // Forward bind group
                            } else {
                                compute_pass.set_bind_group(0, &params.bind_group, &[]); // Regular bind group
                            }

                            // Calculate dispatch size based on layer type
                            let workgroup_count = match self.layers[i].as_any_mut().type_id() {
                                t if t == TypeId::of::<Conv2DLayer>() => {
                                    let out_size = params.activation_buffer.size() as u32 / 4;
                                    ((out_size as f32).cbrt().ceil() as u32 + 7) / 8
                                },
                                _ => (current_input.len() as u32 + 255) / 256
                            };

                            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                        }

                        // Submit command buffer
                        queue.submit(Some(encoder.finish()));

                        // Read results back to CPU
                        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Staging Buffer"),
                            size: params.activation_buffer.size(),
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });

                        // Copy from output buffer to staging buffer
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Copy to Staging Encoder"),
                        });
                        encoder.copy_buffer_to_buffer(
                            &params.activation_buffer,
                            0,
                            &staging_buffer,
                            0,
                            params.activation_buffer.size(),
                        );
                        queue.submit(Some(encoder.finish()));

                        // Read data back
                        let slice = staging_buffer.slice(..);
                        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                        slice.map_async(wgpu::MapMode::Read, move |result| {
                            sender.send(result).unwrap();
                        });
                        device.poll(wgpu::Maintain::Wait);
                        receiver.receive().await.unwrap().unwrap();

                        let data = slice.get_mapped_range();
                        current_input = Array1::from_vec(bytemuck::cast_slice(&data).to_vec());
                        layer_inputs.push(current_input.clone());
                        drop(data);
                        staging_buffer.unmap();
                    }

                    // Calculate loss
                    let output = layer_inputs.last().unwrap();
                    total_loss += self.loss.calculate(output, target);

                    // Initial gradient based on loss function
                    let mut grad_output = match self.loss {
                        Loss::CrossEntropyLoss => output - target
                    };

                    // Backward pass through each layer in reverse
                    for i in (0..self.layers.len()).rev() {
                        let params = &self.gpu_layer_params.as_ref().unwrap()[i];
                        let pipeline = &self.backward_pipelines.as_ref().unwrap()[i];

                        // Write gradients to GPU
                        queue.write_buffer(
                            &params.activation_buffer,
                            0,
                            bytemuck::cast_slice(grad_output.as_slice().unwrap())
                        );

                        // Create command encoder
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Backward Pass Encoder"),
                        });

                        // Create compute pass
                        {
                            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                            compute_pass.set_pipeline(pipeline);
                            
                            // Use appropriate bind group based on layer type
                            if self.layers[i].as_any_mut().type_id() == TypeId::of::<Conv2DLayer>() {
                                // Use backward bind group for Conv2D layers
                                compute_pass.set_bind_group(0, params.backward_bind_group.as_ref().unwrap(), &[]);
                            } else {
                                compute_pass.set_bind_group(0, &params.bind_group, &[]);
                            }

                            // Calculate dispatch size based on layer type
                            let workgroup_count = match self.layers[i].as_any_mut().type_id() {
                                t if t == TypeId::of::<Conv2DLayer>() => {
                                    // For Conv2D, we need three separate dispatches
                                    compute_pass.dispatch_workgroups(
                                        (params.weight_grads_buffer.size() as u32 + 63) / 64,
                                        1,
                                        1
                                    ); // for weights
                                    compute_pass.dispatch_workgroups(
                                        (params.bias_grads_buffer.size() as u32 + 63) / 64,
                                        1,
                                        1
                                    ); // for biases
                                    (params.weights_buffer.size() as u32 + 255) / 256 // for input gradients
                                },
                                t if t == TypeId::of::<MaxPoolLayer>() => {
                                    let (channels, height, width) = self.layers[i]
                                        .as_any_mut()
                                        .downcast_ref::<MaxPoolLayer>()
                                        .unwrap()
                                        .input_shape;
                                    ((height * width * channels) as u32 + 63) / 64
                                },
                                _ => (grad_output.len() as u32 + 255) / 256
                            };

                            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                        }

                        // Submit command buffer
                        queue.submit(Some(encoder.finish()));

                        // Read gradients back
                        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Backward Staging Buffer"),
                            size: params.weights_buffer.size(),
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });

                        // Copy from output buffer to staging buffer
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Copy Gradients to Staging Encoder"),
                        });
                        encoder.copy_buffer_to_buffer(
                            &params.weights_buffer,
                            0,
                            &staging_buffer,
                            0,
                            params.weights_buffer.size(),
                        );
                        queue.submit(Some(encoder.finish()));

                        // Read gradient data back
                        let slice = staging_buffer.slice(..);
                        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                        slice.map_async(wgpu::MapMode::Read, move |result| {
                            sender.send(result).unwrap();
                        });
                        device.poll(wgpu::Maintain::Wait);
                        receiver.receive().await.unwrap().unwrap();

                        let data = slice.get_mapped_range();
                        grad_output = Array1::from_vec(bytemuck::cast_slice(&data).to_vec());
                        drop(data);
                        staging_buffer.unmap();

                        // Read weight and bias gradients if layer has them
                        if !matches!(self.layers[i].as_any_mut().type_id(),
                            t if t == TypeId::of::<MaxPoolLayer>() ||
                                t == TypeId::of::<DropoutLayer>())
                        {
                            // Read weight gradients
                            let weight_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Weight Gradients Staging Buffer"),
                                size: params.weight_grads_buffer.size(),
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });

                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Copy Weight Gradients Encoder"),
                            });
                            encoder.copy_buffer_to_buffer(
                                &params.weight_grads_buffer,
                                0,
                                &weight_staging_buffer,
                                0,
                                params.weight_grads_buffer.size(),
                            );
                            queue.submit(Some(encoder.finish()));

                            // Read weight gradients back
                            let slice = weight_staging_buffer.slice(..);
                            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                            slice.map_async(wgpu::MapMode::Read, move |result| {
                                sender.send(result).unwrap();
                            });
                            device.poll(wgpu::Maintain::Wait);
                            receiver.receive().await.unwrap().unwrap();

                            let data = slice.get_mapped_range();
                            let weight_grads = Array2::from_shape_vec(
                                self.layers[i].params().weight_grads.dim(),
                                bytemuck::cast_slice(&data).to_vec()
                            ).unwrap();
                            self.layers[i].set_weight_grads(weight_grads);
                            drop(data);
                            weight_staging_buffer.unmap();

                            // Read bias gradients
                            let bias_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Bias Gradients Staging Buffer"),
                                size: params.bias_grads_buffer.size(),
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });

                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Copy Bias Gradients Encoder"),
                            });
                            encoder.copy_buffer_to_buffer(
                                &params.bias_grads_buffer,
                                0,
                                &bias_staging_buffer,
                                0,
                                params.bias_grads_buffer.size(),
                            );
                            queue.submit(Some(encoder.finish()));

                            // Read bias gradients back
                            let slice = bias_staging_buffer.slice(..);
                            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                            slice.map_async(wgpu::MapMode::Read, move |result| {
                                sender.send(result).unwrap();
                            });
                            device.poll(wgpu::Maintain::Wait);
                            receiver.receive().await.unwrap().unwrap();

                            let data = slice.get_mapped_range();
                            let bias_grads = Array1::from_vec(
                                bytemuck::cast_slice(&data).to_vec()
                            );
                            self.layers[i].set_bias_grads(bias_grads);
                            drop(data);
                            bias_staging_buffer.unmap();
                        }

                        // Apply regularization gradients if enabled
                        self.layers[i].apply_regularization_gradients();
                    }
                }

                // Average gradients over batch size
                for layer in &mut self.layers {
                    layer.set_weight_grads(&layer.params().weight_grads / batch_size as f32);
                    layer.set_bias_grads(&layer.params().bias_grads / batch_size as f32);
                }

                self.update_parameters();
                (total_loss + regularization_loss) / batch_size as f32
            },
            _ => {
                // Calculate regularization loss from all layers that have regularization enabled
                let regularization_loss: f32 = self.layers
                    .iter()
                    .map(|layer| layer.regularization_loss())
                    .sum();
                
                for (input, target) in inputs.iter().zip(targets.iter()) {
                    // Forward pass with caching
                    let mut layer_inputs = vec![input.clone()];
                    let mut current_input = input.clone();
                    
                    for layer in &mut self.layers {
                        current_input = layer.forward(&current_input);
                        layer_inputs.push(current_input.clone());
                    }
                    
                    let output = layer_inputs.last().unwrap();
                    total_loss += self.loss.calculate(output, target);
                    
                    // Initial gradient based on loss function
                    let mut grad_output = match self.loss {
                        Loss::CrossEntropyLoss => {
                            output - target
                        }
                    };
                    
                    // Backward pass through each layer
                    for i in (0..self.layers.len()).rev() {
                        let prev_cache = if i > 0 {
                            Some(&layer_inputs[i])
                        } else {
                            None
                        };
                        
                        grad_output = self.layers[i].backward(
                            &layer_inputs[i],
                            &grad_output,
                            prev_cache
                        );
                        
                        // Apply regularization gradients if regularization is enabled for this layer
                        self.layers[i].apply_regularization_gradients();
                    }
                }
                
                // Average gradients over batch size
                let batch_size = batch_size as f32;
                for layer in &mut self.layers {
                    layer.set_weight_grads(&layer.params().weight_grads / batch_size);
                    layer.set_bias_grads(&layer.params().bias_grads / batch_size);
                }
                
                self.update_parameters();
                
                // Return average loss including regularization
                (total_loss + regularization_loss) / batch_size
            }
        }
    }

    pub async fn train(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        epochs: usize,
        batch_size: usize,
    ) {
        let total_samples = inputs.len();
        
        for epoch in 0..epochs {
            println!("Epoch {}", epoch);
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            
            // Process data in batches
            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_samples);
                let current_batch_size = batch_end - batch_start;
                
                // Extract batch
                let batch_inputs: Vec<Array1<f32>> = inputs[batch_start..batch_end].to_vec();
                let batch_targets: Vec<Array1<f32>> = targets[batch_start..batch_end].to_vec();
                
                // Train on batch
                let batch_loss = self.train_batch(batch_inputs, batch_targets, current_batch_size).await;
                total_loss += batch_loss;
                batch_count += 1;
                
                // Print progress every 100 batches
                if batch_count % 100 == 0 {
                    println!(
                        "Batch {} / {}, Average Loss: {:.4}", 
                        batch_count, 
                        (total_samples + batch_size - 1) / batch_size,
                        total_loss / batch_count as f32
                    );
                }
            }
            
            println!(
                "Epoch {} complete. Average loss: {:.4}", 
                epoch, 
                total_loss / batch_count as f32
            );
        }
    }

    pub fn update_parameters(&mut self) {
        for layer in &mut self.layers {
            layer.params_mut().weights = &layer.params().weights - 
                self.optimizer.learning_rate * &layer.params().weight_grads;
            layer.params_mut().bias = &layer.params().bias - 
                self.optimizer.learning_rate * &layer.params().bias_grads;
        }
    }

    pub fn zero_gradients(&mut self) {
        if let (Some(_), Some(queue)) = (&self.device, &self.queue) {
            // Zero GPU gradient buffers
            if let Some(gpu_params) = &self.gpu_layer_params {
                for params in gpu_params {
                    // Create zero data
                    let weight_zeros = vec![0.0f32; params.weight_grads_buffer.size() as usize / 4];
                    let bias_zeros = vec![0.0f32; params.bias_grads_buffer.size() as usize / 4];

                    // Write zeros to GPU buffers
                    queue.write_buffer(
                        &params.weight_grads_buffer,
                        0,
                        bytemuck::cast_slice(&weight_zeros)
                    );
                    queue.write_buffer(
                        &params.bias_grads_buffer,
                        0,
                        bytemuck::cast_slice(&bias_zeros)
                    );
                }
            }
        } else {
            for layer in &mut self.layers {
                layer.params_mut().weight_grads.fill(0.0);
                layer.params_mut().bias_grads.fill(0.0);
            }
        }
    }

    fn init_gpu_resources(&mut self) {
        let mut gpu_params = Vec::new();
        let mut shader_modules = Vec::new();
        let mut forward_pipelines = Vec::new();
        let mut backward_pipelines = Vec::new();

        for layer in &mut self.layers {
            // Create GPU buffers
            let params = layer.create_gpu_buffers(self.device.as_ref().unwrap());
            
            // Create shader modules and pipelines based on layer type
            if let Some(_) = layer.as_any_mut().downcast_ref::<FeedForwardLayer>() {
                let forward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Feed Forward Forward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/feed_forward_forward.wgsl").into()),
                    });
                
                let backward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Feed Forward Backward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/feed_forward_backward.wgsl").into()),
                    });

                // Create pipeline layout
                let pipeline_layout = self.device.as_ref().unwrap()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Feed Forward Pipeline Layout"),
                        bind_group_layouts: &[&params.bind_group_layout],  // Use stored layout
                        push_constant_ranges: &[],
                    });

                // Create compute pipelines
                let forward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Feed Forward Forward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &forward_shader,
                        entry_point: Some("forward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                let backward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Feed Forward Backward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &backward_shader,
                        entry_point: Some("backward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                shader_modules.push((forward_shader, backward_shader));
                forward_pipelines.push(forward_pipeline);
                backward_pipelines.push(backward_pipeline);
            } 
            else if let Some(_) = layer.as_any_mut().downcast_ref::<Conv2DLayer>() {
                let forward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Conv2D Forward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/conv2d_forward.wgsl").into()),
                    });
                
                let backward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Conv2D Backward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/conv2d_backward.wgsl").into()),
                    });

                let pipeline_layout = self.device.as_ref().unwrap()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Conv2D Pipeline Layout"),
                        bind_group_layouts: &[&params.bind_group_layout],  // Use stored layout
                        push_constant_ranges: &[],
                    });

                let forward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Conv2D Forward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &forward_shader,
                        entry_point: Some("forward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                let backward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Conv2D Backward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &backward_shader,
                        entry_point: Some("backward_weights"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                shader_modules.push((forward_shader, backward_shader));
                forward_pipelines.push(forward_pipeline);
                backward_pipelines.push(backward_pipeline);
            }
            else if let Some(_) = layer.as_any_mut().downcast_ref::<MaxPoolLayer>() {
                let forward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("MaxPool Forward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/max_pool_forward.wgsl").into()),
                    });
                
                let backward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("MaxPool Backward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/max_pool_backward.wgsl").into()),
                    });

                let pipeline_layout = self.device.as_ref().unwrap()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("MaxPool Pipeline Layout"),
                        bind_group_layouts: &[&params.bind_group_layout],  // Use stored layout
                        push_constant_ranges: &[],
                    });

                let forward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("MaxPool Forward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &forward_shader,
                        entry_point: Some("forward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                let backward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("MaxPool Backward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &backward_shader,
                        entry_point: Some("backward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                shader_modules.push((forward_shader, backward_shader));
                forward_pipelines.push(forward_pipeline);
                backward_pipelines.push(backward_pipeline);
            }
            else if let Some(_) = layer.as_any_mut().downcast_ref::<DropoutLayer>() {
                let forward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Dropout Forward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/dropout_forward.wgsl").into()),
                    });
                
                let backward_shader = self.device.as_ref().unwrap()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Dropout Backward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/dropout_backward.wgsl").into()),
                    });

                let pipeline_layout = self.device.as_ref().unwrap()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Dropout Pipeline Layout"),
                        bind_group_layouts: &[&params.bind_group_layout],  // Use stored layout
                        push_constant_ranges: &[],
                    });

                let forward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Dropout Forward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &forward_shader,
                        entry_point: Some("forward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                let backward_pipeline = self.device.as_ref().unwrap()
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Dropout Backward Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &backward_shader,
                        entry_point: Some("backward"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });

                shader_modules.push((forward_shader, backward_shader));
                forward_pipelines.push(forward_pipeline);
                backward_pipelines.push(backward_pipeline);
            }

            gpu_params.push(params);
        }

        self.gpu_layer_params = Some(gpu_params);
        self.shader_modules = Some(shader_modules);
        self.forward_pipelines = Some(forward_pipelines);
        self.backward_pipelines = Some(backward_pipelines);
    }
}
