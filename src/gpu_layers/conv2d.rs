use wgpu::{self, BindGroup, Buffer, ComputePipeline, Device, PipelineCompilationOptions, Queue};

const WORKGROUP_SIZE: u32 = 8;

pub struct Conv2DGPU {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    bind_group: Option<BindGroup>,
    input_buffer: Option<Buffer>,
    weight_buffer: Option<Buffer>,
    bias_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
    params_buffer: Option<Buffer>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvParams {
    pub batch_size: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub input_height: u32,
    pub input_width: u32,
    pub kernel_height: u32,
    pub kernel_width: u32,
    pub stride: u32,
    pub padding: u32,
    pub output_height: u32,
    pub output_width: u32,
}

impl Conv2DGPU {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        
        // Request an adapter for Metal backend
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();

        // Create the device and queue
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

        // Load and create the compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Convolution Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("conv2d_shader.wgsl"))),
        });

        // Create the compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Convolution Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("conv2d_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Conv2DGPU {
            device,
            queue,
            pipeline,
            bind_group: None,
            input_buffer: None,
            weight_buffer: None,
            bias_buffer: None,
            output_buffer: None,
            params_buffer: None,
        }
    }

    pub fn prepare_buffers(
        &mut self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        params: ConvParams,
    ) {
        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Buffer"),
            size: (input.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create weight buffer
        let weight_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weight Buffer"),
            size: (weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bias buffer
        let bias_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bias Buffer"),
            size: (bias.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer (now with COPY_SRC for staging pattern)
        let output_size = (params.batch_size * params.out_channels * params.output_height * params.output_width) as usize;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer (with COPY_SRC for staging pattern)
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<ConvParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));
        self.queue.write_buffer(&weight_buffer, 0, bytemuck::cast_slice(weights));
        self.queue.write_buffer(&bias_buffer, 0, bytemuck::cast_slice(bias));
        self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Convolution Bind Group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        self.input_buffer = Some(input_buffer);
        self.weight_buffer = Some(weight_buffer);
        self.bias_buffer = Some(bias_buffer);
        self.output_buffer = Some(output_buffer);
        self.params_buffer = Some(params_buffer);
        self.bind_group = Some(bind_group);
    }

    pub async fn compute(&self) -> Vec<f32> {
        // First, let's read the params using a staging buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Params Read Encoder"),
        });

        // Create staging buffer for params
        let params_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Staging Buffer"),
            size: std::mem::size_of::<ConvParams>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy params to staging buffer
        encoder.copy_buffer_to_buffer(
            self.params_buffer.as_ref().unwrap(),
            0,
            &params_staging_buffer,
            0,
            std::mem::size_of::<ConvParams>() as u64,
        );

        // Submit params reading command
        self.queue.submit(Some(encoder.finish()));

        // Read params from staging buffer
        let params_slice = params_staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        params_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();

        let params_data = params_slice.get_mapped_range();
        let params: ConvParams = *bytemuck::from_bytes(&params_data);
        drop(params_data);
        params_staging_buffer.unmap();

        // Create compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Convolution Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Convolution Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);

            compute_pass.dispatch_workgroups(
                (params.output_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                (params.output_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                params.out_channels
            );
        }

        // Create output staging buffer
        let output_size = (params.batch_size * params.out_channels * params.output_height * params.output_width) as usize;
        let output_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            self.output_buffer.as_ref().unwrap(),
            0,
            &output_staging_buffer,
            0,
            (output_size * std::mem::size_of::<f32>()) as u64,
        );

        // Submit all commands
        self.queue.submit(Some(encoder.finish()));

        // Read the results from staging buffer
        let output_slice = output_staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        output_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();

        let data = output_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        output_staging_buffer.unmap();

        result
    }
}
