use core::mem;

use bevy_ecs::{
    component::Component,
    query::QueryItem,
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_render::{
    camera::ExtractedCamera,
    globals::GlobalsUniform,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{
        binding_types::{
            sampler, texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer,
        },
        AddressMode, BindGroup, BindGroupLayout, BindGroupLayoutEntries, Buffer,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
        FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderDefVal,
        ShaderStages, SpecializedComputePipeline, StorageTextureAccess, TextureDataOrder,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
        TextureView, TextureViewDescriptor,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    texture::CachedTexture,
    view::{ViewUniform, ViewUniformOffset},
};

use crate::ssao::ScreenSpaceAmbientOcclusionQualityLevel;

use super::plugin::{
    PREPROCESS_DEPTH_SHADER_HANDLE, SPATIAL_DENOISE_SHADER_HANDLE, SSAO_SHADER_HANDLE,
};

// https://www.shadertoy.com/view/3tB3z3
const HILBERT_WIDTH: u16 = 64;
fn hilbert_index(mut x: u16, mut y: u16) -> u16 {
    let mut index = 0;

    let mut level: u16 = HILBERT_WIDTH / 2;
    while level > 0 {
        let region_x = (x & level > 0) as u16;
        let region_y = (y & level > 0) as u16;
        index += level * level * ((3 * region_x) ^ region_y);

        if region_y == 0 {
            if region_x == 1 {
                x = HILBERT_WIDTH - 1 - x;
                y = HILBERT_WIDTH - 1 - y;
            }

            mem::swap(&mut x, &mut y);
        }

        level /= 2;
    }

    index
}

#[derive(Resource)]
pub struct SsaoPipelines {
    preprocess_depth_pipeline: CachedComputePipelineId,
    spatial_denoise_pipeline: CachedComputePipelineId,

    pub common_bind_group_layout: BindGroupLayout,
    pub preprocess_depth_bind_group_layout: BindGroupLayout,
    pub ssao_bind_group_layout: BindGroupLayout,
    pub spatial_denoise_bind_group_layout: BindGroupLayout,

    pub hilbert_index_lut: TextureView,
    pub point_clamp_sampler: Sampler,
    pub linear_clamp_sampler: Sampler,
}

impl FromWorld for SsaoPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let hilbert_index_lut = render_device
            .create_texture_with_data(
                render_queue,
                &(TextureDescriptor {
                    label: Some("ssao_hilbert_index_lut"),
                    size: Extent3d {
                        width: HILBERT_WIDTH as u32,
                        height: HILBERT_WIDTH as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R16Uint,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
                TextureDataOrder::default(),
                bytemuck::cast_slice(&generate_hilbert_index_lut()),
            )
            .create_view(&TextureViewDescriptor::default());

        let point_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });
        let linear_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        let common_bind_group_layout = render_device.create_bind_group_layout(
            "ssao_common_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    sampler(SamplerBindingType::NonFiltering),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<ViewUniform>(true),
                ),
            ),
        );

        let preprocess_depth_bind_group_layout = render_device.create_bind_group_layout(
            "ssao_preprocess_depth_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_depth_2d(),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let ssao_bind_group_layout = render_device.create_bind_group_layout(
            "ssao_ssao_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Uint),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                    uniform_buffer::<GlobalsUniform>(false),
                    uniform_buffer::<f32>(false),
                ),
            ),
        );

        let spatial_denoise_bind_group_layout = render_device.create_bind_group_layout(
            "ssao_spatial_denoise_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Uint),
                    texture_storage_2d(TextureFormat::R16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let preprocess_depth_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ssao_preprocess_depth_pipeline".into()),
                layout: vec![
                    preprocess_depth_bind_group_layout.clone(),
                    common_bind_group_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: PREPROCESS_DEPTH_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: "preprocess_depth".into(),
                zero_initialize_workgroup_memory: false,
            });

        let spatial_denoise_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ssao_spatial_denoise_pipeline".into()),
                layout: vec![
                    spatial_denoise_bind_group_layout.clone(),
                    common_bind_group_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: SPATIAL_DENOISE_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: "spatial_denoise".into(),
                zero_initialize_workgroup_memory: false,
            });

        Self {
            preprocess_depth_pipeline,
            spatial_denoise_pipeline,

            common_bind_group_layout,
            preprocess_depth_bind_group_layout,
            ssao_bind_group_layout,
            spatial_denoise_bind_group_layout,

            hilbert_index_lut,
            point_clamp_sampler,
            linear_clamp_sampler,
        }
    }
}

impl SpecializedComputePipeline for SsaoPipelines {
    type Key = SsaoPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let (slice_count, samples_per_slice_side) = key.quality_level.sample_counts();

        let mut shader_defs = vec![
            ShaderDefVal::Int("SLICE_COUNT".to_string(), slice_count as i32),
            ShaderDefVal::Int(
                "SAMPLES_PER_SLICE_SIDE".to_string(),
                samples_per_slice_side as i32,
            ),
        ];

        if key.temporal_jitter {
            shader_defs.push("TEMPORAL_JITTER".into());
        }

        ComputePipelineDescriptor {
            label: Some("ssao_ssao_pipeline".into()),
            layout: vec![
                self.ssao_bind_group_layout.clone(),
                self.common_bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SSAO_SHADER_HANDLE,
            shader_defs,
            entry_point: "ssao".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct SsaoPipelineKey {
    pub quality_level: ScreenSpaceAmbientOcclusionQualityLevel,
    pub temporal_jitter: bool,
}

#[derive(Component)]
pub struct ScreenSpaceAmbientOcclusionResources {
    pub preprocessed_depth_texture: CachedTexture,
    pub ssao_noisy_texture: CachedTexture, // Pre-spatially denoised texture
    pub screen_space_ambient_occlusion_texture: CachedTexture, // Spatially denoised texture
    pub depth_differences_texture: CachedTexture,
    pub thickness_buffer: Buffer,
}

#[derive(Component)]
pub struct SsaoPipelineId(pub CachedComputePipelineId);

#[derive(Component)]
pub struct SsaoBindGroups {
    pub common_bind_group: BindGroup,
    pub preprocess_depth_bind_group: BindGroup,
    pub ssao_bind_group: BindGroup,
    pub spatial_denoise_bind_group: BindGroup,
}

#[derive(Default)]
pub struct SsaoNode {}

impl ViewNode for SsaoNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static SsaoPipelineId,
        &'static SsaoBindGroups,
        &'static ViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, pipeline_id, bind_groups, view_uniform_offset): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<SsaoPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (
            Some(camera_size),
            Some(preprocess_depth_pipeline),
            Some(spatial_denoise_pipeline),
            Some(ssao_pipeline),
        ) = (
            camera.physical_viewport_size,
            pipeline_cache.get_compute_pipeline(pipelines.preprocess_depth_pipeline),
            pipeline_cache.get_compute_pipeline(pipelines.spatial_denoise_pipeline),
            pipeline_cache.get_compute_pipeline(pipeline_id.0),
        )
        else {
            return Ok(());
        };

        render_context.command_encoder().push_debug_group("ssao");

        {
            let mut preprocess_depth_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ssao_preprocess_depth_pass"),
                        timestamp_writes: None,
                    });
            preprocess_depth_pass.set_pipeline(preprocess_depth_pipeline);
            preprocess_depth_pass.set_bind_group(0, &bind_groups.preprocess_depth_bind_group, &[]);
            preprocess_depth_pass.set_bind_group(
                1,
                &bind_groups.common_bind_group,
                &[view_uniform_offset.offset],
            );
            preprocess_depth_pass.dispatch_workgroups(
                camera_size.x.div_ceil(16),
                camera_size.y.div_ceil(16),
                1,
            );
        }

        {
            let mut ssao_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ssao_ssao_pass"),
                        timestamp_writes: None,
                    });
            ssao_pass.set_pipeline(ssao_pipeline);
            ssao_pass.set_bind_group(0, &bind_groups.ssao_bind_group, &[]);
            ssao_pass.set_bind_group(
                1,
                &bind_groups.common_bind_group,
                &[view_uniform_offset.offset],
            );
            ssao_pass.dispatch_workgroups(camera_size.x.div_ceil(8), camera_size.y.div_ceil(8), 1);
        }

        {
            let mut spatial_denoise_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ssao_spatial_denoise_pass"),
                        timestamp_writes: None,
                    });
            spatial_denoise_pass.set_pipeline(spatial_denoise_pipeline);
            spatial_denoise_pass.set_bind_group(0, &bind_groups.spatial_denoise_bind_group, &[]);
            spatial_denoise_pass.set_bind_group(
                1,
                &bind_groups.common_bind_group,
                &[view_uniform_offset.offset],
            );
            spatial_denoise_pass.dispatch_workgroups(
                camera_size.x.div_ceil(8),
                camera_size.y.div_ceil(8),
                1,
            );
        }

        render_context.command_encoder().pop_debug_group();
        Ok(())
    }
}

fn generate_hilbert_index_lut() -> [[u16; 64]; 64] {
    use core::array::from_fn;
    from_fn(|x| from_fn(|y| hilbert_index(x as u16, y as u16)))
}
