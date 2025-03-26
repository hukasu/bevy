use bevy_ecs::{
    component::Component,
    query::QueryItem,
    resource::Resource,
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_math::{Mat4, UVec2, Vec3Swizzles};
use bevy_render::{
    extract_component::DynamicUniformIndex,
    render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
    render_resource::{
        binding_types, AddressMode, BindGroup, BindGroupLayout, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePass, ComputePassDescriptor, ComputePipelineDescriptor,
        DynamicUniformBuffer, FilterMode, PipelineCache, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, ShaderType, StorageTextureAccess, TextureFormat,
        TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice},
    view::{ViewUniform, ViewUniformOffset},
};

use crate::{
    light::plugin::shader_type::GpuLights, mesh_pipeline::pipeline::view::ViewLightsUniformOffset,
};

use super::{
    plugin::{AERIAL_VIEW_LUT, MULTISCATTERING_LUT, SKY_VIEW_LUT, TRANSMITTANCE_LUT},
    Atmosphere, AtmosphereSettings,
};

#[derive(Resource)]
pub(crate) struct AtmosphereLutPipelines {
    pub transmittance_lut: CachedComputePipelineId,
    pub multiscattering_lut: CachedComputePipelineId,
    pub sky_view_lut: CachedComputePipelineId,
    pub aerial_view_lut: CachedComputePipelineId,
}

impl FromWorld for AtmosphereLutPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<AtmosphereBindGroupLayouts>();

        let transmittance_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transmittance_lut_pipeline".into()),
            layout: vec![layouts.transmittance_lut.clone()],
            push_constant_ranges: vec![],
            shader: TRANSMITTANCE_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let multiscattering_lut =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("multi_scattering_lut_pipeline".into()),
                layout: vec![layouts.multiscattering_lut.clone()],
                push_constant_ranges: vec![],
                shader: MULTISCATTERING_LUT,
                shader_defs: vec![],
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        let sky_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("sky_view_lut_pipeline".into()),
            layout: vec![layouts.sky_view_lut.clone()],
            push_constant_ranges: vec![],
            shader: SKY_VIEW_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let aerial_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("aerial_view_lut_pipeline".into()),
            layout: vec![layouts.aerial_view_lut.clone()],
            push_constant_ranges: vec![],
            shader: AERIAL_VIEW_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
        }
    }
}

#[derive(Component)]
pub(crate) struct AtmosphereBindGroups {
    pub transmittance_lut: BindGroup,
    pub multiscattering_lut: BindGroup,
    pub sky_view_lut: BindGroup,
    pub aerial_view_lut: BindGroup,
    pub render_sky: BindGroup,
}

#[derive(Resource)]
pub struct AtmosphereBindGroupLayouts {
    pub transmittance_lut: BindGroupLayout,
    pub multiscattering_lut: BindGroupLayout,
    pub sky_view_lut: BindGroupLayout,
    pub aerial_view_lut: BindGroupLayout,
}

impl FromWorld for AtmosphereBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let transmittance_lut = render_device.create_bind_group_layout(
            "transmittance_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, binding_types::uniform_buffer::<Atmosphere>(true)),
                    (1, binding_types::uniform_buffer::<AtmosphereSettings>(true)),
                    (
                        // transmittance lut storage texture
                        13,
                        binding_types::texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let multiscattering_lut = render_device.create_bind_group_layout(
            "multiscattering_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, binding_types::uniform_buffer::<Atmosphere>(true)),
                    (1, binding_types::uniform_buffer::<AtmosphereSettings>(true)),
                    (
                        5,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //transmittance lut and sampler
                    (6, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        //multiscattering lut storage texture
                        13,
                        binding_types::texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let sky_view_lut = render_device.create_bind_group_layout(
            "sky_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, binding_types::uniform_buffer::<Atmosphere>(true)),
                    (1, binding_types::uniform_buffer::<AtmosphereSettings>(true)),
                    (
                        2,
                        binding_types::uniform_buffer::<AtmosphereTransform>(true),
                    ),
                    (3, binding_types::uniform_buffer::<ViewUniform>(true)),
                    (4, binding_types::uniform_buffer::<GpuLights>(true)),
                    (
                        5,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //transmittance lut and sampler
                    (6, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        7,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //multiscattering lut and sampler
                    (8, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        13,
                        binding_types::texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let aerial_view_lut = render_device.create_bind_group_layout(
            "aerial_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, binding_types::uniform_buffer::<Atmosphere>(true)),
                    (1, binding_types::uniform_buffer::<AtmosphereSettings>(true)),
                    (3, binding_types::uniform_buffer::<ViewUniform>(true)),
                    (4, binding_types::uniform_buffer::<GpuLights>(true)),
                    (
                        5,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //transmittance lut and sampler
                    (6, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        7,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //multiscattering lut and sampler
                    (8, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        //Aerial view lut storage texture
                        13,
                        binding_types::texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
        }
    }
}

#[derive(Resource)]
pub struct AtmosphereSamplers {
    pub transmittance_lut: Sampler,
    pub multiscattering_lut: Sampler,
    pub sky_view_lut: Sampler,
    pub aerial_view_lut: Sampler,
}

impl FromWorld for AtmosphereSamplers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let base_sampler = SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        };

        let transmittance_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("transmittance_lut_sampler"),
            ..base_sampler
        });

        let multiscattering_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("multiscattering_lut_sampler"),
            ..base_sampler
        });

        let sky_view_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("sky_view_lut_sampler"),
            address_mode_u: AddressMode::Repeat,
            ..base_sampler
        });

        let aerial_view_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("aerial_view_lut_sampler"),
            ..base_sampler
        });

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
        }
    }
}

#[derive(Resource, Default)]
pub struct AtmosphereTransforms {
    uniforms: DynamicUniformBuffer<AtmosphereTransform>,
}

impl AtmosphereTransforms {
    #[inline]
    pub fn uniforms(&self) -> &DynamicUniformBuffer<AtmosphereTransform> {
        &self.uniforms
    }

    #[inline]
    pub fn uniforms_mut(&mut self) -> &mut DynamicUniformBuffer<AtmosphereTransform> {
        &mut self.uniforms
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum AtmosphereNode {
    RenderLuts,
    RenderSky,
}

#[derive(Default)]
pub struct AtmosphereLutsNode {}

impl ViewNode for AtmosphereLutsNode {
    type ViewQuery = (
        Read<AtmosphereSettings>,
        Read<AtmosphereBindGroups>,
        Read<DynamicUniformIndex<Atmosphere>>,
        Read<DynamicUniformIndex<AtmosphereSettings>>,
        Read<AtmosphereTransformsOffset>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            settings,
            bind_groups,
            atmosphere_uniforms_offset,
            settings_uniforms_offset,
            atmosphere_transforms_offset,
            view_uniforms_offset,
            lights_uniforms_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<AtmosphereLutPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (
            Some(transmittance_lut_pipeline),
            Some(multiscattering_lut_pipeline),
            Some(sky_view_lut_pipeline),
            Some(aerial_view_lut_pipeline),
        ) = (
            pipeline_cache.get_compute_pipeline(pipelines.transmittance_lut),
            pipeline_cache.get_compute_pipeline(pipelines.multiscattering_lut),
            pipeline_cache.get_compute_pipeline(pipelines.sky_view_lut),
            pipeline_cache.get_compute_pipeline(pipelines.aerial_view_lut),
        )
        else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();

        let mut luts_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("atmosphere_luts_pass"),
            timestamp_writes: None,
        });

        fn dispatch_2d(compute_pass: &mut ComputePass, size: UVec2) {
            const WORKGROUP_SIZE: u32 = 16;
            let workgroups_x = size.x.div_ceil(WORKGROUP_SIZE);
            let workgroups_y = size.y.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Transmittance LUT

        luts_pass.set_pipeline(transmittance_lut_pipeline);
        luts_pass.set_bind_group(
            0,
            &bind_groups.transmittance_lut,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
            ],
        );

        dispatch_2d(&mut luts_pass, settings.transmittance_lut_size);

        // Multiscattering LUT

        luts_pass.set_pipeline(multiscattering_lut_pipeline);
        luts_pass.set_bind_group(
            0,
            &bind_groups.multiscattering_lut,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
            ],
        );

        luts_pass.dispatch_workgroups(
            settings.multiscattering_lut_size.x,
            settings.multiscattering_lut_size.y,
            1,
        );

        // Sky View LUT

        luts_pass.set_pipeline(sky_view_lut_pipeline);
        luts_pass.set_bind_group(
            0,
            &bind_groups.sky_view_lut,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );

        dispatch_2d(&mut luts_pass, settings.sky_view_lut_size);

        // Aerial View LUT

        luts_pass.set_pipeline(aerial_view_lut_pipeline);
        luts_pass.set_bind_group(
            0,
            &bind_groups.aerial_view_lut,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );

        dispatch_2d(&mut luts_pass, settings.aerial_view_lut_size.xy());

        Ok(())
    }
}

#[derive(ShaderType)]
pub struct AtmosphereTransform {
    pub world_from_atmosphere: Mat4,
    pub atmosphere_from_world: Mat4,
}

#[derive(Component)]
pub struct AtmosphereTransformsOffset {
    pub index: u32,
}

impl AtmosphereTransformsOffset {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}
