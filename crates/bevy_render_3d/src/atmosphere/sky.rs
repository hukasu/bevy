use bevy_core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy_ecs::{
    component::Component,
    query::QueryItem,
    resource::Resource,
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_render::{
    extract_component::DynamicUniformIndex,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{
        binding_types, BindGroupLayout, BindGroupLayoutEntries, BlendComponent, BlendFactor,
        BlendOperation, BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites,
        FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPassDescriptor,
        RenderPipelineDescriptor, SamplerBindingType, ShaderStages, SpecializedRenderPipeline,
        TextureFormat, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice},
    view::{ViewTarget, ViewUniform, ViewUniformOffset},
};

use crate::{
    atmosphere::{Atmosphere, AtmosphereSettings},
    light::{render::GpuLights, ViewLightsUniformOffset},
};

use super::{
    atmosphere::{AtmosphereBindGroups, AtmosphereTransform, AtmosphereTransformsOffset},
    plugin::RENDER_SKY,
};

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct RenderSkyPipelineKey {
    pub msaa_samples: u32,
    pub hdr: bool,
}

#[derive(Component)]
pub struct RenderSkyPipelineId(pub CachedRenderPipelineId);

#[derive(Resource)]
pub struct RenderSkyBindGroupLayouts {
    pub render_sky: BindGroupLayout,
    pub render_sky_msaa: BindGroupLayout,
}

impl FromWorld for RenderSkyBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_sky = render_device.create_bind_group_layout(
            "render_sky_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
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
                        9,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //sky view lut and sampler
                    (10, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        // aerial view lut and sampler
                        11,
                        binding_types::texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    (12, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        //view depth texture
                        13,
                        binding_types::texture_2d(TextureSampleType::Depth),
                    ),
                ),
            ),
        );

        let render_sky_msaa = render_device.create_bind_group_layout(
            "render_sky_msaa_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
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
                        9,
                        binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    ), //sky view lut and sampler
                    (10, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        // aerial view lut and sampler
                        11,
                        binding_types::texture_3d(TextureSampleType::Float { filterable: true }),
                    ),
                    (12, binding_types::sampler(SamplerBindingType::Filtering)),
                    (
                        //view depth texture
                        13,
                        binding_types::texture_2d_multisampled(TextureSampleType::Depth),
                    ),
                ),
            ),
        );

        Self {
            render_sky,
            render_sky_msaa,
        }
    }
}

impl SpecializedRenderPipeline for RenderSkyBindGroupLayouts {
    type Key = RenderSkyPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        if key.msaa_samples > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if key.hdr {
            shader_defs.push("TONEMAP_IN_SHADER".into());
        }

        RenderPipelineDescriptor {
            label: Some(format!("render_sky_pipeline_{}", key.msaa_samples).into()),
            layout: vec![if key.msaa_samples == 1 {
                self.render_sky.clone()
            } else {
                self.render_sky_msaa.clone()
            }],
            push_constant_ranges: vec![],
            vertex: fullscreen_shader_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
            fragment: Some(FragmentState {
                shader: RENDER_SKY.clone(),
                shader_defs,
                entry_point: "main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::Src1,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
        }
    }
}

#[derive(Default)]
pub struct RenderSkyNode;

impl ViewNode for RenderSkyNode {
    type ViewQuery = (
        Read<AtmosphereBindGroups>,
        Read<ViewTarget>,
        Read<DynamicUniformIndex<Atmosphere>>,
        Read<DynamicUniformIndex<AtmosphereSettings>>,
        Read<AtmosphereTransformsOffset>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<RenderSkyPipelineId>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            atmosphere_bind_groups,
            view_target,
            atmosphere_uniforms_offset,
            settings_uniforms_offset,
            atmosphere_transforms_offset,
            view_uniforms_offset,
            lights_uniforms_offset,
            render_sky_pipeline_id,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(render_sky_pipeline) =
            pipeline_cache.get_render_pipeline(render_sky_pipeline_id.0)
        else {
            return Ok(());
        }; //TODO: warning

        let mut render_sky_pass =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("render_sky_pass"),
                    color_attachments: &[Some(view_target.get_color_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

        render_sky_pass.set_pipeline(render_sky_pipeline);
        render_sky_pass.set_bind_group(
            0,
            &atmosphere_bind_groups.render_sky,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );
        render_sky_pass.draw(0..3, 0..1);

        Ok(())
    }
}
