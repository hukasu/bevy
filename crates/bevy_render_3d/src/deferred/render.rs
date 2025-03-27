use bevy_core_pipeline::deferred::{
    copy_lighting_id::DeferredLightingIdDepthTexture, DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
};
use bevy_ecs::{
    component::Component,
    query::QueryItem,
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_image::BevyDefault;
use bevy_render::{
    extract_component::ComponentUniforms,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{
        binding_types::uniform_buffer, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
        DepthStencilState, FragmentState, LoadOp, MultisampleState, Operations, PipelineCache,
        PrimitiveState, RenderPassDepthStencilAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, ShaderDefVal, ShaderStages, SpecializedRenderPipeline,
        StencilFaceState, StencilState, StoreOp, TextureFormat, VertexState,
    },
    renderer::{RenderContext, RenderDevice},
    view::{ViewTarget, ViewUniformOffset},
};

use crate::{
    distance_fog::fog::ViewFogUniformOffset,
    light::ViewLightsUniformOffset,
    light_probe::{
        environment_map::ViewEnvironmentMapUniformOffset,
        light_probes::ViewLightProbesUniformOffset,
    },
    mesh_pipeline::render::{
        pipeline::{MeshPipeline, MeshPipelineKey},
        MeshViewBindGroup, TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
        TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
    },
    ssr::render::ViewScreenSpaceReflectionsUniformOffset,
};

use super::{plugin::DEFERRED_LIGHTING_SHADER_HANDLE, DeferredLightingDepthId};

#[derive(Default)]
pub struct DeferredOpaquePass3dLightingNode;

impl ViewNode for DeferredOpaquePass3dLightingNode {
    type ViewQuery = (
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewFogUniformOffset,
        &'static ViewLightProbesUniformOffset,
        &'static ViewScreenSpaceReflectionsUniformOffset,
        &'static ViewEnvironmentMapUniformOffset,
        &'static MeshViewBindGroup,
        &'static ViewTarget,
        &'static DeferredLightingIdDepthTexture,
        &'static DeferredLightingPipeline,
    );

    fn run(
        &self,
        _graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            view_uniform_offset,
            view_lights_offset,
            view_fog_offset,
            view_light_probes_offset,
            view_ssr_offset,
            view_environment_map_offset,
            mesh_view_bind_group,
            target,
            deferred_lighting_id_depth_texture,
            deferred_lighting_pipeline,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let deferred_lighting_layout = world.resource::<DeferredLightingLayout>();

        let Some(pipeline) =
            pipeline_cache.get_render_pipeline(deferred_lighting_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let deferred_lighting_pass_id =
            world.resource::<ComponentUniforms<DeferredLightingDepthId>>();
        let Some(deferred_lighting_pass_id_binding) =
            deferred_lighting_pass_id.uniforms().binding()
        else {
            return Ok(());
        };

        let bind_group_1 = render_context.render_device().create_bind_group(
            "deferred_lighting_layout_group_1",
            &deferred_lighting_layout.bind_group_layout_1,
            &BindGroupEntries::single(deferred_lighting_pass_id_binding),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("deferred_lighting_pass"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &deferred_lighting_id_depth_texture.texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &mesh_view_bind_group.value,
            &[
                view_uniform_offset.offset,
                view_lights_offset.offset,
                view_fog_offset.offset,
                **view_light_probes_offset,
                **view_ssr_offset,
                **view_environment_map_offset,
            ],
        );
        render_pass.set_bind_group(1, &bind_group_1, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
pub struct DeferredLightingLayout {
    mesh_pipeline: MeshPipeline,
    bind_group_layout_1: BindGroupLayout,
}

#[derive(Component)]
pub struct DeferredLightingPipeline {
    pub pipeline_id: CachedRenderPipelineId,
}

impl SpecializedRenderPipeline for DeferredLightingLayout {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        // Let the shader code know that it's running in a deferred pipeline.
        shader_defs.push("DEFERRED_LIGHTING_PIPELINE".into());

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        shader_defs.push("WEBGL2".into());

        if key.contains(MeshPipelineKey::TONEMAP_IN_SHADER) {
            shader_defs.push("TONEMAP_IN_SHADER".into());
            shader_defs.push(ShaderDefVal::UInt(
                "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
                TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
            ));
            shader_defs.push(ShaderDefVal::UInt(
                "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
                TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
            ));

            let method = key.intersection(MeshPipelineKey::TONEMAP_METHOD_RESERVED_BITS);

            if method == MeshPipelineKey::TONEMAP_METHOD_NONE {
                shader_defs.push("TONEMAP_METHOD_NONE".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD {
                shader_defs.push("TONEMAP_METHOD_REINHARD".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED {
                shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_AGX {
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC {
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE {
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }

            // Debanding is tied to tonemapping in the shader, cannot run without it.
            if key.contains(MeshPipelineKey::DEBAND_DITHER) {
                shader_defs.push("DEBAND_DITHER".into());
            }
        }

        if key.contains(MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
            shader_defs.push("SCREEN_SPACE_AMBIENT_OCCLUSION".into());
        }

        if key.contains(MeshPipelineKey::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }

        if key.contains(MeshPipelineKey::IRRADIANCE_VOLUME) {
            shader_defs.push("IRRADIANCE_VOLUME".into());
        }

        if key.contains(MeshPipelineKey::NORMAL_PREPASS) {
            shader_defs.push("NORMAL_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            shader_defs.push("MOTION_VECTOR_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::SCREEN_SPACE_REFLECTIONS) {
            shader_defs.push("SCREEN_SPACE_REFLECTIONS".into());
        }

        if key.contains(MeshPipelineKey::HAS_PREVIOUS_SKIN) {
            shader_defs.push("HAS_PREVIOUS_SKIN".into());
        }

        if key.contains(MeshPipelineKey::HAS_PREVIOUS_MORPH) {
            shader_defs.push("HAS_PREVIOUS_MORPH".into());
        }

        if key.contains(MeshPipelineKey::DISTANCE_FOG) {
            shader_defs.push("DISTANCE_FOG".into());
        }

        // Always true, since we're in the deferred lighting pipeline
        shader_defs.push("DEFERRED_PREPASS".into());

        let shadow_filter_method =
            key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
        if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN {
            shader_defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
            shader_defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
        }

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        shader_defs.push("SIXTEEN_BYTE_ALIGNMENT".into());

        RenderPipelineDescriptor {
            label: Some("deferred_lighting_pipeline".into()),
            layout: vec![
                self.mesh_pipeline.get_view_layout(key.into()).clone(),
                self.bind_group_layout_1.clone(),
            ],
            vertex: VertexState {
                shader: DEFERRED_LIGHTING_SHADER_HANDLE,
                shader_defs: shader_defs.clone(),
                entry_point: "vertex".into(),
                buffers: Vec::new(),
            },
            fragment: Some(FragmentState {
                shader: DEFERRED_LIGHTING_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: if key.contains(MeshPipelineKey::HDR) {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Equal,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl FromWorld for DeferredLightingLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "deferred_lighting_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX_FRAGMENT,
                uniform_buffer::<DeferredLightingDepthId>(false),
            ),
        );
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            bind_group_layout_1: layout,
        }
    }
}
