use bevy_core_pipeline::fullscreen_vertex_shader;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    query::QueryItem,
    resource::Resource,
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_image::BevyDefault;
use bevy_render::{
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{
        binding_types, AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, DynamicUniformBuffer, FilterMode,
        FragmentState, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        ShaderType, SpecializedRenderPipeline, TextureFormat, TextureSampleType,
    },
    renderer::{RenderAdapter, RenderContext, RenderDevice},
    view::{ViewTarget, ViewUniformOffset},
};
use bevy_utils::default;

use crate::{
    binding_arrays_are_usable,
    distance_fog::fog::ViewFogUniformOffset,
    light::ViewLightsUniformOffset,
    light_probe::{
        environment_map::ViewEnvironmentMapUniformOffset,
        light_probes::ViewLightProbesUniformOffset,
    },
    mesh_pipeline::render::{
        pipeline::{MeshPipelineViewLayoutKey, MeshPipelineViewLayouts},
        MeshViewBindGroup,
    },
};

use super::{plugin::SSR_SHADER_HANDLE, ScreenSpaceReflections};

/// Information relating to the render pipeline for the screen space reflections
/// shader.
#[derive(Resource)]
pub struct ScreenSpaceReflectionsPipeline {
    mesh_view_layouts: MeshPipelineViewLayouts,
    color_sampler: Sampler,
    depth_linear_sampler: Sampler,
    depth_nearest_sampler: Sampler,
    bind_group_layout: BindGroupLayout,
    binding_arrays_are_usable: bool,
}

impl FromWorld for ScreenSpaceReflectionsPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_view_layouts = world.resource::<MeshPipelineViewLayouts>().clone();
        let render_device = world.resource::<RenderDevice>();
        let render_adapter = world.resource::<RenderAdapter>();

        // Create the bind group layout.
        let bind_group_layout = render_device.create_bind_group_layout(
            "SSR bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
                    binding_types::sampler(SamplerBindingType::Filtering),
                    binding_types::sampler(SamplerBindingType::Filtering),
                    binding_types::sampler(SamplerBindingType::NonFiltering),
                ),
            ),
        );

        // Create the samplers we need.

        let color_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: "SSR color sampler".into(),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        let depth_linear_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: "SSR depth linear sampler".into(),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        let depth_nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: "SSR depth nearest sampler".into(),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..default()
        });

        Self {
            mesh_view_layouts,
            color_sampler,
            depth_linear_sampler,
            depth_nearest_sampler,
            bind_group_layout,
            binding_arrays_are_usable: binding_arrays_are_usable(render_device, render_adapter),
        }
    }
}

impl SpecializedRenderPipeline for ScreenSpaceReflectionsPipeline {
    type Key = ScreenSpaceReflectionsPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mesh_view_layout = self
            .mesh_view_layouts
            .get_view_layout(key.mesh_pipeline_view_key);

        let mut shader_defs = vec![
            "DEPTH_PREPASS".into(),
            "DEFERRED_PREPASS".into(),
            "SCREEN_SPACE_REFLECTIONS".into(),
        ];

        if key.has_environment_maps {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }

        if self.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
        }

        RenderPipelineDescriptor {
            label: Some("SSR pipeline".into()),
            layout: vec![mesh_view_layout.clone(), self.bind_group_layout.clone()],
            vertex: fullscreen_vertex_shader::fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: SSR_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: if key.is_hdr {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            push_constant_ranges: vec![],
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// Identifies which screen space reflections render pipeline a view needs.
#[derive(Component, Deref, DerefMut)]
pub struct ScreenSpaceReflectionsPipelineId(pub CachedRenderPipelineId);

/// Identifies a specific configuration of the SSR pipeline shader.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScreenSpaceReflectionsPipelineKey {
    pub mesh_pipeline_view_key: MeshPipelineViewLayoutKey,
    pub is_hdr: bool,
    pub has_environment_maps: bool,
}

/// A GPU buffer that stores the screen space reflection settings for each view.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct ScreenSpaceReflectionsBuffer(pub DynamicUniformBuffer<ScreenSpaceReflectionsUniform>);

/// The node in the render graph that traces screen space reflections.
#[derive(Default)]
pub struct ScreenSpaceReflectionsNode;

impl ViewNode for ScreenSpaceReflectionsNode {
    type ViewQuery = (
        Read<ViewTarget>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<ViewFogUniformOffset>,
        Read<ViewLightProbesUniformOffset>,
        Read<ViewScreenSpaceReflectionsUniformOffset>,
        Read<ViewEnvironmentMapUniformOffset>,
        Read<MeshViewBindGroup>,
        Read<ScreenSpaceReflectionsPipelineId>,
    );

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            view_target,
            view_uniform_offset,
            view_lights_offset,
            view_fog_offset,
            view_light_probes_offset,
            view_ssr_offset,
            view_environment_map_offset,
            view_bind_group,
            ssr_pipeline_id,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Grab the render pipeline.
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(**ssr_pipeline_id) else {
            return Ok(());
        };

        // Set up a standard pair of postprocessing textures.
        let postprocess = view_target.post_process_write();

        // Create the bind group for this view.
        let ssr_pipeline = world.resource::<ScreenSpaceReflectionsPipeline>();
        let ssr_bind_group = render_context.render_device().create_bind_group(
            "SSR bind group",
            &ssr_pipeline.bind_group_layout,
            &BindGroupEntries::sequential((
                postprocess.source,
                &ssr_pipeline.color_sampler,
                &ssr_pipeline.depth_linear_sampler,
                &ssr_pipeline.depth_nearest_sampler,
            )),
        );

        // Build the SSR render pass.
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("SSR pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: postprocess.destination,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set bind groups.
        render_pass.set_render_pipeline(render_pipeline);
        render_pass.set_bind_group(
            0,
            &view_bind_group.value,
            &[
                view_uniform_offset.offset,
                view_lights_offset.offset,
                view_fog_offset.offset,
                **view_light_probes_offset,
                **view_ssr_offset,
                **view_environment_map_offset,
            ],
        );

        // Perform the SSR render pass.
        render_pass.set_bind_group(1, &ssr_bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

/// A version of [`ScreenSpaceReflections`] for upload to the GPU.
///
/// For more information on these fields, see the corresponding documentation in
/// [`ScreenSpaceReflections`].
#[derive(Clone, Copy, Component, ShaderType)]
pub struct ScreenSpaceReflectionsUniform {
    perceptual_roughness_threshold: f32,
    thickness: f32,
    linear_steps: u32,
    linear_march_exponent: f32,
    bisection_steps: u32,
    /// A boolean converted to a `u32`.
    use_secant: u32,
}

impl From<ScreenSpaceReflections> for ScreenSpaceReflectionsUniform {
    fn from(settings: ScreenSpaceReflections) -> Self {
        Self {
            perceptual_roughness_threshold: settings.perceptual_roughness_threshold,
            thickness: settings.thickness,
            linear_steps: settings.linear_steps,
            linear_march_exponent: settings.linear_march_exponent,
            bisection_steps: settings.bisection_steps,
            use_secant: settings.use_secant as u32,
        }
    }
}

/// A component that stores the offset within the
/// [`ScreenSpaceReflectionsBuffer`] for each view.
#[derive(Component, Default, Deref, DerefMut)]
pub struct ViewScreenSpaceReflectionsUniformOffset(pub u32);
