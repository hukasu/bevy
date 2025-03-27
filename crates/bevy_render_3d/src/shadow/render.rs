use core::marker::PhantomData;

use alloc::string::String;

use bevy_asset::UntypedAssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::{Component, Tick},
    query::{Has, QueryState},
    resource::Resource,
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_platform_support::collections::HashMap;
use bevy_render::{
    diagnostic::RecordDiagnostics,
    experimental::occlusion_culling::OcclusionCulling,
    mesh::allocator::SlabId,
    render_graph::{Node, NodeRunError, RenderGraphContext},
    render_phase::{
        DrawFunctionId, PhaseItemBatchSetKey, TrackedRenderPass, ViewBinnedRenderPhases,
    },
    render_resource::{
        AddressMode, CachedRenderPipelineId, CommandEncoderDescriptor, CompareFunction, FilterMode,
        RenderPassDescriptor, Sampler, SamplerDescriptor, StoreOp, Texture, TextureView,
    },
    renderer::{RenderContext, RenderDevice},
    sync_world::MainEntityHashMap,
    texture::DepthAttachment,
    view::{ExtractedView, RetainedViewEntity},
};
use bevy_utils::prelude::default;
use tracing::error;

use crate::light::ViewLightEntities;

use super::phase_item::Shadow;

#[derive(Component)]
pub struct ShadowView {
    pub depth_attachment: DepthAttachment,
    pub pass_name: String,
}

#[derive(Resource, Clone)]
pub struct ShadowSamplers {
    pub point_light_comparison_sampler: Sampler,
    #[cfg(feature = "experimental_pbr_pcss")]
    pub point_light_linear_sampler: Sampler,
    pub directional_light_comparison_sampler: Sampler,
    #[cfg(feature = "experimental_pbr_pcss")]
    pub directional_light_linear_sampler: Sampler,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for ShadowSamplers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let base_sampler_descriptor = SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..default()
        };

        ShadowSamplers {
            point_light_comparison_sampler: render_device.create_sampler(&SamplerDescriptor {
                compare: Some(CompareFunction::GreaterEqual),
                ..base_sampler_descriptor
            }),
            #[cfg(feature = "experimental_pbr_pcss")]
            point_light_linear_sampler: render_device.create_sampler(&base_sampler_descriptor),
            directional_light_comparison_sampler: render_device.create_sampler(
                &SamplerDescriptor {
                    compare: Some(CompareFunction::GreaterEqual),
                    ..base_sampler_descriptor
                },
            ),
            #[cfg(feature = "experimental_pbr_pcss")]
            directional_light_linear_sampler: render_device
                .create_sampler(&base_sampler_descriptor),
        }
    }
}

#[derive(Component)]
pub struct ViewShadowBindings {
    pub point_light_depth_texture: Texture,
    pub point_light_depth_texture_view: TextureView,
    pub directional_light_depth_texture: Texture,
    pub directional_light_depth_texture_view: TextureView,
}

/// Information that must be identical in order to place opaque meshes in the
/// same *batch set*.
///
/// A batch set is a set of batches that can be multi-drawn together, if
/// multi-draw is in use.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShadowBatchSetKey {
    /// The identifier of the render pipeline.
    pub pipeline: CachedRenderPipelineId,

    /// The function used to draw.
    pub draw_function: DrawFunctionId,

    /// The ID of a bind group specific to the material.
    ///
    /// In the case of PBR, this is the `MaterialBindGroupIndex`.
    pub material_bind_group_index: Option<u32>,

    /// The ID of the slab of GPU memory that contains vertex data.
    ///
    /// For non-mesh items, you can fill this with 0 if your items can be
    /// multi-drawn, or with a unique value if they can't.
    pub vertex_slab: SlabId,

    /// The ID of the slab of GPU memory that contains index data, if present.
    ///
    /// For non-mesh items, you can safely fill this with `None`.
    pub index_slab: Option<SlabId>,
}

impl PhaseItemBatchSetKey for ShadowBatchSetKey {
    fn indexed(&self) -> bool {
        self.index_slab.is_some()
    }
}

/// Data used to bin each object in the shadow map phase.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShadowBinKey {
    /// The object.
    pub asset_id: UntypedAssetId,
}

#[derive(Resource, Deref, DerefMut)]
pub struct SpecializedShadowMaterialPipelineCache<M> {
    // view light entity -> view pipeline cache
    #[deref]
    map: HashMap<RetainedViewEntity, SpecializedShadowMaterialViewPipelineCache<M>>,
    marker: PhantomData<M>,
}

#[derive(Deref, DerefMut)]
pub struct SpecializedShadowMaterialViewPipelineCache<M> {
    #[deref]
    map: MainEntityHashMap<(Tick, CachedRenderPipelineId)>,
    marker: PhantomData<M>,
}

impl<M> Default for SpecializedShadowMaterialPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            marker: PhantomData,
        }
    }
}

impl<M> Default for SpecializedShadowMaterialViewPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: MainEntityHashMap::default(),
            marker: PhantomData,
        }
    }
}

/// The rendering node that renders meshes that were "visible" (so to speak)
/// from a light last frame.
///
/// If occlusion culling for a light is disabled, then this node simply renders
/// all meshes in range of the light.
#[derive(Deref, DerefMut)]
pub struct EarlyShadowPassNode(ShadowPassNode);

/// The rendering node that renders meshes that became newly "visible" (so to
/// speak) from a light this frame.
///
/// If occlusion culling for a light is disabled, then this node does nothing.
#[derive(Deref, DerefMut)]
pub struct LateShadowPassNode(ShadowPassNode);

/// Encapsulates rendering logic shared between the early and late shadow pass
/// nodes.
pub struct ShadowPassNode {
    /// The query that finds cameras in which shadows are visible.
    main_view_query: QueryState<Read<ViewLightEntities>>,
    /// The query that finds shadow cascades.
    view_light_query: QueryState<(Read<ShadowView>, Read<ExtractedView>, Has<OcclusionCulling>)>,
}

impl FromWorld for EarlyShadowPassNode {
    fn from_world(world: &mut World) -> Self {
        Self(ShadowPassNode::from_world(world))
    }
}

impl FromWorld for LateShadowPassNode {
    fn from_world(world: &mut World) -> Self {
        Self(ShadowPassNode::from_world(world))
    }
}

impl FromWorld for ShadowPassNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
            view_light_query: QueryState::new(world),
        }
    }
}

impl Node for EarlyShadowPassNode {
    fn update(&mut self, world: &mut World) {
        self.0.update(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        self.0.run(graph, render_context, world, false)
    }
}

impl Node for LateShadowPassNode {
    fn update(&mut self, world: &mut World) {
        self.0.update(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        self.0.run(graph, render_context, world, true)
    }
}

impl ShadowPassNode {
    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
        self.view_light_query.update_archetypes(world);
    }

    /// Runs the node logic.
    ///
    /// `is_late` is true if this is the late shadow pass or false if this is
    /// the early shadow pass.
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
        is_late: bool,
    ) -> Result<(), NodeRunError> {
        let diagnostics = render_context.diagnostic_recorder();

        let view_entity = graph.view_entity();

        let Some(shadow_render_phases) = world.get_resource::<ViewBinnedRenderPhases<Shadow>>()
        else {
            return Ok(());
        };

        let time_span = diagnostics.time_span(render_context.command_encoder(), "shadows");

        if let Ok(view_lights) = self.main_view_query.get_manual(world, view_entity) {
            for view_light_entity in view_lights.lights.iter().copied() {
                let Ok((view_light, extracted_light_view, occlusion_culling)) =
                    self.view_light_query.get_manual(world, view_light_entity)
                else {
                    continue;
                };

                // There's no need for a late shadow pass if the light isn't
                // using occlusion culling.
                if is_late && !occlusion_culling {
                    continue;
                }

                let Some(shadow_phase) =
                    shadow_render_phases.get(&extracted_light_view.retained_view_entity)
                else {
                    continue;
                };

                let depth_stencil_attachment =
                    Some(view_light.depth_attachment.get_attachment(StoreOp::Store));

                let diagnostics = render_context.diagnostic_recorder();
                render_context.add_command_buffer_generation_task(move |render_device| {
                    #[cfg(feature = "trace")]
                    let _shadow_pass_span = info_span!("", "{}", view_light.pass_name).entered();
                    let mut command_encoder =
                        render_device.create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("shadow_pass_command_encoder"),
                        });

                    let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some(&view_light.pass_name),
                        color_attachments: &[],
                        depth_stencil_attachment,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);
                    let pass_span =
                        diagnostics.pass_span(&mut render_pass, view_light.pass_name.clone());

                    if let Err(err) =
                        shadow_phase.render(&mut render_pass, world, view_light_entity)
                    {
                        error!("Error encountered while rendering the shadow phase {err:?}");
                    }

                    pass_span.end(&mut render_pass);
                    drop(render_pass);
                    command_encoder.finish()
                });
            }
        }

        time_span.end(render_context.command_encoder());

        Ok(())
    }
}
