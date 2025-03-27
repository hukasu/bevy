use std::num::{NonZero, NonZeroU64};

use bevy_core_pipeline::{
    experimental::mip_generation::ViewDepthPyramid,
    prepass::{DepthPrepass, PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Has, Or, QueryState, With, Without},
    resource::Resource,
    system::{lifetimeless::Read, Query},
    world::{FromWorld, World},
};
use bevy_render::{
    batching::gpu_preprocessing::{
        BatchedInstanceBuffers, GpuOcclusionCullingWorkItemBuffers, IndirectBatchSet,
        IndirectParametersBuffers, IndirectParametersCpuMetadata, IndirectParametersGpuMetadata,
        IndirectParametersIndexed, IndirectParametersNonIndexed,
        LatePreprocessWorkItemIndirectParameters, PreprocessWorkItem, PreprocessWorkItemBuffers,
        UntypedPhaseBatchedInstanceBuffers, UntypedPhaseIndirectParametersBuffers,
    },
    experimental::occlusion_culling::OcclusionCulling,
    render_graph::{Node, NodeRunError, RenderGraphContext},
    render_resource::{
        binding_types, BindGroup, BindGroupEntries, BindGroupLayout, BindingResource, Buffer,
        BufferBinding, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        DynamicBindGroupLayoutEntries, PipelineCache, PushConstantRange, RawBufferVec,
        ShaderStages, ShaderType, SpecializedComputePipeline, SpecializedComputePipelines,
        TextureSampleType, UninitBufferVec,
    },
    renderer::{RenderContext, RenderDevice},
    view::{ExtractedView, NoIndirectDrawing, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_utils::TypeIdMap;

use smallvec::{smallvec, SmallVec};
use tracing::warn;

use crate::{
    light::ViewLightEntities,
    mesh_pipeline::render::{
        MeshCullingData, MeshCullingDataBuffer, MeshInputUniform, MeshUniform,
    },
    shadow::render::ShadowView,
};

use super::{
    BUILD_INDIRECT_PARAMS_SHADER_HANDLE, MESH_PREPROCESS_SHADER_HANDLE,
    RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE, WORKGROUP_SIZE,
};

/// The render node that clears out the GPU-side indirect metadata buffers.
///
/// This is only used when indirect drawing is enabled.
#[derive(Default)]
pub struct ClearIndirectParametersMetadataNode;

/// The render node for the first mesh preprocessing pass.
///
/// This pass runs a compute shader to cull meshes outside the view frustum (if
/// that wasn't done by the CPU), cull meshes that weren't visible last frame
/// (if occlusion culling is on), transform them, and, if indirect drawing is
/// on, populate indirect draw parameter metadata for the subsequent
/// [`EarlyPrepassBuildIndirectParametersNode`].
pub struct EarlyGpuPreprocessNode {
    view_query: QueryState<
        (
            Read<ExtractedView>,
            Option<Read<PreprocessBindGroups>>,
            Option<Read<ViewUniformOffset>>,
            Has<NoIndirectDrawing>,
            Has<OcclusionCulling>,
        ),
        Without<SkipGpuPreprocess>,
    >,
    main_view_query: QueryState<Read<ViewLightEntities>>,
}

/// The render node for the second mesh preprocessing pass.
///
/// This pass runs a compute shader to cull meshes outside the view frustum (if
/// that wasn't done by the CPU), cull meshes that were neither visible last
/// frame nor visible this frame (if occlusion culling is on), transform them,
/// and, if indirect drawing is on, populate the indirect draw parameter
/// metadata for the subsequent [`LatePrepassBuildIndirectParametersNode`].
pub struct LateGpuPreprocessNode {
    view_query: QueryState<
        (
            Read<ExtractedView>,
            Read<PreprocessBindGroups>,
            Read<ViewUniformOffset>,
        ),
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            With<OcclusionCulling>,
            With<DepthPrepass>,
        ),
    >,
}

/// The render node for the part of the indirect parameter building pass that
/// draws the meshes visible from the previous frame.
///
/// This node runs a compute shader on the output of the
/// [`EarlyGpuPreprocessNode`] in order to transform the
/// [`IndirectParametersGpuMetadata`] into properly-formatted
/// [`IndirectParametersIndexed`] and [`IndirectParametersNonIndexed`].
pub struct EarlyPrepassBuildIndirectParametersNode {
    view_query: QueryState<
        Read<PreprocessBindGroups>,
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            Or<(With<DepthPrepass>, With<ShadowView>)>,
        ),
    >,
}

/// The render node for the part of the indirect parameter building pass that
/// draws the meshes that are potentially visible on this frame but weren't
/// visible on the previous frame.
///
/// This node runs a compute shader on the output of the
/// [`LateGpuPreprocessNode`] in order to transform the
/// [`IndirectParametersGpuMetadata`] into properly-formatted
/// [`IndirectParametersIndexed`] and [`IndirectParametersNonIndexed`].
pub struct LatePrepassBuildIndirectParametersNode {
    view_query: QueryState<
        Read<PreprocessBindGroups>,
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            Or<(With<DepthPrepass>, With<ShadowView>)>,
            With<OcclusionCulling>,
        ),
    >,
}

/// The render node for the part of the indirect parameter building pass that
/// draws all meshes, both those that are newly-visible on this frame and those
/// that were visible last frame.
///
/// This node runs a compute shader on the output of the
/// [`EarlyGpuPreprocessNode`] and [`LateGpuPreprocessNode`] in order to
/// transform the [`IndirectParametersGpuMetadata`] into properly-formatted
/// [`IndirectParametersIndexed`] and [`IndirectParametersNonIndexed`].
pub struct MainBuildIndirectParametersNode {
    view_query: QueryState<
        Read<PreprocessBindGroups>,
        (Without<SkipGpuPreprocess>, Without<NoIndirectDrawing>),
    >,
}

/// The compute shader pipelines for the GPU mesh preprocessing and indirect
/// parameter building passes.
#[derive(Resource)]
pub struct PreprocessPipelines {
    /// The pipeline used for CPU culling. This pipeline doesn't populate
    /// indirect parameter metadata.
    pub direct_preprocess: PreprocessPipeline,
    /// The pipeline used for mesh preprocessing when GPU frustum culling is in
    /// use, but occlusion culling isn't.
    ///
    /// This pipeline populates indirect parameter metadata.
    pub gpu_frustum_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for the first phase of occlusion culling.
    ///
    /// This pipeline culls, transforms meshes, and populates indirect parameter
    /// metadata.
    pub early_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for the second phase of occlusion culling.
    ///
    /// This pipeline culls, transforms meshes, and populates indirect parameter
    /// metadata.
    pub late_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    /// The pipeline that builds indirect draw parameters for indexed meshes,
    /// when frustum culling is enabled but occlusion culling *isn't* enabled.
    pub gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// The pipeline that builds indirect draw parameters for non-indexed
    /// meshes, when frustum culling is enabled but occlusion culling *isn't*
    /// enabled.
    pub gpu_frustum_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// Compute shader pipelines for the early prepass phase that draws meshes
    /// visible in the previous frame.
    pub early_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the late prepass phase that draws meshes
    /// that weren't visible in the previous frame, but became visible this
    /// frame.
    pub late_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the main color phase.
    pub main_phase: PreprocessPhasePipelines,
}

/// Compute shader pipelines for a specific phase: early, late, or main.
///
/// The distinction between these phases is relevant for occlusion culling.
#[derive(Clone)]
pub struct PreprocessPhasePipelines {
    /// The pipeline that resets the indirect draw counts used in
    /// `multi_draw_indirect_count` to 0 in preparation for a new pass.
    pub reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline,
    /// The pipeline used for indexed indirect parameter building.
    ///
    /// This pipeline converts indirect parameter metadata into indexed indirect
    /// parameters.
    pub gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// The pipeline used for non-indexed indirect parameter building.
    ///
    /// This pipeline converts indirect parameter metadata into non-indexed
    /// indirect parameters.
    pub gpu_occlusion_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
}

/// The pipeline for the GPU mesh preprocessing shader.
pub struct PreprocessPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayout,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the batch set count reset shader.
///
/// This shader resets the indirect batch set count to 0 for each view. It runs
/// in between every phase (early, late, and main).
#[derive(Clone)]
pub struct ResetIndirectBatchSetsPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayout,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the indirect parameter building shader.
#[derive(Clone)]
pub struct BuildIndirectParametersPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayout,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

bitflags::bitflags! {
    /// Specifies variants of the mesh preprocessing shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PreprocessPipelineKey: u8 {
        /// Whether GPU frustum culling is in use.
        ///
        /// This `#define`'s `FRUSTUM_CULLING` in the shader.
        const FRUSTUM_CULLING = 1;
        /// Whether GPU two-phase occlusion culling is in use.
        ///
        /// This `#define`'s `OCCLUSION_CULLING` in the shader.
        const OCCLUSION_CULLING = 2;
        /// Whether this is the early phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `EARLY_PHASE` in the shader.
        const EARLY_PHASE = 4;
    }

    /// Specifies variants of the indirect parameter building shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BuildIndirectParametersPipelineKey: u8 {
        /// Whether the indirect parameter building shader is processing indexed
        /// meshes (those that have index buffers).
        ///
        /// This defines `INDEXED` in the shader.
        const INDEXED = 1;
        /// Whether the GPU and driver supports `multi_draw_indirect_count`.
        ///
        /// This defines `MULTI_DRAW_INDIRECT_COUNT_SUPPORTED` in the shader.
        const MULTI_DRAW_INDIRECT_COUNT_SUPPORTED = 2;
        /// Whether GPU two-phase occlusion culling is in use.
        ///
        /// This `#define`'s `OCCLUSION_CULLING` in the shader.
        const OCCLUSION_CULLING = 4;
        /// Whether this is the early phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `EARLY_PHASE` in the shader.
        const EARLY_PHASE = 8;
        /// Whether this is the late phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `LATE_PHASE` in the shader.
        const LATE_PHASE = 16;
        /// Whether this is the phase that runs after the early and late phases,
        /// and right before the main drawing logic, when GPU two-phase
        /// occlusion culling is in use.
        ///
        /// This `#define`'s `MAIN_PHASE` in the shader.
        const MAIN_PHASE = 32;
    }
}

/// The compute shader bind group for the mesh preprocessing pass for each
/// render phase.
///
/// This goes on the view. It maps the [`core::any::TypeId`] of a render phase
/// (e.g.  [`bevy_core_pipeline::core_3d::Opaque3d`]) to the
/// [`PhasePreprocessBindGroups`] for that phase.
#[derive(Component, Clone, Deref, DerefMut)]
pub struct PreprocessBindGroups(pub TypeIdMap<PhasePreprocessBindGroups>);

/// The compute shader bind group for the mesh preprocessing step for a single
/// render phase on a single view.
#[derive(Clone)]
pub enum PhasePreprocessBindGroups {
    /// The bind group used for the single invocation of the compute shader when
    /// indirect drawing is *not* being used.
    ///
    /// Because direct drawing doesn't require splitting the meshes into indexed
    /// and non-indexed meshes, there's only one bind group in this case.
    Direct(BindGroup),

    /// The bind groups used for the compute shader when indirect drawing is
    /// being used, but occlusion culling isn't being used.
    ///
    /// Because indirect drawing requires splitting the meshes into indexed and
    /// non-indexed meshes, there are two bind groups here.
    IndirectFrustumCulling {
        /// The bind group for indexed meshes.
        indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes.
        non_indexed: Option<BindGroup>,
    },

    /// The bind groups used for the compute shader when indirect drawing is
    /// being used, but occlusion culling isn't being used.
    ///
    /// Because indirect drawing requires splitting the meshes into indexed and
    /// non-indexed meshes, and because occlusion culling requires splitting
    /// this phase into early and late versions, there are four bind groups
    /// here.
    IndirectOcclusionCulling {
        /// The bind group for indexed meshes during the early mesh
        /// preprocessing phase.
        early_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the early mesh
        /// preprocessing phase.
        early_non_indexed: Option<BindGroup>,
        /// The bind group for indexed meshes during the late mesh preprocessing
        /// phase.
        late_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the late mesh
        /// preprocessing phase.
        late_non_indexed: Option<BindGroup>,
    },
}

/// The bind groups for the compute shaders that reset indirect draw counts and
/// build indirect parameters.
///
/// There's one set of bind group for each phase. Phases are keyed off their
/// [`core::any::TypeId`].
#[derive(Resource, Default, Deref, DerefMut)]
pub struct BuildIndirectParametersBindGroups(pub TypeIdMap<PhaseBuildIndirectParametersBindGroups>);

impl BuildIndirectParametersBindGroups {
    /// Creates a new, empty [`BuildIndirectParametersBindGroups`] table.
    pub fn new() -> BuildIndirectParametersBindGroups {
        Self::default()
    }
}

/// The per-phase set of bind groups for the compute shaders that reset indirect
/// draw counts and build indirect parameters.
pub struct PhaseBuildIndirectParametersBindGroups {
    /// The bind group for the `reset_indirect_batch_sets.wgsl` shader, for
    /// indexed meshes.
    pub reset_indexed_indirect_batch_sets: Option<BindGroup>,
    /// The bind group for the `reset_indirect_batch_sets.wgsl` shader, for
    /// non-indexed meshes.
    pub reset_non_indexed_indirect_batch_sets: Option<BindGroup>,
    /// The bind group for the `build_indirect_params.wgsl` shader, for indexed
    /// meshes.
    pub build_indexed_indirect: Option<BindGroup>,
    /// The bind group for the `build_indirect_params.wgsl` shader, for
    /// non-indexed meshes.
    pub build_non_indexed_indirect: Option<BindGroup>,
}

/// Stops the `GpuPreprocessNode` attempting to generate the buffer for this view
/// useful to avoid duplicating effort if the bind group is shared between views
#[derive(Component, Default)]
pub struct SkipGpuPreprocess;

impl Node for ClearIndirectParametersMetadataNode {
    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(indirect_parameters_buffers) = world.get_resource::<IndirectParametersBuffers>()
        else {
            return Ok(());
        };

        // Clear out each indexed and non-indexed GPU-side buffer.
        for phase_indirect_parameters_buffers in indirect_parameters_buffers.values() {
            if let Some(indexed_gpu_metadata_buffer) = phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer()
            {
                render_context.command_encoder().clear_buffer(
                    indexed_gpu_metadata_buffer,
                    0,
                    Some(
                        phase_indirect_parameters_buffers.indexed.batch_count() as u64
                            * size_of::<IndirectParametersGpuMetadata>() as u64,
                    ),
                );
            }

            if let Some(non_indexed_gpu_metadata_buffer) = phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer()
            {
                render_context.command_encoder().clear_buffer(
                    non_indexed_gpu_metadata_buffer,
                    0,
                    Some(
                        phase_indirect_parameters_buffers.non_indexed.batch_count() as u64
                            * size_of::<IndirectParametersGpuMetadata>() as u64,
                    ),
                );
            }
        }

        Ok(())
    }
}

impl FromWorld for EarlyGpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
            main_view_query: QueryState::new(world),
        }
    }
}

impl Node for EarlyGpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
        self.main_view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Grab the [`BatchedInstanceBuffers`].
        let batched_instance_buffers =
            world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("early mesh preprocessing"),
                    timestamp_writes: None,
                });

        let mut all_views: SmallVec<[_; 8]> = SmallVec::new();
        all_views.push(graph.view_entity());
        if let Ok(shadow_cascade_views) =
            self.main_view_query.get_manual(world, graph.view_entity())
        {
            all_views.extend(shadow_cascade_views.lights.iter().copied());
        }

        // Run the compute passes.

        for view_entity in all_views {
            let Ok((
                view,
                bind_groups,
                view_uniform_offset,
                no_indirect_drawing,
                occlusion_culling,
            )) = self.view_query.get_manual(world, view_entity)
            else {
                continue;
            };

            let Some(bind_groups) = bind_groups else {
                continue;
            };
            let Some(view_uniform_offset) = view_uniform_offset else {
                continue;
            };

            // Select the right pipeline, depending on whether GPU culling is in
            // use.
            let maybe_pipeline_id = if no_indirect_drawing {
                preprocess_pipelines.direct_preprocess.pipeline_id
            } else if occlusion_culling {
                preprocess_pipelines
                    .early_gpu_occlusion_culling_preprocess
                    .pipeline_id
            } else {
                preprocess_pipelines
                    .gpu_frustum_culling_preprocess
                    .pipeline_id
            };

            // Fetch the pipeline.
            let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
                warn!("The build mesh uniforms pipeline wasn't ready");
                continue;
            };

            let Some(preprocess_pipeline) =
                pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
            else {
                // This will happen while the pipeline is being compiled and is fine.
                continue;
            };

            compute_pass.set_pipeline(preprocess_pipeline);

            // Loop over each render phase.
            for (phase_type_id, batched_phase_instance_buffers) in
                &batched_instance_buffers.phase_instance_buffers
            {
                // Grab the work item buffers for this view.
                let Some(work_item_buffers) = batched_phase_instance_buffers
                    .work_item_buffers
                    .get(&view.retained_view_entity)
                else {
                    continue;
                };

                // Fetch the bind group for the render phase.
                let Some(phase_bind_groups) = bind_groups.get(phase_type_id) else {
                    continue;
                };

                // Make sure the mesh preprocessing shader has access to the
                // view info it needs to do culling and motion vector
                // computation.
                let dynamic_offsets = [view_uniform_offset.offset];

                // Are we drawing directly or indirectly?
                match *phase_bind_groups {
                    PhasePreprocessBindGroups::Direct(ref bind_group) => {
                        // Invoke the mesh preprocessing shader to transform
                        // meshes only, but not cull.
                        let PreprocessWorkItemBuffers::Direct(work_item_buffer) = work_item_buffers
                        else {
                            continue;
                        };
                        compute_pass.set_bind_group(0, bind_group, &dynamic_offsets);
                        let workgroup_count = work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }

                    PhasePreprocessBindGroups::IndirectFrustumCulling {
                        indexed: ref maybe_indexed_bind_group,
                        non_indexed: ref maybe_non_indexed_bind_group,
                    }
                    | PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        early_indexed: ref maybe_indexed_bind_group,
                        early_non_indexed: ref maybe_non_indexed_bind_group,
                        ..
                    } => {
                        // Invoke the mesh preprocessing shader to transform and
                        // cull the meshes.
                        let PreprocessWorkItemBuffers::Indirect {
                            indexed: indexed_buffer,
                            non_indexed: non_indexed_buffer,
                            ..
                        } = work_item_buffers
                        else {
                            continue;
                        };

                        // Transform and cull indexed meshes if there are any.
                        if let Some(indexed_bind_group) = maybe_indexed_bind_group {
                            if let PreprocessWorkItemBuffers::Indirect {
                                gpu_occlusion_culling:
                                    Some(GpuOcclusionCullingWorkItemBuffers {
                                        late_indirect_parameters_indexed_offset,
                                        ..
                                    }),
                                ..
                            } = *work_item_buffers
                            {
                                compute_pass.set_push_constants(
                                    0,
                                    bytemuck::bytes_of(&late_indirect_parameters_indexed_offset),
                                );
                            }

                            compute_pass.set_bind_group(0, indexed_bind_group, &dynamic_offsets);
                            let workgroup_count = indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                            }
                        }

                        // Transform and cull non-indexed meshes if there are any.
                        if let Some(non_indexed_bind_group) = maybe_non_indexed_bind_group {
                            if let PreprocessWorkItemBuffers::Indirect {
                                gpu_occlusion_culling:
                                    Some(GpuOcclusionCullingWorkItemBuffers {
                                        late_indirect_parameters_non_indexed_offset,
                                        ..
                                    }),
                                ..
                            } = *work_item_buffers
                            {
                                compute_pass.set_push_constants(
                                    0,
                                    bytemuck::bytes_of(
                                        &late_indirect_parameters_non_indexed_offset,
                                    ),
                                );
                            }

                            compute_pass.set_bind_group(
                                0,
                                non_indexed_bind_group,
                                &dynamic_offsets,
                            );
                            let workgroup_count = non_indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl FromWorld for EarlyPrepassBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for LatePrepassBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for MainBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for LateGpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for LateGpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // Grab the [`BatchedInstanceBuffers`].
        let batched_instance_buffers =
            world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("late mesh preprocessing"),
                    timestamp_writes: None,
                });

        // Run the compute passes.
        for (view, bind_groups, view_uniform_offset) in self.view_query.iter_manual(world) {
            let maybe_pipeline_id = preprocess_pipelines
                .late_gpu_occlusion_culling_preprocess
                .pipeline_id;

            // Fetch the pipeline.
            let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
                warn!("The build mesh uniforms pipeline wasn't ready");
                return Ok(());
            };

            let Some(preprocess_pipeline) =
                pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
            else {
                // This will happen while the pipeline is being compiled and is fine.
                return Ok(());
            };

            compute_pass.set_pipeline(preprocess_pipeline);

            // Loop over each phase. Because we built the phases in parallel,
            // each phase has a separate set of instance buffers.
            for (phase_type_id, batched_phase_instance_buffers) in
                &batched_instance_buffers.phase_instance_buffers
            {
                let UntypedPhaseBatchedInstanceBuffers {
                    ref work_item_buffers,
                    ref late_indexed_indirect_parameters_buffer,
                    ref late_non_indexed_indirect_parameters_buffer,
                    ..
                } = *batched_phase_instance_buffers;

                // Grab the work item buffers for this view.
                let Some(phase_work_item_buffers) =
                    work_item_buffers.get(&view.retained_view_entity)
                else {
                    continue;
                };

                let (
                    PreprocessWorkItemBuffers::Indirect {
                        gpu_occlusion_culling:
                            Some(GpuOcclusionCullingWorkItemBuffers {
                                late_indirect_parameters_indexed_offset,
                                late_indirect_parameters_non_indexed_offset,
                                ..
                            }),
                        ..
                    },
                    Some(PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        late_indexed: maybe_late_indexed_bind_group,
                        late_non_indexed: maybe_late_non_indexed_bind_group,
                        ..
                    }),
                    Some(late_indexed_indirect_parameters_buffer),
                    Some(late_non_indexed_indirect_parameters_buffer),
                ) = (
                    phase_work_item_buffers,
                    bind_groups.get(phase_type_id),
                    late_indexed_indirect_parameters_buffer.buffer(),
                    late_non_indexed_indirect_parameters_buffer.buffer(),
                )
                else {
                    continue;
                };

                let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
                dynamic_offsets.push(view_uniform_offset.offset);

                // If there's no space reserved for work items, then don't
                // bother doing the dispatch, as there can't possibly be any
                // meshes of the given class (indexed or non-indexed) in this
                // phase.

                // Transform and cull indexed meshes if there are any.
                if let Some(late_indexed_bind_group) = maybe_late_indexed_bind_group {
                    compute_pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(late_indirect_parameters_indexed_offset),
                    );

                    compute_pass.set_bind_group(0, late_indexed_bind_group, &dynamic_offsets);
                    compute_pass.dispatch_workgroups_indirect(
                        late_indexed_indirect_parameters_buffer,
                        (*late_indirect_parameters_indexed_offset as u64)
                            * (size_of::<LatePreprocessWorkItemIndirectParameters>() as u64),
                    );
                }

                // Transform and cull non-indexed meshes if there are any.
                if let Some(late_non_indexed_bind_group) = maybe_late_non_indexed_bind_group {
                    compute_pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(late_indirect_parameters_non_indexed_offset),
                    );

                    compute_pass.set_bind_group(0, late_non_indexed_bind_group, &dynamic_offsets);
                    compute_pass.dispatch_workgroups_indirect(
                        late_non_indexed_indirect_parameters_buffer,
                        (*late_indirect_parameters_non_indexed_offset as u64)
                            * (size_of::<LatePreprocessWorkItemIndirectParameters>() as u64),
                    );
                }
            }
        }

        Ok(())
    }
}

impl Node for EarlyPrepassBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        // If there are no views with a depth prepass enabled, we don't need to
        // run this.
        if self.view_query.iter_manual(world).next().is_none() {
            return Ok(());
        }

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.early_phase,
            "early prepass indirect parameters building",
        )
    }
}

impl Node for LatePrepassBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        // If there are no views with occlusion culling enabled, we don't need
        // to run this.
        if self.view_query.iter_manual(world).next().is_none() {
            return Ok(());
        }

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.late_phase,
            "late prepass indirect parameters building",
        )
    }
}

impl Node for MainBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.main_phase,
            "main indirect parameters building",
        )
    }
}

impl PreprocessPipelines {
    /// Returns true if the preprocessing and indirect parameters pipelines have
    /// been loaded or false otherwise.
    pub(crate) fn pipelines_are_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.direct_preprocess.is_loaded(pipeline_cache)
            && self
                .gpu_frustum_culling_preprocess
                .is_loaded(pipeline_cache)
            && self
                .early_gpu_occlusion_culling_preprocess
                .is_loaded(pipeline_cache)
            && self
                .late_gpu_occlusion_culling_preprocess
                .is_loaded(pipeline_cache)
            && self
                .gpu_frustum_culling_build_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self
                .gpu_frustum_culling_build_non_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self.early_phase.is_loaded(pipeline_cache)
            && self.late_phase.is_loaded(pipeline_cache)
            && self.main_phase.is_loaded(pipeline_cache)
    }
}

impl PreprocessPhasePipelines {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.reset_indirect_batch_sets.is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_non_indexed_indirect_params
                .is_loaded(pipeline_cache)
    }
}

impl PreprocessPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl BuildIndirectParametersPipeline {
    /// Returns true if this pipeline has been loaded into the pipeline cache or
    /// false otherwise.
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl SpecializedComputePipeline for PreprocessPipeline {
    type Key = PreprocessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec!["WRITE_INDIRECT_PARAMETERS_METADATA".into()];
        if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
            shader_defs.push("INDIRECT".into());
            shader_defs.push("FRUSTUM_CULLING".into());
        }
        if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
            if key.contains(PreprocessPipelineKey::EARLY_PHASE) {
                shader_defs.push("EARLY_PHASE".into());
            } else {
                shader_defs.push("LATE_PHASE".into());
            }
        }

        ComputePipelineDescriptor {
            label: Some(
                format!(
                    "mesh preprocessing ({})",
                    if key.contains(
                        PreprocessPipelineKey::OCCLUSION_CULLING
                            | PreprocessPipelineKey::EARLY_PHASE
                    ) {
                        "early GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                        "late GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
                        "GPU frustum culling"
                    } else {
                        "direct"
                    }
                )
                .into(),
            ),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..4,
                }]
            } else {
                vec![]
            },
            shader: MESH_PREPROCESS_SHADER_HANDLE,
            shader_defs,
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl FromWorld for PreprocessPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // GPU culling bind group parameters are a superset of those in the CPU
        // culling (direct) shader.
        let direct_bind_group_layout_entries = preprocess_direct_bind_group_layout_entries();
        let gpu_frustum_culling_bind_group_layout_entries = gpu_culling_bind_group_layout_entries();
        let gpu_early_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries().extend_with_indices(((
                11,
                binding_types::storage_buffer::<PreprocessWorkItem>(
                    /*has_dynamic_offset=*/ false,
                ),
            ),));
        let gpu_late_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries();

        let reset_indirect_batch_sets_bind_group_layout_entries =
            DynamicBindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (binding_types::storage_buffer::<IndirectBatchSet>(false),),
            );

        // Indexed and non-indexed bind group parameters share all the bind
        // group layout entries except the final one.
        let build_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries().extend_sequential((
                binding_types::storage_buffer::<IndirectParametersIndexed>(false),
            ));
        let build_non_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries().extend_sequential((
                binding_types::storage_buffer::<IndirectParametersNonIndexed>(false),
            ));

        // Create the bind group layouts.
        let direct_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms direct bind group layout",
            &direct_bind_group_layout_entries,
        );
        let gpu_frustum_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU frustum culling bind group layout",
            &gpu_frustum_culling_bind_group_layout_entries,
        );
        let gpu_early_occlusion_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU early occlusion culling bind group layout",
            &gpu_early_occlusion_culling_bind_group_layout_entries,
        );
        let gpu_late_occlusion_culling_bind_group_layout = render_device.create_bind_group_layout(
            "build mesh uniforms GPU late occlusion culling bind group layout",
            &gpu_late_occlusion_culling_bind_group_layout_entries,
        );
        let reset_indirect_batch_sets_bind_group_layout = render_device.create_bind_group_layout(
            "reset indirect batch sets bind group layout",
            &reset_indirect_batch_sets_bind_group_layout_entries,
        );
        let build_indexed_indirect_params_bind_group_layout = render_device
            .create_bind_group_layout(
                "build indexed indirect parameters bind group layout",
                &build_indexed_indirect_params_bind_group_layout_entries,
            );
        let build_non_indexed_indirect_params_bind_group_layout = render_device
            .create_bind_group_layout(
                "build non-indexed indirect parameters bind group layout",
                &build_non_indexed_indirect_params_bind_group_layout_entries,
            );

        let preprocess_phase_pipelines = PreprocessPhasePipelines {
            reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline {
                bind_group_layout: reset_indirect_batch_sets_bind_group_layout.clone(),
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    pipeline_id: None,
                },
        };

        PreprocessPipelines {
            direct_preprocess: PreprocessPipeline {
                bind_group_layout: direct_bind_group_layout,
                pipeline_id: None,
            },
            gpu_frustum_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_frustum_culling_bind_group_layout,
                pipeline_id: None,
            },
            early_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_early_occlusion_culling_bind_group_layout,
                pipeline_id: None,
            },
            late_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_late_occlusion_culling_bind_group_layout,
                pipeline_id: None,
            },
            gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                pipeline_id: None,
            },
            gpu_frustum_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    pipeline_id: None,
                },
            early_phase: preprocess_phase_pipelines.clone(),
            late_phase: preprocess_phase_pipelines.clone(),
            main_phase: preprocess_phase_pipelines.clone(),
        }
    }
}
impl PreprocessPipeline {
    pub(crate) fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<PreprocessPipeline>,
        key: PreprocessPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let preprocess_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(preprocess_pipeline_id);
    }
}

impl SpecializedComputePipeline for ResetIndirectBatchSetsPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("reset indirect batch sets".into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl SpecializedComputePipeline for BuildIndirectParametersPipeline {
    type Key = BuildIndirectParametersPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
            shader_defs.push("INDEXED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED) {
            shader_defs.push("MULTI_DRAW_INDIRECT_COUNT_SUPPORTED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
            shader_defs.push("EARLY_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
            shader_defs.push("LATE_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MAIN_PHASE) {
            shader_defs.push("MAIN_PHASE".into());
        }

        let label = format!(
            "{} build {}indexed indirect parameters",
            if !key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
                "frustum culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
                "early occlusion culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
                "late occlusion culling"
            } else {
                "main occlusion culling"
            },
            if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
                ""
            } else {
                "non-"
            }
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: vec![],
            shader: BUILD_INDIRECT_PARAMS_SHADER_HANDLE,
            shader_defs,
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

impl ResetIndirectBatchSetsPipeline {
    pub(crate) fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let reset_indirect_batch_sets_pipeline_id = pipelines.specialize(pipeline_cache, self, ());
        self.pipeline_id = Some(reset_indirect_batch_sets_pipeline_id);
    }
}

impl BuildIndirectParametersPipeline {
    pub(crate) fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<BuildIndirectParametersPipeline>,
        key: BuildIndirectParametersPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let build_indirect_parameters_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(build_indirect_parameters_pipeline_id);
    }
}

/// A temporary structure that stores all the information needed to construct
/// bind groups for the mesh preprocessing shader.
pub(super) struct PreprocessBindGroupBuilder<'a> {
    /// The render-world entity corresponding to the current view.
    pub view: Entity,
    /// The indirect compute dispatch parameters buffer for indexed meshes in
    /// the late prepass.
    pub late_indexed_indirect_parameters_buffer:
        &'a RawBufferVec<LatePreprocessWorkItemIndirectParameters>,
    /// The indirect compute dispatch parameters buffer for non-indexed meshes
    /// in the late prepass.
    pub late_non_indexed_indirect_parameters_buffer:
        &'a RawBufferVec<LatePreprocessWorkItemIndirectParameters>,
    /// The device.
    pub render_device: &'a RenderDevice,
    /// The buffers that store indirect draw parameters.
    pub phase_indirect_parameters_buffers: &'a UntypedPhaseIndirectParametersBuffers,
    /// The GPU buffer that stores the information needed to cull each mesh.
    pub mesh_culling_data_buffer: &'a MeshCullingDataBuffer,
    /// The GPU buffer that stores information about the view.
    pub view_uniforms: &'a ViewUniforms,
    /// The GPU buffer that stores information about the view from last frame.
    pub previous_view_uniforms: &'a PreviousViewUniforms,
    /// The pipelines for the mesh preprocessing shader.
    pub pipelines: &'a PreprocessPipelines,
    /// The GPU buffer containing the list of [`MeshInputUniform`]s for the
    /// current frame.
    pub current_input_buffer: &'a Buffer,
    /// The GPU buffer containing the list of [`MeshInputUniform`]s for the
    /// previous frame.
    pub previous_input_buffer: &'a Buffer,
    /// The GPU buffer containing the list of [`MeshUniform`]s for the current
    /// frame.
    ///
    /// This is the buffer containing the mesh's final transforms that the
    /// shaders will write to.
    pub data_buffer: &'a Buffer,
}

impl<'a> PreprocessBindGroupBuilder<'a> {
    /// Creates the bind groups for mesh preprocessing when GPU frustum culling
    /// and GPU occlusion culling are both disabled.
    pub(crate) fn create_direct_preprocess_bind_groups(
        &self,
        work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<PhasePreprocessBindGroups> {
        // Don't use `as_entire_binding()` here; the shader reads the array
        // length and the underlying buffer may be longer than the actual size
        // of the vector.
        let work_item_buffer_size = NonZero::<u64>::try_from(
            work_item_buffer.len() as u64 * u64::from(PreprocessWorkItem::min_size()),
        )
        .ok();

        Some(PhasePreprocessBindGroups::Direct(
            self.render_device.create_bind_group(
                "preprocess_direct_bind_group",
                &self.pipelines.direct_preprocess.bind_group_layout,
                &BindGroupEntries::with_indices((
                    (0, self.view_uniforms.uniforms.binding()?),
                    (3, self.current_input_buffer.as_entire_binding()),
                    (4, self.previous_input_buffer.as_entire_binding()),
                    (
                        5,
                        BindingResource::Buffer(BufferBinding {
                            buffer: work_item_buffer.buffer()?,
                            offset: 0,
                            size: work_item_buffer_size,
                        }),
                    ),
                    (6, self.data_buffer.as_entire_binding()),
                )),
            ),
        ))
    }

    /// Creates the bind groups for mesh preprocessing when GPU occlusion
    /// culling is enabled.
    pub(crate) fn create_indirect_occlusion_culling_preprocess_bind_groups(
        &self,
        view_depth_pyramids: &Query<(&ViewDepthPyramid, &PreviousViewUniformOffset)>,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        gpu_occlusion_culling_work_item_buffers: &GpuOcclusionCullingWorkItemBuffers,
    ) -> Option<PhasePreprocessBindGroups> {
        let GpuOcclusionCullingWorkItemBuffers {
            late_indexed: ref late_indexed_work_item_buffer,
            late_non_indexed: ref late_non_indexed_work_item_buffer,
            ..
        } = *gpu_occlusion_culling_work_item_buffers;

        let (view_depth_pyramid, previous_view_uniform_offset) =
            view_depth_pyramids.get(self.view).ok()?;

        Some(PhasePreprocessBindGroups::IndirectOcclusionCulling {
            early_indexed: self.create_indirect_occlusion_culling_early_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                indexed_work_item_buffer,
                late_indexed_work_item_buffer,
            ),

            early_non_indexed: self.create_indirect_occlusion_culling_early_non_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                non_indexed_work_item_buffer,
                late_non_indexed_work_item_buffer,
            ),

            late_indexed: self.create_indirect_occlusion_culling_late_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                late_indexed_work_item_buffer,
            ),

            late_non_indexed: self.create_indirect_occlusion_culling_late_non_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                late_non_indexed_work_item_buffer,
            ),
        })
    }

    /// Creates the bind group for the first phase of mesh preprocessing of
    /// indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_early_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        late_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer(),
            indexed_work_item_buffer.buffer(),
            late_indexed_work_item_buffer.buffer(),
            self.late_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(indexed_cpu_metadata_buffer),
                Some(indexed_gpu_metadata_buffer),
                Some(indexed_work_item_gpu_buffer),
                Some(late_indexed_work_item_gpu_buffer),
                Some(late_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_early_indexed_gpu_occlusion_culling_bind_group",
                        &self
                            .pipelines
                            .early_gpu_occlusion_culling_preprocess
                            .bind_group_layout,
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                11,
                                BufferBinding {
                                    buffer: late_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the first phase of mesh preprocessing of
    /// non-indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_early_non_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        late_non_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer(),
            non_indexed_work_item_buffer.buffer(),
            late_non_indexed_work_item_buffer.buffer(),
            self.late_non_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(non_indexed_cpu_metadata_buffer),
                Some(non_indexed_gpu_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
                Some(late_non_indexed_work_item_buffer),
                Some(late_non_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_early_non_indexed_gpu_occlusion_culling_bind_group",
                        &self
                            .pipelines
                            .early_gpu_occlusion_culling_preprocess
                            .bind_group_layout,
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, non_indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, non_indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                11,
                                BufferBinding {
                                    buffer: late_non_indexed_work_item_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_non_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_non_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the second phase of mesh preprocessing of
    /// indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_late_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        late_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer(),
            late_indexed_work_item_buffer.buffer(),
            self.late_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(indexed_cpu_metadata_buffer),
                Some(indexed_gpu_metadata_buffer),
                Some(late_indexed_work_item_gpu_buffer),
                Some(late_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let late_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    late_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_late_indexed_gpu_occlusion_culling_bind_group",
                        &self
                            .pipelines
                            .late_gpu_occlusion_culling_preprocess
                            .bind_group_layout,
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: late_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: late_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the second phase of mesh preprocessing of
    /// non-indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_late_non_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        late_non_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer(),
            late_non_indexed_work_item_buffer.buffer(),
            self.late_non_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(non_indexed_cpu_metadata_buffer),
                Some(non_indexed_gpu_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
                Some(late_non_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    late_non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_late_non_indexed_gpu_occlusion_culling_bind_group",
                        &self
                            .pipelines
                            .late_gpu_occlusion_culling_preprocess
                            .bind_group_layout,
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, non_indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, non_indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_non_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_non_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind groups for mesh preprocessing when GPU frustum culling
    /// is enabled, but GPU occlusion culling is disabled.
    pub(crate) fn create_indirect_frustum_culling_preprocess_bind_groups(
        &self,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<PhasePreprocessBindGroups> {
        Some(PhasePreprocessBindGroups::IndirectFrustumCulling {
            indexed: self
                .create_indirect_frustum_culling_indexed_bind_group(indexed_work_item_buffer),
            non_indexed: self.create_indirect_frustum_culling_non_indexed_bind_group(
                non_indexed_work_item_buffer,
            ),
        })
    }

    /// Creates the bind group for mesh preprocessing of indexed meshes when GPU
    /// frustum culling is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_indexed_bind_group(
        &self,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer(),
            indexed_work_item_buffer.buffer(),
        ) {
            (
                Some(indexed_cpu_metadata_buffer),
                Some(indexed_gpu_metadata_buffer),
                Some(indexed_work_item_gpu_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_gpu_indexed_frustum_culling_bind_group",
                        &self
                            .pipelines
                            .gpu_frustum_culling_preprocess
                            .bind_group_layout,
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for mesh preprocessing of non-indexed meshes when
    /// GPU frustum culling is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_non_indexed_bind_group(
        &self,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer(),
            non_indexed_work_item_buffer.buffer(),
        ) {
            (
                Some(non_indexed_cpu_metadata_buffer),
                Some(non_indexed_gpu_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_gpu_non_indexed_frustum_culling_bind_group",
                        &self
                            .pipelines
                            .gpu_frustum_culling_preprocess
                            .bind_group_layout,
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, non_indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, non_indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }
}

fn run_build_indirect_parameters_node(
    render_context: &mut RenderContext,
    world: &World,
    preprocess_phase_pipelines: &PreprocessPhasePipelines,
    label: &'static str,
) -> Result<(), NodeRunError> {
    let Some(build_indirect_params_bind_groups) =
        world.get_resource::<BuildIndirectParametersBindGroups>()
    else {
        return Ok(());
    };

    let pipeline_cache = world.resource::<PipelineCache>();
    let indirect_parameters_buffers = world.resource::<IndirectParametersBuffers>();

    let mut compute_pass =
        render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });

    // Fetch the pipeline.
    let (
        Some(reset_indirect_batch_sets_pipeline_id),
        Some(build_indexed_indirect_params_pipeline_id),
        Some(build_non_indexed_indirect_params_pipeline_id),
    ) = (
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .pipeline_id,
    )
    else {
        warn!("The build indirect parameters pipelines weren't ready");
        return Ok(());
    };

    let (
        Some(reset_indirect_batch_sets_pipeline),
        Some(build_indexed_indirect_params_pipeline),
        Some(build_non_indexed_indirect_params_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(reset_indirect_batch_sets_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_indexed_indirect_params_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_non_indexed_indirect_params_pipeline_id),
    )
    else {
        // This will happen while the pipeline is being compiled and is fine.
        return Ok(());
    };

    // Loop over each phase. As each has as separate set of buffers, we need to
    // build indirect parameters individually for each phase.
    for (phase_type_id, phase_build_indirect_params_bind_groups) in
        build_indirect_params_bind_groups.iter()
    {
        let Some(phase_indirect_parameters_buffers) =
            indirect_parameters_buffers.get(phase_type_id)
        else {
            continue;
        };

        // Build indexed indirect parameters.
        if let (
            Some(reset_indexed_indirect_batch_sets_bind_group),
            Some(build_indirect_indexed_params_bind_group),
        ) = (
            &phase_build_indirect_params_bind_groups.reset_indexed_indirect_batch_sets,
            &phase_build_indirect_params_bind_groups.build_indexed_indirect,
        ) {
            compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
            compute_pass.set_bind_group(0, reset_indexed_indirect_batch_sets_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .batch_set_count(true)
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            compute_pass.set_pipeline(build_indexed_indirect_params_pipeline);
            compute_pass.set_bind_group(0, build_indirect_indexed_params_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .indexed
                .batch_count()
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }
        }

        // Build non-indexed indirect parameters.
        if let (
            Some(reset_non_indexed_indirect_batch_sets_bind_group),
            Some(build_indirect_non_indexed_params_bind_group),
        ) = (
            &phase_build_indirect_params_bind_groups.reset_non_indexed_indirect_batch_sets,
            &phase_build_indirect_params_bind_groups.build_non_indexed_indirect,
        ) {
            compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
            compute_pass.set_bind_group(0, reset_non_indexed_indirect_batch_sets_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .batch_set_count(false)
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            compute_pass.set_pipeline(build_non_indexed_indirect_params_pipeline);
            compute_pass.set_bind_group(0, build_indirect_non_indexed_params_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .non_indexed
                .batch_count()
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }
        }
    }

    Ok(())
}

fn preprocess_direct_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            // `view`
            (
                0,
                binding_types::uniform_buffer::<ViewUniform>(/* has_dynamic_offset= */ true),
            ),
            // `current_input`
            (
                3,
                binding_types::storage_buffer_read_only::<MeshInputUniform>(false),
            ),
            // `previous_input`
            (
                4,
                binding_types::storage_buffer_read_only::<MeshInputUniform>(false),
            ),
            // `indices`
            (
                5,
                binding_types::storage_buffer_read_only::<PreprocessWorkItem>(false),
            ),
            // `output`
            (6, binding_types::storage_buffer::<MeshUniform>(false)),
        ),
    )
}

// Returns the first 4 bind group layout entries shared between all invocations
// of the indirect parameters building shader.
fn build_indirect_params_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            (
                0,
                binding_types::storage_buffer_read_only::<MeshInputUniform>(false),
            ),
            (
                1,
                binding_types::storage_buffer_read_only::<IndirectParametersCpuMetadata>(false),
            ),
            (
                2,
                binding_types::storage_buffer_read_only::<IndirectParametersGpuMetadata>(false),
            ),
            (3, binding_types::storage_buffer::<IndirectBatchSet>(false)),
        ),
    )
}

/// A system that specializes the `mesh_preprocess.wgsl` and
/// `build_indirect_params.wgsl` pipelines if necessary.
fn gpu_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    // GPU culling bind group parameters are a superset of those in the CPU
    // culling (direct) shader.
    preprocess_direct_bind_group_layout_entries().extend_with_indices((
        // `indirect_parameters_cpu_metadata`
        (
            7,
            binding_types::storage_buffer_read_only::<IndirectParametersCpuMetadata>(
                /* has_dynamic_offset= */ false,
            ),
        ),
        // `indirect_parameters_gpu_metadata`
        (
            8,
            binding_types::storage_buffer::<IndirectParametersGpuMetadata>(
                /* has_dynamic_offset= */ false,
            ),
        ),
        // `mesh_culling_data`
        (
            9,
            binding_types::storage_buffer_read_only::<MeshCullingData>(
                /* has_dynamic_offset= */ false,
            ),
        ),
    ))
}

fn gpu_occlusion_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    gpu_culling_bind_group_layout_entries().extend_with_indices((
        (
            2,
            binding_types::uniform_buffer::<PreviousViewData>(/*has_dynamic_offset=*/ false),
        ),
        (
            10,
            binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (
            12,
            binding_types::storage_buffer::<LatePreprocessWorkItemIndirectParameters>(
                /*has_dynamic_offset=*/ false,
            ),
        ),
    ))
}
