use core::num::NonZeroU64;

use bevy_core_pipeline::{
    experimental::mip_generation::ViewDepthPyramid,
    prepass::{PreviousViewUniformOffset, PreviousViewUniforms},
};
use bevy_ecs::{
    entity::Entity,
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    batching::gpu_preprocessing::{
        BatchedInstanceBuffers, IndirectParametersBuffers, IndirectParametersCpuMetadata,
        IndirectParametersGpuMetadata, PreprocessWorkItemBuffers,
        UntypedPhaseBatchedInstanceBuffers,
    },
    render_resource::{
        BindGroupEntries, Buffer, BufferBinding, PipelineCache, SpecializedComputePipelines,
    },
    renderer::{RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    view::{ExtractedView, ViewUniforms},
};
use bevy_utils::TypeIdMap;

use crate::mesh_pipeline::render::{MeshCullingDataBuffer, MeshInputUniform, MeshUniform};

use super::render::{
    BuildIndirectParametersBindGroups, BuildIndirectParametersPipeline,
    BuildIndirectParametersPipelineKey, PhaseBuildIndirectParametersBindGroups,
    PreprocessBindGroupBuilder, PreprocessBindGroups, PreprocessPipeline, PreprocessPipelineKey,
    PreprocessPipelines, ResetIndirectBatchSetsPipeline,
};

/// A system that specializes the `mesh_preprocess.wgsl` pipelines if necessary.
pub fn prepare_preprocess_pipelines(
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut specialized_preprocess_pipelines: ResMut<SpecializedComputePipelines<PreprocessPipeline>>,
    mut specialized_reset_indirect_batch_sets_pipelines: ResMut<
        SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    >,
    mut specialized_build_indirect_parameters_pipelines: ResMut<
        SpecializedComputePipelines<BuildIndirectParametersPipeline>,
    >,
    preprocess_pipelines: ResMut<PreprocessPipelines>,
) {
    let preprocess_pipelines = preprocess_pipelines.into_inner();

    preprocess_pipelines.direct_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::empty(),
    );
    preprocess_pipelines.gpu_frustum_culling_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::FRUSTUM_CULLING,
    );
    preprocess_pipelines
        .early_gpu_occlusion_culling_preprocess
        .prepare(
            &pipeline_cache,
            &mut specialized_preprocess_pipelines,
            PreprocessPipelineKey::FRUSTUM_CULLING
                | PreprocessPipelineKey::OCCLUSION_CULLING
                | PreprocessPipelineKey::EARLY_PHASE,
        );
    preprocess_pipelines
        .late_gpu_occlusion_culling_preprocess
        .prepare(
            &pipeline_cache,
            &mut specialized_preprocess_pipelines,
            PreprocessPipelineKey::FRUSTUM_CULLING | PreprocessPipelineKey::OCCLUSION_CULLING,
        );

    let mut build_indirect_parameters_pipeline_key = BuildIndirectParametersPipelineKey::empty();

    // If the GPU and driver support `multi_draw_indirect_count`, tell the
    // shader that.
    if render_device
        .wgpu_device()
        .features()
        .contains(WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT)
    {
        build_indirect_parameters_pipeline_key
            .insert(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED);
    }

    preprocess_pipelines
        .gpu_frustum_culling_build_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key | BuildIndirectParametersPipelineKey::INDEXED,
        );
    preprocess_pipelines
        .gpu_frustum_culling_build_non_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key,
        );

    for (preprocess_phase_pipelines, build_indirect_parameters_phase_pipeline_key) in [
        (
            &mut preprocess_pipelines.early_phase,
            BuildIndirectParametersPipelineKey::EARLY_PHASE,
        ),
        (
            &mut preprocess_pipelines.late_phase,
            BuildIndirectParametersPipelineKey::LATE_PHASE,
        ),
        (
            &mut preprocess_pipelines.main_phase,
            BuildIndirectParametersPipelineKey::MAIN_PHASE,
        ),
    ] {
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .prepare(
                &pipeline_cache,
                &mut specialized_reset_indirect_batch_sets_pipelines,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::INDEXED
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
    }
}

/// A system that attaches the mesh uniform buffers to the bind groups for the
/// variants of the mesh preprocessing compute shader.
#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of arguments"
)]
pub fn prepare_preprocess_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedView)>,
    view_depth_pyramids: Query<(&ViewDepthPyramid, &PreviousViewUniformOffset)>,
    render_device: Res<RenderDevice>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    mesh_culling_data_buffer: Res<MeshCullingDataBuffer>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    pipelines: Res<PreprocessPipelines>,
) {
    // Grab the `BatchedInstanceBuffers`.
    let BatchedInstanceBuffers {
        current_input_buffer: current_input_buffer_vec,
        previous_input_buffer: previous_input_buffer_vec,
        phase_instance_buffers,
    } = batched_instance_buffers.into_inner();

    let (Some(current_input_buffer), Some(previous_input_buffer)) = (
        current_input_buffer_vec.buffer().buffer(),
        previous_input_buffer_vec.buffer().buffer(),
    ) else {
        return;
    };

    // Record whether we have any meshes that are to be drawn indirectly. If we
    // don't, then we can skip building indirect parameters.
    let mut any_indirect = false;

    // Loop over each view.
    for (view_entity, view) in &views {
        let mut bind_groups = TypeIdMap::default();

        // Loop over each phase.
        for (phase_type_id, phase_instance_buffers) in phase_instance_buffers {
            let UntypedPhaseBatchedInstanceBuffers {
                data_buffer: ref data_buffer_vec,
                ref work_item_buffers,
                ref late_indexed_indirect_parameters_buffer,
                ref late_non_indexed_indirect_parameters_buffer,
            } = *phase_instance_buffers;

            let Some(data_buffer) = data_buffer_vec.buffer() else {
                continue;
            };

            // Grab the indirect parameters buffers for this phase.
            let Some(phase_indirect_parameters_buffers) =
                indirect_parameters_buffers.get(phase_type_id)
            else {
                continue;
            };

            let Some(work_item_buffers) = work_item_buffers.get(&view.retained_view_entity) else {
                continue;
            };

            // Create the `PreprocessBindGroupBuilder`.
            let preprocess_bind_group_builder = PreprocessBindGroupBuilder {
                view: view_entity,
                late_indexed_indirect_parameters_buffer,
                late_non_indexed_indirect_parameters_buffer,
                render_device: &render_device,
                phase_indirect_parameters_buffers,
                mesh_culling_data_buffer: &mesh_culling_data_buffer,
                view_uniforms: &view_uniforms,
                previous_view_uniforms: &previous_view_uniforms,
                pipelines: &pipelines,
                current_input_buffer,
                previous_input_buffer,
                data_buffer,
            };

            // Depending on the type of work items we have, construct the
            // appropriate bind groups.
            let (was_indirect, bind_group) = match *work_item_buffers {
                PreprocessWorkItemBuffers::Direct(ref work_item_buffer) => (
                    false,
                    preprocess_bind_group_builder
                        .create_direct_preprocess_bind_groups(work_item_buffer),
                ),

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: Some(ref gpu_occlusion_culling_work_item_buffers),
                } => (
                    true,
                    preprocess_bind_group_builder
                        .create_indirect_occlusion_culling_preprocess_bind_groups(
                            &view_depth_pyramids,
                            indexed_work_item_buffer,
                            non_indexed_work_item_buffer,
                            gpu_occlusion_culling_work_item_buffers,
                        ),
                ),

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: None,
                } => (
                    true,
                    preprocess_bind_group_builder
                        .create_indirect_frustum_culling_preprocess_bind_groups(
                            indexed_work_item_buffer,
                            non_indexed_work_item_buffer,
                        ),
                ),
            };

            // Write that bind group in.
            if let Some(bind_group) = bind_group {
                any_indirect = any_indirect || was_indirect;
                bind_groups.insert(*phase_type_id, bind_group);
            }
        }

        // Save the bind groups.
        commands
            .entity(view_entity)
            .insert(PreprocessBindGroups(bind_groups));
    }

    // Now, if there were any indirect draw commands, create the bind groups for
    // the indirect parameters building shader.
    if any_indirect {
        create_build_indirect_parameters_bind_groups(
            &mut commands,
            &render_device,
            &pipelines,
            current_input_buffer,
            &indirect_parameters_buffers,
        );
    }
}

/// Writes the information needed to do GPU mesh culling to the GPU.
pub fn write_mesh_culling_data_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut mesh_culling_data_buffer: ResMut<MeshCullingDataBuffer>,
) {
    mesh_culling_data_buffer.write_buffer(&render_device, &render_queue);
}

/// A system that creates bind groups from the indirect parameters metadata and
/// data buffers for the indirect batch set reset shader and the indirect
/// parameter building shader.
fn create_build_indirect_parameters_bind_groups(
    commands: &mut Commands,
    render_device: &RenderDevice,
    pipelines: &PreprocessPipelines,
    current_input_buffer: &Buffer,
    indirect_parameters_buffers: &IndirectParametersBuffers,
) {
    let mut build_indirect_parameters_bind_groups = BuildIndirectParametersBindGroups::new();

    for (phase_type_id, phase_indirect_parameters_buffer) in indirect_parameters_buffers.iter() {
        build_indirect_parameters_bind_groups.insert(
            *phase_type_id,
            PhaseBuildIndirectParametersBindGroups {
                reset_indexed_indirect_batch_sets: match (phase_indirect_parameters_buffer
                    .indexed
                    .batch_sets_buffer(),)
                {
                    (Some(indexed_batch_sets_buffer),) => Some(
                        render_device.create_bind_group(
                            "reset_indexed_indirect_batch_sets_bind_group",
                            // The early bind group is good for the main phase and late
                            // phase too. They bind the same buffers.
                            &pipelines
                                .early_phase
                                .reset_indirect_batch_sets
                                .bind_group_layout,
                            &BindGroupEntries::sequential((
                                indexed_batch_sets_buffer.as_entire_binding(),
                            )),
                        ),
                    ),
                    _ => None,
                },

                reset_non_indexed_indirect_batch_sets: match (phase_indirect_parameters_buffer
                    .non_indexed
                    .batch_sets_buffer(),)
                {
                    (Some(non_indexed_batch_sets_buffer),) => Some(
                        render_device.create_bind_group(
                            "reset_non_indexed_indirect_batch_sets_bind_group",
                            // The early bind group is good for the main phase and late
                            // phase too. They bind the same buffers.
                            &pipelines
                                .early_phase
                                .reset_indirect_batch_sets
                                .bind_group_layout,
                            &BindGroupEntries::sequential((
                                non_indexed_batch_sets_buffer.as_entire_binding(),
                            )),
                        ),
                    ),
                    _ => None,
                },

                build_indexed_indirect: match (
                    phase_indirect_parameters_buffer
                        .indexed
                        .cpu_metadata_buffer(),
                    phase_indirect_parameters_buffer
                        .indexed
                        .gpu_metadata_buffer(),
                    phase_indirect_parameters_buffer.indexed.data_buffer(),
                    phase_indirect_parameters_buffer.indexed.batch_sets_buffer(),
                ) {
                    (
                        Some(indexed_indirect_parameters_cpu_metadata_buffer),
                        Some(indexed_indirect_parameters_gpu_metadata_buffer),
                        Some(indexed_indirect_parameters_data_buffer),
                        Some(indexed_batch_sets_buffer),
                    ) => Some(
                        render_device.create_bind_group(
                            "build_indexed_indirect_parameters_bind_group",
                            // The frustum culling bind group is good for occlusion culling
                            // too. They bind the same buffers.
                            &pipelines
                                .gpu_frustum_culling_build_indexed_indirect_params
                                .bind_group_layout,
                            &BindGroupEntries::sequential((
                                current_input_buffer.as_entire_binding(),
                                // Don't use `as_entire_binding` here; the shader reads
                                // the length and `RawBufferVec` overallocates.
                                BufferBinding {
                                    buffer: indexed_indirect_parameters_cpu_metadata_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        phase_indirect_parameters_buffer.indexed.batch_count()
                                            as u64
                                            * size_of::<IndirectParametersCpuMetadata>() as u64,
                                    ),
                                },
                                BufferBinding {
                                    buffer: indexed_indirect_parameters_gpu_metadata_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        phase_indirect_parameters_buffer.indexed.batch_count()
                                            as u64
                                            * size_of::<IndirectParametersGpuMetadata>() as u64,
                                    ),
                                },
                                indexed_batch_sets_buffer.as_entire_binding(),
                                indexed_indirect_parameters_data_buffer.as_entire_binding(),
                            )),
                        ),
                    ),
                    _ => None,
                },

                build_non_indexed_indirect: match (
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .cpu_metadata_buffer(),
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .gpu_metadata_buffer(),
                    phase_indirect_parameters_buffer.non_indexed.data_buffer(),
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .batch_sets_buffer(),
                ) {
                    (
                        Some(non_indexed_indirect_parameters_cpu_metadata_buffer),
                        Some(non_indexed_indirect_parameters_gpu_metadata_buffer),
                        Some(non_indexed_indirect_parameters_data_buffer),
                        Some(non_indexed_batch_sets_buffer),
                    ) => Some(
                        render_device.create_bind_group(
                            "build_non_indexed_indirect_parameters_bind_group",
                            // The frustum culling bind group is good for occlusion culling
                            // too. They bind the same buffers.
                            &pipelines
                                .gpu_frustum_culling_build_non_indexed_indirect_params
                                .bind_group_layout,
                            &BindGroupEntries::sequential((
                                current_input_buffer.as_entire_binding(),
                                // Don't use `as_entire_binding` here; the shader reads
                                // the length and `RawBufferVec` overallocates.
                                BufferBinding {
                                    buffer: non_indexed_indirect_parameters_cpu_metadata_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        phase_indirect_parameters_buffer.non_indexed.batch_count()
                                            as u64
                                            * size_of::<IndirectParametersCpuMetadata>() as u64,
                                    ),
                                },
                                BufferBinding {
                                    buffer: non_indexed_indirect_parameters_gpu_metadata_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        phase_indirect_parameters_buffer.non_indexed.batch_count()
                                            as u64
                                            * size_of::<IndirectParametersGpuMetadata>() as u64,
                                    ),
                                },
                                non_indexed_batch_sets_buffer.as_entire_binding(),
                                non_indexed_indirect_parameters_data_buffer.as_entire_binding(),
                            )),
                        ),
                    ),
                    _ => None,
                },
            },
        );
    }

    commands.insert_resource(build_indirect_parameters_bind_groups);
}
