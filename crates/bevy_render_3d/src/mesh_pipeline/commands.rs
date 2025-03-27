use std::any::TypeId;

use bevy_core_pipeline::{
    oit::OrderIndependentTransparencySettingsOffset, prepass::MotionVectorPrepass,
};
use bevy_ecs::{
    query::{Has, ROQueryItem},
    system::{
        lifetimeless::{Read, SRes},
        SystemParamItem,
    },
};
use bevy_render::{
    batching::gpu_preprocessing::{
        IndirectBatchSet, IndirectParametersBuffers, IndirectParametersIndexed,
        IndirectParametersNonIndexed,
    },
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{
        PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, TrackedRenderPass,
    },
    render_resource::PipelineCache,
    renderer::RenderDevice,
    view::ViewUniformOffset,
};
use smallvec::{smallvec, SmallVec};
use tracing::warn;

use crate::{
    distance_fog::fog::ViewFogUniformOffset,
    gpu_preprocess::render::{PreprocessBindGroups, PreprocessPipelines},
    light::ViewLightsUniformOffset,
    light_probe::{
        environment_map::ViewEnvironmentMapUniformOffset,
        light_probes::ViewLightProbesUniformOffset,
    },
    lightmap::RenderLightmaps,
    morph::data::MorphIndices,
    skin::uniforms::{skins_use_uniform_buffers, SkinUniforms},
    ssr::render::ViewScreenSpaceReflectionsUniformOffset,
};

use super::render::{instance::RenderMeshInstances, MeshBindGroups, MeshViewBindGroup};

pub struct SetMeshViewBindGroup<const I: usize>;

pub struct SetMeshBindGroup<const I: usize>;

pub struct DrawMesh;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetMeshViewBindGroup<I> {
    type Param = ();
    type ViewQuery = (
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<ViewFogUniformOffset>,
        Read<ViewLightProbesUniformOffset>,
        Read<ViewScreenSpaceReflectionsUniformOffset>,
        Read<ViewEnvironmentMapUniformOffset>,
        Read<MeshViewBindGroup>,
        Option<Read<OrderIndependentTransparencySettingsOffset>>,
    );
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        (
            view_uniform,
            view_lights,
            view_fog,
            view_light_probes,
            view_ssr,
            view_environment_map,
            mesh_view_bind_group,
            maybe_oit_layers_count_offset,
        ): ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<()>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mut offsets: SmallVec<[u32; 8]> = smallvec![
            view_uniform.offset,
            view_lights.offset,
            view_fog.offset,
            **view_light_probes,
            **view_ssr,
            **view_environment_map,
        ];
        if let Some(layers_count_offset) = maybe_oit_layers_count_offset {
            offsets.push(layers_count_offset.offset);
        }
        pass.set_bind_group(I, &mesh_view_bind_group.value, &offsets);

        RenderCommandResult::Success
    }
}

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetMeshBindGroup<I> {
    type Param = (
        SRes<RenderDevice>,
        SRes<MeshBindGroups>,
        SRes<RenderMeshInstances>,
        SRes<SkinUniforms>,
        SRes<MorphIndices>,
        SRes<RenderLightmaps>,
    );
    type ViewQuery = Has<MotionVectorPrepass>;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        has_motion_vector_prepass: bool,
        _item_query: Option<()>,
        (
            render_device,
            bind_groups,
            mesh_instances,
            skin_uniforms,
            morph_indices,
            lightmaps,
        ): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_groups = bind_groups.into_inner();
        let mesh_instances = mesh_instances.into_inner();
        let skin_uniforms = skin_uniforms.into_inner();
        let morph_indices = morph_indices.into_inner();

        let entity = &item.main_entity();

        let Some(mesh_asset_id) = mesh_instances.mesh_asset_id(*entity) else {
            return RenderCommandResult::Success;
        };

        let current_skin_byte_offset = skin_uniforms.skin_byte_offset(*entity);
        let current_morph_index = morph_indices.current.get(entity);
        let prev_morph_index = morph_indices.prev.get(entity);

        let is_skinned = current_skin_byte_offset.is_some();
        let is_morphed = current_morph_index.is_some();

        let lightmap_slab_index = lightmaps
            .render_lightmaps
            .get(entity)
            .map(|render_lightmap| render_lightmap.slab_index);

        let Some(mesh_phase_bind_groups) = (match *bind_groups {
            MeshBindGroups::CpuPreprocessing(ref mesh_phase_bind_groups) => {
                Some(mesh_phase_bind_groups)
            }
            MeshBindGroups::GpuPreprocessing(ref mesh_phase_bind_groups) => {
                mesh_phase_bind_groups.get(&TypeId::of::<P>())
            }
        }) else {
            // This is harmless if e.g. we're rendering the `Shadow` phase and
            // there weren't any shadows.
            return RenderCommandResult::Success;
        };

        let Some(bind_group) = mesh_phase_bind_groups.get(
            mesh_asset_id,
            lightmap_slab_index,
            is_skinned,
            is_morphed,
            has_motion_vector_prepass,
        ) else {
            return RenderCommandResult::Failure(
                "The MeshBindGroups resource wasn't set in the render phase. \
                It should be set by the prepare_mesh_bind_group system.\n\
                This is a bevy bug! Please open an issue.",
            );
        };

        let mut dynamic_offsets: [u32; 3] = Default::default();
        let mut offset_count = 0;
        if let PhaseItemExtraIndex::DynamicOffset(dynamic_offset) = item.extra_index() {
            dynamic_offsets[offset_count] = dynamic_offset;
            offset_count += 1;
        }
        if let Some(current_skin_index) = current_skin_byte_offset {
            if skins_use_uniform_buffers(&render_device) {
                dynamic_offsets[offset_count] = current_skin_index.byte_offset;
                offset_count += 1;
            }
        }
        if let Some(current_morph_index) = current_morph_index {
            dynamic_offsets[offset_count] = current_morph_index.index;
            offset_count += 1;
        }

        // Attach motion vectors if needed.
        if has_motion_vector_prepass {
            // Attach the previous skin index for motion vector computation.
            if skins_use_uniform_buffers(&render_device) {
                if let Some(current_skin_byte_offset) = current_skin_byte_offset {
                    dynamic_offsets[offset_count] = current_skin_byte_offset.byte_offset;
                    offset_count += 1;
                }
            }

            // Attach the previous morph index for motion vector computation. If
            // there isn't one, just use zero as the shader will ignore it.
            if current_morph_index.is_some() {
                match prev_morph_index {
                    Some(prev_morph_index) => {
                        dynamic_offsets[offset_count] = prev_morph_index.index;
                    }
                    None => dynamic_offsets[offset_count] = 0,
                }
                offset_count += 1;
            }
        }

        pass.set_bind_group(I, bind_group, &dynamic_offsets[0..offset_count]);

        RenderCommandResult::Success
    }
}

impl<P: PhaseItem> RenderCommand<P> for DrawMesh {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<IndirectParametersBuffers>,
        SRes<PipelineCache>,
        SRes<MeshAllocator>,
        Option<SRes<PreprocessPipelines>>,
    );
    type ViewQuery = Has<PreprocessBindGroups>;
    type ItemQuery = ();
    #[inline]
    fn render<'w>(
        item: &P,
        has_preprocess_bind_group: ROQueryItem<Self::ViewQuery>,
        _item_query: Option<()>,
        (
            meshes,
            mesh_instances,
            indirect_parameters_buffer,
            pipeline_cache,
            mesh_allocator,
            preprocess_pipelines,
        ): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // If we're using GPU preprocessing, then we're dependent on that
        // compute shader having been run, which of course can only happen if
        // it's compiled. Otherwise, our mesh instance data won't be present.
        if let Some(preprocess_pipelines) = preprocess_pipelines {
            if !has_preprocess_bind_group
                || !preprocess_pipelines.pipelines_are_loaded(&pipeline_cache)
            {
                return RenderCommandResult::Skip;
            }
        }

        let meshes = meshes.into_inner();
        let mesh_instances = mesh_instances.into_inner();
        let indirect_parameters_buffer = indirect_parameters_buffer.into_inner();
        let mesh_allocator = mesh_allocator.into_inner();

        let Some(mesh_asset_id) = mesh_instances.mesh_asset_id(item.main_entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.get(mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

        let batch_range = item.batch_range();

        // Draw either directly or indirectly, as appropriate. If we're in
        // indirect mode, we can additionally multi-draw. (We can't multi-draw
        // in direct mode because `wgpu` doesn't expose that functionality.)
        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);

                match item.extra_index() {
                    PhaseItemExtraIndex::None | PhaseItemExtraIndex::DynamicOffset(_) => {
                        pass.draw_indexed(
                            index_buffer_slice.range.start
                                ..(index_buffer_slice.range.start + *count),
                            vertex_buffer_slice.range.start as i32,
                            batch_range.clone(),
                        );
                    }
                    PhaseItemExtraIndex::IndirectParametersIndex {
                        range: indirect_parameters_range,
                        batch_set_index,
                    } => {
                        // Look up the indirect parameters buffer, as well as
                        // the buffer we're going to use for
                        // `multi_draw_indexed_indirect_count` (if available).
                        let Some(phase_indirect_parameters_buffers) =
                            indirect_parameters_buffer.get(&TypeId::of::<P>())
                        else {
                            warn!(
                                "Not rendering mesh because indexed indirect parameters buffer \
                                 wasn't present for this phase",
                            );
                            return RenderCommandResult::Skip;
                        };
                        let (Some(indirect_parameters_buffer), Some(batch_sets_buffer)) = (
                            phase_indirect_parameters_buffers.indexed.data_buffer(),
                            phase_indirect_parameters_buffers
                                .indexed
                                .batch_sets_buffer(),
                        ) else {
                            warn!(
                                "Not rendering mesh because indexed indirect parameters buffer \
                                 wasn't present",
                            );
                            return RenderCommandResult::Skip;
                        };

                        // Calculate the location of the indirect parameters
                        // within the buffer.
                        let indirect_parameters_offset = indirect_parameters_range.start as u64
                            * size_of::<IndirectParametersIndexed>() as u64;
                        let indirect_parameters_count =
                            indirect_parameters_range.end - indirect_parameters_range.start;

                        // If we're using `multi_draw_indirect_count`, take the
                        // number of batches from the appropriate position in
                        // the batch sets buffer. Otherwise, supply the size of
                        // the batch set.
                        match batch_set_index {
                            Some(batch_set_index) => {
                                let count_offset = u32::from(batch_set_index)
                                    * (size_of::<IndirectBatchSet>() as u32);
                                pass.multi_draw_indexed_indirect_count(
                                    indirect_parameters_buffer,
                                    indirect_parameters_offset,
                                    batch_sets_buffer,
                                    count_offset as u64,
                                    indirect_parameters_count,
                                );
                            }
                            None => {
                                pass.multi_draw_indexed_indirect(
                                    indirect_parameters_buffer,
                                    indirect_parameters_offset,
                                    indirect_parameters_count,
                                );
                            }
                        }
                    }
                }
            }

            RenderMeshBufferInfo::NonIndexed => match item.extra_index() {
                PhaseItemExtraIndex::None | PhaseItemExtraIndex::DynamicOffset(_) => {
                    pass.draw(vertex_buffer_slice.range, batch_range.clone());
                }
                PhaseItemExtraIndex::IndirectParametersIndex {
                    range: indirect_parameters_range,
                    batch_set_index,
                } => {
                    // Look up the indirect parameters buffer, as well as the
                    // buffer we're going to use for
                    // `multi_draw_indirect_count` (if available).
                    let Some(phase_indirect_parameters_buffers) =
                        indirect_parameters_buffer.get(&TypeId::of::<P>())
                    else {
                        warn!(
                            "Not rendering mesh because non-indexed indirect parameters buffer \
                                 wasn't present for this phase",
                        );
                        return RenderCommandResult::Skip;
                    };
                    let (Some(indirect_parameters_buffer), Some(batch_sets_buffer)) = (
                        phase_indirect_parameters_buffers.non_indexed.data_buffer(),
                        phase_indirect_parameters_buffers
                            .non_indexed
                            .batch_sets_buffer(),
                    ) else {
                        warn!(
                            "Not rendering mesh because non-indexed indirect parameters buffer \
                             wasn't present"
                        );
                        return RenderCommandResult::Skip;
                    };

                    // Calculate the location of the indirect parameters within
                    // the buffer.
                    let indirect_parameters_offset = indirect_parameters_range.start as u64
                        * size_of::<IndirectParametersNonIndexed>() as u64;
                    let indirect_parameters_count =
                        indirect_parameters_range.end - indirect_parameters_range.start;

                    // If we're using `multi_draw_indirect_count`, take the
                    // number of batches from the appropriate position in the
                    // batch sets buffer. Otherwise, supply the size of the
                    // batch set.
                    match batch_set_index {
                        Some(batch_set_index) => {
                            let count_offset =
                                u32::from(batch_set_index) * (size_of::<IndirectBatchSet>() as u32);
                            pass.multi_draw_indirect_count(
                                indirect_parameters_buffer,
                                indirect_parameters_offset,
                                batch_sets_buffer,
                                count_offset as u64,
                                indirect_parameters_count,
                            );
                        }
                        None => {
                            pass.multi_draw_indirect(
                                indirect_parameters_buffer,
                                indirect_parameters_offset,
                                indirect_parameters_count,
                            );
                        }
                    }
                }
            },
        }
        RenderCommandResult::Success
    }
}
