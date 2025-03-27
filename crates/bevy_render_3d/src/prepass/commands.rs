use bevy_core_pipeline::prepass::{MotionVectorPrepass, PreviousViewUniformOffset};
use bevy_ecs::{
    query::Has,
    system::{
        lifetimeless::{Read, SRes},
        SystemParamItem,
    },
};
use bevy_render::{
    render_phase::{
        PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
    },
    view::ViewUniformOffset,
};

use crate::{
    material::commands::SetMaterialBindGroup,
    mesh_pipeline::commands::{DrawMesh, SetMeshBindGroup},
};

use super::render::PrepassViewBindGroup;

pub type DrawPrepass<M> = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetMaterialBindGroup<M, 2>,
    DrawMesh,
);

pub struct SetPrepassViewBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetPrepassViewBindGroup<I> {
    type Param = SRes<PrepassViewBindGroup>;
    type ViewQuery = (
        Read<ViewUniformOffset>,
        Has<MotionVectorPrepass>,
        Option<Read<PreviousViewUniformOffset>>,
    );
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        (view_uniform_offset, has_motion_vector_prepass, previous_view_uniform_offset): (
            &'_ ViewUniformOffset,
            bool,
            Option<&'_ PreviousViewUniformOffset>,
        ),
        _entity: Option<()>,
        prepass_view_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let prepass_view_bind_group = prepass_view_bind_group.into_inner();

        match previous_view_uniform_offset {
            Some(previous_view_uniform_offset) if has_motion_vector_prepass => {
                pass.set_bind_group(
                    I,
                    prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
                    &[
                        view_uniform_offset.offset,
                        previous_view_uniform_offset.offset,
                    ],
                );
            }
            _ => {
                pass.set_bind_group(
                    I,
                    prepass_view_bind_group.no_motion_vectors.as_ref().unwrap(),
                    &[view_uniform_offset.offset],
                );
            }
        }

        RenderCommandResult::Success
    }
}
