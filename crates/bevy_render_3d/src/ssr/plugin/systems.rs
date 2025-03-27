use bevy_core_pipeline::prepass::{
    DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
};
use bevy_ecs::{
    entity::Entity,
    query::{Has, With},
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    render_resource::{PipelineCache, SpecializedRenderPipelines},
    renderer::{RenderDevice, RenderQueue},
    view::{ExtractedView, Msaa},
};

use crate::{
    light_probe::{light_probes::RenderViewLightProbes, EnvironmentMapLight},
    mesh_pipeline::render::pipeline::MeshPipelineViewLayoutKey,
    ssr::render::{
        ScreenSpaceReflectionsBuffer, ScreenSpaceReflectionsPipeline,
        ScreenSpaceReflectionsPipelineId, ScreenSpaceReflectionsPipelineKey,
        ScreenSpaceReflectionsUniform, ViewScreenSpaceReflectionsUniformOffset,
    },
};

/// Sets up screen space reflection pipelines for each applicable view.
pub fn prepare_ssr_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ScreenSpaceReflectionsPipeline>>,
    ssr_pipeline: Res<ScreenSpaceReflectionsPipeline>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            Has<RenderViewLightProbes<EnvironmentMapLight>>,
            Has<NormalPrepass>,
            Has<MotionVectorPrepass>,
        ),
        (
            With<ScreenSpaceReflectionsUniform>,
            With<DepthPrepass>,
            With<DeferredPrepass>,
        ),
    >,
) {
    for (
        entity,
        extracted_view,
        has_environment_maps,
        has_normal_prepass,
        has_motion_vector_prepass,
    ) in &views
    {
        // SSR is only supported in the deferred pipeline, which has no MSAA
        // support. Thus we can assume MSAA is off.
        let mut mesh_pipeline_view_key = MeshPipelineViewLayoutKey::from(Msaa::Off)
            | MeshPipelineViewLayoutKey::DEPTH_PREPASS
            | MeshPipelineViewLayoutKey::DEFERRED_PREPASS;
        mesh_pipeline_view_key.set(
            MeshPipelineViewLayoutKey::NORMAL_PREPASS,
            has_normal_prepass,
        );
        mesh_pipeline_view_key.set(
            MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS,
            has_motion_vector_prepass,
        );

        // Build the pipeline.
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &ssr_pipeline,
            ScreenSpaceReflectionsPipelineKey {
                mesh_pipeline_view_key,
                is_hdr: extracted_view.hdr,
                has_environment_maps,
            },
        );

        // Note which pipeline ID was used.
        commands
            .entity(entity)
            .insert(ScreenSpaceReflectionsPipelineId(pipeline_id));
    }
}

/// Gathers up screen space reflection settings for each applicable view and
/// writes them into a GPU buffer.
pub fn prepare_ssr_settings(
    mut commands: Commands,
    views: Query<(Entity, Option<&ScreenSpaceReflectionsUniform>), With<ExtractedView>>,
    mut ssr_settings_buffer: ResMut<ScreenSpaceReflectionsBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut writer) =
        ssr_settings_buffer.get_writer(views.iter().len(), &render_device, &render_queue)
    else {
        return;
    };

    for (view, ssr_uniform) in views.iter() {
        let uniform_offset = match ssr_uniform {
            None => 0,
            Some(ssr_uniform) => writer.write(ssr_uniform),
        };
        commands
            .entity(view)
            .insert(ViewScreenSpaceReflectionsUniformOffset(uniform_offset));
    }
}
