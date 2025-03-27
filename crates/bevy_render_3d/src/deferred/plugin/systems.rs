use bevy_core_pipeline::{
    prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_ecs::{
    entity::Entity,
    query::{Has, With, Without},
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    render_resource::{PipelineCache, SpecializedRenderPipelines},
    view::ExtractedView,
};

use crate::{
    deferred::{
        render::{DeferredLightingLayout, DeferredLightingPipeline},
        DeferredLightingDepthId,
    },
    distance_fog::DistanceFog,
    light_probe::{light_probes::RenderViewLightProbes, EnvironmentMapLight, IrradianceVolume},
    mesh_pipeline::render::pipeline::MeshPipelineKey,
    shadow::ShadowFilteringMethod,
    ssao::ScreenSpaceAmbientOcclusion,
    ssr::render::ScreenSpaceReflectionsUniform,
};

pub fn insert_deferred_lighting_pass_id_component(
    mut commands: Commands,
    views: Query<Entity, (With<DeferredPrepass>, Without<DeferredLightingDepthId>)>,
) {
    for entity in views.iter() {
        commands
            .entity(entity)
            .insert(DeferredLightingDepthId::default());
    }
}

pub fn prepare_deferred_lighting_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DeferredLightingLayout>>,
    deferred_lighting_layout: Res<DeferredLightingLayout>,
    views: Query<(
        Entity,
        &ExtractedView,
        Option<&Tonemapping>,
        Option<&DebandDither>,
        Option<&ShadowFilteringMethod>,
        (
            Has<ScreenSpaceAmbientOcclusion>,
            Has<ScreenSpaceReflectionsUniform>,
            Has<DistanceFog>,
        ),
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        Has<RenderViewLightProbes<EnvironmentMapLight>>,
        Has<RenderViewLightProbes<IrradianceVolume>>,
    )>,
) {
    for (
        entity,
        view,
        tonemapping,
        dither,
        shadow_filter_method,
        (ssao, ssr, distance_fog),
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
        has_environment_maps,
        has_irradiance_volumes,
    ) in &views
    {
        // If there is no deferred prepass, remove the old pipeline if there was
        // one. This handles the case in which a view using deferred stops using
        // it.
        if !deferred_prepass {
            commands.entity(entity).remove::<DeferredLightingPipeline>();
            continue;
        }

        let mut view_key = MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        // Always true, since we're in the deferred lighting pipeline
        view_key |= MeshPipelineKey::DEFERRED_PREPASS;

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                view_key |= match tonemapping {
                    Tonemapping::None => MeshPipelineKey::TONEMAP_METHOD_NONE,
                    Tonemapping::Reinhard => MeshPipelineKey::TONEMAP_METHOD_REINHARD,
                    Tonemapping::ReinhardLuminance => {
                        MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE
                    }
                    Tonemapping::AcesFitted => MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
                    Tonemapping::AgX => MeshPipelineKey::TONEMAP_METHOD_AGX,
                    Tonemapping::SomewhatBoringDisplayTransform => {
                        MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
                    }
                    Tonemapping::TonyMcMapface => MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
                    Tonemapping::BlenderFilmic => MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
                };
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= MeshPipelineKey::DEBAND_DITHER;
            }
        }

        if ssao {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if ssr {
            view_key |= MeshPipelineKey::SCREEN_SPACE_REFLECTIONS;
        }
        if distance_fog {
            view_key |= MeshPipelineKey::DISTANCE_FOG;
        }

        // We don't need to check to see whether the environment map is loaded
        // because [`gather_light_probes`] already checked that for us before
        // adding the [`RenderViewEnvironmentMaps`] component.
        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }

        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }

        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        let pipeline_id =
            pipelines.specialize(&pipeline_cache, &deferred_lighting_layout, view_key);

        commands
            .entity(entity)
            .insert(DeferredLightingPipeline { pipeline_id });
    }
}
