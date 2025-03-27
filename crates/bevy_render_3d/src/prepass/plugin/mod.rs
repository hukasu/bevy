mod systems;

use core::{hash::Hash, marker::PhantomData};

use bevy_app::{App, Plugin, PreUpdate};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_core_pipeline::{
    deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
    prepass::{AlphaMask3dPrepass, Opaque3dPrepass},
};
use bevy_ecs::{resource::Resource, schedule::IntoScheduleConfigs};
use bevy_render::{
    mesh::RenderMesh,
    render_asset::prepare_assets,
    render_phase::{AddRenderCommand, BinnedRenderPhasePlugin},
    render_resource::{Shader, SpecializedMeshPipelines},
    ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSet,
};

use crate::{
    material::{plugin::MaterialRenderSystems, Material, PreparedMaterial},
    mesh_pipeline::{render::pipeline::MeshPipeline, GpuMeshBuildingSet},
    prepass::render::{PrepassPipeline, PrepassViewBindGroup},
};

use systems::{
    check_prepass_views_need_specialization, extract_camera_previous_view_data,
    prepare_prepass_view_bind_group, prepare_previous_view_uniforms, queue_prepass_material_meshes,
    specialize_prepass_material_meshes, update_mesh_previous_global_transforms,
    update_previous_view_data,
};

use super::{
    commands::DrawPrepass,
    render::{
        SpecializedPrepassMaterialPipelineCache, ViewKeyPrepassCache,
        ViewPrepassSpecializationTicks,
    },
};

pub(super) const PREPASS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("ce810284-f1ae-4439-ab2e-0d6b204b6284");
const PREPASS_BINDINGS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("3e83537e-ae17-489c-a18a-999bc9c1d252");
const PREPASS_UTILS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("02e4643a-a14b-48eb-a339-0c47aeab0d7e");
const PREPASS_IO_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("1c065187-c99b-4b7c-ba59-c1575482d2c9");

/// Sets up everything required to use the prepass pipeline.
///
/// This does not add the actual prepasses, see [`PrepassPlugin`] for that.
pub struct PrepassPipelinePlugin<M: Material>(PhantomData<M>);

impl<M: Material> Default for PrepassPipelinePlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<M: Material> Plugin for PrepassPipelinePlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PREPASS_SHADER_HANDLE,
            "shaders/prepass.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            PREPASS_BINDINGS_SHADER_HANDLE,
            "shaders/prepass_bindings.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            PREPASS_UTILS_SHADER_HANDLE,
            "shaders/prepass_utils.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            PREPASS_IO_SHADER_HANDLE,
            "shaders/prepass_io.wgsl",
            Shader::from_wgsl
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                Render,
                prepare_prepass_view_bind_group::<M>.in_set(RenderSet::PrepareBindGroups),
            )
            .init_resource::<PrepassViewBindGroup>()
            .init_resource::<SpecializedMeshPipelines<PrepassPipeline<M>>>()
            .allow_ambiguous_resource::<SpecializedMeshPipelines<PrepassPipeline<M>>>();
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<PrepassPipeline<M>>();
    }
}

/// Sets up the prepasses for a [`Material`].
///
/// This depends on the [`PrepassPipelinePlugin`].
pub struct PrepassPlugin<M: Material> {
    /// Debugging flags that can optionally be set when constructing the renderer.
    pub debug_flags: RenderDebugFlags,
    pub phantom: PhantomData<M>,
}

impl<M: Material> PrepassPlugin<M> {
    /// Creates a new [`PrepassPlugin`] with the given debug flags.
    pub fn new(debug_flags: RenderDebugFlags) -> Self {
        PrepassPlugin {
            debug_flags,
            phantom: PhantomData,
        }
    }
}

impl<M: Material> Plugin for PrepassPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        let no_prepass_plugin_loaded = app
            .world()
            .get_resource::<AnyPrepassPluginLoaded>()
            .is_none();

        if no_prepass_plugin_loaded {
            app.insert_resource(AnyPrepassPluginLoaded)
                // At the start of each frame, last frame's GlobalTransforms become this frame's PreviousGlobalTransforms
                // and last frame's view projection matrices become this frame's PreviousViewProjections
                .add_systems(
                    PreUpdate,
                    (
                        update_mesh_previous_global_transforms,
                        update_previous_view_data,
                    ),
                )
                .add_plugins((
                    BinnedRenderPhasePlugin::<Opaque3dPrepass, MeshPipeline>::new(self.debug_flags),
                    BinnedRenderPhasePlugin::<AlphaMask3dPrepass, MeshPipeline>::new(
                        self.debug_flags,
                    ),
                ));
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        if no_prepass_plugin_loaded {
            render_app
                .add_systems(ExtractSchedule, extract_camera_previous_view_data)
                .add_systems(
                    Render,
                    prepare_previous_view_uniforms.in_set(RenderSet::PrepareResources),
                );
        }

        render_app
            .init_resource::<ViewPrepassSpecializationTicks>()
            .init_resource::<ViewKeyPrepassCache>()
            .init_resource::<SpecializedPrepassMaterialPipelineCache<M>>()
            .add_render_command::<Opaque3dPrepass, DrawPrepass<M>>()
            .add_render_command::<AlphaMask3dPrepass, DrawPrepass<M>>()
            .add_render_command::<Opaque3dDeferred, DrawPrepass<M>>()
            .add_render_command::<AlphaMask3dDeferred, DrawPrepass<M>>()
            .add_systems(
                Render,
                (
                    check_prepass_views_need_specialization.in_set(RenderSet::PrepareAssets),
                    specialize_prepass_material_meshes::<M>
                        .in_set(RenderSet::PrepareMeshes)
                        .after(prepare_assets::<PreparedMaterial<M>>)
                        .after(prepare_assets::<RenderMesh>)
                        .after(GpuMeshBuildingSet),
                    queue_prepass_material_meshes::<M>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<M>>)
                        .ambiguous_with(MaterialRenderSystems::QueueMeshes),
                ),
            );
    }
}

#[derive(Resource)]
struct AnyPrepassPluginLoaded;
