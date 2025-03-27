mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{render_resource::Shader, ExtractSchedule, RenderApp};

use crate::mesh_pipeline::ExtractMeshesSet;

use super::lightmap::RenderLightmaps;

use systems::extract_lightmaps;

/// The ID of the lightmap shader.
const LIGHTMAP_SHADER_HANDLE: Handle<Shader> = weak_handle!("fc28203f-f258-47f3-973c-ce7d1dd70e59");

/// A plugin that provides an implementation of lightmaps.
pub struct LightmapPlugin;

impl Plugin for LightmapPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            LIGHTMAP_SHADER_HANDLE,
            "lightmap.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<RenderLightmaps>()
            .add_systems(ExtractSchedule, extract_lightmaps.after(ExtractMeshesSet));
    }
}
