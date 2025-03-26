mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_component::ExtractComponentPlugin, render_resource::Shader, Render, RenderApp,
    RenderSet,
};

use crate::distance_fog::{fog::FogMeta, DistanceFog};

use systems::prepare_fog;

/// Handle for the fog WGSL Shader internal asset
const FOG_SHADER_HANDLE: Handle<Shader> = weak_handle!("e943f446-2856-471c-af5e-68dd276eec42");

/// A plugin that consolidates fog extraction, preparation and related resources/assets
pub struct FogPlugin;

impl Plugin for FogPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, FOG_SHADER_HANDLE, "fog.wgsl", Shader::from_wgsl);

        app.register_type::<DistanceFog>();
        app.add_plugins(ExtractComponentPlugin::<DistanceFog>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<FogMeta>()
                .add_systems(Render, prepare_fog.in_set(RenderSet::PrepareResources));
        }
    }
}
