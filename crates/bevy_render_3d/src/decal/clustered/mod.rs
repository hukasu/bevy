mod decals;
mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_component::ExtractComponentPlugin,
    render_resource::Shader,
    renderer::{RenderAdapter, RenderDevice},
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use decals::{DecalsBuffer, RenderClusteredDecals};

use crate::{
    binding_arrays_are_usable, cluster::plugin::ClusterableObjectPlugin, decal::ClusteredDecal,
    light::plugin::LightSystems,
};

use systems::{extract_decals, prepare_decals, upload_decals};

/// The handle to the `clustered.wgsl` shader.
const CLUSTERED_DECAL_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("87929002-3509-42f1-8279-2d2765dd145c");

/// The maximum number of decals that can be present in a view.
///
/// This number is currently relatively low in order to work around the lack of
/// first-class binding arrays in `wgpu`. When that feature is implemented, this
/// limit can be increased.
const MAX_VIEW_DECALS: usize = 8;

/// A plugin that adds support for clustered decals.
///
/// In environments where bindless textures aren't available, clustered decals
/// can still be added to a scene, but they won't project any decals.
pub struct ClusteredDecalPlugin;

impl Plugin for ClusteredDecalPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CLUSTERED_DECAL_SHADER_HANDLE,
            "clustered.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(ExtractComponentPlugin::<ClusteredDecal>::default())
            .register_type::<ClusteredDecal>();

        if let (Some(render_device), Some(render_adapter)) =
            (app.world().get_resource(), app.world().get_resource())
        {
            // Add decals if the current platform supports them.
            if clustered_decals_are_usable(&render_device, &render_adapter) {
                app.add_plugins(ClusterableObjectPlugin::<4, ClusteredDecal>::default());
            }
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DecalsBuffer>()
            .init_resource::<RenderClusteredDecals>()
            .add_systems(ExtractSchedule, extract_decals)
            .add_systems(
                Render,
                prepare_decals
                    .in_set(RenderSet::ManageViews)
                    .after(LightSystems::Prepare),
            )
            .add_systems(Render, upload_decals.in_set(RenderSet::PrepareResources));
    }
}

/// Returns true if clustered decals are usable on the current platform or false
/// otherwise.
///
/// Clustered decals are currently disabled on macOS and iOS due to insufficient
/// texture bindings and limited bindless support in `wgpu`.
fn clustered_decals_are_usable(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> bool {
    // Disable binding arrays on Metal. There aren't enough texture bindings available.
    // See issue #17553.
    // Re-enable this when `wgpu` has first-class bindless.
    binding_arrays_are_usable(render_device, render_adapter)
        && cfg!(not(any(target_os = "macos", target_os = "ios")))
}
