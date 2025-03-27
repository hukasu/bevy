pub(crate) mod decals;
mod systems;

use core::num::NonZero;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_component::ExtractComponentPlugin,
    render_resource::{
        binding_types, BindGroupLayoutEntryBuilder, SamplerBindingType, Shader, TextureSampleType,
    },
    renderer::{RenderAdapter, RenderDevice},
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use decals::{DecalsBuffer, RenderClusteredDecal, RenderClusteredDecals};

use crate::{
    cluster::plugin::ClusterableObjectPlugin, clustered_decals_are_usable, decal::ClusteredDecal,
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
            if clustered_decals_are_usable(render_device, render_adapter) {
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

/// Returns the layout for the clustered-decal-related bind group entries for a
/// single view.
pub(crate) fn get_bind_group_layout_entries(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> Option<[BindGroupLayoutEntryBuilder; 3]> {
    // If binding arrays aren't supported on the current platform, we have no
    // bind group layout entries.
    if !clustered_decals_are_usable(render_device, render_adapter) {
        return None;
    }

    let Some(max_view_decals) = u32::try_from(MAX_VIEW_DECALS)
        .ok()
        .and_then(|max_view_decals| NonZero::<u32>::new(max_view_decals))
    else {
        unreachable!("`MAX_VIEW_DECALS` should never be zero or exceed u32.");
    };

    Some([
        // `decals`
        binding_types::storage_buffer_read_only::<RenderClusteredDecal>(false),
        // `decal_textures`
        binding_types::texture_2d(TextureSampleType::Float { filterable: true })
            .count(max_view_decals),
        // `decal_sampler`
        binding_types::sampler(SamplerBindingType::Filtering),
    ])
}
