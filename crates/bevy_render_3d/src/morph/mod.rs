pub(super) mod data;
mod systems;

use bevy_app::{Plugin, PostUpdate};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{render_resource::Shader, ExtractSchedule, Render, RenderApp, RenderSet};

use data::{MorphIndices, MorphUniforms};
use systems::{extract_morphs, no_automatic_morph_batching, prepare_morphs};

const MORPH_HANDLE: Handle<Shader> = weak_handle!("da30aac7-34cc-431d-a07f-15b1a783008c");

pub struct MorphPlugin;

impl Plugin for MorphPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        load_internal_asset!(app, MORPH_HANDLE, "morph.wgsl", Shader::from_wgsl);

        app.add_systems(PostUpdate, no_automatic_morph_batching);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MorphUniforms>()
                .init_resource::<MorphIndices>();

            render_app
                .add_systems(ExtractSchedule, extract_morphs)
                .add_systems(Render, prepare_morphs.in_set(RenderSet::PrepareResources));
        }
    }
}
