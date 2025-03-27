mod systems;
pub(super) mod uniforms;

use bevy_app::{Plugin, PostUpdate};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_math::Mat4;
use bevy_render::{render_resource::Shader, ExtractSchedule, Render, RenderApp, RenderSet};

use systems::{extract_skins, no_automatic_skin_batching, prepare_skins};
use uniforms::SkinUniforms;

/// Maximum number of joints supported for skinned meshes.
///
/// It is used to allocate buffers.
/// The correctness of the value depends on the GPU/platform.
/// The current value is chosen because it is guaranteed to work everywhere.
/// To allow for bigger values, a check must be made for the limits
/// of the GPU at runtime, which would mean not using consts anymore.
pub const MAX_JOINTS: usize = 256;

/// The total number of joints we support.
///
/// This is 256 GiB worth of joint matrices, which we will never hit under any
/// reasonable circumstances.
const MAX_TOTAL_JOINTS: u32 = 1024 * 1024 * 1024;

/// The number of joints that we allocate at a time.
///
/// Some hardware requires that uniforms be allocated on 256-byte boundaries, so
/// we need to allocate 4 64-byte matrices at a time to satisfy alignment
/// requirements.
const JOINTS_PER_ALLOCATION_UNIT: u32 = (256 / size_of::<Mat4>()) as u32;

/// The maximum ratio of the number of entities whose transforms changed to the
/// total number of joints before we re-extract all joints.
///
/// We use this as a heuristic to decide whether it's worth switching over to
/// fine-grained detection to determine which skins need extraction. If the
/// number of changed entities is over this threshold, we skip change detection
/// and simply re-extract the transforms of all joints.
const JOINT_EXTRACTION_THRESHOLD_FACTOR: f64 = 0.25;

pub const SKINNING_HANDLE: Handle<Shader> = weak_handle!("7474e812-2506-4cbf-9de3-fe07e5c6ff24");

pub struct SkinPlugin;

impl Plugin for SkinPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        load_internal_asset!(app, SKINNING_HANDLE, "skinning.wgsl", Shader::from_wgsl);

        app.add_systems(PostUpdate, no_automatic_skin_batching);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<SkinUniforms>()
                .add_systems(ExtractSchedule, extract_skins)
                .add_systems(Render, prepare_skins.in_set(RenderSet::PrepareResources));
        }
    }
}
