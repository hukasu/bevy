use bevy_app::Plugin;
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_render::render_resource::Shader;

// This uses the same UUID as the one from `meshlet`
const MESHLET_VISIBILITY_BUFFER_RESOLVE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("69187376-3dea-4d0f-b3f5-185bde63d6a2");

/// Setup dummy shaders for when [`MeshletPlugin`](crate::meshlet::plugin::MeshletPlugin)
/// is not used to prevent shader import errors.
pub struct DummyMeshletPlugin;

impl Plugin for DummyMeshletPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        load_internal_asset!(
            app,
            MESHLET_VISIBILITY_BUFFER_RESOLVE_SHADER_HANDLE,
            "dummy_visibility_buffer_resolve.wgsl",
            Shader::from_wgsl
        );
    }
}
