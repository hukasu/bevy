mod decals;

use core::marker::PhantomData;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Assets, Handle};
use bevy_math::{prelude::Rectangle, Quat, Vec2, Vec3};
use bevy_mesh::{Mesh, MeshBuilder, Meshable};
use bevy_render::{render_resource::Shader, RenderDebugFlags};
use bevy_utils::default;

use crate::{
    decal::ForwardDecal,
    material::{plugin::MaterialPlugin, Material},
};

use decals::ForwardDecalMaterial;

pub(super) const FORWARD_DECAL_MESH_HANDLE: Handle<Mesh> =
    weak_handle!("afa817f9-1869-4e0c-ac0d-d8cd1552d38a");
const FORWARD_DECAL_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("f8dfbef4-d88b-42ae-9af4-d9661e9f1648");

/// Plugin to render [`ForwardDecal`]s.
pub struct ForwardDecalPlugin<M: Material>(PhantomData<M>);

impl<M: Material> Default for ForwardDecalPlugin<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: Material> Plugin for ForwardDecalPlugin<M>
where
    MaterialPlugin<ForwardDecalMaterial<M>>: Plugin,
{
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<BaseForwardDecalPlugin>() {
            app.add_plugins(BaseForwardDecalPlugin);
        }

        app.add_plugins(MaterialPlugin::<ForwardDecalMaterial<M>> {
            prepass_enabled: false,
            shadows_enabled: false,
            debug_flags: RenderDebugFlags::default(),
            ..default()
        });
    }
}

struct BaseForwardDecalPlugin;

impl Plugin for BaseForwardDecalPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            FORWARD_DECAL_SHADER_HANDLE,
            "forward_decal.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ForwardDecal>();

        app.world_mut().resource_mut::<Assets<Mesh>>().insert(
            FORWARD_DECAL_MESH_HANDLE.id(),
            Rectangle::from_size(Vec2::ONE)
                .mesh()
                .build()
                .rotated_by(Quat::from_rotation_arc(Vec3::Z, Vec3::Y))
                .with_generated_tangents()
                .unwrap(),
        );
    }
}
