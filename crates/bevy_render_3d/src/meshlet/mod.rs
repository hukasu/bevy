//! Render high-poly 3d meshes using an efficient GPU-driven method. See [`MeshletPlugin`] and [`MeshletMesh`] for details.

mod asset;
#[cfg(feature = "meshlet_processor")]
mod from_mesh;
pub mod graph;
pub mod plugin;
mod render;

use bevy_asset::{AssetId, Handle};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::view::{Visibility, VisibilityClass};
use bevy_transform::components::Transform;

use derive_more::From;

use crate::prepass::PreviousGlobalTransform;

pub use self::asset::{
    MeshletMesh, MeshletMeshLoader, MeshletMeshSaver, MESHLET_MESH_ASSET_VERSION,
};
#[cfg(feature = "meshlet_processor")]
pub use self::from_mesh::{
    MeshToMeshletMeshConversionError, MESHLET_DEFAULT_VERTEX_POSITION_QUANTIZATION_FACTOR,
};

/// The meshlet mesh equivalent of [`bevy_render::mesh::Mesh3d`].
#[derive(Component, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(Transform, PreviousGlobalTransform, Visibility, VisibilityClass)]
#[component(on_add = bevy_render::view::add_visibility_class::<MeshletMesh3d>)]
pub struct MeshletMesh3d(pub Handle<MeshletMesh>);

impl From<MeshletMesh3d> for AssetId<MeshletMesh> {
    fn from(mesh: MeshletMesh3d) -> Self {
        mesh.id()
    }
}

impl From<&MeshletMesh3d> for AssetId<MeshletMesh> {
    fn from(mesh: &MeshletMesh3d) -> Self {
        mesh.id()
    }
}
