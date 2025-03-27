//! Decal rendering.
//!
//! Decals are a material that render on top of the surface that they're placed above.
//! They can be used to render signs, paint, snow, impact craters, and other effects on top of surfaces.

// TODO: Once other decal types are added, write a paragraph comparing the different types in the module docs.

pub mod clustered;
pub mod forward;

use bevy_asset::Handle;
use bevy_ecs::{component::Component, entity::Entity, query::With, reflect::ReflectComponent};
use bevy_image::Image;
use bevy_reflect::Reflect;
use bevy_render::{
    extract_component::ExtractComponent,
    mesh::Mesh3d,
    view::{Visibility, VisibilityClass},
};
use bevy_transform::components::{GlobalTransform, Transform};
use forward::FORWARD_DECAL_MESH_HANDLE;

use crate::{cluster::plugin::ClusterAssignable, light::LightVisibilityClass};

/// A decal that renders via a 1x1 transparent quad mesh, smoothly alpha-blending with the underlying
/// geometry towards the edges.
///
/// Because forward decals are meshes, you can use arbitrary materials to control their appearance.
///
/// # Usage Notes
///
/// * Spawn this component on an entity with a [`crate::MeshMaterial3d`] component holding a [`ForwardDecalMaterial`].
/// * Any camera rendering a forward decal must have the [`bevy_core_pipeline::prepass::DepthPrepass`] component.
/// * Looking at forward decals at a steep angle can cause distortion. This can be mitigated by padding your decal's
///   texture with extra transparent pixels on the edges.
#[derive(Component, Reflect)]
#[require(Mesh3d(|| Mesh3d(FORWARD_DECAL_MESH_HANDLE)))]
pub struct ForwardDecal;

/// An object that projects a decal onto surfaces within its bounds.
///
/// Conceptually, a clustered decal is a 1×1×1 cube centered on its origin. It
/// projects the given [`Self::image`] onto surfaces in the +Z direction (thus
/// you may find [`Transform::looking_at`] useful).
///
/// Clustered decals are the highest-quality types of decals that Bevy supports,
/// but they require bindless textures. This means that they presently can't be
/// used on WebGL 2, WebGPU, macOS, or iOS. Bevy's clustered decals can be used
/// with forward or deferred rendering and don't require a prepass.
#[derive(Component, Debug, Clone, Reflect, ExtractComponent)]
#[reflect(Component, Debug, Clone)]
#[require(Transform, Visibility, VisibilityClass)]
#[component(on_add = bevy_render::view::add_visibility_class::<LightVisibilityClass>)]
pub struct ClusteredDecal {
    /// The image that the clustered decal projects.
    ///
    /// This must be a 2D image. If it has an alpha channel, it'll be alpha
    /// blended with the underlying surface and/or other decals. All decal
    /// images in the scene must use the same sampler.
    pub image: Handle<Image>,

    /// An application-specific tag you can use for any purpose you want.
    ///
    /// See the `clustered_decals` example for an example of use.
    pub tag: u32,
}

impl ClusterAssignable for ClusteredDecal {
    type Query<'a> = (Entity, &'a GlobalTransform);
    type Filter = With<ClusteredDecal>;

    fn entity(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Entity {
        query_data.0
    }

    fn transform(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> GlobalTransform {
        *query_data.1
    }

    fn range(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> f32 {
        query_data.1.scale().length()
    }

    fn shadows_enabled(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        false
    }

    fn volumetric(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        false
    }

    fn render_layers(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Option<bevy_render::view::RenderLayers> {
        None
    }

    fn cull_method(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Option<Box<dyn crate::cluster::plugin::ClusterAssignableCullMethod>> {
        None
    }

    fn visible(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        true
    }
}
