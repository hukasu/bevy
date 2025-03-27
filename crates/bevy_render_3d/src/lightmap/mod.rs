//! Lightmaps, baked lighting textures that can be applied at runtime to provide
//! diffuse global illumination.
//!
//! Bevy doesn't currently have any way to actually bake lightmaps, but they can
//! be baked in an external tool like [Blender](http://blender.org), for example
//! with an addon like [The Lightmapper]. The tools in the [`bevy-baked-gi`]
//! project support other lightmap baking methods.
//!
//! When a [`Lightmap`] component is added to an entity with a [`Mesh3d`] and a
//! [`MeshMaterial3d<StandardMaterial>`], Bevy applies the lightmap when rendering. The brightness
//! of the lightmap may be controlled with the `lightmap_exposure` field on
//! [`StandardMaterial`].
//!
//! During the rendering extraction phase, we extract all lightmaps into the
//! [`RenderLightmaps`] table, which lives in the render world. Mesh bindgroup
//! and mesh uniform creation consults this table to determine which lightmap to
//! supply to the shader. Essentially, the lightmap is a special type of texture
//! that is part of the mesh instance rather than part of the material (because
//! multiple meshes can share the same material, whereas sharing lightmaps is
//! nonsensical).
//!
//! Note that multiple meshes can't be drawn in a single drawcall if they use
//! different lightmap textures, unless bindless textures are in use. If you
//! want to instance a lightmapped mesh, and your platform doesn't support
//! bindless textures, combine the lightmap textures into a single atlas, and
//! set the `uv_rect` field on [`Lightmap`] appropriately.
//!
//! [The Lightmapper]: https://github.com/Naxela/The_Lightmapper
//! [`Mesh3d`]: bevy_render::mesh::Mesh3d
//! [`MeshMaterial3d<StandardMaterial>`]: crate::StandardMaterial
//! [`StandardMaterial`]: crate::StandardMaterial
//! [`bevy-baked-gi`]: https://github.com/pcwalton/bevy-baked-gi

mod lightmap;
pub mod plugin;

use bevy_asset::Handle;
use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_image::Image;
use bevy_math::Rect;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

/// A component that applies baked indirect diffuse global illumination from a
/// lightmap.
///
/// When assigned to an entity that contains a [`Mesh3d`](bevy_render::mesh::Mesh3d) and a
/// [`MeshMaterial3d<StandardMaterial>`](crate::StandardMaterial), if the mesh
/// has a second UV layer ([`ATTRIBUTE_UV_1`](bevy_render::mesh::Mesh::ATTRIBUTE_UV_1)),
/// then the lightmap will render using those UVs.
#[derive(Component, Clone, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct Lightmap {
    /// The lightmap texture.
    pub image: Handle<Image>,

    /// The rectangle within the lightmap texture that the UVs are relative to.
    ///
    /// The top left coordinate is the `min` part of the rect, and the bottom
    /// right coordinate is the `max` part of the rect. The rect ranges from (0,
    /// 0) to (1, 1).
    ///
    /// This field allows lightmaps for a variety of meshes to be packed into a
    /// single atlas.
    pub uv_rect: Rect,

    /// Whether bicubic sampling should be used for sampling this lightmap.
    ///
    /// Bicubic sampling is higher quality, but slower, and may lead to light leaks.
    ///
    /// If true, the lightmap texture's sampler must be set to [`bevy_image::ImageSampler::linear`].
    pub bicubic_sampling: bool,
}

impl Default for Lightmap {
    fn default() -> Self {
        Self {
            image: Default::default(),
            uv_rect: Rect::new(0.0, 0.0, 1.0, 1.0),
            bicubic_sampling: false,
        }
    }
}
