use bevy_asset::Asset;
use bevy_reflect::TypePath;
use bevy_render::{
    alpha::AlphaMode,
    mesh::MeshVertexBufferLayoutRef,
    render_resource::{
        AsBindGroup, CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    },
};

use crate::{
    extended_material::{
        pipeline::{MaterialExtensionKey, MaterialExtensionPipeline},
        ExtendedMaterial, MaterialExtension,
    },
    material::Material,
};

/// Type alias for an extended material with a [`ForwardDecalMaterialExt`] extension.
///
/// Make sure to register the [`MaterialPlugin`] for this material in your app setup.
///
/// [`StandardMaterial`] comes with out of the box support for forward decals.
#[expect(type_alias_bounds, reason = "Type alias generics not yet stable")]
pub type ForwardDecalMaterial<B: Material> = ExtendedMaterial<B, ForwardDecalMaterialExt>;

/// Material extension for a [`ForwardDecal`].
///
/// In addition to wrapping your material type with this extension, your shader must use
/// the `bevy_render_3d::decal::forward::get_forward_decal_info` function.
///
/// The `FORWARD_DECAL` shader define will be made available to your shader so that you can gate
/// the forward decal code behind an ifdef.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct ForwardDecalMaterialExt {
    /// Controls how far away a surface must be before the decal will stop blending with it, and instead render as opaque.
    ///
    /// Decreasing this value will cause the decal to blend only to surfaces closer to it.
    ///
    /// Units are in meters.
    #[uniform(200)]
    pub depth_fade_factor: f32,
}

impl MaterialExtension for ForwardDecalMaterialExt {
    fn alpha_mode() -> Option<AlphaMode> {
        Some(AlphaMode::Blend)
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.depth_stencil.as_mut().unwrap().depth_compare = CompareFunction::Always;

        descriptor.vertex.shader_defs.push("FORWARD_DECAL".into());

        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs.push("FORWARD_DECAL".into());
        }

        if let Some(label) = &mut descriptor.label {
            *label = format!("forward_decal_{}", label).into();
        }

        Ok(())
    }
}

impl Default for ForwardDecalMaterialExt {
    fn default() -> Self {
        Self {
            depth_fade_factor: 8.0,
        }
    }
}
