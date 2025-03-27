#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! Provides functionality for rendering in 3d

pub mod atmosphere;
pub mod cluster;
pub mod decal;
pub mod distance_fog;
#[cfg(not(feature = "meshlet"))]
mod dummy_meshlet;
pub mod extended_material;
pub mod gpu_preprocess;
pub mod light;
pub mod light_probe;
pub mod lightmap;
pub mod material;
pub mod mesh_pipeline;
#[cfg(feature = "meshlet")]
mod meshlet;
pub mod morph;
pub mod prepass;
pub mod shadow;
pub mod skin;
pub mod ssao;
pub mod ssr;
pub mod volumetric_fog;

extern crate alloc;

use core::num::NonZero;

use bevy_app::{PluginGroup, PluginGroupBuilder};
use bevy_mesh::{Mesh, MeshVertexBufferLayoutRef, VertexAttributeDescriptor};
use bevy_render::{
    render_resource::{
        binding_types, BindGroupLayout, BindGroupLayoutEntryBuilder, BufferBindingType,
        ShaderDefVal,
    },
    renderer::{RenderAdapter, RenderDevice},
    settings::WgpuFeatures,
    RenderDebugFlags,
};

use light_probe::MAX_VIEW_LIGHT_PROBES;
use mesh_pipeline::render::{pipeline::MeshPipelineKey, MeshLayouts};

/// Experimental features that are not yet finished. Please report any issues you encounter!
///
/// Expect bugs, missing features, compatibility issues, low performance, and/or future breaking changes.
#[cfg(feature = "meshlet")]
pub mod experimental {
    /// Render high-poly 3d meshes using an efficient GPU-driven method.
    /// See [`MeshletPlugin`](crate::meshlet::plugin::MeshletPlugin) and
    /// [`MeshletMesh`](crate::meshlet::MeshletMesh) for details.
    pub mod meshlet {
        pub use crate::meshlet::*;
    }
}

pub mod prelude {
    pub use super::atmosphere::{Atmosphere, AtmosphereSettings};
    pub use super::decal::{ClusteredDecal, ForwardDecal};
    pub use super::distance_fog::{DistanceFog, FogFalloff};
    pub use super::extended_material::{ExtendedMaterial, MaterialExtension};
    pub use super::light::{AmbientLight, DirectionalLight, PointLight, SpotLight};
    pub use super::light_probe::{EnvironmentMapLight, IrradianceVolume, LightProbe};
    pub use super::lightmap::Lightmap;
    pub use super::material::{Material, MeshMaterial3d};
    #[cfg(feature = "meshlet")]
    pub mod experimental {
        pub use crate::meshlet::{MeshletMesh, MeshletMesh3d};
    }
    pub use super::shadow::ShadowFilteringMethod;
    pub use super::ssao::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel};
    pub use super::ssr::ScreenSpaceReflections;
    pub use super::volumetric_fog::{FogVolume, VolumetricFog, VolumetricLight};
}

pub struct Render3dPluginGroup {
    debug_flags: RenderDebugFlags,
}

impl PluginGroup for Render3dPluginGroup {
    fn build(self) -> PluginGroupBuilder {
        let pg = PluginGroupBuilder::start::<Self>()
            .add(atmosphere::plugin::AtmospherePlugin)
            .add(decal::clustered::ClusteredDecalPlugin)
            .add(distance_fog::plugin::FogPlugin)
            .add(light_probe::plugin::LightProbePlugin)
            .add(lightmap::plugin::LightmapPlugin)
            .add(morph::MorphPlugin)
            .add(skin::SkinPlugin)
            .add(ssao::plugin::ScreenSpaceAmbientOcclusionPlugin)
            .add(ssr::plugin::ScreenSpaceReflectionsPlugin)
            .add(volumetric_fog::plugin::VolumetricFogPlugin)
            .add(mesh_pipeline::MeshRenderPlugin::new(self.debug_flags));

        #[cfg(not(feature = "meshlet"))]
        let pg = pg.add(dummy_meshlet::DummyMeshletPlugin);

        pg
    }
}

/// Many things can go wrong when attempting to use texture binding arrays
/// (a.k.a. bindless textures). This function checks for these pitfalls:
///
/// 1. If GLSL support is enabled at the feature level, then in debug mode
///    `naga_oil` will attempt to compile all shader modules under GLSL to check
///    validity of names, even if GLSL isn't actually used. This will cause a crash
///    if binding arrays are enabled, because binding arrays are currently
///    unimplemented in the GLSL backend of Naga. Therefore, we disable binding
///    arrays if the `shader_format_glsl` feature is present.
///
/// 2. If there aren't enough texture bindings available to accommodate all the
///    binding arrays, the driver will panic. So we also bail out if there aren't
///    enough texture bindings available in the fragment shader.
///
/// 3. If binding arrays aren't supported on the hardware, then we obviously
///    can't use them. Adreno <= 610 claims to support bindless, but seems to be
///    too buggy to be usable.
///
/// 4. If binding arrays are supported on the hardware, but they can only be
///    accessed by uniform indices, that's not good enough, and we bail out.
///
/// If binding arrays aren't usable, we disable reflection probes and limit the
/// number of irradiance volumes in the scene to 1.
pub(crate) fn binding_arrays_are_usable(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> bool {
    /// How many texture bindings are used in the fragment shader, *not* counting
    /// environment maps or irradiance volumes.
    const STANDARD_MATERIAL_FRAGMENT_SHADER_MIN_TEXTURE_BINDINGS: usize = 16;

    let Ok(maximum_bindings) = u32::try_from(
        STANDARD_MATERIAL_FRAGMENT_SHADER_MIN_TEXTURE_BINDINGS + MAX_VIEW_LIGHT_PROBES,
    ) else {
        unreachable!("Bindings should never reach more that {}.", u32::MAX);
    };

    !cfg!(feature = "shader_format_glsl")
        && bevy_render::get_adreno_model(render_adapter).is_none_or(|model| model > 610)
        && render_device.limits().max_storage_textures_per_shader_stage >= maximum_bindings
        && render_device.features().contains(
            WgpuFeatures::TEXTURE_BINDING_ARRAY
                | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        )
}

fn buffer_layout(
    buffer_binding_type: BufferBindingType,
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZero<u64>>,
) -> BindGroupLayoutEntryBuilder {
    match buffer_binding_type {
        BufferBindingType::Uniform => {
            binding_types::uniform_buffer_sized(has_dynamic_offset, min_binding_size)
        }
        BufferBindingType::Storage { read_only } => {
            if read_only {
                binding_types::storage_buffer_read_only_sized(has_dynamic_offset, min_binding_size)
            } else {
                binding_types::storage_buffer_sized(has_dynamic_offset, min_binding_size)
            }
        }
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

fn is_skinned(layout: &MeshVertexBufferLayoutRef) -> bool {
    layout.0.contains(Mesh::ATTRIBUTE_JOINT_INDEX)
        && layout.0.contains(Mesh::ATTRIBUTE_JOINT_WEIGHT)
}

fn setup_morph_and_skinning_defs(
    mesh_layouts: &MeshLayouts,
    layout: &MeshVertexBufferLayoutRef,
    offset: u32,
    key: &MeshPipelineKey,
    shader_defs: &mut Vec<ShaderDefVal>,
    vertex_attributes: &mut Vec<VertexAttributeDescriptor>,
    skins_use_uniform_buffers: bool,
) -> BindGroupLayout {
    let is_morphed = key.intersects(MeshPipelineKey::MORPH_TARGETS);
    let is_lightmapped = key.intersects(MeshPipelineKey::LIGHTMAPPED);
    let motion_vector_prepass = key.intersects(MeshPipelineKey::MOTION_VECTOR_PREPASS);

    if skins_use_uniform_buffers {
        shader_defs.push("SKINS_USE_UNIFORM_BUFFERS".into());
    }

    let mut add_skin_data = || {
        shader_defs.push("SKINNED".into());
        vertex_attributes.push(Mesh::ATTRIBUTE_JOINT_INDEX.at_shader_location(offset));
        vertex_attributes.push(Mesh::ATTRIBUTE_JOINT_WEIGHT.at_shader_location(offset + 1));
    };

    match (
        is_skinned(layout),
        is_morphed,
        is_lightmapped,
        motion_vector_prepass,
    ) {
        (true, false, _, true) => {
            add_skin_data();
            mesh_layouts.skinned_motion.clone()
        }
        (true, false, _, false) => {
            add_skin_data();
            mesh_layouts.skinned.clone()
        }
        (true, true, _, true) => {
            add_skin_data();
            shader_defs.push("MORPH_TARGETS".into());
            mesh_layouts.morphed_skinned_motion.clone()
        }
        (true, true, _, false) => {
            add_skin_data();
            shader_defs.push("MORPH_TARGETS".into());
            mesh_layouts.morphed_skinned.clone()
        }
        (false, true, _, true) => {
            shader_defs.push("MORPH_TARGETS".into());
            mesh_layouts.morphed_motion.clone()
        }
        (false, true, _, false) => {
            shader_defs.push("MORPH_TARGETS".into());
            mesh_layouts.morphed.clone()
        }
        (false, false, true, _) => mesh_layouts.lightmapped.clone(),
        (false, false, false, _) => mesh_layouts.model_only.clone(),
    }
}
