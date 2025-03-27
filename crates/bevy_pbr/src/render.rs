use bevy_asset::Handle;
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_math::{Mat3, Vec2, Vec3, Vec4};
use bevy_render::{
    alpha::AlphaMode,
    mesh::MeshVertexBufferLayoutRef,
    render_asset::RenderAssets,
    render_resource::{
        AsBindGroupShaderType, Face, RenderPipelineDescriptor, ShaderRef, ShaderType,
        SpecializedMeshPipelineError, TextureFormat,
    },
    texture::GpuImage,
};
use bevy_render_3d::{
    material::{MaterialPipeline, MaterialPipelineKey},
    mesh_pipeline::render_method::OpaqueRendererMethod,
    prelude::Material,
};

use crate::{
    plugin::{PBR_PREPASS_SHADER_HANDLE, PBR_SHADER_HANDLE},
    prelude::ParallaxMappingMethod,
    StandardMaterial, UvChannel,
};

// NOTE: These must match the bit flags in bevy_pbr/src/render/pbr_types.wgsl!
bitflags::bitflags! {
    /// Bitflags info about the material a shader is currently rendering.
    /// This is accessible in the shader in the [`StandardMaterialUniform`]
    #[repr(transparent)]
    pub struct StandardMaterialFlags: u32 {
        const BASE_COLOR_TEXTURE         = 1 << 0;
        const EMISSIVE_TEXTURE           = 1 << 1;
        const METALLIC_ROUGHNESS_TEXTURE = 1 << 2;
        const OCCLUSION_TEXTURE          = 1 << 3;
        const DOUBLE_SIDED               = 1 << 4;
        const UNLIT                      = 1 << 5;
        const TWO_COMPONENT_NORMAL_MAP   = 1 << 6;
        const FLIP_NORMAL_MAP_Y          = 1 << 7;
        const FOG_ENABLED                = 1 << 8;
        const DEPTH_MAP                  = 1 << 9; // Used for parallax mapping
        const SPECULAR_TRANSMISSION_TEXTURE = 1 << 10;
        const THICKNESS_TEXTURE          = 1 << 11;
        const DIFFUSE_TRANSMISSION_TEXTURE = 1 << 12;
        const ATTENUATION_ENABLED        = 1 << 13;
        const CLEARCOAT_TEXTURE          = 1 << 14;
        const CLEARCOAT_ROUGHNESS_TEXTURE = 1 << 15;
        const CLEARCOAT_NORMAL_TEXTURE   = 1 << 16;
        const ANISOTROPY_TEXTURE         = 1 << 17;
        const SPECULAR_TEXTURE           = 1 << 18;
        const SPECULAR_TINT_TEXTURE      = 1 << 19;
        const ALPHA_MODE_RESERVED_BITS   = Self::ALPHA_MODE_MASK_BITS << Self::ALPHA_MODE_SHIFT_BITS; // ← Bitmask reserving bits for the `AlphaMode`
        const ALPHA_MODE_OPAQUE          = 0 << Self::ALPHA_MODE_SHIFT_BITS;                          // ← Values are just sequential values bitshifted into
        const ALPHA_MODE_MASK            = 1 << Self::ALPHA_MODE_SHIFT_BITS;                          //   the bitmask, and can range from 0 to 7.
        const ALPHA_MODE_BLEND           = 2 << Self::ALPHA_MODE_SHIFT_BITS;                          //
        const ALPHA_MODE_PREMULTIPLIED   = 3 << Self::ALPHA_MODE_SHIFT_BITS;                          //
        const ALPHA_MODE_ADD             = 4 << Self::ALPHA_MODE_SHIFT_BITS;                          //   Right now only values 0–5 are used, which still gives
        const ALPHA_MODE_MULTIPLY        = 5 << Self::ALPHA_MODE_SHIFT_BITS;                          // ← us "room" for two more modes without adding more bits
        const ALPHA_MODE_ALPHA_TO_COVERAGE = 6 << Self::ALPHA_MODE_SHIFT_BITS;
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

impl StandardMaterialFlags {
    const ALPHA_MODE_MASK_BITS: u32 = 0b111;
    const ALPHA_MODE_SHIFT_BITS: u32 = 32 - Self::ALPHA_MODE_MASK_BITS.count_ones();
}

/// The GPU representation of the uniform data of a [`StandardMaterial`].
#[derive(Clone, Default, ShaderType)]
pub struct StandardMaterialUniform {
    /// Doubles as diffuse albedo for non-metallic, specular for metallic and a mix for everything
    /// in between.
    pub base_color: Vec4,
    // Use a color for user-friendliness even though we technically don't use the alpha channel
    // Might be used in the future for exposure correction in HDR
    pub emissive: Vec4,
    /// Color white light takes after traveling through the attenuation distance underneath the material surface
    pub attenuation_color: Vec4,
    /// The transform applied to the UVs corresponding to `ATTRIBUTE_UV_0` on the mesh before sampling. Default is identity.
    pub uv_transform: Mat3,
    /// Specular intensity for non-metals on a linear scale of [0.0, 1.0]
    /// defaults to 0.5 which is mapped to 4% reflectance in the shader
    pub reflectance: Vec3,
    /// Linear perceptual roughness, clamped to [0.089, 1.0] in the shader
    /// Defaults to minimum of 0.089
    pub roughness: f32,
    /// From [0.0, 1.0], dielectric to pure metallic
    pub metallic: f32,
    /// Amount of diffuse light transmitted through the material
    pub diffuse_transmission: f32,
    /// Amount of specular light transmitted through the material
    pub specular_transmission: f32,
    /// Thickness of the volume underneath the material surface
    pub thickness: f32,
    /// Index of Refraction
    pub ior: f32,
    /// How far light travels through the volume underneath the material surface before being absorbed
    pub attenuation_distance: f32,
    pub clearcoat: f32,
    pub clearcoat_perceptual_roughness: f32,
    pub anisotropy_strength: f32,
    pub anisotropy_rotation: Vec2,
    /// The [`StandardMaterialFlags`] accessible in the `wgsl` shader.
    pub flags: u32,
    /// When the alpha mode mask flag is set, any base color alpha above this cutoff means fully opaque,
    /// and any below means fully transparent.
    pub alpha_cutoff: f32,
    /// The depth of the [`StandardMaterial::depth_map`] to apply.
    pub parallax_depth_scale: f32,
    /// In how many layers to split the depth maps for Steep parallax mapping.
    ///
    /// If your `parallax_depth_scale` is >0.1 and you are seeing jaggy edges,
    /// increase this value. However, this incurs a performance cost.
    pub max_parallax_layer_count: f32,
    /// The exposure (brightness) level of the lightmap, if present.
    pub lightmap_exposure: f32,
    /// Using [`ParallaxMappingMethod::Relief`], how many additional
    /// steps to use at most to find the depth value.
    pub max_relief_mapping_search_steps: u32,
    /// ID for specifying which deferred lighting pass should be used for rendering this material, if any.
    pub deferred_lighting_pass_id: u32,
}

impl AsBindGroupShaderType<StandardMaterialUniform> for StandardMaterial {
    fn as_bind_group_shader_type(
        &self,
        images: &RenderAssets<GpuImage>,
    ) -> StandardMaterialUniform {
        let mut flags = StandardMaterialFlags::NONE;
        if self.base_color_texture.is_some() {
            flags |= StandardMaterialFlags::BASE_COLOR_TEXTURE;
        }
        if self.emissive_texture.is_some() {
            flags |= StandardMaterialFlags::EMISSIVE_TEXTURE;
        }
        if self.metallic_roughness_texture.is_some() {
            flags |= StandardMaterialFlags::METALLIC_ROUGHNESS_TEXTURE;
        }
        if self.occlusion_texture.is_some() {
            flags |= StandardMaterialFlags::OCCLUSION_TEXTURE;
        }
        if self.double_sided {
            flags |= StandardMaterialFlags::DOUBLE_SIDED;
        }
        if self.unlit {
            flags |= StandardMaterialFlags::UNLIT;
        }
        if self.fog_enabled {
            flags |= StandardMaterialFlags::FOG_ENABLED;
        }
        if self.depth_map.is_some() {
            flags |= StandardMaterialFlags::DEPTH_MAP;
        }
        #[cfg(feature = "pbr_transmission_textures")]
        {
            if self.specular_transmission_texture.is_some() {
                flags |= StandardMaterialFlags::SPECULAR_TRANSMISSION_TEXTURE;
            }
            if self.thickness_texture.is_some() {
                flags |= StandardMaterialFlags::THICKNESS_TEXTURE;
            }
            if self.diffuse_transmission_texture.is_some() {
                flags |= StandardMaterialFlags::DIFFUSE_TRANSMISSION_TEXTURE;
            }
        }

        #[cfg(feature = "pbr_anisotropy_texture")]
        {
            if self.anisotropy_texture.is_some() {
                flags |= StandardMaterialFlags::ANISOTROPY_TEXTURE;
            }
        }

        #[cfg(feature = "pbr_specular_textures")]
        {
            if self.specular_texture.is_some() {
                flags |= StandardMaterialFlags::SPECULAR_TEXTURE;
            }
            if self.specular_tint_texture.is_some() {
                flags |= StandardMaterialFlags::SPECULAR_TINT_TEXTURE;
            }
        }

        #[cfg(feature = "pbr_multi_layer_material_textures")]
        {
            if self.clearcoat_texture.is_some() {
                flags |= StandardMaterialFlags::CLEARCOAT_TEXTURE;
            }
            if self.clearcoat_roughness_texture.is_some() {
                flags |= StandardMaterialFlags::CLEARCOAT_ROUGHNESS_TEXTURE;
            }
            if self.clearcoat_normal_texture.is_some() {
                flags |= StandardMaterialFlags::CLEARCOAT_NORMAL_TEXTURE;
            }
        }

        let has_normal_map = self.normal_map_texture.is_some();
        if has_normal_map {
            let normal_map_id = self.normal_map_texture.as_ref().map(Handle::id).unwrap();
            if let Some(texture) = images.get(normal_map_id) {
                match texture.texture_format {
                    // All 2-component unorm formats
                    TextureFormat::Rg8Unorm
                    | TextureFormat::Rg16Unorm
                    | TextureFormat::Bc5RgUnorm
                    | TextureFormat::EacRg11Unorm => {
                        flags |= StandardMaterialFlags::TWO_COMPONENT_NORMAL_MAP;
                    }
                    _ => {}
                }
            }
            if self.flip_normal_map_y {
                flags |= StandardMaterialFlags::FLIP_NORMAL_MAP_Y;
            }
        }
        // NOTE: 0.5 is from the glTF default - do we want this?
        let mut alpha_cutoff = 0.5;
        match self.alpha_mode {
            AlphaMode::Opaque => flags |= StandardMaterialFlags::ALPHA_MODE_OPAQUE,
            AlphaMode::Mask(c) => {
                alpha_cutoff = c;
                flags |= StandardMaterialFlags::ALPHA_MODE_MASK;
            }
            AlphaMode::Blend => flags |= StandardMaterialFlags::ALPHA_MODE_BLEND,
            AlphaMode::Premultiplied => flags |= StandardMaterialFlags::ALPHA_MODE_PREMULTIPLIED,
            AlphaMode::Add => flags |= StandardMaterialFlags::ALPHA_MODE_ADD,
            AlphaMode::Multiply => flags |= StandardMaterialFlags::ALPHA_MODE_MULTIPLY,
            AlphaMode::AlphaToCoverage => {
                flags |= StandardMaterialFlags::ALPHA_MODE_ALPHA_TO_COVERAGE;
            }
        };

        if self.attenuation_distance.is_finite() {
            flags |= StandardMaterialFlags::ATTENUATION_ENABLED;
        }

        let mut emissive = self.emissive.to_vec4();
        emissive[3] = self.emissive_exposure_weight;

        // Doing this up front saves having to do this repeatedly in the fragment shader.
        let anisotropy_rotation = Vec2::from_angle(self.anisotropy_rotation);

        StandardMaterialUniform {
            base_color: LinearRgba::from(self.base_color).to_vec4(),
            emissive,
            roughness: self.perceptual_roughness,
            metallic: self.metallic,
            reflectance: LinearRgba::from(self.specular_tint).to_vec3() * self.reflectance,
            clearcoat: self.clearcoat,
            clearcoat_perceptual_roughness: self.clearcoat_perceptual_roughness,
            anisotropy_strength: self.anisotropy_strength,
            anisotropy_rotation,
            diffuse_transmission: self.diffuse_transmission,
            specular_transmission: self.specular_transmission,
            thickness: self.thickness,
            ior: self.ior,
            attenuation_distance: self.attenuation_distance,
            attenuation_color: LinearRgba::from(self.attenuation_color)
                .to_f32_array()
                .into(),
            flags: flags.bits(),
            alpha_cutoff,
            parallax_depth_scale: self.parallax_depth_scale,
            max_parallax_layer_count: self.max_parallax_layer_count,
            lightmap_exposure: self.lightmap_exposure,
            max_relief_mapping_search_steps: self.parallax_mapping_method.max_steps(),
            deferred_lighting_pass_id: self.deferred_lighting_pass_id as u32,
            uv_transform: self.uv_transform.into(),
        }
    }
}

bitflags::bitflags! {
    /// The pipeline key for `StandardMaterial`, packed into 64 bits.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StandardMaterialKey: u64 {
        const CULL_FRONT               = 0x000001;
        const CULL_BACK                = 0x000002;
        const NORMAL_MAP               = 0x000004;
        const RELIEF_MAPPING           = 0x000008;
        const DIFFUSE_TRANSMISSION     = 0x000010;
        const SPECULAR_TRANSMISSION    = 0x000020;
        const CLEARCOAT                = 0x000040;
        const CLEARCOAT_NORMAL_MAP     = 0x000080;
        const ANISOTROPY               = 0x000100;
        const BASE_COLOR_UV            = 0x000200;
        const EMISSIVE_UV              = 0x000400;
        const METALLIC_ROUGHNESS_UV    = 0x000800;
        const OCCLUSION_UV             = 0x001000;
        const SPECULAR_TRANSMISSION_UV = 0x002000;
        const THICKNESS_UV             = 0x004000;
        const DIFFUSE_TRANSMISSION_UV  = 0x008000;
        const NORMAL_MAP_UV            = 0x010000;
        const ANISOTROPY_UV            = 0x020000;
        const CLEARCOAT_UV             = 0x040000;
        const CLEARCOAT_ROUGHNESS_UV   = 0x080000;
        const CLEARCOAT_NORMAL_UV      = 0x100000;
        const SPECULAR_UV              = 0x200000;
        const SPECULAR_TINT_UV         = 0x400000;
        const DEPTH_BIAS               = 0xffffffff_00000000;
    }
}

const STANDARD_MATERIAL_KEY_DEPTH_BIAS_SHIFT: u64 = 32;

impl From<&StandardMaterial> for StandardMaterialKey {
    fn from(material: &StandardMaterial) -> Self {
        let mut key = StandardMaterialKey::empty();
        key.set(
            StandardMaterialKey::CULL_FRONT,
            material.cull_mode == Some(Face::Front),
        );
        key.set(
            StandardMaterialKey::CULL_BACK,
            material.cull_mode == Some(Face::Back),
        );
        key.set(
            StandardMaterialKey::NORMAL_MAP,
            material.normal_map_texture.is_some(),
        );
        key.set(
            StandardMaterialKey::RELIEF_MAPPING,
            matches!(
                material.parallax_mapping_method,
                ParallaxMappingMethod::Relief { .. }
            ),
        );
        key.set(
            StandardMaterialKey::DIFFUSE_TRANSMISSION,
            material.diffuse_transmission > 0.0,
        );
        key.set(
            StandardMaterialKey::SPECULAR_TRANSMISSION,
            material.specular_transmission > 0.0,
        );

        key.set(StandardMaterialKey::CLEARCOAT, material.clearcoat > 0.0);

        #[cfg(feature = "pbr_multi_layer_material_textures")]
        key.set(
            StandardMaterialKey::CLEARCOAT_NORMAL_MAP,
            material.clearcoat > 0.0 && material.clearcoat_normal_texture.is_some(),
        );

        key.set(
            StandardMaterialKey::ANISOTROPY,
            material.anisotropy_strength > 0.0,
        );

        key.set(
            StandardMaterialKey::BASE_COLOR_UV,
            material.base_color_channel != UvChannel::Uv0,
        );

        key.set(
            StandardMaterialKey::EMISSIVE_UV,
            material.emissive_channel != UvChannel::Uv0,
        );
        key.set(
            StandardMaterialKey::METALLIC_ROUGHNESS_UV,
            material.metallic_roughness_channel != UvChannel::Uv0,
        );
        key.set(
            StandardMaterialKey::OCCLUSION_UV,
            material.occlusion_channel != UvChannel::Uv0,
        );
        #[cfg(feature = "pbr_transmission_textures")]
        {
            key.set(
                StandardMaterialKey::SPECULAR_TRANSMISSION_UV,
                material.specular_transmission_channel != UvChannel::Uv0,
            );
            key.set(
                StandardMaterialKey::THICKNESS_UV,
                material.thickness_channel != UvChannel::Uv0,
            );
            key.set(
                StandardMaterialKey::DIFFUSE_TRANSMISSION_UV,
                material.diffuse_transmission_channel != UvChannel::Uv0,
            );
        }

        key.set(
            StandardMaterialKey::NORMAL_MAP_UV,
            material.normal_map_channel != UvChannel::Uv0,
        );

        #[cfg(feature = "pbr_anisotropy_texture")]
        {
            key.set(
                StandardMaterialKey::ANISOTROPY_UV,
                material.anisotropy_channel != UvChannel::Uv0,
            );
        }

        #[cfg(feature = "pbr_specular_textures")]
        {
            key.set(
                StandardMaterialKey::SPECULAR_UV,
                material.specular_channel != UvChannel::Uv0,
            );
            key.set(
                StandardMaterialKey::SPECULAR_TINT_UV,
                material.specular_tint_channel != UvChannel::Uv0,
            );
        }

        #[cfg(feature = "pbr_multi_layer_material_textures")]
        {
            key.set(
                StandardMaterialKey::CLEARCOAT_UV,
                material.clearcoat_channel != UvChannel::Uv0,
            );
            key.set(
                StandardMaterialKey::CLEARCOAT_ROUGHNESS_UV,
                material.clearcoat_roughness_channel != UvChannel::Uv0,
            );
            key.set(
                StandardMaterialKey::CLEARCOAT_NORMAL_UV,
                material.clearcoat_normal_channel != UvChannel::Uv0,
            );
        }

        key.insert(StandardMaterialKey::from_bits_retain(
            // Casting to i32 first to ensure the full i32 range is preserved.
            // (wgpu expects the depth_bias as an i32 when this is extracted in a later step)
            (material.depth_bias as i32 as u64) << STANDARD_MATERIAL_KEY_DEPTH_BIAS_SHIFT,
        ));
        key
    }
}

impl Material for StandardMaterial {
    fn fragment_shader() -> ShaderRef {
        PBR_SHADER_HANDLE.into()
    }

    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    #[inline]
    fn opaque_render_method(&self) -> OpaqueRendererMethod {
        match self.opaque_render_method {
            // For now, diffuse transmission doesn't work under deferred rendering as we don't pack
            // the required data into the GBuffer. If this material is set to `Auto`, we report it as
            // `Forward` so that it's rendered correctly, even when the `DefaultOpaqueRendererMethod`
            // is set to `Deferred`.
            //
            // If the developer explicitly sets the `OpaqueRendererMethod` to `Deferred`, we assume
            // they know what they're doing and don't override it.
            OpaqueRendererMethod::Auto if self.diffuse_transmission > 0.0 => {
                OpaqueRendererMethod::Forward
            }
            other => other,
        }
    }

    #[inline]
    fn depth_bias(&self) -> f32 {
        self.depth_bias
    }

    #[inline]
    fn reads_view_transmission_texture(&self) -> bool {
        self.specular_transmission > 0.0
    }

    fn prepass_fragment_shader() -> ShaderRef {
        PBR_PREPASS_SHADER_HANDLE.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        PBR_SHADER_HANDLE.into()
    }

    #[cfg(feature = "meshlet")]
    fn meshlet_mesh_fragment_shader() -> ShaderRef {
        Self::fragment_shader()
    }

    #[cfg(feature = "meshlet")]
    fn meshlet_mesh_prepass_fragment_shader() -> ShaderRef {
        Self::prepass_fragment_shader()
    }

    #[cfg(feature = "meshlet")]
    fn meshlet_mesh_deferred_fragment_shader() -> ShaderRef {
        Self::deferred_fragment_shader()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut() {
            let shader_defs = &mut fragment.shader_defs;

            for (flags, shader_def) in [
                (
                    StandardMaterialKey::NORMAL_MAP,
                    "STANDARD_MATERIAL_NORMAL_MAP",
                ),
                (StandardMaterialKey::RELIEF_MAPPING, "RELIEF_MAPPING"),
                (
                    StandardMaterialKey::DIFFUSE_TRANSMISSION,
                    "STANDARD_MATERIAL_DIFFUSE_TRANSMISSION",
                ),
                (
                    StandardMaterialKey::SPECULAR_TRANSMISSION,
                    "STANDARD_MATERIAL_SPECULAR_TRANSMISSION",
                ),
                (
                    StandardMaterialKey::DIFFUSE_TRANSMISSION
                        | StandardMaterialKey::SPECULAR_TRANSMISSION,
                    "STANDARD_MATERIAL_DIFFUSE_OR_SPECULAR_TRANSMISSION",
                ),
                (
                    StandardMaterialKey::CLEARCOAT,
                    "STANDARD_MATERIAL_CLEARCOAT",
                ),
                (
                    StandardMaterialKey::CLEARCOAT_NORMAL_MAP,
                    "STANDARD_MATERIAL_CLEARCOAT_NORMAL_MAP",
                ),
                (
                    StandardMaterialKey::ANISOTROPY,
                    "STANDARD_MATERIAL_ANISOTROPY",
                ),
                (
                    StandardMaterialKey::BASE_COLOR_UV,
                    "STANDARD_MATERIAL_BASE_COLOR_UV_B",
                ),
                (
                    StandardMaterialKey::EMISSIVE_UV,
                    "STANDARD_MATERIAL_EMISSIVE_UV_B",
                ),
                (
                    StandardMaterialKey::METALLIC_ROUGHNESS_UV,
                    "STANDARD_MATERIAL_METALLIC_ROUGHNESS_UV_B",
                ),
                (
                    StandardMaterialKey::OCCLUSION_UV,
                    "STANDARD_MATERIAL_OCCLUSION_UV_B",
                ),
                (
                    StandardMaterialKey::SPECULAR_TRANSMISSION_UV,
                    "STANDARD_MATERIAL_SPECULAR_TRANSMISSION_UV_B",
                ),
                (
                    StandardMaterialKey::THICKNESS_UV,
                    "STANDARD_MATERIAL_THICKNESS_UV_B",
                ),
                (
                    StandardMaterialKey::DIFFUSE_TRANSMISSION_UV,
                    "STANDARD_MATERIAL_DIFFUSE_TRANSMISSION_UV_B",
                ),
                (
                    StandardMaterialKey::NORMAL_MAP_UV,
                    "STANDARD_MATERIAL_NORMAL_MAP_UV_B",
                ),
                (
                    StandardMaterialKey::CLEARCOAT_UV,
                    "STANDARD_MATERIAL_CLEARCOAT_UV_B",
                ),
                (
                    StandardMaterialKey::CLEARCOAT_ROUGHNESS_UV,
                    "STANDARD_MATERIAL_CLEARCOAT_ROUGHNESS_UV_B",
                ),
                (
                    StandardMaterialKey::CLEARCOAT_NORMAL_UV,
                    "STANDARD_MATERIAL_CLEARCOAT_NORMAL_UV_B",
                ),
                (
                    StandardMaterialKey::ANISOTROPY_UV,
                    "STANDARD_MATERIAL_ANISOTROPY_UV_B",
                ),
                (
                    StandardMaterialKey::SPECULAR_UV,
                    "STANDARD_MATERIAL_SPECULAR_UV_B",
                ),
                (
                    StandardMaterialKey::SPECULAR_TINT_UV,
                    "STANDARD_MATERIAL_SPECULAR_TINT_UV_B",
                ),
            ] {
                if key.bind_group_data.intersects(flags) {
                    shader_defs.push(shader_def.into());
                }
            }
        }

        descriptor.primitive.cull_mode = if key
            .bind_group_data
            .contains(StandardMaterialKey::CULL_FRONT)
        {
            Some(Face::Front)
        } else if key.bind_group_data.contains(StandardMaterialKey::CULL_BACK) {
            Some(Face::Back)
        } else {
            None
        };

        if let Some(label) = &mut descriptor.label {
            *label = format!("pbr_{}", *label).into();
        }
        if let Some(depth_stencil) = descriptor.depth_stencil.as_mut() {
            depth_stencil.bias.constant =
                (key.bind_group_data.bits() >> STANDARD_MATERIAL_KEY_DEPTH_BIAS_SHIFT) as i32;
        }
        Ok(())
    }
}
