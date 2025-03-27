use bevy_asset::{AssetId, Handle};
use bevy_core_pipeline::{
    core_3d::{ScreenSpaceTransmissionQuality, CORE_3D_DEPTH_FORMAT},
    oit::{resolve::is_oit_supported, OrderIndependentTransparencySettings},
    prepass::ViewPrepassTextures,
    tonemapping::{get_lut_bind_group_layout_entries, Tonemapping},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    entity::Entity,
    resource::Resource,
    system::{lifetimeless::SRes, Res, SystemParamItem, SystemState},
    world::{FromWorld, World},
};
use bevy_image::{BevyDefault, Image, ImageSampler, TextureFormatPixelInfo};
use bevy_math::Vec4;
use bevy_mesh::{BaseMeshPipelineKey, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy_platform_support::sync::Arc;
use bevy_render::{
    batching::{
        gpu_preprocessing::{IndirectParametersCpuMetadata, UntypedPhaseIndirectParametersBuffers},
        GetBatchData, GetFullBatchData,
    },
    globals::GlobalsUniform,
    mesh::{allocator::MeshAllocator, RenderMesh},
    render_asset::RenderAssets,
    render_resource::{
        binding_types, BindGroupLayout, BindGroupLayoutEntry, BindingType, BlendComponent,
        BlendFactor, BlendOperation, BlendState, BufferBindingType, ColorTargetState, ColorWrites,
        CompareFunction, DepthBiasState, DepthStencilState, DynamicBindGroupLayoutEntries, Face,
        FragmentState, FrontFace, GpuArrayBuffer, MultisampleState, PolygonMode, PrimitiveState,
        RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderDefVal, ShaderStages,
        ShaderType, SpecializedMeshPipeline, SpecializedMeshPipelineError, StencilFaceState,
        StencilState, TexelCopyBufferLayout, TextureFormat, TextureSampleType, TextureView,
        TextureViewDescriptor, VertexState,
    },
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
    sync_world::MainEntity,
    texture::{DefaultImageSampler, GpuImage},
    view::{Msaa, ViewTarget, ViewUniform, VISIBILITY_RANGES_STORAGE_BUFFER_COUNT},
};

use bevy_utils::once;
use nonmax::NonMaxU32;
use tracing::{error, warn};

use crate::{
    binding_arrays_are_usable, buffer_layout,
    cluster::{
        cluster::ViewClusterBindings, clusterable_objects::GpuClusterableObjects,
        CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT,
    },
    clustered_decals_are_usable, decal,
    distance_fog::fog::GpuFog,
    light::render::GpuLights,
    light_probe::{
        environment_map, irradiance_volume, light_probes::LightProbesUniform,
        IRRADIANCE_VOLUMES_ARE_USABLE,
    },
    lightmap::{LightmapSlabIndex, RenderLightmaps},
    material::MaterialBindGroupIndex,
    mesh_pipeline::{MESH_PIPELINE_VIEW_LAYOUT_SAFE_MAX_TEXTURES, MESH_SHADER_HANDLE},
    prepass, setup_morph_and_skinning_defs,
    skin::uniforms::SkinUniforms,
    ssr::render::ScreenSpaceReflectionsUniform,
};

use super::{
    instance::RenderMeshInstances, MeshInputUniform, MeshLayouts, MeshUniform,
    TONEMAPPING_LUT_SAMPLER_BINDING_INDEX, TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
};

/// All data needed to construct a pipeline for rendering 3D meshes.
#[derive(Resource, Clone)]
pub struct MeshPipeline {
    /// A reference to all the mesh pipeline view layouts.
    pub view_layouts: MeshPipelineViewLayouts,
    // This dummy white texture is to be used in place of optional StandardMaterial textures
    pub dummy_white_gpu_image: GpuImage,
    pub clustered_forward_buffer_binding_type: BufferBindingType,
    pub mesh_layouts: MeshLayouts,
    /// `MeshUniform`s are stored in arrays in buffers. If storage buffers are available, they
    /// are used and this will be `None`, otherwise uniform buffers will be used with batches
    /// of this many `MeshUniform`s, stored at dynamic offsets within the uniform buffer.
    /// Use code like this in custom shaders:
    /// ```wgsl
    /// ##ifdef PER_OBJECT_BUFFER_BATCH_SIZE
    /// @group(1) @binding(0) var<uniform> mesh: array<Mesh, #{PER_OBJECT_BUFFER_BATCH_SIZE}u>;
    /// ##else
    /// @group(1) @binding(0) var<storage> mesh: array<Mesh>;
    /// ##endif // PER_OBJECT_BUFFER_BATCH_SIZE
    /// ```
    pub per_object_buffer_batch_size: Option<u32>,

    /// Whether binding arrays (a.k.a. bindless textures) are usable on the
    /// current render device.
    ///
    /// This affects whether reflection probes can be used.
    pub binding_arrays_are_usable: bool,

    /// Whether clustered decals are usable on the current render device.
    pub clustered_decals_are_usable: bool,

    /// Whether skins will use uniform buffers on account of storage buffers
    /// being unavailable on this platform.
    pub skins_use_uniform_buffers: bool,
}

impl FromWorld for MeshPipeline {
    fn from_world(world: &mut World) -> Self {
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<RenderAdapter>,
            Res<DefaultImageSampler>,
            Res<RenderQueue>,
            Res<MeshPipelineViewLayouts>,
        )> = SystemState::new(world);
        let (render_device, render_adapter, default_sampler, render_queue, view_layouts) =
            system_state.get_mut(world);

        let clustered_forward_buffer_binding_type = render_device
            .get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT);

        // A 1x1x1 'all 1.0' texture to use as a dummy texture to use in place of optional StandardMaterial textures
        let dummy_white_gpu_image = {
            let image = Image::default();
            let texture = render_device.create_texture(&image.texture_descriptor);
            let sampler = match image.sampler {
                ImageSampler::Default => (**default_sampler).clone(),
                ImageSampler::Descriptor(ref descriptor) => {
                    render_device.create_sampler(&descriptor.as_wgpu())
                }
            };

            let format_size = image.texture_descriptor.format.pixel_size();
            render_queue.write_texture(
                texture.as_image_copy(),
                image.data.as_ref().expect("Image was created without data"),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(image.width() * format_size as u32),
                    rows_per_image: None,
                },
                image.texture_descriptor.size,
            );

            let texture_view = texture.create_view(&TextureViewDescriptor::default());
            GpuImage {
                texture,
                texture_view,
                texture_format: image.texture_descriptor.format,
                sampler,
                size: image.texture_descriptor.size,
                mip_level_count: image.texture_descriptor.mip_level_count,
            }
        };

        MeshPipeline {
            view_layouts: view_layouts.clone(),
            clustered_forward_buffer_binding_type,
            dummy_white_gpu_image,
            mesh_layouts: MeshLayouts::new(&render_device, &render_adapter),
            per_object_buffer_batch_size: GpuArrayBuffer::<MeshUniform>::batch_size(&render_device),
            binding_arrays_are_usable: binding_arrays_are_usable(&render_device, &render_adapter),
            clustered_decals_are_usable: clustered_decals_are_usable(
                &render_device,
                &render_adapter,
            ),
            skins_use_uniform_buffers: crate::skin::uniforms::skins_use_uniform_buffers(
                &render_device,
            ),
        }
    }
}

impl MeshPipeline {
    pub fn get_image_texture<'a>(
        &'a self,
        gpu_images: &'a RenderAssets<GpuImage>,
        handle_option: &Option<Handle<Image>>,
    ) -> Option<(&'a TextureView, &'a Sampler)> {
        if let Some(handle) = handle_option {
            let gpu_image = gpu_images.get(handle)?;
            Some((&gpu_image.texture_view, &gpu_image.sampler))
        } else {
            Some((
                &self.dummy_white_gpu_image.texture_view,
                &self.dummy_white_gpu_image.sampler,
            ))
        }
    }

    pub fn get_view_layout(&self, layout_key: MeshPipelineViewLayoutKey) -> &BindGroupLayout {
        self.view_layouts.get_view_layout(layout_key)
    }
}

impl GetBatchData for MeshPipeline {
    type Param = (
        SRes<RenderMeshInstances>,
        SRes<RenderLightmaps>,
        SRes<RenderAssets<RenderMesh>>,
        SRes<MeshAllocator>,
        SRes<SkinUniforms>,
    );
    // The material bind group ID, the mesh ID, and the lightmap ID,
    // respectively.
    type CompareData = (
        MaterialBindGroupIndex,
        AssetId<Mesh>,
        Option<LightmapSlabIndex>,
    );

    type BufferData = MeshUniform;

    fn get_batch_data(
        (mesh_instances, lightmaps, _, mesh_allocator, skin_uniforms): &SystemParamItem<
            Self::Param,
        >,
        (_entity, main_entity): (Entity, MainEntity),
    ) -> Option<(Self::BufferData, Option<Self::CompareData>)> {
        let RenderMeshInstances::CpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_batch_data` should never be called in GPU mesh uniform \
                building mode"
            );
            return None;
        };
        let mesh_instance = mesh_instances.get(&main_entity)?;
        let first_vertex_index =
            match mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id) {
                Some(mesh_vertex_slice) => mesh_vertex_slice.range.start,
                None => 0,
            };
        let maybe_lightmap = lightmaps.render_lightmaps.get(&main_entity);

        let current_skin_index = skin_uniforms.skin_index(main_entity);
        let material_bind_group_index = mesh_instance.material_bindings_index;

        Some((
            MeshUniform::new(
                &mesh_instance.transforms,
                first_vertex_index,
                material_bind_group_index.slot,
                maybe_lightmap.map(|lightmap| (lightmap.slot_index, lightmap.uv_rect)),
                current_skin_index,
                Some(mesh_instance.tag),
            ),
            mesh_instance.should_batch().then_some((
                material_bind_group_index.group,
                mesh_instance.mesh_asset_id,
                maybe_lightmap.map(|lightmap| lightmap.slab_index),
            )),
        ))
    }
}

impl GetFullBatchData for MeshPipeline {
    type BufferInputData = MeshInputUniform;

    fn get_index_and_compare_data(
        (mesh_instances, lightmaps, _, _, _): &SystemParamItem<Self::Param>,
        main_entity: MainEntity,
    ) -> Option<(NonMaxU32, Option<Self::CompareData>)> {
        // This should only be called during GPU building.
        let RenderMeshInstances::GpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_index_and_compare_data` should never be called in CPU mesh uniform building \
                mode"
            );
            return None;
        };

        let mesh_instance = mesh_instances.get(&main_entity)?;
        let maybe_lightmap = lightmaps.render_lightmaps.get(&main_entity);

        Some((
            mesh_instance.current_uniform_index,
            mesh_instance.should_batch().then_some((
                mesh_instance.material_bindings_index.group,
                mesh_instance.mesh_asset_id,
                maybe_lightmap.map(|lightmap| lightmap.slab_index),
            )),
        ))
    }

    fn get_binned_batch_data(
        (mesh_instances, lightmaps, _, mesh_allocator, skin_uniforms): &SystemParamItem<
            Self::Param,
        >,
        main_entity: MainEntity,
    ) -> Option<Self::BufferData> {
        let RenderMeshInstances::CpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_binned_batch_data` should never be called in GPU mesh uniform building mode"
            );
            return None;
        };
        let mesh_instance = mesh_instances.get(&main_entity)?;
        let first_vertex_index =
            match mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id) {
                Some(mesh_vertex_slice) => mesh_vertex_slice.range.start,
                None => 0,
            };
        let maybe_lightmap = lightmaps.render_lightmaps.get(&main_entity);

        let current_skin_index = skin_uniforms.skin_index(main_entity);

        Some(MeshUniform::new(
            &mesh_instance.transforms,
            first_vertex_index,
            mesh_instance.material_bindings_index.slot,
            maybe_lightmap.map(|lightmap| (lightmap.slot_index, lightmap.uv_rect)),
            current_skin_index,
            Some(mesh_instance.tag),
        ))
    }

    fn get_binned_index(
        (mesh_instances, _, _, _, _): &SystemParamItem<Self::Param>,
        main_entity: MainEntity,
    ) -> Option<NonMaxU32> {
        // This should only be called during GPU building.
        let RenderMeshInstances::GpuBuilding(ref mesh_instances) = **mesh_instances else {
            error!(
                "`get_binned_index` should never be called in CPU mesh uniform \
                building mode"
            );
            return None;
        };

        mesh_instances
            .get(&main_entity)
            .map(|entity| entity.current_uniform_index)
    }

    fn write_batch_indirect_parameters_metadata(
        indexed: bool,
        base_output_index: u32,
        batch_set_index: Option<NonMaxU32>,
        phase_indirect_parameters_buffers: &mut UntypedPhaseIndirectParametersBuffers,
        indirect_parameters_offset: u32,
    ) {
        let indirect_parameters = IndirectParametersCpuMetadata {
            base_output_index,
            batch_set_index: match batch_set_index {
                Some(batch_set_index) => u32::from(batch_set_index),
                None => !0,
            },
        };

        if indexed {
            phase_indirect_parameters_buffers
                .indexed
                .set(indirect_parameters_offset, indirect_parameters);
        } else {
            phase_indirect_parameters_buffers
                .non_indexed
                .set(indirect_parameters_offset, indirect_parameters);
        }
    }
}

impl SpecializedMeshPipeline for MeshPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut shader_defs = Vec::new();
        let mut vertex_attributes = Vec::new();

        // Let the shader code know that it's running in a mesh pipeline.
        shader_defs.push("MESH_PIPELINE".into());

        shader_defs.push("VERTEX_OUTPUT_INSTANCE_INDEX".into());

        if layout.0.contains(Mesh::ATTRIBUTE_POSITION) {
            shader_defs.push("VERTEX_POSITIONS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_POSITION.at_shader_location(0));
        }

        if layout.0.contains(Mesh::ATTRIBUTE_NORMAL) {
            shader_defs.push("VERTEX_NORMALS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_NORMAL.at_shader_location(1));
        }

        if layout.0.contains(Mesh::ATTRIBUTE_UV_0) {
            shader_defs.push("VERTEX_UVS".into());
            shader_defs.push("VERTEX_UVS_A".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
        }

        if layout.0.contains(Mesh::ATTRIBUTE_UV_1) {
            shader_defs.push("VERTEX_UVS".into());
            shader_defs.push("VERTEX_UVS_B".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_UV_1.at_shader_location(3));
        }

        if layout.0.contains(Mesh::ATTRIBUTE_TANGENT) {
            shader_defs.push("VERTEX_TANGENTS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_TANGENT.at_shader_location(4));
        }

        if layout.0.contains(Mesh::ATTRIBUTE_COLOR) {
            shader_defs.push("VERTEX_COLORS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_COLOR.at_shader_location(5));
        }

        if cfg!(feature = "pbr_transmission_textures") {
            shader_defs.push("PBR_TRANSMISSION_TEXTURES_SUPPORTED".into());
        }
        if cfg!(feature = "pbr_multi_layer_material_textures") {
            shader_defs.push("PBR_MULTI_LAYER_MATERIAL_TEXTURES_SUPPORTED".into());
        }
        if cfg!(feature = "pbr_anisotropy_texture") {
            shader_defs.push("PBR_ANISOTROPY_TEXTURE_SUPPORTED".into());
        }
        if cfg!(feature = "pbr_specular_textures") {
            shader_defs.push("PBR_SPECULAR_TEXTURES_SUPPORTED".into());
        }

        let mut bind_group_layout = vec![self.get_view_layout(key.into()).clone()];

        if key.msaa_samples() > 1 {
            shader_defs.push("MULTISAMPLED".into());
        };

        bind_group_layout.push(setup_morph_and_skinning_defs(
            &self.mesh_layouts,
            layout,
            6,
            &key,
            &mut shader_defs,
            &mut vertex_attributes,
            self.skins_use_uniform_buffers,
        ));

        if key.contains(MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
            shader_defs.push("SCREEN_SPACE_AMBIENT_OCCLUSION".into());
        }

        let vertex_buffer_layout = layout.0.get_layout(&vertex_attributes)?;

        let (label, blend, depth_write_enabled);
        let pass = key.intersection(MeshPipelineKey::BLEND_RESERVED_BITS);
        let (mut is_opaque, mut alpha_to_coverage_enabled) = (false, false);
        if key.contains(MeshPipelineKey::OIT_ENABLED) && pass == MeshPipelineKey::BLEND_ALPHA {
            label = "oit_mesh_pipeline".into();
            // TODO tail blending would need alpha blending
            blend = None;
            shader_defs.push("OIT_ENABLED".into());
            // TODO it should be possible to use this to combine MSAA and OIT
            // alpha_to_coverage_enabled = true;
            depth_write_enabled = false;
        } else if pass == MeshPipelineKey::BLEND_ALPHA {
            label = "alpha_blend_mesh_pipeline".into();
            blend = Some(BlendState::ALPHA_BLENDING);
            // For the transparent pass, fragments that are closer will be alpha blended
            // but their depth is not written to the depth buffer
            depth_write_enabled = false;
        } else if pass == MeshPipelineKey::BLEND_PREMULTIPLIED_ALPHA {
            label = "premultiplied_alpha_mesh_pipeline".into();
            blend = Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING);
            shader_defs.push("PREMULTIPLY_ALPHA".into());
            shader_defs.push("BLEND_PREMULTIPLIED_ALPHA".into());
            // For the transparent pass, fragments that are closer will be alpha blended
            // but their depth is not written to the depth buffer
            depth_write_enabled = false;
        } else if pass == MeshPipelineKey::BLEND_MULTIPLY {
            label = "multiply_mesh_pipeline".into();
            blend = Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::Dst,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent::OVER,
            });
            shader_defs.push("PREMULTIPLY_ALPHA".into());
            shader_defs.push("BLEND_MULTIPLY".into());
            // For the multiply pass, fragments that are closer will be alpha blended
            // but their depth is not written to the depth buffer
            depth_write_enabled = false;
        } else if pass == MeshPipelineKey::BLEND_ALPHA_TO_COVERAGE {
            label = "alpha_to_coverage_mesh_pipeline".into();
            // BlendState::REPLACE is not needed here, and None will be potentially much faster in some cases
            blend = None;
            // For the opaque and alpha mask passes, fragments that are closer will replace
            // the current fragment value in the output and the depth is written to the
            // depth buffer
            depth_write_enabled = true;
            is_opaque = !key.contains(MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE);
            alpha_to_coverage_enabled = true;
            shader_defs.push("ALPHA_TO_COVERAGE".into());
        } else {
            label = "opaque_mesh_pipeline".into();
            // BlendState::REPLACE is not needed here, and None will be potentially much faster in some cases
            blend = None;
            // For the opaque and alpha mask passes, fragments that are closer will replace
            // the current fragment value in the output and the depth is written to the
            // depth buffer
            depth_write_enabled = true;
            is_opaque = !key.contains(MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE);
        }

        if key.contains(MeshPipelineKey::NORMAL_PREPASS) {
            shader_defs.push("NORMAL_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            shader_defs.push("MOTION_VECTOR_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::HAS_PREVIOUS_SKIN) {
            shader_defs.push("HAS_PREVIOUS_SKIN".into());
        }

        if key.contains(MeshPipelineKey::HAS_PREVIOUS_MORPH) {
            shader_defs.push("HAS_PREVIOUS_MORPH".into());
        }

        if key.contains(MeshPipelineKey::DEFERRED_PREPASS) {
            shader_defs.push("DEFERRED_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::NORMAL_PREPASS) && key.msaa_samples() == 1 && is_opaque {
            shader_defs.push("LOAD_PREPASS_NORMALS".into());
        }

        let view_projection = key.intersection(MeshPipelineKey::VIEW_PROJECTION_RESERVED_BITS);
        if view_projection == MeshPipelineKey::VIEW_PROJECTION_NONSTANDARD {
            shader_defs.push("VIEW_PROJECTION_NONSTANDARD".into());
        } else if view_projection == MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE {
            shader_defs.push("VIEW_PROJECTION_PERSPECTIVE".into());
        } else if view_projection == MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC {
            shader_defs.push("VIEW_PROJECTION_ORTHOGRAPHIC".into());
        }

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        shader_defs.push("WEBGL2".into());

        #[cfg(feature = "experimental_pbr_pcss")]
        shader_defs.push("PCSS_SAMPLERS_AVAILABLE".into());

        if key.contains(MeshPipelineKey::TONEMAP_IN_SHADER) {
            shader_defs.push("TONEMAP_IN_SHADER".into());
            shader_defs.push(ShaderDefVal::UInt(
                "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
                TONEMAPPING_LUT_TEXTURE_BINDING_INDEX,
            ));
            shader_defs.push(ShaderDefVal::UInt(
                "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
                TONEMAPPING_LUT_SAMPLER_BINDING_INDEX,
            ));

            let method = key.intersection(MeshPipelineKey::TONEMAP_METHOD_RESERVED_BITS);

            if method == MeshPipelineKey::TONEMAP_METHOD_NONE {
                shader_defs.push("TONEMAP_METHOD_NONE".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD {
                shader_defs.push("TONEMAP_METHOD_REINHARD".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED {
                shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_AGX {
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC {
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE {
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }

            // Debanding is tied to tonemapping in the shader, cannot run without it.
            if key.contains(MeshPipelineKey::DEBAND_DITHER) {
                shader_defs.push("DEBAND_DITHER".into());
            }
        }

        if key.contains(MeshPipelineKey::MAY_DISCARD) {
            shader_defs.push("MAY_DISCARD".into());
        }

        if key.contains(MeshPipelineKey::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }

        if key.contains(MeshPipelineKey::IRRADIANCE_VOLUME) && IRRADIANCE_VOLUMES_ARE_USABLE {
            shader_defs.push("IRRADIANCE_VOLUME".into());
        }

        if key.contains(MeshPipelineKey::LIGHTMAPPED) {
            shader_defs.push("LIGHTMAP".into());
        }
        if key.contains(MeshPipelineKey::LIGHTMAP_BICUBIC_SAMPLING) {
            shader_defs.push("LIGHTMAP_BICUBIC_SAMPLING".into());
        }

        if key.contains(MeshPipelineKey::TEMPORAL_JITTER) {
            shader_defs.push("TEMPORAL_JITTER".into());
        }

        let shadow_filter_method =
            key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
        if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN {
            shader_defs.push("SHADOW_FILTER_METHOD_GAUSSIAN".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL {
            shader_defs.push("SHADOW_FILTER_METHOD_TEMPORAL".into());
        }

        let blur_quality =
            key.intersection(MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_RESERVED_BITS);

        shader_defs.push(ShaderDefVal::Int(
            "SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS".into(),
            match blur_quality {
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_LOW => 4,
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_MEDIUM => 8,
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_HIGH => 16,
                MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_ULTRA => 32,
                _ => unreachable!(), // Not possible, since the mask is 2 bits, and we've covered all 4 cases
            },
        ));

        if key.contains(MeshPipelineKey::VISIBILITY_RANGE_DITHER) {
            shader_defs.push("VISIBILITY_RANGE_DITHER".into());
        }

        if key.contains(MeshPipelineKey::DISTANCE_FOG) {
            shader_defs.push("DISTANCE_FOG".into());
        }

        if self.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
            shader_defs.push("MULTIPLE_LIGHTMAPS_IN_ARRAY".into());
        }

        if IRRADIANCE_VOLUMES_ARE_USABLE {
            shader_defs.push("IRRADIANCE_VOLUMES_ARE_USABLE".into());
        }

        if self.clustered_decals_are_usable {
            shader_defs.push("CLUSTERED_DECALS_ARE_USABLE".into());
        }

        let format = if key.contains(MeshPipelineKey::HDR) {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        // This is defined here so that custom shaders that use something other than
        // the mesh binding from bevy_pbr::mesh_bindings can easily make use of this
        // in their own shaders.
        if let Some(per_object_buffer_batch_size) = self.per_object_buffer_batch_size {
            shader_defs.push(ShaderDefVal::UInt(
                "PER_OBJECT_BUFFER_BATCH_SIZE".into(),
                per_object_buffer_batch_size,
            ));
        }

        Ok(RenderPipelineDescriptor {
            vertex: VertexState {
                shader: MESH_SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: MESH_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout: bind_group_layout,
            push_constant_ranges: vec![],
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: key.primitive_topology(),
                strip_index_format: None,
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled,
            },
            label: Some(label),
            zero_initialize_workgroup_memory: false,
        })
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    // NOTE: Apparently quadro drivers support up to 64x MSAA.
    /// MSAA uses the highest 3 bits for the MSAA log2(sample count) to support up to 128x MSAA.
    pub struct MeshPipelineKey: u64 {
        // Nothing
        const NONE                              = 0;

        // Inherited bits
        const MORPH_TARGETS                     = BaseMeshPipelineKey::MORPH_TARGETS.bits();

        // Flag bits
        const HDR                               = 1 << 0;
        const TONEMAP_IN_SHADER                 = 1 << 1;
        const DEBAND_DITHER                     = 1 << 2;
        const DEPTH_PREPASS                     = 1 << 3;
        const NORMAL_PREPASS                    = 1 << 4;
        const DEFERRED_PREPASS                  = 1 << 5;
        const MOTION_VECTOR_PREPASS             = 1 << 6;
        const MAY_DISCARD                       = 1 << 7; // Guards shader codepaths that may discard, allowing early depth tests in most cases
                                                            // See: https://www.khronos.org/opengl/wiki/Early_Fragment_Test
        const ENVIRONMENT_MAP                   = 1 << 8;
        const SCREEN_SPACE_AMBIENT_OCCLUSION    = 1 << 9;
        const UNCLIPPED_DEPTH_ORTHO             = 1 << 10; // Disables depth clipping for use with directional light shadow views
                                                            // Emulated via fragment shader depth on hardware that doesn't support it natively
                                                            // See: https://www.w3.org/TR/webgpu/#depth-clipping and https://therealmjp.github.io/posts/shadow-maps/#disabling-z-clipping
        const TEMPORAL_JITTER                   = 1 << 11;
        const READS_VIEW_TRANSMISSION_TEXTURE   = 1 << 12;
        const LIGHTMAPPED                       = 1 << 13;
        const LIGHTMAP_BICUBIC_SAMPLING         = 1 << 14;
        const IRRADIANCE_VOLUME                 = 1 << 15;
        const VISIBILITY_RANGE_DITHER           = 1 << 16;
        const SCREEN_SPACE_REFLECTIONS          = 1 << 17;
        const HAS_PREVIOUS_SKIN                 = 1 << 18;
        const HAS_PREVIOUS_MORPH                = 1 << 19;
        const OIT_ENABLED                       = 1 << 20;
        const DISTANCE_FOG                      = 1 << 21;
        const LAST_FLAG                         = Self::DISTANCE_FOG.bits();

        // Bitfields
        const MSAA_RESERVED_BITS                = Self::MSAA_MASK_BITS << Self::MSAA_SHIFT_BITS;
        const BLEND_RESERVED_BITS               = Self::BLEND_MASK_BITS << Self::BLEND_SHIFT_BITS; // ← Bitmask reserving bits for the blend state
        const BLEND_OPAQUE                      = 0 << Self::BLEND_SHIFT_BITS;                     // ← Values are just sequential within the mask
        const BLEND_PREMULTIPLIED_ALPHA         = 1 << Self::BLEND_SHIFT_BITS;                     // ← As blend states is on 3 bits, it can range from 0 to 7
        const BLEND_MULTIPLY                    = 2 << Self::BLEND_SHIFT_BITS;                     // ← See `BLEND_MASK_BITS` for the number of bits available
        const BLEND_ALPHA                       = 3 << Self::BLEND_SHIFT_BITS;                     //
        const BLEND_ALPHA_TO_COVERAGE           = 4 << Self::BLEND_SHIFT_BITS;                     // ← We still have room for three more values without adding more bits
        const TONEMAP_METHOD_RESERVED_BITS      = Self::TONEMAP_METHOD_MASK_BITS << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_NONE               = 0 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_REINHARD           = 1 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_REINHARD_LUMINANCE = 2 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_ACES_FITTED        = 3 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_AGX                = 4 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM = 5 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_TONY_MC_MAPFACE    = 6 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const TONEMAP_METHOD_BLENDER_FILMIC     = 7 << Self::TONEMAP_METHOD_SHIFT_BITS;
        const SHADOW_FILTER_METHOD_RESERVED_BITS = Self::SHADOW_FILTER_METHOD_MASK_BITS << Self::SHADOW_FILTER_METHOD_SHIFT_BITS;
        const SHADOW_FILTER_METHOD_HARDWARE_2X2  = 0 << Self::SHADOW_FILTER_METHOD_SHIFT_BITS;
        const SHADOW_FILTER_METHOD_GAUSSIAN      = 1 << Self::SHADOW_FILTER_METHOD_SHIFT_BITS;
        const SHADOW_FILTER_METHOD_TEMPORAL      = 2 << Self::SHADOW_FILTER_METHOD_SHIFT_BITS;
        const VIEW_PROJECTION_RESERVED_BITS     = Self::VIEW_PROJECTION_MASK_BITS << Self::VIEW_PROJECTION_SHIFT_BITS;
        const VIEW_PROJECTION_NONSTANDARD       = 0 << Self::VIEW_PROJECTION_SHIFT_BITS;
        const VIEW_PROJECTION_PERSPECTIVE       = 1 << Self::VIEW_PROJECTION_SHIFT_BITS;
        const VIEW_PROJECTION_ORTHOGRAPHIC      = 2 << Self::VIEW_PROJECTION_SHIFT_BITS;
        const VIEW_PROJECTION_RESERVED          = 3 << Self::VIEW_PROJECTION_SHIFT_BITS;
        const SCREEN_SPACE_SPECULAR_TRANSMISSION_RESERVED_BITS = Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_MASK_BITS << Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_SHIFT_BITS;
        const SCREEN_SPACE_SPECULAR_TRANSMISSION_LOW    = 0 << Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_SHIFT_BITS;
        const SCREEN_SPACE_SPECULAR_TRANSMISSION_MEDIUM = 1 << Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_SHIFT_BITS;
        const SCREEN_SPACE_SPECULAR_TRANSMISSION_HIGH   = 2 << Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_SHIFT_BITS;
        const SCREEN_SPACE_SPECULAR_TRANSMISSION_ULTRA  = 3 << Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_SHIFT_BITS;
        const ALL_RESERVED_BITS =
            Self::BLEND_RESERVED_BITS.bits() |
            Self::MSAA_RESERVED_BITS.bits() |
            Self::TONEMAP_METHOD_RESERVED_BITS.bits() |
            Self::SHADOW_FILTER_METHOD_RESERVED_BITS.bits() |
            Self::VIEW_PROJECTION_RESERVED_BITS.bits() |
            Self::SCREEN_SPACE_SPECULAR_TRANSMISSION_RESERVED_BITS.bits();
    }
}

impl MeshPipelineKey {
    const MSAA_MASK_BITS: u64 = 0b111;
    const MSAA_SHIFT_BITS: u64 = Self::LAST_FLAG.bits().trailing_zeros() as u64 + 1;

    const BLEND_MASK_BITS: u64 = 0b111;
    const BLEND_SHIFT_BITS: u64 = Self::MSAA_MASK_BITS.count_ones() as u64 + Self::MSAA_SHIFT_BITS;

    const TONEMAP_METHOD_MASK_BITS: u64 = 0b111;
    const TONEMAP_METHOD_SHIFT_BITS: u64 =
        Self::BLEND_MASK_BITS.count_ones() as u64 + Self::BLEND_SHIFT_BITS;

    const SHADOW_FILTER_METHOD_MASK_BITS: u64 = 0b11;
    const SHADOW_FILTER_METHOD_SHIFT_BITS: u64 =
        Self::TONEMAP_METHOD_MASK_BITS.count_ones() as u64 + Self::TONEMAP_METHOD_SHIFT_BITS;

    const VIEW_PROJECTION_MASK_BITS: u64 = 0b11;
    const VIEW_PROJECTION_SHIFT_BITS: u64 = Self::SHADOW_FILTER_METHOD_MASK_BITS.count_ones()
        as u64
        + Self::SHADOW_FILTER_METHOD_SHIFT_BITS;

    const SCREEN_SPACE_SPECULAR_TRANSMISSION_MASK_BITS: u64 = 0b11;
    const SCREEN_SPACE_SPECULAR_TRANSMISSION_SHIFT_BITS: u64 =
        Self::VIEW_PROJECTION_MASK_BITS.count_ones() as u64 + Self::VIEW_PROJECTION_SHIFT_BITS;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits =
            (msaa_samples.trailing_zeros() as u64 & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        Self::from_bits_retain(msaa_bits)
    }

    pub fn from_hdr(hdr: bool) -> Self {
        if hdr {
            MeshPipelineKey::HDR
        } else {
            MeshPipelineKey::NONE
        }
    }

    pub fn msaa_samples(&self) -> u32 {
        1 << ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS)
    }

    pub fn from_primitive_topology(primitive_topology: PrimitiveTopology) -> Self {
        let primitive_topology_bits = ((primitive_topology as u64)
            & BaseMeshPipelineKey::PRIMITIVE_TOPOLOGY_MASK_BITS)
            << BaseMeshPipelineKey::PRIMITIVE_TOPOLOGY_SHIFT_BITS;
        Self::from_bits_retain(primitive_topology_bits)
    }

    pub fn primitive_topology(&self) -> PrimitiveTopology {
        let primitive_topology_bits = (self.bits()
            >> BaseMeshPipelineKey::PRIMITIVE_TOPOLOGY_SHIFT_BITS)
            & BaseMeshPipelineKey::PRIMITIVE_TOPOLOGY_MASK_BITS;
        match primitive_topology_bits {
            x if x == PrimitiveTopology::PointList as u64 => PrimitiveTopology::PointList,
            x if x == PrimitiveTopology::LineList as u64 => PrimitiveTopology::LineList,
            x if x == PrimitiveTopology::LineStrip as u64 => PrimitiveTopology::LineStrip,
            x if x == PrimitiveTopology::TriangleList as u64 => PrimitiveTopology::TriangleList,
            x if x == PrimitiveTopology::TriangleStrip as u64 => PrimitiveTopology::TriangleStrip,
            _ => PrimitiveTopology::default(),
        }
    }
}

/// Stores the view layouts for every combination of pipeline keys.
///
/// This is wrapped in an [`Arc`] so that it can be efficiently cloned and
/// placed inside specializable pipeline types.
#[derive(Resource, Clone, Deref, DerefMut)]
pub struct MeshPipelineViewLayouts(
    pub Arc<[MeshPipelineViewLayout; MeshPipelineViewLayoutKey::COUNT]>,
);

impl FromWorld for MeshPipelineViewLayouts {
    fn from_world(world: &mut World) -> Self {
        // Generates all possible view layouts for the mesh pipeline, based on all combinations of
        // [`MeshPipelineViewLayoutKey`] flags.

        let render_device = world.resource::<RenderDevice>();
        let render_adapter = world.resource::<RenderAdapter>();

        let clustered_forward_buffer_binding_type = render_device
            .get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT);
        let visibility_ranges_buffer_binding_type = render_device
            .get_supported_read_only_binding_type(VISIBILITY_RANGES_STORAGE_BUFFER_COUNT);

        Self(Arc::new(core::array::from_fn(|i| {
            let key = MeshPipelineViewLayoutKey::from_bits_truncate(i as u32);
            let entries = layout_entries(
                clustered_forward_buffer_binding_type,
                visibility_ranges_buffer_binding_type,
                key,
                render_device,
                render_adapter,
            );
            #[cfg(debug_assertions)]
            let texture_count: usize = entries
                .iter()
                .filter(|entry| matches!(entry.ty, BindingType::Texture { .. }))
                .count();

            MeshPipelineViewLayout {
                bind_group_layout: render_device
                    .create_bind_group_layout(key.label().as_str(), &entries),
                #[cfg(debug_assertions)]
                texture_count,
            }
        })))
    }
}

impl MeshPipelineViewLayouts {
    pub fn get_view_layout(&self, layout_key: MeshPipelineViewLayoutKey) -> &BindGroupLayout {
        let index = layout_key.bits() as usize;
        let layout = &self[index];

        #[cfg(debug_assertions)]
        if layout.texture_count > MESH_PIPELINE_VIEW_LAYOUT_SAFE_MAX_TEXTURES {
            // Issue our own warning here because Naga's error message is a bit cryptic in this situation
            once!(warn!("Too many textures in mesh pipeline view layout, this might cause us to hit `wgpu::Limits::max_sampled_textures_per_shader_stage` in some environments."));
        }

        &layout.bind_group_layout
    }
}

#[derive(Clone)]
pub struct MeshPipelineViewLayout {
    pub bind_group_layout: BindGroupLayout,

    #[cfg(debug_assertions)]
    pub texture_count: usize,
}

bitflags::bitflags! {
    /// A key that uniquely identifies a [`MeshPipelineViewLayout`].
    ///
    /// Used to generate all possible layouts for the mesh pipeline in [`generate_view_layouts`],
    /// so special care must be taken to not add too many flags, as the number of possible layouts
    /// will grow exponentially.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct MeshPipelineViewLayoutKey: u32 {
        const MULTISAMPLED                = 1 << 0;
        const DEPTH_PREPASS               = 1 << 1;
        const NORMAL_PREPASS              = 1 << 2;
        const MOTION_VECTOR_PREPASS       = 1 << 3;
        const DEFERRED_PREPASS            = 1 << 4;
        const OIT_ENABLED                 = 1 << 5;
    }
}

impl MeshPipelineViewLayoutKey {
    // The number of possible layouts
    pub const COUNT: usize = Self::all().bits() as usize + 1;

    /// Builds a unique label for each layout based on the flags
    pub fn label(&self) -> String {
        use MeshPipelineViewLayoutKey as Key;

        format!(
            "mesh_view_layout{}{}{}{}{}{}",
            self.contains(Key::MULTISAMPLED)
                .then_some("_multisampled")
                .unwrap_or_default(),
            self.contains(Key::DEPTH_PREPASS)
                .then_some("_depth")
                .unwrap_or_default(),
            self.contains(Key::NORMAL_PREPASS)
                .then_some("_normal")
                .unwrap_or_default(),
            self.contains(Key::MOTION_VECTOR_PREPASS)
                .then_some("_motion")
                .unwrap_or_default(),
            self.contains(Key::DEFERRED_PREPASS)
                .then_some("_deferred")
                .unwrap_or_default(),
            self.contains(Key::OIT_ENABLED)
                .then_some("_oit")
                .unwrap_or_default(),
        )
    }
}

impl From<MeshPipelineKey> for MeshPipelineViewLayoutKey {
    fn from(value: MeshPipelineKey) -> Self {
        let mut result = MeshPipelineViewLayoutKey::empty();

        if value.msaa_samples() > 1 {
            result |= MeshPipelineViewLayoutKey::MULTISAMPLED;
        }
        if value.contains(MeshPipelineKey::DEPTH_PREPASS) {
            result |= MeshPipelineViewLayoutKey::DEPTH_PREPASS;
        }
        if value.contains(MeshPipelineKey::NORMAL_PREPASS) {
            result |= MeshPipelineViewLayoutKey::NORMAL_PREPASS;
        }
        if value.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            result |= MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS;
        }
        if value.contains(MeshPipelineKey::DEFERRED_PREPASS) {
            result |= MeshPipelineViewLayoutKey::DEFERRED_PREPASS;
        }
        if value.contains(MeshPipelineKey::OIT_ENABLED) {
            result |= MeshPipelineViewLayoutKey::OIT_ENABLED;
        }

        result
    }
}

impl From<Msaa> for MeshPipelineViewLayoutKey {
    fn from(value: Msaa) -> Self {
        let mut result = MeshPipelineViewLayoutKey::empty();

        if value.samples() > 1 {
            result |= MeshPipelineViewLayoutKey::MULTISAMPLED;
        }

        result
    }
}

impl From<Option<&ViewPrepassTextures>> for MeshPipelineViewLayoutKey {
    fn from(value: Option<&ViewPrepassTextures>) -> Self {
        let mut result = MeshPipelineViewLayoutKey::empty();

        if let Some(prepass_textures) = value {
            if prepass_textures.depth.is_some() {
                result |= MeshPipelineViewLayoutKey::DEPTH_PREPASS;
            }
            if prepass_textures.normal.is_some() {
                result |= MeshPipelineViewLayoutKey::NORMAL_PREPASS;
            }
            if prepass_textures.motion_vectors.is_some() {
                result |= MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS;
            }
            if prepass_textures.deferred.is_some() {
                result |= MeshPipelineViewLayoutKey::DEFERRED_PREPASS;
            }
        }

        result
    }
}

/// Returns the appropriate bind group layout vec based on the parameters
fn layout_entries(
    clustered_forward_buffer_binding_type: BufferBindingType,
    visibility_ranges_buffer_binding_type: BufferBindingType,
    layout_key: MeshPipelineViewLayoutKey,
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> Vec<BindGroupLayoutEntry> {
    let mut entries = DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::FRAGMENT,
        (
            // View
            (
                0,
                binding_types::uniform_buffer::<ViewUniform>(true)
                    .visibility(ShaderStages::VERTEX_FRAGMENT),
            ),
            // Lights
            (1, binding_types::uniform_buffer::<GpuLights>(true)),
            // Point Shadow Texture Cube Array
            (
                2,
                #[cfg(all(
                    not(target_abi = "sim"),
                    any(
                        not(feature = "webgl"),
                        not(target_arch = "wasm32"),
                        feature = "webgpu"
                    )
                ))]
                binding_types::texture_cube_array(TextureSampleType::Depth),
                #[cfg(any(
                    target_abi = "sim",
                    all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu"))
                ))]
                texture_cube(TextureSampleType::Depth),
            ),
            // Point Shadow Texture Array Comparison Sampler
            (3, binding_types::sampler(SamplerBindingType::Comparison)),
            // Point Shadow Texture Array Linear Sampler
            #[cfg(feature = "experimental_pbr_pcss")]
            (4, binding_types::sampler(SamplerBindingType::Filtering)),
            // Directional Shadow Texture Array
            (
                5,
                #[cfg(any(
                    not(feature = "webgl"),
                    not(target_arch = "wasm32"),
                    feature = "webgpu"
                ))]
                binding_types::texture_2d_array(TextureSampleType::Depth),
                #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
                binding_types::texture_2d(TextureSampleType::Depth),
            ),
            // Directional Shadow Texture Array Comparison Sampler
            (6, binding_types::sampler(SamplerBindingType::Comparison)),
            // Directional Shadow Texture Array Linear Sampler
            #[cfg(feature = "experimental_pbr_pcss")]
            (7, binding_types::sampler(SamplerBindingType::Filtering)),
            // PointLights
            (
                8,
                buffer_layout(
                    clustered_forward_buffer_binding_type,
                    false,
                    Some(GpuClusterableObjects::min_size(
                        clustered_forward_buffer_binding_type,
                    )),
                ),
            ),
            // ClusteredLightIndexLists
            (
                9,
                buffer_layout(
                    clustered_forward_buffer_binding_type,
                    false,
                    Some(
                        ViewClusterBindings::min_size_clusterable_object_index_lists(
                            clustered_forward_buffer_binding_type,
                        ),
                    ),
                ),
            ),
            // ClusterOffsetsAndCounts
            (
                10,
                buffer_layout(
                    clustered_forward_buffer_binding_type,
                    false,
                    Some(ViewClusterBindings::min_size_cluster_offsets_and_counts(
                        clustered_forward_buffer_binding_type,
                    )),
                ),
            ),
            // Globals
            (
                11,
                binding_types::uniform_buffer::<GlobalsUniform>(false)
                    .visibility(ShaderStages::VERTEX_FRAGMENT),
            ),
            // Fog
            (12, binding_types::uniform_buffer::<GpuFog>(true)),
            // Light probes
            (
                13,
                binding_types::uniform_buffer::<LightProbesUniform>(true),
            ),
            // Visibility ranges
            (
                14,
                buffer_layout(
                    visibility_ranges_buffer_binding_type,
                    false,
                    Some(Vec4::min_size()),
                )
                .visibility(ShaderStages::VERTEX),
            ),
            // Screen space reflection settings
            (
                15,
                binding_types::uniform_buffer::<ScreenSpaceReflectionsUniform>(true),
            ),
            // Screen space ambient occlusion texture
            (
                16,
                binding_types::texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );

    // EnvironmentMapLight
    let environment_map_entries =
        environment_map::get_bind_group_layout_entries(render_device, render_adapter);
    entries = entries.extend_with_indices((
        (17, environment_map_entries[0]),
        (18, environment_map_entries[1]),
        (19, environment_map_entries[2]),
        (20, environment_map_entries[3]),
    ));

    // Irradiance volumes
    if IRRADIANCE_VOLUMES_ARE_USABLE {
        let irradiance_volume_entries =
            irradiance_volume::get_bind_group_layout_entries(render_device, render_adapter);
        entries = entries.extend_with_indices((
            (21, irradiance_volume_entries[0]),
            (22, irradiance_volume_entries[1]),
        ));
    }

    // Clustered decals
    if let Some(clustered_decal_entries) =
        decal::clustered::get_bind_group_layout_entries(render_device, render_adapter)
    {
        entries = entries.extend_with_indices((
            (23, clustered_decal_entries[0]),
            (24, clustered_decal_entries[1]),
            (25, clustered_decal_entries[2]),
        ));
    }

    // Tonemapping
    let tonemapping_lut_entries = get_lut_bind_group_layout_entries();
    entries = entries.extend_with_indices((
        (26, tonemapping_lut_entries[0]),
        (27, tonemapping_lut_entries[1]),
    ));

    // Prepass
    if cfg!(any(not(feature = "webgl"), not(target_arch = "wasm32")))
        || (cfg!(all(feature = "webgl", target_arch = "wasm32"))
            && !layout_key.contains(MeshPipelineViewLayoutKey::MULTISAMPLED))
    {
        for (entry, binding) in prepass::get_bind_group_layout_entries(layout_key)
            .iter()
            .zip([28, 29, 30, 31])
        {
            if let Some(entry) = entry {
                entries = entries.extend_with_indices(((binding as u32, *entry),));
            }
        }
    }

    // View Transmission Texture
    entries = entries.extend_with_indices((
        (
            32,
            binding_types::texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (33, binding_types::sampler(SamplerBindingType::Filtering)),
    ));

    // OIT
    if layout_key.contains(MeshPipelineViewLayoutKey::OIT_ENABLED) {
        // Check if we can use OIT. This is a hack to avoid errors on webgl --
        // the OIT plugin will warn the user that OIT is not supported on their
        // platform, so we don't need to do it here.
        if is_oit_supported(render_adapter, render_device, false) {
            entries = entries.extend_with_indices((
                // oit_layers
                (34, binding_types::storage_buffer_sized(false, None)),
                // oit_layer_ids,
                (35, binding_types::storage_buffer_sized(false, None)),
                // oit_layer_count
                (
                    36,
                    binding_types::uniform_buffer::<OrderIndependentTransparencySettings>(true),
                ),
            ));
        }
    }

    entries.to_vec()
}

pub const fn tonemapping_pipeline_key(tonemapping: Tonemapping) -> MeshPipelineKey {
    match tonemapping {
        Tonemapping::None => MeshPipelineKey::TONEMAP_METHOD_NONE,
        Tonemapping::Reinhard => MeshPipelineKey::TONEMAP_METHOD_REINHARD,
        Tonemapping::ReinhardLuminance => MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE,
        Tonemapping::AcesFitted => MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
        Tonemapping::AgX => MeshPipelineKey::TONEMAP_METHOD_AGX,
        Tonemapping::SomewhatBoringDisplayTransform => {
            MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
        }
        Tonemapping::TonyMcMapface => MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
        Tonemapping::BlenderFilmic => MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
    }
}

pub const fn screen_space_specular_transmission_pipeline_key(
    screen_space_transmissive_blur_quality: ScreenSpaceTransmissionQuality,
) -> MeshPipelineKey {
    match screen_space_transmissive_blur_quality {
        ScreenSpaceTransmissionQuality::Low => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_LOW
        }
        ScreenSpaceTransmissionQuality::Medium => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_MEDIUM
        }
        ScreenSpaceTransmissionQuality::High => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_HIGH
        }
        ScreenSpaceTransmissionQuality::Ultra => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_ULTRA
        }
    }
}
