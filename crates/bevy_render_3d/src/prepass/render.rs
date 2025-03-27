use core::{hash::Hash, marker::PhantomData};

use bevy_asset::{AssetServer, Handle};
use bevy_core_pipeline::{
    core_3d::CORE_3D_DEPTH_FORMAT,
    prepass::{prepass_target_descriptors, PreviousViewData},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Tick,
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_math::Vec4;
use bevy_mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy_platform_support::collections::HashMap;
use bevy_render::{
    globals::GlobalsUniform,
    render_resource::{
        binding_types, BindGroup, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
        CompareFunction, DepthBiasState, DepthStencilState, FragmentState, FrontFace,
        MultisampleState, PolygonMode, PrimitiveState, RenderPipelineDescriptor, Shader, ShaderRef,
        ShaderStages, ShaderType, SpecializedMeshPipeline, SpecializedMeshPipelineError,
        StencilFaceState, StencilState, VertexState,
    },
    renderer::{RenderAdapter, RenderDevice},
    settings::WgpuFeatures,
    sync_world::MainEntityHashMap,
    view::{RetainedViewEntity, ViewUniform, VISIBILITY_RANGES_STORAGE_BUFFER_COUNT},
};

use tracing::warn;

use crate::{
    binding_arrays_are_usable, buffer_layout,
    material::{Material, MaterialPipeline, MaterialPipelineKey},
    mesh_pipeline::render::{
        pipeline::{MeshPipeline, MeshPipelineKey},
        MeshLayouts,
    },
    setup_morph_and_skinning_defs,
    skin::uniforms::skins_use_uniform_buffers,
};

use super::plugin::PREPASS_SHADER_HANDLE;

#[derive(Resource)]
pub struct PrepassPipeline<M: Material> {
    pub view_layout_motion_vectors: BindGroupLayout,
    pub view_layout_no_motion_vectors: BindGroupLayout,
    pub mesh_layouts: MeshLayouts,
    pub material_layout: BindGroupLayout,
    pub prepass_material_vertex_shader: Option<Handle<Shader>>,
    pub prepass_material_fragment_shader: Option<Handle<Shader>>,
    pub deferred_material_vertex_shader: Option<Handle<Shader>>,
    pub deferred_material_fragment_shader: Option<Handle<Shader>>,
    pub material_pipeline: MaterialPipeline<M>,

    /// Whether skins will use uniform buffers on account of storage buffers
    /// being unavailable on this platform.
    pub skins_use_uniform_buffers: bool,

    pub depth_clip_control_supported: bool,

    /// Whether binding arrays (a.k.a. bindless textures) are usable on the
    /// current render device.
    pub binding_arrays_are_usable: bool,

    _marker: PhantomData<M>,
}

impl<M: Material> FromWorld for PrepassPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_adapter = world.resource::<RenderAdapter>();
        let asset_server = world.resource::<AssetServer>();

        let visibility_ranges_buffer_binding_type = render_device
            .get_supported_read_only_binding_type(VISIBILITY_RANGES_STORAGE_BUFFER_COUNT);

        let view_layout_motion_vectors = render_device.create_bind_group_layout(
            "prepass_view_layout_motion_vectors",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    // View
                    (0, binding_types::uniform_buffer::<ViewUniform>(true)),
                    // Globals
                    (1, binding_types::uniform_buffer::<GlobalsUniform>(false)),
                    // PreviousViewUniforms
                    (2, binding_types::uniform_buffer::<PreviousViewData>(true)),
                    // VisibilityRanges
                    (
                        14,
                        buffer_layout(
                            visibility_ranges_buffer_binding_type,
                            false,
                            Some(Vec4::min_size()),
                        )
                        .visibility(ShaderStages::VERTEX),
                    ),
                ),
            ),
        );

        let view_layout_no_motion_vectors = render_device.create_bind_group_layout(
            "prepass_view_layout_no_motion_vectors",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    // View
                    (0, binding_types::uniform_buffer::<ViewUniform>(true)),
                    // Globals
                    (1, binding_types::uniform_buffer::<GlobalsUniform>(false)),
                    // VisibilityRanges
                    (
                        14,
                        buffer_layout(
                            visibility_ranges_buffer_binding_type,
                            false,
                            Some(Vec4::min_size()),
                        )
                        .visibility(ShaderStages::VERTEX),
                    ),
                ),
            ),
        );

        let mesh_pipeline = world.resource::<MeshPipeline>();

        let depth_clip_control_supported = render_device
            .features()
            .contains(WgpuFeatures::DEPTH_CLIP_CONTROL);

        PrepassPipeline {
            view_layout_motion_vectors,
            view_layout_no_motion_vectors,
            mesh_layouts: mesh_pipeline.mesh_layouts.clone(),
            prepass_material_vertex_shader: match M::prepass_vertex_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            prepass_material_fragment_shader: match M::prepass_fragment_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            deferred_material_vertex_shader: match M::deferred_vertex_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            deferred_material_fragment_shader: match M::deferred_fragment_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            material_layout: M::bind_group_layout(render_device),
            material_pipeline: world.resource::<MaterialPipeline<M>>().clone(),
            skins_use_uniform_buffers: skins_use_uniform_buffers(render_device),
            depth_clip_control_supported,
            binding_arrays_are_usable: binding_arrays_are_usable(render_device, render_adapter),
            _marker: PhantomData,
        }
    }
}

impl<M: Material> SpecializedMeshPipeline for PrepassPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = MaterialPipelineKey<M>;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut bind_group_layouts = vec![if key
            .mesh_key
            .contains(MeshPipelineKey::MOTION_VECTOR_PREPASS)
        {
            self.view_layout_motion_vectors.clone()
        } else {
            self.view_layout_no_motion_vectors.clone()
        }];
        let mut shader_defs = Vec::new();
        let mut vertex_attributes = Vec::new();

        // Let the shader code know that it's running in a prepass pipeline.
        // (PBR code will use this to detect that it's running in deferred mode,
        // since that's the only time it gets called from a prepass pipeline.)
        shader_defs.push("PREPASS_PIPELINE".into());

        // NOTE: Eventually, it would be nice to only add this when the shaders are overloaded by the Material.
        // The main limitation right now is that bind group order is hardcoded in shaders.
        bind_group_layouts.push(self.material_layout.clone());

        #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
        shader_defs.push("WEBGL2".into());

        shader_defs.push("VERTEX_OUTPUT_INSTANCE_INDEX".into());

        if key.mesh_key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }

        if key.mesh_key.contains(MeshPipelineKey::MAY_DISCARD) {
            shader_defs.push("MAY_DISCARD".into());
        }

        let blend_key = key
            .mesh_key
            .intersection(MeshPipelineKey::BLEND_RESERVED_BITS);
        if blend_key == MeshPipelineKey::BLEND_PREMULTIPLIED_ALPHA {
            shader_defs.push("BLEND_PREMULTIPLIED_ALPHA".into());
        }
        if blend_key == MeshPipelineKey::BLEND_ALPHA {
            shader_defs.push("BLEND_ALPHA".into());
        }

        if layout.0.contains(Mesh::ATTRIBUTE_POSITION) {
            shader_defs.push("VERTEX_POSITIONS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_POSITION.at_shader_location(0));
        }

        // For directional light shadow map views, use unclipped depth via either the native GPU feature,
        // or emulated by setting depth in the fragment shader for GPUs that don't support it natively.
        let emulate_unclipped_depth = key
            .mesh_key
            .contains(MeshPipelineKey::UNCLIPPED_DEPTH_ORTHO)
            && !self.depth_clip_control_supported;
        if emulate_unclipped_depth {
            shader_defs.push("UNCLIPPED_DEPTH_ORTHO_EMULATION".into());
            // PERF: This line forces the "prepass fragment shader" to always run in
            // common scenarios like "directional light calculation". Doing so resolves
            // a pretty nasty depth clamping bug, but it also feels a bit excessive.
            // We should try to find a way to resolve this without forcing the fragment
            // shader to run.
            // https://github.com/bevyengine/bevy/pull/8877
            shader_defs.push("PREPASS_FRAGMENT".into());
        }
        let unclipped_depth = key
            .mesh_key
            .contains(MeshPipelineKey::UNCLIPPED_DEPTH_ORTHO)
            && self.depth_clip_control_supported;

        if layout.0.contains(Mesh::ATTRIBUTE_UV_0) {
            shader_defs.push("VERTEX_UVS".into());
            shader_defs.push("VERTEX_UVS_A".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(1));
        }

        if layout.0.contains(Mesh::ATTRIBUTE_UV_1) {
            shader_defs.push("VERTEX_UVS".into());
            shader_defs.push("VERTEX_UVS_B".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_UV_1.at_shader_location(2));
        }

        if key.mesh_key.contains(MeshPipelineKey::NORMAL_PREPASS) {
            shader_defs.push("NORMAL_PREPASS".into());
        }

        if key
            .mesh_key
            .intersects(MeshPipelineKey::NORMAL_PREPASS | MeshPipelineKey::DEFERRED_PREPASS)
        {
            shader_defs.push("NORMAL_PREPASS_OR_DEFERRED_PREPASS".into());
            if layout.0.contains(Mesh::ATTRIBUTE_NORMAL) {
                shader_defs.push("VERTEX_NORMALS".into());
                vertex_attributes.push(Mesh::ATTRIBUTE_NORMAL.at_shader_location(3));
            } else if key.mesh_key.contains(MeshPipelineKey::NORMAL_PREPASS) {
                warn!(
                    "The default normal prepass expects the mesh to have vertex normal attributes."
                );
            }
            if layout.0.contains(Mesh::ATTRIBUTE_TANGENT) {
                shader_defs.push("VERTEX_TANGENTS".into());
                vertex_attributes.push(Mesh::ATTRIBUTE_TANGENT.at_shader_location(4));
            }
        }

        if key
            .mesh_key
            .intersects(MeshPipelineKey::MOTION_VECTOR_PREPASS | MeshPipelineKey::DEFERRED_PREPASS)
        {
            shader_defs.push("MOTION_VECTOR_PREPASS_OR_DEFERRED_PREPASS".into());
        }

        if key.mesh_key.contains(MeshPipelineKey::DEFERRED_PREPASS) {
            shader_defs.push("DEFERRED_PREPASS".into());
        }

        if key.mesh_key.contains(MeshPipelineKey::LIGHTMAPPED) {
            shader_defs.push("LIGHTMAP".into());
        }
        if key
            .mesh_key
            .contains(MeshPipelineKey::LIGHTMAP_BICUBIC_SAMPLING)
        {
            shader_defs.push("LIGHTMAP_BICUBIC_SAMPLING".into());
        }

        if layout.0.contains(Mesh::ATTRIBUTE_COLOR) {
            shader_defs.push("VERTEX_COLORS".into());
            vertex_attributes.push(Mesh::ATTRIBUTE_COLOR.at_shader_location(7));
        }

        if key
            .mesh_key
            .contains(MeshPipelineKey::MOTION_VECTOR_PREPASS)
        {
            shader_defs.push("MOTION_VECTOR_PREPASS".into());
        }

        if key.mesh_key.contains(MeshPipelineKey::HAS_PREVIOUS_SKIN) {
            shader_defs.push("HAS_PREVIOUS_SKIN".into());
        }

        if key.mesh_key.contains(MeshPipelineKey::HAS_PREVIOUS_MORPH) {
            shader_defs.push("HAS_PREVIOUS_MORPH".into());
        }

        // If bindless mode is on, add a `BINDLESS` define.
        if self.material_pipeline.bindless {
            shader_defs.push("BINDLESS".into());
        }

        if self.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHTMAPS_IN_ARRAY".into());
        }

        if key
            .mesh_key
            .contains(MeshPipelineKey::VISIBILITY_RANGE_DITHER)
        {
            shader_defs.push("VISIBILITY_RANGE_DITHER".into());
        }

        if key.mesh_key.intersects(
            MeshPipelineKey::NORMAL_PREPASS
                | MeshPipelineKey::MOTION_VECTOR_PREPASS
                | MeshPipelineKey::DEFERRED_PREPASS,
        ) {
            shader_defs.push("PREPASS_FRAGMENT".into());
        }

        let bind_group = setup_morph_and_skinning_defs(
            &self.mesh_layouts,
            layout,
            5,
            &key.mesh_key,
            &mut shader_defs,
            &mut vertex_attributes,
            self.skins_use_uniform_buffers,
        );
        bind_group_layouts.insert(1, bind_group);

        let vertex_buffer_layout = layout.0.get_layout(&vertex_attributes)?;

        // Setup prepass fragment targets - normals in slot 0 (or None if not needed), motion vectors in slot 1
        let mut targets = prepass_target_descriptors(
            key.mesh_key.contains(MeshPipelineKey::NORMAL_PREPASS),
            key.mesh_key
                .contains(MeshPipelineKey::MOTION_VECTOR_PREPASS),
            key.mesh_key.contains(MeshPipelineKey::DEFERRED_PREPASS),
        );

        if targets.iter().all(Option::is_none) {
            // if no targets are required then clear the list, so that no fragment shader is required
            // (though one may still be used for discarding depth buffer writes)
            targets.clear();
        }

        // The fragment shader is only used when the normal prepass or motion vectors prepass
        // is enabled, the material uses alpha cutoff values and doesn't rely on the standard
        // prepass shader, or we are emulating unclipped depth in the fragment shader.
        let fragment_required = !targets.is_empty()
            || emulate_unclipped_depth
            || (key.mesh_key.contains(MeshPipelineKey::MAY_DISCARD)
                && self.prepass_material_fragment_shader.is_some());

        let fragment = fragment_required.then(|| {
            // Use the fragment shader from the material
            let frag_shader_handle = if key.mesh_key.contains(MeshPipelineKey::DEFERRED_PREPASS) {
                match self.deferred_material_fragment_shader.clone() {
                    Some(frag_shader_handle) => frag_shader_handle,
                    _ => PREPASS_SHADER_HANDLE,
                }
            } else {
                match self.prepass_material_fragment_shader.clone() {
                    Some(frag_shader_handle) => frag_shader_handle,
                    _ => PREPASS_SHADER_HANDLE,
                }
            };

            FragmentState {
                shader: frag_shader_handle,
                entry_point: "fragment".into(),
                shader_defs: shader_defs.clone(),
                targets,
            }
        });

        // Use the vertex shader from the material if present
        let vert_shader_handle = if key.mesh_key.contains(MeshPipelineKey::DEFERRED_PREPASS) {
            if let Some(handle) = &self.deferred_material_vertex_shader {
                handle.clone()
            } else {
                PREPASS_SHADER_HANDLE
            }
        } else if let Some(handle) = &self.prepass_material_vertex_shader {
            handle.clone()
        } else {
            PREPASS_SHADER_HANDLE
        };

        let mut descriptor = RenderPipelineDescriptor {
            vertex: VertexState {
                shader: vert_shader_handle,
                entry_point: "vertex".into(),
                shader_defs,
                buffers: vec![vertex_buffer_layout],
            },
            fragment,
            layout: bind_group_layouts,
            primitive: PrimitiveState {
                topology: key.mesh_key.primitive_topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
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
                count: key.mesh_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: vec![],
            label: Some("prepass_pipeline".into()),
            zero_initialize_workgroup_memory: false,
        };

        // This is a bit risky because it's possible to change something that would
        // break the prepass but be fine in the main pass.
        // Since this api is pretty low-level it doesn't matter that much, but it is a potential issue.
        M::specialize(&self.material_pipeline, &mut descriptor, layout, key)?;

        Ok(descriptor)
    }
}

#[derive(Default, Resource)]
pub struct PrepassViewBindGroup {
    pub motion_vectors: Option<BindGroup>,
    pub no_motion_vectors: Option<BindGroup>,
}

/// Stores the [`SpecializedPrepassMaterialViewPipelineCache`] for each view.
#[derive(Resource, Deref, DerefMut)]
pub struct SpecializedPrepassMaterialPipelineCache<M> {
    // view_entity -> view pipeline cache
    #[deref]
    map: HashMap<RetainedViewEntity, SpecializedPrepassMaterialViewPipelineCache<M>>,
    marker: PhantomData<M>,
}

impl<M> Default for SpecializedPrepassMaterialPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            marker: PhantomData,
        }
    }
}

/// Stores the cached render pipeline ID for each entity in a single view, as
/// well as the last time it was changed.
#[derive(Deref, DerefMut)]
pub struct SpecializedPrepassMaterialViewPipelineCache<M> {
    // material entity -> (tick, pipeline_id)
    #[deref]
    map: MainEntityHashMap<(Tick, CachedRenderPipelineId)>,
    marker: PhantomData<M>,
}

impl<M> Default for SpecializedPrepassMaterialViewPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            marker: PhantomData,
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default, Clone)]
pub struct ViewKeyPrepassCache(HashMap<RetainedViewEntity, MeshPipelineKey>);

#[derive(Resource, Deref, DerefMut, Default, Clone)]
pub struct ViewPrepassSpecializationTicks(HashMap<RetainedViewEntity, Tick>);
