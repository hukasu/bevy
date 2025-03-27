mod systems;

use core::{hash::Hash, marker::PhantomData};

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, AssetApp, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{prelude::resource_exists, schedule::IntoScheduleConfigs};
use bevy_render::{
    render_asset::prepare_assets,
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::Shader,
    renderer::RenderDevice,
    settings::WgpuFeatures,
    view::prepare_view_targets,
    ExtractSchedule, Render, RenderApp, RenderSet,
};

use systems::{
    configure_meshlet_views, extract_meshlet_mesh_entities, perform_pending_meshlet_mesh_writes,
    prepare_material_meshlet_meshes_main_opaque_pass, prepare_material_meshlet_meshes_prepass,
    prepare_meshlet_per_frame_resources, prepare_meshlet_view_bind_groups,
    queue_material_meshlet_meshes,
};
use tracing::error;

use crate::{
    material::{Material, PreparedMaterial},
    mesh_pipeline::graph::NodeRender3d,
    meshlet::{MeshletMesh, MeshletMeshLoader},
};

use super::{
    graph::NodeMeshlet,
    render::{
        InstanceManager, MeshletDeferredGBufferPrepassNode, MeshletMainOpaquePass3dNode,
        MeshletMeshManager, MeshletPipelines, MeshletPrepassNode,
        MeshletVisibilityBufferRasterPassNode, ResourceManager,
    },
};

const MESHLET_BINDINGS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("d90ac78c-500f-48aa-b488-cc98eb3f6314");
const MESHLET_MESH_MATERIAL_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("db8d9001-6ca7-4d00-968a-d5f5b96b89c3");
const MESHLET_VISIBILITY_BUFFER_RESOLVE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("69187376-3dea-4d0f-b3f5-185bde63d6a2");
pub const MESHLET_CLEAR_VISIBILITY_BUFFER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("a4bf48e4-5605-4d1c-987e-29c7b1ec95dc");
pub const MESHLET_FILL_CLUSTER_BUFFERS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("80ccea4a-8234-4ee0-af74-77b3cad503cf");
pub const MESHLET_CULLING_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("d71c5879-97fa-49d1-943e-ed9162fe8adb");
pub const MESHLET_VISIBILITY_BUFFER_SOFTWARE_RASTER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("68cc6826-8321-43d1-93d5-4f61f0456c13");
pub const MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("4b4e3020-748f-4baf-b011-87d9d2a12796");
pub const MESHLET_RESOLVE_RENDER_TARGETS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("c218ce17-cf59-4268-8898-13ecf384f133");
pub const MESHLET_REMAP_1D_TO_2D_DISPATCH_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("f5b7edfc-2eac-4407-8f5c-1265d4d795c2");

/// Provides a plugin for rendering large amounts of high-poly 3d meshes using an efficient GPU-driven method. See also [`MeshletMesh`].
///
/// Rendering dense scenes made of high-poly meshes with thousands or millions of triangles is extremely expensive in Bevy's standard renderer.
/// Once meshes are pre-processed into a [`MeshletMesh`], this plugin can render these kinds of scenes very efficiently.
///
/// In comparison to Bevy's standard renderer:
/// * Much more efficient culling. Meshlets can be culled individually, instead of all or nothing culling for entire meshes at a time.
///   Additionally, occlusion culling can eliminate meshlets that would cause overdraw.
/// * Much more efficient batching. All geometry can be rasterized in a single draw.
/// * Scales better with large amounts of dense geometry and overdraw. Bevy's standard renderer will bottleneck sooner.
/// * Near-seamless level of detail (LOD).
/// * Much greater base overhead. Rendering will be slower and use more memory than Bevy's standard renderer
///   with small amounts of geometry and overdraw.
/// * Requires preprocessing meshes. See [`MeshletMesh`] for details.
/// * Limitations on the kinds of materials you can use. See [`MeshletMesh`] for details.
///
/// This plugin requires a fairly recent GPU that supports [`WgpuFeatures::TEXTURE_INT64_ATOMIC`].
///
/// This plugin currently works only on the Vulkan and Metal backends.
///
/// This plugin is not compatible with [`Msaa`]. Any camera rendering a [`MeshletMesh`] must have
/// [`Msaa`] set to [`Msaa::Off`].
///
/// Mixing forward+prepass and deferred rendering for opaque materials is not currently supported when using this plugin.
/// You must use one or the other by setting [`crate::DefaultOpaqueRendererMethod`].
/// Do not override [`crate::Material::opaque_render_method`] for any material when using this plugin.
///
/// ![A render of the Stanford dragon as a `MeshletMesh`](https://raw.githubusercontent.com/bevyengine/bevy/main/crates/bevy_pbr/src/meshlet/meshlet_preview.png)
pub struct MeshletPlugin {
    /// The maximum amount of clusters that can be processed at once,
    /// used to control the size of a pre-allocated GPU buffer.
    ///
    /// If this number is too low, you'll see rendering artifacts like missing or blinking meshes.
    ///
    /// Each cluster slot costs 4 bytes of VRAM.
    ///
    /// Must not be greater than 2^25.
    pub cluster_buffer_slots: u32,
}

impl MeshletPlugin {
    /// [`WgpuFeatures`] required for this plugin to function.
    pub fn required_wgpu_features() -> WgpuFeatures {
        WgpuFeatures::TEXTURE_INT64_ATOMIC
            | WgpuFeatures::TEXTURE_ATOMIC
            | WgpuFeatures::SHADER_INT64
            | WgpuFeatures::SUBGROUP
            | WgpuFeatures::DEPTH_CLIP_CONTROL
            | WgpuFeatures::PUSH_CONSTANTS
    }
}

impl Plugin for MeshletPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_endian = "big")]
        compile_error!("MeshletPlugin is only supported on little-endian processors.");

        if self.cluster_buffer_slots > 2_u32.pow(25) {
            error!("MeshletPlugin::cluster_buffer_slots must not be greater than 2^25.");
            std::process::exit(1);
        }

        load_internal_asset!(
            app,
            MESHLET_CLEAR_VISIBILITY_BUFFER_SHADER_HANDLE,
            "shaders/clear_visibility_buffer.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_BINDINGS_SHADER_HANDLE,
            "shaders/meshlet_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_VISIBILITY_BUFFER_RESOLVE_SHADER_HANDLE,
            "shaders/visibility_buffer_resolve.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_FILL_CLUSTER_BUFFERS_SHADER_HANDLE,
            "shaders/fill_cluster_buffers.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_CULLING_SHADER_HANDLE,
            "shaders/cull_clusters.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_VISIBILITY_BUFFER_SOFTWARE_RASTER_SHADER_HANDLE,
            "shaders/visibility_buffer_software_raster.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
            "shaders/visibility_buffer_hardware_raster.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_MESH_MATERIAL_SHADER_HANDLE,
            "shaders/meshlet_mesh_material.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_RESOLVE_RENDER_TARGETS_SHADER_HANDLE,
            "shaders/resolve_render_targets.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESHLET_REMAP_1D_TO_2D_DISPATCH_SHADER_HANDLE,
            "shaders/remap_1d_to_2d_dispatch.wgsl",
            Shader::from_wgsl
        );

        app.init_asset::<MeshletMesh>()
            .register_asset_loader(MeshletMeshLoader);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>().clone();
        let features = render_device.features();
        if !features.contains(Self::required_wgpu_features()) {
            error!(
                "MeshletPlugin can't be used. GPU lacks support for required features: {:?}.",
                Self::required_wgpu_features().difference(features)
            );
            std::process::exit(1);
        }

        render_app
            .add_render_graph_node::<MeshletVisibilityBufferRasterPassNode>(
                Core3d,
                NodeMeshlet::VisibilityBufferRasterPass,
            )
            .add_render_graph_node::<ViewNodeRunner<MeshletPrepassNode>>(
                Core3d,
                NodeMeshlet::Prepass,
            )
            .add_render_graph_node::<ViewNodeRunner<MeshletDeferredGBufferPrepassNode>>(
                Core3d,
                NodeMeshlet::DeferredPrepass,
            )
            .add_render_graph_node::<ViewNodeRunner<MeshletMainOpaquePass3dNode>>(
                Core3d,
                NodeMeshlet::MainOpaquePass,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    NodeMeshlet::VisibilityBufferRasterPass,
                    NodeRender3d::EarlyShadowPass,
                    //
                    NodeMeshlet::Prepass,
                    //
                    NodeMeshlet::DeferredPrepass,
                    Node3d::EndPrepasses,
                    //
                    Node3d::StartMainPass,
                    NodeMeshlet::MainOpaquePass,
                    Node3d::MainOpaquePass,
                    Node3d::EndMainPass,
                ),
            )
            .init_resource::<MeshletMeshManager>()
            .insert_resource(InstanceManager::new())
            .insert_resource(ResourceManager::new(
                self.cluster_buffer_slots,
                &render_device,
            ))
            .init_resource::<MeshletPipelines>()
            .add_systems(ExtractSchedule, extract_meshlet_mesh_entities)
            .add_systems(
                Render,
                (
                    perform_pending_meshlet_mesh_writes.in_set(RenderSet::PrepareAssets),
                    configure_meshlet_views
                        .after(prepare_view_targets)
                        .in_set(RenderSet::ManageViews),
                    prepare_meshlet_per_frame_resources.in_set(RenderSet::PrepareResources),
                    prepare_meshlet_view_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            );
    }
}

pub struct RenderMeshletPlugin<M: Material> {
    prepass_enabled: bool,
    _data: PhantomData<M>,
}

impl<M: Material> Default for RenderMeshletPlugin<M> {
    fn default() -> Self {
        Self {
            prepass_enabled: false,
            _data: PhantomData,
        }
    }
}

impl<M: Material> Plugin for RenderMeshletPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                queue_material_meshlet_meshes::<M>
                    .in_set(RenderSet::QueueMeshes)
                    .run_if(resource_exists::<InstanceManager>),
            );

            render_app.add_systems(
                Render,
                prepare_material_meshlet_meshes_main_opaque_pass::<M>
                    .in_set(RenderSet::QueueMeshes)
                    .after(prepare_assets::<PreparedMaterial<M>>)
                    .before(queue_material_meshlet_meshes::<M>)
                    .run_if(resource_exists::<InstanceManager>),
            );

            if self.prepass_enabled {
                render_app.add_systems(
                    Render,
                    prepare_material_meshlet_meshes_prepass::<M>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<M>>)
                        .before(queue_material_meshlet_meshes::<M>)
                        .run_if(resource_exists::<InstanceManager>),
                );
            }
        }
    }
}
