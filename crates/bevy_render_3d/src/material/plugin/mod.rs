mod systems;

use core::{hash::Hash, marker::PhantomData};

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{AssetApp, AssetEvents};
use bevy_core_pipeline::core_3d::{AlphaMask3d, Opaque3d, Transmissive3d, Transparent3d};
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_render::{
    mesh::{mark_3d_meshes_as_changed_if_their_assets_changed, RenderMesh},
    render_asset::{prepare_assets, RenderAssetPlugin},
    render_phase::AddRenderCommand,
    render_resource::SpecializedMeshPipelines,
    ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSet,
};

use crate::{
    mesh_pipeline::{
        specialization::{EntitiesNeedingSpecialization, EntitySpecializationTicks},
        ExtractMeshesSet, GpuMeshBuildingSet,
    },
    prepass::plugin::{PrepassPipelinePlugin, PrepassPlugin},
    shadow::plugin::ShadowPlugin,
};

use super::{
    commands::DrawMaterial,
    material::{
        MaterialPipeline, PreparedMaterial, RenderMaterialInstances,
        SpecializedMaterialPipelineCache,
    },
    Material, MaterialBindGroupAllocator, MeshMaterial3d,
};

use systems::{
    check_entities_needing_specialization, extract_entities_needs_specialization,
    extract_mesh_materials, mark_meshes_as_changed_if_their_materials_changed,
    prepare_material_bind_groups, queue_material_meshes, specialize_material_meshes,
    write_material_bind_group_buffers,
};

/// Adds the necessary ECS resources and render logic to enable rendering entities using the given [`Material`]
/// asset type.
pub struct MaterialPlugin<M: Material> {
    /// Controls if the prepass is enabled for the Material.
    /// For more information about what a prepass is, see the [`bevy_core_pipeline::prepass`] docs.
    ///
    /// When it is enabled, it will automatically add the [`PrepassPlugin`]
    /// required to make the prepass work on this Material.
    pub prepass_enabled: bool,
    /// Controls if shadows are enabled for the Material.
    pub shadows_enabled: bool,
    /// Debugging flags that can optionally be set when constructing the renderer.
    pub debug_flags: RenderDebugFlags,
    pub _marker: PhantomData<M>,
}

impl<M: Material> Default for MaterialPlugin<M> {
    fn default() -> Self {
        Self {
            prepass_enabled: true,
            shadows_enabled: true,
            debug_flags: RenderDebugFlags::default(),
            _marker: Default::default(),
        }
    }
}

impl<M: Material> Plugin for MaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.configure_sets(
            PostUpdate,
            MaterialSystems::SpecializationCheck.after(AssetEvents),
        );

        app.init_asset::<M>()
            .register_type::<MeshMaterial3d<M>>()
            .init_resource::<EntitiesNeedingSpecialization<M>>()
            .add_plugins(RenderAssetPlugin::<PreparedMaterial<M>>::default())
            .add_systems(
                PostUpdate,
                (
                    mark_meshes_as_changed_if_their_materials_changed::<M>.ambiguous_with_all(),
                    check_entities_needing_specialization::<M>
                        .in_set(MaterialSystems::SpecializationCheck),
                )
                    .after(mark_3d_meshes_as_changed_if_their_assets_changed),
            );

        // #[cfg(feature = "meshlet")]
        // app.add_plugins(meshlet::RenderMeshletPlugin::<M>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.configure_sets(
                Render,
                MaterialRenderSystems::QueueMeshes
                    .in_set(RenderSet::QueueMeshes)
                    .after(prepare_assets::<PreparedMaterial<M>>),
            );

            render_app
                .init_resource::<EntitySpecializationTicks<M>>()
                .init_resource::<SpecializedMaterialPipelineCache<M>>()
                .init_resource::<RenderMaterialInstances<M>>()
                .add_render_command::<Transmissive3d, DrawMaterial<M>>()
                .add_render_command::<Transparent3d, DrawMaterial<M>>()
                .add_render_command::<Opaque3d, DrawMaterial<M>>()
                .add_render_command::<AlphaMask3d, DrawMaterial<M>>()
                .init_resource::<SpecializedMeshPipelines<MaterialPipeline<M>>>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_mesh_materials::<M>.before(ExtractMeshesSet),
                        extract_entities_needs_specialization::<M>,
                    ),
                )
                .add_systems(
                    Render,
                    (
                        specialize_material_meshes::<M>
                            .in_set(RenderSet::PrepareMeshes)
                            .after(prepare_assets::<PreparedMaterial<M>>)
                            .after(prepare_assets::<RenderMesh>)
                            .after(GpuMeshBuildingSet),
                        queue_material_meshes::<M>.in_set(MaterialRenderSystems::QueueMeshes),
                    ),
                )
                .add_systems(
                    Render,
                    (
                        prepare_material_bind_groups::<M>,
                        write_material_bind_group_buffers::<M>,
                    )
                        .chain()
                        .in_set(RenderSet::PrepareBindGroups)
                        .after(prepare_assets::<PreparedMaterial<M>>),
                );

            if self.shadows_enabled {
                render_app.add_plugins(ShadowPlugin::<M>::default());
            }
        }

        if self.shadows_enabled || self.prepass_enabled {
            // PrepassPipelinePlugin is required for shadow mapping and the optional PrepassPlugin
            app.add_plugins(PrepassPipelinePlugin::<M>::default());
        }

        if self.prepass_enabled {
            app.add_plugins(PrepassPlugin::<M>::new(self.debug_flags));
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MaterialPipeline<M>>()
                .init_resource::<MaterialBindGroupAllocator<M>>();
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum MaterialSystems {
    /// After [`AssetEvents`]
    SpecializationCheck,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum MaterialRenderSystems {
    /// In set [`RenderSet::QueueMeshes`] and after [`prepare_assets`]
    QueueMeshes,
}
