mod systems;

use core::marker::PhantomData;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_ecs::{
    entity::Entity,
    query::{QueryData, QueryFilter},
    schedule::{IntoScheduleConfigs, SystemSet},
};
use bevy_math::{Mat4, Vec3A};
use bevy_render::{
    camera::CameraUpdateSystem,
    primitives::{Aabb, Sphere},
    render_resource::BufferBindingType,
    renderer::RenderDevice,
    view::{RenderLayers, VisibilitySystems},
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::{components::GlobalTransform, TransformSystem};

use super::{
    clusterable_objects::{GlobalClusterableObjectMeta, GlobalVisibleClusterableObjects},
    ClusterConfig, CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT,
};

use systems::{
    add_clusters, cluster_assignment, extract_clusters, post_cluster_assignment,
    pre_cluster_assignment, prepare_clusters, sort_cluster_assigned_objects,
};

pub struct ClusterableObjectPlugin<const O: u8, C: ClusterAssignable>(PhantomData<C>);

impl<const O: u8, C: ClusterAssignable> Default for ClusterableObjectPlugin<O, C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const O: u8, C: ClusterAssignable> Plugin for ClusterableObjectPlugin<O, C> {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ClusterPlugin>() {
            app.add_plugins(ClusterPlugin);
        }

        app.add_systems(
            PostUpdate,
            cluster_assignment::<O, C>
                .in_set(ClusterSystems::AssignLightsToClusters)
                .after(pre_cluster_assignment)
                .before(sort_cluster_assigned_objects),
        );
    }
}

struct ClusterPlugin;

impl Plugin for ClusterPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ClusterConfig>()
            .init_resource::<GlobalVisibleClusterableObjects>();

        app.configure_sets(
            PostUpdate,
            (
                ClusterSystems::AddClusters,
                ClusterSystems::AssignLightsToClusters,
            )
                .chain(),
        )
        .configure_sets(
            PostUpdate,
            (
                ClusterSystems::AddClusters.after(CameraUpdateSystem),
                ClusterSystems::AssignLightsToClusters
                    .after(TransformSystem::TransformPropagate)
                    .after(VisibilitySystems::CheckVisibility)
                    .after(CameraUpdateSystem),
            ),
        );

        app.add_systems(
            PostUpdate,
            (
                add_clusters.in_set(ClusterSystems::AddClusters),
                (pre_cluster_assignment, post_cluster_assignment)
                    .chain()
                    .in_set(ClusterSystems::AssignLightsToClusters),
            ),
        );

        if let Some(render_device) = app.world().get_resource::<RenderDevice>() {
            let clustered_forward_buffer_binding_type = render_device
                .get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT);
            let supports_storage_buffers = matches!(
                clustered_forward_buffer_binding_type,
                BufferBindingType::Storage { .. }
            );
            if !supports_storage_buffers {
                app.add_systems(
                    PostUpdate,
                    sort_cluster_assigned_objects
                        .in_set(ClusterSystems::AssignLightsToClusters)
                        .after(pre_cluster_assignment)
                        .before(post_cluster_assignment),
                );
            }
        }

        if let Some(render_app) = app.get_sub_app(RenderApp) {
            render_app
                .add_systems(ExtractSchedule, extract_clusters)
                .add_systems(Render, prepare_clusters.in_set(RenderSet::PrepareResources));
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<GlobalClusterableObjectMeta>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum ClusterSystems {
    AddClusters,
    AssignLightsToClusters,
}

pub trait ClusterAssignable: Sync + Send + 'static {
    type Query<'a>: QueryData;
    type Filter: QueryFilter;

    fn entity(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> Entity;
    fn transform(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> GlobalTransform;
    fn range(query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>)
        -> f32;
    fn shadows_enabled(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> bool;
    fn volumetric(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> bool;
    fn render_layers(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> Option<RenderLayers>;
    fn cull_method(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> Option<Box<dyn ClusterAssignableCullMethod>>;

    fn visible(
        query_data: &<<Self::Query<'_> as QueryData>::ReadOnly as QueryData>::Item<'_>,
    ) -> bool;
}

pub trait ClusterAssignableCullMethod: Send + Sync + 'static {
    fn cull(
        &self,
        transform: &GlobalTransform,
        center: &Vec3A,
        range: f32,
        view_from_world: &Mat4,
        view_from_world_scale_max: f32,
        aabb: &Aabb,
        cluster_aabb_sphere: &mut Option<Sphere>,
    ) -> bool;
}
