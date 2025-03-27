mod systems;

use core::marker::PhantomData;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::{
    entity::Entity,
    query::{QueryData, QueryFilter},
    schedule::{IntoScheduleConfigs, SystemSet},
};
use bevy_math::{ops, Mat4, Vec2, Vec3A};
use bevy_render::{
    camera::CameraUpdateSystem,
    primitives::{Aabb, Sphere},
    render_resource::Shader,
    view::{RenderLayers, VisibilitySystems},
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::{components::GlobalTransform, TransformSystem};

use super::{
    clusterable_objects::{GlobalClusterableObjectMeta, GlobalVisibleClusterableObjects},
    ClusterConfig,
};

use systems::{
    add_clusters, cluster_assignment, extract_clusters, post_cluster_assignment,
    pre_cluster_assignment, prepare_clusters, sort_cluster_assigned_objects,
    sort_cluster_assigned_objects_condition,
};

pub const CLUSTERED_FORWARD_HANDLE: Handle<Shader> =
    weak_handle!("f8e3b4c6-60b7-4b23-8b2e-a6b27bb4ddce");

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
        load_internal_asset!(
            app,
            CLUSTERED_FORWARD_HANDLE,
            "clustered_forward.wgsl",
            Shader::from_wgsl
        );

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

        app.add_systems(
            PostUpdate,
            sort_cluster_assigned_objects
                .in_set(ClusterSystems::AssignLightsToClusters)
                .after(pre_cluster_assignment)
                .before(post_cluster_assignment)
                .run_if(sort_cluster_assigned_objects_condition),
        );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
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

pub(crate) fn calculate_cluster_factors(
    near: f32,
    far: f32,
    z_slices: f32,
    is_orthographic: bool,
) -> Vec2 {
    if is_orthographic {
        Vec2::new(-near, z_slices / (-far - -near))
    } else {
        let z_slices_of_ln_zfar_over_znear = (z_slices - 1.0) / ops::ln(far / near);
        Vec2::new(
            z_slices_of_ln_zfar_over_znear,
            ops::ln(near) * z_slices_of_ln_zfar_over_znear,
        )
    }
}
