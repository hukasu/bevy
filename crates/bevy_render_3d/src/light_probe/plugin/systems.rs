use bevy_core_pipeline::core_3d::Camera3d;
use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_render::{
    primitives::Frustum,
    render_asset::RenderAssets,
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    texture::GpuImage,
    view::ExtractedView,
    Extract,
};
use bevy_transform::components::{GlobalTransform, Transform};

use tracing::error;

use crate::light_probe::{
    environment_map::{
        EnvironmentMapUniform, EnvironmentMapUniformBuffer, ViewEnvironmentMapUniformOffset,
    },
    light_probes::{
        LightProbeInfo, LightProbesBuffer, LightProbesUniform, RenderLightProbe,
        RenderViewLightProbes, ViewLightProbesUniformOffset,
    },
    EnvironmentMapLight, IrradianceVolume, LightProbe, LightProbeComponent, MAX_VIEW_LIGHT_PROBES,
};

/// Extracts [`EnvironmentMapLight`] from views and creates [`EnvironmentMapUniform`] for them.
///
/// Compared to the `ExtractComponentPlugin`, this implementation will create a default instance
/// if one does not already exist.
pub fn gather_environment_map_uniform(
    view_query: Extract<Query<(RenderEntity, Option<&EnvironmentMapLight>), With<Camera3d>>>,
    mut commands: Commands,
) {
    for (view_entity, environment_map_light) in view_query.iter() {
        let environment_map_uniform = if let Some(environment_map_light) = environment_map_light {
            EnvironmentMapUniform {
                transform: Transform::from_rotation(environment_map_light.rotation)
                    .compute_matrix()
                    .inverse(),
            }
        } else {
            EnvironmentMapUniform::default()
        };
        commands
            .get_entity(view_entity)
            .expect("Environment map light entity wasn't synced.")
            .insert(environment_map_uniform);
    }
}

/// Gathers up all light probes of a single type in the scene and assigns them
/// to views, performing frustum culling and distance sorting in the process.
pub fn gather_light_probes<C>(
    image_assets: Res<RenderAssets<GpuImage>>,
    light_probe_query: Extract<Query<(&GlobalTransform, &C), With<LightProbe>>>,
    view_query: Extract<
        Query<(RenderEntity, &GlobalTransform, &Frustum, Option<&C>), With<Camera3d>>,
    >,
    mut reflection_probes: Local<Vec<LightProbeInfo<C>>>,
    mut view_reflection_probes: Local<Vec<LightProbeInfo<C>>>,
    mut commands: Commands,
) where
    C: LightProbeComponent,
{
    // Create [`LightProbeInfo`] for every light probe in the scene.
    reflection_probes.clear();
    reflection_probes.extend(
        light_probe_query
            .iter()
            .filter_map(|query_row| LightProbeInfo::new(query_row, &image_assets)),
    );

    // Build up the light probes uniform and the key table.
    for (view_entity, view_transform, view_frustum, view_component) in view_query.iter() {
        // Cull light probes outside the view frustum.
        view_reflection_probes.clear();
        view_reflection_probes.extend(
            reflection_probes
                .iter()
                .filter(|light_probe_info| light_probe_info.frustum_cull(view_frustum))
                .cloned(),
        );

        // Sort by distance to camera.
        view_reflection_probes.sort_by_cached_key(|light_probe_info| {
            light_probe_info.camera_distance_sort_key(view_transform)
        });

        // Create the light probes list.
        let mut render_view_light_probes =
            C::create_render_view_light_probes(view_component, &image_assets);

        // Gather up the light probes in the list.
        render_view_light_probes.maybe_gather_light_probes(&view_reflection_probes);

        // Record the per-view light probes.
        if render_view_light_probes.is_empty() {
            commands
                .get_entity(view_entity)
                .expect("View entity wasn't synced.")
                .remove::<RenderViewLightProbes<C>>();
        } else {
            commands
                .get_entity(view_entity)
                .expect("View entity wasn't synced.")
                .insert(render_view_light_probes);
        }
    }
}

/// Gathers up environment map settings for each applicable view and
/// writes them into a GPU buffer.
pub fn prepare_environment_uniform_buffer(
    mut commands: Commands,
    views: Query<(Entity, Option<&EnvironmentMapUniform>), With<ExtractedView>>,
    mut environment_uniform_buffer: ResMut<EnvironmentMapUniformBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(mut writer) =
        environment_uniform_buffer.get_writer(views.iter().len(), &render_device, &render_queue)
    else {
        return;
    };

    for (view, environment_uniform) in views.iter() {
        let uniform_offset = match environment_uniform {
            None => 0,
            Some(environment_uniform) => writer.write(environment_uniform),
        };
        commands
            .entity(view)
            .insert(ViewEnvironmentMapUniformOffset(uniform_offset));
    }
}

// A system that runs after [`gather_light_probes`] and populates the GPU
// uniforms with the results.
//
// Note that, unlike [`gather_light_probes`], this system is not generic over
// the type of light probe. It collects light probes of all types together into
// a single structure, ready to be passed to the shader.
pub fn upload_light_probes(
    mut commands: Commands,
    views: Query<Entity, With<ExtractedView>>,
    mut light_probes_buffer: ResMut<LightProbesBuffer>,
    mut view_light_probes_query: Query<(
        Option<&RenderViewLightProbes<EnvironmentMapLight>>,
        Option<&RenderViewLightProbes<IrradianceVolume>>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // If there are no views, bail.
    if views.is_empty() {
        return;
    }

    // Initialize the uniform buffer writer.
    let mut writer = light_probes_buffer
        .get_writer(views.iter().len(), &render_device, &render_queue)
        .unwrap();

    // Process each view.
    for view_entity in views.iter() {
        let Ok((render_view_environment_maps, render_view_irradiance_volumes)) =
            view_light_probes_query.get_mut(view_entity)
        else {
            error!("Failed to find `RenderViewLightProbes` for the view!");
            continue;
        };

        // Initialize the uniform with only the view environment map, if there
        // is one.
        let mut light_probes_uniform = LightProbesUniform {
            reflection_probes: [RenderLightProbe::default(); MAX_VIEW_LIGHT_PROBES],
            irradiance_volumes: [RenderLightProbe::default(); MAX_VIEW_LIGHT_PROBES],
            reflection_probe_count: render_view_environment_maps
                .map(RenderViewLightProbes::len)
                .unwrap_or_default()
                .min(MAX_VIEW_LIGHT_PROBES) as i32,
            irradiance_volume_count: render_view_irradiance_volumes
                .map(RenderViewLightProbes::len)
                .unwrap_or_default()
                .min(MAX_VIEW_LIGHT_PROBES) as i32,
            view_cubemap_index: render_view_environment_maps
                .map(|maps| maps.view_light_probe_info.cubemap_index)
                .unwrap_or(-1),
            smallest_specular_mip_level_for_view: render_view_environment_maps
                .map(|maps| maps.view_light_probe_info.smallest_specular_mip_level)
                .unwrap_or(0),
            intensity_for_view: render_view_environment_maps
                .map(|maps| maps.view_light_probe_info.intensity)
                .unwrap_or(1.0),
            view_environment_map_affects_lightmapped_mesh_diffuse: render_view_environment_maps
                .map(|maps| maps.view_light_probe_info.affects_lightmapped_mesh_diffuse as u32)
                .unwrap_or(1),
        };

        // Add any environment maps that [`gather_light_probes`] found to the
        // uniform.
        if let Some(render_view_environment_maps) = render_view_environment_maps {
            render_view_environment_maps.add_to_uniform(
                &mut light_probes_uniform.reflection_probes,
                &mut light_probes_uniform.reflection_probe_count,
            );
        }

        // Add any irradiance volumes that [`gather_light_probes`] found to the
        // uniform.
        if let Some(render_view_irradiance_volumes) = render_view_irradiance_volumes {
            render_view_irradiance_volumes.add_to_uniform(
                &mut light_probes_uniform.irradiance_volumes,
                &mut light_probes_uniform.irradiance_volume_count,
            );
        }

        // Queue the view's uniforms to be written to the GPU.
        let uniform_offset = writer.write(&light_probes_uniform);

        commands
            .entity(view_entity)
            .insert(ViewLightProbesUniformOffset(uniform_offset));
    }
}
