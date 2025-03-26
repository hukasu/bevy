use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::Vec3;
use bevy_render::{
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};

use crate::distance_fog::{
    fog::{FogMeta, GpuFog, ViewFogUniformOffset},
    DistanceFog, FogFalloff,
};

// Important: These must be kept in sync with `mesh_view_types.wgsl`
const GPU_FOG_MODE_OFF: u32 = 0;
const GPU_FOG_MODE_LINEAR: u32 = 1;
const GPU_FOG_MODE_EXPONENTIAL: u32 = 2;
const GPU_FOG_MODE_EXPONENTIAL_SQUARED: u32 = 3;
const GPU_FOG_MODE_ATMOSPHERIC: u32 = 4;

/// Prepares fog metadata and writes the fog-related uniform buffers to the GPU
pub fn prepare_fog(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut fog_meta: ResMut<FogMeta>,
    views: Query<(Entity, Option<&DistanceFog>), With<ExtractedView>>,
) {
    let views_iter = views.iter();
    let view_count = views_iter.len();
    let Some(mut writer) = fog_meta
        .gpu_fogs
        .get_writer(view_count, &render_device, &render_queue)
    else {
        return;
    };
    for (entity, fog) in views_iter {
        let gpu_fog = if let Some(fog) = fog {
            match &fog.falloff {
                FogFalloff::Linear { start, end } => GpuFog {
                    mode: GPU_FOG_MODE_LINEAR,
                    base_color: LinearRgba::from(fog.color).to_vec4(),
                    directional_light_color: LinearRgba::from(fog.directional_light_color)
                        .to_vec4(),
                    directional_light_exponent: fog.directional_light_exponent,
                    be: Vec3::new(*start, *end, 0.0),
                    ..Default::default()
                },
                FogFalloff::Exponential { density } => GpuFog {
                    mode: GPU_FOG_MODE_EXPONENTIAL,
                    base_color: LinearRgba::from(fog.color).to_vec4(),
                    directional_light_color: LinearRgba::from(fog.directional_light_color)
                        .to_vec4(),
                    directional_light_exponent: fog.directional_light_exponent,
                    be: Vec3::new(*density, 0.0, 0.0),
                    ..Default::default()
                },
                FogFalloff::ExponentialSquared { density } => GpuFog {
                    mode: GPU_FOG_MODE_EXPONENTIAL_SQUARED,
                    base_color: LinearRgba::from(fog.color).to_vec4(),
                    directional_light_color: LinearRgba::from(fog.directional_light_color)
                        .to_vec4(),
                    directional_light_exponent: fog.directional_light_exponent,
                    be: Vec3::new(*density, 0.0, 0.0),
                    ..Default::default()
                },
                FogFalloff::Atmospheric {
                    extinction,
                    inscattering,
                } => GpuFog {
                    mode: GPU_FOG_MODE_ATMOSPHERIC,
                    base_color: LinearRgba::from(fog.color).to_vec4(),
                    directional_light_color: LinearRgba::from(fog.directional_light_color)
                        .to_vec4(),
                    directional_light_exponent: fog.directional_light_exponent,
                    be: *extinction,
                    bi: *inscattering,
                },
            }
        } else {
            // If no fog is added to a camera, by default it's off
            GpuFog {
                mode: GPU_FOG_MODE_OFF,
                ..Default::default()
            }
        };

        // This is later read by `SetMeshViewBindGroup<I>`
        commands.entity(entity).insert(ViewFogUniformOffset {
            offset: writer.write(&gpu_fog),
        });
    }
}
