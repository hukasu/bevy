mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_instances::ExtractInstancesPlugin,
    render_resource::{BufferBindingType, Shader},
    renderer::RenderDevice,
    ExtractSchedule, Render, RenderApp, RenderSet,
};

use crate::{
    cluster::{plugin::ClusterableObjectPlugin, CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT},
    light_probe::{EnvironmentMapLight, IrradianceVolume},
};

use super::{
    environment_map::{EnvironmentMapIds, EnvironmentMapUniformBuffer},
    light_probes::LightProbesBuffer,
    LightProbe,
};

use systems::{
    gather_environment_map_uniform, gather_light_probes, prepare_environment_uniform_buffer,
    upload_light_probes,
};

const LIGHT_PROBE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("e80a2ae6-1c5a-4d9a-a852-d66ff0e6bf7f");
/// A handle to the environment map helper shader.
const ENVIRONMENT_MAP_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("d38c4ec4-e84c-468f-b485-bf44745db937");
const IRRADIANCE_VOLUME_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("7fc7dcd8-3f90-4124-b093-be0e53e08205");

/// Adds support for light probes: cuboid bounding regions that apply global
/// illumination to objects within them.
///
/// This also adds support for view environment maps: diffuse and specular
/// cubemaps applied to all objects that a view renders.
pub struct LightProbePlugin;

impl Plugin for LightProbePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            LIGHT_PROBE_SHADER_HANDLE,
            "shaders/light_probe.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            ENVIRONMENT_MAP_SHADER_HANDLE,
            "shaders/environment_map.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            IRRADIANCE_VOLUME_SHADER_HANDLE,
            "shaders/irradiance_volume.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<LightProbe>()
            .register_type::<EnvironmentMapLight>()
            .register_type::<IrradianceVolume>();

        if let Some(render_device) = app.world().get_resource::<RenderDevice>() {
            let clustered_forward_buffer_binding_type = render_device
                .get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT);
            let supports_storage_buffers = matches!(
                clustered_forward_buffer_binding_type,
                BufferBindingType::Storage { .. }
            );

            // Gather up light probes, but only if we're clustering them.
            //
            // UBOs aren't large enough to hold indices for light probes, so we can't
            // cluster light probes on such platforms (mainly WebGL 2). Besides, those
            // platforms typically lack bindless textures, so multiple light probes
            // wouldn't be supported anyhow.
            if supports_storage_buffers {
                app.add_plugins(ClusterableObjectPlugin::<2, EnvironmentMapLight>::default());
                app.add_plugins(ClusterableObjectPlugin::<3, IrradianceVolume>::default());
            }
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_plugins(ExtractInstancesPlugin::<EnvironmentMapIds>::new())
            .init_resource::<LightProbesBuffer>()
            .init_resource::<EnvironmentMapUniformBuffer>()
            .add_systems(ExtractSchedule, gather_environment_map_uniform)
            .add_systems(ExtractSchedule, gather_light_probes::<EnvironmentMapLight>)
            .add_systems(ExtractSchedule, gather_light_probes::<IrradianceVolume>)
            .add_systems(
                Render,
                (upload_light_probes, prepare_environment_uniform_buffer)
                    .in_set(RenderSet::PrepareResources),
            );
    }
}
