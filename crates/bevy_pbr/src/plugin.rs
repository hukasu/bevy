use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, AssetApp, Assets, Handle};
use bevy_color::Color;
use bevy_render::{render_resource::Shader, RenderDebugFlags};
use bevy_render_3d::{
    decal::forward::ForwardDecalPlugin, deferred::plugin::DeferredLightingPlugin,
    light::plugin::LightPlugin, material::plugin::MaterialPlugin, shadow::plugin::ShadowPlugin,
};

use crate::prelude::StandardMaterial;

pub const PBR_TYPES_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("b0330585-2335-4268-9032-a6c4c2d932f6");
pub const PBR_BINDINGS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("13834c18-c7ec-4c4b-bbbd-432c3ba4cace");
pub const UTILS_HANDLE: Handle<Shader> = weak_handle!("0a32978f-2744-4608-98b6-4c3000a0638d");
pub const PBR_LIGHTING_HANDLE: Handle<Shader> =
    weak_handle!("de0cf697-2876-49a0-aa0f-f015216f70c2");
pub const PBR_TRANSMISSION_HANDLE: Handle<Shader> =
    weak_handle!("22482185-36bb-4c16-9b93-a20e6d4a2725");
pub const SHADOWS_HANDLE: Handle<Shader> = weak_handle!("ff758c5a-3927-4a15-94c3-3fbdfc362590");
pub const SHADOW_SAMPLING_HANDLE: Handle<Shader> =
    weak_handle!("f6bf5843-54bc-4e39-bd9d-56bfcd77b033");
pub const PBR_FRAGMENT_HANDLE: Handle<Shader> =
    weak_handle!("1bd3c10d-851b-400c-934a-db489d99cc50");
pub const PBR_SHADER_HANDLE: Handle<Shader> = weak_handle!("0eba65ed-3e5b-4752-93ed-e8097e7b0c84");
pub const PBR_PREPASS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("9afeaeab-7c45-43ce-b322-4b97799eaeb9");
pub const PBR_FUNCTIONS_HANDLE: Handle<Shader> =
    weak_handle!("815b8618-f557-4a96-91a5-a2fb7e249fb0");
pub const PBR_AMBIENT_HANDLE: Handle<Shader> = weak_handle!("4a90b95b-112a-4a10-9145-7590d6f14260");
pub const PARALLAX_MAPPING_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("6cf57d9f-222a-429a-bba4-55ba9586e1d4");
pub const VIEW_TRANSFORMATIONS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("ec047703-cde3-4876-94df-fed121544abb");
pub const PBR_PREPASS_FUNCTIONS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("77b1bd3a-877c-4b2c-981b-b9c68d1b774a");
pub const RGB9E5_FUNCTIONS_HANDLE: Handle<Shader> =
    weak_handle!("90c19aa3-6a11-4252-8586-d9299352e94f");

/// Sets up the entire PBR infrastructure of bevy.
pub struct PbrPlugin {
    /// Controls if the prepass is enabled for the [`StandardMaterial`].
    /// For more information about what a prepass is, see the [`bevy_core_pipeline::prepass`] docs.
    pub prepass_enabled: bool,
    /// Controls if [`DeferredPbrLightingPlugin`] is added.
    pub add_default_deferred_lighting_plugin: bool,
    /// Debugging flags that can optionally be set when constructing the renderer.
    pub debug_flags: RenderDebugFlags,
}

impl Default for PbrPlugin {
    fn default() -> Self {
        Self {
            prepass_enabled: true,
            add_default_deferred_lighting_plugin: true,
            debug_flags: RenderDebugFlags::default(),
        }
    }
}

impl Plugin for PbrPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PBR_TYPES_SHADER_HANDLE,
            "shaders/pbr_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_BINDINGS_SHADER_HANDLE,
            "shaders/pbr_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, UTILS_HANDLE, "shaders/utils.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            PBR_LIGHTING_HANDLE,
            "shaders/pbr_lighting.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_TRANSMISSION_HANDLE,
            "shaders/pbr_transmission.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SHADOWS_HANDLE,
            "shaders/shadows.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SHADOW_SAMPLING_HANDLE,
            "shaders/shadow_sampling.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_FUNCTIONS_HANDLE,
            "shaders/pbr_functions.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RGB9E5_FUNCTIONS_HANDLE,
            "shaders/rgb9e5.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_AMBIENT_HANDLE,
            "shaders/pbr_ambient.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_FRAGMENT_HANDLE,
            "shaders/pbr_fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_SHADER_HANDLE,
            "shaders/pbr.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_PREPASS_FUNCTIONS_SHADER_HANDLE,
            "shaders/pbr_prepass_functions.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PBR_PREPASS_SHADER_HANDLE,
            "shaders/pbr_prepass.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            PARALLAX_MAPPING_SHADER_HANDLE,
            "shaders/parallax_mapping.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            VIEW_TRANSFORMATIONS_SHADER_HANDLE,
            "shaders/view_transformations.wgsl",
            Shader::from_wgsl
        );

        app.register_asset_reflect::<StandardMaterial>()
            .add_plugins((
                MaterialPlugin::<StandardMaterial> {
                    prepass_enabled: self.prepass_enabled,
                    debug_flags: self.debug_flags,
                    ..Default::default()
                },
                LightPlugin::<StandardMaterial>::new(true),
                ShadowPlugin::<StandardMaterial>::default(),
                ForwardDecalPlugin::<StandardMaterial>::default(),
            ));

        if self.add_default_deferred_lighting_plugin {
            app.add_plugins(DeferredLightingPlugin);
        }

        // Initialize the default material handle.
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .insert(
                &Handle::<StandardMaterial>::default(),
                StandardMaterial {
                    base_color: Color::srgb(1.0, 0.0, 0.5),
                    ..Default::default()
                },
            );
    }
}
