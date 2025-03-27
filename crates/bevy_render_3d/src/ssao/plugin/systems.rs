use bevy_core_pipeline::{
    core_3d::Camera3d,
    prepass::{DepthPrepass, NormalPrepass, ViewPrepassTextures},
};
use bevy_ecs::{
    entity::Entity,
    query::{Has, With},
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    camera::{Camera, ExtractedCamera, TemporalJitter},
    globals::GlobalsBuffer,
    render_resource::{
        BindGroupEntries, BufferInitDescriptor, BufferUsages, Extent3d, PipelineCache,
        SpecializedComputePipelines, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureViewDescriptor, TextureViewDimension,
    },
    renderer::RenderDevice,
    sync_world::RenderEntity,
    texture::TextureCache,
    view::{Msaa, ViewUniforms},
    Extract,
};
use bevy_utils::default;
use tracing::error;

use crate::ssao::{
    ssao::{
        ScreenSpaceAmbientOcclusionResources, SsaoBindGroups, SsaoPipelineId, SsaoPipelineKey,
        SsaoPipelines,
    },
    ScreenSpaceAmbientOcclusion,
};

pub fn extract_ssao_settings(
    mut commands: Commands,
    cameras: Extract<
        Query<
            (RenderEntity, &Camera, &ScreenSpaceAmbientOcclusion, &Msaa),
            (With<Camera3d>, With<DepthPrepass>, With<NormalPrepass>),
        >,
    >,
) {
    for (entity, camera, ssao_settings, msaa) in &cameras {
        if *msaa != Msaa::Off {
            error!(
                "SSAO is being used which requires Msaa::Off, but Msaa is currently set to Msaa::{:?}",
                *msaa
            );
            return;
        }
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("SSAO entity wasn't synced.");
        if camera.is_active {
            entity_commands.insert(ssao_settings.clone());
        } else {
            entity_commands.remove::<ScreenSpaceAmbientOcclusion>();
        }
    }
}

pub fn prepare_ssao_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &ScreenSpaceAmbientOcclusion)>,
) {
    for (entity, camera, ssao_settings) in &views {
        let Some(physical_viewport_size) = camera.physical_viewport_size else {
            continue;
        };
        let size = Extent3d {
            width: physical_viewport_size.x,
            height: physical_viewport_size.y,
            depth_or_array_layers: 1,
        };

        let preprocessed_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssao_preprocessed_depth_texture"),
                size,
                mip_level_count: 5,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let ssao_noisy_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssao_noisy_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let ssao_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssao_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let depth_differences_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssao_depth_differences_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let thickness_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("thickness_buffer"),
            contents: &ssao_settings.constant_object_thickness.to_le_bytes(),
            usage: BufferUsages::UNIFORM,
        });

        commands
            .entity(entity)
            .insert(ScreenSpaceAmbientOcclusionResources {
                preprocessed_depth_texture,
                ssao_noisy_texture,
                screen_space_ambient_occlusion_texture: ssao_texture,
                depth_differences_texture,
                thickness_buffer,
            });
    }
}

pub fn prepare_ssao_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<SsaoPipelines>>,
    pipeline: Res<SsaoPipelines>,
    views: Query<(Entity, &ScreenSpaceAmbientOcclusion, Has<TemporalJitter>)>,
) {
    for (entity, ssao_settings, temporal_jitter) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SsaoPipelineKey {
                quality_level: ssao_settings.quality_level,
                temporal_jitter,
            },
        );

        commands.entity(entity).insert(SsaoPipelineId(pipeline_id));
    }
}

pub fn prepare_ssao_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipelines: Res<SsaoPipelines>,
    view_uniforms: Res<ViewUniforms>,
    global_uniforms: Res<GlobalsBuffer>,
    views: Query<(
        Entity,
        &ScreenSpaceAmbientOcclusionResources,
        &ViewPrepassTextures,
    )>,
) {
    let (Some(view_uniforms), Some(globals_uniforms)) = (
        view_uniforms.uniforms.binding(),
        global_uniforms.buffer.binding(),
    ) else {
        return;
    };

    for (entity, ssao_resources, prepass_textures) in &views {
        let common_bind_group = render_device.create_bind_group(
            "ssao_common_bind_group",
            &pipelines.common_bind_group_layout,
            &BindGroupEntries::sequential((
                &pipelines.point_clamp_sampler,
                &pipelines.linear_clamp_sampler,
                view_uniforms.clone(),
            )),
        );

        let create_depth_view = |mip_level| {
            ssao_resources
                .preprocessed_depth_texture
                .texture
                .create_view(&TextureViewDescriptor {
                    label: Some("ssao_preprocessed_depth_texture_mip_view"),
                    base_mip_level: mip_level,
                    format: Some(TextureFormat::R16Float),
                    dimension: Some(TextureViewDimension::D2),
                    mip_level_count: Some(1),
                    ..default()
                })
        };

        let preprocess_depth_bind_group = render_device.create_bind_group(
            "ssao_preprocess_depth_bind_group",
            &pipelines.preprocess_depth_bind_group_layout,
            &BindGroupEntries::sequential((
                prepass_textures.depth_view().unwrap(),
                &create_depth_view(0),
                &create_depth_view(1),
                &create_depth_view(2),
                &create_depth_view(3),
                &create_depth_view(4),
            )),
        );

        let ssao_bind_group = render_device.create_bind_group(
            "ssao_ssao_bind_group",
            &pipelines.ssao_bind_group_layout,
            &BindGroupEntries::sequential((
                &ssao_resources.preprocessed_depth_texture.default_view,
                prepass_textures.normal_view().unwrap(),
                &pipelines.hilbert_index_lut,
                &ssao_resources.ssao_noisy_texture.default_view,
                &ssao_resources.depth_differences_texture.default_view,
                globals_uniforms.clone(),
                ssao_resources.thickness_buffer.as_entire_binding(),
            )),
        );

        let spatial_denoise_bind_group = render_device.create_bind_group(
            "ssao_spatial_denoise_bind_group",
            &pipelines.spatial_denoise_bind_group_layout,
            &BindGroupEntries::sequential((
                &ssao_resources.ssao_noisy_texture.default_view,
                &ssao_resources.depth_differences_texture.default_view,
                &ssao_resources
                    .screen_space_ambient_occlusion_texture
                    .default_view,
            )),
        );

        commands.entity(entity).insert(SsaoBindGroups {
            common_bind_group,
            preprocess_depth_bind_group,
            ssao_bind_group,
            spatial_denoise_bind_group,
        });
    }
}
