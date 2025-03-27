pub mod commands;
pub mod plugin;
pub(crate) mod render;

use bevy_ecs::component::Component;
use bevy_math::Affine3A;
use bevy_render::render_resource::{binding_types, BindGroupLayoutEntryBuilder, TextureSampleType};

use crate::mesh_pipeline::render::pipeline::MeshPipelineViewLayoutKey;

#[derive(Component, PartialEq, Default)]
pub struct PreviousGlobalTransform(pub Affine3A);

pub(crate) fn get_bind_group_layout_entries(
    layout_key: MeshPipelineViewLayoutKey,
) -> [Option<BindGroupLayoutEntryBuilder>; 4] {
    let mut entries: [Option<BindGroupLayoutEntryBuilder>; 4] = [None; 4];

    let multisampled = layout_key.contains(MeshPipelineViewLayoutKey::MULTISAMPLED);

    if layout_key.contains(MeshPipelineViewLayoutKey::DEPTH_PREPASS) {
        // Depth texture
        entries[0] = if multisampled {
            Some(binding_types::texture_depth_2d_multisampled())
        } else {
            Some(binding_types::texture_depth_2d())
        };
    }

    if layout_key.contains(MeshPipelineViewLayoutKey::NORMAL_PREPASS) {
        // Normal texture
        entries[1] = if multisampled {
            Some(binding_types::texture_2d_multisampled(
                TextureSampleType::Float { filterable: false },
            ))
        } else {
            Some(binding_types::texture_2d(TextureSampleType::Float {
                filterable: false,
            }))
        };
    }

    if layout_key.contains(MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS) {
        // Motion Vectors texture
        entries[2] = if multisampled {
            Some(binding_types::texture_2d_multisampled(
                TextureSampleType::Float { filterable: false },
            ))
        } else {
            Some(binding_types::texture_2d(TextureSampleType::Float {
                filterable: false,
            }))
        };
    }

    if layout_key.contains(MeshPipelineViewLayoutKey::DEFERRED_PREPASS) {
        // Deferred texture
        entries[3] = Some(binding_types::texture_2d(TextureSampleType::Uint));
    }

    entries
}
