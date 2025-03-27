use bevy_asset::Handle;
use bevy_render::render_resource::{BindGroupLayout, Shader};

use crate::mesh_pipeline::render::pipeline::{MeshPipeline, MeshPipelineKey};

use super::MaterialExtension;

pub struct MaterialExtensionPipeline {
    pub mesh_pipeline: MeshPipeline,
    pub material_layout: BindGroupLayout,
    pub vertex_shader: Option<Handle<Shader>>,
    pub fragment_shader: Option<Handle<Shader>>,
    pub bindless: bool,
}

pub struct MaterialExtensionKey<E: MaterialExtension> {
    pub mesh_key: MeshPipelineKey,
    pub bind_group_data: E::Data,
}
