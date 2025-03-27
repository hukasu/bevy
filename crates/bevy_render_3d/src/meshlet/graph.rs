use bevy_render::render_graph::RenderLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub enum NodeMeshlet {
    VisibilityBufferRasterPass,
    Prepass,
    DeferredPrepass,
    MainOpaquePass,
}
