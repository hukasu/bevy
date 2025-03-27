use bevy_render::render_graph::RenderLabel;

/// Render graph nodes specific to 3D PBR rendering.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub enum NodeRender3d {
    /// Label for the shadow pass node that draws meshes that were visible
    /// from the light last frame.
    EarlyShadowPass,
    /// Label for the shadow pass node that draws meshes that became visible
    /// from the light this frame.
    LateShadowPass,
    /// Label for the screen space ambient occlusion render node.
    ScreenSpaceAmbientOcclusion,
    DeferredLightingPass,
    /// Label for the volumetric lighting pass.
    VolumetricFog,
    /// Label for the shader that transforms and culls meshes that were
    /// visible last frame.
    EarlyGpuPreprocess,
    /// Label for the shader that transforms and culls meshes that became
    /// visible this frame.
    LateGpuPreprocess,
    /// Label for the screen space reflections pass.
    ScreenSpaceReflections,
    /// Label for the node that builds indirect draw parameters for meshes
    /// that were visible last frame.
    EarlyPrepassBuildIndirectParameters,
    /// Label for the node that builds indirect draw parameters for meshes
    /// that became visible this frame.
    LatePrepassBuildIndirectParameters,
    /// Label for the node that builds indirect draw parameters for the main
    /// rendering pass, containing all meshes that are visible this frame.
    MainBuildIndirectParameters,
    ClearIndirectParametersMetadata,
}
