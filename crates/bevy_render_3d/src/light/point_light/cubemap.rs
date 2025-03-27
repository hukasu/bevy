use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_math::Vec3;
use bevy_reflect::{prelude::ReflectDefault, Reflect};

use crate::light::VisibleMeshEntities;

pub(crate) struct CubeMapFace {
    pub(crate) target: Vec3,
    pub(crate) up: Vec3,
}

// Cubemap faces are [+X, -X, +Y, -Y, +Z, -Z], per https://www.w3.org/TR/webgpu/#texture-view-creation
// Note: Cubemap coordinates are left-handed y-up, unlike the rest of Bevy.
// See https://registry.khronos.org/vulkan/specs/1.2/html/chap16.html#_cube_map_face_selection
//
// For each cubemap face, we take care to specify the appropriate target/up axis such that the rendered
// texture using Bevy's right-handed y-up coordinate space matches the expected cubemap face in
// left-handed y-up cubemap coordinates.
pub(crate) const CUBE_MAP_FACES: [CubeMapFace; 6] = [
    // +X
    CubeMapFace {
        target: Vec3::X,
        up: Vec3::Y,
    },
    // -X
    CubeMapFace {
        target: Vec3::NEG_X,
        up: Vec3::Y,
    },
    // +Y
    CubeMapFace {
        target: Vec3::Y,
        up: Vec3::Z,
    },
    // -Y
    CubeMapFace {
        target: Vec3::NEG_Y,
        up: Vec3::NEG_Z,
    },
    // +Z (with left-handed conventions, pointing forwards)
    CubeMapFace {
        target: Vec3::NEG_Z,
        up: Vec3::Y,
    },
    // -Z (with left-handed conventions, pointing backwards)
    CubeMapFace {
        target: Vec3::Z,
        up: Vec3::Y,
    },
];

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct CubemapVisibleEntities {
    #[reflect(ignore, clone)]
    data: [VisibleMeshEntities; 6],
}

impl CubemapVisibleEntities {
    pub fn get(&self, i: usize) -> &VisibleMeshEntities {
        &self.data[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut VisibleMeshEntities {
        &mut self.data[i]
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &VisibleMeshEntities> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut VisibleMeshEntities> {
        self.data.iter_mut()
    }
}
