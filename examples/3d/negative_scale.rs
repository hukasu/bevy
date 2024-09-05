//! Demonstrates lighting on models flipped with negative scale

use bevy::{
    app::{App, Startup, Update},
    asset::Assets,
    color::Color,
    math::{Dir3, Vec2, Vec3},
    pbr::{DirectionalLightBundle, PbrBundle, StandardMaterial},
    prelude::{Camera3dBundle, Commands, Gizmos, Mesh, Plane3d, ResMut, Transform},
    DefaultPlugins,
};

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins);

    app.add_systems(
        Startup,
        (spawn_camera, spawn_directional_light, spawn_planes),
    )
    .add_systems(Update, draw_gizmos);

    app.run();
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0., 0., 10.).looking_at(Vec3::splat(0.), Dir3::Y),
        ..Default::default()
    });
}

fn spawn_directional_light(mut commands: Commands) {
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_xyz(0., 0., 10.).looking_at(Vec3::splat(0.), Dir3::Y),
        ..Default::default()
    });
}

fn spawn_planes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let plane = Plane3d::new(Vec3::Z, Vec2::splat(1.));
    let plane_handle = meshes.add(plane);

    let material = StandardMaterial {
        base_color: Color::WHITE,
        cull_mode: None,
        ..Default::default()
    };
    let material_handle = materials.add(material);

    commands.spawn(PbrBundle {
        mesh: plane_handle.clone(),
        material: material_handle.clone(),
        transform: Transform::from_xyz(-2., 2., 0.),
        ..Default::default()
    });
    commands.spawn(PbrBundle {
        mesh: plane_handle.clone(),
        material: material_handle.clone(),
        transform: Transform::from_xyz(-2., -2., 0.).with_scale(Vec3::new(-1., 1., 1.)),
        ..Default::default()
    });
    commands.spawn(PbrBundle {
        mesh: plane_handle.clone(),
        material: material_handle.clone(),
        transform: Transform::from_xyz(2., 2., 0.).with_scale(Vec3::new(1., -1., 1.)),
        ..Default::default()
    });
    commands.spawn(PbrBundle {
        mesh: plane_handle.clone(),
        material: material_handle.clone(),
        transform: Transform::from_xyz(2., -2., 0.).with_scale(Vec3::new(1., 1., -1.)),
        ..Default::default()
    });
}

fn draw_gizmos(mut gizmos: Gizmos) {
    let color = Color::BLACK;
    // -X
    gizmos.line(Vec3::new(-2.5, -2., 1.), Vec3::new(-2., -2., 1.), color);
    gizmos.line(Vec3::new(-2., -2.25, 1.), Vec3::new(-1.5, -1.75, 1.), color);
    gizmos.line(Vec3::new(-2., -1.75, 1.), Vec3::new(-1.5, -2.25, 1.), color);

    // -Y
    gizmos.line(Vec3::new(1.5, 2., 1.), Vec3::new(2., 2., 1.), color);
    gizmos.line(Vec3::new(2., 2.25, 1.), Vec3::new(2.25, 2., 1.), color);
    gizmos.line(Vec3::new(2.5, 2.25, 1.), Vec3::new(2.25, 2., 1.), color);
    gizmos.line(Vec3::new(2.25, 1.75, 1.), Vec3::new(2.25, 2., 1.), color);

    // -Z
    gizmos.line(Vec3::new(1.5, -2., 1.), Vec3::new(2., -2., 1.), color);
    gizmos.line(Vec3::new(2., -1.75, 1.), Vec3::new(2.5, -1.75, 1.), color);
    gizmos.line(Vec3::new(2.5, -1.75, 1.), Vec3::new(2., -2.25, 1.), color);
    gizmos.line(Vec3::new(2., -2.25, 1.), Vec3::new(2.5, -2.25, 1.), color);
}
