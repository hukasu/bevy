use std::time::Duration;

use uuid::Uuid;

use bevy::{
    animation::{AnimationTarget, AnimationTargetId},
    app::App,
    asset::{DirectAssetAccessExt, Handle},
    core::Name,
    math::Quat,
    prelude::{
        AnimationClip, AnimationGraph, AnimationPlayer, AnimationTransitions, BuildWorldChildren,
        Transform, TransformBundle, VariableCurve,
    },
};

/// [Issue 14776](https://github.com/bevyengine/bevy/issues/14766)
///
/// An [`AnimationClip`] that has an empty [`VariableCurve`] would causes an `attempt to subtract with overflow`
/// on play.
#[test]
fn empty_variable_curve() {
    let mut app = App::new();

    app.add_plugins((
        bevy_internal::asset::AssetPlugin::default(),
        bevy_internal::time::TimePlugin,
        bevy_internal::animation::AnimationPlugin,
    ));

    let root_target_id = AnimationTargetId(Uuid::new_v4());
    let point_target_id = AnimationTargetId(Uuid::new_v4());

    let mut animation_clip = AnimationClip::default();
    animation_clip.add_curve_to_target(
        root_target_id,
        VariableCurve {
            keyframe_timestamps: vec![0., 16.],
            keyframes: bevy::prelude::Keyframes::Rotation(vec![
                Quat::default(),
                Quat::from_euler(
                    bevy::math::EulerRot::XYZ,
                    std::f32::consts::FRAC_PI_2,
                    0.,
                    0.,
                ),
            ]),
            interpolation: bevy::prelude::Interpolation::Linear,
        },
    );
    animation_clip.add_curve_to_target(
        point_target_id,
        VariableCurve {
            keyframe_timestamps: vec![],
            keyframes: bevy::prelude::Keyframes::Rotation(vec![]),
            interpolation: bevy::prelude::Interpolation::Linear,
        },
    );

    let animation_clip_handle = app.world_mut().add_asset(animation_clip);

    let (animation_graph, node_index) = AnimationGraph::from_clip(animation_clip_handle);

    let animation_graph_handle: Handle<AnimationGraph> = app.world_mut().add_asset(animation_graph);

    let mut animation_player = AnimationPlayer::default();

    let mut animation_transitions = AnimationTransitions::new();
    animation_transitions.play(&mut animation_player, node_index, Duration::ZERO);

    let root = app
        .world_mut()
        .spawn((
            Name::new("root"),
            TransformBundle::default(),
            animation_player,
            animation_transitions,
            animation_graph_handle,
        ))
        .id();

    app.world_mut()
        .entity_mut(root)
        .insert(AnimationTarget {
            id: root_target_id,
            player: root,
        })
        .with_children(|builder| {
            builder.spawn((
                Name::new("point"),
                TransformBundle {
                    local: Transform::from_xyz(3., 0., 0.),
                    ..Default::default()
                },
                AnimationTarget {
                    id: point_target_id,
                    player: root,
                },
            ));
        });

    app.update();
}
