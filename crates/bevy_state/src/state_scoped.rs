#[cfg(feature = "bevy_reflect")]
use bevy_ecs::reflect::ReflectComponent;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    entity_disabling::Disabled,
    message::MessageReader,
    query::Allow,
    system::{Commands, Query},
};
#[cfg(feature = "bevy_reflect")]
use bevy_reflect::prelude::*;

use crate::state::{StateTransitionEvent, States};

/// Entities marked with this component will be removed
/// when the world's state of the matching type no longer matches the supplied value.
///
/// If you need to disable this behavior, add the attribute `#[states(scoped_entities = false)]` when deriving [`States`].
///
/// ```
/// use bevy_state::prelude::*;
/// use bevy_ecs::prelude::*;
/// use bevy_ecs::system::ScheduleSystem;
///
/// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, States)]
/// enum GameState {
///     #[default]
///     MainMenu,
///     SettingsMenu,
///     InGame,
/// }
///
/// # #[derive(Component)]
/// # struct Player;
///
/// fn spawn_player(mut commands: Commands) {
///     commands.spawn((
///         DespawnOnExit(GameState::InGame),
///         Player
///     ));
/// }
///
/// # struct AppMock;
/// # impl AppMock {
/// #     fn init_state<S>(&mut self) {}
/// #     fn add_systems<S, M>(&mut self, schedule: S, systems: impl IntoScheduleConfigs<ScheduleSystem, M>) {}
/// # }
/// # struct Update;
/// # let mut app = AppMock;
///
/// app.init_state::<GameState>();
/// app.add_systems(OnEnter(GameState::InGame), spawn_player);
/// ```
#[derive(Component, Clone)]
#[cfg_attr(feature = "bevy_reflect", derive(Reflect), reflect(Component, Clone))]
pub struct DespawnOnExit<S: States>(pub S);

impl<S> Default for DespawnOnExit<S>
where
    S: States + Default,
{
    fn default() -> Self {
        Self(S::default())
    }
}

/// Despawns entities marked with [`DespawnOnExit<S>`] when their state no
/// longer matches the world state.
///
/// If the entity has already been despawned no warning will be emitted.
pub fn despawn_entities_on_exit_state<S: States>(
    mut commands: Commands,
    mut transitions: MessageReader<StateTransitionEvent<S>>,
    query: Query<(Entity, &DespawnOnExit<S>), Allow<Disabled>>,
) {
    // We use the latest event, because state machine internals generate at most 1
    // transition event (per type) each frame. No event means no change happened
    // and we skip iterating all entities.
    let Some(transition) = transitions.read().last() else {
        return;
    };
    if transition.entered == transition.exited {
        return;
    }
    let Some(exited) = &transition.exited else {
        return;
    };
    for (entity, binding) in &query {
        if binding.0 == *exited {
            commands.entity(entity).try_despawn();
        }
    }
}

/// Entities marked with this component will be despawned
/// upon entering the given state.
///
/// ```
/// use bevy_state::prelude::*;
/// use bevy_ecs::{prelude::*, system::ScheduleSystem};
///
/// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, States)]
/// enum GameState {
///     #[default]
///     MainMenu,
///     SettingsMenu,
///     InGame,
/// }
///
/// # #[derive(Component)]
/// # struct Player;
///
/// fn spawn_player(mut commands: Commands) {
///     commands.spawn((
///         DespawnOnEnter(GameState::MainMenu),
///         Player
///     ));
/// }
///
/// # struct AppMock;
/// # impl AppMock {
/// #     fn init_state<S>(&mut self) {}
/// #     fn add_systems<S, M>(&mut self, schedule: S, systems: impl IntoScheduleConfigs<ScheduleSystem, M>) {}
/// # }
/// # struct Update;
/// # let mut app = AppMock;
///
/// app.init_state::<GameState>();
/// app.add_systems(OnEnter(GameState::InGame), spawn_player);
/// ```
#[derive(Component, Clone)]
#[cfg_attr(feature = "bevy_reflect", derive(Reflect), reflect(Component))]
pub struct DespawnOnEnter<S: States>(pub S);

/// Despawns entities marked with [`DespawnOnEnter<S>`] when their state
/// matches the world state.
///
/// If the entity has already been despawned no warning will be emitted.
pub fn despawn_entities_on_enter_state<S: States>(
    mut commands: Commands,
    mut transitions: MessageReader<StateTransitionEvent<S>>,
    query: Query<(Entity, &DespawnOnEnter<S>), Allow<Disabled>>,
) {
    // We use the latest event, because state machine internals generate at most 1
    // transition event (per type) each frame. No event means no change happened
    // and we skip iterating all entities.
    let Some(transition) = transitions.read().last() else {
        return;
    };
    if transition.entered == transition.exited {
        return;
    }
    let Some(entered) = &transition.entered else {
        return;
    };
    for (entity, binding) in &query {
        if binding.0 == *entered {
            commands.entity(entity).try_despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bevy_app::App;
    use bevy_ecs::entity_disabling::Disabled;

    use crate::{
        app::{AppExtStates, StatesPlugin},
        prelude::CommandsStatesExt,
    };

    /// Tests that [`DespawnOnExit`] and [`DespawnOnExit`] properly despawns
    /// an entity that has a disabling component.
    ///
    /// # Note
    /// Currently there is no query filter to allow all disabling components, so
    /// there is only a guarantee for [`Disabled`].
    /// <https://github.com/bevyengine/bevy/issues/21615>
    #[test]
    fn despawn_on_enter_exit_on_disabled_entity() {
        #[derive(Component)]
        struct AltDisabled;
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, States)]
        enum State {
            On,
            Off,
        }

        let mut app = App::new();
        app.add_plugins(StatesPlugin);
        app.register_disabling_component::<AltDisabled>();

        app.insert_state(State::On);
        app.update();

        let disabled_exit = app
            .world_mut()
            .spawn((DespawnOnExit(State::On), Disabled))
            .id();
        let alt_disabled_exit = app
            .world_mut()
            .spawn((DespawnOnExit(State::On), AltDisabled))
            .id();
        let disabled_enter = app
            .world_mut()
            .spawn((DespawnOnEnter(State::Off), Disabled))
            .id();
        let alt_disabled_enter = app
            .world_mut()
            .spawn((DespawnOnEnter(State::Off), AltDisabled))
            .id();

        assert!(app.world().get_entity(disabled_exit).is_ok());
        assert!(app.world().get_entity(alt_disabled_exit).is_ok());
        assert!(app.world().get_entity(disabled_enter).is_ok());
        assert!(app.world().get_entity(alt_disabled_enter).is_ok());

        app.world_mut().commands().set_state(State::Off);
        app.update();

        assert!(app.world().get_entity(disabled_exit).is_err());
        // TODO assert!(app.world().get_entity(alt_disabled_exit).is_err());
        assert!(app.world().get_entity(disabled_enter).is_err());
        // TODO assert!(app.world().get_entity(alt_disabled_enter).is_err());
    }
}
