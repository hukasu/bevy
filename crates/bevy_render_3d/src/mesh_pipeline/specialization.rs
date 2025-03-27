use core::marker::PhantomData;

use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Tick, entity::Entity, resource::Resource};
use bevy_platform_support::collections::HashMap;
use bevy_render::sync_world::MainEntityHashMap;
use bevy_render::view::RetainedViewEntity;

#[derive(Resource, Deref, DerefMut, Clone, Debug)]
pub struct EntitiesNeedingSpecialization<M> {
    #[deref]
    pub entities: Vec<Entity>,
    _marker: PhantomData<M>,
}

impl<M> Default for EntitiesNeedingSpecialization<M> {
    fn default() -> Self {
        Self {
            entities: Default::default(),
            _marker: Default::default(),
        }
    }
}

#[derive(Resource, Deref, DerefMut, Clone, Debug)]
pub struct EntitySpecializationTicks<M> {
    #[deref]
    pub entities: MainEntityHashMap<Tick>,
    _marker: PhantomData<M>,
}

impl<M> Default for EntitySpecializationTicks<M> {
    fn default() -> Self {
        Self {
            entities: MainEntityHashMap::default(),
            _marker: Default::default(),
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default, Debug, Clone)]
pub struct ViewSpecializationTicks(HashMap<RetainedViewEntity, Tick>);
