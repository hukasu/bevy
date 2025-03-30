use crate::components::{GlobalTransform, Transform};

use bevy_ecs::prelude::*;

#[cfg(feature = "std")]
pub use parallel::propagate_parent_transforms;
#[cfg(feature = "std")]
pub(super) use parallel::{get_propagation_roots, PropagationRoot};
#[cfg(feature = "std")]
#[expect(
    unused_imports,
    reason = "They need the same visibility as `propagate_parent_transforms`"
)]
pub(super) use parallel::{Promess, Reader, Writer};
#[cfg(not(feature = "std"))]
pub use serial::propagate_parent_transforms;

/// Update [`GlobalTransform`] component of entities that aren't in the hierarchy
///
/// Third party plugins should ensure that this is used in concert with
/// [`propagate_parent_transforms`] and [`compute_transform_leaves`].
pub fn sync_simple_transforms(
    mut query: ParamSet<(
        Query<
            (&Transform, &mut GlobalTransform),
            (
                Or<(Changed<Transform>, Added<GlobalTransform>)>,
                Without<ChildOf>,
                Without<Children>,
            ),
        >,
        Query<(Ref<Transform>, &mut GlobalTransform), (Without<ChildOf>, Without<Children>)>,
    )>,
    mut orphaned: RemovedComponents<ChildOf>,
) {
    // Update changed entities.
    query
        .p0()
        .par_iter_mut()
        .for_each(|(transform, mut global_transform)| {
            *global_transform = GlobalTransform::from(*transform);
        });
    // Update orphaned entities.
    let mut query = query.p1();
    let mut iter = query.iter_many_mut(orphaned.read());
    while let Some((transform, mut global_transform)) = iter.fetch_next() {
        if !transform.is_changed() && !global_transform.is_added() {
            *global_transform = GlobalTransform::from(*transform);
        }
    }
}

/// Compute leaf [`GlobalTransform`]s in parallel.
///
/// This is run after [`propagate_parent_transforms`], to ensure the parents' [`GlobalTransform`]s
/// have been computed. This makes computing leaf nodes at different levels of the hierarchy much
/// more cache friendly, because data can be iterated over densely from the same archetype.
pub fn compute_transform_leaves(
    parents: Query<Ref<GlobalTransform>, With<Children>>,
    mut leaves: Query<(Ref<Transform>, &mut GlobalTransform, &ChildOf), Without<Children>>,
) {
    leaves
        .par_iter_mut()
        .for_each(|(transform, mut global_transform, child_of)| {
            let Ok(parent_transform) = parents.get(child_of.parent) else {
                return;
            };
            if parent_transform.is_changed()
                || transform.is_changed()
                || global_transform.is_added()
            {
                *global_transform = parent_transform.mul_transform(*transform);
            }
        });
}

// TODO: This serial implementation isn't actually serial, it parallelizes across the roots.
// Additionally, this couples "no_std" with "single_threaded" when these two features should be
// independent.
//
// What we want to do in a future refactor is take the current "single threaded" implementation, and
// actually make it single threaded. This will remove any overhead associated with working on a task
// pool when you only have a single thread, and will have the benefit of removing the need for any
// unsafe. We would then make the multithreaded implementation work across std and no_std, but this
// is blocked a no_std compatible Channel, which is why this TODO is not yet implemented.
//
// This complexity might also not be needed. If the multithreaded implementation on a single thread
// is as fast as the single threaded implementation, we could simply remove the entire serial
// module, and make the multithreaded module no_std compatible.
//
/// Serial hierarchy traversal. Useful in `no_std` or single threaded contexts.
#[cfg(not(feature = "std"))]
mod serial {
    use crate::prelude::*;
    use alloc::vec::Vec;
    use bevy_ecs::prelude::*;

    /// Update [`GlobalTransform`] component of entities based on entity hierarchy and [`Transform`]
    /// component.
    ///
    /// Third party plugins should ensure that this is used in concert with
    /// [`sync_simple_transforms`](super::sync_simple_transforms) and
    /// [`compute_transform_leaves`](super::compute_transform_leaves).
    pub fn propagate_parent_transforms(
        mut root_query: Query<
            (Entity, &Children, Ref<Transform>, &mut GlobalTransform),
            Without<ChildOf>,
        >,
        mut orphaned: RemovedComponents<ChildOf>,
        transform_query: Query<
            (Ref<Transform>, &mut GlobalTransform, Option<&Children>),
            (With<ChildOf>, With<Children>),
        >,
        child_query: Query<(Entity, Ref<ChildOf>), With<GlobalTransform>>,
        mut orphaned_entities: Local<Vec<Entity>>,
    ) {
        orphaned_entities.clear();
        orphaned_entities.extend(orphaned.read());
        orphaned_entities.sort_unstable();
        root_query.par_iter_mut().for_each(
        |(entity, children, transform, mut global_transform)| {
            let changed = transform.is_changed() || global_transform.is_added() || orphaned_entities.binary_search(&entity).is_ok();
            if changed {
                *global_transform = GlobalTransform::from(*transform);
            }

            for (child, child_of) in child_query.iter_many(children) {
                assert_eq!(
                    child_of.parent, entity,
                    "Malformed hierarchy. This probably means that your hierarchy has been improperly maintained, or contains a cycle"
                );
                // SAFETY:
                // - `child` must have consistent parentage, or the above assertion would panic.
                //   Since `child` is parented to a root entity, the entire hierarchy leading to it
                //   is consistent.
                // - We may operate as if all descendants are consistent, since
                //   `propagate_recursive` will panic before continuing to propagate if it
                //   encounters an entity with inconsistent parentage.
                // - Since each root entity is unique and the hierarchy is consistent and
                //   forest-like, other root entities' `propagate_recursive` calls will not conflict
                //   with this one.
                // - Since this is the only place where `transform_query` gets used, there will be
                //   no conflicting fetches elsewhere.
                #[expect(unsafe_code, reason = "`propagate_recursive()` is unsafe due to its use of `Query::get_unchecked()`.")]
                unsafe {
                    propagate_recursive(
                        &global_transform,
                        &transform_query,
                        &child_query,
                        child,
                        changed || child_of.is_changed(),
                    );
                }
            }
        },
    );
    }

    /// Recursively propagates the transforms for `entity` and all of its descendants.
    ///
    /// # Panics
    ///
    /// If `entity`'s descendants have a malformed hierarchy, this function will panic occur before
    /// propagating the transforms of any malformed entities and their descendants.
    ///
    /// # Safety
    ///
    /// - While this function is running, `transform_query` must not have any fetches for `entity`,
    ///   nor any of its descendants.
    /// - The caller must ensure that the hierarchy leading to `entity` is well-formed and must
    ///   remain as a tree or a forest. Each entity must have at most one parent.
    #[expect(
        unsafe_code,
        reason = "This function uses `Query::get_unchecked()`, which can result in multiple mutable references if the preconditions are not met."
    )]
    unsafe fn propagate_recursive(
        parent: &GlobalTransform,
        transform_query: &Query<
            (Ref<Transform>, &mut GlobalTransform, Option<&Children>),
            (With<ChildOf>, With<Children>),
        >,
        child_query: &Query<(Entity, Ref<ChildOf>), With<GlobalTransform>>,
        entity: Entity,
        mut changed: bool,
    ) {
        let (global_matrix, children) = {
            let Ok((transform, mut global_transform, children)) =
            // SAFETY: This call cannot create aliased mutable references.
            //   - The top level iteration parallelizes on the roots of the hierarchy.
            //   - The caller ensures that each child has one and only one unique parent throughout
            //     the entire hierarchy.
            //
            // For example, consider the following malformed hierarchy:
            //
            //     A
            //   /   \
            //  B     C
            //   \   /
            //     D
            //
            // D has two parents, B and C. If the propagation passes through C, but the ChildOf
            // component on D points to B, the above check will panic as the origin parent does
            // match the recorded parent.
            //
            // Also consider the following case, where A and B are roots:
            //
            //  A       B
            //   \     /
            //    C   D
            //     \ /
            //      E
            //
            // Even if these A and B start two separate tasks running in parallel, one of them will
            // panic before attempting to mutably access E.
            (unsafe { transform_query.get_unchecked(entity) }) else {
                return;
            };

            changed |= transform.is_changed() || global_transform.is_added();
            if changed {
                *global_transform = parent.mul_transform(*transform);
            }
            (global_transform, children)
        };

        let Some(children) = children else { return };
        for (child, child_of) in child_query.iter_many(children) {
            assert_eq!(
            child_of.parent, entity,
            "Malformed hierarchy. This probably means that your hierarchy has been improperly maintained, or contains a cycle"
        );
            // SAFETY: The caller guarantees that `transform_query` will not be fetched for any
            // descendants of `entity`, so it is safe to call `propagate_recursive` for each child.
            //
            // The above assertion ensures that each child has one and only one unique parent
            // throughout the entire hierarchy.
            unsafe {
                propagate_recursive(
                    global_matrix.as_ref(),
                    transform_query,
                    child_query,
                    child,
                    changed || child_of.is_changed(),
                );
            }
        }
    }
}

// TODO: Relies on `std` until a `no_std` `mpsc` channel is available.
//
/// Parallel hierarchy traversal with a batched work sharing scheduler. Often 2-5 times faster than
/// the serial version.
#[cfg(feature = "std")]
mod parallel {
    use alloc::{collections::VecDeque, sync::Arc, vec::Vec};
    use core::{marker::PhantomData, mem};

    use bevy_ecs::{
        entity::{hash_set::EntityHashSet, unique_vec::UniqueEntityVec},
        prelude::*,
    };
    use bevy_platform_support::{
        collections::{HashMap, HashSet},
        hash::FixedHasher,
        sync::OnceLock,
    };
    use bevy_tasks::{futures_lite::future::yield_now, ComputeTaskPool, TaskPool};

    use crate::prelude::*;

    #[derive(Default, Resource)]
    pub struct PropagationRoot {
        roots: Vec<Entity>,
    }

    /// Create a set of entities that had their [`Transform`] changed.
    ///
    /// Only the entity of a hierarchy closest to the root is taken.
    pub fn get_propagation_roots(
        mut propagation_roots: ResMut<PropagationRoot>,
        transforms_changed: Query<Entity, Changed<Transform>>,
        parents: Query<&ChildOf>,
        // Stored as a local to preserve capacity across frames
        mut root_candidates: Local<Vec<Entity>>,
    ) {
        let mut changed = HashSet::new();
        let mut root_entities = HashSet::new();
        let mut connections = HashMap::new();
        let mut cycle_detection = HashSet::new();

        'outer: for transform_changed in transforms_changed {
            changed.insert(transform_changed);

            let mut cur_entity = transform_changed;
            cycle_detection.clear();
            // Doing this manually to reference to the last entity
            while let Ok(parent) = parents.get(cur_entity) {
                if !cycle_detection.insert(cur_entity) {
                    panic!("Unsound hierarchy. Hierarchy had a cycle.");
                }
                connections
                    .entry(parent.parent)
                    .and_modify(|set: &mut HashSet<_>| {
                        set.insert(cur_entity);
                    })
                    .or_insert_with(|| {
                        let mut set = HashSet::with_hasher(FixedHasher);
                        set.insert(cur_entity);
                        set
                    });
                if connections.contains_key(&parent.parent)
                    || root_entities.contains(&parent.parent)
                {
                    continue 'outer;
                }
                cur_entity = parent.parent;
            }
            root_entities.insert(cur_entity);
        }

        let mut queue = VecDeque::from_iter(root_entities);
        while let Some(entity) = queue.pop_front() {
            if changed.contains(&entity) {
                root_candidates.push(entity);
            } else {
                let Some(connected_to) = connections.remove(&entity) else {
                    unreachable!("Should never be None.");
                };
                queue.extend(connected_to);
            }
        }
        // tracing::info!("{:?}", propagation_roots);
        mem::swap(&mut propagation_roots.roots, &mut (*root_candidates));
    }

    /// Update [`GlobalTransform`] component of entities based on entity hierarchy and [`Transform`]
    /// component.
    ///
    /// Third party plugins should ensure that this is used in concert with
    /// [`sync_simple_transforms`](super::sync_simple_transforms) and
    /// [`compute_transform_leaves`](super::compute_transform_leaves).
    pub fn propagate_parent_transforms(
        mut propagation_roots: ResMut<PropagationRoot>,
        parents: Query<&ChildOf>,
        children: Query<&Children>,
        mut transforms: Query<(Entity, &Transform, &mut GlobalTransform)>,
        // Caching capacity by having them on locals
        mut tasks: Local<
            Vec<(
                usize,
                Entity,
                Promess<GlobalTransform, Reader>,
                Promess<GlobalTransform, Writer>,
            )>,
        >,
        mut entity_cache: Local<EntityHashSet>,
    ) {
        entity_cache.clear();
        entity_cache.insert(Entity::PLACEHOLDER);

        let mut branch = 0;
        for root in propagation_roots.roots.drain(..) {
            let promess = Promess::new();
            let parent = parents
                .get(root)
                .map(|parent| parent.parent)
                .unwrap_or(Entity::PLACEHOLDER);
            if let Ok((_transform_owner, _, global_transform)) = transforms.get(parent) {
                #[cfg(debug_assertions)]
                assert_eq!(_transform_owner, parent);
                promess.set(*global_transform);
            } else {
                promess.set(GlobalTransform::default());
            }
            depth_first_tasks_build(
                root,
                parent,
                promess.reader(),
                &mut tasks,
                children,
                0,
                &mut branch,
                &mut entity_cache,
            );
        }

        let unique_vec = UniqueEntityVec::from_iter(tasks.iter().map(|task| task.1));
        let transforms = transforms.iter_many_unique_mut(unique_vec);
        let mut tasks = transforms
            .zip(tasks.drain(..))
            .map(|(transform, task)| {
                #[cfg(debug_assertions)]
                assert_eq!(transform.0, task.1);
                (task.0, transform.1, transform.2, task.2, task.3)
            })
            .collect::<Vec<_>>();

        let task_pool = ComputeTaskPool::get_or_init(TaskPool::default);
        task_pool.scope(|s| {
            for task_group in tasks.chunk_by_mut(|a, b| a.0 == b.0) {
                s.spawn(propagate_task_group(task_group));
            }
        });
    }

    fn depth_first_tasks_build(
        entity: Entity,
        parent: Entity,
        promess: Promess<GlobalTransform, Reader>,
        tasks: &mut Vec<(
            usize,
            Entity,
            Promess<GlobalTransform, Reader>,
            Promess<GlobalTransform, Writer>,
        )>,
        children: Query<&Children>,
        depth: usize,
        branch: &mut usize,
        cache: &mut EntityHashSet,
    ) {
        if depth < 5 && cache.contains(&parent) {
            *branch += 1;
        }
        cache.insert(parent);

        let new_promess = Promess::new();
        let promess_reader = new_promess.reader();
        tasks.push((*branch, entity, promess, new_promess));

        if let Ok(ent_children) = children.get(entity) {
            for child in ent_children.iter() {
                depth_first_tasks_build(
                    child,
                    entity,
                    promess_reader.clone(),
                    tasks,
                    children,
                    depth + 1,
                    branch,
                    cache,
                );
            }
        }
    }

    async fn propagate_task_group(
        tasks: &mut [(
            usize,
            &Transform,
            Mut<'_, GlobalTransform>,
            Promess<GlobalTransform, Reader>,
            Promess<GlobalTransform, Writer>,
        )],
    ) {
        for (_, transform, global_transform, reader, writer) in tasks.iter_mut() {
            **global_transform = reader.get().await.mul_transform(**transform);
            writer.set(**global_transform);
        }
    }

    pub struct Reader;
    pub struct Writer;
    pub struct Promess<O, M> {
        output: Arc<OnceLock<O>>,
        _data: PhantomData<M>,
    }

    impl<O> Promess<O, Writer> {
        fn new() -> Promess<O, Writer> {
            let lock = Arc::new(OnceLock::new());
            Promess {
                output: lock,
                _data: PhantomData,
            }
        }

        fn reader(&self) -> Promess<O, Reader> {
            Promess {
                output: self.output.clone(),
                _data: PhantomData,
            }
        }

        fn set(&self, data: O) {
            if self.output.set(data).is_err() {
                unreachable!("`set` should only be called once.");
            }
        }
    }

    impl<O> Promess<O, Reader> {
        async fn get(&self) -> &O {
            while self.output.get().is_none() {
                // TODO busy wait, replace for something reasonable
                yield_now().await;
            }
            let Some(value) = self.output.get() else {
                unreachable!("Value should be available here.");
            };
            value
        }
    }

    impl<O> Clone for Promess<O, Reader> {
        fn clone(&self) -> Self {
            Self {
                output: self.output.clone(),
                _data: PhantomData,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use alloc::{vec, vec::Vec};
    use bevy_app::prelude::*;
    use bevy_ecs::{prelude::*, world::CommandQueue};
    use bevy_math::{vec3, Vec3};
    use bevy_tasks::{ComputeTaskPool, TaskPool};

    use crate::systems::*;

    #[test]
    fn correct_parent_removed() {
        ComputeTaskPool::get_or_init(TaskPool::default);
        let mut world = World::default();
        let offset_global_transform =
            |offset| GlobalTransform::from(Transform::from_xyz(offset, offset, offset));
        let offset_transform = |offset| Transform::from_xyz(offset, offset, offset);

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                sync_simple_transforms,
                propagate_parent_transforms,
                compute_transform_leaves,
            )
                .chain(),
        );

        let mut command_queue = CommandQueue::default();
        let mut commands = Commands::new(&mut command_queue, &world);
        let root = commands.spawn(offset_transform(3.3)).id();
        let parent = commands.spawn(offset_transform(4.4)).id();
        let child = commands.spawn(offset_transform(5.5)).id();
        commands.entity(parent).insert(ChildOf { parent: root });
        commands.entity(child).insert(ChildOf { parent });
        command_queue.apply(&mut world);
        schedule.run(&mut world);

        assert_eq!(
            world.get::<GlobalTransform>(parent).unwrap(),
            &offset_global_transform(4.4 + 3.3),
            "The transform systems didn't run, ie: `GlobalTransform` wasn't updated",
        );

        // Remove parent of `parent`
        let mut command_queue = CommandQueue::default();
        let mut commands = Commands::new(&mut command_queue, &world);
        commands.entity(parent).remove::<ChildOf>();
        command_queue.apply(&mut world);
        schedule.run(&mut world);

        assert_eq!(
            world.get::<GlobalTransform>(parent).unwrap(),
            &offset_global_transform(4.4),
            "The global transform of an orphaned entity wasn't updated properly",
        );

        // Remove parent of `child`
        let mut command_queue = CommandQueue::default();
        let mut commands = Commands::new(&mut command_queue, &world);
        commands.entity(child).remove::<ChildOf>();
        command_queue.apply(&mut world);
        schedule.run(&mut world);

        assert_eq!(
            world.get::<GlobalTransform>(child).unwrap(),
            &offset_global_transform(5.5),
            "The global transform of an orphaned entity wasn't updated properly",
        );
    }

    #[test]
    fn did_propagate() {
        ComputeTaskPool::get_or_init(TaskPool::default);
        let mut world = World::default();

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                sync_simple_transforms,
                propagate_parent_transforms,
                compute_transform_leaves,
            )
                .chain(),
        );

        // Root entity
        world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));

        let mut children = Vec::new();
        world
            .spawn(Transform::from_xyz(1.0, 0.0, 0.0))
            .with_children(|parent| {
                children.push(parent.spawn(Transform::from_xyz(0.0, 2.0, 0.)).id());
                children.push(parent.spawn(Transform::from_xyz(0.0, 0.0, 3.)).id());
            });
        schedule.run(&mut world);

        assert_eq!(
            *world.get::<GlobalTransform>(children[0]).unwrap(),
            GlobalTransform::from_xyz(1.0, 0.0, 0.0) * Transform::from_xyz(0.0, 2.0, 0.0)
        );

        assert_eq!(
            *world.get::<GlobalTransform>(children[1]).unwrap(),
            GlobalTransform::from_xyz(1.0, 0.0, 0.0) * Transform::from_xyz(0.0, 0.0, 3.0)
        );
    }

    #[test]
    fn did_propagate_command_buffer() {
        let mut world = World::default();

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                sync_simple_transforms,
                propagate_parent_transforms,
                compute_transform_leaves,
            )
                .chain(),
        );

        // Root entity
        let mut queue = CommandQueue::default();
        let mut commands = Commands::new(&mut queue, &world);
        let mut children = Vec::new();
        commands
            .spawn(Transform::from_xyz(1.0, 0.0, 0.0))
            .with_children(|parent| {
                children.push(parent.spawn(Transform::from_xyz(0.0, 2.0, 0.0)).id());
                children.push(parent.spawn(Transform::from_xyz(0.0, 0.0, 3.0)).id());
            });
        queue.apply(&mut world);
        schedule.run(&mut world);

        assert_eq!(
            *world.get::<GlobalTransform>(children[0]).unwrap(),
            GlobalTransform::from_xyz(1.0, 0.0, 0.0) * Transform::from_xyz(0.0, 2.0, 0.0)
        );

        assert_eq!(
            *world.get::<GlobalTransform>(children[1]).unwrap(),
            GlobalTransform::from_xyz(1.0, 0.0, 0.0) * Transform::from_xyz(0.0, 0.0, 3.0)
        );
    }

    #[test]
    fn correct_children() {
        ComputeTaskPool::get_or_init(TaskPool::default);
        let mut world = World::default();

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                sync_simple_transforms,
                propagate_parent_transforms,
                compute_transform_leaves,
            )
                .chain(),
        );

        // Add parent entities
        let mut children = Vec::new();
        let parent = {
            let mut command_queue = CommandQueue::default();
            let mut commands = Commands::new(&mut command_queue, &world);
            let parent = commands.spawn(Transform::from_xyz(1.0, 0.0, 0.0)).id();
            commands.entity(parent).with_children(|parent| {
                children.push(parent.spawn(Transform::from_xyz(0.0, 2.0, 0.0)).id());
                children.push(parent.spawn(Transform::from_xyz(0.0, 3.0, 0.0)).id());
            });
            command_queue.apply(&mut world);
            schedule.run(&mut world);
            parent
        };

        assert_eq!(
            world
                .get::<Children>(parent)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            children,
        );

        // Parent `e1` to `e2`.
        {
            let mut command_queue = CommandQueue::default();
            let mut commands = Commands::new(&mut command_queue, &world);
            commands.entity(children[1]).add_child(children[0]);
            command_queue.apply(&mut world);
            schedule.run(&mut world);
        }

        assert_eq!(
            world
                .get::<Children>(parent)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![children[1]]
        );

        assert_eq!(
            world
                .get::<Children>(children[1])
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![children[0]]
        );

        assert!(world.despawn(children[0]));

        schedule.run(&mut world);

        assert_eq!(
            world
                .get::<Children>(parent)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![children[1]]
        );
    }

    #[test]
    fn correct_transforms_when_no_children() {
        let mut app = App::new();
        ComputeTaskPool::get_or_init(TaskPool::default);

        app.add_systems(
            Update,
            (
                sync_simple_transforms,
                propagate_parent_transforms,
                compute_transform_leaves,
            )
                .chain(),
        );

        let translation = vec3(1.0, 0.0, 0.0);

        // These will be overwritten.
        let mut child = Entity::from_raw(0);
        let mut grandchild = Entity::from_raw(1);
        let parent = app
            .world_mut()
            .spawn(Transform::from_translation(translation))
            .with_children(|builder| {
                child = builder
                    .spawn(Transform::IDENTITY)
                    .with_children(|builder| {
                        grandchild = builder.spawn(Transform::IDENTITY).id();
                    })
                    .id();
            })
            .id();

        app.update();

        // check the `Children` structure is spawned
        assert_eq!(&**app.world().get::<Children>(parent).unwrap(), &[child]);
        assert_eq!(
            &**app.world().get::<Children>(child).unwrap(),
            &[grandchild]
        );
        // Note that at this point, the `GlobalTransform`s will not have updated yet, due to
        // `Commands` delay
        app.update();

        let mut state = app.world_mut().query::<&GlobalTransform>();
        for global in state.iter(app.world()) {
            assert_eq!(global, &GlobalTransform::from_translation(translation));
        }
    }

    #[test]
    #[should_panic]
    fn panic_when_hierarchy_cycle() {
        ComputeTaskPool::get_or_init(TaskPool::default);
        // We cannot directly edit ChildOf and Children, so we use a temp world to break the
        // hierarchy's invariants.
        let mut temp = World::new();
        let mut app = App::new();

        app.add_systems(
            Update,
            (
                propagate_parent_transforms,
                sync_simple_transforms,
                compute_transform_leaves,
            )
                .chain(),
        );

        fn setup_world(world: &mut World) -> (Entity, Entity) {
            let mut grandchild = Entity::from_raw(0);
            let child = world
                .spawn(Transform::IDENTITY)
                .with_children(|builder| {
                    grandchild = builder.spawn(Transform::IDENTITY).id();
                })
                .id();
            (child, grandchild)
        }

        let (temp_child, temp_grandchild) = setup_world(&mut temp);
        let (child, grandchild) = setup_world(app.world_mut());

        assert_eq!(temp_child, child);
        assert_eq!(temp_grandchild, grandchild);

        app.world_mut()
            .spawn(Transform::IDENTITY)
            .add_children(&[child]);

        let mut child_entity = app.world_mut().entity_mut(child);

        let mut grandchild_entity = temp.entity_mut(grandchild);

        #[expect(
            unsafe_code,
            reason = "ChildOf is not mutable but this is for a test to produce a scenario that cannot happen"
        )]
        // SAFETY: ChildOf is not mutable but this is for a test to produce a scenario that
        // cannot happen
        let mut a = unsafe { child_entity.get_mut_assume_mutable::<ChildOf>().unwrap() };

        // SAFETY: ChildOf is not mutable but this is for a test to produce a scenario that
        // cannot happen
        #[expect(
            unsafe_code,
            reason = "ChildOf is not mutable but this is for a test to produce a scenario that cannot happen"
        )]
        let mut b = unsafe {
            grandchild_entity
                .get_mut_assume_mutable::<ChildOf>()
                .unwrap()
        };

        core::mem::swap(a.as_mut(), b.as_mut());

        app.update();
    }

    #[test]
    fn global_transform_should_not_be_overwritten_after_reparenting() {
        let translation = Vec3::ONE;
        let mut world = World::new();

        // Create transform propagation schedule
        let mut schedule = Schedule::default();
        schedule.add_systems((
            sync_simple_transforms,
            propagate_parent_transforms,
            compute_transform_leaves,
        ));

        // Spawn a `Transform` entity with a local translation of `Vec3::ONE`
        let mut spawn_transform_bundle =
            || world.spawn(Transform::from_translation(translation)).id();

        // Spawn parent and child with identical transform bundles
        let parent = spawn_transform_bundle();
        let child = spawn_transform_bundle();
        world.entity_mut(parent).add_child(child);

        // Run schedule to propagate transforms
        schedule.run(&mut world);

        // Child should be positioned relative to its parent
        let parent_global_transform = *world.entity(parent).get::<GlobalTransform>().unwrap();
        let child_global_transform = *world.entity(child).get::<GlobalTransform>().unwrap();
        assert!(parent_global_transform
            .translation()
            .abs_diff_eq(translation, 0.1));
        assert!(child_global_transform
            .translation()
            .abs_diff_eq(2. * translation, 0.1));

        // Reparent child
        world.entity_mut(child).remove::<ChildOf>();
        world.entity_mut(parent).add_child(child);

        // Run schedule to propagate transforms
        schedule.run(&mut world);

        // Translations should be unchanged after update
        assert_eq!(
            parent_global_transform,
            *world.entity(parent).get::<GlobalTransform>().unwrap()
        );
        assert_eq!(
            child_global_transform,
            *world.entity(child).get::<GlobalTransform>().unwrap()
        );
    }
}
