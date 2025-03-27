use core::marker::PhantomData;

use bevy_platform_support::collections::HashSet;
use bevy_render::{
    render_resource::{
        BindGroupEntry, BindGroupLayout, Buffer, BufferInitDescriptor, BufferUsages,
        OwnedBindingResource, PreparedBindGroup, UnpreparedBindGroup,
    },
    renderer::RenderDevice,
};
use bevy_utils::default;

use crate::material::{material::MaterialBindGroupSlot, Material};

use super::material::{MaterialBindGroupIndex, MaterialBindingId};

/// The allocator that stores bind groups for non-bindless materials.
pub struct MaterialBindGroupNonBindlessAllocator<M>
where
    M: Material,
{
    /// A mapping from [`MaterialBindGroupIndex`] to the bind group allocated in
    /// each slot.
    bind_groups: Vec<Option<MaterialNonBindlessAllocatedBindGroup<M>>>,
    /// The bind groups that are dirty and need to be prepared.
    ///
    /// To prepare the bind groups, call
    /// [`MaterialBindGroupAllocator::prepare_bind_groups`].
    to_prepare: HashSet<MaterialBindGroupIndex>,
    /// A list of free bind group indices.
    free_indices: Vec<MaterialBindGroupIndex>,
    phantom: PhantomData<M>,
}
impl<M> MaterialBindGroupNonBindlessAllocator<M>
where
    M: Material,
{
    /// Creates a new [`MaterialBindGroupNonBindlessAllocator`] managing the
    /// bind groups for a single non-bindless material.
    pub fn new() -> MaterialBindGroupNonBindlessAllocator<M> {
        MaterialBindGroupNonBindlessAllocator {
            bind_groups: vec![],
            to_prepare: HashSet::default(),
            free_indices: vec![],
            phantom: PhantomData,
        }
    }

    /// Inserts a bind group, either unprepared or prepared, into this allocator
    /// and returns a [`MaterialBindingId`].
    ///
    /// The returned [`MaterialBindingId`] can later be used to fetch the bind
    /// group.
    pub fn allocate(
        &mut self,
        bind_group: MaterialNonBindlessAllocatedBindGroup<M>,
    ) -> MaterialBindingId {
        let group_id = self
            .free_indices
            .pop()
            .unwrap_or(MaterialBindGroupIndex(self.bind_groups.len() as u32));
        if self.bind_groups.len() < *group_id as usize + 1 {
            self.bind_groups
                .resize_with(*group_id as usize + 1, || None);
        }

        if matches!(
            bind_group,
            MaterialNonBindlessAllocatedBindGroup::Unprepared { .. }
        ) {
            self.to_prepare.insert(group_id);
        }

        self.bind_groups[*group_id as usize] = Some(bind_group);

        MaterialBindingId {
            group: group_id,
            slot: default(),
        }
    }

    /// Inserts an unprepared bind group into this allocator and returns a
    /// [`MaterialBindingId`].
    pub fn allocate_unprepared(
        &mut self,
        unprepared_bind_group: UnpreparedBindGroup<M::Data>,
        bind_group_layout: BindGroupLayout,
    ) -> MaterialBindingId {
        self.allocate(MaterialNonBindlessAllocatedBindGroup::Unprepared {
            bind_group: unprepared_bind_group,
            layout: bind_group_layout,
        })
    }

    /// Inserts an prepared bind group into this allocator and returns a
    /// [`MaterialBindingId`].
    pub fn allocate_prepared(
        &mut self,
        prepared_bind_group: PreparedBindGroup<M::Data>,
    ) -> MaterialBindingId {
        self.allocate(MaterialNonBindlessAllocatedBindGroup::Prepared {
            bind_group: prepared_bind_group,
            uniform_buffers: vec![],
        })
    }

    /// Deallocates the bind group with the given binding ID.
    pub fn free(&mut self, binding_id: MaterialBindingId) {
        debug_assert_eq!(binding_id.slot, MaterialBindGroupSlot(0));
        debug_assert!(self.bind_groups[*binding_id.group as usize].is_some());
        self.bind_groups[*binding_id.group as usize] = None;
        self.to_prepare.remove(&binding_id.group);
        self.free_indices.push(binding_id.group);
    }

    /// Returns a wrapper around the bind group with the given index.
    pub fn get(&self, group: MaterialBindGroupIndex) -> Option<MaterialNonBindlessSlab<M>> {
        self.bind_groups[group.0 as usize]
            .as_ref()
            .map(|bind_group| match bind_group {
                MaterialNonBindlessAllocatedBindGroup::Prepared { bind_group, .. } => {
                    MaterialNonBindlessSlab::Prepared(bind_group)
                }
                MaterialNonBindlessAllocatedBindGroup::Unprepared { bind_group, .. } => {
                    MaterialNonBindlessSlab::Unprepared(bind_group)
                }
            })
    }

    /// Prepares any as-yet unprepared bind groups that this allocator is
    /// managing.
    ///
    /// Unprepared bind groups can be added to this allocator with
    /// [`Self::allocate_unprepared`]. Such bind groups will defer being
    /// prepared until the next time this method is called.
    pub fn prepare_bind_groups(&mut self, render_device: &RenderDevice) {
        for bind_group_index in core::mem::take(&mut self.to_prepare) {
            let Some(MaterialNonBindlessAllocatedBindGroup::Unprepared {
                bind_group: unprepared_bind_group,
                layout: bind_group_layout,
            }) = core::mem::take(&mut self.bind_groups[*bind_group_index as usize])
            else {
                panic!("Allocation didn't exist or was already prepared");
            };

            // Pack any `Data` into uniform buffers.
            let mut uniform_buffers = vec![];
            for (index, binding) in unprepared_bind_group.bindings.iter() {
                let OwnedBindingResource::Data(ref owned_data) = *binding else {
                    continue;
                };
                let label = format!("material uniform data {}", *index);
                let uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some(&label),
                    contents: &owned_data.0,
                    usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                });
                uniform_buffers.push(uniform_buffer);
            }

            // Create bind group entries.
            let mut bind_group_entries = vec![];
            let mut uniform_buffers_iter = uniform_buffers.iter();
            for (index, binding) in unprepared_bind_group.bindings.iter() {
                match *binding {
                    OwnedBindingResource::Data(_) => {
                        bind_group_entries.push(BindGroupEntry {
                            binding: *index,
                            resource: uniform_buffers_iter
                                .next()
                                .expect("We should have created uniform buffers for each `Data`")
                                .as_entire_binding(),
                        });
                    }
                    _ => bind_group_entries.push(BindGroupEntry {
                        binding: *index,
                        resource: binding.get_binding(),
                    }),
                }
            }

            // Create the bind group.
            let bind_group = render_device.create_bind_group(
                M::label(),
                &bind_group_layout,
                &bind_group_entries,
            );

            self.bind_groups[*bind_group_index as usize] =
                Some(MaterialNonBindlessAllocatedBindGroup::Prepared {
                    bind_group: PreparedBindGroup {
                        bindings: unprepared_bind_group.bindings,
                        bind_group,
                        data: unprepared_bind_group.data,
                    },
                    uniform_buffers,
                });
        }
    }
}

/// A single bind group that a [`MaterialBindGroupNonBindlessAllocator`] is
/// currently managing.
enum MaterialNonBindlessAllocatedBindGroup<M>
where
    M: Material,
{
    /// An unprepared bind group.
    ///
    /// The allocator prepares all outstanding unprepared bind groups when
    /// [`MaterialBindGroupNonBindlessAllocator::prepare_bind_groups`] is
    /// called.
    Unprepared {
        /// The unprepared bind group, including extra data.
        bind_group: UnpreparedBindGroup<M::Data>,
        /// The layout of that bind group.
        layout: BindGroupLayout,
    },
    /// A bind group that's already been prepared.
    Prepared {
        bind_group: PreparedBindGroup<M::Data>,
        #[expect(dead_code, reason = "These buffers are only referenced by bind groups")]
        uniform_buffers: Vec<Buffer>,
    },
}

/// A single bind group that the [`MaterialBindGroupNonBindlessAllocator`]
/// manages.
pub enum MaterialNonBindlessSlab<'a, M>
where
    M: Material,
{
    /// A slab that has a bind group.
    Prepared(&'a PreparedBindGroup<M::Data>),
    /// A slab that doesn't yet have a bind group.
    Unprepared(&'a UnpreparedBindGroup<M::Data>),
}
