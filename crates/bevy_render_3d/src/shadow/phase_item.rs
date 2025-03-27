use core::ops::Range;

use bevy_ecs::entity::Entity;
use bevy_render::{
    render_phase::{
        BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
        PhaseItemExtraIndex,
    },
    render_resource::CachedRenderPipelineId,
    sync_world::MainEntity,
};

use super::render::{ShadowBatchSetKey, ShadowBinKey};

pub struct Shadow {
    /// Determines which objects can be placed into a *batch set*.
    ///
    /// Objects in a single batch set can potentially be multi-drawn together,
    /// if it's enabled and the current platform supports it.
    batch_set_key: ShadowBatchSetKey,
    /// Information that separates items into bins.
    bin_key: ShadowBinKey,
    representative_entity: (Entity, MainEntity),
    batch_range: Range<u32>,
    extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for Shadow {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.batch_set_key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for Shadow {
    type BatchSetKey = ShadowBatchSetKey;
    type BinKey = ShadowBinKey;

    #[inline]
    fn new(
        batch_set_key: Self::BatchSetKey,
        bin_key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Shadow {
            batch_set_key,
            bin_key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for Shadow {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.batch_set_key.pipeline
    }
}
