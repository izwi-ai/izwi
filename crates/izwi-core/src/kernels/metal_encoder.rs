use candle_metal_kernels::metal::{Buffer, CommandsGuard};
use objc2_metal::MTLSize;

pub(crate) trait IzwiMetalCommandEncoderExt {
    fn set_input_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize);
    fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize);
    fn set_bytes<T>(&self, index: usize, data: &T);
    fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize);
    fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    );
}

impl IzwiMetalCommandEncoderExt for CommandsGuard<'_> {
    fn set_input_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        self.as_ref().set_input_buffer(index, buffer, offset);
    }

    fn set_output_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        self.as_ref().set_output_buffer(index, buffer, offset);
    }

    fn set_bytes<T>(&self, index: usize, data: &T) {
        self.as_ref().set_bytes(index, data);
    }

    fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.as_ref()
            .dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.as_ref()
            .dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }
}
