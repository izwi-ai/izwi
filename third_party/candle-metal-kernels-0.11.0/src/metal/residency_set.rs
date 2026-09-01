use super::{Buffer, Device};
use objc2::{
    rc::Retained,
    runtime::{AnyClass, ProtocolObject},
};
use objc2_metal::{MTLAllocation, MTLDevice as _, MTLResidencySet, MTLResidencySetDescriptor};

/// Keeps Metal buffers resident in GPU memory, removing the need for `useResource` calls.
///
/// Add the set to the command queue once via `MTLCommandQueue::addResidencySet`. Then
/// register every buffer at allocation time and unregister at free time.
pub struct ResidencySet {
    raw: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
}

unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    pub fn new(device: &Device) -> Self {
        // Residency sets were introduced in macOS 15. Candle supports older
        // Metal runtimes, where objc2's generated class accessor panics before
        // the existing `raw: None` fallback can be reached.
        if AnyClass::get(c"MTLResidencySetDescriptor").is_none() {
            return ResidencySet { raw: None };
        }
        let descriptor = MTLResidencySetDescriptor::new();
        let raw = device
            .as_ref()
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()
            .inspect(|set| set.requestResidency());
        ResidencySet { raw }
    }

    pub fn raw(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.raw.as_deref()
    }

    pub fn insert(&self, buf: &Buffer) {
        if let Some(set) = &self.raw {
            set.addAllocation(as_allocation(buf));
            set.commit();
        }
    }

    pub fn remove(&self, buf: &Buffer) {
        if let Some(set) = &self.raw {
            set.removeAllocation(as_allocation(buf));
            set.commit();
        }
    }
}

/// Cast a `&Buffer` to `&ProtocolObject<dyn MTLAllocation>`.
///
/// Safe because `MTLBuffer: MTLResource: MTLAllocation`. All `ProtocolObject<P>` are
/// `repr(C)` thin ObjC pointers — the cast only changes the static protocol marker,
/// not the pointer value or runtime dispatch.
fn as_allocation(buf: &Buffer) -> &ProtocolObject<dyn MTLAllocation> {
    unsafe {
        &*(buf.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}
