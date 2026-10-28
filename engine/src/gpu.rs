//! Per-process Windows GPU Engine PDH telemetry. Construct and poll this meter
//! on a normal telemetry thread, never in an audio callback. No NVML dependency.
#[cfg(any(windows, test))]
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub const MIN_SAMPLE_INTERVAL: Duration = Duration::from_millis(200);

/// Process contribution to the busiest matching physical engine, not board-wide
/// utilization and not a sum of simultaneously running engines.
#[derive(Clone, Debug)]
pub struct GpuSnapshot {
    pub utilization: f32,
    pub three_d: f32,
    pub compute: f32,
    pub engines: Vec<EngineSample>,
}
#[derive(Clone, Debug)]
pub struct EngineSample {
    pub instance: String,
    pub utilization: f32,
}
#[cfg(any(windows, test))]
fn engine_identity(instance: &str, pid: u32) -> Option<(&str, &str)> {
    let remainder = instance.strip_prefix("pid_")?;
    let (number, device) = remainder.split_once('_')?;
    if number.parse::<u32>().ok()? != pid {
        return None;
    }
    let (identity, kind) = device.split_once("_engtype_")?;
    let kind = kind.split('#').next()?;
    if kind == "3D" || kind.starts_with("Compute") {
        Some((identity, kind))
    } else {
        None
    }
}
#[cfg(any(windows, test))]
fn aggregate(pid: u32, rows: impl IntoIterator<Item = (String, f64)>) -> GpuSnapshot {
    let mut engines: BTreeMap<String, EngineSample> = BTreeMap::new();
    let (mut three_d, mut compute) = (0.0_f32, 0.0_f32);
    for (instance, percentage) in rows {
        let Some((identity, kind)) = engine_identity(&instance, pid) else {
            continue;
        };
        if !percentage.is_finite() {
            continue;
        }
        let value = (percentage / 100.0).clamp(0.0, 1.0) as f32;
        if kind == "3D" {
            three_d = three_d.max(value);
        } else {
            compute = compute.max(value);
        }
        let entry = engines
            .entry(identity.to_owned())
            .or_insert_with(|| EngineSample {
                instance: instance.clone(),
                utilization: value,
            });
        if value > entry.utilization {
            entry.utilization = value;
            entry.instance = instance;
        }
    }
    let utilization = engines
        .values()
        .map(|item| item.utilization)
        .fold(0.0_f32, f32::max);
    GpuSnapshot {
        utilization,
        three_d,
        compute,
        engines: engines.into_values().collect(),
    }
}

pub struct GpuMeter {
    pid: u32,
    last_collect: Instant,
    latest: Option<GpuSnapshot>,
    #[cfg(windows)]
    pdh: windows_pdh::Query,
}
impl GpuMeter {
    /// Monitor this process. The initial counter collection happens here.
    pub fn new() -> Option<Self> {
        Self::for_pid(std::process::id())
    }
    /// Diagnostic helper: monitor an explicitly named process, not the probe's PID.
    pub fn for_pid(pid: u32) -> Option<Self> {
        if pid == 0 {
            return None;
        }
        #[cfg(windows)]
        {
            let pdh = windows_pdh::Query::new()?;
            Some(Self {
                pid,
                last_collect: Instant::now(),
                latest: None,
                pdh,
            })
        }
        #[cfg(not(windows))]
        {
            let _ = pid;
            None
        }
    }
    /// Too-early calls and unavailable/invalid PDH data return None without sleep.
    /// Two successful PDH collections are required. Values are always in 0..=1.
    pub fn sample(&mut self) -> Option<f32> {
        if self.last_collect.elapsed() < MIN_SAMPLE_INTERVAL {
            return None;
        }
        self.last_collect = Instant::now();
        #[cfg(windows)]
        {
            let rows = self.pdh.collect()?;
            let matched = rows
                .iter()
                .any(|(name, _)| engine_identity(name, self.pid).is_some());
            if matched
                && !rows.iter().any(|(name, value)| {
                    engine_identity(name, self.pid).is_some() && value.is_finite()
                })
            {
                return None;
            }
            let snapshot = aggregate(self.pid, rows);
            let value = snapshot.utilization;
            self.latest = Some(snapshot);
            Some(value)
        }
        #[cfg(not(windows))]
        {
            None
        }
    }
    pub fn pid(&self) -> u32 {
        self.pid
    }
    pub fn latest(&self) -> Option<&GpuSnapshot> {
        self.latest.as_ref()
    }
}

#[cfg(windows)]
mod windows_pdh {
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};
    type Handle = *mut c_void;
    const SUCCESS: u32 = 0;
    const MORE_DATA: u32 = 0x8000_07D2;
    const NO_DATA: u32 = 0x8000_07D5;
    const FMT_DOUBLE: u32 = 0x200;
    #[repr(C)]
    union Number {
        double: f64,
    }
    #[repr(C)]
    struct Value {
        status: u32,
        number: Number,
    }
    #[repr(C)]
    struct Item {
        name: *mut u16,
        value: Value,
    }
    #[link(name = "pdh")]
    unsafe extern "system" {
        fn PdhOpenQueryW(source: *const u16, user: usize, query: *mut Handle) -> u32;
        fn PdhAddEnglishCounterW(
            query: Handle,
            path: *const u16,
            user: usize,
            counter: *mut Handle,
        ) -> u32;
        fn PdhCollectQueryData(query: Handle) -> u32;
        fn PdhGetFormattedCounterArrayW(
            counter: Handle,
            format: u32,
            bytes: *mut u32,
            count: *mut u32,
            items: *mut Item,
        ) -> u32;
        fn PdhCloseQuery(query: Handle) -> u32;
    }
    pub(super) struct Query {
        handle: Handle,
        counter: Handle,
        primed: bool,
    }
    impl Query {
        pub(super) fn new() -> Option<Self> {
            let mut handle = null_mut();
            // SAFETY: output handles are valid local pointers; null source means live PDH.
            if unsafe { PdhOpenQueryW(null(), 0, &mut handle) } != SUCCESS {
                return None;
            }
            let mut query = Self {
                handle,
                counter: null_mut(),
                primed: false,
            };
            let path: Vec<u16> = "\\GPU Engine(*)\\Utilization Percentage\0"
                .encode_utf16()
                .collect();
            // English counter names are locale-independent on Chinese Windows.
            if unsafe { PdhAddEnglishCounterW(handle, path.as_ptr(), 0, &mut query.counter) }
                != SUCCESS
            {
                return None;
            }
            let status = unsafe { PdhCollectQueryData(handle) };
            if status != SUCCESS && status != NO_DATA {
                return None;
            }
            query.primed = status == SUCCESS;
            Some(query)
        }
        pub(super) fn collect(&mut self) -> Option<Vec<(String, f64)>> {
            // SAFETY: this live query is uniquely borrowed and closed only in Drop.
            if unsafe { PdhCollectQueryData(self.handle) } != SUCCESS {
                return None;
            }
            if !self.primed {
                self.primed = true;
                return None;
            }
            let (mut bytes, mut count) = (0_u32, 0_u32);
            let result = unsafe {
                PdhGetFormattedCounterArrayW(
                    self.counter,
                    FMT_DOUBLE,
                    &mut bytes,
                    &mut count,
                    null_mut(),
                )
            };
            if result == SUCCESS && count == 0 {
                return Some(Vec::new());
            }
            if result != MORE_DATA || bytes == 0 {
                return None;
            }
            // PDH writes a variable array plus UTF-16 names. u64 storage provides
            // the required 8-byte Item alignment on Windows x64 and ARM64.
            for _ in 0..3 {
                let mut storage = vec![0_u64; (bytes as usize).div_ceil(8)];
                let capacity_bytes = storage.len() * 8;
                bytes = capacity_bytes.try_into().ok()?;
                let result = unsafe {
                    PdhGetFormattedCounterArrayW(
                        self.counter,
                        FMT_DOUBLE,
                        &mut bytes,
                        &mut count,
                        storage.as_mut_ptr().cast(),
                    )
                };
                if result == MORE_DATA {
                    continue;
                }
                if result != SUCCESS
                    || count as usize > capacity_bytes / std::mem::size_of::<Item>()
                {
                    return None;
                }
                let items = unsafe {
                    std::slice::from_raw_parts(storage.as_ptr().cast::<Item>(), count as usize)
                };
                let base = storage.as_ptr() as usize;
                let end = base + capacity_bytes;
                let mut rows = Vec::with_capacity(items.len());
                for item in items {
                    let address = item.name as usize;
                    if address < base || address >= end || address % 2 != 0 {
                        continue;
                    }
                    let words =
                        unsafe { std::slice::from_raw_parts(item.name, (end - address) / 2) };
                    let Some(length) = words.iter().position(|&word| word == 0) else {
                        continue;
                    };
                    let name = String::from_utf16_lossy(&words[..length]);
                    // FMT_DOUBLE selected this union member in the successful call.
                    // Keep invalid instance names so the caller can distinguish
                    // a genuinely idle PID from matching but unprimed counters.
                    let value = if item.value.status <= 1 {
                        unsafe { item.value.number.double }
                    } else {
                        f64::NAN
                    };
                    rows.push((name, value));
                }
                return Some(rows);
            }
            None
        }
    }
    impl Drop for Query {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    PdhCloseQuery(self.handle);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_meter_filters_pid_kind_and_nonfinite() {
        let rows = vec![
            ("pid_12_luid_0x1_0x2_phys_0_eng_0_engtype_3D".into(), 30.0),
            ("pid_123_luid_0x1_0x2_phys_0_eng_0_engtype_3D".into(), 99.0),
            (
                "pid_12_luid_0x1_0x2_phys_0_eng_1_engtype_Compute_0".into(),
                45.0,
            ),
            (
                "pid_12_luid_0x1_0x2_phys_0_eng_2_engtype_Copy".into(),
                100.0,
            ),
            (
                "pid_12_luid_0x1_0x2_phys_0_eng_3_engtype_3D".into(),
                f64::NAN,
            ),
        ];
        let value = aggregate(12, rows);
        assert_eq!(value.three_d, 0.3);
        assert_eq!(value.compute, 0.45);
        assert_eq!(value.utilization, 0.45);
        assert_eq!(value.engines.len(), 2);
    }
    #[test]
    fn gpu_meter_uses_busiest_engine_instead_of_double_counting() {
        let value = aggregate(
            12,
            vec![
                ("pid_12_luid_0x1_0x2_phys_0_eng_0_engtype_3D".into(), 70.0),
                ("pid_12_luid_0x1_0x2_phys_0_eng_0_engtype_3D#1".into(), 65.0),
                (
                    "pid_12_luid_0x1_0x2_phys_0_eng_1_engtype_Compute".into(),
                    60.0,
                ),
            ],
        );
        assert_eq!(value.engines.len(), 2);
        assert_eq!(value.utilization, 0.7);
        assert_eq!(
            aggregate(
                12,
                vec![("pid_12_luid_x_phys_0_eng_0_engtype_3D".into(), 125.0)]
            )
            .utilization,
            1.0
        );
        assert_eq!(
            aggregate(
                12,
                vec![("pid_12_luid_x_phys_0_eng_0_engtype_3D".into(), -1.0)]
            )
            .utilization,
            0.0
        );
    }
}
