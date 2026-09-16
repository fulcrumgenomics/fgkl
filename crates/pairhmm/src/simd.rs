//! The small SIMD vocabulary the kernel is written against, with a scalar implementation, a
//! lane-multiplying wrapper, and NEON / AVX2 / AVX-512 backends.
//!
//! Every backend method is `#[inline(always)]` so that it is compiled inside the
//! `#[target_feature]` entry point that instantiates it; the x86 backends must never be used from
//! code compiled without their features.

use std::ops::{Add, Deref, DerefMut, Mul};

/// Alignment of every kernel buffer: a full AVX-512 vector, so no lane-group load or store
/// crosses a cache line.
pub const BUFFER_ALIGN: usize = 64;

/// A growable buffer whose slice starts on a [`BUFFER_ALIGN`]-byte boundary. Built on `Vec`
/// with slack at the front, so it needs no unsafe code.
#[derive(Clone, Debug, Default)]
pub struct AlignedVec<T> {
    buf: Vec<T>,
    offset: usize,
    len: usize,
}

impl<T: Copy + Default> AlignedVec<T> {
    pub fn new() -> Self {
        AlignedVec { buf: Vec::new(), offset: 0, len: 0 }
    }

    /// Resizes to `len` elements and fills them all with `value`.
    pub fn resize(&mut self, len: usize, value: T) {
        self.resize_no_fill(len);
        self.fill(value);
    }

    /// Resizes to `len` elements without clearing them: the contents are whatever an earlier
    /// call left there, or zero where the buffer had to grow. Scratch memory that is always
    /// written before it is read uses this to avoid clearing megabytes per batch.
    pub fn resize_no_fill(&mut self, len: usize) {
        let slack = BUFFER_ALIGN / std::mem::size_of::<T>().max(1);
        if self.buf.len() < len + slack {
            self.buf.clear();
            self.buf.resize(len + slack, T::default());
        }
        let addr = self.buf.as_ptr() as usize;
        self.offset =
            (BUFFER_ALIGN - addr % BUFFER_ALIGN) % BUFFER_ALIGN / std::mem::size_of::<T>().max(1);
        self.len = len;
    }

    pub fn fill(&mut self, value: T) {
        self.buf[self.offset..self.offset + self.len].fill(value);
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.buf[self.offset..self.offset + self.len]
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf[self.offset..self.offset + self.len]
    }
}

/// Element type of a kernel: `f32` or `f64`, with the scaling constants GATK and GKL use for it.
pub trait Float:
    Copy
    + Default
    + PartialOrd
    + Send
    + Sync
    + std::fmt::Debug
    + Add<Output = Self>
    + Mul<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    /// Initial deletion probability, chosen to keep the linear-space DP inside the exponent range.
    const INITIAL_CONSTANT: Self;
    /// Raw results below this are treated as having lost precision and are recomputed in `f64`.
    const MIN_ACCEPTED: Option<f64>;
    fn log10_initial_constant() -> f64;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    /// The larger of two values; both are always finite and non-negative in the kernels.
    fn max(self, other: Self) -> Self;
}

impl Float for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    // 2^120, as in GKL's Context<float>.
    const INITIAL_CONSTANT: f32 = f32::from_bits(0x7B80_0000);
    const MIN_ACCEPTED: Option<f64> = Some(1e-28);
    #[inline(always)]
    fn log10_initial_constant() -> f64 {
        120.0 * std::f64::consts::LOG10_2
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn max(self, other: f32) -> f32 {
        f32::max(self, other)
    }
}

impl Float for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    // 2^1020, as in GATK's LoglessPairHMM.
    const INITIAL_CONSTANT: f64 = f64::from_bits(0x7FB0_0000_0000_0000);
    const MIN_ACCEPTED: Option<f64> = None;
    #[inline(always)]
    fn log10_initial_constant() -> f64 {
        1020.0 * std::f64::consts::LOG10_2
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn max(self, other: f64) -> f64 {
        f64::max(self, other)
    }
}

/// A fixed-width vector of [`Float`]s.
pub trait Simd: Copy + Send + Sync + 'static {
    type Elem: Float;
    const LANES: usize;
    fn splat(v: Self::Elem) -> Self;
    /// Loads `LANES` elements starting at `src`.
    ///
    /// # Safety
    /// `src` must point to at least `LANES` readable elements.
    unsafe fn load_ptr(src: *const Self::Elem) -> Self;
    /// Stores `LANES` elements starting at `dst`.
    ///
    /// # Safety
    /// `dst` must point to at least `LANES` writable elements.
    unsafe fn store_ptr(self, dst: *mut Self::Elem);
    /// Loads the first `LANES` elements of `src`; panics if `src` is shorter.
    #[inline(always)]
    fn load(src: &[Self::Elem]) -> Self {
        let src = &src[..Self::LANES];
        // SAFETY: the slice was just checked to hold `LANES` elements.
        unsafe { Self::load_ptr(src.as_ptr()) }
    }
    /// Stores into the first `LANES` elements of `dst`; panics if `dst` is shorter.
    #[inline(always)]
    fn store(self, dst: &mut [Self::Elem]) {
        let dst = &mut dst[..Self::LANES];
        // SAFETY: the slice was just checked to hold `LANES` elements.
        unsafe { self.store_ptr(dst.as_mut_ptr()) }
    }
    fn add(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    /// `self * mul + add`, fused where the hardware supports it.
    fn mul_add(self, mul: Self, add: Self) -> Self;
    /// Lane-wise maximum. Kernel values are finite and non-negative, so the NaN and signed-zero
    /// conventions of the backends never matter.
    fn max(self, other: Self) -> Self;
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(Self::Elem::ZERO)
    }
}

/// Runs `f` with x86 flush-to-zero and denormals-are-zero set, restoring the caller's MXCSR
/// afterwards. Single-precision DP values routinely fall into the subnormal range, where x86
/// arithmetic takes microcode assists costing over a hundred cycles per operation; flushing them
/// to zero costs nothing numerically because any pair whose result is that small is recomputed
/// in double precision anyway. GKL enables the same mode.
#[cfg(target_arch = "x86_64")]
pub(crate) fn with_flush_to_zero<R>(f: impl FnOnce() -> R) -> R {
    /// Restores the saved MXCSR when dropped, so a panic unwinding out of `f` (caught by the JNI
    /// layer) does not leave the JVM thread flushing subnormals.
    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: ldmxcsr only writes the MXCSR register from a valid u32.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &self.0, options(nostack, preserves_flags))
            };
        }
    }
    const FTZ_DAZ: u32 = 0x8040;
    let mut saved: u32 = 0;
    // SAFETY: stmxcsr/ldmxcsr only read and write the MXCSR register through a valid u32.
    unsafe {
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut saved, options(nostack, preserves_flags))
    };
    let _restore = Restore(saved);
    let flushed = saved | FTZ_DAZ;
    unsafe {
        core::arch::asm!("ldmxcsr [{}]", in(reg) &flushed, options(nostack, preserves_flags))
    };
    f()
}

/// Subnormals cost nothing extra on aarch64, so nothing to do.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn with_flush_to_zero<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// One lane, for CPUs without a supported vector unit and for checking the vector backends.
#[derive(Clone, Copy)]
pub struct Scalar<T>(pub T);

impl<T: Float> Simd for Scalar<T> {
    type Elem = T;
    const LANES: usize = 1;
    #[inline(always)]
    fn splat(v: T) -> Self {
        Scalar(v)
    }
    #[inline(always)]
    unsafe fn load_ptr(src: *const T) -> Self {
        // SAFETY: the caller guarantees one readable element.
        Scalar(unsafe { *src })
    }
    #[inline(always)]
    unsafe fn store_ptr(self, dst: *mut T) {
        // SAFETY: the caller guarantees one writable element.
        unsafe { *dst = self.0 }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Scalar(self.0 + other.0)
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Scalar(self.0 * other.0)
    }
    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        Scalar(self.0 * mul.0 + add.0)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Scalar(self.0.max(other.0))
    }
}

/// `K` vectors treated as one, giving the kernel `K` independent dependency chains per cell so
/// that FMA latency is hidden on cores with several vector pipes.
#[derive(Clone, Copy)]
pub struct Wide<S, const K: usize>(pub [S; K]);

impl<S: Simd, const K: usize> Simd for Wide<S, K> {
    type Elem = S::Elem;
    const LANES: usize = S::LANES * K;
    #[inline(always)]
    fn splat(v: S::Elem) -> Self {
        Wide([S::splat(v); K])
    }
    #[inline(always)]
    unsafe fn load_ptr(src: *const S::Elem) -> Self {
        // SAFETY: the caller guarantees `K * S::LANES` elements; group `i` starts at `i * S::LANES`.
        Wide(std::array::from_fn(|i| unsafe { S::load_ptr(src.add(i * S::LANES)) }))
    }
    #[inline(always)]
    unsafe fn store_ptr(self, dst: *mut S::Elem) {
        for (i, v) in self.0.into_iter().enumerate() {
            // SAFETY: as for `load_ptr`.
            unsafe { v.store_ptr(dst.add(i * S::LANES)) }
        }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Wide(std::array::from_fn(|i| self.0[i].add(other.0[i])))
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Wide(std::array::from_fn(|i| self.0[i].mul(other.0[i])))
    }
    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        Wide(std::array::from_fn(|i| self.0[i].mul_add(mul.0[i], add.0[i])))
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Wide(std::array::from_fn(|i| self.0[i].max(other.0[i])))
    }
}

pub type ScalarF32 = Wide<Scalar<f32>, 4>;
pub type ScalarF64 = Wide<Scalar<f64>, 4>;

// SAFETY (whole module): NEON is part of the aarch64 baseline, so every intrinsic call below is
// sound in any aarch64 build; `load_ptr` and `store_ptr` touch exactly the `LANES` elements their
// callers guarantee.
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::{Simd, Wide};
    use core::arch::aarch64::*;

    pub type NeonF32 = Wide<F32x4, 2>;
    pub type NeonF64 = Wide<F64x2, 2>;

    #[derive(Clone, Copy)]
    pub struct F32x4(float32x4_t);

    impl Simd for F32x4 {
        type Elem = f32;
        const LANES: usize = 4;
        #[inline(always)]
        fn splat(v: f32) -> Self {
            F32x4(unsafe { vdupq_n_f32(v) })
        }
        #[inline(always)]
        unsafe fn load_ptr(src: *const f32) -> Self {
            F32x4(unsafe { vld1q_f32(src) })
        }
        #[inline(always)]
        unsafe fn store_ptr(self, dst: *mut f32) {
            unsafe { vst1q_f32(dst, self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            F32x4(unsafe { vaddq_f32(self.0, other.0) })
        }
        #[inline(always)]
        fn mul(self, other: Self) -> Self {
            F32x4(unsafe { vmulq_f32(self.0, other.0) })
        }
        #[inline(always)]
        fn mul_add(self, mul: Self, add: Self) -> Self {
            F32x4(unsafe { vfmaq_f32(add.0, self.0, mul.0) })
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            F32x4(unsafe { vmaxq_f32(self.0, other.0) })
        }
    }

    #[derive(Clone, Copy)]
    pub struct F64x2(float64x2_t);

    impl Simd for F64x2 {
        type Elem = f64;
        const LANES: usize = 2;
        #[inline(always)]
        fn splat(v: f64) -> Self {
            F64x2(unsafe { vdupq_n_f64(v) })
        }
        #[inline(always)]
        unsafe fn load_ptr(src: *const f64) -> Self {
            F64x2(unsafe { vld1q_f64(src) })
        }
        #[inline(always)]
        unsafe fn store_ptr(self, dst: *mut f64) {
            unsafe { vst1q_f64(dst, self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            F64x2(unsafe { vaddq_f64(self.0, other.0) })
        }
        #[inline(always)]
        fn mul(self, other: Self) -> Self {
            F64x2(unsafe { vmulq_f64(self.0, other.0) })
        }
        #[inline(always)]
        fn mul_add(self, mul: Self, add: Self) -> Self {
            F64x2(unsafe { vfmaq_f64(add.0, self.0, mul.0) })
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            F64x2(unsafe { vmaxq_f64(self.0, other.0) })
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::{Simd, Wide};
    use core::arch::x86_64::*;

    // AVX2 has 16 vector registers, so a second lane group spills and costs about 15%;
    // AVX-512's 32 registers hold two groups, which hides FMA latency for a 6% gain.
    pub type Avx2F32 = F32x8;
    pub type Avx2F64 = F64x4;
    pub type Avx512F32 = Wide<F32x16, 2>;
    pub type Avx512F64 = Wide<F64x8, 2>;
    /// Single lane group for the reads a region has left over after its full 32-lane batches.
    pub type Avx512F32Narrow = F32x16;
    pub type Avx512F64Narrow = F64x8;

    macro_rules! x86_vector {
        ($name:ident, $reg:ty, $elem:ty, $lanes:expr, $set1:ident, $loadu:ident, $storeu:ident, $add:ident, $mul:ident, $fmadd:ident, $max:ident) => {
            #[derive(Clone, Copy)]
            pub struct $name($reg);

            // SAFETY (all intrinsics below): the backend is only instantiated inside functions
            // compiled with the matching `#[target_feature]`, after runtime detection; `load_ptr`
            // and `store_ptr` touch exactly the `LANES` elements their callers guarantee.
            impl Simd for $name {
                type Elem = $elem;
                const LANES: usize = $lanes;
                #[inline(always)]
                fn splat(v: $elem) -> Self {
                    $name(unsafe { $set1(v) })
                }
                #[inline(always)]
                unsafe fn load_ptr(src: *const $elem) -> Self {
                    $name(unsafe { $loadu(src) })
                }
                #[inline(always)]
                unsafe fn store_ptr(self, dst: *mut $elem) {
                    unsafe { $storeu(dst, self.0) }
                }
                #[inline(always)]
                fn add(self, other: Self) -> Self {
                    $name(unsafe { $add(self.0, other.0) })
                }
                #[inline(always)]
                fn mul(self, other: Self) -> Self {
                    $name(unsafe { $mul(self.0, other.0) })
                }
                #[inline(always)]
                fn mul_add(self, mul: Self, add: Self) -> Self {
                    $name(unsafe { $fmadd(self.0, mul.0, add.0) })
                }
                #[inline(always)]
                fn max(self, other: Self) -> Self {
                    $name(unsafe { $max(self.0, other.0) })
                }
            }
        };
    }

    x86_vector!(
        F32x8,
        __m256,
        f32,
        8,
        _mm256_set1_ps,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        _mm256_fmadd_ps,
        _mm256_max_ps
    );
    x86_vector!(
        F64x4,
        __m256d,
        f64,
        4,
        _mm256_set1_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_mul_pd,
        _mm256_fmadd_pd,
        _mm256_max_pd
    );
    x86_vector!(
        F32x16,
        __m512,
        f32,
        16,
        _mm512_set1_ps,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        _mm512_fmadd_ps,
        _mm512_max_ps
    );
    x86_vector!(
        F64x8,
        __m512d,
        f64,
        8,
        _mm512_set1_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        _mm512_mul_pd,
        _mm512_fmadd_pd,
        _mm512_max_pd
    );
}
