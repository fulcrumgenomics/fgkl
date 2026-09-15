//! Integer SIMD vocabulary for the anti-diagonal aligner: `i32` lanes, lane masks, and the
//! scalar / NEON / AVX2 / AVX-512 backends. The same rules as the PairHMM backends apply: every
//! method is `#[inline(always)]` and the x86 types are only instantiated inside functions compiled
//! with their target features.

/// A vector of `i32` lanes with a per-lane mask type.
pub trait SimdI32: Copy + Send + Sync + 'static {
    type Mask: Copy;
    const LANES: usize;
    fn splat(v: i32) -> Self;
    /// Loads `LANES` values starting at `src[0]`; panics if `src` is shorter.
    fn load(src: &[i32]) -> Self;
    fn store(self, dst: &mut [i32]);
    fn add(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    /// Lanes where `self > other`.
    fn gt(self, other: Self) -> Self::Mask;
    /// Lanes where `self == other`.
    fn eq(self, other: Self) -> Self::Mask;
    fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    fn mask_not(a: Self::Mask) -> Self::Mask;
    /// `a` where the mask is set, else `b`.
    fn select(mask: Self::Mask, a: Self, b: Self) -> Self;
    /// Stores the low byte of every lane to `dst[0..LANES]`.
    #[inline(always)]
    fn store_low_bytes(self, dst: &mut [u8]) {
        let mut tmp = [0i32; 64];
        self.store(&mut tmp[..Self::LANES]);
        for (d, v) in dst[..Self::LANES].iter_mut().zip(&tmp[..Self::LANES]) {
            *d = *v as u8;
        }
    }
    /// Lanes where `self >= other`.
    #[inline(always)]
    fn ge(self, other: Self) -> Self::Mask {
        Self::mask_not(other.gt(self))
    }
}

#[derive(Clone, Copy)]
pub struct ScalarI32(pub i32);

impl SimdI32 for ScalarI32 {
    type Mask = bool;
    const LANES: usize = 1;
    #[inline(always)]
    fn splat(v: i32) -> Self {
        ScalarI32(v)
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        ScalarI32(src[0])
    }
    #[inline(always)]
    fn store(self, dst: &mut [i32]) {
        dst[0] = self.0;
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        ScalarI32(self.0.wrapping_add(other.0))
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        ScalarI32(self.0.max(other.0))
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self.0 > other.0
    }
    #[inline(always)]
    fn eq(self, other: Self) -> bool {
        self.0 == other.0
    }
    #[inline(always)]
    fn mask_and(a: bool, b: bool) -> bool {
        a && b
    }
    #[inline(always)]
    fn mask_not(a: bool) -> bool {
        !a
    }
    #[inline(always)]
    fn select(mask: bool, a: Self, b: Self) -> Self {
        if mask { a } else { b }
    }
}

// SAFETY (whole module): NEON is part of the aarch64 baseline; loads and stores touch exactly
// `LANES` elements of slices checked to be long enough.
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::SimdI32;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct I32x4(int32x4_t);

    impl SimdI32 for I32x4 {
        type Mask = uint32x4_t;
        const LANES: usize = 4;
        #[inline(always)]
        fn splat(v: i32) -> Self {
            I32x4(unsafe { vdupq_n_s32(v) })
        }
        #[inline(always)]
        fn load(src: &[i32]) -> Self {
            let src = &src[..4];
            I32x4(unsafe { vld1q_s32(src.as_ptr()) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i32]) {
            let dst = &mut dst[..4];
            unsafe { vst1q_s32(dst.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            I32x4(unsafe { vaddq_s32(self.0, other.0) })
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            I32x4(unsafe { vmaxq_s32(self.0, other.0) })
        }
        #[inline(always)]
        fn gt(self, other: Self) -> uint32x4_t {
            unsafe { vcgtq_s32(self.0, other.0) }
        }
        #[inline(always)]
        fn eq(self, other: Self) -> uint32x4_t {
            unsafe { vceqq_s32(self.0, other.0) }
        }
        #[inline(always)]
        fn mask_and(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
            unsafe { vandq_u32(a, b) }
        }
        #[inline(always)]
        fn mask_not(a: uint32x4_t) -> uint32x4_t {
            unsafe { vmvnq_u32(a) }
        }
        #[inline(always)]
        fn select(mask: uint32x4_t, a: Self, b: Self) -> Self {
            I32x4(unsafe { vbslq_s32(mask, a.0, b.0) })
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::SimdI32;
    use core::arch::x86_64::*;

    // SAFETY (this module): only instantiated inside functions compiled with the matching
    // target features after runtime detection; loads and stores touch exactly `LANES` elements
    // of slices checked to be long enough.

    #[derive(Clone, Copy)]
    pub struct I32x8(__m256i);

    impl SimdI32 for I32x8 {
        type Mask = __m256i;
        const LANES: usize = 8;
        #[inline(always)]
        fn splat(v: i32) -> Self {
            I32x8(unsafe { _mm256_set1_epi32(v) })
        }
        #[inline(always)]
        fn load(src: &[i32]) -> Self {
            let src = &src[..8];
            I32x8(unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i32]) {
            let dst = &mut dst[..8];
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            I32x8(unsafe { _mm256_add_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            I32x8(unsafe { _mm256_max_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn gt(self, other: Self) -> __m256i {
            unsafe { _mm256_cmpgt_epi32(self.0, other.0) }
        }
        #[inline(always)]
        fn eq(self, other: Self) -> __m256i {
            unsafe { _mm256_cmpeq_epi32(self.0, other.0) }
        }
        #[inline(always)]
        fn mask_and(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_and_si256(a, b) }
        }
        #[inline(always)]
        fn mask_not(a: __m256i) -> __m256i {
            unsafe { _mm256_xor_si256(a, _mm256_set1_epi32(-1)) }
        }
        #[inline(always)]
        fn select(mask: __m256i, a: Self, b: Self) -> Self {
            I32x8(unsafe { _mm256_blendv_epi8(b.0, a.0, mask) })
        }
    }

    #[derive(Clone, Copy)]
    pub struct I32x16(__m512i);

    impl SimdI32 for I32x16 {
        type Mask = __mmask16;
        const LANES: usize = 16;
        #[inline(always)]
        fn splat(v: i32) -> Self {
            I32x16(unsafe { _mm512_set1_epi32(v) })
        }
        #[inline(always)]
        fn load(src: &[i32]) -> Self {
            let src = &src[..16];
            I32x16(unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i32]) {
            let dst = &mut dst[..16];
            unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            I32x16(unsafe { _mm512_add_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn max(self, other: Self) -> Self {
            I32x16(unsafe { _mm512_max_epi32(self.0, other.0) })
        }
        #[inline(always)]
        fn gt(self, other: Self) -> __mmask16 {
            unsafe { _mm512_cmpgt_epi32_mask(self.0, other.0) }
        }
        #[inline(always)]
        fn eq(self, other: Self) -> __mmask16 {
            unsafe { _mm512_cmpeq_epi32_mask(self.0, other.0) }
        }
        #[inline(always)]
        fn mask_and(a: __mmask16, b: __mmask16) -> __mmask16 {
            a & b
        }
        #[inline(always)]
        fn mask_not(a: __mmask16) -> __mmask16 {
            !a
        }
        #[inline(always)]
        fn select(mask: __mmask16, a: Self, b: Self) -> Self {
            I32x16(unsafe { _mm512_mask_blend_epi32(mask, b.0, a.0) })
        }
    }
}
