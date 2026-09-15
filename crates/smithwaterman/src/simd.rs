//! Integer SIMD vocabulary for the anti-diagonal aligner: vectors of `i16` or `i32` lanes with a
//! per-lane mask type, on scalar / NEON / AVX2 / AVX-512 backends. The same rules as the PairHMM
//! backends apply: every method is `#[inline(always)]` and the x86 types are only instantiated
//! inside functions compiled with their target features.

/// A lane element type. `i16` saturates on addition and `FLOOR` is its saturation floor; `i32`
/// never saturates for the sequence lengths GATK aligns and `FLOOR` is just a sentinel.
pub trait Lane: Copy + PartialEq + Send + Sync + 'static {
    const FLOOR: Self;
    /// Constant added to every stored score. Encoded scores are never positive, so shifting them
    /// towards the top of the lane's range costs nothing and nearly doubles the depth a score can
    /// reach before saturating.
    const HEADROOM: i32;
    /// Converts a constant, saturating to the lane's range.
    fn from_i32(v: i32) -> Self;
    fn to_i32(self) -> i32;
}

impl Lane for i16 {
    const FLOOR: Self = i16::MIN;
    const HEADROOM: i32 = 32_000;
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
}

impl Lane for i32 {
    const FLOOR: Self = i32::MIN / 2;
    const HEADROOM: i32 = 0;
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
}

/// A vector of integer lanes with a per-lane mask type.
pub trait SimdInt: Copy + Send + Sync + 'static {
    type Elem: Lane;
    type Mask: Copy;
    const LANES: usize;
    fn splat(v: Self::Elem) -> Self;
    /// Loads `LANES` values starting at `src[0]`; panics if `src` is shorter.
    fn load(src: &[Self::Elem]) -> Self;
    fn store(self, dst: &mut [Self::Elem]);
    /// Lane-wise sum, saturating for `i16` lanes.
    fn add(self, other: Self) -> Self;
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
        let mut tmp = [Self::Elem::FLOOR; 64];
        self.store(&mut tmp[..Self::LANES]);
        for (d, v) in dst[..Self::LANES].iter_mut().zip(&tmp[..Self::LANES]) {
            *d = v.to_i32() as u8;
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

impl SimdInt for ScalarI32 {
    type Elem = i32;
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

#[derive(Clone, Copy)]
pub struct ScalarI16(pub i16);

impl SimdInt for ScalarI16 {
    type Elem = i16;
    type Mask = bool;
    const LANES: usize = 1;
    #[inline(always)]
    fn splat(v: i16) -> Self {
        ScalarI16(v)
    }
    #[inline(always)]
    fn load(src: &[i16]) -> Self {
        ScalarI16(src[0])
    }
    #[inline(always)]
    fn store(self, dst: &mut [i16]) {
        dst[0] = self.0;
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        ScalarI16(self.0.saturating_add(other.0))
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
    use super::SimdInt;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct I32x4(int32x4_t);

    impl SimdInt for I32x4 {
        type Elem = i32;
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

    #[derive(Clone, Copy)]
    pub struct I16x8(int16x8_t);

    impl SimdInt for I16x8 {
        type Elem = i16;
        type Mask = uint16x8_t;
        const LANES: usize = 8;
        #[inline(always)]
        fn splat(v: i16) -> Self {
            I16x8(unsafe { vdupq_n_s16(v) })
        }
        #[inline(always)]
        fn load(src: &[i16]) -> Self {
            let src = &src[..8];
            I16x8(unsafe { vld1q_s16(src.as_ptr()) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i16]) {
            let dst = &mut dst[..8];
            unsafe { vst1q_s16(dst.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            I16x8(unsafe { vqaddq_s16(self.0, other.0) })
        }
        #[inline(always)]
        fn gt(self, other: Self) -> uint16x8_t {
            unsafe { vcgtq_s16(self.0, other.0) }
        }
        #[inline(always)]
        fn eq(self, other: Self) -> uint16x8_t {
            unsafe { vceqq_s16(self.0, other.0) }
        }
        #[inline(always)]
        fn mask_and(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            unsafe { vandq_u16(a, b) }
        }
        #[inline(always)]
        fn mask_not(a: uint16x8_t) -> uint16x8_t {
            unsafe { vmvnq_u16(a) }
        }
        #[inline(always)]
        fn select(mask: uint16x8_t, a: Self, b: Self) -> Self {
            I16x8(unsafe { vbslq_s16(mask, a.0, b.0) })
        }
        #[inline(always)]
        fn store_low_bytes(self, dst: &mut [u8]) {
            let dst = &mut dst[..8];
            unsafe { vst1_s8(dst.as_mut_ptr() as *mut i8, vmovn_s16(self.0)) }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::SimdInt;
    use core::arch::x86_64::*;

    // SAFETY (this module): only instantiated inside functions compiled with the matching
    // target features after runtime detection; loads and stores touch exactly `LANES` elements
    // of slices checked to be long enough.

    #[derive(Clone, Copy)]
    pub struct I32x8(__m256i);

    impl SimdInt for I32x8 {
        type Elem = i32;
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
    pub struct I16x16(__m256i);

    impl SimdInt for I16x16 {
        type Elem = i16;
        type Mask = __m256i;
        const LANES: usize = 16;
        #[inline(always)]
        fn splat(v: i16) -> Self {
            I16x16(unsafe { _mm256_set1_epi16(v) })
        }
        #[inline(always)]
        fn load(src: &[i16]) -> Self {
            let src = &src[..16];
            I16x16(unsafe { _mm256_loadu_si256(src.as_ptr() as *const __m256i) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i16]) {
            let dst = &mut dst[..16];
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            I16x16(unsafe { _mm256_adds_epi16(self.0, other.0) })
        }
        #[inline(always)]
        fn gt(self, other: Self) -> __m256i {
            unsafe { _mm256_cmpgt_epi16(self.0, other.0) }
        }
        #[inline(always)]
        fn eq(self, other: Self) -> __m256i {
            unsafe { _mm256_cmpeq_epi16(self.0, other.0) }
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
            // Mask lanes are all-ones or all-zeros, so a byte-granular blend selects whole lanes.
            I16x16(unsafe { _mm256_blendv_epi8(b.0, a.0, mask) })
        }
    }

    #[derive(Clone, Copy)]
    pub struct I32x16(__m512i);

    impl SimdInt for I32x16 {
        type Elem = i32;
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
        #[inline(always)]
        fn store_low_bytes(self, dst: &mut [u8]) {
            let dst = &mut dst[..16];
            unsafe {
                _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, _mm512_cvtepi32_epi8(self.0))
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct I16x32(__m512i);

    impl SimdInt for I16x32 {
        type Elem = i16;
        type Mask = __mmask32;
        const LANES: usize = 32;
        #[inline(always)]
        fn splat(v: i16) -> Self {
            I16x32(unsafe { _mm512_set1_epi16(v) })
        }
        #[inline(always)]
        fn load(src: &[i16]) -> Self {
            let src = &src[..32];
            I16x32(unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) })
        }
        #[inline(always)]
        fn store(self, dst: &mut [i16]) {
            let dst = &mut dst[..32];
            unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, self.0) }
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            I16x32(unsafe { _mm512_adds_epi16(self.0, other.0) })
        }
        #[inline(always)]
        fn gt(self, other: Self) -> __mmask32 {
            unsafe { _mm512_cmpgt_epi16_mask(self.0, other.0) }
        }
        #[inline(always)]
        fn eq(self, other: Self) -> __mmask32 {
            unsafe { _mm512_cmpeq_epi16_mask(self.0, other.0) }
        }
        #[inline(always)]
        fn mask_and(a: __mmask32, b: __mmask32) -> __mmask32 {
            a & b
        }
        #[inline(always)]
        fn mask_not(a: __mmask32) -> __mmask32 {
            !a
        }
        #[inline(always)]
        fn select(mask: __mmask32, a: Self, b: Self) -> Self {
            I16x32(unsafe { _mm512_mask_blend_epi16(mask, b.0, a.0) })
        }
        #[inline(always)]
        fn store_low_bytes(self, dst: &mut [u8]) {
            let dst = &mut dst[..32];
            unsafe {
                _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, _mm512_cvtepi16_epi8(self.0))
            }
        }
    }
}
