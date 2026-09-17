//! JNI entry points for `com.fulcrumgenomics.fgkl.pairhmm.FgklPairHmm`,
//! `com.fulcrumgenomics.fgkl.pdhmm.FgklPdHmm` and
//! `com.fulcrumgenomics.fgkl.smithwaterman.FgklSmithWaterman`.
//!
//! Inputs arrive as primitive arrays and are copied into Rust memory before any computation, so
//! no JNI critical section is held while a kernel runs and the JVM is free to garbage collect.
//! Results are copied back with a single region write. Every entry point runs its body through
//! `EnvUnowned::with_env`, which catches panics, and the `Throw` policy turns a failure into
//! the Java exception it names and a panic into a `RuntimeException` instead of an abort.

use std::cell::RefCell;

use fgkl_pairhmm::{Backend, Config, PairHmm, PdHaplotype, PdPairHmm, Precision, ReadRef};
use fgkl_smithwaterman::{Aligner, OverhangStrategy, SwParameters};
use jni::errors::ErrorPolicy;
use jni::objects::{JByteArray, JClass, JDoubleArray, JIntArray, JObject};
use jni::strings::JNIString;
use jni::sys::{jboolean, jint};
use jni::{Env, EnvUnowned};

const ILLEGAL_ARGUMENT: &str = "java/lang/IllegalArgumentException";
const RUNTIME_EXCEPTION: &str = "java/lang/RuntimeException";

thread_local! {
    /// One aligner per thread so its scratch buffers are reused across calls.
    static SW_ALIGNER: RefCell<Aligner> = RefCell::new(Aligner::new());
}

/// A Java exception to raise: its class and message.
#[derive(Debug)]
struct Failure {
    class: &'static str,
    message: String,
}

/// How a native method's outcome reaches Java: a [`Failure`] becomes the exception it names and
/// a panic a `RuntimeException`, and the method returns the type's default (`()` or `null`). An
/// exception already pending takes precedence.
struct Throw;

impl<T: Default> ErrorPolicy<T, Failure> for Throw {
    type Captures<'unowned_env_local: 'native_method, 'native_method> = ();

    fn on_error<'unowned_env_local: 'native_method, 'native_method>(
        env: &mut Env<'unowned_env_local>,
        _captures: &mut (),
        err: Failure,
    ) -> jni::errors::Result<T> {
        if !env.exception_check() {
            // Throwing reports the now-pending exception as an error; it is the intended result.
            let _ = env.throw_new(JNIString::from(err.class), JNIString::from(err.message));
        }
        Ok(T::default())
    }

    fn on_panic<'unowned_env_local: 'native_method, 'native_method>(
        env: &mut Env<'unowned_env_local>,
        _captures: &mut (),
        payload: Box<dyn std::any::Any + Send + 'static>,
    ) -> jni::errors::Result<T> {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_string());
        if !env.exception_check() {
            let _ = env.throw_new(
                JNIString::from(RUNTIME_EXCEPTION),
                JNIString::from(format!("fgkl native code panicked: {message}")),
            );
        }
        Ok(T::default())
    }
}

impl From<jni::errors::Error> for Failure {
    fn from(e: jni::errors::Error) -> Self {
        Failure { class: RUNTIME_EXCEPTION, message: format!("JNI error: {e}") }
    }
}

impl From<fgkl_pairhmm::Error> for Failure {
    fn from(e: fgkl_pairhmm::Error) -> Self {
        Failure { class: ILLEGAL_ARGUMENT, message: e.to_string() }
    }
}

impl From<fgkl_smithwaterman::Error> for Failure {
    fn from(e: fgkl_smithwaterman::Error) -> Self {
        Failure { class: ILLEGAL_ARGUMENT, message: e.to_string() }
    }
}

/// The five read arrays and their offset table, copied out of the JVM.
struct ReadBatch {
    bases: Vec<u8>,
    quals: Vec<u8>,
    ins_gop: Vec<u8>,
    del_gop: Vec<u8>,
    gcp: Vec<u8>,
    offsets: Vec<i32>,
}

impl ReadBatch {
    fn fetch(
        env: &mut Env<'_>,
        bases: &JByteArray<'_>,
        quals: &JByteArray<'_>,
        ins_gop: &JByteArray<'_>,
        del_gop: &JByteArray<'_>,
        gcp: &JByteArray<'_>,
        offsets: &JIntArray<'_>,
    ) -> Result<Self, Failure> {
        let batch = ReadBatch {
            bases: env.convert_byte_array(bases)?,
            quals: env.convert_byte_array(quals)?,
            ins_gop: env.convert_byte_array(ins_gop)?,
            del_gop: env.convert_byte_array(del_gop)?,
            gcp: env.convert_byte_array(gcp)?,
            offsets: int_array(env, offsets)?,
        };
        let n = batch.bases.len();
        if [batch.quals.len(), batch.ins_gop.len(), batch.del_gop.len(), batch.gcp.len()]
            .iter()
            .any(|&l| l != n)
        {
            return Err(Failure {
                class: ILLEGAL_ARGUMENT,
                message: "read bases, qualities and penalties differ in total length".to_string(),
            });
        }
        Ok(batch)
    }

    fn reads(&self) -> Result<Vec<ReadRef<'_>>, Failure> {
        Ok(split(&self.offsets, self.bases.len())?
            .into_iter()
            .map(|(lo, hi)| ReadRef {
                bases: &self.bases[lo..hi],
                quals: &self.quals[lo..hi],
                ins_gop: &self.ins_gop[lo..hi],
                del_gop: &self.del_gop[lo..hi],
                gcp: &self.gcp[lo..hi],
            })
            .collect())
    }
}

/// `FgklPairHmm.backendNative()`: the name of the kernel backend selected for this CPU.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fulcrumgenomics_fgkl_pairhmm_FgklPairHmm_backendNative<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
) -> JObject<'local> {
    unowned_env.with_env(|env| backend_name(env)).resolve::<Throw>()
}

/// `FgklPairHmm.computeNative(...)`; see the Java declaration for the array layout.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_fulcrumgenomics_fgkl_pairhmm_FgklPairHmm_computeNative<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    bases: JByteArray<'local>,
    quals: JByteArray<'local>,
    ins_gop: JByteArray<'local>,
    del_gop: JByteArray<'local>,
    gcp: JByteArray<'local>,
    read_offsets: JIntArray<'local>,
    hap_bases: JByteArray<'local>,
    hap_offsets: JIntArray<'local>,
    double_precision: jboolean,
    likelihoods: JDoubleArray<'local>,
) {
    unowned_env
        .with_env(|env| -> Result<(), Failure> {
            let batch =
                ReadBatch::fetch(env, &bases, &quals, &ins_gop, &del_gop, &gcp, &read_offsets)?;
            let reads = batch.reads()?;
            let hap_bases = env.convert_byte_array(&hap_bases)?;
            let hap_offsets = int_array(env, &hap_offsets)?;
            let haplotypes: Vec<&[u8]> = split(&hap_offsets, hap_bases.len())?
                .into_iter()
                .map(|(lo, hi)| &hap_bases[lo..hi])
                .collect();
            let expected = reads.len() * haplotypes.len();
            check_output(env, &likelihoods, expected)?;
            let hmm = PairHmm::new(&config(double_precision))?;
            let mut out = vec![0.0f64; expected];
            hmm.compute_log10_likelihoods(&reads, &haplotypes, &mut out)?;
            likelihoods.set_region(env, 0, &out)?;
            Ok(())
        })
        .resolve::<Throw>()
}

/// `FgklPdHmm.backendNative()`: the name of the kernel backend selected for this CPU.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_fulcrumgenomics_fgkl_pdhmm_FgklPdHmm_backendNative<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
) -> JObject<'local> {
    unowned_env.with_env(|env| backend_name(env)).resolve::<Throw>()
}

/// `FgklPdHmm.computeNative(...)`; see the Java declaration for the array layout.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_fulcrumgenomics_fgkl_pdhmm_FgklPdHmm_computeNative<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    bases: JByteArray<'local>,
    quals: JByteArray<'local>,
    ins_gop: JByteArray<'local>,
    del_gop: JByteArray<'local>,
    gcp: JByteArray<'local>,
    read_offsets: JIntArray<'local>,
    hap_bases: JByteArray<'local>,
    hap_flags: JByteArray<'local>,
    hap_offsets: JIntArray<'local>,
    double_precision: jboolean,
    likelihoods: JDoubleArray<'local>,
) {
    unowned_env
        .with_env(|env| -> Result<(), Failure> {
            let batch =
                ReadBatch::fetch(env, &bases, &quals, &ins_gop, &del_gop, &gcp, &read_offsets)?;
            let reads = batch.reads()?;
            let hap_bases = env.convert_byte_array(&hap_bases)?;
            let hap_flags = env.convert_byte_array(&hap_flags)?;
            let hap_offsets = int_array(env, &hap_offsets)?;
            if hap_flags.len() != hap_bases.len() {
                return Err(Failure {
                    class: ILLEGAL_ARGUMENT,
                    message: "haplotype bases and PD flags differ in total length".to_string(),
                });
            }
            let haplotypes: Vec<PdHaplotype<'_>> = split(&hap_offsets, hap_bases.len())?
                .into_iter()
                .map(|(lo, hi)| PdHaplotype {
                    bases: &hap_bases[lo..hi],
                    flags: &hap_flags[lo..hi],
                })
                .collect();
            let expected = reads.len() * haplotypes.len();
            check_output(env, &likelihoods, expected)?;
            let hmm = PdPairHmm::new(&config(double_precision))?;
            let mut out = vec![0.0f64; expected];
            hmm.compute_log10_likelihoods(&reads, &haplotypes, &mut out)?;
            likelihoods.set_region(env, 0, &out)?;
            Ok(())
        })
        .resolve::<Throw>()
}

/// `FgklSmithWaterman.alignNative(...)`: returns the CIGAR string and writes the alignment offset
/// into `offset[0]`. `strategy` is the code `FgklSmithWaterman` assigns to each
/// `SWOverhangStrategy`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_fulcrumgenomics_fgkl_smithwaterman_FgklSmithWaterman_alignNative<
    'local,
>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    reference: JByteArray<'local>,
    alternate: JByteArray<'local>,
    match_value: jint,
    mismatch_penalty: jint,
    gap_open: jint,
    gap_extend: jint,
    strategy: jint,
    offset: JIntArray<'local>,
) -> JObject<'local> {
    unowned_env
        .with_env(|env| -> Result<JObject<'local>, Failure> {
            let strategy = match strategy {
                0 => OverhangStrategy::SoftClip,
                1 => OverhangStrategy::Indel,
                2 => OverhangStrategy::LeadingIndel,
                3 => OverhangStrategy::Ignore,
                other => {
                    return Err(Failure {
                        class: ILLEGAL_ARGUMENT,
                        message: format!("unknown overhang strategy code {other}"),
                    });
                }
            };
            let params = SwParameters::new(match_value, mismatch_penalty, gap_open, gap_extend);
            let reference = env.convert_byte_array(&reference)?;
            let alternate = env.convert_byte_array(&alternate)?;
            let alignment = SW_ALIGNER
                .with(|a| a.borrow_mut().align(&reference, &alternate, &params, strategy))?;
            offset.set_region(env, 0, &[alignment.offset])?;
            Ok(env.new_string(alignment.cigar_string())?.into())
        })
        .resolve::<Throw>()
}

fn backend_name<'local>(env: &mut Env<'local>) -> Result<JObject<'local>, Failure> {
    Ok(env.new_string(Backend::detect().name())?.into())
}

fn config(double_precision: jboolean) -> Config {
    let precision = if double_precision { Precision::Double } else { Precision::Float };
    Config { precision, backend: None, double_fallback: true }
}

/// Checks the Java output array holds exactly `expected` elements.
fn check_output(
    env: &Env<'_>,
    likelihoods: &JDoubleArray<'_>,
    expected: usize,
) -> Result<(), Failure> {
    let actual = likelihoods.len(env)?;
    if actual != expected {
        return Err(fgkl_pairhmm::Error::OutputLength { expected, actual }.into());
    }
    Ok(())
}

fn int_array(env: &Env<'_>, array: &JIntArray<'_>) -> Result<Vec<i32>, Failure> {
    let mut values = vec![0i32; array.len(env)?];
    array.get_region(env, 0, &mut values)?;
    Ok(values)
}

/// Turns an offset table into `(start, end)` ranges, checking it starts at zero, is
/// non-decreasing and ends at `total`.
fn split(offsets: &[i32], total: usize) -> Result<Vec<(usize, usize)>, Failure> {
    let bad = |message: &str| Failure { class: ILLEGAL_ARGUMENT, message: message.to_string() };
    if offsets.first() != Some(&0) {
        return Err(bad("offset table must start at zero"));
    }
    if offsets.last().is_none_or(|&last| last < 0 || last as usize != total) {
        return Err(bad("offset table must end at the array length"));
    }
    offsets
        .windows(2)
        .map(|w| {
            if w[1] < w[0] {
                Err(bad("offset table must be non-decreasing"))
            } else {
                Ok((w[0] as usize, w[1] as usize))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(offsets: &[i32], total: usize) -> Result<Vec<(usize, usize)>, String> {
        split(offsets, total).map_err(|f| f.message)
    }

    #[test]
    fn well_formed_offsets_become_ranges() {
        assert_eq!(ranges(&[0, 3, 3, 7], 7).unwrap(), vec![(0, 3), (3, 3), (3, 7)]);
        assert_eq!(ranges(&[0], 0).unwrap(), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn offsets_must_start_at_zero() {
        assert!(ranges(&[1, 4], 4).unwrap_err().contains("start at zero"));
        assert!(ranges(&[], 0).unwrap_err().contains("start at zero"));
    }

    #[test]
    fn offsets_must_end_at_the_array_length() {
        assert!(ranges(&[0, 3], 4).unwrap_err().contains("end at the array length"));
        assert!(ranges(&[0, -1], 0).unwrap_err().contains("end at the array length"));
    }

    #[test]
    fn offsets_must_not_decrease() {
        assert!(ranges(&[0, 5, 2, 6], 6).unwrap_err().contains("non-decreasing"));
    }
}
