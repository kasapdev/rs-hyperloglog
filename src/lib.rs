//! A from-scratch implementation of the **HyperLogLog** cardinality
//! estimator — the probabilistic data structure behind Redis's `PFCOUNT`
//! and similar "how many distinct things have I seen" features in
//! analytics pipelines.
//!
//! HyperLogLog lets you estimate the number of *distinct* elements in a
//! multiset while using a tiny, fixed amount of memory (`2^p` single-byte
//! registers), regardless of whether you insert a thousand items or a
//! billion. The trade-off is that the answer is an estimate with a known,
//! bounded relative error rather than an exact count.
//!
//! See the crate README for a full explanation of the algorithm
//! (registers, rank, harmonic mean, alpha bias correction, and the
//! small-range linear-counting correction).
//!
//! # Example
//!
//! ```
//! use rs_hyperloglog::HyperLogLog;
//!
//! let mut hll = HyperLogLog::new(14).unwrap();
//! for i in 0..1_000 {
//!     hll.insert(&format!("user-{i}"));
//! }
//!
//! let estimate = hll.estimate();
//! // With p = 14 the expected relative error is small; 1000 truly
//! // distinct items should be estimated well within +/-5%.
//! assert!((estimate - 1000.0).abs() / 1000.0 < 0.05);
//! ```

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Smallest allowed precision (`p`). `p = 4` gives `m = 16` registers.
pub const MIN_PRECISION: u8 = 4;
/// Largest allowed precision (`p`). `p = 16` gives `m = 65536` registers.
pub const MAX_PRECISION: u8 = 16;

/// Errors that can occur when constructing or combining [`HyperLogLog`]
/// estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperLogLogError {
    /// The requested precision `p` was outside the supported range
    /// `[MIN_PRECISION, MAX_PRECISION]`.
    PrecisionOutOfRange {
        /// The value of `p` that was requested.
        requested: u8,
    },
    /// [`HyperLogLog::merge`] was called with an estimator built with a
    /// different precision `p`. Only estimators sharing the same register
    /// layout can be merged.
    PrecisionMismatch {
        /// Precision of `self`.
        self_p: u8,
        /// Precision of the estimator that was passed in.
        other_p: u8,
    },
}

impl fmt::Display for HyperLogLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperLogLogError::PrecisionOutOfRange { requested } => write!(
                f,
                "precision p = {requested} is out of range; must be between {MIN_PRECISION} and {MAX_PRECISION} inclusive"
            ),
            HyperLogLogError::PrecisionMismatch { self_p, other_p } => write!(
                f,
                "cannot merge HyperLogLog with p = {self_p} into one with p = {other_p}; precisions must match"
            ),
        }
    }
}

impl Error for HyperLogLogError {}

/// A HyperLogLog cardinality estimator.
///
/// Construct one with [`HyperLogLog::new`], feed it items with
/// [`HyperLogLog::insert`], and read the estimated number of distinct
/// items inserted so far with [`HyperLogLog::estimate`].
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    /// Number of bits used to select a register (`m = 2^p`).
    p: u8,
    /// Number of registers, `m = 2^p`.
    m: usize,
    /// The registers themselves: `registers[i]` holds the maximum rank
    /// (leftmost-1-bit position) seen among all hashed items that mapped
    /// to bucket `i`.
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Creates a new, empty `HyperLogLog` with precision `p`.
    ///
    /// `p` controls the number of registers `m = 2^p`, and therefore both
    /// the memory usage (`m` bytes) and the expected relative error of the
    /// estimate, which is approximately `1.04 / sqrt(m)`.
    ///
    /// `p` must be in `MIN_PRECISION..=MAX_PRECISION` (4..=16). Values
    /// outside that range are rejected rather than silently clamped,
    /// since a `p` that's too small gives a useless estimator and a `p`
    /// that's too large wastes memory for no accuracy benefit in
    /// practice.
    ///
    /// # Errors
    ///
    /// Returns [`HyperLogLogError::PrecisionOutOfRange`] if `p` is not in
    /// `MIN_PRECISION..=MAX_PRECISION`.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let hll = HyperLogLog::new(14).unwrap();
    /// assert_eq!(hll.precision(), 14);
    /// assert_eq!(hll.num_registers(), 16_384);
    ///
    /// assert!(HyperLogLog::new(2).is_err());
    /// assert!(HyperLogLog::new(20).is_err());
    /// ```
    pub fn new(p: u8) -> Result<Self, HyperLogLogError> {
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&p) {
            return Err(HyperLogLogError::PrecisionOutOfRange { requested: p });
        }
        let m = 1usize << p;
        Ok(Self {
            p,
            m,
            registers: vec![0u8; m],
        })
    }

    /// Returns the precision `p` this estimator was constructed with.
    pub fn precision(&self) -> u8 {
        self.p
    }

    /// Returns the number of registers `m = 2^p`.
    pub fn num_registers(&self) -> usize {
        self.m
    }

    /// Returns the expected relative standard error of [`estimate`](Self::estimate)
    /// for this precision, `1.04 / sqrt(m)` with `m = 2^p` registers.
    ///
    /// About 68% of estimates fall within one standard error of the true
    /// cardinality and about 95% within two, once enough items have been
    /// inserted for the asymptotic formula to apply.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let hll = HyperLogLog::new(14).unwrap();
    /// assert!((hll.standard_error() - 0.008125).abs() < 1e-9); // 1.04 / 128
    /// ```
    pub fn standard_error(&self) -> f64 {
        1.04 / (self.m as f64).sqrt()
    }

    /// Returns `true` if no item has been inserted (every register is zero).
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let mut hll = HyperLogLog::new(10).unwrap();
    /// assert!(hll.is_empty());
    /// hll.insert("x");
    /// assert!(!hll.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&r| r == 0)
    }

    /// Resets the estimator to its just-constructed state, keeping the
    /// precision and the allocated registers. Useful for reusing one sketch
    /// across time windows without reallocating.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let mut hll = HyperLogLog::new(10).unwrap();
    /// hll.insert("x");
    /// hll.clear();
    /// assert!(hll.is_empty());
    /// assert_eq!(hll.estimate(), 0.0);
    /// assert_eq!(hll.precision(), 10);
    /// ```
    pub fn clear(&mut self) {
        self.registers.fill(0);
    }

    /// Hashes `item` and folds it into the estimator.
    ///
    /// Uses [`std::collections::hash_map::DefaultHasher`] (SipHash), which
    /// is **not cryptographically secure** and is not guaranteed to be
    /// stable across Rust versions, but has good avalanche properties
    /// (small input changes flip roughly half the output bits), which is
    /// the property HyperLogLog actually depends on for accuracy. A
    /// production system processing adversarial input, or one that needs
    /// hash stability across versions/processes, would typically swap in
    /// a dedicated fast 64-bit hash such as xxHash instead.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let mut hll = HyperLogLog::new(10).unwrap();
    /// hll.insert(&"hello");
    /// hll.insert(&"world");
    /// assert!(hll.estimate() > 0.0);
    /// ```
    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        self.insert_hash(hash);
    }

    /// Core algorithm step, factored out so it can be unit-tested directly
    /// against hand-picked hash values.
    fn insert_hash(&mut self, hash: u64) {
        // The first `p` bits of the hash select which of the `m = 2^p`
        // registers this item updates.
        let index = (hash >> (64 - self.p)) as usize;

        // Shift the index bits out; what's left (in the top `64 - p` bits
        // of `remaining`, zero-padded below) is the part of the hash used
        // to compute the rank.
        let remaining = hash << self.p;

        // Rank = 1-based position of the leftmost 1-bit among the
        // remaining `64 - p` bits. If all remaining bits are zero (an
        // extremely rare event), the rank is defined as `64 - p + 1`,
        // the maximum possible position + 1, matching the classic
        // definition of the statistic.
        let bits_remaining = 64 - self.p;
        let rank: u8 = if remaining == 0 {
            bits_remaining + 1
        } else {
            // `leading_zeros` on the shifted value counts leading zero
            // bits within the full u64, which is exactly the count of
            // leading zero bits within the `bits_remaining`-bit window
            // since it stops at the first 1, found within that window.
            remaining.leading_zeros() as u8 + 1
        };

        if rank > self.registers[index] {
            self.registers[index] = rank;
        }
    }

    /// Returns the estimated number of distinct items inserted so far.
    ///
    /// Uses the standard HyperLogLog estimator: the (bias-corrected)
    /// harmonic mean of `2^register` across all registers, with a
    /// small-range correction (linear counting) applied when the raw
    /// estimate is small and some registers are still empty. See the
    /// crate README for the full derivation.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let hll = HyperLogLog::new(10).unwrap();
    /// // No items inserted: estimate should be (approximately) zero.
    /// assert!(hll.estimate() < 1.0);
    /// ```
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let alpha = alpha_m(self.m);

        let indicator_sum: f64 = self.registers.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        let raw_estimate = alpha * m * m / indicator_sum;

        let zero_registers = self.registers.iter().filter(|&&r| r == 0).count();

        if raw_estimate <= 2.5 * m && zero_registers > 0 {
            // Small-range correction: linear counting.
            m * (m / zero_registers as f64).ln()
        } else {
            raw_estimate
        }
    }

    /// Merges `other` into `self`, so that `self` afterwards estimates the
    /// cardinality of the *union* of everything inserted into either
    /// estimator.
    ///
    /// This works by taking the register-wise maximum, which is valid
    /// because each register already holds the maximum rank seen for its
    /// bucket; merging two sketches built with the same `p` is equivalent
    /// to having inserted every item into a single sketch.
    ///
    /// # Errors
    ///
    /// Returns [`HyperLogLogError::PrecisionMismatch`] if `other` was
    /// constructed with a different precision `p`, since the register
    /// arrays would not correspond to the same hash-bit layout.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_hyperloglog::HyperLogLog;
    ///
    /// let mut a = HyperLogLog::new(12).unwrap();
    /// let mut b = HyperLogLog::new(12).unwrap();
    /// for i in 0..500 {
    ///     a.insert(&format!("a-{i}"));
    /// }
    /// for i in 0..500 {
    ///     b.insert(&format!("b-{i}"));
    /// }
    /// a.merge(&b).unwrap();
    /// // The union has ~1000 distinct items.
    /// assert!((a.estimate() - 1000.0).abs() / 1000.0 < 0.1);
    /// ```
    pub fn merge(&mut self, other: &HyperLogLog) -> Result<(), HyperLogLogError> {
        if self.p != other.p {
            return Err(HyperLogLogError::PrecisionMismatch {
                self_p: self.p,
                other_p: other.p,
            });
        }
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *b > *a {
                *a = *b;
            }
        }
        Ok(())
    }
}

/// The bias-correction constant `alpha_m` from the original HyperLogLog
/// paper (Flajolet et al.), as a function of the number of registers `m`.
impl Default for HyperLogLog {
    /// Creates an estimator with precision 14: 16,384 registers (16 KiB) and
    /// a standard error of about 0.81%, the usual default for cardinality
    /// sketches.
    fn default() -> Self {
        Self::new(14).expect("precision 14 is within MIN_PRECISION..=MAX_PRECISION")
    }
}

fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_out_of_range_precision() {
        assert_eq!(
            HyperLogLog::new(3).unwrap_err(),
            HyperLogLogError::PrecisionOutOfRange { requested: 3 }
        );
        assert_eq!(
            HyperLogLog::new(17).unwrap_err(),
            HyperLogLogError::PrecisionOutOfRange { requested: 17 }
        );
        assert!(HyperLogLog::new(4).is_ok());
        assert!(HyperLogLog::new(16).is_ok());
    }

    #[test]
    fn empty_estimator_estimates_near_zero() {
        let hll = HyperLogLog::new(10).unwrap();
        assert!(hll.estimate() < 1.0);
    }

    #[test]
    fn merge_rejects_mismatched_precision() {
        let mut a = HyperLogLog::new(10).unwrap();
        let b = HyperLogLog::new(12).unwrap();
        assert_eq!(
            a.merge(&b),
            Err(HyperLogLogError::PrecisionMismatch {
                self_p: 10,
                other_p: 12,
            })
        );
    }

    /// Small-cardinality sanity check: with only 100 distinct items and
    /// p = 14 (m = 16384), the vast majority of registers stay at zero,
    /// so `estimate()` should take the linear-counting branch. Verify the
    /// result is close to the true count of 100.
    #[test]
    fn small_cardinality_is_roughly_correct() {
        let mut hll = HyperLogLog::new(14).unwrap();
        for i in 0..100 {
            hll.insert(&format!("small-item-{i}"));
        }
        let estimate = hll.estimate();
        let true_count = 100.0;
        let relative_error = (estimate - true_count).abs() / true_count;
        // Linear counting is very accurate in this regime; allow a
        // generous 15% relative error to keep the test robust.
        assert!(
            relative_error < 0.15,
            "estimate {estimate} too far from true count {true_count} (relative error {relative_error})"
        );
    }

    /// Re-inserting the same items must be a true no-op on the registers:
    /// max(existing_rank, same_rank_again) == existing_rank always. This
    /// is the whole point of storing a max-rank-per-register rather than
    /// a count, so we assert *exact* equality, not just "close".
    #[test]
    fn reinserting_same_items_does_not_change_estimate() {
        let mut hll = HyperLogLog::new(12).unwrap();
        let items: Vec<String> = (0..5_000).map(|i| format!("dup-item-{i}")).collect();

        for item in &items {
            hll.insert(item);
        }
        let estimate_before = hll.estimate();

        for item in &items {
            hll.insert(item);
        }
        let estimate_after = hll.estimate();

        assert_eq!(
            estimate_before, estimate_after,
            "re-inserting identical items must not change the estimate at all"
        );
    }

    /// The main correctness test: feed in exactly 100,000 known-distinct
    /// items with precision p = 14 (m = 16384) and check the estimate
    /// falls within a statistically justified bound of the true count.
    ///
    /// HyperLogLog's expected relative standard error is `1.04 / sqrt(m)`.
    /// For m = 16384, sqrt(m) = 128, so:
    ///
    ///     relative_error ≈ 1.04 / 128 ≈ 0.008125  (0.8125%)
    ///
    /// We allow 4 standard errors of slack (≈ 3.25%) to keep the test
    /// robust against ordinary statistical variance rather than flaky,
    /// while still being tight enough to catch a genuinely broken
    /// estimator (wrong alpha, off-by-one rank, missing bias correction,
    /// etc.) at this scale.
    #[test]
    fn estimates_large_known_cardinality_within_expected_error() {
        const P: u8 = 14;
        const M: f64 = 16_384.0; // 2^14
        const TRUE_CARDINALITY: f64 = 100_000.0;
        const STANDARD_ERRORS_ALLOWED: f64 = 4.0;

        let mut hll = HyperLogLog::new(P).unwrap();
        for i in 0..100_000u32 {
            hll.insert(&format!("item-{i}"));
        }

        let estimate = hll.estimate();

        let relative_standard_error = 1.04 / M.sqrt();
        let tolerance = STANDARD_ERRORS_ALLOWED * relative_standard_error * TRUE_CARDINALITY;
        let lower_bound = TRUE_CARDINALITY - tolerance;
        let upper_bound = TRUE_CARDINALITY + tolerance;

        assert!(
            estimate >= lower_bound && estimate <= upper_bound,
            "estimate {estimate} outside [{lower_bound}, {upper_bound}] \
             (relative standard error {relative_standard_error}, \
             tolerance +/-{tolerance} at {STANDARD_ERRORS_ALLOWED} standard errors)"
        );
    }

    #[test]
    fn standard_error_matches_the_formula_and_shrinks_with_precision() {
        let hll = HyperLogLog::new(14).unwrap();
        assert!((hll.standard_error() - 1.04 / 128.0).abs() < 1e-12);
        let low = HyperLogLog::new(8).unwrap().standard_error();
        let high = HyperLogLog::new(16).unwrap().standard_error();
        assert!(high < low, "more registers must mean a smaller error");
    }

    #[test]
    fn is_empty_reflects_whether_anything_was_inserted() {
        let mut hll = HyperLogLog::new(10).unwrap();
        assert!(hll.is_empty());
        hll.insert(&1u32);
        assert!(!hll.is_empty());
    }

    #[test]
    fn clear_resets_to_empty_and_stays_usable() {
        let mut hll = HyperLogLog::new(12).unwrap();
        for i in 0..5_000u32 {
            hll.insert(&i);
        }
        assert!(hll.estimate() > 1_000.0);

        hll.clear();
        assert!(hll.is_empty());
        assert_eq!(hll.estimate(), 0.0);
        assert_eq!(hll.precision(), 12);
        assert_eq!(hll.num_registers(), 4_096);

        // Reusable: a fresh batch is estimated within a few standard errors.
        for i in 0..2_000u32 {
            hll.insert(&i);
        }
        let error = (hll.estimate() - 2_000.0).abs() / 2_000.0;
        assert!(error < 4.0 * hll.standard_error(), "error {error}");
    }

    #[test]
    fn default_uses_precision_14() {
        let hll = HyperLogLog::default();
        assert_eq!(hll.precision(), 14);
        assert_eq!(hll.num_registers(), 16_384);
        assert!(hll.is_empty());
    }
}
