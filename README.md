# rs-hyperloglog

A from-scratch, dependency-free Rust implementation of **HyperLogLog** — the
probabilistic algorithm that lets you estimate the number of *distinct*
elements in a huge stream of data while using almost no memory.

This is the same class of algorithm behind Redis's `PFCOUNT` / `PFADD`
commands, and it's the standard tool analytics systems reach for whenever
they need an approximate "how many unique users/IPs/URLs did we see today"
answer over billions of events, where keeping an exact `HashSet` would blow
up memory. `rs-hyperloglog` trades a small, bounded amount of statistical
error for the ability to count unique items using a fixed, tiny footprint
(as low as 16 bytes and as high as 64KB per estimator, depending on the
precision you choose) instead of memory proportional to the number of
distinct items.

## Usage

```rust
use rs_hyperloglog::HyperLogLog;

fn main() {
    // p = 14 -> m = 2^14 = 16384 registers, ~0.81% expected relative error.
    let mut hll = HyperLogLog::new(14).unwrap();

    for i in 0..100_000 {
        hll.insert(&format!("user-{i}"));
    }

    let estimate = hll.estimate();
    println!("estimated distinct users: {estimate:.0}");
    // => something very close to 100000, typically within ~1%.
}
```

Two independently-built sketches (with the same precision `p`) can be
combined with `merge` to estimate the cardinality of their union, which is
what makes HyperLogLog useful for distributed counting (merge partial
sketches from many workers/shards into one):

```rust
use rs_hyperloglog::HyperLogLog;

let mut a = HyperLogLog::new(12).unwrap();
let mut b = HyperLogLog::new(12).unwrap();
for i in 0..500 {
    a.insert(&format!("a-{i}"));
}
for i in 0..500 {
    b.insert(&format!("b-{i}"));
}
a.merge(&b).unwrap();
println!("union estimate: {:.0}", a.estimate()); // ~1000
```

Other helpers: `HyperLogLog::default()` builds a `p = 14` sketch, `is_empty()`
tells whether anything was inserted, `clear()` resets a sketch so it can be
reused across time windows without reallocating, and `standard_error()`
reports the expected relative error (`1.04 / sqrt(m)`) so you can put error
bars on an estimate:

```rust
use rs_hyperloglog::HyperLogLog;

let mut hll = HyperLogLog::default(); // p = 14
for i in 0..50_000 {
    hll.insert(&i);
}
let n = hll.estimate();
let margin = n * hll.standard_error();
println!("~{n:.0} distinct (+/- {margin:.0} at one standard error)");

hll.clear();
assert!(hll.is_empty());
```

## How it works

HyperLogLog keeps `m = 2^p` small registers (one byte each here) instead of
the actual items. Every inserted item is reduced to a single 64-bit hash,
and that hash is used twice:

1. **Register selection.** The first `p` bits of the hash pick which of
   the `m` registers this item belongs to. This is what lets many
   different items share a fixed pool of registers.
2. **Rank.** Of the *remaining* `64 - p` bits, we find the position of the
   leftmost `1` bit (1-indexed) — call this the item's "rank". Intuitively,
   seeing a hash whose remaining bits start with a long run of zeros is
   rare (probability `2^-rank`), so observing one is evidence that *many*
   distinct items have hashed into that register. Each register stores the
   **maximum** rank ever observed for it — which is exactly why
   re-inserting the same item again is a no-op: the same item always
   produces the same hash, the same register index, and the same rank, so
   `max(existing, same_rank)` never changes anything.

To turn `m` registers full of small integers back into a cardinality
estimate, we compute the (bias-corrected) **harmonic mean** of `2^register`
across every register:

```text
raw_estimate = alpha_m * m^2 / sum(2^-register[j] for j in 0..m)
```

The harmonic mean is used (rather than, say, an arithmetic mean) because it
is much more robust to the occasional register that got a very high rank
by chance — exactly the kind of outlier a simple average would over-weight.
`alpha_m` is a bias-correction constant derived analytically in the
original Flajolet et al. paper:

- `alpha_16 = 0.673`
- `alpha_32 = 0.697`
- `alpha_64 = 0.709`
- `alpha_m = 0.7213 / (1 + 1.079/m)` for `m >= 128`

**Small-range correction.** The raw harmonic-mean estimator above is known
to be measurably biased when the true cardinality is small relative to
`m` (most registers are still empty, i.e. zero). In that regime,
`rs-hyperloglog` switches to **linear counting** instead, which is far more
accurate for small counts:

```text
estimate = m * ln(m / zero_registers)
```

This correction is applied whenever `raw_estimate <= 2.5 * m` *and* at
least one register is still zero — matching the original HyperLogLog
paper's recommendation.

**On the hash function.** Items are hashed with
[`std::collections::hash_map::DefaultHasher`] (SipHash-1-3). This is
**not a cryptographic hash** and its output is not guaranteed to be stable
across Rust versions or processes — but it does have good *avalanche*
properties (a one-bit change in the input flips roughly half the output
bits), which is the property HyperLogLog's rank calculation actually
depends on for accuracy. A production system, especially one ingesting
adversarial input or needing hash stability across restarts/machines,
would typically swap in a dedicated fast 64-bit hash such as xxHash
instead — this crate deliberately keeps the dependency count at zero, so
it uses what the standard library provides.

[`std::collections::hash_map::DefaultHasher`]: https://doc.rust-lang.org/std/collections/hash_map/struct.DefaultHasher.html

## Testing

`cargo test` runs, among others, one test that is the real correctness
check: inserting exactly **100,000** distinct generated strings
(`format!("item-{i}")` for `i in 0..100_000`) into a `HyperLogLog::new(14)`
(`p = 14`, `m = 16384`) and asserting the estimate lands within a
statistically justified tolerance of the true count.

HyperLogLog's expected relative standard error is the well-known formula:

```text
relative_error ≈ 1.04 / sqrt(m)
```

For `m = 16384`, `sqrt(m) = 128`, so:

```text
relative_error ≈ 1.04 / 128 ≈ 0.008125   (0.8125%)
```

The test allows **4 standard errors** of slack (≈ 3.25% of 100,000, i.e.
±3,250) around the true value of 100,000 — generous enough to avoid
flakiness from ordinary statistical variance, while still tight enough to
catch a genuinely broken estimator (wrong alpha constant, an off-by-one in
the rank calculation, a missing bias correction, etc.). In an actual run
during development, the test produced an estimate of **100,643.6** for the
true count of 100,000 — a relative error of about 0.64%, comfortably inside
the ±3.25% bound.

Other tests cover:

- **Idempotency**: re-inserting the exact same 5,000 items a second time
  leaves the estimate *exactly* (bit-for-bit) unchanged, since it's a
  no-op on the max-rank registers by construction.
- **Small-range correctness**: 100 distinct items at `p = 14` (where
  almost all registers stay empty, exercising the linear-counting branch)
  lands within 15% of the true count of 100.
- Input validation (`p` out of the supported `4..=16` range is rejected)
  and `merge` precision-mismatch handling.

Run the suite yourself:

```sh
cargo test --verbose
```

## License

MIT © 2026 kasapdev
