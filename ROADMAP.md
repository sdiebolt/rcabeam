# Optimization record

Historical goal: reconstruct 200 RCA frames within 400 ms, before clutter filtering.
Current real-time objective: process each 200-frame, 500 Hz ensemble within its
400 ms arrival interval, including filtering and host-input-to-host-PD overhead.
This is a throughput budget for nonoverlapping windows, not acquisition-to-output
latency; overlapping windows require a budget based on their smaller frame step.
Workload: 15 MHz, 80+80 receivers, 16 RC + 16 CR angles, 80×80×157 voxels,
4–12 mm depth, lambda/2 maximum axial spacing. Both targets remain unmet.

## Current recommended execution plan

This execution order supersedes the historical "next" recommendations below.
Completed checkpoints below distinguish implemented filters/baselines from
future temporal compression, preconditioning and reconstruction experiments.
Preserve the current unfiltered reconstruction benchmark and APIs as references.
The historical 400 ms goal is **before filtering**; keep it distinct from the
current complete-pipeline throughput objective above.

### Latest decision and immediate next steps

Steps 1–4 below are implemented and have operator full-size results. The latest
`[3,200)` run is recorded under **Operator full-size optimized filtering result**.
EVD64 is the working candidate for this synthetic high-pass case; SVD128 remains
the gold reference, and the library's complex128 default is unchanged.

1. **Freeze the measured reference and precision/quality checks.** Preserve the
   eight-row benchmark, explicit rank bounds, stage timings and optional JSON.
   Add a small targeted sweep of tissue/blood dynamic range and the spectral gap
   at the cutoff. Compare dominant-only removal separately from upper-cutoff
   filtering; measure weak-component residual/IQ errors as well as whole-PD error.
   Do not accept a precision for acquired-data use from this synthetic scene alone.
2. **Make structural OPW work the main performance track (step 7).** Reconstruction
   is ~1.32 s versus ~0.10 s EVD64 filtering. Finish fresh source-attributed main/
   tail profiling to locate dependent gathers/interpolation stalls, then choose
   one exact, bounded sample-sharing/data-reuse experiment with a stated mechanism.
   Include grouping/conversion costs and independent IQ/PD parity. Earlier blind
   accumulator, shuffle and tail changes did not win; do not repeat them unchanged.
   Keep hardware power/security settings unchanged without explicit authorization.
3. **HERK is a secondary bounded microbenchmark**, not a conditioning fix. Compare
   `Cherk`/`Zherk` against GEMM with the same normalized inputs and precision,
   correct conjugation/triangle semantics, and all layout costs. Require a repeatable
   complete-filter win before integration. EVD64 Gram construction is only 55–64 ms;
   even eliminating it would save about 4% of the current complete pipeline.
4. **Make step 5 conditional on useful retained K.** The fixed-basis commutation
   proof remains valid, but `[3,200)` retains 197 components: exact component OPW
   reduces nominal frame work by only 1.5% before projection overhead. Three
   discarded tissue modes do not mean three reconstructed blood components.
   Do not build/promote a large compressed pipeline for this cutoff. Selecting a
   much smaller K changes the filter and needs an explicit quality/error budget;
   five synthetic blood scatterers are not evidence of acquired blood rank five.
5. **Defer eigensolver preconditioning and broader numerical alternatives.** The
   eigensolve is ~2.3 ms, not the bottleneck. Revisit TSQR/small SVD if EVD64 fails
   required accuracy and direct SVD64 is too costly; residual deflation only when
   weak-mode/upper-cutoff resolution is actually needed. Randomized/sampled methods
   remain separate approximations, not silent replacements.

The numbered sections below retain the detailed gates and historical implementation
order; this immediate queue supersedes their earlier implied next-step ordering.

### 1. Build a meaningful clutter-filtering scene

The current five-target ensemble uses a single common slow-time phase for every
scatterer: each configuration is a fixed channel image times one temporal vector.
It is ideally rank one, so removing its first singular component removes the
simulated targets, not just tissue. Keep it for reconstruction/PSF parity; do not
use it as evidence of blood/tissue separation.

Add a deterministic tissue + blood + noise ensemble, retaining separate noiseless
tissue and blood components as ground truth. Include strong, slowly varying
spatially distributed tissue, weaker blood with distinct/decorrelating temporal
behavior, and reproducible complex noise. Declare motion/PRF, amplitude ratios,
noise level and temporal models; avoid making tissue and blood perfectly
orthogonal by construction. Include multiple regions and rank/dynamic-range
cases, rather than just a perfectly separable low-rank example. Reuse the current
geometry/pulse simulation. Do not change the existing viewer/reference fixture
silently or claim this synthetic scene validates acquired weak-flow sensitivity.

**Gate:** reproducible components and their sum; independently checked channel
simulation/OPW; nontrivial temporal spectrum; known blood-only/tissue-only
reconstructions and a small inexpensive test fixture.

### 2. Implement direct-SVD and temporal-Gram EVD reference filters

For IQ reshaped as `X = (all non-temporal dimensions, T)`, support two explicit
backends for the same global temporal filter:

- Direct thin SVD, `X = U Sigma V^H`: independent numerical reference.
- Hermitian EVD of the temporal Gram matrix, `C = X^H X = V Lambda V^H`:
  reconstruct singular values as `sqrt(Lambda)` and order components descending.
  Accumulate Gram products in bounded row chunks when needed; do not construct
  the enormous spatial `U` matrix or full spatial covariance on this path.

Use an explicit zero-based retained interval `[rank_start, rank_stop)`; do not
invent a universal clinical cutoff. Initially use global filtering, no implicit
centering/windowing, and retain all components above the tissue cutoff unless an
upper noise cutoff is explicitly selected. Choose and label synthetic benchmark
cutoffs from the declared scene, not by tuning for a favorable speed/quality result.
Filtering is `X_f = (X V_keep) V_keep^H`; ordinary PD divides energy by original T.
Keep acquisition-frame outputs distinct from reduced temporal-component outputs.

**Gate:** complex inputs, conjugation/order, rectangular/rank-deficient/zero data,
invalid/nonfinite inputs and rank bounds, retain-all identity, explicit empty-band
behavior, and direct-SVD/EVD projector and filtered-power agreement. Compare
projectors/results rather than individual vectors with arbitrary phases or
rotations inside degenerate eigenspaces. Do not split a degenerate cluster in
an equivalence test. Gram construction squares the condition number: test strong
tissue/weak blood cases and accumulation precision against higher-precision
reference; only clip roundoff-scale negative eigenvalues, not genuine failures.

### 3. Add GPU filtering in both domains

Reuse NumPy for the small reference and the existing optional CuPy dependency
for resident GPU operations; no custom CUDA eigensolver initially. Benchmark
both direct SVD and Gram EVD rather than assume the latter wins end to end.
The small T-by-T eigensolve does not eliminate the cost of forming `X^H X`.

- **Beamformed domain:** OPW IQ -> estimate temporal basis -> filter -> PD.
- **Channel domain:** estimate one basis from vertically stacked raw RC/CR IQ,
  filter each with that shared basis -> OPW -> PD. Gram products can be summed
  across RC/CR without allocating the concatenated input.

These estimate different bases and are different filtering methods; do not assert
that their PD images must match. Keep dimensions, input ownership, working-memory
bounds and precision explicit. Validate GPU results against the reference,
including weak components; do not introduce FP16 projected-component storage yet.
Automatic rank selection and spatially local/blockwise SVD are later quality work.
Nonlinear DMAS/XDoppler/RC-FMAS/St-SW require separately specified filtering
placement: filtering their final responses is not filtering IQ before nonlinearity.

### 4. Integrate and profile the filtered benchmark

Keep the unfiltered all-method table clearly labeled. Add separately named OPW
filtered pipelines on the new scene, comparing direct SVD and Gram EVD in both
domains with the same rank policy per domain. Record basis/Gram construction,
decomposition, projection/filtering, packing, beamforming, reduction and transfers,
plus complete host-input-to-host-PD time and peak working memory. Also report
original T, retained K, spectra, tissue suppression and blood preservation versus
known component references. Avoid CPU IQ round trips in the GPU path.

Use warmed randomized/interleaved trials, identical inputs, finite/reference
checks, independent worker rejection and clock/power/thermal logging. Separate
kernel/resident timings from complete-pipeline cost; new optimization comparisons
should report repeated medians and variation. Preserve the historical benchmark's
best-of-three results as historical, not interchangeable with new medians.

**Gate:** a reproducible filtered reference, explicit rank/precision/timing policy,
and full-cost measurements. Acquire representative real data before making
physiological sensitivity claims or accepting precision changes for production.

### 5. Prove and measure exact temporal-component OPW

For a fixed orthonormal temporal basis V_keep and fixed linear OPW operator B,
`B(X V_keep) = (B X) V_keep`. Beamform K projected components instead of T frames,
then calculate `PD(v) = sum_k |B(X V_keep)(v,k)|^2 / T`, **not divided by K**.
Use the same basis for RC/CR before coherent combination. This is not valid for
nonlinear DMAS or other nonlinear compounding/reductions.

First prove full-unitary-basis PD invariance and retained-subspace equivalence to
explicit filtering with the **same projector**, using random complex ensembles,
RC/CR cancellation and frame/component tails. Then measure basis estimation,
channel projection, packing, reconstruction and PD together. Inspect measured
spectra/blood preservation before assuming a small useful K; discarding 20 of
200 modes leaves 180 and may not repay the projection cost.

**Gate:** independent equivalence tests plus a repeatable complete-pipeline win.
No assumed linear runtime scaling or projected 5x headline without measurements.

### 6. Evaluate sampled bases and sliding-window reuse, conditionally

- Estimate temporal bases from spatially distributed beamformed voxel subsets.
  Compare resulting filters with the full-volume basis across heterogeneous
  motion/flow regions. Rearrangement remains exact for the chosen basis, but
  subset estimation is approximate. Do not use spatial singular vectors U to
  project channels. Region-specific bases may require separate projections.
- Profile the filtered pipeline before choosing sliding-window preconditioning:
  only prioritize it if overlapping-window eigensolves matter to total latency.
  Establish an ordinary Gram-EVD sliding-window reference first. Compare several
  overlaps/window steps, independent blocks, abrupt motion and degenerate spectra.

#### Research candidate: previous-window EVD preconditioning

Sebastian K. Praesius, Kees Joost Batenburg and Jorgen Arendt Jensen,
*Real-Time SVD Clutter Filtering using Preconditioning*, IEEE IUS 2025,
[DOI: 10.1109/IUS62464.2025.11201625](https://doi.org/10.1109/IUS62464.2025.11201625),
[DTU publication/accepted-manuscript link](https://orbit.dtu.dk/en/publications/real-time-svd-clutter-filtering-using-preconditioning/).

The DTU abstract reports 401-frame windows, batches of 32 frames, an RTX 4090,
in vivo rat-kidney data, 1120 eigendecompositions/s (60% higher throughput), and
640 images/s including RF transfer, beamforming, motion correction and filtering.
These are the authors' workload-specific results, not rcabeam predictions; a 60%
throughput increase is not a 60% latency reduction. The abstract was checked;
fetching the linked full manuscript returned HTTP 403. Verify full-paper details
before implementing the reported cyclic row-reordering variant or quoting the
secondary 699/882/1120-per-second comparison as independently checked.

Candidate transform: `C_prime = V_prev^H C V_prev`; diagonalize it as
`W Lambda W^H`, then recover `V = V_prev W`. With a full unitary previous basis
and a converged solve this is a change of coordinates, not an approximation.
Investigate the correct row permutation for the sliding-window frame shift,
including wraparound/new frames. A truncated previous basis is not an equivalent
full-space preconditioner. Warm-start sign/phase/degeneracy issues need tests.

A generic `eigh` call may not exploit near-diagonality. Check solver algorithm,
iteration control and Jacobi availability before adding a custom/native backend.
Count basis rotation, covariance formation, decomposition, recovery, state and
memory costs; fallback to the ordinary solve when needed. Preconditioning does
not remove covariance work, and has no guaranteed benefit for nonoverlapping
200-frame blocks. Motion correction is not currently part of rcabeam; do not
imply replication of the paper's complete pipeline.

### 7. Structural reconstruction experiments after the filtered baseline

Each candidate must state how it reduces dependencies, IQ gathers, arithmetic or
intermediate size before implementation; retain the current kernel as reference.

- Quantify neighboring-voxel overlap of interpolation sample pairs, then test
  exact sharing with voxel-specific weights/phases and all grouping costs.
- Derive exact RCA/OPW decomposition before coding. Geometric separability alone
  does not make interpolation/sample access separable. The existing approximate
  two-stage prototype has unresolved quality errors; keep it opt-in.
- Investigate Fourier per-angle reconstruction and angular DMAS/coherence as
  separate reconstruction/quality experiments. Fourier linearity does not prove
  equivalence to the present aperture/interpolation model; coherent sums cannot
  recover channel-resolved quantities required by current channel DMAS.
  Apply clutter filtering before nonlinear operations where the method requires
  it. F-DMAS terminology generally means filtered DMAS, not Fourier DMAS.
- Explore tensor-core formulations only with explicit data grouping, operand
  precision, conversion overhead and independent weak-signal validation.
- Use source-level stalls to motivate genuine asynchronous loading or shorter
  dependency chains, not another blind register/shuffle rewrite.

### 8. Approximate temporal methods, only with a quality/error budget

Separate randomized SVD (approximate subspace estimation) from random mixtures
for power estimation. For independent zero-mean complex Gaussian entries of R
with variance 1/K, `sum_k |B X R|^2 / T` is an unbiased per-voxel PD estimate with
relative standard deviation 1/sqrt(K) for nonzero fixed signals. At K=32 that is
about 18%, not an exact low-rank replacement. Use the same R across RC/CR.
With fixed clutter projector P, use `X P R`; conditional unbiasedness requires
R independent of the data/basis estimation. Orthogonal designs have different
variance; a complete unitary basis preserves energy exactly. Measure spatially
correlated estimation errors, not just an image-wide average or expected value.
Do not promise time-series recovery. Count projection and basis-estimation cost.

### Parked/rejected work and hardware policy

- Single RC/CR accumulator, shared-to-shuffle phase changes and simple tail
  remapping: tested, no retained win. Larger batches, one-voxel mapping,
  cross-channel register pipelines and aperture-group skipping: slower or
  inconclusive. Revisit only with a changed workload and a new hypothesis.
- Geometry caching, native-output restoration removal, persistent voxel kernels
  and atomic reduction redesign: low priority unless filtered-stage costs change.
- Dimension specialization/autotuning: wait for actual retained-component sizes.
  Compact geometry/texture interpolation: precision-sensitive research, not fixes.
- Redundant finish removal is retained; sample-index reconstruction was not a
  precision bug. Fewer registers/static instructions alone are not a success gate.
- Compare 55 W/80 W or other GPUs only with controlled same-workload trials and
  explicit approval before changing power settings; historical runs cannot
  isolate the power-limit penalty. Keep acquired-data FP16 validation separate.

### Latest profiling availability

After the review experiments, the operator restored per-user R615 profiling
capabilities. A fresh central-tile FP32 main-kernel capture succeeded: 64
registers/thread, no spills, 65.23% achieved occupancy, 1.83 eligible warps per
scheduler, no eligible warp in 30.64% of scheduler cycles, and long-scoreboard
stalls accounting for about 44.6% of average warp cycles between instructions.
Registers still limit theoretical occupancy to 66.7%; the 59/60-register
experiment did not demonstrate a useful occupancy/performance gain. This is
memory-dependency evidence, not a 44.6% speedup prediction or proof of which
producer dominates without source attribution. Clocks/caches were uncontrolled;
replay timings are not throughput measurements. Fresh FP16/source attribution
remains to do. Report and details: `/tmp/rcabeam-opw-review/baseline-main.ncu-repz`
and `baseline-main-details.txt`. Earlier permission-failure entries are historical.

## Clutter scene checkpoint: ready for operator benchmark

Step 1 now has a separate `_support.clutter_scene` / `_support.clutter_benchmark`
harness; the common-phase scene, all-method benchmark and viewers are preserved.
The standard geometry is shared rather than duplicated. Seed 42 generates 16
fixed tissue points (source amplitude 20, heterogeneous 1-5 Hz amplitude/phase
modulation), five blood points (source amplitudes 0.5-1, complex stationary AR(1)
with rho=0.9 and 35-140 Hz modulation), and independent RC/CR complex noise
(std 0.05 per quadrature). PRF is 500 Hz; RC/CR share scatterer temporal weights.
No orthogonalization or rank truncation is used to manufacture separation.

Spatial/temporal factors preserve noiseless tissue/blood ground truth without
retaining four full channel ensembles; materialization and noise generation use
bounded row chunks. This is a sparse, fixed-position point-scatterer phantom,
not diffuse tissue, geometric translation or physical blood flow. Additional
motion/dynamic-range scenarios and acquired data remain quality-validation work.

The operator can run `PYTHONPATH=examples uv run python -m _support.clutter_benchmark`
(default acquisition), or add `--grid 12 --elements 12 --angles 3 --frames 32 --repeat 1`
for a quick checkpoint. The table reports sampled component IQ RMS, first-mode
energy and spectral content. The 0.1%-of-largest spectral reporting threshold is
not a clutter cutoff. Default output is a compact component table and median
OPW time; `--output PATH` optionally saves spectra, all timings and
clock/power/thermal/utilization telemetry as JSON without dumping it to the
terminal. Telemetry spans component preparation,
validation and GPU calls, not exclusively the timed reconstruction intervals.

Only mixture full-grid unfiltered OPW is timed: warmed repeated host-input-to-PD
calls, median and paired per-trial H2D. Scene creation, sampled component IQ/SVD,
independent NumPy DAS checks, power/IQ consistency and component additivity checks
are outside timing. No SVD/EVD clutter filter is implemented in this checkpoint.
The initial full-size benchmark was left for the operator; their completed run
is recorded below. Filter implementation does not change these timing boundaries.

Validation: 151 tests pass, including compact-output/optional-report checks,
the unchanged common-phase fixture and a
small real-GPU scene benchmark. Tests check reproducibility, component sums,
noise statistics, nonorthogonal temporal structure, AR(1) correlation, shared
RC/CR weights, independent NumPy channel simulation/DAS and input validation.
The initial focused 13-test CUDA memcheck reported zero errors. Full-size timings and tissue/
blood separation claims are not inferred from these small correctness checks.

## Operator scene baseline and CPU filter references

The operator ran the default 200-frame scene checkpoint and reported:

| Sampled component | IQ RMS | First-mode energy | Singular values >0.1% of largest |
| --- | ---: | ---: | ---: |
| Mixture | 323.21 | 94.27% | 13 |
| Tissue | 319.09 | 96.70% | 8 |
| Blood | 51.394 | 32.77% | 5 |
| Noise | 2.5001 | 0.87% | 200 |

Unfiltered mixture OPW median: **1499.01 ms**, three warmed complete
host-channel-to-host-PD trials; validation passed. These are operator-supplied
terminal results without the optional detailed telemetry report. Do not compare
this changed scene/unknown clock-power state as a controlled speedup over prior
scenes. The blood rank of five follows from five fixed scatterers; it is not
evidence that acquired blood occupies only five useful dimensions. Tissue's
remaining 3.30% energy after its first mode can still obscure weak signals.

Step 2 now provides `rcabeam.temporal_basis` and `rcabeam.svd_clutter_filter`
NumPy references in `src/rcabeam/clutter.py`. Direct SVD and temporal-Gram EVD
return descending singular values and a complete temporal basis, including wide
and rank-deficient cases. Both use complex128 working precision. The EVD path
accumulates bounded row chunks without allocating spatial singular vectors.
Filtering uses an explicit retained interval, preserves input shape/dtype and
original frame count, and does not mutate the input. No automatic cutoff,
centering, GPU filtering or eigensolver preconditioning was added.

Tests compare known complex factorizations/projectors, filtered power with the
original-T normalization, full/empty bands, rank deficiency and unsplit degenerate
clusters, noncontiguous/chunked inputs, high-dynamic-range weak retained modes,
shared stacked RC/CR, invalid inputs/bounds, overflow and negative-eigenvalue
handling. All **192 tests pass**; focused lint, formatting and typing pass.
These are correctness references, not evidence of Gram-EVD speed on the full
scene. Next implement resident GPU backends and both domain pipelines, then
provide the operator the next useful filtered-benchmark checkpoint.

## Resident GPU filters and filtered benchmark checkpoint

Historical baseline; superseded by the optimization/precision checkpoint below.

Steps 3/4 now have a runnable baseline, not a temporal-compression optimization.
`rcabeam.gpu_clutter` provides direct SVD, temporal-Gram EVD, explicit-basis
filtering and a convenience estimate/filter call on resident CuPy IQ. Tuple
inputs estimate/apply one shared RC/CR basis; the Gram path sums covariance
products without concatenating. Arrays must be finite and on the current device.
Filtering uses the current stream and preserves input ownership, shape/dtype and
original frame count. Complex128 decomposition/projection is explicit in both
backends; no automatic cutoff, sampling or precision fallback was added.

Direct SVD uses cuSOLVER gesvd with only temporal vectors requested, rather than
allocating unused spatial U. The destructive Fortran input copy and solver
workspace remain large. Gram accumulation and full-time-series projection are
row-chunked. Gram conditioning/negative-eigenvalue safeguards match the CPU
reference. Solver status, nonfinite results and invalid/nonorthonormal supplied
bases are checked, and failures propagate.

Run `PYTHONPATH=examples uv run python -m _support.clutter_benchmark --rank-start 2`
for the next operator checkpoint; optionally add `--output /tmp/clutter-filtered.json`.
Rank 2 is an illustrative synthetic choice, not a clinical default or a rank
selected for a promised speedup. Without `--rank-start`, the scene-only benchmark
is preserved. `--rank-stop` is stop-exclusive and defaults to original T.
The small smoke command can add `--rank-start 2` as well.

The comparison adds beamformed-domain and shared-channel-domain filtering with
both solvers. Every case still reconstructs/filters all original frames and
normalizes power by T. New trials are warmed and randomized/interleaved; foreign
worker rejection and per-trial clock/power/thermal telemetry remain active.
Each pipeline includes allocation, raw H2D, packing, basis estimation, full
filtering, OPW, PD reduction and final PD D2H. Unused raw/packed/workspace buffers
are released before image-domain decomposition to bound working memory.
The unfiltered comparator is fused power-only; filtered rows materialize IQ,
so differences in total time are not solely filter cost.

Compact output reports total/filter time and sampled noiseless tissue-power
residual/blood-power retention through each pipeline's own fixed projector.
These are not image purity or acquired-flow sensitivity metrics. Optional JSON
retains stages (matrix construction and decomposition separately, synchronized
wall times), spectra/bases, per-trial timings/telemetry and peak active CuPy-pool
allocated bytes; driver/cuBLAS allocations are excluded from that memory metric.
Small diagnostic transfers and checks are outside the timed PD path. Explicit
GPU-memory skips do not substitute another algorithm or precision. Non-memory
solver/validation failures abort rather than produce misleading timings.

Proof: GPU direct/EVD outputs agree with independent NumPy complex128 references
for wide/tall, zero, noncontiguous, shared RC/CR and weak-band cases. Tests cover
input/rank/basis validation, propagated solver failure and explicit memory skips.
Small end-to-end comparisons check direct/EVD agreement within each domain,
full-volume repeatability, and sampled PD against CPU filtering of sampled OPW
IQ with the same basis. Channel and image bases are intentionally not equated.
All **229 tests pass**, focused lint/format/type checks pass, and a 51-test CUDA
memcheck run reports zero errors. Full-size comparative timings remain for the
operator; no SVD/EVD speedup is claimed from small correctness tests.

Next gate: inspect operator stage timings and component preservation before
choosing cutoffs or prioritizing exact component-space OPW. In particular,
preconditioning only targets eigensolve cost, not Gram construction/projection.

## Optimized projector and explicit precision checkpoint

The operator's full 200-frame result exposed the original baseline's cost:
image Gram filtering ~2280 ms and channel Gram filtering ~2149 ms. JSON stage
medians isolated ~902/815 ms in covariance construction, ~6 ms in EVD and
~1395/1325 ms in full retained-subspace projection. Individual stage medians do
not have to add to combined medians. Preconditioning a 6 ms eigensolve cannot
solve the dominant cost; earlier "optimized Gram" terminology was premature.

The current implementation removes that redundant large projection. For fixed
orthonormal V, it chooses the smaller of retained and discarded subspaces:
`X_f = X V_r V_r^H` or `X_f = X - X V_c V_c^H`. Retaining `[2, 200)` projects two
columns, not 198. Full/empty/middle bands and noncontiguous inputs are covered.
This reduces projection multiplication work, not covariance work, and does not
by itself imply a measured end-to-end speedup.

Image PD reduces chunked residual IQ directly without allocating the full
filtered IQ volume. It does **not** subtract nearly equal total and clutter
energies; a constructed 1e8:1 amplitude fixture demonstrates why that loses weak
power. Residual arithmetic follows the chosen precision; real/imaginary squares
and power accumulation are float64, normalized by original T. Channel filtering
still reconstructs T channel frames before full-frame OPW. Component-space OPW
and any precision/sampling fallback remain unimplemented.

This checkpoint initially kept one full complex128 SVD reference and optimized
Gram in complex128/complex64 per domain (six rows total; expanded below). The reference obtains
spatial U in cuSOLVER's overwritten input copy and reconstructs `U Sigma V^H`
with discarded singular values zeroed; no complementary shortcut is used in the
reference. The independent NumPy SVD reference likewise reconstructs from U/S/Vh.
Basis-only tall-matrix SVD retains its temporal-vectors-only allocation behavior. Previous
full-retained-projector benchmark variants are removed rather than retained as
extra rows.

Gram precision is explicit; complex128 remains the default. Global maximum
real/imaginary component scaling and row normalization bound the covariance.
Componentwise real division avoids overflow in complex division itself, and
normalization promotes to float64 before division for complex128 work. Subnormal
float32 amplitude scaling also uses real float64 division before the explicit
complex64 cast. Diagnostics record the normalized covariance's minimum
eigenvalue, negative count and dtype-dependent tolerance; small negatives are
clipped before square root, significant negatives/nonfinite decompositions fail.
Returned singular values are float64, rescaled to the original input units.

Negative **eigenvalues**, not singular values, are the relevant finite-precision
failure: Hermitian is not sufficient for PSD. Scaling prevents amplitude
under/overflow, not squared-condition-number loss. An adversarial 24x6 complex128
fixture with singular values `[1e5, 2, 1, .3, .1, .01]`, retaining `[1, 5)`, showed
~3.3e-6 relative IQ error for Gram128 versus direct SVD, but ~44.5% for Gram64,
with two roundoff-negative eigenvalues and otherwise finite arithmetic. This is
a numerical stress fixture, not a physiological sensitivity benchmark.

Output stays compact: total/filter time, PD relative-L2 error against SVD128,
and sampled component fractions. The fractions are evaluated with each estimated
projector in CPU complex128; they do not validate complex64 arithmetic. Optional
JSON includes peak-normalized maximum PD error, clipping diagnostics, projector
size, per-stage/trial timing and telemetry. Relative-L2 or peak-normalized maximum
PD error >0.5% is an engineering warning, not a clinical quality threshold.
Numerical EVD64 failures are labeled failed, memory limits skipped; neither falls
back. Image filter timing includes its PD reduction; comparisons to earlier
filter-only stage timings must account for that boundary change.

Proof: **246 tests pass**; **68 GPU/scene tests under CUDA memcheck report zero
errors**. Focused Ruff formatting/lint, ty and diff checks pass. End-to-end tests
exercise both precisions and the SVD reference at 16 and 200 frames, compact
terminal/optional JSON output, explicit memory skips, numerical failures and
visible precision warnings. Projector-size checks prove the 200-frame high-pass
uses two modes; overflow/subnormal, weak-band and residual-power cancellation
fixtures cover the numerical changes. No full-size timing claim is made.

Next operator checkpoint (full-size speedup deliberately not inferred from tests):

```bash
PYTHONPATH=examples uv run python -m _support.clutter_benchmark \\
  --rank-start 2 --output /tmp/clutter-optimized.json
```

Inspect actual PD errors and covariance/application costs before selecting a
precision or attempting temporal-component OPW. Acquired-data weak-flow
sensitivity remains unvalidated.

## Direct SVD64 comparison

Added SVD64 as a precision control alongside SVD128, Gram128 and Gram64 in both
domains (eight rows). This helps separate single-precision loss from the extra
conditioning loss caused by forming `X^H X`. SVD128 remains the sole gold
reference and the default direct-SVD precision; no defaults or cutoffs change.

Both direct precisions use the same full-SVD, zero-singular-values reconstruction,
with cuSOLVER `cgesvd`/`zgesvd` and matching workspace/value dtypes. Basis-only,
wide-matrix and shared RC/CR paths also accept explicit complex64. Nonfinite
complex128-to-complex64 casts fail before entering the solver. SVD64 numerical
failures and PD-error warnings follow the same explicit reporting as Gram64,
without fallback. All PD errors remain relative to same-domain SVD128.

On the preceding adversarial weak-band fixture, relative filtered-IQ errors
against independent NumPy SVD were ~6.6e-12 (SVD128), **0.180% (SVD64)**,
~3.3e-6 (Gram128), and **44.5% (Gram64)**. This illustrates the conditioning
difference, not universal SVD64 safety or physiological sensitivity.

Proof: **256 tests pass**, including wide/tall/single-frame, noncontiguous,
shared-input, full/empty-band, solver-failure and cast-overflow checks.
**78 GPU/scene tests pass CUDA memcheck with zero errors**; focused Ruff and ty
checks pass. The existing full benchmark command now includes SVD64 automatically;
full-size timings remain for the operator.

## Operator full-size optimized filtering result

Source: operator run and `/tmp/clutter-optimized.json` (local artifact, not bundled
in the repository), three warmed randomized/interleaved trials per case:

```bash
PYTHONPATH=examples uv run python -m _support.clutter_benchmark \\
  --rank-start 3 --output /tmp/clutter-optimized.json
```

200 frames at 500 Hz represent a 400 ms nonoverlapping slow-time ensemble.
Retained interval `[3,200)` is the operator's explicit synthetic cutoff, not an
automatic/clinical default. Unfiltered fused-PD OPW median: **1424.50 ms**.

| Pipeline | Total ms | Filter ms | PD relative-L2 error vs same-domain SVD128 | Tissue power left | Blood power kept |
|---|---:|---:|---:|---:|---:|
| Image SVD128 | 5966.55 | 4500.29 | reference | 3.673% | 97.89% |
| Image SVD64 | 2050.15 | 576.91 | 0.00345% | 3.673% | 97.89% |
| Image EVD128 | 2582.35 | 1110.76 | 4.44e-7% | 3.673% | 97.89% |
| Image EVD64 | 1592.23 | 114.31 | 6.91e-5% | 3.673% | 97.89% |
| Channel SVD128 | 5650.99 | 4176.89 | reference | 3.253% | 97.93% |
| Channel SVD64 | 1982.79 | 433.03 | 0.00353% | 3.253% | 97.93% |
| Channel EVD128 | 2523.30 | 1039.36 | 6.77e-11% | 3.253% | 97.93% |
| Channel EVD64 | 1573.46 | 98.51 | 3.84e-5% | 3.253% | 97.93% |

Both EVD precisions had zero negative Gram eigenvalues in warm and timed trials.
EVD64 is faster and closer to SVD128 PD than direct SVD64 **in this dominant-only
removal case**, not universally. Clocks vary across algorithms; these are actual
pipeline measurements, not isolated precision-throughput ratios. Prior rank-2
runs and earlier hardware conditions are not controlled same-workload speedups.
The 19 ms image/channel total difference is too small to declare a domain winner.

| EVD64 stage median | Image ms | Channel ms |
|---|---:|---:|
| Raw H2D | 137.04 | 123.60 |
| OPW reconstruction | 1323.37 | 1320.93 |
| Gram construction | 64.05 | 55.39 |
| Eigensolve | 2.25 | 2.28 |
| Filter application | 39.51 | 32.91 |

Individual stage medians need not add to combined/total medians. Image filtering
includes its PD reduction; channel PD is a separate ~13.09 ms stage. Packing,
workspace, validation and allocation/cleanup also count in complete time.
The fastest total is **3.93x** the 400 ms budget; reconstruction alone is **3.30x**
that entire budget. Eliminating filtering cannot reach real time.

The similar domain sizes explain their similar costs:

- RC: `(368 * 80 * 16, 200) = (471040, 200)`; CR is the same.
- Shared channel basis: conceptually `(942080, 200)`, **1.404 GiB** complex64;
  Gram construction sums RC/CR products without full concatenation.
- Image: `(80 * 157 * 80, 200) = (1004800, 200)`, **1.497 GiB** complex64,
  only **6.66%** more rows than the combined channels.
- Both Gram matrices are `200 x 200`; discarded tissue basis is `200 x 3`.
  Both current pipelines still beamform all 200 frames. Complex128 full-input
  copies double the listed input storage, excluding additional workspace.

Component fractions use sampled noiseless components and each estimated basis,
not image purity. From sampled RMS/retention, residual tissue component power is
still ~1.3–1.4x retained blood power; cross-terms mean mixture PD is not simply
the sum of component PD. Small whole-PD error does not establish weak-flow accuracy.

## LLM conditioning/HERK report: assessment and bounded candidates

The operator supplied a numerical-analysis report recommending HERK, dominant
subspace removal, accumulation changes, deflation, QR/TSQR and randomized SVD.
Treat these as hypotheses and distinguish stability from performance:

- **Already implemented:** explicit Hermitian symmetrization, Hermitian `eigh`,
  global amplitude scaling, selectable covariance precision, and smaller-subspace
  filtering. Global scaling prevents amplitude under/overflow, not conditioning.
  Hermitian symmetry does not guarantee PSD; clipping small negatives is not
  recovery of lost spectral information.
- **HERK:** [NVIDIA cuBLAS section 2.7.14](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-herk)
  specifies one-triangle `alpha A^H A + beta C` for conjugate-transpose mode.
  It can avoid computing both triangles and the explicit symmetrization pass,
  but still squares the condition number. No guaranteed 2x speedup or improved
  weak-mode accuracy. Triangle storage still uses a dense buffer; input storage
  is not halved. Current row-major chunks require correct column-major/conjugation
  handling; count any copy/conversion. Never average an uninitialized triangle
  into the computed one. Compare both precision variants, then retain only wins.
- **Eigensolver:** inspected installed CuPy: the current two-dimensional CUDA
  `eigh` path already uses one-triangle cuSOLVER `xsyevd`. Calling `Cheevd` directly
  is not automatically a different/better algorithm and only targets ~2.3 ms here.
- **Dominant projector versus weak spectrum:** perturbations relative to the
  cutoff's spectral gap matter. Inaccurate small eigenvalues need not imply a bad
  dominant tissue projector, but small projector errors acting on strong tissue
  can still overwhelm weak residual blood. An upper noise cutoff additionally
  requires weak-mode separation. Inspect filtered IQ/weak-component errors, not
  just negative counts, eigenvalues or global PD norms.
- **Corrected stress-fixture interpretation:** on the same 24x6 fixture above,
  `[1,5)` gives SVD64 **0.180%** versus EVD64 **44.5%** relative filtered-IQ error;
  removing only the dominant mode, `[1,6)`, gives SVD64 **0.195%** versus EVD64
  **0.510%**. The latter was an ad hoc numerical check, not a new automated gate
  or acquired-data result. Do not use the upper-cutoff stress failure to label
  every EVD64 high-pass filter unusable.
- **FP32 arithmetic/accumulation:** verify effective cuBLAS compute/math settings
  and any TF32 override before attributing errors to IEEE FP32 alone. FP64 summation
  of FP32 partial Grams only improves inter-chunk summation; it cannot recover
  errors inside each product. Casting an already rounded Gram to FP64 is not a
  precision remedy. HERK likewise is not proof of full-mantissa multiplication.
- **QR/TSQR plus small SVD:** avoids explicit normal equations and is a plausible
  accuracy/performance alternative if needed. Retain only compact R factors,
  count all passes/layout/workspace, and avoid materializing huge Q. It is not
  implemented or established as faster than native direct SVD for this workload.
- **Residual deflation/refinement:** another pass on `R = X - X V_r V_r^H` may
  improve resolution of weaker modes, but adds input passes and is limited by
  initial subspace/projection cancellation errors. Not needed merely to discard
  three dominant components already matching SVD128 closely.
- **Explicit semantics required:** mean removal and spatial normalization/whitening
  change the estimated filter. New upper cutoffs and randomized subspaces require
  declared semantics/error budgets, not silent substitution. Adding a scalar
  diagonal shift leaves exact eigenvectors unchanged and cannot repair the desired
  projector.

Relevant reference: Theo Mary, *Error Analysis of the Gram Low-Rank Approximation
(and Why it is Not as Unstable as One May Think)*, SIAM J. Matrix Anal. Appl. 46(2),
1444–1459 (2025), [DOI 10.1137/24M1687649](https://doi.org/10.1137/24M1687649).
Publication metadata/abstract were checked; direct full-text fetch failed. Do not
claim its low-rank-approximation bounds guarantee a small relative error in our
weak clutter-filtered residual without checking the theorem assumptions and norm.
The earlier IUS preconditioning report remains documented in step 6; present
measurements deprioritize it because decomposition is not the bottleneck.

## Retained implementation

- OPW: 64-frame batches, 16,384-voxel tiles, vectorized two-complex loads,
  within-channel prefetch, and a specialized eight-frame tail.
- Fuse complex phase rotation into FP32 accumulation; share phase coefficients
  within each warp with explicit synchronization.
- Traverse power-only Cartesian grids in 2×4 lateral patches, preserving odd
  tails and canonical output order. IQ export keeps canonical traversal.
- Opt-in FP16 packed samples, with FP32 geometry, interpolation weights,
  phase rotation, accumulation and output. Packing rejects nonfinite and
  overflowing components. Inputs/raw H2D remain complex64; default stays FP32.
- Native bindings release the GIL and skip unused allocations. FPM power-only
  reconstruction compounds/reduces chunks without retaining full ensemble IQ.
- Benchmark rows report full end-to-end time and measured same-run raw H2D.
  S4 FPM remains an opt-in, idealized multiplexed comparison, not acquired data.

The obsolete OPW ensemble kernel, `--compare-baseline`, and redundant
`opw_ensemble_pd_from_channels` wrapper/test were removed. Use
`ensemble_pd_from_channels(..., method="opw")` in Python and the packed
`rcabeam_ensemble_device` API in C. Independent NumPy references, scalar
fallbacks, and precision/range/tail/device-buffer regression checks remain.

## Agent experiment results

Experiments used randomized/interleaved trials, output parity checks,
compute-worker monitoring, and temperature/SM-clock/power logging on an
RTX 5080 Laptop. Desktop activity and variable clocks qualify the results.
The GPU enforced an 80 W limit; no power or security settings were changed.

Full timings include allocation, H2D, packing, reconstruction/reduction and
D2H, excluding simulation. A verified local hook selected the previous CUDA
kernel for matched comparisons on a five-target scene.

### Power-only: five trials per mode

| Mode | Median full | Raw H2D in that trial |
| --- | ---: | ---: |
| Previous kernel/traversal | 1480.82 ms | 93.72 ms |
| Retained FP32 | 1383.65 ms | 95.23 ms |
| Opt-in FP16 storage | 1279.95 ms | 96.24 ms |

### IQ export: three trials per mode

| Mode | Median full | Raw H2D in that trial |
| --- | ---: | ---: |
| Previous kernel | 1668.83 ms | 91.75 ms |
| Retained FP32 | 1599.79 ms | 90.47 ms |
| Opt-in FP16 storage | 1529.22 ms | 89.04 ms |

Synthetic FP16 power maximum absolute error was approximately 2.1e-6 of the
FP32 peak. This does not validate acquired weak-flow signals, especially after
clutter filtering. Int16 acquisition does not imply FP16-equivalent precision.

### Rejected or inconclusive approaches

- Larger 128/256-frame batches and one-voxel warp mappings were slower.
- Batch192, four-voxel mapping, group aperture skipping and separable geometry
  showed only about 1–3% gains, insufficient against clock/run variation.
- Read-only loads matched baseline; bypassing L1 and per-pair ballots lost.
- Narrow descriptor addressing did not improve throughput.
- Register-based cross-channel pipelining increased registers from 64 to 78
  without spills and was slower. Shared coefficients alone were also slower;
  retained only the measured FMA/shared combination.
- Direction-aware shared-memory RC/CR spatial transposition lost to simple
  traversal reordering.
- Tiled IQ traversal with host reshape/transpose restoration took 1745.98 ms
  versus 1623.96 ms for canonical FP32 in a separate matched run. Not retained.

No rejected kernels or profiling harnesses are shipped. Local experiment and
sanitizer artifacts remain in `/tmp/rcabeam-opw-batches/`; this record preserves
conclusions without relying on temporary files. After cleanup, 63 library tests
pass; focused CUDA memcheck, synccheck and racecheck cover the retained kernels.
Formatting, lint and source/example type checks also pass.

## GPU-resident workspace checkpoint

`rcabeam.gpu.OpwWorkspace` now owns packed samples, geometry scratch and
canonical IQ/power output. It uses the existing C API without another DAS
kernel. Metadata is copied once; new frames can be packed into existing buffers.
Device/default-stream guards, serialized calls, failure invalidation and
borrowed-output lifetime are explicit. CuPy output restoration enables the
spatial traversal for IQ without a host copy. It allocates an extra full IQ
buffer, so spatial traversal remains an explicit option.

Same current five-target workload, five interleaved trials within each
precision, with compute-worker monitoring and clock/thermal/power sampling:

| Storage/traversal | Resident reconstruct + GPU restoration | Pack in matching trial |
| --- | ---: | ---: |
| FP32 canonical | 1244.90 ms | 6.06 ms |
| FP32 spatial | 1160.99 ms | 5.97 ms |
| FP16 canonical | 1201.91 ms | 4.37 ms |
| FP16 spatial | 1146.94 ms | 4.37 ms |

These resident profiling figures exclude one-time setup, raw H2D, packing and
output D2H. Geometry generation remains included; metadata snapshots are reused,
not fully precomputed per-volume geometry. FP16 IQ sample maximum error was
approximately 4.7e-5 of the sampled FP32 peak, not an acquired-flow validation.

A separate five-trial steady-state **complete host-IQ round trip** updates
preallocated raw GPU buffers, packs/reconstructs and downloads canonical IQ
into a preallocated host output:

| Storage/traversal | Median steady round trip | Raw H2D in that trial |
| --- | ---: | ---: |
| FP32 canonical | 1485.04 ms | 88.85 ms |
| FP32 spatial | 1392.79 ms | 86.55 ms |
| FP16 canonical | 1415.78 ms | 84.80 ms |
| FP16 spatial | 1356.45 ms | 91.74 ms |

Unlike the allocating API figures above, these exclude one-time allocation and
metadata upload. Do not compare them as a same-run allocating-API speedup.
Spatial traversal consistently improved matched runs; the extra FP16 speedup
was modest and remains qualified by sequential precision groups/clock variation.
400 ms remains unmet even without transfers.

Local logs: `/tmp/rcabeam-resident-{results,roundtrip,ncu}.log`. A fresh Nsight
attempt still returned `ERR_NVGPUCTRPERM`; security/power settings were unchanged.
Workspace tests cover scalar/cooperative tails, odd grids, the geometry tile
boundary, precision, metadata snapshots, repacking, invalid inputs/streams,
output invalidation, buffer identity and allocation-free warmed reconstruction.
77 library tests pass; focused 21-test memcheck, synccheck and racecheck runs
report zero errors/hazards. Formatting, lint and source/example type checks pass.
Hardware FP16 RF is an ingress follow-up; no RF-to-IQ conversion or DMA claim
is made from these complex-IQ tests.

## Matched FPM mach/ffdas checkpoint

Same RTX 5080 Laptop, 94×94×157 voxels, 1024 receivers, five waves,
200 compounded frames, 368 complex64 IQ samples, 20 MS/s, 15 MHz carrier/full
bandwidth, 100 µm pitch and F-number 1. Both libraries receive the same
`(frames, receivers, waves, samples)` acquisition layout. Coherent compounding
precedes any power reduction; output is canonical voxel/frame complex64 IQ.
ffdas 0.1.1 core was built from unmodified commit
`6466de37dc3cb2f18946a2ef4543b788af783e95`, Release/SM120, CUDA 13.4;
mach-beamform 0.2.0 and CuPy 14.2.0 ran in an isolated Python 3.12 environment.

### Safety before speed

Initial 128/200-frame ALG2 timings were **discarded**, despite numerical checks
passing: CUDA memcheck subsequently found invalid local accumulator reads.
Short calls also produced invalid local writes, NaNs and reference mismatches.
`das_alg2_kernels.cuh` sizes its accumulator for `M*N`, but the output loop
continues to the full batch size, beyond the kernel's `N`-frame tile. Small
vectorized dispatches can also require more accumulator entries than allocated.
No neighboring ffdas files were changed.

The retained comparison uses 64-frame calls; ALG2 zero-pads the final eight
frames to 64 and counts padding, conversion and extra DAS work. Only the actual
200 frames are restored/exported. A 200-frame, 64-receiver, odd-grid smoke run
covering all five backends and the padded remainder reports **zero memcheck
errors**. Every timed output is finite and agrees at target/random/boundary
voxels and selected slow-time/chunk-boundary frames with independent float64
NumPy geometry, interpolation, radial aperture and phase rotation. Downloaded
IQ and power are checked too. This is synthetic parity, not weak-flow validation.

### Three uncontaminated interleaved trials per backend/mode

| Backend | Resident canonical IQ | Full IQ → unfiltered power | Raw H2D in that power trial |
| --- | ---: | ---: | ---: |
| mach FP32 | 5317.66 ms | 5819.77 ms | 212.23 ms |
| ffdas ALG2 FP32 samples | 5265.13 ms | 5538.03 ms | 206.97 ms |
| ffdas ALG2 FP16 samples/interpolation | 4239.42 ms | 4491.65 ms | 206.73 ms |
| ffdas ALG4 TF32 | 5971.52 ms | 6213.40 ms | 232.33 ms |
| ffdas ALG4 FP16 | 4411.96 ms | 4747.86 ms | 207.53 ms |

Resident excludes setup, H2D and D2H, but includes layout changes, library
workspace operations, coherent compounding, ALG2 tail padding and GPU canonical
restoration. Full power includes warmed allocation requests, 3.015 GB raw H2D,
metadata upload, full-IQ reconstruction, unfiltered mean-power reduction and
D2H. It is **not** the existing bounded-memory streaming-power benchmark.
ALG2 FP32 still uses FP16 fractional weights; its FP16 mode interpolates FP16
samples before FP32 accumulation. ALG4 TF32/FP16 are tensor-arithmetic modes,
not simply different storage. Peak-normalized reference errors were about
0.000053/0.000086/0.000093/0.000784/0.000149 respectively.

Fastest valid resident result is about **47 compounded volumes/s**, not 500;
FP16 ALG2 reduced resident time about 20% versus mach. FP32 ALG2's roughly 1%
resident difference is inconclusive. This is not an RCA throughput ceiling or
a reproduction/refutation of a different GPU's single-observation headline.

Complete 2.22 GB IQ export was also measured, but highly variable: median
mach/ALG2-FP32/ALG2-FP16/ALG4-TF32/ALG4-FP16 times were
8731.51/8370.46/7936.03/9977.65/7761.54 ms, with individual runs spanning
7457.75–15982.46 ms. Do **not** use these to rank backends: a subsequent host
inspection showed almost-full 4 GB swap and heavy I/O wait. A clean host rerun
with separately measured D2H is needed before drawing transfer conclusions.
No swap, power or security settings were changed.

GPU medians were near 79.4–79.6 W; resident/power temperatures were 58–67 °C
with variable clocks and desktop activity. A separate run was aborted when an
Effusive/MATLAB compute worker appeared; its results were not retained. Monitor
processes run independently of the GIL, since ffdas's blocking native calls can
prevent a Python monitor thread from observing overlaps. Nsight Systems is
available; Nsight Compute counters remain permission-blocked. Earlier unsafe
ALG2 profiling timings are not retained as performance evidence.

Reproduce with `PYTHONPATH=examples uv run python -m _support.matrix_backends`. Local accepted log:
`/tmp/rcabeam-fpm-safe-three-trials.log`; sanitizer:
`/tmp/rcabeam-fpm-safe-memcheck.log`. Earlier unsafe/overlapped logs are retained
locally only to diagnose the rejection, not to support throughput claims.
77 library tests, lint/format and source/example type checks pass. No example
tests, generated images, PDF, external source edits or commits were added.

## Paper-like single-observation TC checkpoint

A closer comparison uses `PYTHONPATH=examples uv run python -m _support.ffdas_paper`: 1024 receivers,
128-frame batch, Q=1, K=256 complex64 IQ samples, λ receiver pitch, native
batch-first GPU output and ffdas's CUDA-event timer. Internal permutation,
FP16 conversion and warmed output allocation requests are included; H2D,
metadata upload, simulation, validation, canonical output restoration and D2H
are excluded. The source/build/GPU are unchanged from the preceding checkpoint.
ALG2 is omitted because its 128-frame calls failed memory-safety checks.

**Exact reproduction remains unavailable:** Table 1 supplies voxel spacing but
not grid origin/depth bounds, IQ sampling ratio, receive cone or benchmark
source. The public Python reconstruction example supplies one IQ sample per
carrier wavelength, normal plane-wave transmit, no receive cone and z_min=20λ.
These are explicit assumptions, not recovered paper settings. Deterministic
random finite IQ avoids costly simulation without changing geometry-dependent
memory access; the paper explicitly states simulation fidelity does not affect
its reconstruction performance measurements.

Two 128³ grids, five interleaved trials per precision after two warmups:

| Geometry | Voxel spacing | TC-FP16 batch | TC-FP16 volume | Volumes/s | TC-TF32 batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published spacing; assumed origin | 0.5λ | 474.45 ms | 3.707 ms | 269.8 | 678.28 ms |
| Public example fixed ROI, resampled to 128³ | 0.25197λ | 356.30 ms | 2.784 ms | 359.2 | 502.38 ms |

First grid: x/y ±31.75λ, z=20–83.5λ. Second: x/y ±16λ,
z=20–52λ. The second **does not** match Table 1's λ/2 spacing; it demonstrates
sensitivity to an otherwise unspecified field of view. All required receive
samples fall within K=256. Native wall medians closely track event medians:
474.54/356.33 ms for FP16, so CPU timing boundaries are not responsible for
these results. FP16 batch ranges were 467.40–499.06 and 353.90–359.10 ms.

Paper Figure 5 is approximately 1.3 ms/single-wave volume at 2.1 million voxels
(graph estimate, about 770 volumes/s). Our table-spacing run is roughly 2.9×
slower, and the fixed-ROI variant roughly 2.1× slower. This narrows the earlier
rough 4–5× workload-adjusted comparison, but does not reproduce the headline.
Different GPU power/clocks and unpublished geometry/settings remain unresolved;
these runs do not establish how much each explains.

FP16 reference RMS error was 0.000388/0.000383, peak-normalized maximum
0.000441/0.000361, using independent float64 geometry/interpolation across all
128 frames at selected random and boundary voxels. Every timed output is
finite. A focused 1024-receiver/128-frame/256-sample/8³-grid TC smoke reports
zero CUDA memcheck errors. No overlapping compute workers were detected.
Sampled FP16 SM medians were 1867/1567 MHz, power medians 70.89/55.38 W,
temperatures 55–63 °C; clocks were not locked and desktop activity remains.
No power/security settings or external sources were changed.

Accepted local logs: `/tmp/ffdas-paper-{table,example}.log`; sanitizer:
`/tmp/ffdas-paper-memcheck.log`. The new script reuses the existing benchmark's
process-based GPU monitor through public helpers. No PDF or commits were added.

## Nsight counters and native-layout experiment

The operator granted their account R615 profiler-device/profiler-context/trace
capability access. Actual Nsight Compute 2026.3 collection now succeeds; earlier
permission failures above are historical. No power settings were changed.
Profiles explicitly use `--clock-control none --cache-control none`. Replay
profiles have variable clocks/cache state and are **not throughput benchmarks**.

Current five-target RCA workload, spatial traversal, 200 frames: the main kernel
covers 192 frames with three 64-frame grid-y batches, followed by one eight-frame
tail kernel **per voxel tile**. Central tile 30 was selected with 60 matching
launches skipped; edge tile 60 with 120. FP32/FP16 central profiles show:

| Metric | FP32 | FP16 storage |
| --- | ---: | ---: |
| DRAM utilization | 8.44% | 5.00% |
| L2 hit rate | 96.39% | 96.28% |
| L1/TEX hit rate | 71.06% | 66.14% |
| SM utilization | 63.36% | 78.06% |
| Registers/thread | 64 | 64 |
| Achieved occupancy | 66.62% | 66.46% |
| Register spills | 0 | 0 |

FP32 long-scoreboard stalls account for approximately 48% of average warp
cycles between issued instructions. PC samples concentrate at the IQ
interpolation's `second-first` FADD consumers of vector global loads, rather
than geometry generation. FP16 lowers memory traffic and shifts pressure toward
the shared FMA-heavy pipeline (78.1% utilization). Branch efficiency is 99.84%;
excessive global sectors are negligible. Thus this is not an exhausted-DRAM or
obviously uncoalesced-load problem. These percentages are diagnostic, not
additive speedup predictions; occupancy is register-limited without spilling.

A separate Nsight Systems FP16 reconstruction trace attributes 1079.09 ms to
main DAS kernels, 113.07 ms to eight-frame tails, **4.34 ms to geometry**,
6.34 ms to restoration, and 0.09 ms to finish kernels. Packing and output
validation are separate. Geometry caching cannot remove hundreds of
milliseconds here; the costly tail represents about 9.5% of DAS time but still
cannot explain the overall threefold gap.

A seven-trial randomized, unprofiled layout experiment reused the same spatial
workspace, geometry and packed data. Native-order output was checked bitwise
against canonical sampled IQ, with all native output finite. No product API was
changed; the temporary harness accesses owned buffers only to bypass restoration.

| Storage | Canonical output median | Native output median | Separate restoration median |
| --- | ---: | ---: | ---: |
| FP32 | 1236.54 ms | 1242.97 ms | 6.20 ms |
| FP16 | 1199.39 ms | 1185.70 ms | 6.42 ms |

End-to-end differences are small against trial/clock variability. Removing
restoration costs roughly half a percent, not a major speedup; it could still
save the extra 1.61 GB IQ buffer if downstream processing accepts native order.
No new native-layout API is retained merely for this inconclusive timing change.
These are separate runs, not a regression comparison to prior workspace figures.
No foreign compute workers were detected; desktop activity remains.

Local artifacts: `/tmp/rcabeam-current-{fp32,fp16}-center30.ncu-repz`,
`/tmp/rcabeam-current-fp32-edge60.ncu-repz`, their detail/source exports,
`/tmp/rcabeam-current-stage-times.nsys-rep`, and
`/tmp/rcabeam-native-layout-{float32,float16}.log`.

## Historical post-Nsight recommendations

Superseded in priority by the current execution plan at the top; retained here
as context for the measurements that preceded the clutter-filtering discussion.

- Prioritize a bounded RCA tensor-core prototype to reduce scalar FMA work and
  share IQ loads across neighboring voxels; include grouping/conversion costs
  and independently validate any newly quantized operands.
- Treat geometry caching and restoration removal as low-priority latency work
  given the measured stage costs.
- Profile genuine asynchronous loading or shorter accumulation chains before
  introducing another pipeline or changing interpolation precision.
- Validate FP16 storage on acquired data; reserve H200 for final validation.

## Review-driven OPW accumulator, tail and phase experiments

Same five-target RCA workload on the RTX 5080 Laptop, but the GPU now reports
an enforced **55 W cap**, not the earlier 80 W. No power/permission settings
were changed. These are same-run comparisons, not regressions against earlier
absolute timings. Desktop activity, variable clocks and thermals still qualify
small differences. The temporary monitor allowed the verified Obsidian Electron
GPU process in addition to the usual desktop applications; other compute workers
remained rejected.

Temporary nvcc `-O3 -lineinfo -Xptxas=-v`, sm_120 libraries compared the original
kernel, one RC/CR accumulator, register phase shuffles, and their combination.
The eight-frame tail also used 4- and 2-voxel mappings with minimum legal
16- and 32-frame template batches; unused frame pairs remained guarded.
Nine randomized/interleaved CUDA-event trials per precision timed central tile
30, ten repeated launches per case, excluding geometry, clearing and packing.

| Original-accumulator tail mapping | FP32 median/tile | FP16 median/tile |
| --- | ---: | ---: |
| 8 voxels/warp, batch 8 (retained) | 2.851 ms | 2.261 ms |
| 4 voxels/warp, batch 16 | 3.048 ms | 3.224 ms |
| 2 voxels/warp, batch 32 | 5.835 ms | 4.997 ms |

Short-tail accumulator/shuffle differences were clock-sensitive and did not
establish a repeatable win. A follow-up used seven randomized/interleaved
full-volume trials per precision, reusing packed data and preallocated buffers.
Host wall timing included all 62 voxel tiles, geometry, PD clearing, main/tail
kernels and the original finish launches, but excluded packing, transfers and
canonical output restoration. IQ output remained enabled for every variant.

| Full-volume variant | FP32 median | FP16 median |
| --- | ---: | ---: |
| Original kernel | 1860.55 ms | 1772.13 ms |
| One RC/CR accumulator | 1900.12 ms | 1818.34 ms |
| Register phase shuffles | 1971.84 ms | 1774.04 ms |

One accumulator reduced main registers/thread from 64 to **59 FP32 / 60 FP16**,
with zero spills, but did not improve time. Phase shuffles retained 64 registers.
SASS inspection found fewer static instructions in both variants, but unchanged
vector IQ-load and FFMA instruction counts; fewer instructions/registers alone
were not a throughput win. Fresh Nsight Compute collection failed with
`ERR_NVGPUCTRPERM`, so there are no new scheduler/stall-counter conclusions.

All central-tile candidate IQ/power outputs matched the original within 5e-5
peak-relative tolerance. Full-volume checks used 1024 deterministic sampled
voxels/all frames plus a full-output finiteness check: single-accumulator IQ
maximum error was at most 2.47e-6 of the reference peak; shuffle comparisons
were exact in those checks. This is synthetic parity, not acquired-flow proof.

**Retained:** only skip `finish` for methods 0, 2 and 3, preserving it for 1 and
4 and preserving launch-error checks. No kernel tuning options or experimental
kernels were added. Independent NumPy tail tests now cover 2/4/6/8 frames,
64/66/68/70/72 frames and the 200-frame workload with 1/17 channels and odd voxel
counts. All 137 tests pass; focused 42-test CUDA memcheck reports zero errors.
Local reproducible harnesses, sources, build/SASS logs, raw timing/telemetry JSON
and validation logs: `/tmp/rcabeam-opw-review/`. These temporary artifacts are
not shipped; this checkpoint preserves the measured conclusions.

## Experimental paper-inspired two-stage OPW checkpoint

Implemented an opt-in `rcabeam.two_stage` prototype, not a replacement for the
retained OPW kernel. Stage 1 beamforms one centered, depth-padded reference plane
per angle/orientation; Stage 2 applies the plane-wave mapping and coherently sums
not-a-knot cubic-spline interpolants. Wavelength uses `geom.f_demod` (assumed to
match the carrier). Our positive transmit-sine convention gives
`z_ref = z + (elevation - reference_elevation) * tan(theta/2)`.
The current linear fast-time interpolation is retained, whereas the paper used
cubic interpolation in channel DAS. All samples, spline work and outputs remain
FP32/complex64. No spatial transmit apodization was added to our OPW model.

`TwoStageOpwWorkspace` accepts raw resident IQ without packing, owns canonical
IQ and plane/spline scratch, and snapshots geometry. Synchronous default-stream,
device and input-layout guards and serialized calls are explicit. The borrowed
output is overwritten on reconstruction. Thirty-two-frame chunks bound scratch;
there is no persistent all-frame/all-angle plane volume or FP16 prototype.
Conventional APIs, exports and defaults are unchanged.

### Matched default workload: five interleaved trials per method/mode/density

RTX 5080 Laptop, unchanged 80 W cap, 368 samples, 80+80 receivers, 16+16 waves,
200 frames, 80x80x157 output, 15 MHz carrier/full bandwidth, 20 MS/s IQ, F-number 1.
Five targets and common slow-time phase modulation match `benchmark.py`.
Baseline is the retained **FP32 spatial workspace with GPU canonical restoration**.
Trials were randomized within each density/mode and monitored independently for
foreign compute workers; no overlaps were detected. Known desktop GPU apps,
including Slack, are allowed but remain interference sources. The shared monitor
can now be imported without requiring ffdas/mach; those imports are deferred until
FPM reconstruction. No external source or power/security settings were changed.

| Plane samples/lambda | Reference depths | Scratch GB | Resident conventional / prototype | Speedup | Host round-trip conventional / prototype | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 672 | 0.881 | 1253.12 / 549.98 ms | 2.28x | 1683.62 / 963.59 ms | 1.75x |
| 16 | 1339 | 1.755 | 1238.68 / 959.55 ms | 1.29x | 1746.98 / 1400.91 ms | 1.25x |
| 32 | 2673 | 3.504 | 1207.28 / 1571.38 ms | 0.77x | 1629.14 / 1975.29 ms | 0.82x |

Scratch is planes plus spline second derivatives, excluding the 1.61 GB IQ output
and 1.51 GB raw input. Resident counts dense DAS, spline solves and expansion.
Host round-trip updates preallocated raw GPU arrays, packs conventional inputs,
reconstructs and downloads all IQ into preallocated host output. One-time setup,
JIT, simulation and validation are excluded. Prototype reconstruction avoids
packing; that is correctly counted as a pipeline difference. Timings are not the
allocating power-only benchmark or a complete fUSI pipeline. Host transfer costs
in this run are higher than earlier runs; do not compare those as a regression.
Desktop activity, changing clocks and host memory pressure qualify the results.
Combined per-mode telemetry medians were approximately 1545-1807 MHz, 56.6-74.4 W
and 60-67 C; individual trial variation is retained in the JSON log.

### Quality is not sufficient to promote

Full-volume errors below compare frame-zero complex IQ against conventional OPW.
Other frames share a global phase, so their unfiltered power has the same spatial
error. These are synthetic metrics, not acquired weak-flow validation.

| Plane samples/lambda | Complex relative RMS | Peak-normalized max IQ error | Amplitude relative RMS | Unfiltered power relative RMS | Bright phase RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 21.03% | 5.37% | 13.27% | 8.21% | 0.0819 rad |
| 16 | 20.70% | 4.84% | 12.78% | 4.62% | 0.0827 rad |
| 32 | 20.70% | 4.84% | 12.78% | 4.60% | 0.0828 rad |

Bright phase metrics use conventional amplitude above 10% of its peak.
The persistent discrepancy at 16/32 samples/lambda is evidence that interpolation
density alone cannot fix this configuration. Complex RMS includes sidelobes and
cancellation regions, so it should not be interpreted as 21% peak-amplitude loss.

Independent fine single-target axis traces keep the reference planes at x=y=0
by including the FoV endpoints. Targets: centered at 4.5 mm, (x,y)=(3,-3) mm at
4.5 mm, and (2.6,2.5) mm at 11.5 mm. At density 8, off-axis lateral FWHM increased
7.70-9.87%, with FWTM increases up to 4.95%. At density 16, off-axis lateral FWHM
still increased 7.02-9.15%; density 32 did not materially improve those cases.
Axial interpolation artifacts mostly improved with density. Thus the paper's
reported <=1.19% width change does **not** transfer to our current acquisition,
receive-aperture model and shallow FoV. No amplitude correction, extra reference
planes or change in apodization was slipped into the prototype to mask that result.

A separate single-reconstruction Nsight Systems diagnostic at density 8 attributes
283.67 ms to dense DAS planes, 217.99 ms to expansion, and 21.87 ms to spline solves.
This profiled run is not another throughput trial. Both reconstruction stages need
optimization for further speed; spline solving is not the main cost. The 400 ms
resident target remains unmet, even before considering quality/clutter filtering.

### Proof and reproduction

- NumPy prototype reuses existing channel DAS; spline algebra agrees with an
  independent SciPy `CubicSpline` reference.
- Tests cover the mapping sign, exact normal-incidence geometry, physical moving
  target lag phase in the exact closest-element case, odd layouts, frame/chunk
  tails, two-sample receive-window boundaries, snapshots and input/stream guards.
- Conventional benchmark output is checked at targets/random/boundary voxels
  against independent NumPy DAS. Timed outputs are finite and frame-zero repeat
  checks are outside timing. Common-phase slow-time checks include target voxels
  and frame boundaries, but do not validate arbitrary flow ensembles.
- 90 library tests pass. Focused prototype CUDA memcheck/synccheck/racecheck report
  zero errors/hazards. Targeted Python lint/format/type checks pass; wheel build
  includes both prototype `.py` and RawModule `.cu` sources. Broader type checking
  still encounters existing native-extension resolution/test typing issues, and
  whole-tree formatting flags the pre-existing README workspace snippet.

Reproduce with `PYTHONPATH=examples uv run python -m _support.two_stage_benchmark`.
Research output includes the 8/16/32 density sweep, all individual timings, quality
metrics, telemetry and single-target axis diagnostics. Local initial logs:
`/tmp/rcabeam-two-stage-default.log`, `/tmp/rcabeam-two-stage-psf.log`,
`/tmp/rcabeam-two-stage-{memcheck,synccheck,racecheck}.log`, and
`/tmp/rcabeam-two-stage-profile.nsys-rep` / `-stages.csv`.
No PDFs, generated images or commits were added.

**Stop condition reached:** there is a measurable speed gain, but quality is worse
than the paper's reported equivalence. Keep this opt-in. Next work should first
investigate the receive-aperture/trajectory approximation (potentially bounded
multiple reference planes), not promote it or assume denser sampling solves it.

## Example consolidation checkpoint

Only three top-level entry points remain: `examples/benchmark.py`,
`examples/visualize.py`, and unchanged `examples/storage_estimate.py`.
Internal modules under `_support/` retain the scientific implementations rather
than adding standalone scripts or moving example orchestration into the library.
The unused point-target demo and duplicated monolithic RCA reference were removed.

At this checkpoint the benchmark dispatched `rca`, `two-stage`, `matrix-backends`,
`ffdas-paper`, and `matrix-chunks`; the no-argument update below supersedes that CLI.
Each research harness retains its existing timing boundary and best/median reporting convention;
this consolidation does not make unlike timings directly comparable. ALG2 batch
padding/guards, independent references, precision checks, and discarded-trial
rules remain. The chunk tuner now uses the same independent process-based GPU
monitor as the comparison suites, rather than its Python polling thread. Monitor
failures abort; worker overlaps and allocation failures discard trials.

At this checkpoint visualization selected `rca`, `two-stage`, or `interpolation`;
the no-selection viewer update below supersedes that CLI. They share
one napari renderer and physical `(z,y,x)` millimeter display coordinates.
OPW/interpolation comparisons use a shared reference scale; nonlinear RCA outputs
retain independent normalization because their units/scales differ. Signed DMAS
responses are clipped to the display floor, not interpreted as conventional PD.
`--headless` preserves a compute-only workflow. The two-stage viewer reconstructs
one frame directly and does not run benchmark repetitions or PSF sweeps.

The five-target list and channel simulation are shared. A pre/post small-scene
check found bitwise-identical inputs; the two-stage benchmark's setup, quality
metrics, and fine PSF diagnostics were exactly unchanged (timings excluded).
Numerical library code was not changed during this cleanup.

Validation: 108 tests pass; entry-point routing/help, shared-scene phase,
monitor rejection/failure handling, and viewer reference scaling are covered.
RCA all-method and `--full`/NumPy-reference smokes, two-stage resident,
mach FPM/matched-backend, full/S4 chunk tuning, all three visualization modes,
and storage estimates ran successfully. Actual napari layer construction was
checked headlessly for all comparisons, including the optional FPM layer.
Interactive rendering was not manually checked. Native ffdas execution was not
rerun: its separate Python 3.12+ source-built environment is still required;
its suite help and preserved source checks remain available. Targeted lint,
format and type checks pass (type checking excludes unavailable native imports).
README and historical reproduction commands now use the consolidated entry points.

## No-argument benchmark

`uv run examples/benchmark.py` now runs every RCA method in one Rich table:
OPW, two-stage OPW, XDoppler, RC-FMAS, St-SW, and DMAS-CCF-ACF. The mach FPM
reference runs automatically when installed; otherwise its row is explicitly
skipped. No suite or configuration arguments are accepted. The fixed acquisition
matches the shared five-target scene: 80+80 elements, 16+16 angles, 200 frames,
80×80×157 voxels (x/y/z), 15 MHz carrier/full bandwidth, and 20 MS/s IQ.

RCA rows use the same host-channel-to-host-volume boundary, including allocations,
raw H2D, reconstruction/reduction and final volume D2H. The two-stage row uses
8 samples/wavelength and 32-frame chunks, retains its full GPU IQ and then reduces
power on GPU; it does not download full IQ. The table explicitly marks it as an
approximation and reports power RMS error against OPW. No clutter filtering is
included. RCA trials are warmed, interleaved in deterministic shuffled order,
validated, and reported as best of three with raw H2D paired to that same trial.
Independent NumPy channel DAS checks OPW power at targets and boundary voxels.
Overlapping compute workers and monitor failures abort rather than report a
contaminated result. FPM retains its existing streaming best-of-three boundary;
its different acquisition/grid is marked and receives no RCA speedup ratio.

Research density/PSF sweeps, native ffdas comparisons, S4/chunk tuning, FP16 and
single-frame/staged diagnostics remain available as `_support` modules, invoked
with `PYTHONPATH=examples uv run python -m _support.MODULE`. The old configurable
RCA driver is now `_support.rca_diagnostics`; none of these research sweeps run
in the everyday benchmark. Their numerical checks, timing boundaries and ALG2
safety guards are unchanged.

A complete default all-method run, including mach FPM, is recorded locally in
`/tmp/rcabeam-all-methods-benchmark.log`. The synthetic power error is 8.2%,
consistent with the earlier two-stage experiment. These timing samples remain
qualified by desktop activity, clocks and thermal state; they are not a new
controlled performance claim. Acquired-data/clutter-filtered quality remains
unvalidated.

Validation: 114 tests pass, including all-method execution, paired best-trial
upload timing, missing/installed FPM handling, invalid-volume rejection, GPU
complex-IQ power reduction and a small real-GPU all-method smoke. The complete
default workload ran with mach FPM. Lint, changed-Python formatting, targeted
typing and `git diff --check` pass. All five research-module help paths work;
native ffdas execution and interactive napari rendering were not rerun.

## All-method visualization

`uv run examples/visualize.py` now reconstructs one frame of the same default
five-target acquisition and opens every RCA method in one napari viewer:
OPW, density-8 two-stage OPW, XDoppler, RC-FMAS, St-SW and DMAS-CCF-ACF.
The shared method list, two-stage GPU power reduction and finite-volume checks
are reused by the benchmark without changing its timing boundary. The viewer
runs no timing trials or PSF sweeps. Suite, method and geometry arguments are
removed; `--headless` remains for compute-only checks.

Conventional and two-stage OPW share the conventional OPW peak. The renderer
accepts a per-layer peak dictionary so nonlinear methods remain independently
normalized in that same window; scalar shared-reference behavior for research
comparisons is unchanged. Signed DMAS is clipped to the display floor, not
squared. The two-stage layer explicitly labels the approximation. Physical
millimeter axes and canonical-to-display `(x,z,y)` to `(z,y,x)` permutation remain.
Six layers use a 2×3 grid; installed mach adds the five-wave 32×32 FPM reference
on matched display axes and uses a 2×4 grid. Missing mach produces a skip notice.

Interpolation and the 8/16/32 two-stage density viewer remain in research modules:
`PYTHONPATH=examples uv run python -m _support.interpolation` and
`PYTHONPATH=examples uv run python -m _support.two_stage_visualization`.
The default viewer does not run these experiments. The conventional library
APIs/defaults and two-stage kernels are unchanged.

The full default headless run constructed seven finite volumes including mach
FPM (`/tmp/rcabeam-all-methods-visualize.log`). An offscreen real-napari check
constructed seven image layers, checked display dimensions/physical axes,
2×4 grid and selective shared normalization
(`/tmp/rcabeam-all-methods-napari.log`). Actual interactive rendering is not
manually validated; offscreen Qt reports its expected OpenGL-widget limitation.
Acquired-data/clutter-filtered two-stage quality remains unvalidated.

Validation: 125 tests pass, including no-selection entry points, exactly one
reconstruction per method, optional/missing FPM, finite-volume rejection,
shared OPW versus independent nonlinear normalization, signed-response display,
and a real-GPU all-method smoke. Lint, changed-Python formatting, targeted typing
and `git diff --check` pass. Both research viewer module help paths work. The
benchmark's existing numerical/timing tests still pass after sharing helpers.
