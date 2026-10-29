# AMD395 Benchmark Protocol

## Fixed configuration

- Target: AMD Ryzen AI Max+ 395, Radeon 8060S, `gfx1151`, ROCm 7.2.1.
- Environment: `/home/aup/envs/mujoco-warp-amd-py312`.
- MuJoCo: 3.8.1; MuJoCo-Warp: 3.8.1; Warp: `1.13.0+rocm.0`.
- MJCF: `upstream/mujoco_warp/benchmarks/humanoid/humanoid.xml`.
- Workload: 1024 worlds, 1000 steps, `nconmax=128`, `njmax=128`.
- Same initial state, controls, timestep, solver, tolerances, and iteration cap.
- `MJW_HIP_SINGLE_STREAM=1` for a stable, comparable HIP execution path.
- Warm-up and graph-build time are reported separately from steady-state time.

## Variants

### Eager HIP solver loop

`WP_HIP_CONDITIONAL_NATIVE=0` and `WP_HIP_CONDITIONAL_EMULATION=0` keep the
solver in its regular loop around HIP iteration kernels. This is the baseline.

### Native HIP conditional handle

`WP_HIP_CONDITIONAL_NATIVE=1` and `WP_HIP_CONDITIONAL_EMULATION=0` use the
patched HIP/CLR runtime. The solver creates a device-resident conditional
handle, attaches the captured body graph to a while node, and updates the
condition from device code. There is no host convergence copy during replay.

### Legacy fixed-unroll compatibility path

`WP_HIP_CONDITIONAL_EMULATION=1` captures a fixed upper bound and gates solver
kernels from device state. It remains useful for historical comparison but is
not the native result reported below.

## Correctness gates

1. Both variants complete with exit code 0.
2. `converged_worlds` equals the requested world count.
3. No NaN or Inf is present in the state or solver output.
4. A paired 256-world, 100-step run compares `qpos` and `qvel` against the
   eager reference with tolerance `1e-5`.
5. The exact environment, source revisions, command, logs, and SHA256 values
   are stored with the result.

The recorded state errors are:

```text
max_abs_qpos_error = 6.17932528e-7
max_abs_qvel_error = 9.53674316e-7
numerically_equivalent = true
```

## Primary native result

Artifact: `results/mjwarp_native_conditional_amd395/summary.json`.

| Variant | Runtime | Worlds/s |
| --- | ---: | ---: |
| Eager HIP solver loop | 2.084074 s | 491,345 |
| Native HIP conditional handle | 1.334484 s | 767,338 |

The native path is `1.562x` faster and reduces runtime by `35.97%` in this
fixed workload. The paired state errors are `6.18e-7` for `qpos`, `9.54e-7`
for `qvel`, and `0` for simulation time.

The larger ALOHA side result is intentionally separate: the native prototype
currently reaches 8,937 worlds/s versus 19,117 worlds/s for eager execution.
It identifies graph-build and replay overhead that must be reduced before
claiming a universal speedup.

## P1 sleeping/wakeup protocol

### Full physics-step graph benchmark

`scripts/run_aloha_physical_graph_benchmark.sh` loads the ALOHA pot scene and
replay and runs the complete MuJoCo-Warp step. The frozen AMD395 configuration
uses 16 worlds, 16 warm-up steps, five timed repeats, MuJoCo 3.8.1, ROCm 7.2.1,
the patched HIP/CLR runtime, and a single HIP stream. A long run measures 256
steps per repeat. The policy process sharing the GPU is paused during timing
and resumed immediately afterward.

| Phase | Eager worlds/s | Static worlds/s | Native worlds/s | Native/static | Predicate true |
| --- | ---: | ---: | ---: | ---: | ---: |
| settled | 5,213.66 | 2,847.32 | 2,840.38 | 0.9976x | 45.86% |
| active | 4,268.00 | 2,597.71 | 2,852.14 | 1.0979x | 100% |
| wake cycle | 5,169.52 | 3,524.32 | 2,643.81 | 0.7502x | 100% |

The active one-step native/static state gate records maximum errors of
`9.31e-10` for `qpos` and `2.88e-8` for `qvel`. Long contact trajectories
diverge, so only the settled native/static ratio is used as the P1 throughput
conclusion: the paths are effectively tied at 16 worlds. The active and wake
cycle rows are integration and overhead measurements.

The 64-world static full-step graph did not complete within the expected
runtime after the eager variant reached 17,317 worlds/s. That scan was stopped
and excluded from the result table. It identifies graph scheduling and graph
size as the next runtime work item.

The ALOHA benchmark uses the same MJCF, world count, warm-up, control stimulus,
wake interval, and rollout length for all variants:

| Variant | Cache policy | Graph path |
| --- | --- | --- |
| eager | rebuild every step | direct step calls |
| static | reuse cached collision context | one captured graph |
| native | device wake predicate | native `capture_if` |

Run it with `scripts/fetch_aloha_assets.sh` followed by
`scripts/run_aloha_sleeping_if_benchmark.sh`. The wrapper supports
`STIMULUS=joint_target|none`, `WAKE_MODE=motion|manual`, and
`FIXTURE=dynamic|freeze_after_warmup`. The default dynamic fixture preserves
the physical rollout. The freeze fixture is a graph-path measurement after
warm-up and must not be interpreted as a physical sleeping-task score.

The AMD395 dynamic revalidation used 32 worlds, 16 warm-up steps, 64 timed
steps, zero gravity, and no external actuator stimulus:

| Variant | Worlds/s | Wake predicate | Converged worlds |
| --- | ---: | ---: | ---: |
| eager | 1,278.16 | n/a | 32/32 |
| static | 1,332.28 | n/a | 32/32 |
| native `capture_if` | 1,145.19 | 1.00 | 32/32 |

The native branch executed correctly and remained conservative: every timed
step detected geometry motion, so it rebuilt the broadphase instead of reusing
stale pairs. This is why the dynamic ALOHA run is not presented as a speedup.
The paired result is stored in
`results/aloha_sleeping_if_amd395_revalidation/summary.json`.

### Manual branch coverage

To verify both sides of the conditional node independently from physical
scene motion, the same runner supports `WAKE_MODE=manual` with
`FIXTURE=freeze_after_warmup`. On AMD395, a 16-world, 128-step run toggled the
device predicate every eight steps: the native false branch covered 87.5% of
timed steps, all 16 worlds converged, and native throughput was 1.015x eager
and 1.008x static. The artifact is
`results/aloha_sleeping_if_amd395_manual_branch/summary.json`. It verifies
branch coverage and convergence; it is not a physical sleeping-task score.

The output includes elapsed time, worlds/s, collision count statistics, solver
iteration statistics, final-state digests, wake-predicate telemetry, and
`SHA256SUMS`. A P1 result must not replace the frozen P0 humanoid result or be
combined with it. Cross-process final-state hashes are provenance only because
GPU atomic ordering can vary; correctness is gated by convergence, finite
state, collision statistics, and the explicit workload protocol.

The later files named `*_revalidation.log` are retained as a shared-GPU
repeat. They completed correctly but were slower because a long-lived policy
service was using the same Radeon. They must not be averaged into the frozen
primary result.

## Device telemetry

The primary run recorded 45-47 C edge temperature, 100% reported GPU busy,
and 76% VRAM allocation. A policy service was present on the shared target;
telemetry is therefore labeled shared-host telemetry.

## Reproduction command

```bash
ROOT=/path/to/mujoco-warp-amd-conditional-graph
ENV=/home/aup/envs/mujoco-warp-amd-py312
HIP_CLR_LIB=/path/to/patched/libamdhip64.so \
ROOT="$ROOT" ENV="$ENV" \
  bash "$ROOT/scripts/run_native_mjwarp_benchmark.sh"
```

The script runs both native and eager variants, a paired state comparison, and
emits `SHA256SUMS`. `scripts/run_mjwarp_amd_benchmark.sh` is a compatibility
alias for the same native benchmark.
