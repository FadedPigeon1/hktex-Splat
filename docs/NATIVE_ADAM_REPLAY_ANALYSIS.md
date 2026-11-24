# Native replay: Adam amplifies gradient differences

The user's run `direct_tiles_native_20261117T040251_653086Z_b79be9f3b5`
provides the first complete failed native replay snapshots. Both passes start
from model hash `eab9312fcdb4bfc1a4e046c12fd4e10314eda647448aecd7e7c591ed4eef80d5`,
empty optimizer states and identical learning rates. Loss, all gradient arrays,
Adam moments and location updates pass the existing tolerance. Updated angles
(3 / 5,000 entries) and kernel colors (7 / 15,000 entries) fail.

This is genuine update variability, even though the absolute gradient tolerance
accepts the underlying gradients. At color coordinate `[4374, 2]`, the saved
gradients are `-4.637553843e-7` and `+1.336534297e-6`. With Adam's original
learning rate `0.006154297191437912` and epsilon `1e-8`, the updated values are
`-0.1491356343` and `-0.1612686217`, about `0.012133` apart. At angle `[4113]`,
gradients `-8.981226429e-9` and `-1.458639698e-8` also produce a much larger
parameter difference, approximately `0.011922`, because they lie near epsilon.

For a fresh Adam state, the bias-corrected update is approximately
`-lr * gradient / (abs(gradient) + epsilon)`. Therefore an absolute gradient
tolerance cannot be interpreted as an equivalent bound on the Adam update,
especially for tiny gradients or sign changes.

## Independent CPU reconstruction

The installed, unmodified Torch Adam was run on CPU using the saved initial
parameters and **each pass's own saved gradients**, original configuration and
post-step parameter bounds. All six Adam parameter groups in both passes
reconstruct the saved GPU parameters within the unchanged tolerance. Maximum
CPU-versus-GPU reconstruction error is `2.384185791015625e-7`.

This supports correct Adam behavior for these two saved steps and rules out
different initial optimizer states as their explanation. It does not validate
GeodesicOpt independently or establish why native forward/backward values vary.
No new GPU job, optimizer change, tolerance change or gate bypass was performed.
The CPU analysis helper stayed under `/tmp`; its results and all ten divergent
parameter entries are in
[native_adam_reconstruction.json](hktex_vs_hks_benchmark/native_adam_reconstruction.json).

The unresolved investigation is upstream native forward/backward repeatability,
including discrete outer-KNN/top-30 admission and floating-point arithmetic.
The existing snapshots contain gradients and updates, not per-pass candidate
IDs, so they do not prove a selector or floating-point root cause. Full matched
benchmark measurements remain blocked by the strict gate; this analysis is
not permission to treat the failed replay as a pass.
