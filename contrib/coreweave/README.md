# CoreWeave-specific helpers

Scripts here bind to CoreWeave's cluster naming and their published
kubeconfig layout. Everything in this directory is provider-specific
and NOT required to run the study — the core harness is cloud-agnostic
and needs only a `KUBECONFIG` pointing at any Kubernetes cluster with
the required GPU capacity.

External contributors adding equivalent helpers for other providers
(EKS, GKE, AKS, ...) can add sibling directories: `contrib/aws-eks/`,
`contrib/gke/`, etc.

## Contents

### `select-cluster.sh`

Iterates through a hard-coded list of CoreWeave cluster kubeconfigs the
operator has downloaded to `~/Downloads/CWKubeconfig_<name>_<region>`
and picks the first one with enough free GPUs to host the two spokes
(`MRL_MIN_FREE_GPUS`, default 6). Prints `export CLUSTER=... KUBECONFIG=...`
that the operator can `eval`.

Usage:

```bash
eval "$(contrib/coreweave/select-cluster.sh)"
echo "on cluster ${CLUSTER}, KUBECONFIG=${KUBECONFIG}"
```

Environment overrides recognized:

- `KUBECONFIG_PIGGY`, `KUBECONFIG_FOZZIE`, `KUBECONFIG_KERMIT`
  — override the default paths in `~/Downloads/`.
- `MRL_MIN_FREE_GPUS` — override the minimum free-GPU threshold.

## For non-CoreWeave users

Skip this directory entirely. Point `KUBECONFIG` at your own cluster
before running `make prep` / `make calibrate` / `make cell`:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
make prep
```

The rest of the harness has no CoreWeave-specific dependencies.
