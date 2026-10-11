# In-cluster harness image.
#
# UBI9 + Python 3.12 base fits the OCP restricted-SCC (non-root UID, read-only /).
# The image bundles the datasets and transformers libraries so tokenization for
# reasoning-token accounting happens locally without a network hop on every request.
FROM registry.access.redhat.com/ubi9/python-312:latest

USER 0
WORKDIR /app

# Layer 1: build tools + wheel cache for pydantic-core, tokenizers, etc.
RUN dnf -y install --setopt=install_weak_deps=False \
      git gcc gcc-c++ make tar gzip && \
    dnf clean all

# Layer 2: helm binary. Both the sequential orchestrator (when
# --install-router is passed) and the uber-harness controller shell out
# to `helm` to install/uninstall the per-model router Helm release.
ARG HELM_VERSION=v3.16.4
ARG HELM_ARCH=linux-amd64
RUN curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-${HELM_ARCH}.tar.gz" \
      | tar -xzO "${HELM_ARCH}/helm" > /usr/local/bin/helm && \
    chmod 0755 /usr/local/bin/helm && \
    helm version --short

# Layer 3: install the harness package
COPY pyproject.toml README.md* ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Drop back to unprivileged UID for runtime (matches OCP restricted-SCC).
USER 1001

# Sensible defaults for HuggingFace caches inside the pod's writable /results
# volume — orchestrator passes --results-dir=/results at runtime.
ENV HF_HOME=/results/dataset_cache \
    HF_DATASETS_CACHE=/results/dataset_cache \
    TRANSFORMERS_CACHE=/results/dataset_cache \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "pricing_reversal_harness"]
