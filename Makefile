# mrl_study — head-to-head study driver
#
# Targets mirror the runbook in docs/mrl-headtohead.md. Every target
# except `prep`, `images`, and `clean` is parameterized by DATASET;
# most cell targets also take SCORER and ARRIVAL. LAMBDA is optional
# on `cell` and `study`: if unset, calibration.json is required.
#
# Examples:
#   make prep
#   make images
#   make calibrate DATASET=totrandom-tail472 TARGET_RHO=0.90
#   make study     DATASET=totrandom-tail472 SCORER=mrl
#   make headtohead DATASET=totrandom-tail472
#   make clean

SHELL := /usr/bin/env bash

# --- Parameter surface ----------------------------------------------------

DATASET     ?=
SCORER      ?=
ARRIVAL     ?=
ROWS        ?=
TARGET_RHO  ?=
LAMBDA      ?=
NAMESPACE   ?= mrl-headtohead

# --- Guard helpers --------------------------------------------------------

# `$(call require,VARNAME)` fails at parse time with a clear message
# rather than passing an empty value into a script that would then
# misinterpret it.
require = $(if $($(1)),,$(error $(1) is required, e.g. make $@ $(1)=<value>))

# --- Top-level targets ----------------------------------------------------

.PHONY: help prep images calibrate cell study headtohead clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | sort | \
	 awk 'BEGIN{FS=":.*?## "} {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Parameters (set with VAR=value):"
	@echo "  DATASET     dataset name (basename of a JSON in harness/data/)"
	@echo "  SCORER      qd|ar|lat|mrl"
	@echo "  ARRIVAL     poisson|weibull-0.5|weibull-0.3"
	@echo "  ROWS        cell row count (defaults to dataset size)"
	@echo "  TARGET_RHO  target utilization for calibration"
	@echo "  LAMBDA      operator-supplied arrivals/sec (bypasses calibration.json)"
	@echo "  NAMESPACE   K8s namespace (default: mrl-headtohead)"

prep: ## GPU preflight, deploy both spokes, ensure PVC/RBAC (assumes KUBECONFIG is exported)
	@if [ -z "$$KUBECONFIG" ]; then \
	  echo "ERROR: KUBECONFIG must be set. Point it at the kubeconfig for a cluster with capacity."; \
	  echo "       CoreWeave users can eval contrib/coreweave/select-cluster.sh to pick automatically."; \
	  exit 1; \
	fi
	scripts/check-gpu-availability.sh
	scripts/deploy-spoke.sh a
	scripts/deploy-spoke.sh b
	kubectl apply -n $(NAMESPACE) -f stack/harness/pvc.yaml
	kubectl apply -n $(NAMESPACE) -f stack/harness/rbac.yaml

images: ## Build + push the EPP and harness images
	scripts/build-image.sh
	scripts/build-harness-image.sh

# --- Calibration ----------------------------------------------------------
#
# calibrate runs the staged bisection probe (mode=full by default).
# Pass LAMBDA to switch to fast mode: a single-stage probe that reports
# the observed rho and warns if outside the target band, without
# iterating.

calibrate: ## Probe the fleet for the study_lambda that lands at TARGET_RHO
	$(call require,DATASET)
	$(call require,TARGET_RHO)
	@echo "[make calibrate] deploying QD Hub (calibration reference)"
	HUB_VALUES=stack/router/hub-values-qd.yaml scripts/deploy-hub.sh
	scripts/hub-smoke.sh
	@if [ -n "$(LAMBDA)" ]; then \
	  echo "[make calibrate] fast mode: single-stage validation of LAMBDA=$(LAMBDA)"; \
	  scripts/validate_lambda_probe.sh --mode fast --dataset $(DATASET) \
	    --target-rho $(TARGET_RHO) --lambda $(LAMBDA); \
	else \
	  echo "[make calibrate] full mode: staged bisection"; \
	  scripts/validate_lambda_probe.sh --mode full --dataset $(DATASET) \
	    --target-rho $(TARGET_RHO); \
	fi

# --- Cells (single scorer × single regime) --------------------------------

cell: ## Deploy scorer's Hub, hub-smoke, run one cell
	$(call require,DATASET)
	$(call require,SCORER)
	$(call require,ARRIVAL)
	@echo "[make cell] deploying $(SCORER) Hub"
	HUB_VALUES=stack/router/hub-values-$(SCORER).yaml scripts/deploy-hub.sh
	scripts/hub-smoke.sh
	@rows_arg="$${ROWS:-$$(python3 -c 'import json,sys; \
	  print(json.load(open(sys.argv[1]))["total"])' \
	  harness/src/pricing_reversal_harness/data/$(DATASET).json)}"; \
	 lambda_arg=""; if [ -n "$(LAMBDA)" ]; then lambda_arg="--lambda $(LAMBDA)"; fi; \
	 scripts/run-config.sh --scorer $(SCORER) --arrival $(ARRIVAL) \
	   --rows "$$rows_arg" --dataset $(DATASET) $$lambda_arg

# --- Study (single scorer × three regimes) --------------------------------

study: ## Run all three regimes for SCORER on DATASET
	$(call require,DATASET)
	$(call require,SCORER)
	@echo "[make study] deploying $(SCORER) Hub (one-time)"
	HUB_VALUES=stack/router/hub-values-$(SCORER).yaml scripts/deploy-hub.sh
	scripts/hub-smoke.sh
	@rows_arg="$${ROWS:-$$(python3 -c 'import json,sys; \
	  print(json.load(open(sys.argv[1]))["total"])' \
	  harness/src/pricing_reversal_harness/data/$(DATASET).json)}"; \
	 lambda_arg=""; if [ -n "$(LAMBDA)" ]; then lambda_arg="--lambda $(LAMBDA)"; fi; \
	 for arrival in poisson weibull-0.5 weibull-0.3; do \
	   echo "[make study] cell: scorer=$(SCORER) arrival=$$arrival dataset=$(DATASET)"; \
	   scripts/run-config.sh --scorer $(SCORER) --arrival $$arrival \
	     --rows "$$rows_arg" --dataset $(DATASET) $$lambda_arg \
	     || { echo "[make study] cell FAILED: scorer=$(SCORER) arrival=$$arrival"; exit 1; }; \
	 done

# --- Head-to-head aggregator ---------------------------------------------

headtohead: ## Run the 4-scorer × 3-regime aggregator for DATASET
	$(call require,DATASET)
	python3 scripts/build_headtohead_table.py \
	  --dataset $(DATASET) \
	  --out local_results/$(DATASET)/head-to-head/

# --- Teardown -------------------------------------------------------------
#
# Removes every K8s resource this study creates: uninstalls Hubs from
# every scorer (idempotent — helm silently no-ops on absent releases),
# deletes both spokes, drops the harness PVC + RBAC + ConfigMaps + all
# harness Jobs, and finally deletes the namespace itself. Does NOT
# touch local_results/ on disk.

clean: ## Tear down all K8s resources this study creates
	@echo "[make clean] uninstalling any deployed Hub releases"
	-helm -n $(NAMESPACE) uninstall $(NAMESPACE)-hub 2>/dev/null || true
	@echo "[make clean] deleting spokes"
	-scripts/deploy-spoke.sh --delete a 2>/dev/null || \
	  helm -n $(NAMESPACE)-cw-a uninstall spoke-gemma-a 2>/dev/null || true
	-scripts/deploy-spoke.sh --delete b 2>/dev/null || \
	  helm -n $(NAMESPACE)-cw-b uninstall spoke-gemma-b 2>/dev/null || true
	@echo "[make clean] deleting all harness resources in $(NAMESPACE)"
	-kubectl delete jobs -n $(NAMESPACE) --all --ignore-not-found
	-kubectl delete configmaps -n $(NAMESPACE) \
	  -l llm-d.ai/guide=mrl-headtohead --ignore-not-found
	-kubectl delete pvc -n $(NAMESPACE) mrl-headtohead-results --ignore-not-found
	-kubectl delete rolebinding,role,serviceaccount -n $(NAMESPACE) \
	  pricing-reversal-harness --ignore-not-found
	@echo "[make clean] deleting namespaces"
	-kubectl delete namespace $(NAMESPACE) --ignore-not-found
	-kubectl delete namespace $(NAMESPACE)-cw-a --ignore-not-found
	-kubectl delete namespace $(NAMESPACE)-cw-b --ignore-not-found
	@echo "[make clean] done. local_results/ on disk left intact."
