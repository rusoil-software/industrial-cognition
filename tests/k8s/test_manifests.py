# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Oct-4
# ==============================================================================
"""Contract tests for the Kubernetes manifests (see doc/k8s.md).

These encode the defects that the `a532152` manifest dump actually had, so the
same class of breakage cannot be reintroduced:

* Deployments referenced a ConfigMap and a Secret that were never created.
* The API `Service` was declared three times with three different ports.
* Stateful data lived on Deployments with bare PVCs.
* An Erlang/broker credential and a Grafana admin password were literals.
* `robot` (a Modbus/RS-485 singleton) could have been scaled.

`tests/k8s/kustomize_min.py` renders the overlays so the suite runs without a
cluster or `kubectl`; when kubectl *is* present the renderer is cross-checked
against it.
"""

from __future__ import annotations

import pytest
import yaml

from tests.k8s.kustomize_min import (
    RenderedObject,
    render,
    render_with_kubectl,
    repo_root,
)

OVERLAYS = ["k8s/overlays/dev", "k8s/overlays/prod"]
WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet"}
STATEFUL_COMPONENTS = ["postgres", "rabbitmq", "redis", "minio"]
PLACEHOLDER_TOKENS = [
    "your-registry",
    "<REPLACE",
    "<BASE64",
    "changeme",
    "TODO:",
    "REPLACE_ME",
]


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=OVERLAYS)
def rendered(request) -> list[RenderedObject]:
    return render(request.param)


def objects_of(rendered: list[RenderedObject], kind: str) -> list[RenderedObject]:
    return [obj for obj in rendered if obj.kind == kind]


def by_component(rendered: list[RenderedObject], component: str) -> RenderedObject:
    for obj in rendered:
        labels = (obj.document.get("metadata") or {}).get("labels") or {}
        if labels.get("app.kubernetes.io/component") == component and obj.kind in WORKLOAD_KINDS:
            return obj
    raise AssertionError(f"no workload found for component {component!r}")


def containers(workload: RenderedObject) -> list[dict]:
    spec = workload.document["spec"]["template"]["spec"]
    return list(spec.get("initContainers") or []) + list(spec.get("containers") or [])


def iter_containers(rendered: list[RenderedObject]):
    """Yield ``(workload, container)`` for every container in the overlay."""
    for obj in rendered:
        if obj.kind not in WORKLOAD_KINDS:
            continue
        spec = (obj.document.get("spec") or {}).get("template", {}).get("spec", {})
        for container in (spec.get("initContainers") or []) + (spec.get("containers") or []):
            yield obj, container


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("overlay", OVERLAYS)
def test_overlay_renders_without_unsupported_features(overlay: str) -> None:
    assert render(overlay), f"{overlay} rendered no objects"


def test_rendered_documents_are_well_formed(rendered: list[RenderedObject]) -> None:
    # Namespace-scoped objects must land in `cogni-ns`. Cluster-scoped ones may
    # carry a namespace on their metadata (Kustomize's namespace transformer
    # sets it and the API server ignores it there), so only the namespace-scoped
    # side is enforced - that is the side `kubectl apply` actually depends on.
    for obj in rendered:
        assert obj.kind, f"object without kind: {obj.document}"
        assert obj.name, f"object without metadata.name: {obj.document}"
        assert obj.document.get("apiVersion"), f"{obj.identifier} has no apiVersion"
        if obj.document.get("kind") == "Namespace":
            continue
        assert obj.namespace in ("", "cogni-ns"), (
            f"{obj.identifier} is outside the cogni-ns namespace"
        )
    # At least the workloads must be namespaced.
    for obj in rendered:
        if obj.kind in WORKLOAD_KINDS:
            assert obj.namespace == "cogni-ns", f"{obj.identifier} is not in cogni-ns"


def test_no_duplicate_object_identifiers(rendered: list[RenderedObject]) -> None:
    """The old manifests declared `industrial-cognition-api` Service twice."""
    seen: dict[tuple, str] = {}
    for obj in rendered:
        key = (obj.kind, obj.namespace, obj.name)
        assert key not in seen, (
            f"{obj.kind} {obj.namespace}/{obj.name} is declared more than once, "
            "which makes `kubectl apply` order-dependent"
        )
        seen[key] = obj.name


def test_every_object_carries_the_project_labels(rendered: list[RenderedObject]) -> None:
    for obj in rendered:
        labels = (obj.document.get("metadata") or {}).get("labels") or {}
        assert labels.get("app.kubernetes.io/part-of") == "industrial-cognition", (
            f"{obj.identifier} is missing app.kubernetes.io/part-of"
        )


def test_kubectl_render_agrees_with_the_python_renderer() -> None:
    """Differential check; skipped when kubectl is not installed."""
    for overlay in OVERLAYS:
        expected = render_with_kubectl(overlay)
        if expected is None:
            pytest.skip("kubectl is not available")
        actual = render(overlay)
        expected_ids = {
            (doc.get("kind"), (doc.get("metadata") or {}).get("name")) for doc in expected
        }
        actual_ids = {(obj.kind, obj.name) for obj in actual}
        assert expected_ids == actual_ids, (
            f"{overlay}: renderer and kubectl disagree on the object set; "
            f"only in python: {actual_ids - expected_ids}, "
            f"only in kubectl: {expected_ids - actual_ids}"
        )


# --------------------------------------------------------------------------- #
# Reference integrity - the exact failure mode of the previous manifests
# --------------------------------------------------------------------------- #
def test_configmap_and_secret_references_resolve(rendered: list[RenderedObject]) -> None:
    config_maps = {(obj.namespace, obj.name) for obj in objects_of(rendered, "ConfigMap")}
    secrets = {(obj.namespace, obj.name) for obj in objects_of(rendered, "Secret")}

    assert ("cogni-ns", "industrial-cognition-config") in config_maps
    assert ("cogni-ns", "industrial-cognition-secrets") in secrets

    for obj in rendered:
        if obj.kind not in WORKLOAD_KINDS:
            continue
        spec = obj.document["spec"]["template"]["spec"]
        for container in (spec.get("initContainers") or []) + (spec.get("containers") or []):
            for source in container.get("envFrom") or []:
                if "configMapRef" in source:
                    ref = (obj.namespace, source["configMapRef"]["name"])
                    assert ref in config_maps, f"{obj.name}: missing ConfigMap {ref[1]}"
                if "secretRef" in source:
                    ref = (obj.namespace, source["secretRef"]["name"])
                    assert ref in secrets, f"{obj.name}: missing Secret {ref[1]}"
            for entry in container.get("env") or []:
                value_from = entry.get("valueFrom") or {}
                if "configMapKeyRef" in value_from:
                    ref = value_from["configMapKeyRef"]
                    assert (obj.namespace, ref["name"]) in config_maps, (
                        f"{obj.name}: env {entry['name']} references missing ConfigMap {ref['name']}"
                    )
                    assert _key_exists(
                        rendered, "ConfigMap", obj.namespace, ref["name"], ref["key"]
                    ), f"{obj.name}: ConfigMap {ref['name']} has no key {ref['key']!r}"
                if "secretKeyRef" in value_from:
                    ref = value_from["secretKeyRef"]
                    assert (obj.namespace, ref["name"]) in secrets, (
                        f"{obj.name}: env {entry['name']} references missing Secret {ref['name']}"
                    )
                    assert _key_exists(
                        rendered, "Secret", obj.namespace, ref["name"], ref["key"]
                    ), f"{obj.name}: Secret {ref['name']} has no key {ref['key']!r}"

        for volume in spec.get("volumes") or []:
            if "configMap" in volume:
                ref = (obj.namespace, volume["configMap"]["name"])
                assert ref in config_maps, f"{obj.name}: volume references missing ConfigMap {ref[1]}"
            if "secret" in volume:
                ref = (obj.namespace, volume["secret"]["secretName"])
                assert ref in secrets, f"{obj.name}: volume references missing Secret {ref[1]}"


def _key_exists(
    rendered: list[RenderedObject], kind: str, namespace: str, name: str, key: str
) -> bool:
    for obj in objects_of(rendered, kind):
        if obj.namespace == namespace and obj.name == name:
            data = obj.document.get("data") or obj.document.get("stringData") or {}
            return key in data
    return False


def test_service_selectors_match_a_workload(rendered: list[RenderedObject]) -> None:
    for service in objects_of(rendered, "Service"):
        selector = service.document.get("spec", {}).get("selector") or {}
        assert selector, f"Service {service.name} has an empty selector"
        matches = [
            obj
            for obj in rendered
            if obj.kind in WORKLOAD_KINDS
            and _selector_in(selector, obj.document["spec"]["template"]["metadata"]["labels"])
        ]
        assert matches, f"Service {service.name} selects no workload: {selector}"


def _selector_in(selector: dict, labels: dict) -> bool:
    return all(labels.get(key) == value for key, value in selector.items())


def test_service_target_ports_exist_on_the_workload(rendered: list[RenderedObject]) -> None:
    for service in objects_of(rendered, "Service"):
        selector = service.document["spec"].get("selector") or {}
        targets = [
            obj
            for obj in rendered
            if obj.kind in WORKLOAD_KINDS
            and _selector_in(selector, obj.document["spec"]["template"]["metadata"]["labels"])
        ]
        for port in service.document["spec"].get("ports") or []:
            target = port.get("targetPort")
            if isinstance(target, int):
                continue  # numeric targetPort cannot be validated without a schema
            available = {
                container_port.get("name")
                for workload in targets
                for container in containers(workload)
                for container_port in container.get("ports") or []
            }
            assert target in available, (
                f"Service {service.name} forwards to named port {target!r}, "
                f"but the selected pods only expose {sorted(available)}"
            )


def test_ingress_backends_resolve_to_services(rendered: list[RenderedObject]) -> None:
    services = {(obj.namespace, obj.name) for obj in objects_of(rendered, "Service")}
    for ingress in objects_of(rendered, "Ingress"):
        for rule in ingress.document["spec"].get("rules") or []:
            for path in rule["http"]["paths"]:
                backend = path["backend"]["service"]
                assert (ingress.namespace, backend["name"]) in services, (
                    f"Ingress path {path['path']} points at missing Service {backend['name']}"
                )


# --------------------------------------------------------------------------- #
# Configuration contract
# --------------------------------------------------------------------------- #
REQUIRED_CONFIG_KEYS = [
    "ENVIRONMENT",
    "LOG_LEVEL",
    "OWL2_MODEL_PATH",
    "OWL2_EXECUTION_PROVIDER",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "DATABASE_URL",
    "RABBITMQ_HOST",
    "REDIS_HOST",
    "MINIO_ENDPOINT",
    "CAMERA_STREAMS",
    "MODBUS_INTERFACE",
]

REQUIRED_SECRET_KEYS = [
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "RABBITMQ_DEFAULT_USER",
    "RABBITMQ_DEFAULT_PASS",
    "RABBITMQ_ERLANG_COOKIE",
    "REDIS_PASSWORD",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
]


def test_app_config_map_exposes_the_application_contract(rendered: list[RenderedObject]) -> None:
    config = next(
        obj for obj in objects_of(rendered, "ConfigMap")
        if obj.name == "industrial-cognition-config"
    )
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in (config.document["data"] or {})]
    assert not missing, f"industrial-cognition-config is missing keys: {missing}"


def test_secret_carries_every_credential_the_workloads_read(
    rendered: list[RenderedObject],
) -> None:
    secret = next(
        obj for obj in objects_of(rendered, "Secret")
        if obj.name == "industrial-cognition-secrets"
    )
    data = secret.document.get("stringData") or secret.document.get("data") or {}
    missing = [key for key in REQUIRED_SECRET_KEYS if key not in data]
    assert not missing, f"industrial-cognition-secrets is missing keys: {missing}"


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_config_map_keys_used_by_the_python_code_are_present(
    overlay: str,
) -> None:
    """`src/owl/inference/config.py` reads these through pydantic BaseSettings."""
    rendered = render(overlay)
    config = next(
        obj for obj in objects_of(rendered, "ConfigMap")
        if obj.name == "industrial-cognition-config"
    )
    data = config.document["data"] or {}
    for key in ("OWL2_MODEL_PATH", "OWL2_EXECUTION_PROVIDER", "OWL2_MAX_BATCH_SIZE"):
        assert key in data, f"{key} is read by src/owl/inference/config.py but not configured"


# --------------------------------------------------------------------------- #
# Hardening
# --------------------------------------------------------------------------- #
def test_no_placeholder_tokens_remain(rendered: list[RenderedObject]) -> None:
    for obj in rendered:
        blob = yaml.safe_dump(obj.document)
        for token in PLACEHOLDER_TOKENS:
            assert token not in blob, f"{obj.identifier} still contains placeholder {token!r}"


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_every_application_container_has_probes_resources_and_a_secure_context(
    overlay: str,
) -> None:
    rendered = render(overlay)
    exempt = {
        # The log collector has no HTTP surface and is intentionally root-owned
        # because it reads /var/log/pods; see the DaemonSet's own comment.
        "alloy",
        # One-shot bucket bootstrap. It retries via `backoffLimit`, and a Job
        # pod has no Service to become Ready behind.
        "mc",
    }
    for workload, container in iter_containers(rendered):
        name = container["name"]
        if name in exempt:
            continue
        assert container.get("resources", {}).get("requests"), (
            f"{workload.name}/{name} has no resource requests"
        )
        assert container.get("resources", {}).get("limits"), (
            f"{workload.name}/{name} has no resource limits"
        )
        security = container.get("securityContext") or {}
        assert security.get("allowPrivilegeEscalation") is False, (
            f"{workload.name}/{name} must set allowPrivilegeEscalation: false"
        )
        assert security.get("privileged") in (False, None), (
            f"{workload.name}/{name} must not be privileged"
        )
        assert (security.get("capabilities") or {}).get("drop"), (
            f"{workload.name}/{name} must drop capabilities"
        )
        if workload.kind == "DaemonSet":
            continue
        # Init containers run to completion before the pod serves traffic, so
        # Kubernetes has nothing to probe.
        if name in {c["name"] for c in _init_containers(workload)}:
            continue
        assert container.get("livenessProbe"), f"{workload.name}/{name} has no livenessProbe"
        assert container.get("readinessProbe"), f"{workload.name}/{name} has no readinessProbe"


def _init_containers(workload: RenderedObject) -> list[dict]:
    return list(workload.document["spec"]["template"]["spec"].get("initContainers") or [])


def test_no_claim_disables_dynamic_provisioning(rendered: list[RenderedObject]) -> None:
    """`storageClassName: ""` is not "use the default" - it is "no class".

    Regression test for a bug that only a real cluster exposes: an empty string
    disables dynamic provisioning, so every claim sat `Pending` forever on
    `FailedBinding: no persistent volumes available for this claim and no storage
    class is set`, and because the pods mount those claims,
    `postgres`/`rabbitmq`/`redis`/`minio` and the vision worker never scheduled.

    Omitting the field is what selects the default StorageClass; an overlay that
    wants an explicit class names one.
    """
    claims: list[tuple[str, dict]] = []
    for obj in objects_of(rendered, "PersistentVolumeClaim"):
        claims.append((obj.name, obj.document["spec"]))
    for obj in objects_of(rendered, "StatefulSet"):
        for template in obj.document["spec"].get("volumeClaimTemplates") or []:
            claims.append((f"{obj.name}/{template['metadata']['name']}", template["spec"]))

    assert claims, "no persistent volume claims were found; did the templates move?"
    for name, spec in claims:
        assert "storageClassName" not in spec or spec["storageClassName"] != "", (
            f"{name} sets storageClassName to the empty string, which disables "
            "dynamic provisioning; omit the field to use the cluster default"
        )


def test_every_pod_runs_as_non_root(rendered: list[RenderedObject]) -> None:
    for obj in rendered:
        if obj.kind not in WORKLOAD_KINDS:
            continue
        pod_security = obj.document["spec"]["template"]["spec"].get("securityContext") or {}
        container_security = [c.get("securityContext") or {} for c in containers(obj)]
        declared_non_root = pod_security.get("runAsNonRoot") is True or any(
            c.get("runAsNonRoot") is True for c in container_security
        )
        if obj.name == "alloy":
            continue  # documented exception: needs root to read /var/log/pods
        assert declared_non_root, f"{obj.kind}/{obj.name} does not declare runAsNonRoot"


def test_stateful_workloads_use_volume_claim_templates(rendered: list[RenderedObject]) -> None:
    """PostgreSQL/RabbitMQ/Redis/MinIO must not be Deployments on bare PVCs."""
    for component in STATEFUL_COMPONENTS:
        workload = by_component(rendered, component)
        assert workload.kind == "StatefulSet", (
            f"{component} is a {workload.kind}; durable state requires a StatefulSet"
        )
        assert workload.document["spec"].get("volumeClaimTemplates"), (
            f"{component} has no volumeClaimTemplates"
        )
        assert workload.document["spec"].get("serviceName"), (
            f"{component} has no headless serviceName"
        )


def test_robot_is_a_singleton_with_recreate_strategy(rendered: list[RenderedObject]) -> None:
    """Two Modbus writers on one RS-485 bus corrupt the protocol."""
    robot = by_component(rendered, "robot")
    assert robot.kind == "Deployment"
    assert robot.document["spec"]["replicas"] == 1
    assert robot.document["spec"]["strategy"]["type"] == "Recreate"


def test_no_autoscaler_targets_the_robot(rendered: list[RenderedObject]) -> None:
    for hpa in objects_of(rendered, "HorizontalPodAutoscaler"):
        target = hpa.document["spec"]["scaleTargetRef"]["name"]
        assert target != "robot", "the Modbus bridge must never be autoscaled"


def test_gpu_request_and_selector_are_consistent(rendered: list[RenderedObject]) -> None:
    for workload, container in iter_containers(rendered):
        resources = container.get("resources") or {}
        gpu_requested = "nvidia.com/gpu" in (resources.get("limits") or {})
        if not gpu_requested:
            continue
        selectors = workload.document["spec"]["template"]["spec"].get("nodeSelector") or {}
        assert selectors.get("nvidia.com/gpu.present") == "true", (
            f"{workload.name} requests a GPU but does not select a GPU node"
        )
        provider = {
            entry["name"]: entry.get("value")
            for entry in container.get("env") or []
        }.get("OWL2_EXECUTION_PROVIDER")
        assert provider == "CUDAExecutionProvider", (
            f"{workload.name} requests a GPU but OWL2_EXECUTION_PROVIDER={provider!r}"
        )


# --------------------------------------------------------------------------- #
# Overlay-specific expectations
# --------------------------------------------------------------------------- #
def test_dev_overlay_stays_lightweight() -> None:
    rendered = render("k8s/overlays/dev")
    kinds = {obj.kind for obj in rendered}
    assert "HorizontalPodAutoscaler" not in kinds, "dev must not require a metrics-server"
    assert "ResourceQuota" not in kinds, "a quota only obstructs local development"
    assert "NetworkPolicy" not in kinds, (
        "dev cannot reach the physical RTSP/Modbus network, so the production "
        "default-deny policies would black-hole the pipeline"
    )
    minio = by_component(rendered, "minio")
    assert minio.document["spec"]["replicas"] == 1, "dev MinIO must be a single server"
    vision = by_component(rendered, "vision")
    assert vision.document["spec"]["replicas"] == 1


def test_prod_overlay_enables_the_full_topology() -> None:
    rendered = render("k8s/overlays/prod")
    kinds = {obj.kind for obj in rendered}
    for expected in ("HorizontalPodAutoscaler", "PodDisruptionBudget", "ResourceQuota", "NetworkPolicy"):
        assert expected in kinds, f"prod is missing {expected}"
    policies = objects_of(rendered, "NetworkPolicy")
    names = {obj.name for obj in policies}
    assert "default-deny-all" in names, "prod must default-deny ingress and egress"
    for component in ("prometheus", "grafana", "loki", "alloy"):
        assert by_component(rendered, component), f"prod is missing {component}"
    minio = by_component(rendered, "minio")
    assert minio.document["spec"]["replicas"] == 4, "prod MinIO is a 4-drive cluster"


def test_monitoring_ingress_paths_are_added_by_the_component() -> None:
    prod = {obj.name: obj for obj in objects_of(render("k8s/overlays/prod"), "Ingress")}
    ingress = prod["industrial-cognition"]
    paths = {
        path["path"]
        for rule in ingress.document["spec"]["rules"]
        for path in rule["http"]["paths"]
    }
    assert "/grafana" in paths, "the monitoring component adds the Grafana path"
    dev = {
        path["path"]
        for obj in objects_of(render("k8s/overlays/dev"), "Ingress")
        for rule in obj.document["spec"]["rules"]
        for path in rule["http"]["paths"]
    }
    assert "/grafana" not in dev, "dev does not ship the monitoring component"


def test_keda_component_renders_the_queue_depth_scaler() -> None:
    rendered = render("k8s/components/keda")
    kinds = [obj.kind for obj in rendered]
    assert kinds.count("ScaledObject") == 1
    assert kinds.count("TriggerAuthentication") == 1
    scaled = next(obj for obj in rendered if obj.kind == "ScaledObject")
    triggers = {trigger["metadata"]["queueName"] for trigger in scaled.document["spec"]["triggers"]}
    assert triggers == {"vision", "vision.priority"}
    # The scaler reads the AMQP URL from the same Secret the workers use.
    auth = next(obj for obj in rendered if obj.kind == "TriggerAuthentication")
    assert auth.document["spec"]["secretTargetRef"][0]["key"] == "RABBITMQ_URL"


def test_exactly_one_vision_autoscaler_is_selectable() -> None:
    """CPU-depth (`hpa-vision-cpu.yaml`) and queue-depth (KEDA) are exclusive.

    Two controllers writing `spec.replicas` for one Deployment flips it every few
    seconds, so `prod` must ship exactly one of them and the swap must be a
    one-line `resources:`/`components:` change - which is why the vision CPU HPA
    lives in its own file rather than inside `autoscaling.yaml`.
    """
    prod = repo_root() / "k8s" / "overlays" / "prod"
    kustomization = yaml.safe_load((prod / "kustomization.yaml").read_text())
    resources = kustomization["resources"]
    components = [entry for entry in kustomization.get("components") or []]

    assert "hpa-vision-cpu.yaml" in resources, (
        "prod must ship the CPU-based vision HPA by default"
    )
    assert "../../components/keda" not in components, (
        "KEDA is opt-in; shipping it by default breaks clusters without the operator"
    )
    # The CPU HPA must not also live inside the shared autoscaling file, or the
    # documented swap would leave two vision HPAs behind.
    shared_hpas = [
        doc["metadata"]["name"]
        for doc in yaml.safe_load_all((prod / "autoscaling.yaml").read_text())
        if doc
    ]
    assert "vision" not in shared_hpas, (
        f"autoscaling.yaml must not declare a vision HPA (found {shared_hpas}); "
        "it would collide with the standalone hpa-vision-cpu.yaml"
    )


def test_generated_secret_is_not_committed_with_hash_suffix() -> None:
    """`secretKeyRef.name` must equal the generator name exactly."""
    base = yaml.safe_load((repo_root() / "k8s" / "base" / "kustomization.yaml").read_text())
    options = base["generatorOptions"]
    assert options["disableNameSuffixHash"] is True


def test_every_manifest_file_is_referenced_by_a_kustomization() -> None:
    """Dead YAML is how the previous dump accumulated unreachable objects."""
    root = repo_root() / "k8s"
    referenced: set[str] = set()
    for kustomization in root.rglob("kustomization.yaml"):
        document = yaml.safe_load(kustomization.read_text())
        for entry in (document.get("resources") or []) + (document.get("components") or []):
            referenced.add(str((kustomization.parent / entry).resolve()))
        for entry in document.get("patches") or []:
            if "path" in entry:
                referenced.add(str((kustomization.parent / entry["path"]).resolve()))
        # Generator inputs (`files:`, `envs:`) are loaded into ConfigMaps and
        # Secrets rather than applied directly, so they count as referenced too.
        for section in ("configMapGenerator", "secretGenerator"):
            for generator in document.get(section) or []:
                inputs = list(generator.get("files") or []) + list(generator.get("envs") or [])
                for entry in inputs:
                    source = entry.partition("=")[2] or entry
                    referenced.add(str((kustomization.parent / source).resolve()))

    orphans = []
    for manifest in root.rglob("*.yaml"):
        if manifest.name == "kustomization.yaml" or "patches" in manifest.parts:
            continue
        body = [
            line for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not body:
            # A comment-only stub is not a manifest.
            continue
        if str(manifest.resolve()) not in referenced:
            orphans.append(str(manifest.relative_to(root)))
    assert not orphans, f"manifests not referenced by any kustomization: {orphans}"


def test_documentation_references_the_real_entrypoints() -> None:
    """doc/k8s.md is the operator contract; it must not point at dead paths."""
    doc = (repo_root() / "doc" / "k8s.md").read_text(encoding="utf-8")
    for expected in (
        "k8s/overlays/dev",
        "k8s/overlays/prod",
        "k8s/scripts/deploy.sh",
        "k8s/scripts/validate.sh",
        "k8s/components/monitoring",
    ):
        assert expected in doc, f"doc/k8s.md does not mention {expected}"
