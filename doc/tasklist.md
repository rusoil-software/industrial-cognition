# Project Task List: Industrial AI Recognition System

This list tracks all development tasks required to bring the system from blueprint to production. Tasks must be completed sequentially to maintain functional integrity.

**Status Legend:**

* ✅: Completed and verified.
* 🟡: In Progress (Requires active work).
* 🔴: Blocked (Requires external decision or prerequisite task completion).
* ⚫: To Do (Initial task).

---

## 🚀 Development Stages

### **Stage 0: Foundation & Documentation (COMPLETED)**

* ✅ Finalize Core Vision & Tech Stack: (Completed)
* ✅ Define Communication Protocols: (Completed - Modbus/RS-485)
* ✅ Establish Development Conventions: (Completed - KISS, SRP, etc.)

### **Stage 1: Foundational Infrastructure (NEXT FOCUS AREA)**

* ✅ Setup K8s Manifests: Kustomize `base` + `dev`/`prod` overlays covering every microservice, the data
  services, networking, autoscaling and the optional observability/KEDA components. Rendered and
  contract-tested by `tests/k8s` and `.github/workflows/k8s/validate.yaml`; documented in `doc/k8s.md`.
* ✅ Database Schema Migration: Alembic is wired up (`alembic.ini` + `alembic/env.py`), the ORM models in
  `src/api/storage/models.py` define `detection_results` and `system_metadata` per `doc/vision.md`, and the
  initial revision creates both tables plus the `object_type`/`camera_status` enums. Verified against a real
  PostgreSQL: upgrade, full downgrade/re-upgrade round trip, empty second autogenerate, and insert/read-back.
  `pytest tests/db` (12 static + 10 behavioural tests); documented in `doc/database.md`.
* ✅ Base Dependency Management: `requirements/prod.txt` finalised as the union of the runtime dependency sets
  shared by both production images, excluding the `dev`/`export`/`k8s` tiers. The two images still differ in
  exactly one respect — `api` installs `onnxruntime`, `vision` installs `onnxruntime-gpu`, and those cannot
  coexist in one environment. `requirements/api.txt` and `base.txt` gained `psycopg[binary]==3.2.9` to match
  the deployment's `postgresql+psycopg://` DSN.

### **Stage 2: Core Functionality Implementation**

* ✅ Camera Service: Device/RTSP capture implemented in `src/camera/capture.py` (MJPG + 60 FPS negotiation,
  `CAP_PROP_BUFFERSIZE=1` to avoid draining stale frames, JPEG encode at the model's resolution) and unit
  tested. **Remaining:** push results to clients over the FastAPI WebSocket (`/ws/detections`); the payload
  already matches `src/camera/schemas.py::StreamingPayload`. Real device negotiation needs the camera hardware.
* ✅ Vision Service: OWLv2 on ONNX Runtime is implemented end to end and unit tested — preprocessing
  (`src/vision/preprocess.py`), graph decoding (`src/vision/decode.py`: centre→corner, sigmoid, NMS, fp16),
  backend seam (`src/vision/backend.py`) and the per-frame worker (`src/vision/worker.py`). The model contract
  was read from the artifact with `scripts/inspect_model.py`. 72 tests in `tests/vision`.
  *Caveat resolved:* the default prompt is now genuinely CLIP-tokenised (`k8s/dev/text_queries.json`), the
  tokenizer is baked into the API image so prompts can be set offline over HTTP, and the runtime was tuned
  (cgroup-aware thread pool, warmup at worker start): 26.8 s → 10.1 s per frame at the deployed 2-CPU limit.
  Detection *quality* is now verified on a real Apache-2.0 test image
  (`tests/vision/test_detection_quality.py`): 'robot' localises the arm and 'human'/'person' the worker,
  with the two prompts returning different boxes. Multi-object detection is one forward pass with
  every label in the text batch, and each query carries its own threshold, set by the System User
  through `PUT /cameras/{id}/prompt` (`doc/vision-pipeline.md` §15).
* ✅ Robot Interface: Modbus RTU/TCP implemented server-side — `protocol.py` (CRC, framing, PDU builders,
  exception decoding), `transports.py` (RS-485 via pyserial, Modbus TCP, and a PDU-accurate simulated
  controller), `service.py` (bounded retries, typed errors, no retry on device refusal) and
  `integration_manager.py` (detection → millimetres → register units, with STALE_DETECTION / OUT_OF_REACH /
  HUMAN_ON_SITE refusals). 46 tests in `tests/robot`.
  **No hardware yet:** the transport defaults to `simulated`, and the register map / units are placeholders
  to confirm when a converter and the rebuilt robot exist. See `doc/robot-interface.md`.
* ✅ Orchestration: Per-camera prompt API implemented (`src/api/prompts/`: put/get/delete/list at
  `/cameras/{camera_id}/prompt`), tokenised through `src/vision/prompt.py`, stored in
  `system_metadata` via `src/api/storage/prompt_repository.py`, and carried in each frame payload so a
  prompt applies only to its own camera. Multi-worker dispatch implemented and load tested — bounded in-flight window with
  load shedding, capture/dispatch/completion FPS and latency percentiles (`src/pipeline/`), Celery task and
  transport, and the 60 FPS acceptance tests (`tests/pipeline`, 28 tests). **Remaining:** wire the dispatcher
  into the FastAPI app. Done: `src/pipeline/runner.py` owns the capture loop, `src/api/main.py` was rewritten
  around a non-fatal lifespan, and `/health` + `/status` expose the pipeline snapshot.

**Stage 2 throughput note (measured, `doc/vision-pipeline.md` §1):** one OWLv2 pass costs **4053 ms on CPU**
and ~20-40 ms expected on a GPU, so the 60 FPS aggregate target needs **2-3 GPU workers** (or is unreachable
on CPU: 244 workers). The pipeline itself is not the bottleneck; it sustains 60 FPS with adequate workers and
sheds frames with bounded latency otherwise, both asserted by tests.

### **Pre-Stage 3 backlog (agreed 2026-10-17)**

**Status 2026-10-21: 11 of 15 implemented.** Done: 1 (model on PVC), 2 (two workers), 4 (persistence),
5 (readiness), 6 (auth hardening), 8 (queue hygiene), 10 (CI), 14 (live view: tracked boxes,
delete, liveness), 15 (metal-sheet presence: calibration tooling, the `presence` message and the
geometry gate that tells the part from the table), plus the `object_type`
follow-up. Implemented but not yet validated with hardware: 3 (camera discovery + registry -
code, console and tests are in place, no IP camera on the network yet). Open: 7 (ingress/TLS),
9 (multi-camera), 11 (detection quality dataset), 12 (GPU decision - deferred by request after
the CPU presence measurements); 13 (model delivery) is deferred by agreement with the MinIO
replacement. Items 14 and 15 were added while testing item 3, so the original list had 13.

Ordered roughly by what blocks what. Items 1-3 are the ones we named; the rest are gaps this
build exposed that will cost more to fix later than now.

**1. &#9745; Put the model back on the PVC (and fix the loader pod first).** - done 2026-10-20
`k8s/tools/model-loader.yaml` now satisfies PodSecurity `restricted`: `runAsNonRoot`,
`allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, `seccompProfile: RuntimeDefault`,
a read-only root filesystem with an `emptyDir` at `/tmp` for tar, and `fsGroup: 1000` so the
non-root user can write the volume. Without the `fsGroup` the copy fails only at the very end,
after streaming 293 MB.
Verified end to end: model and prompt copied through the loader (2.1 s), `vision` rolled out
`1/1`, the worker loaded the artifact in 1.3 s and warmed in 10.8 s, a real inference returned
sane logits at ~8.5 s/frame with 3434 MiB peak RSS, and the authenticated console path produced
**4 detections on one frame `{cardboard box: 2, robot: 1, person: 1}`** with `queue_ms` 1.7 s.
*Still to do:* seed the default prompt in the installer so a recreated cluster is not a fresh
database with no cognition task.

**2. &#9745; Make the Celery worker lean enough for two.** - done 2026-10-20
Measured with `scripts/measure_footprint.py` (one process per configuration; `VmHWM` never goes
down). The **CPU allocation arena** was the lever: turning it off cuts peak RSS by **41%**,
3465 -> 2042 MiB, at no measurable latency cost in the first measurement (~12% slower in a
second run at 20 threads against a 6-CPU limit - the honest range). Thread count does not affect
memory at all, and disabling memory patterns as well doubled latency for 15 MiB.
`OWL2_CPU_ARENA=1` restores the old behaviour. Verified: `vision` at 2 replicas, both `1/1`,
RabbitMQ reports **2 consumers**, combined RSS ~3.0 GiB (~2.0 GiB each once warmed).
Full table and reasoning in `doc/vision-pipeline.md` §16. Remaining levers, not needed for two:
the 1.55 GB image (`onnxruntime-gpu` dominates and the GPU path needs it) and the architectural
one - a single inference service with thin Celery clients, instead of one model per replica.

**3. &#9744; Find the real camera from the OS and run it end to end.** - implemented 2026-10-20,
hardware validation pending
*Implemented:* `GET /cameras/discover` lists local OpenCV devices (each index tried per
backend and required to deliver a frame) and ONVIF cameras found by WS-Discovery; the
`cameras` table plus `POST /cameras` store a camera with a Fernet-encrypted password
(`CAMERA_CRED_KEY`, separate from `JWT_SECRET`); `POST /cameras/{id}/probe` opens the source,
reads one frame and reports resolution/latency with redacted output; activation is a single
transaction guarded by the partial unique index `ux_cameras_only_one_active`, and the capture
pod follows it by polling (`CAMERA_REGISTRY_WATCH=1`, 5 s) because capture and the API share
only the database. The console gained a **sources** panel: scan, use, register, probe,
activate. Full reference: `doc/camera.md`. 70 new tests (`tests/camera/`, `tests/api/test_cameras.py`).
*Without hardware:* `scripts/rtsp_camera.py` publishes a laptop camera as an RTSP stream
(pure Python, RFC 2435 RTP/JPEG, no `ffmpeg` binary needed) and prints every address it
answers on, so the production RTSP path can be exercised with no IP camera on the network;
`--image PATH` publishes a still frame instead, for a machine whose camera is unusable.
Verified live: OpenCV reads `rtsp://127.0.0.1:8554/cam` at 640x480 and 10.9 fps, the
registry probe reports `{"ok": true, "width": 640, "height": 480}`, the registry smoke passes
15/15 against it, and a camera fed from `--image` produced real rows in `detection_results` -
`robot` 0.275 x10, `person` 0.220 x5 - through capture, dispatch, CPU inference (~11 s/frame)
and the detection sink. Validating this found three bugs unit tests had missed:
`build_frame_source` returned `DeviceFrameSource(config)` for a URL, which **kept the
configured device index and silently ignored the camera address**, so every RTSP camera would
have opened the local webcam; it also rebuilt `CaptureConfig` through `__dict__`, which a
slots dataclass does not have, so the numeric branch raised `AttributeError` (both fixed and
pinned by `tests/camera/test_sources.py`); and RTSP over UDP cannot cross a NAT boundary -
the pod's advertised UDP port is a pod address the host cannot route to - so the same stream
that reads fine on the host produced **no frames at all** from inside a pod. Capture now
negotiates **TCP interleaved by default** (`RTSP_TRANSPORT=tcp`, override `udp`), which
delivered every frame.
*Not yet done:* a real IP camera end to end. There is no IP camera on this network and the
laptop camera is not visible to Docker Desktop/WSL, so the realistic route stays "publish
RTSP from the host, register the URL" - which is now one form submission instead of an env
change and a redeploy.
*Done when:* real frames reach the console with overlays, at a documented resolution/FPS.

**4. &#9745; Persist detections.** - done 2026-10-20
The table existed and nothing wrote to it. Now the capture loop hands every completed frame
to `src/api/detection_sink.py`, which decides whether it becomes rows; the writer is
`src/api/storage/detection_repository.py`. Sampling is deliberate and documented: failed
frames never persist, empty frames are skipped by default, and at most one frame per camera
per `PERSIST_MIN_INTERVAL_SECONDS` (1 s) - 60 FPS would otherwise be a disk-filling bug, and
most frames say the same thing. This is *history*, not an audit log; an event shorter than
the interval can be missed.
Two schema gaps had to be closed first, with migration `c4e7a1f90b12`: `object_type` is a
four-value enum (`metal_sheet`, `detail`, `debris`, `clutter`) that predates free-form prompts,
so new `label` (the operator's text) and `camera_id` (indexed with `timestamp`) columns were
added. Verified live: 27 rows for camera `main` with `label` values `robot`, `person`,
`cardboard box`, coordinates, confidences and frame ids preserved.
*Both follow-ups are closed.* `object_type` became free text in migration `d5f8b2a1c9e3` and was
backfilled from `label` - verified: 244 rows that were all `clutter` now read `cardboard box`
116, `robot` 75, `person` 58, none left as clutter. Retention is the daily CronJob in
`k8s/base/cronjob-retention.yaml` (item 8's section and `doc/database.md`).

**5. &#9745; Make readiness honest.** - done 2026-10-20
`/readyz` answers "should traffic come here", `/health` stays liveness. The policy is a pure
function (`src/api/readiness.py`) parameterised by role, because the answer differs: a capture
pod without capture is useless, while the API serves the console, prompts and history and
holds the page that explains the fault - taking it out of the endpoints would hide the
diagnosis. `READY_REQUIRE_PIPELINE` selects the role: `1` on the camera Deployment, `0` on API
and robot. The database is always required. Verified: `/readyz` 200 with
`{"database": true, "pipeline": true}`, liveness still `/health`, readiness now `/readyz`.

**6. &#9745; Harden auth before anything faces a network.** - done 2026-10-20
Done: credentials are no longer committed (`secrets.env.example` is tracked, the real file is
generated with random values by the installers and git-ignored); `DOCS_ENABLED=0` removes
/docs, /redoc and /openapi.json in the prod overlay; `AUTH_COOKIE_SECURE=1` and a shorter
`AUTH_TTL_SECONDS` are set there too. A login throttle now blocks a caller after `AUTH_MAX_FAILURES` (5) failed attempts for
`AUTH_LOCKOUT_SECONDS` (300), on both the POST and the token-in-URL form - verified live:
4x401 then 429, and the correct token refused while locked. Its state is per process, so the
effective limit multiplies by the replica count; that is documented rather than hidden.
Remaining, not blocking: per-user accounts and roles (today one shared deployment token) and
a rotation procedure that changes the database password on both sides at once.
Dev credentials are committed in `k8s/base/secrets.env`; `AUTH_TOKEN`/`JWT_SECRET` must be
rotated, `AUTH_COOKIE_SECURE=1` behind TLS, token lifetime decided, and `/docs` +
`/openapi.json` closed in production. Today it is one shared deployment token - per-user
accounts and roles are a later step, but the shared token must not survive into a shared
network.

**7. &#9745; Real ingress and TLS.** - done 2026-10-20
The cluster has no route out, so the certificate comes from a **local internal CA**:
`scripts/tls-selfsigned.sh` / `.ps1` create `industrial-cognition-tls` (served by the ingress) and
`industrial-cognition-tls-ca` (the trust anchor). A CA rather than a self-signed leaf, so
operators import one file and renewals are invisible. `k8s/components/tls/cert-manager` holds the
equivalent chain for sites that already run cert-manager - deliberately not referenced by any
overlay, because it needs their CRDs.
Verified live: HTTP/2 200 with `ssl_verify_result=0` trusting only the internal CA, chain issued
by `IndustrialCognitionInternalCA`, and exit 60 without it. The dev overlay still uses a NodePort
because no DNS name exists in a laptop cluster; production terminates TLS at the ingress.

**8. &#9745; Broker-side queue hygiene.** - done 2026-10-20
Both work queues now carry `x-max-length=256`, `x-overflow=drop-head`, a 60 s `x-message-ttl`
and a dead-letter exchange. Queue arguments are immutable, so `scripts/queue-hygiene.sh --force`
deletes them once and the consumers recreate them - verified live: `vision` carries all five
arguments and `vision.priority` matches. The dead-letter queue needed an explicit
`worker_ready` declaration because RabbitMQ drops messages dead-lettered into a missing
exchange, silently; verified by letting a frame expire with no consumer: `vision` 2 -> 0 and
`vision.dlq` 0 -> 3.

**14. &#9744; mTLS between service pods.** Deferred until the cluster has a CNI that can
enforce it - there are no NetworkPolicies in the namespace yet, so mTLS would protect against a
threat the network does not restrict. Needs a certificate per workload and enforcement
(a service mesh, or cert-manager plus policy). Discussed 2026-10-20.

**15. &#9744; Secrets manager (Vault / OpenBao).** `secrets.env` is generated per checkout and
git-ignored, which is adequate for one cluster. A manager becomes worthwhile when several
environments share credentials and rotation has to be auditable. Discussed 2026-10-20.

**9. &#9744; Decide multi-camera shape.**
The runner is one source per process; per-camera prompts already exist but queues, task routing
and stats assume a single capture loop per deployment. Either run one capture Deployment per
camera (simplest, needs per-camera services/ingress paths) or teach the runner multiple sources
with per-camera windows and stats. Also per-camera `PIPELINE_MAX_IN_FLIGHT`, which must be sized
from inference latency, not left at a default.

**10. &#9745; Make the deployment itself testable in CI.** - done 2026-10-20
`.github/workflows/ci.yml`: unit tests plus a compile-all pass and syntax checks for the
PowerShell and shell entry points; a manifests job that renders dev/prod/tools, validates them
with `kubeconform -strict` and runs the contract tests; and a manual cluster job that builds
both images, deploys to kind, migrates through a Job and runs `scripts/smoke_auth.py`. The
cluster job is gated because it takes minutes - it is the one that would have caught the auth,
readiness and stream bugs found by hand.

**11. &#9744; Detection quality beyond one photograph.**
One Apache-2.0 image proves prompt conditioning; thresholds are still chosen from a handful of
scores (`doc/vision-pipeline.md` §14). Build a small ground-truth set of the real workcell per
camera and tune per-query thresholds against it, and use
`POSE-Lab/IndustryShapes` (already noted in `assets/images/README.md`) for a broader evaluation
before Stage 3 claims quality numbers.

**12. &#9744; Decide where the GPU sits.**
The prod overlay requests `nvidia.com/gpu` and the worker image installs `onnxruntime-gpu`, but
nothing has ever run on a GPU: every measurement is CPU at ~11 s/frame. Before performance work
in Stage 3, confirm a GPU runtime actually exists (`CUDAExecutionProvider` visible in the pod)
or state plainly that 60 FPS is out of scope for now.
*Decision 2026-10-21 (owner):* **do not pursue GPU acceleration now.** The development rig is an
RTX 4060 Laptop (8 GB) and the test rig an RTX 5060 (8 GB); both remain unused by the vision
path. The reason is visible in item 15: presence of the real sheet is answerable on CPU once the
prompt is chosen and the threshold calibrated, so GPU work would buy latency the station does
not need yet. Revisit when multi-camera throughput (item 9) or sub-second tracking forces it,
which is also when the 8 GB budget becomes the constraint worth measuring.

**13. &#9744; Model delivery stays manual, by agreement.**
The artifact lives on a PVC and is copied by hand. Revisit together with the MinIO replacement
(Stage 4) so the worker pulls it at startup with checksum verification instead of `kubectl cp`.

**14. &#9745; Make the live view trustworthy: tracked boxes, a delete button, honest liveness.** - done 2026-10-20
Three complaints, one theme: the console could not be believed. *Added while testing item 3,
not in the original 13.*

* **The RTSP client dropped after ~30 s.** Root cause was `_accept_loop` setting a 30-second
  socket timeout on an RTSP control connection that is idle by design after `PLAY`: `recv`
  raised, the session was torn down, the stream closed, and the publisher printed
  `0 client(s)` while the camera was still publishing. Fixed by keeping idle sessions, giving
  every client its own writer thread with a single-slot mailbox (the older frame is dropped,
  never queued), a shared write lock for RTSP replies and interleaved RTP, and
  `shutdown()` before `close()`. Verified: one client held `rtsp://127.0.0.1:8554/cam` for
  75 s and 711 frames, where it used to die at 30 s. The heartbeat now separates the two
  failures it used to conflate: `last frame 12.4s ago` is a silent **camera**, `0 client(s)`
  with `last frame 0.1s ago` is just nobody watching, and each client gets its own line.
* **No way to remove a camera.** `delete` per row in the sources panel calls the existing
  `DELETE /cameras/{id}`; it deletes the registered row *and* the detection task, stops
  capture when the camera is active (or is the one being captured), and clears the stream so
  a deleted camera does not keep showing its last frame. Front end and back end were both
  fixed for the follow-up report that entries survived deletion ("they remain after
  deletion and when attempted to delete again, it says 'no camera'"): `GET /cameras` lists
  the union of registry rows and tasks, so the delete has to remove both - and the console
  now drops the row locally on success instead of waiting for a refresh.
* **A frozen snapshot looked live.** The preview is now a real stream at `STREAM_FPS` (10)
  that ends after `STREAM_IDLE_TIMEOUT` (15 s) without a new frame, and the top bar carries a
  live badge fed by `/status` every 2 s: `live · 0.4s`, `stale · 12.1s`, `camera offline`,
  `waiting for a frame`, `capturing another camera`. The staleness budget is cadence-aware
  (`max(5 s, 3 intervals)`), so a 0.2 FPS dev replay is not called dead. `/cameras` entries
  carry `last_frame_age_seconds` and `streaming`.
* **"One frame in more than 10 seconds, so the movement practically does not exist."** A live
  badge does not make a slideshow watchable: the dev capture cadence was `CAMERA_FPS=0.2`
  (5 s per frame) and the replay source drifted by ±2 px, invisible at 640×960. The dev
  overlay now captures at 5 FPS and `ReplayFrameSource` pans 24 px with a 60-frame period
  (`drift_pixels`, `PAN_PERIOD_FRAMES`), so consecutive frames differ visibly while the
  bounded in-flight window keeps shedding the surplus. Inference latency is unchanged; what
  changes is that the preview is 5 FPS and a still image looks like a slow pan.
* **Follow-up report (same day): dropped frames, duplicate boxes, contradicting boxes.**
  "Lessen the amount of frames dropped", "reload the page to get tracking results", "two
  trackers instead of one", "dotted-line frames that do not coincide with the solid ones".
  - `dropped_capacity` was measuring the cadence mismatch, not a fault: 5 FPS capture against
    ~11 s CPU inference refuses nearly every frame by design. `PIPELINE_DISPATCH_INTERVAL_SECONDS`
    (dev `10`, base `0`) now offers workers one frame per interval and counts preview-only
    frames as `shed_sampling`, so `dropped_capacity` rising again means a real problem.
  - Duplicate boxes: matching used overlap only, so an object moving further than its own width
    between two inferences started a second track. Detections now also match the nearest
    same-label track within 25 % of the frame diagonal (`match_distance`), and two tracks that
    end up on one object (IoU ≥ 0.6) collapse into the older id (`merged` in `/status`).
  - Dashed detection boxes were one whole inference old, so they could never agree with the
    live frame; they are no longer drawn when tracks exist (drawn solid when tracking is off).
  - "Needs a reload": the console's `app.js` is served from the image and was being cached by the
    browser. `/ui` responses now carry `Cache-Control: no-cache, must-revalidate`, so a rebuilt
    console is picked up on the next load instead of a hard reload.
  Tests: 4 tracking cases (distance gate, disabled gate, collapse, label separation) and 2 runner
  cases (sampling vs. zero interval) added.
* **Boxes moved once per inference.** On CPU that is once per ~11 s, so a walking worker's
  box teleported. `src/vision/tracking.py` seeds one OpenCV tracker per detection and
  advances them at `TRACK_FPS` (10) between inferences, publishing
  `{"type": "tracks"}` on the same WebSocket; the console draws tracks as the primary box.
  KCF is
  the default (CSRT is several times more expensive per update), fallback kcf → csrt → mil,
  and a missing tracker, a bad `TRACKER_TYPE` or a tracker exception disables the overlay
  without touching capture or detection. `api.txt` moved to
  `opencv-contrib-python-headless` (same version) for the trackers.
  Verified live in minikube: `tracking {enabled: true, tracker: "kcf", tracks: 4, seeded: 7,
  updated: 23, lost: 0}` and 6 `tracks` messages in 30 s carrying `track_id`, `object_type`,
  normalised geometry, `age` and `stale` for `robot` 0.2745 / `robot` 0.2687 / `person` 0.2197.
  Tracking is an aid, not measurement: rows in `detection_results` still come from inference.
* Tests: 15 new (`tests/pipeline/test_runner_tracking.py`, `tests/api/test_stream_freshness.py`,
  `tests/vision/test_tracking.py`, 5 lifecycle tests in `tests/camera/test_rtsp_server.py`),
  plus 4 tracking cases and 3 runner cases for the follow-up; suite 543 passed / 13 skipped.
  Reference: `doc/camera.md` (console, liveness, RTSP keepalive)
  and `doc/vision-pipeline.md` §18.
* Verified live after the follow-up (2026-10-20 22:52, dev): `captured 463` in 93 s (5 FPS),
  `dispatched 6`, `dropped_capacity 1`, `shed_sampling 456`, inference p50 13.0 s, tracking
  `{tracks: 1, seeded: 4, updated: 463, merged: 0, match_distance: 0.25}` - one box per object,
  and a drop counter that no longer reports the intended cadence as a fault. `/ui/app.js`
  answered `cache-control: no-cache, must-revalidate`, so a rebuilt console needs no hard reload.

**15. &#9745; Presence for the real metal sheet: calibrate before trusting the default.** - done 2026-10-21
The test deployment needs the station question answered - "is the blank on the table or not" -
on a real photograph rather than a warehouse stock image. *Added after the owner provided
`assets/images/metal_sheet_test_image.jpg`. It also settles the GPU question for now: everything
below is CPU, and it is enough.*

* The photograph is a brushed-metal sheet on a white table in a tiled hallway, i.e. exactly as a
  fixed station camera would see it. Top-box scores were measured for 10 phrasings; `steel blank`
  localises the sheet (centre `0.483, 0.588`, size `0.590 x 0.421`) at **0.2459**, `metal blank`
  at 0.2448, the committed `metal sheet` at 0.1564. Absence crops of the same photograph (doorway,
  wall, bare table) stay at or below 0.0465.
* **Consequence:** the shipped default threshold of 0.25 reports "no part" for a sheet that fills
  half the frame. The threshold therefore belongs to the camera; `PUT /cameras/{id}/prompt`
  already supports one per query.
* **Limitation, measured not assumed:** the tiled floor scores 0.19 (`steel blank`), so no
  threshold separates floor from workpiece - the camera must be aimed at the station, and a zone
  constraint (item 4) would make this robust. Cropping tighter makes scores *worse* (0.1235 for
  the table crop, 0.0569 for the sheet alone), because this export uses surrounding context.
* Added `scripts/calibrate_presence.py` (photographs in, threshold out, refuses to print one when
  the groups overlap by less than `2 x margin`), prompt fixtures `assets/prompts/steel_blank.json`
  and `metal_blank.json`, `tests/vision/test_metal_sheet_presence.py` (6 model-marked assertions:
  localisation, absence crops, separation, prompt choice) and
  `tests/scripts/test_calibrate_presence.py` (12 fast tests).
* Verified live in the dev cluster: camera `metal-sheet` on
  `replay:/app/assets/images/metal_sheet_test_image.jpg`, prompts `steel blank` and `metal blank`
  at `0.15`, produced `detection_results` rows `steel blank` 0.202-0.222 and `metal blank` 0.208
  with box centre `(0.48, 0.59)` - the sheet, at a threshold the defaults would have rejected.
  The console lists it active with its two queries; `main` (warehouse replay) stays registered and
  inactive, so switching back is one click.
* Still open for the real station: repeat the calibration with the owner's own photographs of the
  test rig (the tool is the workflow; one photograph is not evidence), and wire the presence
  answer to the robot in Stage 3. Reference: `doc/vision-pipeline.md` §19 and `doc/camera.md`
  "Presence at a station".
* **Follow-up the same day - the best box was the table.** Looking at the console, the owner
  pointed out that the detection finds the *table* the sheet lies on, and the sheet itself only as
  a smaller second box (`steel blank` 0.2459 on the table vs **0.1594** on the part, 90 % inside
  the sheet). Ten rephrasings later (`stainless steel sheet`, `brushed steel plate`,
  `reflective metal plate`, `mirror`, ...) it was clear that phrasing cannot promote the part:
  a brushed sheet, a table top and a tiled floor are the same kind of flat light rectangle. The
  answer is **geometry, not language** - `src/vision/presence.py` gates the existing boxes by
  station region, area, aspect and the calibrated confidence, and answers the station question
  from them. Measured on the real photo the gate refuses 7 of 8 boxes and keeps exactly the part
  (`area 0.2484 > 0.2000 (a table, not the part)`, `only 36% inside the station region`,
  `confidence 0.097 < 0.100`). The runner publishes `{"type": "presence", ...}` per inference,
  `/status` carries `checks`/`present`/`errors`/`last_reason` and the gate in force, and the
  console shows a `part present` / `no part` badge. A failed inference publishes nothing, so a
  worker timeout cannot read as "the part is gone". Config is env-only (`PRESENCE_*`, off by
  default) because the region and size band belong to one physical stand; the dev overlay carries
  this stand's measured values. Tests: `tests/vision/test_presence.py` (45) and
  `tests/pipeline/test_runner_presence.py` (11). Reference: `doc/vision-pipeline.md` §19.2.

* **Follow-up after the first live run - the badge flapped.** Deployed as written, presence
  alternated `part present` / `no part` on every inference. Two causes, both found by measuring
  the frames the worker actually sees across the replay pan: the user's `steel sheet` prompt at
  0.1 only ever produced the *table* box (the part needs `steel blank` at ~0.08), and on the old
  gate the part's own score moves 0.085-0.125 across the eight pan phases while
  `PRESENCE_MIN_CONFIDENCE=0.10` cut exactly the low phases. Lowering the floor alone let the
  **wall behind the stand** in (0.070-0.087, 86 % inside the region - its score overlaps the
  part's). The fix is to make geometry, not confidence, the discriminator: region top edge
  `0.42` (below the wall band) and `PRESENCE_MAX_ASPECT=2.0` (wall 2.27, part 1.31). All eight
  phases now accept the part and the table/floor/wall boxes of the pan are all refused.
  `tests/vision/test_presence.py` pins the eight phases and the wall case. Reference:
  `doc/vision-pipeline.md` §19.2 "Why the confidence floor is 0.05".

### **Stage 3: Polish & Hardening**

* ✅ Operator console: Vue 3 single-page UI (ui/, no build step) served at /ui/ with a
  live MJPEG preview, detection overlays and per-camera cognition tasks (one query per
  object, each with its own threshold). Backend: src/api/stream.py,
  src/camera/sources.py (including a still-image replay source so the whole pipeline
  runs without a camera). See doc/operator-console.md. Missing before any network
  exposure: authentication.

* 🟡 End-to-End Testing: Write and execute full integration tests covering the entire pipeline (Camera -> Vision -> Modbus).
  **Live deployment done (2026-10-17):** the whole stack runs on minikube — all pods `Running`, schema migrated,
  service discovery, prompt repository against real Postgres, and the full
  API → RabbitMQ → vision worker → Redis task path verified with a real JPEG (0.30 s, 2 detections).
  Six deployment bugs were found and fixed; see `doc/k8s.md` "Live run". Remaining: real camera/OWLv2/Modbus
  hardware, and automated assertions rather than a smoke script.
* 🟡 Observability Integration: Hook up Prometheus metrics scraping endpoints to all service checkpoints.
  The Kubernetes side is already in place (`k8s/components/monitoring` scrapes the
  `prometheus.io/scrape` annotations on every Deployment); what remains is the `/metrics` endpoint itself in
  the FastAPI app and the vision worker, plus a Redis exporter for `redis_exporter`. The commented jobs in
  `k8s/components/monitoring/prometheus/prometheus.yml` are the exact targets to enable.
* 🟡 Deployment Automation: Finalize Helm chart package for production deployment.
  Superseded in practice by the Kustomize overlays (`doc/k8s.md` §1 explains why Helm is deferred); revisit
  only if a Helm-native consumer appears.

### **Stage 4: Tracked Issues (deferred until the MVP runs end to end)**

* 🔴 **Object-storage backend: replace MinIO.** `minio/minio` and `minio/mc` no longer resolve on Docker Hub or
  Quay (verified with `docker manifest inspect`; `alpine`/`redis`/`postgres`/`rabbitmq` all resolve, so it is
  the upstream repository, not the network). The manifests currently use `bitnamilegacy/minio:latest`, an
  explicitly unmaintained archive repository — a stopgap, not a target. Evaluate **SeaweedFS**, **Garage**,
  **RustFS**, **Ceph RGW** or **Versity Gateway**, roughly in that order.
  *Scope:* only `k8s/base/statefulset-minio.yaml`, the `minio-bucket-init` Job and the credential key names
  change. The application contract is the S3 API plus `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` /
  `MINIO_SECRET_KEY` / `MINIO_BUCKET_*`, so no service code is affected. See `doc/k8s.md` §2
  "Object storage (S3)".
* 🔴 **RabbitMQ HA.** The broker is a replicated-single StatefulSet. True multi-node clustering needs the
  RabbitMQ Cluster Operator; see `doc/k8s.md` §2 "RabbitMQ clustering".
* 🔴 **Image delivery, and a model+runtime container.** Two linked questions, both deferred until the MVP runs
  end to end (they depend on items 12 and 13).
  * **Publish the images.** Every deployment today builds `api` and `vision` locally (`docker build` plus
    `minikube image load` / `kind load docker-image`). That is fine on a workstation and wrong for a line with
    several nodes and no outbound access: publish to Docker Hub (public) or a private/internal registry, tag by
    commit, and **pin digests** in the overlays. Digest pinning is the part that matters most — `:dev` / `:latest`
    cannot say what is actually running, which is exactly the question an incident asks.
    *Scope:* `scripts/build-images.sh`, the `image:` fields (or a per-overlay kustomize `images:` block), both
    installers and the CI cluster job.
  * **One verified image for model + runtime.** The worker is currently `python:3.11-slim` + pip-installed
    `onnxruntime` (CPU) + a 293 MiB ONNX artifact on a PVC: one session peaks at ~2 GiB with the arena off and
    CPU inference runs 8.5-11 s/frame (`doc/vision-pipeline.md` §13, §16). Two directions to evaluate:
    (a) base the worker on a **vendor-verified runtime image** (ONNX Runtime GPU / TensorRT containers) instead
    of pip-installing a runtime and hoping it matches the hardware, and (b) build a **model image** — the model
    plus runtime baked in, optionally including a TensorRT engine built for the target GPU. That would remove the
    model PVC, the loader Job and the "is the artifact the right one?" class of failure entirely, and should
    reduce per-session RSS and speed up load beyond the measured 1.3 s + 10.8 s warmup.
    *Unknown until measured:* whether a GPU image actually lowers latency on the target hardware — that is item
    12's question, this item is "with what, once we know where". A model image also gives item 13 a different
    answer than MinIO: nothing to deliver if the model ships inside the container, at the cost of a larger image
    that a registry makes cheap to cache and expensive to rebuild blindly.

### **Fixed while doing item 1**

**Result handling in the dispatcher.** A lost or failed task surfaced as an *exception object*
from Celery's `AsyncResult.result`. That object is truthy, so it passed the empty-result check and
`FrameResult.from_wire` died on `payload.get` - and the `AttributeError` escaped `collect()` into
the capture loop, killing the capture thread. The pod stayed `Running` with
`pipeline.running: false`, two frames stuck in flight and no detections, which is exactly the kind
of failure that looks like "the model is slow" from the outside. `_finalise` now recognises a
non-dict payload and records it as a failure with the exception's type and message, and the
runner guards dispatch and collection separately so an unexpected error costs one frame instead
of the loop. `loop_errors` is exposed in `/status`, so "running but degrading" is visible.
Two regression tests cover the exception object and an arbitrary string result.
