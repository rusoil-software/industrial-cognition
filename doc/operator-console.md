# Operator Console (Vue 3)

The System User's view of the cell: the camera image with detection overlays, and the
cognition task for each camera - a set of objects, each with its own confidence
threshold.

```
browser  ──<img src=/cameras/{id}/stream>──►  MJPEG preview (what the camera sees now)
         ──<ws  /ws/detections>────────────►  boxes + confidences (what was analysed)
         ──PUT /cameras/{id}/prompt────────►  the cognition task
```

## Why it looks like this

**No build step.** `ui/` is `index.html`, `app.js`, `styles.css` and a vendored
`vendor/vue.global.prod.js` (Vue 3.5, MIT). A Vite project would add a Node toolchain to
the image and a build step to CI for a single view over three endpoints. It is written
so the component can move into a single-file component unchanged.

**MJPEG for video, WebSocket for results.** `multipart/x-mixed-replace` renders in a
plain `<img>` with no JavaScript decoding, no WebRTC and no HLS. Detections arrive
separately on the existing `/ws/detections`, carrying `metadata.frame_number`.

**The overlay is honest about latency.** The preview is "now" and the boxes are "the last
analysed frame" - on CPU that is an ~15 s gap. The UI shows the frame number and the
inference/queue timings instead of pretending the two are synchronised, because
pretending would make a correct system look broken.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /ui/` | the console (static mount) |
| `GET /cameras` | cameras, whether each is capturing, and its queries |
| `GET /cameras/{id}/snapshot` | the last JPEG the inference actually saw |
| `GET /cameras/{id}/stream` | MJPEG preview, 5 FPS |
| `GET /cameras/{id}/stats` | pipeline, throughput and fan-out snapshot |
| `WS /ws/detections` | results as they complete |
| `PUT/GET/DELETE /cameras/{id}/prompt` | the cognition task |

## Running it without a camera

`CAMERA_SOURCE` accepts `replay:/path/to/still.jpg` (`src/camera/sources.py`). The frame
loop, the queue, the worker, the WebSocket and the UI all run for real; only the pixels
are constant, with a small deterministic drift so consecutive frames differ. The source
logs `replaying ... as a camera: frames are a still image, not live video`, and
`GET /cameras` reports the pipeline as running - there is no hidden fallback, and an
unopenable source is reported rather than replaced with fake frames.

The dev overlay runs it in the `camera` Deployment against
`/app/assets/images/roboflow_test_detect_human_robot.webp` at `CAMERA_FPS=0.2`.

**The console must be opened against the pod that captures.** The latest-frame buffer
lives in the capturing process, so `/cameras/{id}/snapshot` and `/stream` answer `503`
on a pod with capture disabled. The `camera` Deployment runs the same application, so it
serves the whole console; either point the ingress at it, or co-locate capture with the
API. Both pods share the database, so a task set from either applies to both.

```sh
kubectl -n cogni-ns port-forward deploy/camera 8080:8000
# then open http://localhost:8080/ui/
```

## What the live run fixed

Four real bugs, none visible from a unit test:

1. **Redis auth.** The ConfigMap's `CELERY_RESULT_BACKEND` has no password while Redis
   runs with `--requirepass`. Only `api` and `vision` overrode it; the `camera` pod - the
   one that submits frames - crash-looped on its first dispatch with
   `redis.exceptions.AuthenticationError`. Now `camera` and `robot` override it too.
2. **In-flight window leak.** Purging the broker queue left two tasks that could never
   return, so `AsyncResult.ready()` stayed false, the window stayed full and *every*
   subsequent frame was shed - a permanent stall from one lost message. The dispatcher
   now expires entries older than `task_timeout_ms` (default 180 s) and counts them as
   failures.
3. **Fractional capture rates.** `CAMERA_FPS=0.2` (one frame per five seconds, which is
   what CPU inference supports) crashed startup because FPS was parsed as `int`, which
   rounds it to zero. `RunnerConfig.target_fps` is now a float.
4. **In-flight sizing.** With 11 s inference and a 64-frame window, the newest frame
   waited ~10 minutes (`queue_ms` 353 s observed). The window must be sized from
   capacity: `PIPELINE_MAX_IN_FLIGHT=2` brought `queue_ms` to 7 ms.

## Verified

Against the live cluster, on the real photograph: console assets served (Vue 158 KB),
`/cameras` reports the camera capturing, `PUT` accepts three queries with thresholds,
`/snapshot` returns a 120 KB JPEG whose last bytes decode, `/stream` speaks multipart,
and `/ws/detections` delivered **5 detections for one frame** - `{cardboard box: 2,
robot: 2, person: 1}` - with `queue_ms` 7 and `e2e` 15 s on CPU.

## Not built yet

No authentication or authorisation on the console or the prompt endpoints (the deployment
has no auth story yet - this is the first thing to add before it faces a network). No
multi-camera view (one camera per page). No historical playback or result persistence in
the UI. No zone/area constraints per query - deliberately deferred.
