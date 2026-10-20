# Camera sources, discovery and the registry

<!--
  Copyright (c), The Rusoil Software Development Team. All rights reserved.
  Licensed under the Apache License, Version 2.0.
-->

How the service finds a camera, remembers it, and points capture at it. Three source
kinds are supported, and they differ only in how the address is opened:

| kind | example | notes |
| --- | --- | --- |
| `device` | `0`, `1` | a USB/built-in camera by OpenCV index, on the host that runs capture |
| `network` | `rtsp://10.0.0.5:554/stream1`, `http://…/video.mjpg` | IP cameras, RTSP servers, MJPEG URLs |
| `replay` | `replay:/app/assets/images/scene.webp` | a still image used as a fake device; the default on a dev cluster |

`rtsps`, `https` and `http` are accepted for network sources. Anything else is rejected
with a message naming the schemes that are understood.

## Finding a camera

`GET /cameras/discover` probes both directions at once:

* **local devices** - each OpenCV index from `0` to `max_index` is opened per backend
  (`CAP_DSHOW`/`CAP_MSMF` on Windows, `CAP_V4L2` on Linux) and must deliver a real
  frame to be reported. A device that opens but returns nothing is not listed.
* **network cameras** - a WS-Discovery (ONVIF) `Probe` is multicast to
  `239.255.255.250:3702` and the `ProbeMatch` replies are parsed. The reported address
  prefers `rtsp://` over the device-service HTTP URL, because the latter is not a stream.

Both halves never raise: an empty list is a valid answer. Discovery is a *hint*, not a
requirement - most cameras work with a typed RTSP URL even when ONVIF is disabled, and
multicast does not cross VLANs, so a camera on another subnet will not answer.

```bash
curl -s -H "Cookie: ic_session=…" \
  'http://localhost:30080/cameras/discover?include_network=true&timeout=2.0' | jq
```

`credentials_configured` in the response says whether the server can store a password
(see below); the console shows a warning when it is `false`.

## Registering a camera

`POST /cameras` is register-or-update, keyed by `id`:

```json
{
  "id": "door-cam",
  "label": "Workshop door",
  "source": "rtsp://10.0.0.5:554/stream1",
  "username": "operator",
  "password": "s3cret",
  "activate": true
}
```

* `id` is a slug (`[a-z0-9-]`, at most 64 characters). It is also the prompt key: the
  task in the console is stored per camera id.
* Credentials are optional and independent of the URL. Pasting
  `rtsp://operator:s3cret@10.0.0.5:554/stream1` also works: the userinfo is split out,
  stored separately, and the stored `source` stays clean.
* `activate: true` makes this camera the active one and restarts capture on it. The row
  is committed *before* the restart, so a camera that fails to start still exists and
  can be fixed instead of retyped.
* A device source cannot be given a username and password - that is a `422`, not a
  silently ignored field.

## Passwords at rest

Passwords are encrypted with Fernet (`cryptography`) using `CAMERA_CRED_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

* The key must be a 32-byte urlsafe-base64 Fernet key. It is deliberately **separate
  from `JWT_SECRET`**: rotating session signing must not make every stored camera
  password unreadable, and vice versa.
* Without the key the service still starts (the Secret entry is `optional: true`);
  listing and activating cameras that need no password works, and storing a password
  answers `503` naming `CAMERA_CRED_KEY`.
* A password encrypted under a different key fails to decrypt with a message about
  rotation - that is a `503`, not a crash and not a silent empty password.
* What this protects: the database, its backups and dumps. It does not protect against
  an attacker who can read the pod's environment.

The API never returns a password. `GET /cameras` shows `has_credentials: true` and a
redacted source (`rtsp://operator:****@10.0.0.5:554/stream1`).

## One active camera

The `cameras` table has a partial unique index, `ux_cameras_only_one_active`, so "two
cameras are active" is impossible rather than merely unlikely. Activation is a single
transaction: clear the old flag, set the new one.

The active camera is what the **capture process** follows. In production capture runs in
the `camera` Deployment, not the API pod, and the two share only the database - so the
camera pod polls it (`CAMERA_REGISTRY_WATCH=1`, every
`CAMERA_REGISTRY_POLL_SECONDS`, default 5 s) and restarts capture when the active row
changes. A restart that fails is retried on the next poll instead of being remembered as
done, so a camera that was busy when the operator pressed "activate" recovers by itself.
The restart is a stop-then-start because one process owns one device: a USB camera that
allows a single open would otherwise refuse the new one with "device busy".

Polling rather than messaging is deliberate. The state that matters is one row, a restart
is seconds of work, and a poll that misses a change catches it on the next tick; a queue
would add a broker hop and a new failure mode (activation succeeded, notification lost)
for a latency saving nobody can see.

## Probing before you commit

`POST /cameras/{id}/probe` (and `POST /probe` for an unregistered address) opens the
source, reads one frame, and reports the resolution and latency:

```json
{ "ok": true, "source": "rtsp://10.0.0.5:554/stream1", "width": 1920, "height": 1080,
  "latency_ms": 412.7, "detail": "frame read in 412.7 ms", "camera_id": "door-cam" }
```

A probe opens the source explicitly and requires a frame - the same two steps capture
takes. That matters: a replay source decodes nothing until it is opened, so a probe that
only read would call a working camera broken. Typical failures:

| detail | meaning |
| --- | --- |
| `the source did not open: …` | wrong address, wrong credentials, codec not supported, or a missing replay file |
| `the source opened but produced no frame within the read timeout` | the stream path is wrong (often `stream2` where the camera expects `stream1`) or the camera is busy |

Probe output is redacted. Nothing logs a password.

## The console

The camera picker now has a **sources** button. It opens:

1. **scan for cameras** - the discovery results, each with a `use` button that fills the
   form below.
2. **register form** - id, label, source, user, password, `activate now`.
3. **registered cameras** - kind, redacted source, `active` and `password stored` badges,
   plus `probe`, `activate` and `delete` per row.

Registering a camera does not create a task; set the prompts in the usual editor for that
camera id afterwards.

`delete` forgets the camera **and** its detection task: the console lists registered cameras
and configured tasks together, so deleting only the camera row used to leave the entry on
screen - built from the prompt - and the next delete answered `no camera '<id>'`. Both rows
go now, and an id that has neither is a 404. Deleting the active camera (or the one the
running pipeline is capturing) stops capture first, and deleting a camera you are watching
clears the stream, because a dead capture would otherwise leave the browser showing its
last frame forever. Re-registering the same id starts from an empty task.

### Is the camera still there?

An MJPEG stream is a sequence of stills, and a browser keeps the last one on screen when the
stream stops. So the console does not rely on the image to answer "is it live?":

- the **live badge** in the top bar comes from `GET /status` every 2 seconds and shows the
  age of the newest captured frame: `live` under 5 s, `stale` beyond that, `camera offline`
  when capture is not running, `waiting for a frame` when it runs but has produced none, and
  `capturing another camera` when the pipeline follows a different camera than the one
  selected. The same interval carries the tracking state (`tracking kcf` or `tracking off`).
- `/cameras` entries carry `last_frame_age_seconds` and `streaming`.
- `/cameras/{id}/stream` closes the response after `STREAM_IDLE_TIMEOUT` (15 s) without a new
  frame, so the image does not pretend to be live; the console re-opens it after 5 seconds.

### Boxes between inferences

On a GPU-less host one inference takes seconds, which makes detection boxes look frozen on a
moving object. The capture process therefore runs a per-camera *tracker pool*: every
inference seeds an OpenCV tracker per detection, and the trackers are advanced at `TRACK_FPS`
(10) on the newest frames.

- `src/vision/tracking.py` publishes `{"type": "tracks", ...}` messages that carry the same
  normalised geometry keys as detections plus `track_id`, `age` and `stale`.
- The console draws tracks solid (dashed only when the tracker lost its object). Once tracks
  exist it does **not** also draw the last inference: that box is one inference old (seconds
  on CPU) and painting it over the live frame produced two boxes that never agreed, which
  looked like a bug in the tracking. Without tracks — no tracker installed, tracking disabled,
  or nothing detected yet — the last inference is drawn solid with its confidence, because
  then it is the only thing there is to show.
- An object that moves further than its own width between two inferences is still the same
  object: a fresh detection matches an existing track by overlap (IoU >= 0.3) **or** by centre
  distance (within 25 % of the frame diagonal, `match_distance`). Without the distance rule the
  old box stayed on screen while a new one appeared next to it, so a fast object ended up with
  two trackers once it stopped.
- Two same-label boxes that end up on top of each other (IoU >= 0.6) are collapsed into the
  older `track_id` (`merged` in `/status`). Two boxes on one object are worse than one, since
  the operator cannot tell which to trust.
- The seed frame is the *newest* frame, not the one the detection came from: on CPU the
  detection is seconds old, and seeding a stale frame would put every box in the wrong place.
- Tracker choice is `TRACKER_TYPE` (`kcf` by default, then `csrt`, then `mil`). KCF is the
  default because CSRT costs several times more per update and this runs next to inference.
  With no tracker installed, `TRACKING_ENABLED=0`, or a bad name, capture and detection keep
  working and the console says `tracking off` with the reason in `/status` (`tracking.reason`).
- The trackers live in `opencv-contrib-python-headless`, which is why `requirements/api.txt`
  pins the contrib build of the same OpenCV version.

Tracking is an operator aid, not measurement: `detection_results` rows come from inference
only. When a track and a detection disagree, the detection is the answer.

## Publishing a laptop camera as RTSP

Docker Desktop and WSL cannot see a USB camera, so a laptop camera cannot be handed to the
cluster as a device. `scripts/rtsp_camera.py` publishes it as a stream instead, which puts it
on the same code path a shop-floor IP camera uses:

```bash
python scripts/rtsp_camera.py --list          # what this machine can open
python scripts/rtsp_camera.py 0               # device 0, port 8554, path cam
python scripts/rtsp_camera.py 1 --fps 10 --width 1280 --height 720 --quality 75
# a camera whose shutter is closed, or no camera at all
python scripts/rtsp_camera.py --image assets/images/roboflow_test_detect_human_robot.webp
```

It prints every address the stream answers on and says which one is local-only:

```
RTSP addresses for this stream:

  rtsp://192.168.1.20:8554/cam
  rtsp://172.19.0.1:8554/cam
  rtsp://127.0.0.1:8554/cam  (this machine only)
```

Register the address your cluster can route to (console: **Sources** → source → **register
and activate**, or `POST /cameras` with `{"source": "rtsp://192.168.1.20:8554/cam",
"activate": true}`). The `172.x.y.1` entries are Docker/WSL adapters, not your LAN - a pod
usually cannot reach them. Docker Desktop publishes the host as `host.docker.internal`, but
that name resolves in containers, not always in pods; the LAN address is the reliable one.
If the machine has no route to the cluster at all, a probe from inside the cluster fails
with `the source did not open` and that is the honest answer.

The script is a test aid, not a relay:

* **MJPEG only**, encoded by OpenCV. The payload is RFC 2435 RTP/JPEG, which is not "a JPEG
  file in a UDP packet": receivers rebuild the JPEG headers from the RTP header and expect
  only the entropy-coded scan data, so the script splits the encoded frame and ships the
  quantisation tables in-band. Shipping the whole file makes FFmpeg read image bytes as a
  quantisation table and drop every frame.
* **No authentication.** Anyone who can reach the port can watch. Bind a trusted network,
  or use `--port` and a firewall if that is not acceptable.
* **No recording, no H.264, no RTCP.** One stream, one codec, one purpose.
* Capture runs at the publish rate (`--fps`), so a slower camera simply delivers fewer
  frames; the console shows the negotiated rate, not the requested one.
* `--image PATH` publishes a still image instead of a camera. It is encoded once and the RTP
  timestamps still advance, so a receiver sees a live stream. Use it when the camera is
  useless (a closed privacy shutter delivers black frames) or absent: the registry and the
  detector can then be exercised with content the model recognises.

## RTSP transport

RTSP over UDP is the industry default and the wrong default here. A UDP RTP fragment that
crosses a NAT boundary or a Docker bridge is lost, and one lost fragment costs the whole
frame; behind NAT the camera cannot even route its packets back to the receiver.

Capture therefore asks OpenCV for **TCP** by default, and `RTSP_TRANSPORT=udp` restores the
lower-jitter UDP path on a clean LAN. OpenCV has no API for this: the transport is written
into `OPENCV_FFMPEG_CAPTURE_OPTIONS` just before the capture is opened and the previous value
is restored afterwards, so a webcam on the same process is unaffected.

This is a measured difference, not a preference. The same stream that read 640x480 on the
host returned **no frame at all** from inside a pod over UDP (the receiver's UDP port is a pod
address the host cannot route to), and TCP interleaved delivered every frame.

### A client that goes quiet is not a client that left

An RTSP control connection is idle by design: after `PLAY` the client sends nothing for
minutes while the RTP flows on the other channel. The publisher originally set a 30-second
socket timeout on that connection, so `recv` raised, the session was dropped, the stream
closed, and `snapshot()` reported `0 client(s)` while the camera was still publishing - the
console froze and a reader could only guess why.

The server now keeps idle sessions, gives every client its own writer thread, and holds at
most one pending frame per client, dropping the older one rather than queueing. A slow
consumer therefore costs its own frames and nothing else: a stalled client no longer blocks
the accept loop or the other viewers.

`snapshot()` and the script's heartbeat tell the two failures apart:

```
serving cam - 1 client(s), 214 frame(s) sent, 3 dropped, last frame 0.1s ago
client 192.168.1.33:52314 (tcp) - 214 frame(s), 3 dropped
```

* `last frame 0.1s ago` with clients listed: the stream is healthy.
* `last frame 12.4s ago`: the *camera* went silent (shutter closed, unplugged, held by
  another process) - not a network problem.
* `0 client(s)` with a recent frame: nobody is watching; the publisher is fine.

Per-session counters are also in `MjpegRtspServer.snapshot()` (`clients_served`, `sessions`,
`frame_age_seconds`), so the same facts are testable without reading a log.

Verified end to end: OpenCV reads `rtsp://127.0.0.1:8554/cam` at 640x480, the registry probe
reports `{"ok": true, "width": 640, "height": 480}` against it, and a registered camera fed
from `--image` produced real detections in `detection_results` (`robot` 0.275, `person`
0.220) through capture, dispatch, inference and the sink.

## Operating notes

* **Docker Desktop and WSL cannot see a USB camera.** Publish it with
  `scripts/rtsp_camera.py` (above) and register the RTSP URL - the same code path production
  uses. This is why a laptop camera cannot simply be handed to the cluster.
* A camera that answers ONVIF but whose stream is H.265 may open and still produce no
  frames in OpenCV; the probe reports that, and the camera's sub-stream (usually H.264)
  is the workaround.
* `DELETE /cameras/{id}` deletes the registered row *and* the detection task, and stops
  capture first when that camera is active (or is the one being captured), so deleting the
  running camera does not leave a pipeline pointed at a row that no longer exists.
* The dev overlay keeps the `camera` Deployment scaled to zero and lets the API pod
  capture a replay image; in that deployment `activate` restarts the API pod's own
  pipeline. Whatever process has `CAMERA_ENABLED=1` is the one that captures, and
  `CAMERA_REGISTRY_WATCH=1` is what makes it follow the registry. Capture runs at
  `CAMERA_FPS=5` there, and the replay source pans 24 px per 60 frames, so the preview
  moves; a slower cadence looks like a frozen camera even when everything is healthy.

## Verification

```bash
# publish a laptop camera as a stream the cluster can open
python scripts/rtsp_camera.py --list
python scripts/rtsp_camera.py 0
# discovery
curl -s -b cookies.txt 'http://localhost:30080/cameras/discover?include_network=true'
# register + activate an RTSP camera
curl -s -b cookies.txt -X POST http://localhost:30080/cameras -H 'content-type: application/json' \
  -d '{"id":"door-cam","source":"rtsp://10.0.0.5:554/stream1","username":"operator",
       "password":"s3cret","activate":true}' | jq
# check capture
curl -s -b cookies.txt http://localhost:30080/status | jq '.runner.camera_id, .registry'
```

`tests/camera/` covers classification, redaction, credential round-trips (including a
wrong key and a tampered token), WS-Discovery parsing over loopback UDP, the probe's
failure messages, and the supervisor's restart/retry behaviour. The RTSP server has its own
tests: the RFC 2435 payload round-trip, a full OPTIONS→DESCRIBE→SETUP→PLAY handshake over
loopback UDP, and - when the `ffmpeg` binary is present - decoding the stream with FFmpeg,
because that is the decoder OpenCV uses.
