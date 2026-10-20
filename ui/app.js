/*
 * Copyright (c), The Rusoil Software Development Team. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Operator console - Vue 3, no build step.
 *
 * Why no bundler: the whole application is one view over three endpoints
 * (/cameras, /ws/detections, /cameras/{id}/prompt). A Vite project would add a Node
 * toolchain to the image and a build step to CI for no user-visible benefit yet. The
 * component is written so it can be lifted into a single-file component unchanged if
 * that changes - it has no globals beyond `Vue`.
 *
 * Overlay note: the MJPEG preview and the detection stream are independent. The image
 * is "what the camera sees now" (a few frames per second) and the boxes are "what the
 * last analysed frame contained". They are aligned by frame number in the status bar
 * rather than pretended to be frame-synchronised, because the inference latency is
 * seconds on CPU and pretending otherwise would make the overlay look broken.
 */
const { createApp, ref, computed, onMounted, onBeforeUnmount, nextTick } = Vue;


createApp({
  setup() {
    const cameras = ref([]);
    const cameraId = ref("main");
    const newCameraId = ref("");
    const showRegistry = ref(false);
    const scanning = ref(false);
    const scanned = ref(false);
    const discovery = ref({ devices: [], network: [], hint: "" });
    const register = ref({ id: "", label: "", source: "", username: "", password: "", activate: true });
    const registering = ref(false);
    const registryHint = ref("");
    const queries = ref([{ label: "metal sheet", threshold: 0.25 }]);
    const applied = ref([]);
    const detections = ref([]);
    const history = ref([]);
    const timings = ref(null);
    const lastFrameNumber = ref("-");
    const wsConnected = ref(false);
    const pipelineRunning = ref(false);
    const streaming = ref(false);
    const streamError = ref("");
    const streamNonce = ref(0);
    const tracks = ref([]);
    const tracking = ref({ enabled: false, tracker: null, reason: null, tracks: 0 });
    const frameAge = ref(null);
    const liveState = ref("unknown");
    const deleting = ref("");
    const saving = ref(false);
    const message = ref("");
    const messageKind = ref("ok");
    const resultFps = ref(0);
    const authenticated = ref(false);
    const tokenInput = ref("");
    const signingIn = ref(false);
    const viewport = ref(null);
    const overlay = ref(null);

    let socket = null;
    let redrawQueued = false;
    let healthTimer = null;
    let retryTimer = null;
    let resultTimes = [];
    const seenLabels = new Map();
    const palette = ["#4ade80", "#60a5fa", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee"];

    const streamUrl = computed(
      () =>
        `/cameras/${encodeURIComponent(cameraId.value)}/stream` +
        `?ts=${streamNonce.value}`
    );

    function colourFor(label) {
      if (!seenLabels.has(label)) {
        seenLabels.set(label, palette[seenLabels.size % palette.length]);
      }
      return seenLabels.get(label);
    }

    async function api(path, options) {
      const response = await fetch(path, {
        headers: { "content-type": "application/json" },
        ...options,
      });
      const text = await response.text();
      let body = null;
      try {
        body = text ? JSON.parse(text) : null;
      } catch (error) {
        body = { detail: text };
      }
      if (response.status === 401) {
        authenticated.value = false;
        throw new Error("authentication required");
      }
      if (!response.ok) {
        const detail = body && body.detail ? body.detail : response.statusText;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return body;
    }

    function flash(text, kind) {
      message.value = text;
      messageKind.value = kind || "ok";
      window.setTimeout(() => {
        if (message.value === text) message.value = "";
      }, 6000);
    }

    async function refreshCameras() {
      try {
        const body = await api("/cameras");
        cameras.value = body.cameras || [];
        const current = cameras.value.find((camera) => camera.camera_id === cameraId.value);
        if (!current && cameras.value.length) {
          cameraId.value = cameras.value[0].camera_id;
        }
        pipelineRunning.value = cameras.value.some((camera) => camera.running);
      } catch (error) {
        pipelineRunning.value = false;
      }
    }

    function addCamera() {
      const id = newCameraId.value.trim();
      if (!id) return;
      cameraId.value = id;
      newCameraId.value = "";
      refreshCameras();
      loadPrompts();
    }

    function toggleRegistry() {
      showRegistry.value = !showRegistry.value;
      // Scanning opens every local device and waits for ONVIF answers (seconds), so it
      // runs when the operator asks for it - never on page load.
      if (!showRegistry.value) registryHint.value = "";
    }

    async function scanCameras() {
      scanning.value = true;
      registryHint.value = "";
      try {
        const body = await api("/cameras/discover?include_network=true&timeout=2.0");
        discovery.value = {
          devices: body.devices || [],
          network: body.network || [],
          hint: body.hint || "",
        };
        scanned.value = true;
        if (!body.credentials_configured) {
          registryHint.value =
            "CAMERA_CRED_KEY is not configured on the server: cameras that need a password " +
            "cannot be registered until it is set (doc/camera.md).";
        }
      } catch (error) {
        flash(String(error.message || error), "error");
      } finally {
        scanning.value = false;
      }
    }

    function slugifySource(text) {
      const slug = String(text || "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 48);
      return slug || "camera";
    }

    function useSource(source, label) {
      register.value.source = String(source);
      if (!register.value.id.trim()) {
        register.value.id = slugifySource(register.value.source);
      }
      if (label && !register.value.label.trim()) {
        register.value.label = label;
      }
    }

    async function registerCamera() {
      registering.value = true;
      try {
        const body = await api("/cameras", {
          method: "POST",
          body: JSON.stringify({
            id: register.value.id.trim(),
            label: register.value.label.trim() || register.value.id.trim(),
            source: register.value.source.trim(),
            username: register.value.username.trim() || null,
            password: register.value.password || null,
            activate: register.value.activate,
          }),
        });
        flash(`camera ${body.camera_id} registered`, "ok");
        register.value = { id: "", label: "", source: "", username: "", password: "", activate: true };
        cameraId.value = body.camera_id;
        await refreshCameras();
        onCameraChange();
      } catch (error) {
        flash(String(error.message || error), "error");
      } finally {
        registering.value = false;
      }
    }

    async function activateCamera(id) {
      try {
        await api(`/cameras/${encodeURIComponent(id)}/activate`, { method: "POST" });
        flash(`capture follows ${id}`, "ok");
        cameraId.value = id;
        await refreshCameras();
        streaming.value = false;
        nextTick(startStream);
      } catch (error) {
        flash(String(error.message || error), "error");
      }
    }

    async function probeCamera(id) {
      try {
        const body = await api(`/cameras/${encodeURIComponent(id)}/probe`, { method: "POST" });
        registryHint.value = body.ok
          ? `${id}: ${body.width}×${body.height} in ${Math.round(body.latency_ms)} ms`
          : `${id}: ${body.detail}`;
      } catch (error) {
        flash(String(error.message || error), "error");
      }
    }

    async function deleteCamera(id) {
      if (!window.confirm(`Remove camera ${id}? Its detection task is kept.`)) return;
      deleting.value = id;
      try {
        await api(`/cameras/${encodeURIComponent(id)}`, { method: "DELETE" });
        flash(`camera ${id} removed`, "ok");
        tracks.value = [];
        const wasSelected = cameraId.value === id;
        if (wasSelected) {
          // The registry stops capture when the active camera is deleted, so the console
          // must not keep showing the removed camera's stream.
          cameraId.value = "";
          streaming.value = false;
          detections.value = [];
        }
        await refreshCameras();
        if (wasSelected) onCameraChange();
      } catch (error) {
        flash(String(error.message || error), "error");
      } finally {
        deleting.value = "";
      }
    }

    // Polling /status is what makes the console honest about the connection: the MJPEG
    // stream keeps the last frame on screen when capture dies, so only the age of the
    // newest frame tells an operator whether the camera is still there.
    async function pollHealth() {
      try {
        const body = await api("/status");
        const latest = body.latest_frame || null;
        // "Late" is relative to the cadence capture was asked for: a dev replay at 0.2 FPS
        // is healthy at four seconds of age, a 30 FPS camera at four seconds is dead.
        const staleAfter = (latest && latest.stale_after_seconds) || 5;
        frameAge.value = latest ? latest.age_seconds : null;
        const running = Boolean(body.running);
        const capturedCamera = (body.pipeline && body.pipeline.camera_id) || null;
        if (!running || !latest) {
          liveState.value = running ? "waiting" : "offline";
        } else if (capturedCamera && capturedCamera !== cameraId.value) {
          liveState.value = "other";
        } else {
          liveState.value = latest.age_seconds <= staleAfter ? "live" : "stale";
        }
        tracking.value = body.tracking || tracking.value;
        if (!tracking.value.enabled) tracks.value = [];
      } catch (error) {
        liveState.value = "offline";
        frameAge.value = null;
      }
    }

    function retryStream() {
      // A closed MJPEG response leaves the <img> on its last frame; bumping the nonce
      // opens a new stream so a restarted capture comes back without a page reload.
      streamNonce.value += 1;
      streaming.value = true;
    }

    function onCameraChange() {
      streamError.value = "";
      streaming.value = false;
      detections.value = [];
      applied.value = [];
      history.value = [];
      loadPrompts();
      nextTick(startStream);
    }

    async function loadPrompts() {
      try {
        const body = await api(`/cameras/${encodeURIComponent(cameraId.value)}/prompt`);
        applied.value = body.prompts || [];
        queries.value = (body.prompts || []).map((query) => ({
          label: query.label,
          threshold: query.threshold,
        }));
        if (!queries.value.length) queries.value = [{ label: "", threshold: 0.25 }];
      } catch (error) {
        applied.value = [];
        if (!/404|no detection prompt/.test(String(error.message))) {
          flash(error.message, "error");
        }
      }
    }

    async function savePrompts() {
      saving.value = true;
      try {
        const body = await api(`/cameras/${encodeURIComponent(cameraId.value)}/prompt`, {
          method: "PUT",
          body: JSON.stringify({
            prompts: queries.value
              .filter((query) => query.label && query.label.trim())
              .map((query) => ({ label: query.label.trim(), threshold: query.threshold })),
          }),
        });
        applied.value = body.prompts || [];
        flash("task applied - the next frames use it");
      } catch (error) {
        flash(error.message, "error");
      } finally {
        saving.value = false;
      }
    }

    async function clearPrompts() {
      try {
        await api(`/cameras/${encodeURIComponent(cameraId.value)}/prompt`, { method: "DELETE" });
        applied.value = [];
        queries.value = [{ label: "", threshold: 0.25 }];
        flash("task cleared");
      } catch (error) {
        flash(error.message, "error");
      }
    }

    function addQuery() {
      queries.value.push({ label: "", threshold: 0.25 });
    }

    function startStream() {
      streamError.value = "";
      streaming.value = true;
      streamNonce.value += 1;
    }

    function onFeedLoad() {
      streamError.value = "";
      resizeOverlay();
      // Every MJPEG part fires `load`: redraw the overlay here so tracked boxes land on
      // the frame that is actually on screen, not on the previous one.
      scheduleDraw();
    }

    function onFeedError() {
      streamError.value = "no live frames - waiting for the camera";
      streaming.value = false;
      liveState.value = "offline";
      if (retryTimer) window.clearTimeout(retryTimer);
      retryTimer = window.setTimeout(() => {
        if (authenticated.value) retryStream();
      }, 5000);
    }

    function resizeOverlay() {
      const canvas = overlay.value;
      const image = canvas && canvas.parentElement.querySelector(".feed");
      if (!canvas || !image || !image.clientWidth) return;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = image.clientWidth * ratio;
      canvas.height = image.clientHeight * ratio;
      canvas.style.width = `${image.clientWidth}px`;
      canvas.style.height = `${image.clientHeight}px`;
      draw();
    }

    function draw() {
      const canvas = overlay.value;
      if (!canvas) return;
      const context = canvas.getContext("2d");
      const ratio = window.devicePixelRatio || 1;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const width = canvas.width / ratio;
      const height = canvas.height / ratio;
      context.clearRect(0, 0, width, height);

      function box(item) {
        return {
          x: item.x_norm * width,
          y: item.y_norm * height,
          w: item.width_norm * width,
          h: item.height_norm * height,
        };
      }

      function label(text, left, top, colour) {
        context.font = "600 12px system-ui, sans-serif";
        const padding = 4;
        const textWidth = context.measureText(text).width;
        const labelY = top > 18 ? top - 18 : top + 2;
        context.fillStyle = colour;
        context.fillRect(left, labelY, textWidth + padding * 2, 16);
        context.fillStyle = "#0b1120";
        context.fillText(text, left + padding, labelY + 12);
      }

      // The last inference, drawn thin and dashed: this is what the model actually said,
      // which on a GPU-less host may be several seconds old.
      if (tracks.value.length) {
        context.save();
        context.globalAlpha = 0.55;
        context.setLineDash([4, 4]);
        context.lineWidth = 1;
        for (const detection of detections.value) {
          const { x, y, w, h } = box(detection);
          context.strokeStyle = colourFor(detection.object_type);
          context.strokeRect(x, y, w, h);
        }
        context.restore();
      } else {
        for (const detection of detections.value) {
          const { x, y, w, h } = box(detection);
          const colour = colourFor(detection.object_type);
          context.lineWidth = 2;
          context.strokeStyle = colour;
          context.strokeRect(x, y, w, h);
          label(
            `${detection.object_type} ${(detection.confidence * 100).toFixed(0)}%`,
            x,
            y,
            colour
          );
        }
      }

      // Tracked boxes move with the video between inferences; they are the primary
      // signal, so they are solid and sit on top.
      for (const track of tracks.value) {
        const { x, y, w, h } = box(track);
        const colour = colourFor(track.object_type);
        context.save();
        if (track.stale) context.setLineDash([6, 4]);
        context.lineWidth = track.stale ? 2 : 3;
        context.strokeStyle = colour;
        context.strokeRect(x, y, w, h);
        context.restore();
        label(`${track.object_type} #${track.track_id}`, x, y, colour);
      }
    }

    function scheduleDraw() {
      if (redrawQueued) return;
      redrawQueued = true;
      window.requestAnimationFrame(() => {
        redrawQueued = false;
        draw();
      });
    }

    function summarise(items) {
      if (!items.length) return "nothing";
      const counts = {};
      for (const item of items) {
        counts[item.object_type] = (counts[item.object_type] || 0) + 1;
      }
      return Object.entries(counts)
        .map(([label, count]) => `${label}×${count}`)
        .join(", ");
    }

    async function signIn() {
      signingIn.value = true;
      try {
        await api("/auth/login", {
          method: "POST",
          body: JSON.stringify({ token: tokenInput.value.trim() }),
        });
        tokenInput.value = "";
        authenticated.value = true;
        await refreshCameras();
        await loadPrompts();
        startStream();
        connect();
      } catch (error) {
        flash(error.message === "authentication required" ? "invalid token" : error.message, "error");
      } finally {
        signingIn.value = false;
      }
    }

    async function signOut() {
      try {
        await api("/auth/login", { method: "DELETE" });
      } catch (error) {
        // A failed sign-out still drops the local view.
      }
      authenticated.value = false;
      if (socket) socket.close();
      socket = null;
      streaming.value = false;
    }

    function connect() {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${window.location.host}/ws/detections`);

      socket.onopen = () => {
        wsConnected.value = true;
      };

      socket.onclose = () => {
        wsConnected.value = false;
        window.setTimeout(connect, 3000);
      };

      socket.onerror = () => {
        wsConnected.value = false;
      };

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);

        if (payload.type === "hello") {
          pipelineRunning.value = Boolean(payload.pipeline && payload.pipeline.running);
          return;
        }
        if (payload.type === "unavailable") {
          pipelineRunning.value = false;
          return;
        }
        if (payload.type === "tracks") {
          // Tracked boxes arrive at the camera rate, between the (slow) inferences.
          tracks.value = payload.tracks || [];
          scheduleDraw();
          return;
        }

        const metadata = payload.metadata || {};
        lastFrameNumber.value = metadata.frame_number ?? lastFrameNumber.value;
        detections.value = payload.detections || [];
        timings.value = payload.timings || null;
        scheduleDraw();

        resultTimes.push(performance.now());
        if (resultTimes.length > 30) resultTimes.shift();
        if (resultTimes.length > 1) {
          const span = (resultTimes[resultTimes.length - 1] - resultTimes[0]) / 1000;
          resultFps.value = span > 0 ? (resultTimes.length - 1) / span : 0;
        }

        history.value.unshift({
          frame: metadata.frame_number,
          summary: summarise(detections.value),
          inference: payload.timings && payload.timings.inference_ms
            ? `${payload.timings.inference_ms.toFixed(0)} ms`
            : "-",
        });
        history.value = history.value.slice(0, 8);
      };
    }

    onMounted(async () => {
      try {
        const status = await api("/auth/status");
        authenticated.value = status.authenticated || !status.required;
      } catch (error) {
        authenticated.value = false;
      }
      if (!authenticated.value) return;
      await refreshCameras();
      await loadPrompts();
      startStream();
      connect();
      pollHealth();
      healthTimer = window.setInterval(pollHealth, 2000);
      window.addEventListener("resize", resizeOverlay);
    });

    onBeforeUnmount(() => {
      window.removeEventListener("resize", resizeOverlay);
      window.clearInterval(healthTimer);
      window.clearTimeout(retryTimer);
      if (socket) socket.close();
    });

    return {
      authenticated,
      tokenInput,
      signingIn,
      signIn,
      signOut,
      cameras,
      cameraId,
      newCameraId,
      showRegistry,
      scanning,
      scanned,
      discovery,
      register,
      registering,
      registryHint,
      toggleRegistry,
      scanCameras,
      useSource,
      registerCamera,
      activateCamera,
      probeCamera,
      deleteCamera,
      deleting,
      tracks,
      tracking,
      frameAge,
      liveState,
      pollHealth,
      retryStream,
      queries,
      applied,
      detections,
      history,
      timings,
      lastFrameNumber,
      wsConnected,
      pipelineRunning,
      streaming,
      streamError,
      saving,
      message,
      messageKind,
      resultFps,
      viewport,
      overlay,
      streamUrl,
      colourFor,
      addCamera,
      addQuery,
      onCameraChange,
      onFeedLoad,
      onFeedError,
      loadPrompts,
      savePrompts,
      clearPrompts,
      startStream,
    };
  },
}).mount("#app");
