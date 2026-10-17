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
    let resultTimes = [];
    const seenLabels = new Map();
    const palette = ["#4ade80", "#60a5fa", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee"];

    const streamUrl = computed(
      () => `/cameras/${encodeURIComponent(cameraId.value)}/stream?ts=${Date.now()}`
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
    }

    function onFeedLoad() {
      streamError.value = "";
      resizeOverlay();
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

      for (const detection of detections.value) {
        const x = detection.x_norm * width;
        const y = detection.y_norm * height;
        const w = detection.width_norm * width;
        const h = detection.height_norm * height;
        const colour = colourFor(detection.object_type);

        context.lineWidth = 2;
        context.strokeStyle = colour;
        context.strokeRect(x, y, w, h);

        const label = `${detection.object_type} ${(detection.confidence * 100).toFixed(0)}%`;
        context.font = "600 12px system-ui, sans-serif";
        const padding = 4;
        const textWidth = context.measureText(label).width;
        const labelY = y > 18 ? y - 18 : y + 2;
        context.fillStyle = colour;
        context.fillRect(x, labelY, textWidth + padding * 2, 16);
        context.fillStyle = "#0b1120";
        context.fillText(label, x + padding, labelY + 12);
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
      window.addEventListener("resize", resizeOverlay);
    });

    onBeforeUnmount(() => {
      window.removeEventListener("resize", resizeOverlay);
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
      loadPrompts,
      savePrompts,
      clearPrompts,
      startStream,
    };
  },
}).mount("#app");
