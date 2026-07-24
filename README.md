### 🤖 Project Overview: Industrial AI Recognition System

This project implements a high-throughput, industrial-grade computer vision system utilizing a modern, decoupled *
*Microservices Architecture** to perform **Open-World Localization Vision Transformer (OWLv2)** recognition from
multiple real-time industrial cameras.

**Core Functionality & Goals:**

* **Goal:** To autonomously detect, locate, and classify specific parts in an industrial setting with extreme
  reliability.
* **Performance Target:** A primary operational goal is achieving **60 FPS** total throughput, requiring sub-200ms
  end-to-end inference latency.
* **Scope Lock:** The current focus is narrowed to a **single robot arm** interacting with a defined work area,
  demanding precise kinematic control and coordinate mapping.

**Architecture & Technology Stack (The "How"):**
The system is designed for **maximum operational resilience** using a robust, asynchronous pipeline pattern:

1. **Ingestion:** Raw streams are captured via **RTSP cameras** and pre-processed using **FFmpeg** into a standardized
   format.
2. **Orchestration:** A **FastAPI** gateway accepts streams, which are queued into a **RabbitMQ** message broker.
3. **Processing:** Dedicated **Inference Workers** (running on **GPU-accelerated Kubernetes pods**) consume messages,
   run the OWLv2 model via **ONNX Runtime**, and determine object locations.
4. **Persistence:** All raw frames, processed results, and metadata are stored in **MinIO** (S3-compatible object
   storage).
5. **Monitoring:** A comprehensive **Observability Stack** (Prometheus, Loki, Grafana) tracks system health, worker
   load, and data throughput for proactive failure detection.

**Key Technical Constraints & Focus Areas:**

* **Hardware Dependence:** Requires **NVIDIA GPUs** (e.g., RTX 5060 Ti) to meet latency targets.
* **Physical Grounding:** All generated coordinates must be relative to a physically defined coordinate system anchored
  by the camera's **3000mm height** and **45-degree viewing angle**.
* **Failure Handling:** The system must explicitly report and handle diagnostic states like `NO_TARGET`, `OUT_OF_REACH`,
  and `HUMAN_ON_SITE`.

**In essence, the project is building a scalable, highly reliable, real-time, closed-loop vision system designed for
industrial automation, with a clear mandate to favor resilient, decoupled architecture over initial simplicity.**

***Source Memories Used:***

* *`final_state`*: Defines the target visual goal.
* *`tech_stack`*: Details the core tech components (K8s, FastAPI, RabbitMQ, etc.).
* *`functional_spec`*: Outlines hard performance and reliability metrics (25 FPS min, 99.5% uptime).
* *`physical_spec`*: Anchors the digital system to real-world physics (angles, distances).
* *`context_update`*: Synthesizes the operational reality derived from the physical specs.
* *`scope_lock`*: Narrows the scope to a single robot/camera loop with specific diagnostic outputs.
* *`report_analysis`*: Provides the overall blueprint linking everything together.