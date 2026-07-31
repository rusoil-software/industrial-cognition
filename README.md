[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/rusoil-software/industrial-cognition">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Industrial Cognition AI System</h3>

  <p align="center">
    A Project Blueprint for Autonomous Vision and Robotics
    <br />
    <a href="https://github.com/rusoil-software/industrial-cognition/wiki"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/rusoil-software/industrial-cognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/rusoil-software/industrial-cognition/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

# Industrial Cognition AI System

<!-- ABOUT THE PROJECT -->
## About The Project

![MVM Dashboard view](https://github.com/rusoil-software/industrial-cognition/tree/master/images/screenshot.png "MVM Dashboard")

### Project Overview

This project implements a high-throughput, industrial-grade computer vision system utilizing a modern, decoupled *
*Microservices Architecture** to perform **Open-World Localization Vision Transformer (OWLv2)** recognition from
multiple real-time industrial cameras.
It's goal is to create an autonomous, real-time system that uses computer vision to detect and locate specific components (metal sheets, parts) in a physical workspace, and then reliably issues corrective/directive commands to a physical robot controller.

## Architectural Overview (Layered Design)

The system adheres to a strict **Layered Architecture** to maximize resilience and maintainability.

![This picture contains the Machine Vision Module internal structure which is a Kubernetes‑based video analytics pipeline with eight layers: Cameras – Six IP cameras stream 1080p H.264 video over RTSP. FFMPEG Pods – One pod per camera decodes the stream into raw frames. FastAPI Service – A load‑balanced API validates incoming frames and publishes work messages. MinIO Storage – Three‑node S3‑compatible cluster storing frames and inference results. RabbitMQ – Three‑node message broker with priority queues, results queue, and dead‑letter queue. Inference Workers – Four GPU pods running ONNX Runtime, consuming messages and writing results. Results Workers – Two CPU pods performing tracking, post‑processing, and alerting. Observability – Prometheus, Loki, and Grafana for metrics, logs, and dashboards. Supporting components include Kubernetes control‑plane services, CPU and GPU node pools, persistent volumes, autoscaling rules based on CPU/GPU load and queue depth, and network policies restricting traffic to RTSP, AMQP, and HTTPS.](https://github.com/rusoil-software/industrial-cognition/tree/master/images/architecture.png "MVM Architecture")

**Flow:** Data flows from the physical world $\rightarrow$ Camera Service $\rightarrow$ Vision Service $\rightarrow$ Robot Interface $\rightarrow$ Physical Robot.

**Layers:**

Here are main layers of the MVM (Machine Vision Module):

1. **Presentation (FastAPI API):** Provides a modern, websocket-driven control panel for monitoring and control.
2. **Service Layer (Orchestrator):** The main API handles state management, error recovery, and coordinating calls between specialized domain services.
3. **Domain Services:**
    * **Camera Service:** Manages all physical I/O, abstracting the complexity of video streaming via **FFmpeg/GStreamer**.
    * **Vision Service:** Executes the heavy computational lifting using **OpenCV** and **ONNX Runtime** models (e.g., OWLv2).
    * **Robot Interface:** Handles the low-level physical communication, strictly adhering to **Modbus/RS-485** protocols.
4. **Data Layer:** Uses **SQLAlchemy/PostgreSQL** for durable storage of metadata and historical results.

## Key Protocols & Constraints

* **Communication Protocol:** Physical robot control can use **binary Modbus over RS-485** or direct TCP/IP connection to MVM stream of JSON data (API key configuration required).
* **Streaming:** All real-time video streams are managed via **FFmpeg/GStreamer** pipelines to ensure performance consistency.
* **Deployment:** containerized and orchestrated using **Kubernetes (K8s)** for high availability and horizontal scaling.

## Core Functionality & Goals

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

### Built With

* [Kubernetes](https://kubernetes.io)
* [Docker](https://docker.com)
* [FFMPEG](https://ffmpeg.org/)
* [RTSP](https://en.wikipedia.org/wiki/Real-Time_Streaming_Protocol)
* [Modbus](https://modbus.org)
* [FastAPI](https://fastapi.tiangolo.com/)
* [PostgreSQL](https://www.postgresql.org/)
* [RabbitMQ](https://rabbitmq.com)
* [Celery](https://docs.celeryq.dev/en/v5.4.0/getting-started/backends-and-brokers/rabbitmq.html)
* [Redis](https://redis.io)
* [MinIO](https://min.io)
* [NVIDIA RTX](https://www.nvidia.com/en-us/geforce/rtx/)
* [Loki](https://github.com/grafana/loki)
* [Alloy](https://github.com/grafana/alloy)
* [Prometheus](https://prometheus.io)
* [Grafana](https://grafana.com)

<!-- GETTING STARTED -->
## Getting Started

Follow the steps in [Onboarding](https://github.com/rusoil-software/industrial-cognition/tree/master/) docs for new team members to get the entire stack running locally for development.

### Prerequisites

#### Hardware Requirements

##### Minimum Requirements

* **CPU**: Intel Core i7 or equivalent (8 cores, 16 threads)
* **GPU**: NVIDIA GTX 1660 or equivalent (6GB VRAM) * optional but recommended
* **RAM**: 16GB DDR4
* **Storage**: 512GB SSD (for OS, application, and database)
* **Camera Interface**: Compatible with industrial camera (GigE Vision, USB3 Vision, or Camera Link)
* **Network**: Gigabit Ethernet
* 60 FPS 2K *Industrial Camera* with Gigabit Ethernet connectior

##### Recommended Requirements

* **CPU**: Intel Xeon or AMD Ryzen 9 (16+ cores)
* **GPU**: NVIDIA RTX 3070 or equivalent (8GB+ VRAM)
* **RAM**: 32GB DDR4
* **Storage**: 1TB NVMe SSD
* **Camera Interface**: Dedicated frame grabber card if required by camera
* **Network**: 10GbE capable
* 60 FPS 4K *Industrial Camera* with Gigabit Ethernet connectior

### Software Requirements

* *Kubernetes*
* *Docker*
* *Ubuntu Server* or another apt-compatible distributive.

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/rusoil-software/industrial-cognition.git
   ```

2. Go into the directory of the project

   ```sh
   cd industrial-cognition
   ```

3. Run deployment script

   ```sh
   sh scripts/install.sh
   ```

4. Wait until installation and deployment is finished

5. Go to `your-server-public-ip/dashboard` (with deployment via reverse proxy) or `localhost:8080/dashboard` (for development on local machine) to see the MVM Web App

6. Follow initial setup instructions within the MVM App

<!-- USAGE EXAMPLES -->
## Usage

For more examples, please refer to the [Documentation](https://github.com/rusoil-software/industrial-cognition/wiki)

## Development & Process Guide

The following files detail *how* to build and *how* to maintain the system:

* **`doc/workflow.md`**: Mandatory step-by-step guide for all feature development, emphasizing sequential workflow adherence.
* **`doc/conventions.md`**: Contains the hard rules (KISS, SRP, Modbus usage) that every line of code must follow.
* **`doc/tasklist.md`**: The master tracker for tracking all remaining development tasks.

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/rusoil-software/industrial-cognition/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the Apache 2 License. See [LICENSE](https://github.com/rusoil-software/industrial-cognition/blob/master/LICENSE) for more information.

<!-- CONTACT -->
## Contact

Konstantin Ustiuzhanin - [email](ustyuzhaninky@hotmail.com)

Project Link: [Industrial Cognition](https://github.com/rusoil-software/industrial-cognition)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Kubernetes](https://kubernetes.io)
* [Docker](https://docker.com)
* [FFMPEG](https://ffmpeg.org/)
* [Modbus](https://modbus.org)
* [FastAPI](https://fastapi.tiangolo.com/)
* [PostgreSQL](https://www.postgresql.org/)
* [RabbitMQ](https://rabbitmq.com)
* [Celery](https://docs.celeryq.dev/en/v5.4.0/getting-started/backends-and-brokers/rabbitmq.html)
* [Redis](https://redis.io)
* [MinIO](https://min.io)
* [NVIDIA RTX](https://www.nvidia.com/en-us/geforce/rtx/)
* [Loki](https://github.com/grafana/loki)
* [Alloy](https://github.com/grafana/alloy)
* [Prometheus](https://prometheus.io)
* [Grafana](https://grafana.com)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/rusoil-software/industrial-cognition.svg?style=for-the-badge
[contributors-url]: https://github.com/rusoil-software/industrial-cognition/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/rusoil-software/industrial-cognition.svg?style=for-the-badge
[forks-url]: https://github.com/rusoil-software/industrial-cognition/network/members
[stars-shield]: https://img.shields.io/github/stars/rusoil-software/industrial-cognition.svg?style=for-the-badge
[stars-url]: https://github.com/rusoil-software/industrial-cognition/stargazers
[issues-shield]: https://img.shields.io/github/issues/rusoil-software/industrial-cognition.svg?style=for-the-badge
[issues-url]: https://github.com/rusoil-software/industrial-cognition/issues
[license-shield]: https://img.shields.io/github/license/rusoil-software/industrial-cognition.svg?style=for-the-badge
[license-url]: https://github.com/rusoil-software/industrial-cognition/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/konstantine-ustyuzhanin-544551131/
[product-screenshot]: images/screenshot.png
[product-architecture]: images/achitecture.png
