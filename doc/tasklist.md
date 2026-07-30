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

* 🟡 Setup K8s Manifests: Create base Deployment/Service/ConfigMap templates for all microservices.
* 🟡 Database Schema Migration: Run initial Alembic migrations to create all necessary tables (`detection_results`, `system_metadata`).
* 🟡 Base Dependency Management: Finalize `requirements/prod.txt` based on base dependencies.

### **Stage 2: Core Functionality Implementation**

* 🟡 Camera Service: Implement low-level RTSP capture, using OpenCV, and streaming results via WebSocket.
* 🟡 Vision Service: Implement the core detection logic using OWLv2 on ONNX Runtime.
* 🟡 Robot Interface: Implement the Modbus/RS-485 communication service that consumes detection data and sends commands.
* 🟡 Orchestration: Develop the FastAPI layer to tie together the services (Routing/Error Handling).

### **Stage 3: Polish & Hardening**

* 🟡 End-to-End Testing: Write and execute full integration tests covering the entire pipeline (Camera -> Vision -> Modbus).
* 🟡 Observability Integration: Hook up Prometheus metrics scraping endpoints to all service checkpoints.
* 🟡 Deployment Automation: Finalize Helm chart package for production deployment.
