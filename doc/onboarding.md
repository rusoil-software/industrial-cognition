# Onboarding Guide for New Developers

Welcome to the Industrial Cognition team! This guide serves as your single point of truth to get set up and start contributing.

## 🚀 Getting Started Checklist

1.  **Prerequisites Check:** Ensure you have the following installed:
    *   Python 3.9+ (with virtual environment management).
    *   Docker & Docker Compose (for local testing environments).
    *   A local PostgreSQL instance (or access credentials for the staging DB).
    *   NVIDIA Drivers and CUDA Toolkit (for GPU acceleration).
2.  **Clone & Setup:**
    *   Clone the repository: `git clone <repo-url> industrial-cognition`
    *   Navigate: `cd industrial-cognition`
    *   Install dependencies: `pip install -r requirements/dev.txt`
3.  **First Test Run:**
    *   Run the foundational unit tests to verify the environment: `pytest tests/unit/`
4.  **System Check:**
    *   Run the `docker-compose` setup scripts to bring up all services (API, DB, etc.).
    *   Run the health check endpoint: `curl http://localhost:8000/health`

## 🗺️ Key Areas of Focus

*   **Core Logic:** The service logic resides in `src/<domain>/service.py`. Adhere strictly to the **Single Responsibility Principle (SRP)**.
*   **Communication:** Remember the explicit protocols:
    *   Robot Control: **Modbus/RS-485**.
    *   Real-time Streaming: **WebSocket** via FastAPI.
    *   Data Persistence: PostgreSQL via SQLAlchemy ORM.
*   **Documentation Source of Truth:** Always consult `doc/vision.md` and `doc/conventions.md` before writing or changing logic.

## 💡 Quick Start: Adding a Feature

1.  Follow the steps in `doc/workflow.md`.
2.  Create a new `doc/feature-name.md` to document the idea.
3.  Update `doc/tasklist.md` with the new task.
4.  Implement using the structure defined in `doc/test-template.md`.
5.  Update `doc/vision.md` to reflect the architectural extension.