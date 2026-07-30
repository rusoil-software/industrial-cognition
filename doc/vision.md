# Industrial Cognition * Technical Vision

This document outlines the definitive technical blueprint for the industrial-cognition project, establishing a resilient, scalable, and maintainable standard. The vision adheres strictly to the **KISS Principle** while architecting for industrial-grade reliability.

## Core Vision & Mandate

The system's primary function is to provide autonomous, real-time localization and classification of parts within an industrial workspace, enabling robotic systems to adapt dynamically to complex environments.

## ⚙️ Architectural Blueprint (Layered Model)

The system must be strictly layered to ensure Separation of Concerns (SRP):

1. **Presentation/API Layer (FastAPI):** Acts as the single entry point. It exposes WebSocket endpoints (`/ws/detections`) for real-time telemetry and standard REST endpoints (`/status`, `/config`) for synchronous control.
2. **Service Layer (Orchestrator):** This layer manages the flow. It consumes processed data from the Camera Service, passes necessary commands to the Robot Interface, and manages the overall state lifecycle.
3. **Domain Services:** Dedicated, isolated modules for each physical concern:
    * **Camera Service:** Manages all physical stream capture using **FFmpeg/GStreamer** wrappers for robust, multi-stream input.
    * **Vision Service:** Executes complex computer vision tasks (OWLv2) on pre-processed frames.
    * **Robot Interface:** Manages the physical communication protocol, strictly using **binary Modbus over RS-485**.
    * **System Service:** Handles logging, configuration, and health monitoring.
4. **Data Access Layer:** Abstracted via the Repository pattern, ensuring the business logic never talks directly to the database engine.
5. **Persistence:** PostgreSQL 13+ via SQLAlchemy ORM.

## 🌐 Communication Protocols (The Truth Source)

* **Robot Control:** **Modbus/RS-485** (Physical, Low-Level Control).
* **Real-Time Telemetry:** **WebSocket/FastAPI** (High-Throughput Data Stream).
* **Configuration/Status:** **REST** (Standard HTTP calls).

## 💻 Technology Stack

* **Language:** Python 3.9+ (Mandatory use of `asyncio`).
* **AI/CV:** OpenCV 4.5+ linked against ONNX Runtime.
* **Orchestration:** **Kubernetes (K8s)** deployment model using Helm charts.
* **Asynchronous Work:** Must utilize message queuing (RabbitMQ) for decoupling service components.

## 🛠️ Workflow & Development Standards

* **Development:** Strictly follows the **Workflow Guide** (`doc/workflow.md`).
* **Coding:** Adheres to the **Conventions** (`doc/conventions.md`), prioritizing TDD and clean boundaries.

## Project Structure

```tree
src/
├── camera/                  # Camera domain
│   ├── router.py            # Camera API endpoints
│   ├── schemas.py           # Pydantic models for camera
│   ├── models.py            # SQLAlchemy ORM models
│   ├── service.py           # Camera business logic
│   ├── dependencies.py      # Camera route dependencies
│   ├── config.py            # Camera domain settings
│   ├── constants.py         # Camera constants and error codes
│   ├── exceptions.py        # Camera-specific exceptions
│   └── utils.py             # Camera helper functions
├── vision/                  # Vision processing domain
│   ├── router.py            # Vision API endpoints
│   ├── schemas.py           # Pydantic models for vision
│   ├── models.py            # SQLAlchemy ORM models
│   ├── service.py           # Vision business logic
│   ├── dependencies.py      # Vision route dependencies
│   ├── config.py            # Vision domain settings
│   ├── constants.py         # Vision constants and error codes
│   ├── exceptions.py        # Vision-specific exceptions
│   └── utils.py             # Vision helper functions
├── detection/               # Detection domain
│   ├── router.py            # Detection API endpoints
│   ├── schemas.py           # Pydantic models for detection
│   ├── models.py            # SQLAlchemy ORM models
│   ├── service.py           # Detection business logic
│   ├── dependencies.py      # Detection route dependencies
│   ├── config.py            # Detection domain settings
│   ├── constants.py         # Detection constants and error codes
│   ├── exceptions.py        # Detection-specific exceptions
│   └── utils.py             # Detection helper functions
├── system/                  # System domain
│   ├── router.py            # System API endpoints
│   ├── schemas.py           # Pydantic models for system
│   ├── models.py            # SQLAlchemy ORM models
│   ├── service.py           # System business logic
│   ├── dependencies.py      # System route dependencies
│   ├── config.py            # System domain settings
│   ├── constants.py         # System constants and error codes
│   ├── exceptions.py        # System-specific exceptions
│   └── utils.py             # System helper functions
├── config.py                # Global BaseSettings
├── models.py                # Shared Pydantic/ORM bases
├── exceptions.py            # Global exceptions
├── database.py              # Async engine + session factory
└── main.py                  # FastAPI app + lifespan

tests/
├── unit/
├── integration/
└── e2e/

alembic/
├── versions/                # Migration scripts
└── env.py

requirements/
├── base.txt                 # Common dependencies
├── dev.txt                  # Development dependencies
└── prod.txt                 # Production dependencies
```

## Project Architecture

The system follows a layered architecture with clear separation of concerns:

```chartjs
+---------------------+
|    User Interface   |
|  (Web Dashboard)    |
+----------+----------+
           |
+----------v----------+
|      API Layer      |
|  (FastAPI)          |
|  * REST endpoints   |
|  * WebSocket server |
+----------+----------+
           |
+----------v----------+
|   Service Layer     |
|  * Orchestrates     |
|    components       |
|  * Business logic   |
+----------+----------+
           |
+----------v----------+  +------------------+
|  Vision Processing  |  |  Camera Interface |
|  * OpenCV pipeline  |<--|  * Hardware SDK  |
|  * Object detection |  |  * Frame capture |
+----------+----------+  +------------------+
           |
+----------v----------+
|   Data Access Layer |
|  * SQLAlchemy ORM   |
|  * Repository pattern|
+----------+----------+
           |
+----------v----------+
|    Data Storage     |
|  (PostgreSQL)       |
+---------------------+
```

## Data Model

### Detection Result

* **id**: UUID (primary key)
* **timestamp**: DateTime (UTC)
* **frame_id**: Integer (sequential frame counter)
* **object_type**: Enum ("metal_sheet", "detail", "clutter")
* **coordinates**: JSON
  * x: float (normalized 0-1)
  * y: float (normalized 0-1)
  * width: float (normalized 0-1)
  * height: float (normalized 0-1)
* **confidence**: Float (0.0-1.0)
* **metadata**: JSON (additional detection information)

### System Metadata

* **id**: UUID (primary key)
* **camera_status**: Enum ("connected", "disconnected", "error")
* **processing_fps**: Float (frames per second)
* **last_heartbeat**: DateTime (UTC)
* **configuration**: JSON (current system settings)

## User Interactions

### Real-Time Updates

* **WebSocket Endpoint**: `/ws/detections`
  * Pushes detection results in real-time as they're processed
  * Message format: JSON with detection data and metadata
  * Supports multiple concurrent clients

### Control Interface

* **REST API Endpoints**:
  * `GET /health` * System health check
  * `GET /status` * Current system status and statistics
  * `POST /control/start` * Start detection process
  * `POST /control/stop` * Stop detection process
  * `GET /detections` * Retrieve recent detection history
  * `GET /config` * Get current configuration
  * `PUT /config` * Update configuration

## Storage

* **Primary Database**: PostgreSQL 13+
  * Stores all detection results and system metadata
  * Configured with appropriate indexes for time-based queries*
* **Migration Strategy**: Alembic
  * Version-controlled schema changes
  * Automated migration scripts
* **Backup Strategy**: Daily automated backups with 7-day retention
* **Data Retention**: Configurable policy (default: 30 days of detection data)

## Hardware Requirements

### Minimum Requirements

* **CPU**: Intel Core i7 or equivalent (8 cores, 16 threads)
* **GPU**: NVIDIA GTX 1660 or equivalent (6GB VRAM) * optional but recommended
* **RAM**: 16GB DDR4
* **Storage**: 512GB SSD (for OS, application, and database)
* **Camera Interface**: Compatible with industrial camera (GigE Vision, USB3 Vision, or Camera Link)
* **Network**: Gigabit Ethernet

### Recommended Requirements

* **CPU**: Intel Xeon or AMD Ryzen 9 (16+ cores)
* **GPU**: NVIDIA RTX 3070 or equivalent (8GB+ VRAM)
* **RAM**: 32GB DDR4
* **Storage**: 1TB NVMe SSD
* **Camera Interface**: Dedicated frame grabber card if required by camera
* **Network**: 10GbE capable

## Camera Integration

* **Interface**: Direct SDK integration with industrial camera manufacturer's API
* **Supported Protocols**: GigE Vision, USB3 Vision, Camera Link (depending on camera model)
* **Frame Rate**: Up to 60 FPS (configurable)
* **Resolution**: Up to 4K (configurable)
* **Trigger Modes**: Software trigger and continuous acquisition modes
* **Error Handling**: Automatic reconnection and error reporting

## Work Scenarios

### Normal Operation

1. System starts and initializes camera connection
2. Camera begins capturing frames at configured rate
3. Each frame is processed by OpenCV pipeline
4. Detected objects are recorded with timestamps and coordinates
5. Results are stored in database and broadcast via WebSocket
6. System continuously monitors performance and health

### Robot Integration

1. Robotic system connects to WebSocket endpoint
2. Receives real*time updates about object positions
3. Queries REST API for specific information as needed
4. Adjusts robot movements based on current workspace state

### Maintenance Mode

1. System can be paused via REST API
2. Camera continues capturing but processing is suspended
3. Allows for camera cleaning or workspace adjustments
4. Can be resumed via REST API

## Deployment

### Kubernetes Deployment

* **Kubernetes** configuration for all components:
  * Web API service
  * Database service
  * Camera service
  * Vision service
  * Robot Interface service
  * System service
  * RabbitMQ service and Celery worker nodes
  * Redis service
  * MinIO service
  * Optional monitoring services Prometheus, Loki and Grafana

### Configuration

* Environment variables for sensitive data (database credentials, API keys)
* YAML configuration files for system settings
* Configuration reload without restart

### Monitoring

* Health checks exposed via `/health` endpoint
* Metrics endpoint for performance monitoring
* Log aggregation and rotation

## Scaling

### Current Prototype

* Single-instance deployment
* All components run on one industrial server
* Vertical scaling through hardware upgrades

### Future Scaling Options

* **Horizontal Scaling**: Multiple detection nodes with load balancing
* **Microservices**: Split components into separate services
* **Edge Computing**: Distributed processing across multiple devices
* **Cloud Integration**: Offload historical data analysis to cloud

## Configuration Approach

### Configuration Layers

1. **Default Configuration**: Built into the application
2. **File Configuration**: YAML files for environment-specific settings
3. **Environment Variables**: For secrets and deployment-specific values
4. **Runtime Configuration**: API endpoints to modify settings without restart

### Configuration Management

* Configuration validation on startup and reload
* Schema validation using Pydantic
* Configuration change logging
* Rollback capability for failed configurations

## Logging Approach

### Logging Levels

* **DEBUG**: Detailed information for development and troubleshooting
* **INFO**: General operational information
* **WARNING**: Unexpected but recoverable conditions
* **ERROR**: Errors that don't prevent system operation
* **CRITICAL**: System-critical errors that may require intervention

### Log Structure

* JSON format for easy parsing and analysis
* Includes timestamp, level, module, and message
* Contextual information (request ID, session ID, etc.)
* Structured data fields for key values

### Log Management

* Rotating file handler (size-based rotation, 100MB limit)
* Retention of last 7 log files
* Optional syslog integration for centralized logging
* Log filtering by module and level

### Monitoring Integration

* Key metrics exposed through dedicated endpoint
* Integration with Prometheus for monitoring
* Alerting on critical conditions
* Performance metrics collection

This technical vision provides a comprehensive blueprint for the industrial-cognition project, balancing simplicity for
the initial prototype with extensibility for future requirements.
