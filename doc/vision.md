# Industrial Cognition - Technical Vision

This document outlines the technical blueprint for the industrial-cognition project, serving as our starting point for
development. The vision follows the KISS principle while ensuring a solid foundation for future scaling.

## Technologies

- **Core Framework**: FastAPI (Python 3.9+)
    - Chosen for its asynchronous capabilities, automatic OpenAPI documentation, and ease of integration with modern
      Python libraries
- **Vision Processing**: OpenCV 4.5+
    - Direct integration for real-time image processing and computer vision algorithms
- **Database**: PostgreSQL 13+ with SQLAlchemy ORM
    - Used for persistent storage of detection results and system metadata
    - Alembic for database migrations
- **Containerization**: Docker 20.10+
    - Ensures consistent deployment across environments
- **Communication Protocol**: All robotic communication must use **binary Modbus over RS-485** physical layer. This supersedes all previous protocols. Data exchange must strictly adhere to Modbus RTU framing, utilizing specified function codes for read/write operations.

## Development Approach

- **Language**: Python 3.9+
    - Leveraging asyncio for asynchronous processing throughout the pipeline
- **Methodology**: Test-Driven Development (TDD)
    - Unit tests for all components
    - Integration tests for camera-to-API flow
    - End-to-end tests for complete detection pipeline
- **Code Quality**:
    - Type hints throughout
    - Black for code formatting
    - Flake8 for linting
    - MyPy for type checking

## Project Structure

```
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

```
+---------------------+
|    User Interface   |
|  (Web Dashboard)    |
+----------+----------+
           |
+----------v----------+
|      API Layer      |
|  (FastAPI)          |
|  - REST endpoints   |
|  - WebSocket server |
+----------+----------+
           |
+----------v----------+
|   Service Layer     |
|  - Orchestrates     |
|    components       |
|  - Business logic   |
+----------+----------+
           |
+----------v----------+  +------------------+
|  Vision Processing  |  |  Camera Interface |
|  - OpenCV pipeline  |<--|  - Hardware SDK  |
|  - Object detection |  |  - Frame capture |
+----------+----------+  +------------------+
           |
+----------v----------+
|   Data Access Layer |
|  - SQLAlchemy ORM   |
|  - Repository pattern|
+----------+----------+
           |
+----------v----------+
|    Data Storage     |
|  (PostgreSQL)       |
+---------------------+
```

## Data Model

### Detection Result

- **id**: UUID (primary key)
- **timestamp**: DateTime (UTC)
- **frame_id**: Integer (sequential frame counter)
- **object_type**: Enum ("metal_sheet", "detail", "clutter")
- **coordinates**: JSON
    - x: float (normalized 0-1)
    - y: float (normalized 0-1)
    - width: float (normalized 0-1)
    - height: float (normalized 0-1)
- **confidence**: Float (0.0-1.0)
- **metadata**: JSON (additional detection information)

### System Metadata

- **id**: UUID (primary key)
- **camera_status**: Enum ("connected", "disconnected", "error")
- **processing_fps**: Float (frames per second)
- **last_heartbeat**: DateTime (UTC)
- **configuration**: JSON (current system settings)

## User Interactions

### Real-Time Updates

- **WebSocket Endpoint**: `/ws/detections`
    - Pushes detection results in real-time as they're processed
    - Message format: JSON with detection data and metadata
    - Supports multiple concurrent clients

### Control Interface

- **REST API Endpoints**:
    - `GET /health` - System health check
    - `GET /status` - Current system status and statistics
    - `POST /control/start` - Start detection process
    - `POST /control/stop` - Stop detection process
    - `GET /detections` - Retrieve recent detection history
    - `GET /config` - Get current configuration
    - `PUT /config` - Update configuration

## Storage

- **Primary Database**: PostgreSQL 13+
    - Stores all detection results and system metadata
    - Configured with appropriate indexes for time-based queries
- **Migration Strategy**: Alembic
    - Version-controlled schema changes
    - Automated migration scripts
- **Backup Strategy**: Daily automated backups with 7-day retention
- **Data Retention**: Configurable policy (default: 30 days of detection data)

## Hardware Requirements

### Minimum Requirements

- **CPU**: Intel Core i7 or equivalent (8 cores, 16 threads)
- **GPU**: NVIDIA GTX 1660 or equivalent (6GB VRAM) - optional but recommended
- **RAM**: 16GB DDR4
- **Storage**: 512GB SSD (for OS, application, and database)
- **Camera Interface**: Compatible with industrial camera (GigE Vision, USB3 Vision, or Camera Link)
- **Network**: Gigabit Ethernet

### Recommended Requirements

- **CPU**: Intel Xeon or AMD Ryzen 9 (16+ cores)
- **GPU**: NVIDIA RTX 3070 or equivalent (8GB+ VRAM)
- **RAM**: 32GB DDR4
- **Storage**: 1TB NVMe SSD
- **Camera Interface**: Dedicated frame grabber card if required by camera
- **Network**: 10GbE capable

## Camera Integration

- **Interface**: Direct SDK integration with industrial camera manufacturer's API
- **Supported Protocols**: GigE Vision, USB3 Vision, Camera Link (depending on camera model)
- **Frame Rate**: Up to 60 FPS (configurable)
- **Resolution**: Up to 4K (configurable)
- **Trigger Modes**: Software trigger and continuous acquisition modes
- **Error Handling**: Automatic reconnection and error reporting

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
2. Receives real-time updates about object positions
3. Queries REST API for specific information as needed
4. Adjusts robot movements based on current workspace state

### Maintenance Mode

1. System can be paused via REST API
2. Camera continues capturing but processing is suspended
3. Allows for camera cleaning or workspace adjustments
4. Can be resumed via REST API

## Deployment

### Containerized Deployment

- **Docker Compose** configuration for all components:
    - Web API service
    - Database service
    - Optional monitoring services

### Configuration

- Environment variables for sensitive data (database credentials, API keys)
- YAML configuration files for system settings
- Configuration reload without restart

### Monitoring

- Health checks exposed via `/health` endpoint
- Metrics endpoint for performance monitoring
- Log aggregation and rotation

## Scaling

### Current Prototype

- Single-instance deployment
- All components run on one industrial server
- Vertical scaling through hardware upgrades

### Future Scaling Options

- **Horizontal Scaling**: Multiple detection nodes with load balancing
- **Microservices**: Split components into separate services
- **Edge Computing**: Distributed processing across multiple devices
- **Cloud Integration**: Offload historical data analysis to cloud

## Configuration Approach

### Configuration Layers

1. **Default Configuration**: Built into the application
2. **File Configuration**: YAML files for environment-specific settings
3. **Environment Variables**: For secrets and deployment-specific values
4. **Runtime Configuration**: API endpoints to modify settings without restart

### Configuration Management

- Configuration validation on startup and reload
- Schema validation using Pydantic
- Configuration change logging
- Rollback capability for failed configurations

## Logging Approach

### Logging Levels

- **DEBUG**: Detailed information for development and troubleshooting
- **INFO**: General operational information
- **WARNING**: Unexpected but recoverable conditions
- **ERROR**: Errors that don't prevent system operation
- **CRITICAL**: System-critical errors that may require intervention

### Log Structure

- JSON format for easy parsing and analysis
- Includes timestamp, level, module, and message
- Contextual information (request ID, session ID, etc.)
- Structured data fields for key values

### Log Management

- Rotating file handler (size-based rotation, 100MB limit)
- Retention of last 7 log files
- Optional syslog integration for centralized logging
- Log filtering by module and level

### Monitoring Integration

- Key metrics exposed through dedicated endpoint
- Integration with Prometheus for monitoring
- Alerting on critical conditions
- Performance metrics collection

This technical vision provides a comprehensive blueprint for the industrial-cognition project, balancing simplicity for
the initial prototype with extensibility for future requirements.