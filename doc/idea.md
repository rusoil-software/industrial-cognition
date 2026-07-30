# Industrial Cognition - AI Service Idea

## Overview

The `industrial-cognition` AI service is designed as a compact, high-performance machine vision system for industrial automation. It is intended for deployment on-site on an industrial server equipped with a professional industrial camera to detect and track metal sheets, produced details, and clutter within a robot's working zone.

## Core Purpose

The primary function of the system — the machine vision module — is a combined hardware and software solution that:

- Captures images via an industrial camera mounted above the robot's working area (e.g., a robotic arm).
- Processes the video stream in real time using OpenCV algorithms for object detection.
- Determines precise coordinates of objects (metal sheets, parts, debris) on the captured images and communicates these coordinates to the **robot controller via Modbus/RS-485** communication protocol.
- Exposes this spatial data to other robotic systems through a real-time, asynchronous communication channel.

## Technical Architecture

### Backend
- **Framework**: FastAPI (Python)
- **Purpose**: Provides a web control panel and REST/websocket API for system monitoring and integration.
- **Real-Time Communication**: Utilizes WebSocket connections to push object detection results to connected clients without requiring repeated polling requests.

### Vision Processing
- **Library**: OpenCV (direct integration)
- **Functionality**: Real-time image analysis to identify objects based on predefined parameters such as size thresholds, shape, and contrast.
- **Processing Target**: Runs on a high-performance industrial computer capable of handling continuous video input and parallel computation.

### Data Management
- **Database Integration**: Detected object coordinates and metadata are saved into a persistent database for logging, analytics, and system recovery.
- **Asynchronous Design**: The entire pipeline — from image capture to database write and WebSocket broadcast — is implemented asynchronously to ensure low latency and high throughput.

## Deployment Environment

- **Hardware**: On-premise industrial server with:
  - Professional-grade industrial camera (top-down view)
  - High-performance CPU/GPU for real-time AI processing
  - Stable network connectivity for system integration
- **Software**: Containerized deployment (Docker) with FastAPI backend, OpenCV, and database services (e.g., PostgreSQL via Alembic for migrations).

## Integration & Use Case

The system enables autonomous robotic systems to dynamically adapt to changing workspaces by providing real-time spatial awareness. For example, a robotic arm can query the service to locate the next metal sheet to process or processed part or avoid areas with clutter.
