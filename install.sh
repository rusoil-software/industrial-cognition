#!/bin/bash

# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Jun-31
# ==============================================================================


# --- Industrial Cognition System Installer Script (V6.0 - Full Startup) ---
# This script builds all necessary service images and brings up the entire stack 
# for full local development parity.

set -e # Exit immediately if a command exits with a non-zero status.

echo "====================================================================================="
echo "       Starting Industrial Cognition System Deployment (Full Stack)             "
echo "====================================================================================="

WORK_DIR=$(pwd)

echo "-- Checking for core dependencies (Docker/Docker-Compose)... --"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "Error: Neither 'docker compose' nor 'docker-compose' found."
    echo "ERROR: Docker Compose is not installed or not in PATH. Please install Docker Compose."
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    echo "Exiting setup."
    echo "=============================================================================="
    echo "SETUP FAILED: Missing Dependency"
    echo "Please install Docker Compose and re-run this script."
    echo "=============================================================================="
    echo "Press any key to exit..."
    read 1
    exit 1
fi

echo "-- Docker Compose found. Proceeding... ---"
echo "--- Building all necessary service images... ---"
# Build the API Gateway and the Vision Worker first.
echo "--- Building API Gateway... ---"
$COMPOSE_CMD build api
echo "-------------------------------------------------------------------------------------"
# Build the worker service using the Dockerfile placed in src/vision/
echo "--- Building Vision Worker... ---"
$COMPOSE_CMD build vision
echo "-------------------------------------------------------------------------------------"

if [ $? -ne 0 ]; then
    echo "============================================================================="
    echo "BUILD FAILED: Check error messages above. One or more services failed to build."
    echo "The build failure usually points to missing dependencies in the Dockerfile or external system issues."
    echo "=============================================================================="
    exit 1
fi

echo "====================================================================================="
echo "BUILD COMPLETE."
echo "The following services are now ready to run:"
echo "  - api (HTTP/WebSocket)"
echo "  - vision (Message Consumer)"
echo "====================================================================================="
echo ""

echo "--- STARTING CORE SERVICES (Requires Infrastructure to be online) ---"
echo "NOTE: This assumes RabbitMQ, Redis, and MinIO are accessible and running externally."
echo "If these are not running, the startup will fail gracefully with connection errors."
echo "-------------------------------------------------------------------------------------"

# Attempt to start the application stack.
$COMPOSE_CMD up -d api vision

if [ $? -ne 0 ]; then
    echo "====================================================================================="
    echo "WARNING: Startup failed for API Gateway or Worker."
    echo "Action: Check if the external infrastructure (Redis, RabbitMQ, MinIO) is running."
    echo "====================================================================================="
else
    echo "====================================================================================="
    echo "SUCCESS: Core services (API Gateway & Vision Worker) are running in the background."
    echo "====================================================================================="
fi