# SAM2-ET Annotation GUI Web Server

A web-based annotation GUI for SAM2 segmentations with Dask integration for distributed processing.

## Features

- ✨ Web-based interface accessible from any browser
- 🚀 Dask integration for distributed processing
- 📦 Zarr file support for efficient data storage
- 🔄 Real-time synchronization across multiple users
- 🖥️ Command-line interface for easy deployment
- 🌐 Remote access capability via SSH tunneling

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/annotation-gui.git
cd annotation-gui

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Quick Start

### Basic Usage

```bash
# Start the server with local data
annotation-gui --data /path/to/zarr/data --port 8080
```

### With Dask Cluster

```bash
# Connect to existing Dask scheduler
annotation-gui --data /path/to/data --dask-scheduler tcp://localhost:8786

# Start with local Dask cluster
annotation-gui --data /path/to/data --workers 8
```

### Remote Access

```bash
# On remote server
annotation-gui --data /data/zarr --host 0.0.0.0 --port 8080

# On local machine (SSH tunnel)
ssh -L 8080:localhost:8080 user@remote-server

# Access at http://localhost:8080
```

## Command-Line Options

- `--data, -d`: Path to local data directory (required)
- `--output, -o`: Output path for saved annotations
- `--port, -p`: Port to run the server on (default: 8080)
- `--host, -h`: Host to bind to (default: 0.0.0.0)
- `--dask-scheduler`: Dask scheduler address
- `--workers, -w`: Number of Dask workers (default: 4)
- `--class-names, -c`: Comma-separated class names
- `--debug`: Run in debug mode

## API Endpoints

- `GET /`: Main web interface
- `GET /api/runs`: List all available runs
- `GET /api/runs/<run_id>`: Get data for specific run
- `POST /api/save`: Save annotations
- `GET /api/status`: Server and Dask status

## Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8080
CMD ["annotation-gui", "--data", "/data", "--port", "8080"]
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Start development server
annotation-gui --data ./test_data --debug
```

## Architecture

The application consists of:

1. **Flask Web Server**: Serves the HTML interface and REST API
2. **Dask Processor**: Handles distributed data processing
3. **Zarr Backend**: Efficient storage and retrieval of segmentation data
4. **Web Interface**: Interactive annotation GUI

## License

MIT License
