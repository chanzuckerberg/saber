# SAM2-ET Annotation GUI Web Server

A web-based annotation GUI for SAM2 segmentations with Dask integration for distributed processing.

## Features

- ✨ Web-based interface accessible from any browser
- 🚀 Dask integration for distributed processing
- 📦 Zarr file support for efficient data storage
- 🔄 Real-time synchronization across multiple users
- 🖥️ Command-line interface for easy deployment
- 🌐 Remote access capability via SSH tunneling

## Quick Start

### Basic Usage

```bash
# Start the server with local data
saber web --data /path/to/zarr/data
```

### Remote Access

```bash
# On local machine (SSH tunnel)
ssh -L 9090:localhost:9090 user@remote-server

# On remote server
saber web --data /data/zarr

# Access at http://localhost:9090
```

## Command-Line Options

- `--input, -i`: Path to local data directory (required)
- `--output, -o`: Output path for saved annotations
- `--port`: Port to run the server on (default: 9090)
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

## License

MIT License
