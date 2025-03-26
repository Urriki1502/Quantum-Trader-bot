#!/bin/bash

# Start the web application using Gunicorn with our config file
# and direct all output to /dev/null to suppress noise
gunicorn -c gunicorn_config.py wsgi:app > /dev/null 2>&1 &

# Wait a moment to allow the server to start
sleep 2

# Show a minimal startup message
echo "Web application started on http://0.0.0.0:5000"