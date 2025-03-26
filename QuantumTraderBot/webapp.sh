#!/bin/bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --reuse-port --reload 'webapp_launcher:app'