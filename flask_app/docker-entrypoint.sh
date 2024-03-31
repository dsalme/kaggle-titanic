#!/bin/bash
gunicorn -w 1 app:app -b 0.0.0.0 --timeout 300 --reload --reload-extra-file 'templates/index.html'