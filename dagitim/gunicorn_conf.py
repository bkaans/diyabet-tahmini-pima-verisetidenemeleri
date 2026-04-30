"""Gunicorn calisma ayarlari."""

from __future__ import annotations

import multiprocessing
import os


bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
workers = int(os.getenv("GUNICORN_WORKERS", max(2, multiprocessing.cpu_count() // 2)))
worker_class = "uvicorn.workers.UvicornWorker"

wsgi_app = "uygulama.main:app"
chdir = os.getenv("GUNICORN_CHDIR", ".")

timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
accesslog = "-"
errorlog = "-"
