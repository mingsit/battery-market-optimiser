#!/usr/bin/env python
"""Django management entry point for the battery dashboard."""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Django is required for the dashboard. "
            "Install it with: pip install django"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
