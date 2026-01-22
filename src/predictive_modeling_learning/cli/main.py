"""
@file main.py
@brief Root Click CLI entrypoint for predictive modeling learning project.
"""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """
    @brief Predictive Modeling CLI root group.

    Subcommands are grouped by modeling category (regression, classification, etc.).
    """
    pass


if __name__ == "__main__":
    cli()
