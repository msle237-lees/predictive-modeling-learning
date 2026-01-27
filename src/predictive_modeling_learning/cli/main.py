"""
@file main.py
@brief Root Click CLI entrypoint for predictive modeling learning project.
"""

from __future__ import annotations

import click

from predictive_modeling_learning.cli.regression import regression

@click.group()
def cli() -> None:
    """
    @brief Predictive Modeling CLI root group.

    Subcommands are grouped by modeling category (regression, classification, etc.).
    """
    pass

cli.add_command(regression)

if __name__ == "__main__":
    cli()
