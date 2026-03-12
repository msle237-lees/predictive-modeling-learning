"""
@file main.py
@brief Root Click CLI entrypoint for predictive modeling learning project.
"""

import click


@click.group()
def pml():
    """Predictive Modeling Learning CLI"""
    pass


@click.group()
def regression():
    """Regression models"""
    pass


# Add model groups to pml
pml.add_command(regression)
