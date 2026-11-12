"""Shared, framework-free audio/text utilities.

This package must not import ``cosyvoice_runtime``, ``cosyvoice_training``,
``cosyvoice_data`` or ``cosyvoice_evaluation``. Everything else may depend on
``cosyvoice_common``; the dependency direction stays one-way so the packages
can be built, published and versioned independently.
"""
