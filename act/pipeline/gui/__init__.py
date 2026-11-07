"""ACT-side support for ACT-GUI: read-only observation of verification runs.

Engine code reads ``act.pipeline.gui.verifier_gui.enabled`` on hot paths and emits
records only while a GUI sink is registered.
"""
