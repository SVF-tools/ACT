"""ACT-side support for ACT-GUI: observation and control of verification and fuzzing runs.

``common_gui`` holds the shared channel, record base, option and run-control API;
``verifier_gui`` and ``fuzzer_gui`` build their records and options on it. Engine code
reads ``act.pipeline.gui.common_gui.enabled`` on hot paths and emits records only while
a GUI sink is registered.
"""
