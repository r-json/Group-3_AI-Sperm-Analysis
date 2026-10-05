"""Desktop application (PySide6), Model-View-Presenter.

* ``view.py``      - widgets only; emits user intents as signals; no ML code
* ``presenter.py`` - application logic; owns the Predictor; talks to the view
* ``worker.py``    - runs predictions off the UI thread
* ``main.py``      - composition root (wires registry, presenter and view)
"""
