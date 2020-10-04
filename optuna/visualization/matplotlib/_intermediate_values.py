from optuna import core
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes


@core._experimental.experimental("2.2.0")
def plot_intermediate_values() -> Axes:
    raise NotImplementedError("To be implemented soon.")
