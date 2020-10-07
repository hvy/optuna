import colorlog

import optuna
from optuna.core.logging import CRITICAL  # NOQA
from optuna.core.logging import DEBUG  # NOQA
from optuna.core.logging import disable_default_handler  # NOQA
from optuna.core.logging import disable_propagation  # NOQA
from optuna.core.logging import enable_default_handler  # NOQA
from optuna.core.logging import enable_propagation  # NOQA
from optuna.core.logging import ERROR  # NOQA
from optuna.core.logging import FATAL  # NOQA
from optuna.core.logging import get_logger  # NOQA
from optuna.core.logging import get_verbosity  # NOQA
from optuna.core.logging import INFO  # NOQA
from optuna.core.logging import set_verbosity  # NOQA
from optuna.core.logging import WARN  # NOQA
from optuna.core.logging import WARNING  # NOQA


def create_default_formatter() -> colorlog.ColoredFormatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """

    return colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s"
    )


optuna.core.logging._configure_library_root_logger()
with optuna.core.logging._lock:
    assert optuna.core.logging._default_handler is not None
    optuna.core.logging._default_handler.setFormatter(create_default_formatter())
