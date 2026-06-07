import logging
import os

from openpibot.server import logging as logging_mod


def test_configure_logging_preserves_cli_level_on_app_startup(tmp_path):
    root = logging.getLogger()
    old_level = root.level
    old_env_file = os.environ.get("OPENPIBOT_LOG_FILE")
    old_env_level = os.environ.get("OPENPIBOT_LOG_LEVEL")
    old_handlers = list(root.handlers)
    try:
        log_file = tmp_path / "server.log"

        logging_mod.configure_logging(level="debug", log_file=log_file)
        root.setLevel(logging.WARNING)
        logging_mod.configure_logging()

        assert root.level == logging.DEBUG
        assert logging_mod.current_log_file() == log_file
    finally:
        root.setLevel(old_level)
        for handler in list(root.handlers):
            if handler not in old_handlers:
                root.removeHandler(handler)
                handler.close()
        if old_env_file is None:
            os.environ.pop("OPENPIBOT_LOG_FILE", None)
        else:
            os.environ["OPENPIBOT_LOG_FILE"] = old_env_file
        if old_env_level is None:
            os.environ.pop("OPENPIBOT_LOG_LEVEL", None)
        else:
            os.environ["OPENPIBOT_LOG_LEVEL"] = old_env_level
