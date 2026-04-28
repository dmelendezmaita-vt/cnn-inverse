import logging, sys


class LibraryLogFilter(logging.Filter):
    """
    A logging filter that suppresses log records of a given level (or below)
    from a specified library/module (by logger name prefix).

    This is useful when you want to keep verbose logging enabled for your
    own code but silence noisy logs from third-party libraries like matplotlib,
    urllib3, asyncio, etc.

    Args:
        library_prefix : str
            The prefix of the logger name to filter (e.g., "matplotlib").
        level : int
            The logging level at or below which messages will be suppressed
            (e.g., logging.DEBUG, logging.INFO).


    Usage notes:

    - Import matplotlib first (so it can create handlers), then attach this filter.
    - Attach it to the "matplotlib" logger (and to any existing handlers) so
      matplotlib's own handlers are filtered.
    - Alternatively, if you simply want to block all DEBUG logs from matplotlib,
      calling `logging.getLogger("matplotlib").setLevel(logging.INFO)` is simpler.

    Example:

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)
    >>> logging.getLogger().addFilter(LibraryLogFilter("matplotlib", logging.DEBUG))
    >>> mpl_logger = logging.getLogger("matplotlib")
    >>> mpl_logger.debug("This will NOT be shown")
    >>> mpl_logger.info("This WILL be shown")
    >>> logging.getLogger(__name__).debug("Your own debug message WILL be shown")

    Source:

    - Initial implementation with OpenAI GPT-5
    """

    def __init__(self, library_prefix: str, level: int):
        """
        Initialize the filter.

        Args:
            library_prefix : str
                Logger name prefix for which logs should be filtered.
            level : int
                Logging level to suppress (any record with level <= this will be filtered).
        """
        super().__init__()
        self.library_prefix = library_prefix
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Decide whether the specified record should be logged.

        Args:
            record : logging.LogRecord
                The log record to be filtered.

        Returns:
            bool
                True if the record should be logged, False if it should be suppressed.
        """
        name = record.name
        # match, for example, "matplotlib" and "matplotlib.submodule"
        if name == self.library_prefix or name.startswith(self.library_prefix + "."):
            # suppress records at or below the configured level
            if record.levelno <= self.level:
                return False
        return True


def logging_set_up(
    log_file_basename,
    log_file_format="%(asctime)s - %(name)s - %(levelname)-8s %(message)s",
    log_console_level=logging.WARNING,
    log_console_format="%(name)s - %(levelname)-8s %(message)s",
    filter_matplotlib_level=logging.DEBUG,
):
    """Sets up loggers."""
    # set up logging of debug to file
    # Note: The level for `basicConfig` has to be lower than all other handlers added afterwards.
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_file_format,
        datefmt="%y-%m-%d %H:%M",
        filename=str(log_file_basename) + "_debug.log",
        filemode="w",
    )

    # set up formatter
    formatter = logging.Formatter(log_file_format)

    # set up logging of info to file
    handler = logging.FileHandler(str(log_file_basename) + "_info.log", mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    # set up logging of errors to file
    handler = logging.FileHandler(str(log_file_basename) + "_error.log", mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(handler)

    # set up logging to console
    if log_console_level is not None:
        # define a Handler which writes messages to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(log_console_level)
        # set a format for console use
        formatter = logging.Formatter(log_console_format)
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(console)

    # filter messages from Matplotlib at the specified level
    if filter_matplotlib_level:
        mpl_filter = LibraryLogFilter("matplotlib", filter_matplotlib_level)
        # attach filter to root handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(mpl_filter)


def logging_get_logger(name=""):
    """Gets a logger by its name. Returns root logger by default."""
    return logging.getLogger(name)


def logging_add_debug_handler(logger, stream=sys.stderr):
    """Returns a handler that was added to print debug-level messages to a stream (e.g., stderr, stdout).
    To remove the handler, store the returned handler and (later) call `logger.removeHandler(handlerVariable)`.
    """
    debugHandler = logging.StreamHandler(stream)
    debugHandler.setLevel(logging.DEBUG)
    logger.addHandler(debugHandler)
    return debugHandler
