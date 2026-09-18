import sys
import logging

def getLogger(name):
    logger = logging.getLogger(name)
    
    # Prevents duplicate logs if this function is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(asctime)s %(name)s]: %(message)s', '%H:%M:%S'))
        
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

from datetime import datetime


class ProgressBar:
    """A small, dependency-free progress bar for long-running terminal jobs."""

    def __init__(self, total, name="STRESSCF", width=32, stream=None, enabled=None):
        self.total = max(1, int(total))
        self.name = name
        self.width = max(1, int(width))
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = (
            self.stream.isatty() if enabled is None else bool(enabled)
        )
        self._last_line = ""
        self._rendered = False

    def update(self, current):
        """Redraw the bar with *current* completed iterations."""
        if not self.enabled:
            return

        current = min(max(0, int(current)), self.total)
        filled = self.width * current // self.total
        bar = "█" * filled + "░" * (self.width - filled)
        timestamp = datetime.now().strftime("%H:%M:%S")
        counter_width = len(str(self.total))
        self._last_line = (
            f"[{timestamp} {self.name}]: "
            f"[{bar}] {current:>{counter_width}}/{self.total}"
        )
        self.stream.write(f"\r{self._last_line}")
        self.stream.flush()
        self._rendered = True

    def finish(self):
        """End the current progress line so subsequent logs start cleanly."""
        if self.enabled and self._rendered:
            self.stream.write(f"\r{self._last_line}\r\n")
            self.stream.flush()
            self._rendered = False