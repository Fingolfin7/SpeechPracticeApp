import numpy as np
import sounddevice as sd


class AudioPlayer:
    """
    A single‐stream player. Load float32 data with set_data(), then
    play(start_index) or stop(). Uses abort() to immediately cut off
    any previous playback.
    """

    def __init__(self, samplerate: int):
        self.sr = samplerate
        self.data = np.zeros((0,), dtype=np.float32)
        self.idx = 0

        # Larger blocksize + high latency for robustness
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
            latency="high",
        )

    def _callback(self, outdata, frames, time_info, status):
        # PortAudio callback: fill 'frames' samples or stop.
        if self.idx >= self.data.size:
            outdata.fill(0)
            raise sd.CallbackStop()
        end = self.idx + frames
        chunk = self.data[self.idx : end]
        n = chunk.shape[0]
        outdata[:n, 0] = chunk
        if n < frames:
            outdata[n:frames, 0] = 0
            self.idx = self.data.size
            raise sd.CallbackStop()
        self.idx = end

    def set_data(self, data: np.ndarray):
        """
        Load a new waveform (float32). Resets the play index.
        """
        # Stop any existing playback
        self.stop()
        # Ensure contiguous float32
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.idx = 0

    def play(self, start_index: int = 0):
        """
        Seek to start_index (clamped), then start the stream.
        Uses abort() for an immediate reset.
        """
        if self.data.size == 0:
            return
        self.idx = max(0, min(int(start_index), self.data.size - 1))
        # Always try to reset any previous state before starting
        try:
            if self.stream.active:
                self.stream.abort()
            else:
                # Some backends behave better if we abort even when stopped
                # after a CallbackStop; ignore errors when not active
                self.stream.abort()
        except Exception:
            pass
        self.stream.start()

    def pause(self):
        """
        Temporarily stop the stream without rewinding. Call play(idx) or
        play() again to resume.
        """
        if self.stream.active:
            # stop() keeps the current read position, abort() would reset
            self.stream.stop()


    def stop(self):
        """Stop playback immediately."""
        if self.stream.active:
            self.stream.abort()

    def close(self):
        """
        Finish with this player permanently (frees the device handle).
        """
        try:
            if self.stream.active:
                self.stream.abort()
        except Exception:
            pass
        self.stream.close()

    @property
    def active(self) -> bool:
        """Is playback currently running? Consider finished-at-end as not active."""
        try:
            return bool(self.stream.active) and (self.idx < self.data.size)
        except Exception:
            return False