import numpy as np
import pytest

from spectral.pipeline.window import SlidingWindow, StreamBuffer


class TestSlidingWindow:
    def test_window_count(self):
        sw = SlidingWindow(size=10, step=1)
        assert sw.window_count(100) == 91

    def test_window_count_with_step(self):
        sw = SlidingWindow(size=10, step=5)
        assert sw.window_count(100) == 19

    def test_window_shape(self):
        sw = SlidingWindow(size=10, step=1)
        data = np.arange(50, dtype=float)
        windows = sw.transform(data)

        for w in windows:
            assert len(w) == 10

    def test_step_size_non_overlapping(self):
        sw = SlidingWindow(size=10, step=10)
        data = np.arange(30, dtype=float)
        windows = sw.transform(data)

        assert len(windows) == 3
        np.testing.assert_array_equal(windows[0], np.arange(10))
        np.testing.assert_array_equal(windows[1], np.arange(10, 20))

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            SlidingWindow(size=0)

    def test_data_smaller_than_window(self):
        sw = SlidingWindow(size=100, step=1)
        data = np.arange(50, dtype=float)
        windows = sw.transform(data)
        assert len(windows) == 0


class TestStreamBuffer:
    def test_yields_on_full(self):
        buffer = StreamBuffer(size=5, step=1)

        results = []
        for i in range(10):
            window = buffer.push(float(i))
            if window is not None:
                results.append(window)

        # first window at index 4 (0,1,2,3,4), then one per step
        assert len(results) == 6

    def test_step_controls_frequency(self):
        buffer = StreamBuffer(size=5, step=3)

        results = []
        for i in range(20):
            window = buffer.push(float(i))
            if window is not None:
                results.append(window)

        # should yield less frequently with larger step
        assert len(results) < 20

    def test_reset_clears_buffer(self):
        buffer = StreamBuffer(size=5)
        for i in range(3):
            buffer.push(float(i))

        assert buffer.current_size == 3
        buffer.reset()
        assert buffer.current_size == 0

    def test_is_full_property(self):
        buffer = StreamBuffer(size=3)
        assert not buffer.is_full
        buffer.push(1.0)
        buffer.push(2.0)
        assert not buffer.is_full
        buffer.push(3.0)
        assert buffer.is_full
