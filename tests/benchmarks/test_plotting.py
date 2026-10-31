# test_plotting.py

import numpy as np
import plotting
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg


class TestOccupancyFigure:
    @pytest.mark.parametrize("ntrack", range(3, 10))
    def test_full_page_fits(self, ntrack):
        per_page = plotting._occupancy_per_page(ntrack)
        occ = np.zeros(5000)
        tracks = [("FiberHMM (retrained)", occ, "#000000")] * (ntrack - 1)
        molecules = [
            (f"test_{i}", np.full(5000, 0.5), tracks)
            for i in range(per_page)
        ]
        fig = plotting._occupancy_figure(
            molecules, "title\nsubtitle", 5000, 150
        )
        renderer = FigureCanvasAgg(fig).get_renderer()
        width, height = fig.get_size_inches()
        box = fig.get_tightbbox(renderer)
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= width and box.y1 <= height
        boxes = [ax.get_window_extent(renderer) for ax in fig.axes]
        titles = [ax.title for ax in fig.axes if ax.get_title(loc="left")]
        assert len(titles) == per_page
        for title in titles:
            extent = title.get_window_extent(renderer)
            own = title.axes.get_window_extent(renderer)
            assert not any(
                extent.overlaps(b) for b in boxes if b is not own
            )
