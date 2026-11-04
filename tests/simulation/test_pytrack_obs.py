# test_pytrack_obs.py

import numpy as np
import pytest
from catella_cpp import (
    Dump,
    NucPosMCMC,
    createDump,
    createEnergyTracker,
    createPositionTracker,
)

OT = Dump.OutputType
POS_KEYS = {"position_flat", "position_offset"}


def _model(seed=3, nbp=400):
    model = NucPosMCMC(147, nbp, 20, 1.0, -1.0, seed)
    eseq = np.random.default_rng(seed).uniform(-2.0, 2.0, nbp)
    model.setEnergy(eseq.tolist(), 5.0)
    return model


def _run(out_type, nsweep=10, freq=2, start_temp=1.0, end_temp=0.1):
    model = _model()
    dump = createDump(freq, out_type)
    model.addTracker(dump)
    model.run(nsweep, start_temp, end_temp, NucPosMCMC.Cooling.Linear)
    return dump, model


def _combine(*otypes):
    value = 0
    for otype in otypes:
        value |= int(otype)
    return OT(value)


class TestPyTrackObsValues:
    @pytest.mark.parametrize(
        "out_type, keys",
        [
            (OT.All, {"time", "energy", "temp"} | POS_KEYS),
            (OT.Energy, {"time", "energy"}),
            (OT.Temp, {"time", "temp"}),
            (OT.Position, {"time"} | POS_KEYS),
            (_combine(OT.Temp, OT.Energy), {"time", "energy", "temp"}),
        ],
    )
    def test_keys_match_out_type(self, out_type, keys):
        dump, _ = _run(out_type)
        assert set(dump.values()) == keys

    def test_dtypes(self):
        values = _run(OT.All)[0].values()
        assert values["time"].dtype == np.int64
        assert values["energy"].dtype == np.float64
        assert values["temp"].dtype == np.float64
        assert values["position_flat"].dtype == np.int32
        assert values["position_offset"].dtype == np.uint64

    def test_time_follows_print_freq(self):
        values = _run(OT.All, nsweep=10, freq=2)[0].values()
        np.testing.assert_array_equal(values["time"], np.arange(0, 11, 2))

    def test_arrays_are_aligned(self):
        values = _run(OT.All)[0].values()
        n = len(values["time"])
        assert len(values["energy"]) == n
        assert len(values["temp"]) == n
        assert len(values["position_offset"]) == n + 1

    def test_offsets_delimit_flat_positions(self):
        values = _run(OT.Position, nsweep=30, freq=1)[0].values()
        offset = values["position_offset"]
        assert offset[0] == 0
        assert offset[-1] == len(values["position_flat"])
        assert np.all(np.diff(offset.astype(np.int64)) >= 0)

    def test_matches_existing_trackers(self):
        model = _model()
        dump = createDump(1, OT.All)
        pos = createPositionTracker(1)
        energy = createEnergyTracker(1)
        for tracker in (dump, pos, energy):
            model.addTracker(tracker)
        model.run(20, 1.0, 0.1, NucPosMCMC.Cooling.Linear)
        values = dump.values()
        flat, offset = values["position_flat"], values["position_offset"]
        frames = [
            flat[offset[i] : offset[i + 1]] for i in range(len(offset) - 1)
        ]
        expected = pos.values()
        assert len(frames) == len(expected)
        for got, want in zip(frames, expected):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(values["energy"], energy.values())

    def test_temp_follows_linear_cooling(self):
        values = _run(OT.Temp, nsweep=10, freq=1)[0].values()
        assert values["temp"][0] == pytest.approx(1.0)
        assert values["temp"][-1] == pytest.approx(0.1)

    def test_zero_sweeps_records_initial_frame(self):
        values = _run(OT.All, nsweep=0)[0].values()
        np.testing.assert_array_equal(values["time"], [0])
        assert len(values["position_flat"]) == 0
        np.testing.assert_array_equal(values["position_offset"], [0, 0])


class TestPyTrackObsReset:
    def test_reset_clears_values(self):
        dump, _ = _run(OT.All)
        dump.reset()
        values = dump.values()
        for key in ("time", "energy", "temp", "position_flat"):
            assert len(values[key]) == 0
        np.testing.assert_array_equal(values["position_offset"], [0])
