// pytrack_obs.cpp

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dtype.hpp"
#include "mcmc.hpp"
#include "pytrack_obs.hpp"

namespace py = pybind11;

using std::vector;

// Copy a vector into a new 1D NumPy array
template <typename T>
static py::array_t<T> toArray(const vector<T>& vec) {
  return py::array_t<T>(vec.size(), vec.data());
}

PyTrackObs::PyTrackObs(lint printFreq, OutputType otype) :
  PyTrackBase(printFreq), outType(otype), posOffset{0} {}

bool PyTrackObs::has(OutputType otype) const {
  return static_cast<uint>(outType) & static_cast<uint>(otype);
}

void PyTrackObs::initialize(lint time, const NucPosMCMC& model) {}

void PyTrackObs::update(lint time, const NucPosMCMC& model) {
  times.push_back(time);
  if (has(OutputType::Energy)) {
    energy.push_back(model.getEnergy());
  }
  if (has(OutputType::Temp)) {
    temp.push_back(model.getTemp());
  }
  if (has(OutputType::Position)) {
    const auto& pos = model.getNucPos();
    posFlat.insert(posFlat.end(), pos.begin(), pos.end());
    posOffset.push_back(posFlat.size());
  }
}

void PyTrackObs::finalize(lint time, const NucPosMCMC& model) {}

void PyTrackObs::reset() {
  times.clear();
  energy.clear();
  temp.clear();
  posFlat.clear();
  posOffset.assign(1, 0);
}

py::object PyTrackObs::values() {
  py::dict out;
  out["time"] = toArray(times);
  if (has(OutputType::Energy)) {
    out["energy"] = toArray(energy);
  }
  if (has(OutputType::Temp)) {
    out["temp"] = toArray(temp);
  }
  if (has(OutputType::Position)) {
    out["position_flat"] = toArray(posFlat);
    out["position_offset"] = toArray(posOffset);
  }
  return out;
}
