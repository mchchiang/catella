// pytrack.hpp

#ifndef PYTRACK_HPP
#define PYTRACK_HPP

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dtype.hpp"
#include "model.hpp"
#include "tracker.hpp"

namespace py = pybind11;

class NucPosModel;

class PyTrackBase : public Tracker {

public:
  PyTrackBase(lint freq) : Tracker(freq) {};
  virtual ~PyTrackBase() = default;
  virtual void reset() = 0;
  virtual py::object values() = 0;
};

template <class Derived, typename T>
class PyTrack : public PyTrackBase {

protected:
  std::vector<T> data;
  
public:
  using value_type = T;

  PyTrack(lint freq) : PyTrackBase(freq) {};
  
  void reset() override {data.clear();}

  void initialize(lint time, const NucPosModel& model) override final {}
  
  void update(lint time, const NucPosModel& model) override final {
    static_cast<Derived*>(this)->track(time, model);
  }

  void finalize(lint time, const NucPosModel& model) override final {}  

  py::object values() override {
    if constexpr (std::is_arithmetic<T>::value) {
      // Scalar type -> 1D NumPy array
      return py::array_t<T>(data.size(), data.data());
    } else {
      // Non-POD type -> return list of NumPy arrays
      py::list out;
      for (const auto &v : data)
	out.append(py::array_t<typename T::value_type>(v.size(), v.data()));
      return out;
    }
  }
};

#endif
