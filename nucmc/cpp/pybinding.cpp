// pybinding.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include "dtype.hpp"
#include "model.hpp"
#include "tracker.hpp"
#include "pytrack.hpp"
#include "pytrack_factory.hpp"
#include "dump_h5.hpp"

namespace py = pybind11;

using std::shared_ptr;
using std::string;

using release_gil = py::call_guard<py::gil_scoped_release>;

PYBIND11_MODULE(nucmc_cpp, m) {
  py::class_<NucPosModel>(m, "NucPosModel")
    .def(py::init<int,int,int,double,ulint>(),
	 py::arg("nucbp"), py::arg("nbp"), py::arg("llink"), py::arg("mu"),
	 py::arg("seed"), release_gil())
    .def("run", &NucPosModel::run, release_gil())
    .def("reset", &NucPosModel::reset, release_gil())
    .def("initMeth", [](NucPosModel& self, py::array_t<double> array) {
      py::buffer_info info = array.request();
      std::vector<double> data(static_cast<double*>(info.ptr),
			       static_cast<double*>(info.ptr) + info.size);
      self.initMeth(data);
    }, py::arg("data"), release_gil())
    .def("addTracker", &NucPosModel::addTracker, release_gil());

  py::class_<Tracker, shared_ptr<Tracker> >(m, "Tracker");
  
  py::class_<PyTrackBase, Tracker, shared_ptr<PyTrackBase> >(m, "PyTracker")
    .def("values", &PyTrackBase::values, release_gil())
    .def("reset", &PyTrackBase::reset, release_gil());
  
  m.def("createPositionTracker", [](lint freq) {
    PyTrackFactory factory;
    return factory.createPosTrack(freq);
  }, py::arg("freq"), release_gil());
  
  m.def("createEnergyTracker", [](lint freq) {
    PyTrackFactory factory;
    return factory.createEnergyTrack(freq);
  }, py::arg("freq"), release_gil());

  py::class_<DumpH5, Tracker, shared_ptr<DumpH5> > dumpcls(m, "Dump");
  
  py::enum_<DumpH5::OutputType>(dumpcls, "OutputType", py::arithmetic())
    .value("Energy", DumpH5::OutputType::Energy)
    .value("Position", DumpH5::OutputType::Position)
    .value("Temp", DumpH5::OutputType::Temp)
    .value("All", DumpH5::OutputType::All);
  
  m.def("createDump", [](lint freq, const string& file,
			 DumpH5::OutputType otype) ->
	shared_ptr<Tracker> {
	  return std::make_shared<DumpH5>(freq, file, otype);
	}, py::arg("freq"), py::arg("file"), py::arg("out_type"),
	release_gil());
}
