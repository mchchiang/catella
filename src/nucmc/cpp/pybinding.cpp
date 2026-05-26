// pybinding.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <vector>
#include "dtype.hpp"
#include "model.hpp"
#include "tracker.hpp"
#include "pytrack.hpp"
#include "pytrack_factory.hpp"
#include "dump_h5.hpp"

namespace py = pybind11;

using std::shared_ptr;
using std::string;
using std::vector;

PYBIND11_MODULE(nucmc_cpp, m) {
  py::class_<NucPosModel> modelcls(m, "NucPosModel");
  modelcls.def(py::init<int,int,int,double,ulint>(),
	       py::arg("nucbp"), py::arg("nbp"), py::arg("llink"),
	       py::arg("mu"), py::arg("seed"));
  modelcls.def("run", &NucPosModel::run);
  modelcls.def("reset", &NucPosModel::reset);
  modelcls.def("addTracker", &NucPosModel::addTracker);
  modelcls.def("getSeqEnergy", &NucPosModel::getSeqEnergy);
  modelcls.def("setSeqEnergy", [](NucPosModel& self, vector<double> pseq,
				   double emax) {
    self.setSeqEnergy(pseq, emax);
  }, py::arg("pseq"), py::arg("emax"));

  py::enum_<NucPosModel::Cooling>(modelcls, "Cooling")
    .value("Linear", NucPosModel::Cooling::Linear)
    .value("Geometric", NucPosModel::Cooling::Geometric)
    .value("Constant", NucPosModel::Cooling::Constant);
  
  py::class_<Tracker, shared_ptr<Tracker> >(m, "Tracker");
  
  py::class_<PyTrackBase, Tracker, shared_ptr<PyTrackBase> >(m, "PyTracker")
    .def("values", &PyTrackBase::values)
    .def("reset", &PyTrackBase::reset);
  
  m.def("createPositionTracker", [](lint freq) {
    PyTrackFactory factory;
    return factory.createPosTrack(freq);
  }, py::arg("freq"));
  
  m.def("createEnergyTracker", [](lint freq) {
    PyTrackFactory factory;
    return factory.createEnergyTrack(freq);
  }, py::arg("freq"));

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
	}, py::arg("freq"), py::arg("file"), py::arg("out_type"));
}
