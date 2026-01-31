// pybinding.cpp

#include <pybind11/pybind11.h>
#include <memory>
#include <string>
#include "model.hpp"
#include "tracker.hpp"
#include "pytrack.hpp"
#include "pytrack_factory.hpp"
#include "dump.hpp"
#include "dump_factory.hpp"

namespace py = pybind11;

using std::shared_ptr;
using std::string;

PYBIND11_MODULE(nucpy, m) {
  py::class_<NucPosModel>(m, "NucPosModel")
    .def(py::init<int,int,int,double,long>(),
	 py::arg("nucbp"),
	 py::arg("nbp"),
	 py::arg("llink"),
	 py::arg("mu"),
	 py::arg("seed"))    
    .def("run", &NucPosModel::run)    
    .def("reset", &NucPosModel::reset)
    .def("addTracker", &NucPosModel::addTracker);

  py::class_<Tracker, shared_ptr<Tracker> >(m, "Tracker");
  
  py::class_<PyTrackBase, Tracker, shared_ptr<PyTrackBase> >
    (m, "PyTracker")
    .def("values", &PyTrackBase::values)
    .def("reset", &PyTrackBase::reset);
  
  m.def("createPosTracker", [](int freq) {
    PyTrackFactory factory;
    return factory.createPosTrack(freq);
  });
  
  m.def("createEnergyTracker", [](int freq) {
    PyTrackFactory factory;
    return factory.createEnergyTrack(freq);
  });

  py::class_<Dump, Tracker, shared_ptr<Dump> >(m, "Dump");
  
  m.def("createPosDump", [](int freq, string file) {
    DumpFactory factory;
    return factory.createPosDump(freq, file);
  });
  
  m.def("createEnergyDump", [](int freq, string file) {
    DumpFactory factory;
    return factory.createEnergyDump(freq, file);
  });  
}
