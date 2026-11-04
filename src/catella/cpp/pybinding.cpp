// pybinding.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <vector>
#include "dtype.hpp"
#include "mcmc.hpp"
#include "tm.hpp"
#include "tracker.hpp"
#include "pytrack.hpp"
#include "pytrack_factory.hpp"
#include "pytrack_obs.hpp"

namespace py = pybind11;

using std::shared_ptr;
using std::string;
using std::vector;

PYBIND11_MODULE(catella_cpp, m) {
  py::class_<NucPosMCMC> mcmccls(m, "NucPosMCMC");
  mcmccls.def(py::init<int,int,int,double,double,ulint>(),
	      py::arg("nucbp"), py::arg("nbp"), py::arg("llink"),
	      py::arg("elink"), py::arg("mu"), py::arg("seed"));
  mcmccls.def("run", &NucPosMCMC::run);
  mcmccls.def("reset", &NucPosMCMC::reset);
  mcmccls.def("addTracker", &NucPosMCMC::addTracker);
  mcmccls.def("getSeqEnergy", &NucPosMCMC::getSeqEnergy);
  mcmccls.def("setEnergy", [](NucPosMCMC& self, vector<double> eseq,
			      double emax) {
    self.setEnergy(eseq, emax);
  }, py::arg("eseq"), py::arg("emax"));

  py::enum_<NucPosMCMC::Cooling>(mcmccls, "Cooling")
    .value("Linear", NucPosMCMC::Cooling::Linear)
    .value("Geometric", NucPosMCMC::Cooling::Geometric)
    .value("Constant", NucPosMCMC::Cooling::Constant);

  py::class_<NucPosTM> tmcls(m, "NucPosTM");
  tmcls.def(py::init<int,int,int,double,double,ulint>(),
	    py::arg("nucbp"), py::arg("nbp"), py::arg("llink"),
	    py::arg("elink"), py::arg("mu"), py::arg("seed"));
  tmcls.def("setEnergy", [](NucPosTM& self, vector<double> eseq,
			    double emax) {
    self.setEnergy(eseq, emax);
  }, py::arg("eseq"), py::arg("emax"));
  tmcls.def("solve", &NucPosTM::solve, py::arg("temp") = 1.0);
  tmcls.def("getLogZ", &NucPosTM::getLogZ);
  tmcls.def("getStartProb", &NucPosTM::getStartProb);
  tmcls.def("getOccup", &NucPosTM::getOccup);
  tmcls.def("getMeanN", &NucPosTM::getMeanN);
  tmcls.def("getVarN", &NucPosTM::getVarN, py::arg("dmu") = 1e-4);
  tmcls.def("getVarOccup", &NucPosTM::getVarOccup);
  tmcls.def("sampleConfig", &NucPosTM::sampleConfig);
  tmcls.def("getSeqEnergy", &NucPosTM::getSeqEnergy);
  tmcls.def("reset", &NucPosTM::reset);
  tmcls.def("setSeed", &NucPosTM::setSeed, py::arg("seed"));

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

  py::class_<PyTrackObs, PyTrackBase, shared_ptr<PyTrackObs> >
    dumpcls(m, "Dump");

  py::enum_<PyTrackObs::OutputType>(dumpcls, "OutputType", py::arithmetic())
    .value("Energy", PyTrackObs::OutputType::Energy)
    .value("Position", PyTrackObs::OutputType::Position)
    .value("Temp", PyTrackObs::OutputType::Temp)
    .value("All", PyTrackObs::OutputType::All);

  m.def("createDump", [](lint freq, PyTrackObs::OutputType otype) {
    return std::make_shared<PyTrackObs>(freq, otype);
  }, py::arg("freq"), py::arg("out_type"));
}
