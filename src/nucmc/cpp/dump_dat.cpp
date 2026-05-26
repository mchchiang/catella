// dump_dat.cpp

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "dtype.hpp"
#include "tracker.hpp"
#include "model.hpp"
#include "dump_dat.hpp"

using std::cout;
using std::endl;
using std::stringstream;
using std::string;

DumpDat::DumpDat(lint freq, const string& f, bool app) :
  Tracker(freq), file(f), append(app) {}

void DumpDat::initialize(lint time, const NucPosModel& model) {
  if (append) {
    fileStream.open(file, std::ios::out | std::ios::app);
    if (!fileStream) {
      throw std::runtime_error("Cannot open the file " + file);
    }
  }
}

void DumpDat::update(lint time, const NucPosModel& model) {
  if (!append) {
    string ofile;    
    stringstream ss;
    ss.clear();
    ss << file << "." << time;
    ofile = ss.str();
    fileStream.open(ofile);
    if (!fileStream) {
      throw std::runtime_error("Cannot open the file " + ofile);
    }
  }
  dump(time, model, fileStream);
  if (!append) {
    fileStream.close();
  }
}

void DumpDat::finalize(lint time, const NucPosModel& model) {
  if (append) {
    fileStream.close();
  }
}
