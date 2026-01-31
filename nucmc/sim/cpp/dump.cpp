// dump.cpp

#include <iostream>
#include <sstream>
#include <string>
#include "tracker.hpp"
#include "dump.hpp"

using std::cout;
using std::endl;
using std::stringstream;
using std::string;

Dump::Dump(int freq, bool app, string f) :
  Tracker(freq), append(app), file(f) {}

void Dump::update(int time, const NucPosModel& model) {
  string ofile;
  if (!append) {
    stringstream ss;
    ss.clear();
    ss << file << "." << time;
    ofile = ss.str();
    fileStream.open(ofile);
  } else {
    ofile = file;
    fileStream.open(ofile, std::ios::app);
  }
  if (!fileStream) {
    cout << "ERROR: cannot open the file " << ofile << endl;
  }  
  update(time, model, fileStream);
  fileStream.close();
}
