// dump.hpp

#ifndef DUMP_HPP
#define DUMP_HPP

#include <fstream>
#include <string>
#include "model.hpp"
#include "tracker.hpp"

class NucPosModel;

class Dump : public Tracker {

private:
  bool append;
  std::string file;  
  std::ofstream fileStream;  

public:
  Dump(int printFreq, bool append, std::string file);
  virtual ~Dump() {
    if (fileStream.is_open()) fileStream.close();
  };  
  virtual void update(int time, const NucPosModel& model,
		      std::ofstream& writer) = 0;
  void update(int time, const NucPosModel& model) override;
};

#endif
