// model.hpp

#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include <vector>
#include <set>
#include <random>
#include <memory>
#include "dtype.hpp"
#include "tracker.hpp"

//class Dump;
class Tracker;

class NucPosModel {

public:
  struct Params {
    int nucbp;
    int nbp;  
    int llink;
    double mu;
    ulint seed;
  };

  // Protocols for cooling the system
  enum class Cooling {
    Linear,
    Geometric,
    Constant
  };
  
private:
  // Required parameters
  Params params;
  int nucbp;
  int nbp;  
  int llink;
  double mu;
  ulint seed;
  
  // Other variables
  int npos;
  int maxNumOfNuc;
  double temp;
  std::vector<double> eseq;
  std::vector<double> erep;
  std::vector<int> nucpos; // Leftmost position of each nucleosome

  // Distributions generating random numbers
  std::uniform_int_distribution<int> randMode;
  std::uniform_int_distribution<int> randPos;
  std::uniform_real_distribution<double> rand;
  std::mt19937 mt;

  // Trackers
  std::vector<std::shared_ptr<Tracker> > trackers;

public:
  NucPosModel(int nucbp, int nbp, int llink, double mu, ulint seed);
  NucPosModel(const Params& params);
  ~NucPosModel();
  void setSeqEnergy(std::string dataFile, double emax);
  void setSeqEnergy(const std::vector<double>& pseq, double emax);
  void update();
  void reset();
  void output(lint time);
  void run(lint nsweep, double startTemp, double endTemp, Cooling coolOption);
  const std::vector<int>& getNucPos() const;
  double getEnergy() const;
  double getTemp() const;
  const std::vector<double>& getSeqEnergy() const;
  const Params& getParams() const;
  void addTracker(std::shared_ptr<Tracker> tracker);
};

#endif
