// model.hpp

#ifndef MODEL_HPP
#define MODEL_HPP

#include <string>
#include <vector>
#include <set>
#include <random>
#include <memory>
#include "tracker.hpp"

//class Dump;
class Tracker;

class NucPosModel {
private:
  // Required parameters
  int nucbp;
  int nbp;  
  int llink;
  double mu;
  long seed;
  
  // Other variables
  int npos;
  int maxNumOfNuc;
  double temp;
  std::vector<double> emeth;
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
  NucPosModel(int nucbp, int nbp, int llink, double mu, long seed);
  ~NucPosModel();
  void initByMethData(std::string dataFile);
  void update();
  void reset();
  void output(int time);
  void run(int nsweeps, double startTemp, double endTemp, int nincs);
  const std::vector<int>& getNucPos() const;
  int getNucbp() const;
  double getEnergy() const;
  void addTracker(std::shared_ptr<Tracker> tracker);
};

#endif
