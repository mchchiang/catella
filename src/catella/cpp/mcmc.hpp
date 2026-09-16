// mcmc.hpp

#ifndef MCMC_HPP
#define MCMC_HPP

#include <memory>
#include <random>
#include <string>
#include <vector>
#include "dtype.hpp"
#include "model.hpp"

class Tracker;

class NucPosMCMC {

public:
  struct Params {
    NucPosModel::Params model;
    ulint seed;
  };

  // Protocols for cooling the system
  enum class Cooling {
    Linear,
    Geometric,
    Constant
  };

  NucPosMCMC(int nucbp, int nbp, int llink, double elink, double mu,
	     ulint seed);
  explicit NucPosMCMC(const Params& params);
  ~NucPosMCMC();
  void setEnergy(std::string dataFile, double emax);
  void setEnergy(const std::vector<double>& eseq, double emax);
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

private:
  NucPosModel model;
  Params params;
  double temp;
  std::vector<int> nucpos; // Leftmost position of each nucleosome

  // Distributions generating random numbers
  std::uniform_int_distribution<int> randMode;
  std::uniform_int_distribution<int> randPos;
  std::uniform_real_distribution<double> rand;
  std::mt19937 mt;

  // Trackers
  std::vector<std::shared_ptr<Tracker> > trackers;
};

#endif
