#include <iostream>
#include <fstream>
#include <vector>

class Mnist{
public:
    std::vector<std::vector<double> > readTrainingFile(std::string filename);
    std::vector<double> readLabelFile(std::string filename);
};