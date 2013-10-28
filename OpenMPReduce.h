#ifndef OPENMPREDUCE_H
#define OPENMPREDUCE_H

#include <string>
#include <vector>

using namespace std;

class OpenMPReduce {
  public:
    void readFile1();
    void readFile2(); 
    void translate(SgStatement* loopStat, SgLocatedNode *ln);
};

#endif
