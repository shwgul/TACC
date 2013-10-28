#ifndef LINEARTRANSLATOR_H
#define LINEARTRANSLATOR_H

#include <vector>

using namespace std;

class LinearTranslator {
  public:
    void readFile(); 
    void translateLinear(SgStatement* loopStat, SgLocatedNode *ln);
};

#endif
