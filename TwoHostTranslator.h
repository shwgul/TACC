#ifndef TWOHOSTTRANSLATOR_H
#define TWOHOSTTRANSLATOR_H

#include <vector>

using namespace std;

class TwoHostTranslator {
  public:
    void readFile();
    void setInnerLoop();
    void translateTwoHost(SgStatement* loopStat, SgLocatedNode *ln);
  private:
    vector<string> src_file;
};

#endif
