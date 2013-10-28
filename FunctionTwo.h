#ifndef FUNCTIONTWO_H_
#define FUNCTIONTWO_H_

#include "rose.h"

using namespace SageBuilder;
using namespace SageInterface;
using namespace AstFromString;

class SimpleInstrumentationTwo: public SgSimpleProcessing
{
  public:
    void visit(SgNode* astNode);
};


#endif
