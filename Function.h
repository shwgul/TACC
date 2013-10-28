#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "rose.h"

using namespace SageBuilder;
using namespace SageInterface;
using namespace AstFromString;

class SimpleInstrumentation: public SgSimpleProcessing
{
  public:
    void visit(SgNode* astNode);
};


#endif
