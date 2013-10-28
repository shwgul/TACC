#ifndef OPENMPDIRECTIVE_H_
#define OPENMPDIRECTIVE_H_

#include <iostream>
#include <rose.h>
#include <string>

using namespace SageBuilder;
using namespace SageInterface;
using namespace AstFromString;
using namespace std;

class OpenMPDirective
{
  public:
    void parseOpenMPDirective(Rose_STL_Container<SgNode*> nodeList);
};


#endif
