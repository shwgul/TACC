#ifndef PARSERFUNDECL_H_
#define PARSERFUNDECL_H_

#include <iostream>
#include <rose.h>
#include <stdio.h>
#include <string>

using namespace SageBuilder;
using namespace SageInterface;
using namespace AstFromString;
using namespace std;

class ParserFunDecl
{
  public:
    void parseFunDecl(Rose_STL_Container<SgNode*> nodeList, string line, int j);
};


#endif
