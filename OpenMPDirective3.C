#include <iostream>
#include <rose.h>
#include <stdio.h>
#include <string>
#include <rewrite.h>
#include "OpenMPDirective.h"

using namespace SageInterface;
using namespace SageBuilder;
using namespace std;

/**
  * Creates the kernel function based on a linear patter.
  */
void SimpleInstrumentationOpenMPDir::visit(SgNode* astNode) {
  PreprocessingInfo * att;
  SgLocatedNode *ln;
  
  SgGlobal* globalScope = isSgGlobal(astNode);
  if(globalScope != NULL) {
      //SgStatement * scope = getFirstStatement ((SgStatement *) astNode, true);
      att = attachArbitraryText(ln, "#include <omp.h>",  PreprocessingInfo::before);
    
    prependStatement((SgStatement *) att, globalScope);

  } 
}
