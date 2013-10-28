#include <rose.h>
#include <stdio.h>
#include <vector>
#include <rewrite.h>
#include "OpenMPDirective.h"
#include "Data.h"

using namespace SageInterface;
using namespace SageBuilder;
using namespace AstFromString;
using namespace std;


  /* This main function has the parser to find the fun declaration in the AST
  * and start the tranlation.
  */
void OpenMPDirective::parseOpenMPDirective(Rose_STL_Container<SgNode*> nodeList) {
  //Parser variables
  ios::sync_with_stdio();
  int counter = 0;
  
  Rose_STL_Container<SgNode*>::iterator i = nodeList.begin();
  while(i != nodeList.end())
  {
	if(counter == 0) break;
	counter++;
        i++;
  }
  /******************** starting translation **********************************/
  SgStatement* stat = (SgStatement *) (*i);
  SgLocatedNode *ln = isSgLocatedNode((*i));
  PreprocessingInfo * att;

  if(transData.parallel == "OpenMP") {
    att = attachArbitraryText(ln, "#include <omp.h>",  PreprocessingInfo::before);
    if (transData.offload == "Yes") {
       att = attachArbitraryText(ln, "#include <offload.h>" ,  PreprocessingInfo::before);
    }
  }
}
