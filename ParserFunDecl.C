#include <cstring>
#include <rose.h>
#include <vector>
#include <string>
#include <rewrite.h>
#include "ParserFunDecl.h"
#include "Data.h"

using namespace SageInterface;
using namespace SageBuilder;
using namespace AstFromString;
using namespace std;


  /* This main function has the parser to find the fun declaration in the AST
  * and start the tranlation.
  */
void ParserFunDecl::parseFunDecl(Rose_STL_Container<SgNode*> nodeList, string line, int j) {
  //Parser variables
  ios::sync_with_stdio();
  int isMatch;
  int location = -1;
  int location2 = -1;
  string str;
  string str2;
  string str3;
  char *tmp1;
  char *tmp2;

  str3 = pairs[line] + " " + line;

  tmp1 = new char[str3.size()+1];
  strcpy(tmp1, str3.c_str());
  
  Rose_STL_Container<SgNode*>::iterator i = nodeList.begin();

  if(transData.parallel == "CUDA") {
   while(i != nodeList.end())
   {
        str = (*i)->unparseToString().c_str();

        location = str.find_first_of("(");
        str2 = str.substr(0, location);
        tmp2 = new char[str2.size()+1];
        strcpy(tmp2, str2.c_str());
        isMatch = strcmp(tmp1, tmp2);

        if(isMatch == 0 && !(str.at(str.length()-1) == ';')) {
 	 transData.cuda_funDec.push_back((*i)->unparseToString()); 
         cout << " match " << tmp1 << " " << tmp2 << endl;
          break;
        }
	i++;
   }

  }

  else if(transData.parallel == "OpenMP") {
   while(i != nodeList.end())
   {
        str = (*i)->unparseToString().c_str();
        
        location = str.find_first_of("(");
	str2 = str.substr(0, location);	
	tmp2 = new char[str2.size()+1];
  	strcpy(tmp2, str2.c_str());
        isMatch = strcmp(tmp1, tmp2);
        if(isMatch == 0 && str.at(str.length()-1) == ';') {
          cout << " match " << tmp1 << " " << tmp2 << endl;
          break;
        }
        i++;
   }
  }
  /******************** starting translation **********************************/
  SgStatement* stat = (SgStatement *) (*i);
  SgLocatedNode *ln = isSgLocatedNode((*i));
  PreprocessingInfo * att;

  if(transData.parallel == "CUDA") {
    /*if(transData.cuda_typeFun.at(j) == "Device") 
	att = attachArbitraryText(ln, "__device__",  PreprocessingInfo::before);
    else if(transData.cuda_typeFun.at(j) == "Host-Device")
	att = attachArbitraryText(ln, "__Host__ __device__",  PreprocessingInfo::before);
    else if(transData.cuda_typeFun.at(j) == "Device-Device")
        att = attachArbitraryText(ln, "__device__ __device__",  PreprocessingInfo::before);*/
     removeStatement(stat);
  }
  
  else if(transData.parallel == "OpenMP") {
    if(transData.offload == "Yes") {
      att = attachArbitraryText(ln, "__declspec(target(mic))",  PreprocessingInfo::before);
    }
  }
}

