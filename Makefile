DIRINC=-I/work/02653/shgulati/TACC_PROJECT/boost_install/include -I/work/02653/shgulati/TACC_PROJECT/roseCompileTree/include
DIRLIB=-L/work/02653/shgulati/TACC_PROJECT/boost_install/lib -L/work/02653/shgulati/TACC_PROJECT/roseCompileTree/lib
FLAGS=-c
GCC=g++
LIB=-lrose $(DIRLIB) 

OBJECTS= Visitor2.o linearTranslator.o Function.o TwoHostTrans.o FunctionTwo.o OpenMPReduce.o OpenMPDirective.o ParserFunDecl.o 

Visitor2:  ${OBJECTS} 
	${GCC}  -o Visitor2  ${OBJECTS} $(DIRINC) ${LIB} 
	
Visitor2.o:  Visitor2.C
	${GCC}  ${FLAGS} Visitor2.C  $(DIRINC) ${LIB}

linearTranslator.o: linearTranslator.C 
	${GCC} ${FLAGS} linearTranslator.C $(DIRINC) ${LIB}

Function.o: Function.C 
	${GCC} ${FLAGS} Function.C $(DIRINC) ${LIB}

TwoHostTrans.o: TwoHostTrans.C
	${GCC} ${FLAGS} TwoHostTrans.C $(DIRINC) ${LIB}

FunctionTwo.o: FunctionTwo.C 
	${GCC} ${FLAGS} FunctionTwo.C $(DIRINC) ${LIB} 

OpenMPReduce.o: OpenMPReduce.C
	${GCC} ${FLAGS} OpenMPReduce.C $(DIRINC) ${LIB}

OpenMPDirective.o: OpenMPDirective.C
	${GCC} ${FLAGS} OpenMPDirective.C $(DIRINC) ${LIB}

ParserFunDecl.o: ParserFunDecl.C
	${GCC} ${FLAGS} ParserFunDecl.C $(DIRINC) ${LIB}

clean:
	rm -f *.o
