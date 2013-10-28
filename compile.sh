#!/bin/bash
g++ -o Visitor2 Visitor2.C linearTranslator.C Function.C TwoHostTrans.C FunctionTwo.C -I/work/01235/jolaya/fraspa/rose/boost_install/include -L/work/01235/jolaya/fraspa/rose/boost_install/lib -I/work/01235/jolaya/fraspa/rose/roseCompileTree/include -L/work/01235/jolaya/fraspa/rose/roseCompileTree/lib  -lrose 

