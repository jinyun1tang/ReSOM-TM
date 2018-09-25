
def __bootstrap__():
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'resom_mathlib.so')
   __loader__ = None; del __bootstrap__, __loader__
#   imp.load_dynamic(__name__,__file__)
   imp.load_dynamic(__name__,'/Users/jinyuntang/work/github/ReSOM-TM/build/lib.macosx-10.13-x86_64-2.7/ReSOM/resom_mathlib.so')
__bootstrap__()
