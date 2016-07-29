## CUDA C Runtime

cudart库实现了运行时，这个库可以通过cudart.lib或者libcurart.a静态链接到应用程序，或者通过cudard.dll和libcudart.so动态链接到应用程序。应用程序需要的动态链接库cudart.dll或者cudart.so一般在安装包中就已经包含了。

运行库所有的入口点都是以cuda开头的。

在[Heterogeneous Programming](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming)中提到的，CUDA编程模型假设一个系统由一个host和一个device组成，每个都有自己独立的内存空间。[Device Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
