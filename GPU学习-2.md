## Programming Interface

CUDA C提供了一个C程序员熟悉的路径来写CUDA设备支持的程序。

它提供了一个C语言和运行库最小的扩展集合。

语言主要的扩展已经在[Programming Model](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)介绍过了。它们允许程序员定义C函数一样的Kernel，并且kernel函数调用时用一些新的语法来指定grid和block的纬度。关于语言扩展的所有描述可以参考[ C Language Extensions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)。所有包含这些扩展的源文件都需要nvcc来编译，在[Compilation with NVCC](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)里有描述。

[ Compilation Workflow](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-workflow)中介绍了运行时，它提供了执行在host的c函数，可以分配和释放在GPU设备上的内存，转移host内存和设备内存之间的数据，管理不同设备间的系统。关于运行时完整的描述可以在CUDA reference manual上找到。

运行时构建在底层的C API和CUDA的驱动API之上，应用程序也可以调用。驱动API通过暴露底层的概念比如CUDA contexts提供了额外的控制。这个CUDA contexts是主机进程对于GPU设备的模拟。CUDA的模块是GPU设备动态加载库的模拟。大部分应用程序并不需要，因为它们在使用运行时库时并不需要这个层级的控制。context和module管理是内置的，所以可以写出很简明的代码。[ Driver API](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)中介绍了驱动API，并且在参考手册中有完整的描述。

### Compilation with NVCC

kernels代码可以写在CUDA指令集架构上，称之为PTX，这个在PTX的参考手册中有相关描述。它提供了更高级的编程语言比如C语言。在这个例子中，kernels可以通过nvcc编译成二进制代码并在GPU设备上执行。

nvcc是一个编译器驱动简化了编译C代码或者PTX的过程：它提供了简单的和似曾相识的命令行选项，并且调用工具集合来执行代码，这些工具集合实现不同阶段的编译。本节提供了nvcc工作流和命令行选项的概览，完整的描述可以参考nvcc的用户手册。

#### Compilation Workflow

##### Offline Compilation

用nvcc编译的源文件可以包含主机代码和GPU设备代码的混合。nvcc的基本工作流包含从主机代码分离GPU设备代码，并且：
* 方式
