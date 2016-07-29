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
* 编译设备代码到汇编代码(PTX代码)或者二进制代码(cubin对象)，
* 用Kernels使用的语法<<<...>>>来修改主机代码(更多信息参考：[ Execution Configuration](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)),并且需要调用CUDA c运行库函数来加载运行每一个kernel代码，这些kernel代码就是从PTX代码或者cunbin对象编译过来的。

修改后的主机代码的输出要么是c代码，可以用别的工具编译，要么是对象代码，可以直接用在编译阶段nvcc调用主机编译器。

##### Just-in-Time Compilation

一个应用程序在运行时加载的任何PTX代码，最终会被设备驱动编译成二进制代码，这就称之为：实时编译(just-in-time compilation)。

实时编译增加了应用程序的加载时间，但是允许程序在新的设备驱动中提供的新的编译器改进中获取性能提升。

当设备驱动实时编译应用代码时，会自动的缓存一份生成的二进制代码的拷贝，为了避免在应用程序中的子调用中重复编译。这个缓存也可以称之为计算缓存会在设备升级后自动失效，所以应用程序可以在新的设备驱动中内置的实时编译器上提升效果。

需要环境变量来控制实时编译，参考：[CUDA Environment Variables](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)

#### Binary Compatibility

二进制代码是由系统架构决定的，一个cubin对象是由编译器选项-code生成的，这个选项指定了具体的目标架构。比如用-code=sm_35编译的二进制代码是适用在3.5版本的设备上的，参考[ compute capability](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)
二进制兼容性是由从一个小的修订再到接下来的一个修订保证的，而不是从一个小修订到前面的一个修订，或者跨大版本的修改来保证。换句话来说，生成的一个cubin对象的计算兼容性从版本X.y到X.z的设备上都是满足的，只要z $\geq$ y。

#### PTX Compatibility

一些PTX的指令集仅仅在具有更多的计算兼容性的设备上才能支持。比如warp shuffle的指令仅仅在计算兼容性版本在3.0或者更高的设备上才能支持。-arch这个编译选项指定了可以从c代码编译到PTX代码的计算性能，因此包含warp shuffle的代码必须用-arch=sm_30或更高的编译指令。

由在某个计算能力档次上编译的PTX代码同样也可以在同一个计算能力档次或者更高的计算能力档次上编译成二进制代码。

#### Application Compatibility

应用程序在某个具体计算能力档次的设备上执行代码，加载的二进制代码或者PTX代码必须与计算能力想匹配，具体描述参考[Binary Compatibility](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility)和[PTX Compatibility](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility)。实际情况下，为了能够在未来拥有更高计算能力的设备上运行，应用程序必须在这些设备上加载PTX代码用来实时编译。

CUDA c应用程序内置的PTX和二进制代码是由编译选项-arch和-code控制的。
