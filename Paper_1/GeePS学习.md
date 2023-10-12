## GeePS 学习

>  这里只是一个粗浅的对GeePS的认识（从理论上）。



#### 稍微简短一点的介绍

为了使PS能支持在分布式GPU上运行并行ML程序，作者做了三项重要的更改：

* Explicit use of GPU memory for the parameter cache（显式使用GPU内存的参数缓存）
* Batch-based parameter access methods（基于批处理的参数访问方法）
* Parameter server management of GPU memory on behalf of the application（代表应用程序对GPU内存的参数服务器管理）



##### 1. 使用GPU内存

将参数缓存（主要）保留在 GPU 内存中并不是要减少 CPU 和 GPU 内存之间的数据移动，而是使参数服务器客户端库能够在后台执行这些数据移动步骤，并将它们与 GPU 计算活动重叠。

同时将参数缓存放在 GPU 内存中还可以使用 GPU 并行性更新参数缓存状态。



##### 2. 批处理操作

一次性读取和更新模型参数值的操作会显著减慢执行速度。

> To realize sufficient performance, our GPU-specialized parameter server supports batch-based interfaces for reads and updates. Moreover, GeePS exploits the iterative nature of model training to provide batch-wide optimizations, such as pre-built indexes for an entire batch that enable GPU-efficient parallel “gathering” and updating of the set of parameters accessed in a batch. These changes make parameter server accesses much more efficient for GPU-based training.

GeePS实现了操作序列收集机制，在第一次迭代或*虚拟迭代*中收集操作序列。在真正的训练开始之前，应用程序会执行虚拟迭代，所有 GeePS 调用都标有虚拟标志。GeePS记录了操作，但没有采取实际行动。



##### 3. 管理GPU内存

参数服务器使用预分配的 GPU 缓冲区将数据传递到应用程序，而不是将参数数据复制到应用程序提供的缓冲区。当应用程序想要更新参数值时，它也会在 GPU 分配的缓冲区中执行此操作。应用程序还可以在参数服务器中存储本地非参数数据（例如中间状态）。

![](https://cdn.jsdelivr.net/gh/Changocc/img@img/pic/geeps.png)





#### 稍微详细一点的介绍

GeePS是专门服务GPU集群的参数服务器的一个深度学习分布式框架。

GPU设备的特性之一是拥有本地内存，任何数据都要先加载到GPU本地内存后才可以计算。通常流程是：CPU从文件中读取一个mini-batch的训练数据，移动到GPU内存，，然后由单线程worker调用cuBLAS以及cuDNN库，以及NVIDIA的其他库来进行GPU运算。

如下图为单GPU的ML，比如默认的**Caffe**。

![image-20231012141327261](https://cdn.jsdelivr.net/gh/Changocc/img@img/pic/image-20231012141327261.png)

在参数服务器中，模型参数存放在一个共享的分布式Key-Value中，他就是PS本身。运行机器学习程序的Worker节点采用Read和Update两个接口跟PS通信，由PS来控制通信开销和一致性收敛。

![image-20231012141911698](https://cdn.jsdelivr.net/gh/Changocc/img@img/pic/image-20231012141911698.png)

通常的PS设计都是围绕以CPU为中心的体系结构来设计的，由于操作GPU的限制，设计时没有考虑到这部分因素，会引起更高的通信开销。

如下图。参数服务器的参数sharding存放在CPU内存中，而运行在GPU内部的模型参数，在CPU中也有一个本地拷贝（可以对比上面的图）。因此，数据需要在三者之间双向移动。

![image-20231012142137870](https://cdn.jsdelivr.net/gh/Changocc/img@img/pic/image-20231012142137870.png)

也就是说，传统的PS不太适合GPU计算（额外的访问开销太多了），GeePS就是针对这点所提出的PS架构。在GeePS中，**参数**sharding直接存放在**GPU**中，因而把前述架构中的三方数据双向复制减少了，同时把GPU和CPU之间的数据搬迁放到了后台执行。

如图。除了将参数缓存从CPU内存移动到GPU内存之外，该图与上图的不同之处在于，相关的Staging memory现在位于参数服务器库中。它用于在网络和参数缓存之间进行更新，而不是在参数缓存和应用程序的GPU部分之间进行更新。

![image-20231012143058633](https://cdn.jsdelivr.net/gh/Changocc/img@img/pic/image-20231012143058633.png)

单纯把**参数Key-Value**存放的地点从CPU移动到GPU并不能完全解决问题，因为GPU的数据读取会非常慢。放到通用的PS下，由于每个Key-Value的同步读取会大大降低吞吐量，因此GeePS所做的进一步工作是引入**批量参数操作**，管理GPU内存，把当前不需要的参数通过后台线程迁移到CPU，从而最小化因为等待数据传输而造成的同步开销。

具体来说，相比于通用PS的Read和Update接口，GeePS额外提供了PostRead和PreUpdate两个接口，当应用需要读取参数时，参数服务器在GPU内存中分配一块缓冲区，并返回指针，读取结束后，由PostRead负责释放缓冲区，改接口是非阻塞操作；当应用需要更新参数时，首先通过PreUpdate获得缓冲区，而非阻塞的Update接口负责更新参数和释放缓冲区。

利用非阻塞IO在不同内存之间交换数据，提升吞吐量，避免单条记录读取的同步开销引起的性能问题。GeePS的这种缓冲区设计实际上也管理起了整个GPU内存。因为GPU内存总是有限的，把PS的Key-Value都放到GPU内存显然不现实，通过在后台不停歇的在CPU和GPU之间交换数据，从而实现了在少量GPU内存中存放巨大模型参数的可能。如图：

![image-20231012144740621](https://cdn.jsdelivr.net/gh/Changocc/img@img/pic/image-20231012144740621.png)



当所有的参数和本地数据(输入数据和中间状态)无法在GPU内存中容纳时，PS可以使用CPU内存来容纳多余的数据。任何合适的数量都可以固定在GPU内存中，而其余的则根据需要从应用程序可以使用的缓冲区中传输。



#### 
