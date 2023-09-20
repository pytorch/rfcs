# Providing Topology Information to Training Apps/Ranks

**Authors:**
* @kurman


## **Summary**
Provide topology information to each rank/trainer process that can be used by trainers to optimize and potentially automate pytorch distributed models.


## **Motivation**
As the size of jobs are increasing, ML jobs utilize parallelism to a) increase throughput and b) to allow model sizes that don’t fit into a single GPU or even node. Those techniques are known as Data Parallelism (DP), Model Parallelism (MP) and Pipeline Parallelism (PP). Each of them have specific trade-offs and can be combined.

Using those techniques requires some knowledge of underlying topology to make right tradeoffs based on communication overhead (speed, throughput), available GPU memory and use of non-accelerator resources (CPU/RAM). PyTorch has come up with various generic distributed modeling solutions like DTensor and DeviceMesh. However pytorch distributed primitives provide no automation today for taking advantage of topology info to perform model partitioning, it is the job of the user to configure the 2-D or 3-D (MP+DP+PP) parallelism solutions in a topology aware manner for optimal performance.


## **Proposed Implementation**
Provide topology information to each rank/trainer process that can be used to create optimized computational graphs for Torch modules.

Current HPC/Gang scheduled jobs rely on number of variables that is passed to each trainer process:
- `WORLD_SIZE` - total number of workers
- `RANK` - unique rank of a worker (0…WORLD_SIZE-1)
- `LOCAL_RANK` - unique rank of a worker on a node, typically used to exclusively assign accelerators on the host.

New proposed `WORLD_TOPOLOGY_FILE` environment variable will reference a local filesystem file that will provide information about the underlying topology.


### **Content of the topology information file**
Topology information is a representation of graph where nodes are devices assigned to a RANK and edges are connectivity type with optional:
- Latency
- Bandwidth,  and 
- Available number of channels

The connectivity type assumes HPC type of workloads and can be scoped to following types:
- NVLink
- NVSwitch
- PCIe
- IB
- NIC/Ethernet

Eg: 

|     |          RANK0            |           RANK1         |        RANK2         |
|-----|---------------------------|-------------------------|----------------------|
|RANK0|              X            | NVLink [22us,64GB/s, 4] | IB[600us, 0.4GB/s,4] |
|RANK1|NVLink [22us,64GB/s, 4]    |            X            | IB[600us, 0.4GB/s,4] |
|RANK2|IB[600us, 0.4GB/s,4]       | IB[600us, 0.4GB/s,4]    |           X          |



Out-of-scope for now:
- System resources allocated for each rank (eg MEM/CPU/GPU) 
  - Those tend to be homogeneous
  - Most of the can be easily detected at runtime by the trainer code
- More fine-grained details based on communication pattern (p2p vs collectives)

### **Format of the topology information file**

- File format: json
- Define version = 0.1
- Extensible:
    - Avoid using lists and use named properties, this allows adding new properties later.

Eg:
```json
{
  "version": "0.1",
  "ranks": {
    "0": {
      "peers": {
        "1": {
          "connection": {
            "latency": {
              "value": "22",
              "measurement": "us"
            },
            "bandwidth": {
              "value": "64",
              "measurement": "GB/s"
            },
            "channels": {
                "value": "4"
            }
          }
        }
      }
    },
    "1": {"peers": {...}}
  }
}
```


### **Implementation details**
[torchrun/torchelastic](https://pytorch.org/docs/stable/elastic/run.html) will 
- Build the topology representation during the rendezvous process
- Save the file in local filesystem
- Launch the job with `WORLD_TOPOLOGY_FILE` env variable with value of the location of the file.

Topology representation can either reuse or reimplement part of topology discovery used in [NCCL](https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.cc) where it uses information from /sys.

## **Potential of the usecases**

- Locality aware DeviceMesh definition to be used by DTensor using topology information
- Use of this information to place MP on NVLinked accelerators, and DDP/PP on IB connected hosts.
- DistCompiler can be used to construct optimal execution graph
- Replication of checkpoints/snapshots on non-adjacent nodes to provide more reliable fault tolerance


#### Additional Context
https://arxiv.org/abs/1903.04611
https://github.com/pytorch/pytorch/issues/88838
https://pytorch.org/docs/stable/distributed.elastic.html


