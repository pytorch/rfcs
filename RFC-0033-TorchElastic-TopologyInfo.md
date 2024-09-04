# Providing Topology Information to Training Apps/Ranks

**Authors:**
* @kurman


## Summary
Provide topology information to each rank/trainer process that can be used by trainers to optimize and potentially automate pytorch distributed models.


## Motivation
As the size of jobs are increasing, ML jobs utilize parallelism to:
1. increase throughput and 
2. to allow model sizes that don’t fit into a single GPU or even node. 

Those techniques are known as Data Parallelism (DP), Model Parallelism (MP) and Pipeline Parallelism (PP). Each of them have specific trade-offs and can be combined.

Using those techniques requires some knowledge of underlying topology to make right tradeoffs based on communication overhead (speed, throughput), available GPU memory and use of non-accelerator resources (CPU/RAM). PyTorch has come up with various generic distributed modeling solutions like DTensor and DeviceMesh. However pytorch distributed primitives provide no automation today for taking advantage of topology infor,atopm to perform model partitioning, it is the job of the user to configure the 2-D or 3-D (MP+DP+PP) parallelism solutions in a topology aware manner for optimal performance.

Underlying topology data can inform application on communication constraints that can be used to optimize distributed computation graph that will take communication pattern (however still relies underlying comms to actually use them). Regardles of what are the available communication channel sematics, most value will be is grouping nodes by locality. See more details under [usecases](#Usecases) section.

## Scope
The proposal focuses on exposing information about infrastructure to application layer for optimization purposes:
- Contract that applications can use to discover toplogy information when available
- Initial reference implementation that can be substituted by other implementations

Primary focus on defining contract that can be expanded and other potential approaches/algorithms for defining toplogy information is out of scope.

## Proposed Implementation
Provide topology information to each rank/trainer process that can be used to create optimized computational graphs for Torch modules. 

Current HPC/Gang scheduled jobs rely on number of variables that is passed to each trainer process, eg `WORLD_SIZE`, `RANK` and `LOCAL_RANK`.

New proposed optional `WORLD_TOPOLOGY_FILE` environment variable will reference a local filesystem file that will provide information about the underlying topology.

It will be generated on a best efforts basis due to underlying complexity.


### Content of the topology information file
Topology information is a representation of graph where nodes are devices assigned to a RANK and edges are connectivity type with optional:
- Latency
- Bandwidth,  and 
- Available number of channels

The connectivity type assumes HPC type of workloads and can be scoped (but not limited) to following types:
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
- Non-topology system resources allocated for each rank (eg MEM/CPU/GPU) 
  - Those tend to be homogeneous
  - Most of the can be easily detected at runtime by the trainer code
  - Those can be added as an extension if there are specific usecases.
- More fine-grained details based on communication pattern (p2p vs collectives). We are exposing phyiscal toplogy information between peers, however things such as bandwidth will differ based on communication pattern.

### Format of the topology information file

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
            "type" "IB",
            "latency": {
              "value": "22",
              "measurement": "us"
            },
            "bandwidth": {
              "value": "64",
              "measurement": "GB/s"
            },
            "channels": {
                "count": "4"
            }
          }
        }
      }
    },
    "1": {"peers": {...}}
  }
}
```

##### Python API

Python API will be exposed to build respresentation of the toplogy as Python datastructure. `TopologyInfo#build` factory method will use `WORLD_TOPOLOGY_FILE` env variable to discover information about the toplogy.

```python
from enum import Enum
class ConnectionType(Enum):
    ETH = 1
    IB = 2,
    ...

@dataclass 
class Measurement:
    val: str
    measurement: str

@dataclass 
class LatencyInfo(Measurement):
    ...

@dataclass 
class BandwidthInfo(Measurement):
    ...

@dataclass
class ChannelInfo:
    count: int
    ...

@dataclass 
class ConnectionInfo:
    conn_type: ConnectionType
    latency: LatencyInfo
    bandwidth: BandwidthInfo
    channel: ChannelInfo

@dataclass
class PeersInfo:
    connections: Dict[int, ConnectionInfo]

@dataclass
class TopologyInfo:
    version: str
    ranks: Dict[int, PeersInfo]

    @staticmethod
    def build() -> Union[None, "TopologyInfo"]
        ...
```

### Implementation details
[torchrun/torchelastic](https://pytorch.org/docs/stable/elastic/run.html) will 
- Add new flag to inject information toplogy `--with-accelarator-topology`
- Build the topology representation during the rendezvous process
- Save the file in local filesystem
- Launch the job with `WORLD_TOPOLOGY_FILE` env variable with value of the location of the file.

#### Defining/discoverying topology
Initial implementation can be very simple by targeting nvidia devices and built using at a high level following steps:
1. Run `nivida-smi` on the host to list devices
2. Traverse "/sys" on each host to detect connectivity type between devices on the host
3. Traverse "/sys" on each host to detect NET connectivity with guid's
4. Publish as part of node redezvous data
5. On each node build representation:
  - For local ranks, use the data from step 2
  - For other hosts, use NET connectivity information. 

Further improvement can be done is to first NCCL_TOPO_FILE representation used by [NCCL](https://github.com/NVIDIA/nccl/blob/master/src/graph/topo.cc) to build topology that will allow using VM vendor specific topology definition.


## Usecases
We expect most of the usecases will be implemented by at a framework level:

- Locality aware DeviceMesh definition to be used by DTensor using topology information, for example use in [controller](https://github.com/zdevito/single_controller) based execution
- DistCompiler can be used to construct optimal execution graph
- Use of this information to place MP on NVLinked accelerators, and DDP/PP on IB connected hosts.
- Replication of checkpoints/snapshots on non-adjacent nodes to provide more reliable fault tolerance, e.g. [proposal](https://docs.google.com/document/d/1xX8kTBqmm9Ve03KX8cBhnUdBx4xIf4YtbcUMky46Nd8/edit#heading=h.6xffhuojpv2y)



#### Additional Context
- https://arxiv.org/abs/1903.04611
- https://github.com/pytorch/pytorch/issues/88838
- https://pytorch.org/docs/stable/distributed.elastic.html


