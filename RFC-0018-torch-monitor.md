# torch.monitor - Standardizing events and counters for core Pytorch

This RFC proposes a new PyTorch package called `torch.monitor`. This would provide a standardized interface for logging events as well as always on counters to enable monitoring of training and inference jobs.

## Motivation

Generally PyTorch jobs have two systems that are used to log information about the job:

1. Time series logger such as TensorBoard used during training, requiring that the user manually log certain metrics they're interested in.
2. `torch.profiler` which records operator level stats that can be selectively turned on if a user is experiencing performance issues and needs deeper level insight into their model. The profiler has performance overhead so is typically only used when a performance issue arises.

Both of these systems are proactive and require manual user intervention to debug their jobs either by adding new metrics to TensorBoard or by manually enabling profiling to get low level performance details.

For distributed jobs and post-mortem investigation into modelling issues it helps to have always-on metrics and events which can be referred to after the fact to understand the model behavior for training or inference. These always-on metrics can also be used to add high level monitoring to proactively detect issues with training or inference.

There's a number of existing metrics and event logging systems in PyTorch but with no standardization across the subprojects. This aims to provide a shared interface that all PyTorch projects can use in a consistent way.

## Description

The goal of this RFC is to provide an interface to track high level events (`events`) and summary statistics (`stats`). This document defines the high level python interface for these though the core implementation will be done in C++ and tied to the existing PyTorch profiler where possible.

Future work will add integrations into PyTorch core and distributed to log key events and metrics to the users. It'll also include out of the box integrations for commonly logged stats and logging destinations.

### Events

Events have a generic metadata dict to store arbitrary information. These events are intended to have a relatively low QPS compared to the existing `RECORD_FUNCTION` profiler interface and should have the same frequency as one might use stderr logging.

**Interfaces**

```py
@dataclass
class Event:
    # type is a globally unique type name
    # Ex: torch.elastic.RdzvEvent
    type: str
    # message is a human readable string describing the event at a
    # high level
    message: Optional[str]
    # Timestamp is the unix wall clock time of the event in seconds
    timestamp: Optional[float]
    # Metadata can contain any event specific fields and is intended
    # for fields to later be aggregated or filtered on.
    metadata: Dict[str, Union[str, float, int, bool]]
```

All events are sent to all registered event handlers. Event handlers can ignore events that are not relevant to them. The backend for these events will be implemented in C++ and merged with the existing profiler where possible.

Since handlers can be registered in Python, the frequency of logged events must be relatively low to avoid performance impact.

```py
event_handlers = []

class EventHandler(ABC):
    @abstractmethod
    def handle(event: Event) -> None:
        ...
def register(handler: EventHandler):
    global event_handlers
    event_handlers.append(handler)

def log(event: Event) -> None:
    for handler in event_handlers:
        handler.handle(event)
```

**Example Usage**

```py
from torch.monitor import Event, log

@dataclass
class RdzvEvent(Event):
    def __init__(self, run_id: str, rank: int, message: str) -> None:
        super().__init__(
            type="elastic_rdvz",
            metadata={"run_id": run_id, "rank": pid},
            message=message,
        )

event = RdzvEvent(run_id="1234", rank=0, message="rendezvous started")
log(event)
```

```py
class ExampleJSONLHandler(EventHandler):
    def handle(event: Event) -> None:
        with open(self.file_name, 'a') as f:
          json.dump(f, event.asdict())

register(ExampleJSONLHandler('logs.jsonl'))
```

### Stats

These stats are designed to be always on metrics to track various key metrics that can be used for monitoring and debugging the performance of your training jobs or inference systems.

These are defined in Python for readability purposes. The core would be implemented in C++ for minimal overhead tracking of metrics and the ability to log more fine grained metrics automatically from things such as autograd or the optimizers.

**Interfaces**

```py
class StatType(Enum):
    # VALUE exports the most recently set value.
    VALUE = "value"
    # MEAN computes the mean of the set values within the window.
    MEAN = "mean"
    # COUNT tracks the number of times a value is set within the window.
    COUNT = "count"
    # SUM computes the sum of the values set within the window.
    SUM = "sum"

    MAX = "max"
    MIN = "min"

    # These may not be present in the initial implementation:

    # HISTOGRAM computes summary statistics such as P50, P90, P95, P99
    HISTOGRAM = "histogram"
    # STDDEV computes the standard deviation of the values set within
    # the window.
    STDDEV = "stddev"

_collectors: Set[StatCollector] = set()

@dataclass
class Stat:
    # key is the name of the stat.
    # Each type of stat should have a globally unique key.
    key: str
    # Aggregations is how this stat should be aggregated.
    aggregations: Set[StatType]

    def add(self, v: float) -> None:
       for collector in _collectors:
           collector.handle(self, v)

class StatCollector(ABC):
    def handle(self, stat: Stat, v: float) -> None:
        ...

def register_stat_collector(collector: StatCollector) -> None:
    _collectors.add(collector)
```

**Example Usage**

```py
from torch.monitor import Stat, StatType

BATCH_LATENCY = Stat(
    key="training.batch.latency",
    aggregations={StatType.MEAN, StatType.HISTOGRAM},
)
EPOCH_LATENCY = Stat(
    key="training.epoch.latency",
    aggregations={StatType.MEAN, StatType.HISTOGRAM},
)

def train(...):
    for i in range(epochs):
        epoch_start = time.time()
        for x, y in dataloader:
            batch_start = time.time()
            y_pred = model(x)
            ...
            BATCH_LATENCY.add(time.time()-batch_start)
        EPOCH_LATENCY.add(time.time()-epoch_start)
```

Collectors:

```py
class AggregatingStatCollector(StatCollector):
    stats: Set[Stat]

    count: DefaultDict[str, float]
    sum: DefaultDict[str, float]
    value: Dict[str, float]

    def handle(self, stat: Stat, v: float) -> None:
        stats.add(stat)

        if (StatType.MEAN in stat.aggregations
                or StatType.COUNT in stat.aggregations):
            self.count[stat.key] += 1

        if (StatType.MEAN in stat.aggregations
               or StatType.SUM in stat.aggregations):
           self.sum[stat.key] += v

        if StatType.VALUE in stat.aggregations:
            self.value[stat.key] = v

        ...

    def report(self) -> Dict[str, float]:
        out = {}
        for stat in self.stats:
            for type in stat.aggregations:
                if type == StatType.MEAN:
                    out[stat.key + ".mean"] = (
                        self.sum[stat.key] / self.count[stat.key]
                    )
                ...

    def reset(self) -> None:
        ...

collector = AggregatingStatCollector()
register_stat_collector(collector)

# in background thread
while True:
   stats = collector.report()
   collector.reset()
   for k, v in stats.items():
      tensorboard_writer.add_scalar(k, v)
   time.sleep(60)
```

## FAQ

- How does this differ from torch.profiler? Why do we need new ways to track performance?

  - The current profiler is typically only turned on when there is an issue. These metrics and events are intended to be always-on to help monitor production training jobs and inference.
  - We plan to extend the profiler on the C++ side to be able to track these events as well though the user facing interface for defining the stats and events will differ from the existing RECORD_FUNCTION interface.

- Why not log to TensorBoard?

  - These events and metrics can and probably will be logged to TensorBoard for many users. This defines a high level interface that core and related projects can use to surface common information in a standardized way. The pluggable collectors enable logging to any location.

- Where will this be used?

  - The events system will be immediately used to replace:

    - torch.distributed.elastic.events
    - torchx.events
    - PyTorch Lightning -- LightningDeprecationEvent
    - Tentatively collective operations from NCCL/Gloo

  - The counters system will be immediately used to unify a number of existing metrics libraries within pytorch such as:
    - torch.distributed.elastic.metrics
    - torch.distributed.rpc.metrics
    - torch.distributed.rpc.parameter_server.metrics
    - The torch.jit.runtime.static may also be provided through this interface.

- How are system stats tracked?

  - If the user has an existing system metrics tracking system as part of their loggers there's no required system stats tracking.
  - If they don't have one, we plan to provide an out of the box SystemStatsProvider that can be enabled with a couple of lines of code in their main method that provides common stats such as CPU/GPU util and memory usage.

- How fast can we log?
  - For events we expect that they will be used at about the same frequency as you might use the built in python logger. These are intended for rich structured event logging and thus for performance reasons need to be relatively few.
