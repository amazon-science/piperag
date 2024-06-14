from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RetrievalRequest(_message.Message):
    __slots__ = ["query", "num_continuation_chunks", "num_neighbours", "staleness_offset", "seq_len", "interval", "use_perf_model"]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    NUM_CONTINUATION_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    NUM_NEIGHBOURS_FIELD_NUMBER: _ClassVar[int]
    STALENESS_OFFSET_FIELD_NUMBER: _ClassVar[int]
    SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    USE_PERF_MODEL_FIELD_NUMBER: _ClassVar[int]
    query: _containers.RepeatedScalarFieldContainer[str]
    num_continuation_chunks: int
    num_neighbours: int
    staleness_offset: int
    seq_len: int
    interval: int
    use_perf_model: bool
    def __init__(self, query: _Optional[_Iterable[str]] = ..., num_continuation_chunks: _Optional[int] = ..., num_neighbours: _Optional[int] = ..., staleness_offset: _Optional[int] = ..., seq_len: _Optional[int] = ..., interval: _Optional[int] = ..., use_perf_model: bool = ...) -> None: ...

class RetrievalReply(_message.Message):
    __slots__ = ["retrieved_tokens"]
    RETRIEVED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    retrieved_tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, retrieved_tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class SetNprobeRequest(_message.Message):
    __slots__ = ["nprobe"]
    NPROBE_FIELD_NUMBER: _ClassVar[int]
    nprobe: int
    def __init__(self, nprobe: _Optional[int] = ...) -> None: ...

class SetNprobeReply(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: str
    def __init__(self, reply: _Optional[str] = ...) -> None: ...
