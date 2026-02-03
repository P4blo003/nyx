from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelIdentifier(_message.Message):
    __slots__ = ("model_id", "version") # type: ignore
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    version: str
    def __init__(self, model_id: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class ModelStatus(_message.Message):
    __slots__ = ("model_id", "state", "message") # type: ignore
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[ModelStatus.State]
        LOADING: _ClassVar[ModelStatus.State]
        READY: _ClassVar[ModelStatus.State]
        UNLOADED: _ClassVar[ModelStatus.State]
        ERROR: _ClassVar[ModelStatus.State]
    UNKNOWN: ModelStatus.State
    LOADING: ModelStatus.State
    READY: ModelStatus.State
    UNLOADED: ModelStatus.State
    ERROR: ModelStatus.State
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    state: ModelStatus.State
    message: str
    def __init__(self, model_id: _Optional[str] = ..., state: _Optional[_Union[ModelStatus.State, str]] = ..., message: _Optional[str] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ("filter",) # type: ignore
    FILTER_FIELD_NUMBER: _ClassVar[int]
    filter: str
    def __init__(self, filter: _Optional[str] = ...) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",) # type: ignore
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelMetadata]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelMetadata, _Mapping]]] = ...) -> None: ...

class ModelMetadata(_message.Message):
    __slots__ = ("model_id", "version") # type: ignore
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    version: str
    def __init__(self, model_id: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("data_input",) # type: ignore
    DATA_INPUT_FIELD_NUMBER: _ClassVar[int]
    data_input: _containers.RepeatedCompositeFieldContainer[DataInput]
    def __init__(self, data_input: _Optional[_Iterable[_Union[DataInput, _Mapping]]] = ...) -> None: ...

class DataInput(_message.Message):
    __slots__ = ("text", "binary_data") # type: ignore
    TEXT_FIELD_NUMBER: _ClassVar[int]
    BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    text: str
    binary_data: bytes
    def __init__(self, text: _Optional[str] = ..., binary_data: _Optional[bytes] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("embedding",) # type: ignore
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedCompositeFieldContainer[DataOutput]
    def __init__(self, embedding: _Optional[_Iterable[_Union[DataOutput, _Mapping]]] = ...) -> None: ...

class DataOutput(_message.Message):
    __slots__ = ("text", "embedding") # type: ignore
    TEXT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    text: str
    embedding: EmbeddingVector
    def __init__(self, text: _Optional[str] = ..., embedding: _Optional[_Union[EmbeddingVector, _Mapping]] = ...) -> None: ...

class EmbeddingVector(_message.Message):
    __slots__ = ("values",) # type: ignore
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...
