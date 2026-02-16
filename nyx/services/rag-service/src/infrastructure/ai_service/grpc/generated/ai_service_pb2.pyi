from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LoadModelRequest(_message.Message):
    __slots__ = ("name", "version", "server") # type: ignore
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVER_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    server: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., server: _Optional[str] = ...) -> None: ...

class UnloadModelRequest(_message.Message):
    __slots__ = ("name", "version", "server") # type: ignore
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVER_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    server: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., server: _Optional[str] = ...) -> None: ...

class ModelStatus(_message.Message):
    __slots__ = ("name", "version", "server", "state", "message") # type: ignore
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
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVER_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    server: str
    state: ModelStatus.State
    message: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., server: _Optional[str] = ..., state: _Optional[_Union[ModelStatus.State, str]] = ..., message: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("name", "version", "server") # type: ignore
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVER_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    server: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., server: _Optional[str] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("task", "text_batch", "image_batch") # type: ignore
    TASK_FIELD_NUMBER: _ClassVar[int]
    TEXT_BATCH_FIELD_NUMBER: _ClassVar[int]
    IMAGE_BATCH_FIELD_NUMBER: _ClassVar[int]
    task: str
    text_batch: TextBatch
    image_batch: ImageBatch
    def __init__(self, task: _Optional[str] = ..., text_batch: _Optional[_Union[TextBatch, _Mapping]] = ..., image_batch: _Optional[_Union[ImageBatch, _Mapping]] = ...) -> None: ...

class TextBatch(_message.Message):
    __slots__ = ("content",) # type: ignore
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, content: _Optional[_Iterable[str]] = ...) -> None: ...

class ImageBatch(_message.Message):
    __slots__ = ("content",) # type: ignore
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, content: _Optional[_Iterable[bytes]] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("task", "text_batch", "embedding_batch") # type: ignore
    TASK_FIELD_NUMBER: _ClassVar[int]
    TEXT_BATCH_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_BATCH_FIELD_NUMBER: _ClassVar[int]
    task: str
    text_batch: TextBatch
    embedding_batch: EmbeddingBatch
    def __init__(self, task: _Optional[str] = ..., text_batch: _Optional[_Union[TextBatch, _Mapping]] = ..., embedding_batch: _Optional[_Union[EmbeddingBatch, _Mapping]] = ...) -> None: ...

class EmbeddingBatch(_message.Message):
    __slots__ = ("vectors",) # type: ignore
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedCompositeFieldContainer[EmbeddingVector]
    def __init__(self, vectors: _Optional[_Iterable[_Union[EmbeddingVector, _Mapping]]] = ...) -> None: ...

class EmbeddingVector(_message.Message):
    __slots__ = ("values",) # type: ignore
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...
