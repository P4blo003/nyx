import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ContentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTENT_TYPE_UNSPECIFIED: _ClassVar[ContentType]
    CONTENT_TYPE_TEXT: _ClassVar[ContentType]
    CONTENT_TYPE_IMAGE: _ClassVar[ContentType]
    CONTENT_TYPE_TABLE: _ClassVar[ContentType]
    CONTENT_TYPE_DIAGRAM: _ClassVar[ContentType]

class ProcessingState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESSING_STATE_UNSPECIFIED: _ClassVar[ProcessingState]
    PROCESSING_STATE_QUEUED: _ClassVar[ProcessingState]
    PROCESSING_STATE_PARTITIONING: _ClassVar[ProcessingState]
    PROCESSING_STATE_CHUNKING: _ClassVar[ProcessingState]
    PROCESSING_STATE_EMBEDDING: _ClassVar[ProcessingState]
    PROCESSING_STATE_STORING: _ClassVar[ProcessingState]
    PROCESSING_STATE_COMPLETED: _ClassVar[ProcessingState]
    PROCESSING_STATE_FAILED: _ClassVar[ProcessingState]
    PROCESSING_STATE_CANCELLED: _ClassVar[ProcessingState]
CONTENT_TYPE_UNSPECIFIED: ContentType
CONTENT_TYPE_TEXT: ContentType
CONTENT_TYPE_IMAGE: ContentType
CONTENT_TYPE_TABLE: ContentType
CONTENT_TYPE_DIAGRAM: ContentType
PROCESSING_STATE_UNSPECIFIED: ProcessingState
PROCESSING_STATE_QUEUED: ProcessingState
PROCESSING_STATE_PARTITIONING: ProcessingState
PROCESSING_STATE_CHUNKING: ProcessingState
PROCESSING_STATE_EMBEDDING: ProcessingState
PROCESSING_STATE_STORING: ProcessingState
PROCESSING_STATE_COMPLETED: ProcessingState
PROCESSING_STATE_FAILED: ProcessingState
PROCESSING_STATE_CANCELLED: ProcessingState

class UploadDocumentRequest(_message.Message):
    __slots__ = ("metadata", "data") # type: ignore
    METADATA_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    metadata: DocumentMetadata
    data: bytes
    def __init__(self, metadata: _Optional[_Union[DocumentMetadata, _Mapping]] = ..., data: _Optional[bytes] = ...) -> None: ...

class DocumentMetadata(_message.Message):
    __slots__ = ("file_name", "mime_type", "collection", "tags", "chunking_config") # type: ignore
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    CHUNKING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    file_name: str
    mime_type: str
    collection: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    chunking_config: ChunkingConfig
    def __init__(self, file_name: _Optional[str] = ..., mime_type: _Optional[str] = ..., collection: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., chunking_config: _Optional[_Union[ChunkingConfig, _Mapping]] = ...) -> None: ...

class ChunkingConfig(_message.Message):
    __slots__ = ("max_chunk_size", "chunk_overlap", "content_types") # type: ignore
    MAX_CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPES_FIELD_NUMBER: _ClassVar[int]
    max_chunk_size: int
    chunk_overlap: int
    content_types: _containers.RepeatedScalarFieldContainer[ContentType]
    def __init__(self, max_chunk_size: _Optional[int] = ..., chunk_overlap: _Optional[int] = ..., content_types: _Optional[_Iterable[_Union[ContentType, str]]] = ...) -> None: ...

class DocumentReceipt(_message.Message):
    __slots__ = ("document_id", "bytes_received", "accepted_at") # type: ignore
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    BYTES_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_AT_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    bytes_received: int
    accepted_at: _timestamp_pb2.Timestamp
    def __init__(self, document_id: _Optional[str] = ..., bytes_received: _Optional[int] = ..., accepted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CancelProcessingRequest(_message.Message):
    __slots__ = ("document_id",) # type: ignore
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: _Optional[str] = ...) -> None: ...

class GetDocumentStatusRequest(_message.Message):
    __slots__ = ("document_id",) # type: ignore
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: _Optional[str] = ...) -> None: ...

class DocumentStatus(_message.Message):
    __slots__ = ("document_id", "file_name", "collection", "state", "total_chunks", "processed_chunks", "error_message", "created_at", "updated_at") # type: ignore
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    file_name: str
    collection: str
    state: ProcessingState
    total_chunks: int
    processed_chunks: int
    error_message: str
    created_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, document_id: _Optional[str] = ..., file_name: _Optional[str] = ..., collection: _Optional[str] = ..., state: _Optional[_Union[ProcessingState, str]] = ..., total_chunks: _Optional[int] = ..., processed_chunks: _Optional[int] = ..., error_message: _Optional[str] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ListDocumentsRequest(_message.Message):
    __slots__ = ("collection", "state", "tags", "page_size", "page_token") # type: ignore
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    collection: str
    state: ProcessingState
    tags: _containers.RepeatedScalarFieldContainer[str]
    page_size: int
    page_token: str
    def __init__(self, collection: _Optional[str] = ..., state: _Optional[_Union[ProcessingState, str]] = ..., tags: _Optional[_Iterable[str]] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ListDocumentsResponse(_message.Message):
    __slots__ = ("documents", "next_page_token") # type: ignore
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[DocumentStatus]
    next_page_token: str
    def __init__(self, documents: _Optional[_Iterable[_Union[DocumentStatus, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class DeleteDocumentRequest(_message.Message):
    __slots__ = ("document_id",) # type: ignore
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    document_id: str
    def __init__(self, document_id: _Optional[str] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("query", "collection", "top_k", "score_threshold", "content_types", "tags") # type: ignore
    QUERY_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    query: str
    collection: str
    top_k: int
    score_threshold: float
    content_types: _containers.RepeatedScalarFieldContainer[ContentType]
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, query: _Optional[str] = ..., collection: _Optional[str] = ..., top_k: _Optional[int] = ..., score_threshold: _Optional[float] = ..., content_types: _Optional[_Iterable[_Union[ContentType, str]]] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchResult(_message.Message):
    __slots__ = ("chunk", "score") # type: ignore
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    chunk: DocumentChunk
    score: float
    def __init__(self, chunk: _Optional[_Union[DocumentChunk, _Mapping]] = ..., score: _Optional[float] = ...) -> None: ...

class DocumentChunk(_message.Message):
    __slots__ = ("chunk_id", "document_id", "chunk_index", "content_type", "text_content", "binary_content", "origin") # type: ignore
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_CONTENT_FIELD_NUMBER: _ClassVar[int]
    BINARY_CONTENT_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_FIELD_NUMBER: _ClassVar[int]
    chunk_id: str
    document_id: str
    chunk_index: int
    content_type: ContentType
    text_content: str
    binary_content: bytes
    origin: ChunkOrigin
    def __init__(self, chunk_id: _Optional[str] = ..., document_id: _Optional[str] = ..., chunk_index: _Optional[int] = ..., content_type: _Optional[_Union[ContentType, str]] = ..., text_content: _Optional[str] = ..., binary_content: _Optional[bytes] = ..., origin: _Optional[_Union[ChunkOrigin, _Mapping]] = ...) -> None: ...

class ChunkOrigin(_message.Message):
    __slots__ = ("page_number", "start_offset", "end_offset") # type: ignore
    PAGE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    START_OFFSET_FIELD_NUMBER: _ClassVar[int]
    END_OFFSET_FIELD_NUMBER: _ClassVar[int]
    page_number: int
    start_offset: int
    end_offset: int
    def __init__(self, page_number: _Optional[int] = ..., start_offset: _Optional[int] = ..., end_offset: _Optional[int] = ...) -> None: ...
