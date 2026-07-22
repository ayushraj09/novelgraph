"""
Stage 1: Data Schema

Defines the five domain node types as Cognee DataPoint models. cognify() uses
this schema to constrain what it extracts from raw text into the graph.

Edges that cognify() will create from this schema (important for Stage 3):
    Paper       -[method]->        Method
    Paper       -[dataset]->       DatasetNode
    Paper       -[result]->        Result
    DatasetNode -[tasks]->         Task
    Method      -[used_for]->      Task
    Result      -[derived_from]->  Method

Note there is NO direct edge type between Method and DatasetNode - a Paper is
what links them together. Stage 3's novelty query relies on this fact.
"""

from typing import List, Optional
from cognee.infrastructure.engine import DataPoint


class Task(DataPoint):
    name: str
    description: str = ""
    metadata: dict = {"index_fields": ["name", "description"]}


class DatasetNode(DataPoint):
    # Named DatasetNode (not "Dataset") to avoid clashing with Cognee's own
    # internal "Dataset" concept.
    name: str
    description: str = ""
    domain: str = ""
    tasks: List[Task] = []
    metadata: dict = {"index_fields": ["name", "description"]}


class Method(DataPoint):
    name: str
    description: str = ""
    method_type: str = ""
    used_for: List[Task] = []
    metadata: dict = {"index_fields": ["name", "description"]}


class Result(DataPoint):
    metric: str
    value: float
    derived_from: Optional[Method] = None
    metadata: dict = {"index_fields": ["metric"]}


class Paper(DataPoint):
    title: str
    abstract: str = ""
    year: Optional[int] = None
    method: Optional[Method] = None
    dataset: Optional[DatasetNode] = None
    result: Optional[Result] = None
    # Use title only for identity. The same source paper is extracted across
    # multiple chunks, and chunk-level abstracts can vary enough to otherwise
    # create duplicate Paper nodes for one uploaded PDF.
    metadata: dict = {"index_fields": ["title"]}
