from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import StrEnum

class NodeType(StrEnum):
    REASONING = "reasoning"
    TOOL_EXECUTION = "tool_execution"
    DATA_TRANSFORMATION = "data_transformation"

class EdgeCondition(BaseModel):
    condition_expression: str = Field(description="Логическое выражение для перехода (skip logic)")

class Edge(BaseModel):
    source: str = Field(description="ID исходного узла")
    target: str = Field(description="ID целевого узла")
    condition: Optional[EdgeCondition] = Field(default=None, description="Условие перехода")

class Node(BaseModel):
    id: str = Field(description="Уникальный ID узла")
    type: NodeType = Field(description="Тип узла (логика выполнения)")
    tool_name: Optional[str] = Field(default=None, description="Имя инструмента (обязательно для типа tool_execution)")
    description: Optional[str] = Field(default=None, description="Описание шага микро-графа")

class MicroGraph(BaseModel):
    nodes: List[Node] = Field(description="Массив узлов (изолированных этапов выполнения)")
    edges: List[Edge] = Field(description="Массив ребер (направленных связей между узлами)")
