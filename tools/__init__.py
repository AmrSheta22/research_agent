"""Tools — self-describing, composable capabilities for the agent."""

from research_agent.tools.base import BaseTool, ToolResult
from research_agent.tools.data_tools import (
    ReadFileTool,
    CleanDataFrameTool,
    DetectVariableTypeTool,
    ProduceVariableSummaryTool,
    DatasetQueryTool,
)
from research_agent.tools.analysis_tools import AutonomousAnalysisTool
from research_agent.tools.rag_tool import StatisticalRAGTool
from research_agent.tools.search_tools import WebSearchTool, CitationLookupTool, AcademicLiteratureSearchTool
from research_agent.tools.statistics_tools import ConductTestsTool
from research_agent.tools.visualization_tools import PlanVisualizationsTool
from research_agent.tools.latex_paper_tool import CreateLatexPaperTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ReadFileTool",
    "CleanDataFrameTool",
    "DetectVariableTypeTool",
    "ProduceVariableSummaryTool",
    "DatasetQueryTool",
    "AutonomousAnalysisTool",
    "StatisticalRAGTool",
    "WebSearchTool",
    "CitationLookupTool",
    "AcademicLiteratureSearchTool",
    "ConductTestsTool",
    "PlanVisualizationsTool",
    "CreateLatexPaperTool",
]
