from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from .base import BaseCommunityDetector
from .leiden import LeidenDetector
from .sllpa import SLLPADetector

class CommunityDetectorFactory:
    """社区检测器工厂类"""
    
    ALGORITHMS = {
        'leiden': LeidenDetector,
        'sllpa': SLLPADetector
    }
    
    # 根据算法名组装对应社区检测器，传入 GDS 客户端和图连接。
    @classmethod
    def create(cls, algorithm: str, gds: GraphDataScience, graph: Neo4jGraph) -> BaseCommunityDetector:
        algorithm = algorithm.lower()
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"不支持的算法: {algorithm}")
        return cls.ALGORITHMS[algorithm](gds, graph)

__all__ = ['CommunityDetectorFactory', 'BaseCommunityDetector', 
           'LeidenDetector', 'SLLPADetector']