from uuid import UUID
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from web.backend.db.models import KnowledgeNode, KnowledgeEdge
from .base import BaseRepository


class KnowledgeRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def add_node(self, label: str, node_type: str, properties: dict | None = None) -> KnowledgeNode:
        node = KnowledgeNode(user_id=self.user_id, label=label, type=node_type, properties=properties or {})
        self.db.add(node)
        await self.db.flush()
        return node

    async def add_edge(self, source_id: UUID, target_id: UUID, relation_type: str,
                 properties: dict | None = None) -> KnowledgeEdge:
        edge = KnowledgeEdge(
            user_id=self.user_id, source_node_id=source_id, target_node_id=target_id,
            relation_type=relation_type, properties=properties or {},
        )
        self.db.add(edge)
        await self.db.flush()
        return edge

    async def get_nodes(self, node_type: str | None = None) -> list[KnowledgeNode]:
        stmt = select(KnowledgeNode).where(KnowledgeNode.user_id == self.user_id)
        if node_type:
            stmt = stmt.where(KnowledgeNode.type == node_type)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_edges(self) -> list[KnowledgeEdge]:
        result = await self.db.execute(
            select(KnowledgeEdge).where(KnowledgeEdge.user_id == self.user_id)
        )
        return list(result.scalars().all())

    async def find_node_by_label(self, label: str) -> KnowledgeNode | None:
        result = await self.db.execute(
            select(KnowledgeNode).where(
                KnowledgeNode.user_id == self.user_id,
                KnowledgeNode.label == label,
            )
        )
        return result.scalar_one_or_none()

    async def get_neighbors(self, node_id: UUID) -> list[KnowledgeNode]:
        # 获取与 node_id 相连的节点 ID
        stmt_out = select(KnowledgeEdge.target_node_id).where(
            KnowledgeEdge.user_id == self.user_id,
            KnowledgeEdge.source_node_id == node_id,
        )
        stmt_in = select(KnowledgeEdge.source_node_id).where(
            KnowledgeEdge.user_id == self.user_id,
            KnowledgeEdge.target_node_id == node_id,
        )
        result_out = await self.db.execute(stmt_out)
        result_in = await self.db.execute(stmt_in)
        ids = [r[0] for r in result_out.all()] + [r[0] for r in result_in.all()]
        if not ids:
            return []
        result = await self.db.execute(
            select(KnowledgeNode).where(KnowledgeNode.id.in_(ids))
        )
        return list(result.scalars().all())

    async def find_path(self, source_id: UUID, target_id: UUID, max_depth: int = 5) -> list:
        from collections import deque
        visited = {source_id}
        queue = deque([(source_id, [source_id])])
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if current == target_id:
                return path
            result = await self.db.execute(
                select(KnowledgeEdge).where(
                    KnowledgeEdge.user_id == self.user_id,
                    or_(
                        KnowledgeEdge.source_node_id == current,
                        KnowledgeEdge.target_node_id == current,
                    ),
                )
            )
            edges = list(result.scalars().all())
            for edge in edges:
                next_id = edge.target_node_id if edge.source_node_id == current else edge.source_node_id
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))
        return []

    async def get_graph_data(self, exclude_types: list[str] | None = None) -> dict:
        nodes = await self.get_nodes()
        edges = await self.get_edges()

        excluded = set(exclude_types) if exclude_types else set()
        excluded_ids = set()

        filtered_nodes = []
        for n in nodes:
            if n.type in excluded:
                excluded_ids.add(str(n.id))
                continue
            filtered_nodes.append(
                {"id": str(n.id), "label": n.label, "type": n.type, **(n.properties or {})}
            )

        filtered_links = []
        for e in edges:
            src, tgt = str(e.source_node_id), str(e.target_node_id)
            if src in excluded_ids or tgt in excluded_ids:
                continue
            filtered_links.append(
                {"source": src, "target": tgt, "relation": e.relation_type, **(e.properties or {})}
            )

        return {"nodes": filtered_nodes, "links": filtered_links}

    async def get_stats(self, exclude_types: list[str] | None = None) -> dict:
        nodes = await self.get_nodes()
        edges = await self.get_edges()
        excluded = set(exclude_types) if exclude_types else set()
        excluded_ids = {str(n.id) for n in nodes if n.type in excluded}
        filtered_nodes = [n for n in nodes if str(n.id) not in excluded_ids]
        filtered_edges = [
            e for e in edges
            if str(e.source_node_id) not in excluded_ids and str(e.target_node_id) not in excluded_ids
        ]

        node_types = {}
        for n in filtered_nodes:
            node_types[n.type] = node_types.get(n.type, 0) + 1
        edge_types = {}
        for e in filtered_edges:
            edge_types[e.relation_type] = edge_types.get(e.relation_type, 0) + 1
        return {
            "total_nodes": len(filtered_nodes),
            "total_edges": len(filtered_edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }
