from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from deep_research.skills.embeddings import EmbeddingProvider, get_embedding_provider
from deep_research.skills.config import get_skill_memory_config
from deep_research.skills.schema import ResearchSkill, SkillSearchResult


def build_embedding_text(skill: ResearchSkill) -> str:
    return "\n".join(
        [
            f"任务类型：{skill.task_type}",
            f"适用场景：{skill.applicability}",
            f"任务标签：{', '.join(skill.task_tags)}",
            "规划指导：" + "；".join(skill.planning_guidance),
            "研究指导：" + "；".join(skill.research_guidance),
            "写作指导：" + "；".join(skill.writing_guidance),
            "评估指导：" + "；".join(skill.evaluation_guidance),
            "红队审查指导：" + "；".join(skill.red_team_guidance),
            "常见失败模式：" + "；".join(skill.common_failure_modes),
            "不适用场景：" + "；".join(skill.not_applicable_when),
        ]
    )


def _skill_document(skill: ResearchSkill) -> str:
    return skill.model_dump_json()


def _skill_metadata(skill: ResearchSkill) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "task_type": skill.task_type,
        "language": skill.language,
        "quality_score": float(skill.quality_score),
        "usage_count": int(skill.usage_count),
        "success_count": int(skill.success_count),
        "source_run_id": skill.source_run_id,
        "created_at": skill.created_at,
        "updated_at": skill.updated_at,
        "disabled": bool(skill.metadata.get("disabled", False)),
    }


def _vector_score_from_distance(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def _rank_result(skill: ResearchSkill, vector_distance: float, metadata: dict[str, Any]) -> SkillSearchResult:
    vector_score = _vector_score_from_distance(vector_distance)
    quality_score = min(float(metadata.get("quality_score", skill.quality_score)) / 10.0, 1.0)
    usage_count = int(metadata.get("usage_count", skill.usage_count))
    success_count = int(metadata.get("success_count", skill.success_count))
    success_rate = success_count / usage_count if usage_count else 0.5
    final_score = vector_score * 0.55 + quality_score * 0.25 + success_rate * 0.15
    return SkillSearchResult(
        skill=skill,
        vector_distance=vector_distance,
        vector_score=vector_score,
        final_score=final_score,
    )


class FileVectorSkillStore:
    """Small local vector index used when chromadb is unavailable."""

    def __init__(self, persist_directory: str, embedding_provider: EmbeddingProvider) -> None:
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "skills_vector_index.json"
        self.embedding_provider = embedding_provider

    def _load(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_skill(self, skill: ResearchSkill) -> None:
        embedding = self.embedding_provider.embed_query(build_embedding_text(skill))
        records = [record for record in self._load() if record["id"] != skill.skill_id]
        records.append(
            {
                "id": skill.skill_id,
                "embedding": embedding,
                "document": _skill_document(skill),
                "metadata": _skill_metadata(skill),
            }
        )
        self._save(records)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        fetch_k: int,
        min_quality_score: float,
        similarity_threshold: float,
    ) -> list[SkillSearchResult]:
        query_embedding = self.embedding_provider.embed_query(query)
        candidates = []
        for record in self._load():
            metadata = record.get("metadata") or {}
            if metadata.get("disabled") or float(metadata.get("quality_score", 0.0)) < min_quality_score:
                continue
            distance = _cosine_distance(query_embedding, record["embedding"])
            result = _rank_result(ResearchSkill.model_validate_json(record["document"]), distance, metadata)
            if result.vector_score >= similarity_threshold:
                candidates.append(result)
        candidates.sort(key=lambda item: item.final_score, reverse=True)
        return candidates[:top_k]


class ChromaSkillStore:
    def __init__(self, persist_directory: str, collection_name: str, embedding_provider: EmbeddingProvider) -> None:
        import chromadb

        self.embedding_provider = embedding_provider
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_skill(self, skill: ResearchSkill) -> None:
        embedding = self.embedding_provider.embed_query(build_embedding_text(skill))
        self.collection.upsert(
            ids=[skill.skill_id],
            embeddings=[embedding],
            documents=[_skill_document(skill)],
            metadatas=[_skill_metadata(skill)],
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        fetch_k: int,
        min_quality_score: float,
        similarity_threshold: float,
    ) -> list[SkillSearchResult]:
        query_embedding = self.embedding_provider.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where={"quality_score": {"$gte": min_quality_score}},
            include=["documents", "metadatas", "distances"],
        )
        candidates = []
        for doc, metadata, distance in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            if metadata.get("disabled"):
                continue
            result = _rank_result(ResearchSkill.model_validate_json(doc), float(distance), metadata)
            if result.vector_score >= similarity_threshold:
                candidates.append(result)
        candidates.sort(key=lambda item: item.final_score, reverse=True)
        return candidates[:top_k]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    cosine = dot / (left_norm * right_norm)
    return 1.0 - cosine


def get_skill_store(skill_cfg: dict | None = None):
    cfg = skill_cfg or get_skill_memory_config()
    vector_cfg = cfg.get("vector_store") or {}
    embedding_provider = get_embedding_provider(cfg)
    persist_directory = vector_cfg.get("persist_directory")
    collection_name = vector_cfg.get("collection_name") or "research_skills"
    if (vector_cfg.get("backend") or "chroma").lower() == "chroma":
        try:
            return ChromaSkillStore(persist_directory, collection_name, embedding_provider)
        except Exception:
            pass
    return FileVectorSkillStore(persist_directory, embedding_provider)
