"""知识图谱工具 - NetworkX DiGraph，JSON 序列化"""

import json
import logging
import os
import re
import hashlib
from typing import Optional

import httpx
import networkx as nx

from core.database import get_connection, get_paper
from tools.result import ok, fail

logger = logging.getLogger("research-server.knowledge_graph")

KG_PATH = os.path.join(os.environ.get("RESEARCH_DATA_DIR", "./data"), "knowledge_graph.json")
DEFAULT_KG_USER = "default"

# ── 实体提取词典 ────────────────────────────────────────────────────────────
# 每个类别：(canonical_name, [alias1, alias2, ...])
# 匹配时忽略大小写，匹配到后统一使用 canonical_name

METHODS = [
    ("Transformer", ["transformer", "self-attention", "self attention"]),
    ("CNN", ["cnn", "convolutional neural network", "convolutional network"]),
    ("RNN", ["rnn", "recurrent neural network"]),
    ("LSTM", ["lstm", "long short-term memory"]),
    ("GAN", ["gan", "generative adversarial network"]),
    ("Autoencoder", ["autoencoder", "auto-encoder", "vae", "variational autoencoder"]),
    ("Graph Neural Network", ["gnn", "graph neural network", "gcn", "graph convolutional"]),
    ("Attention Mechanism", ["attention mechanism", "attention model", "multi-head attention", "multihead attention"]),
    ("Contrastive Learning", ["contrastive learning", "contrastive loss", "simclr", "moco"]),
    ("Self-Supervised Learning", ["self-supervised learning", "self supervised learning"]),
    ("Transfer Learning", ["transfer learning", "pre-trained", "pretrained", "fine-tuning", "finetuning"]),
    ("Few-Shot Learning", ["few-shot learning", "few shot learning", "meta-learning", "meta learning"]),
    ("Reinforcement Learning", ["reinforcement learning", "rl", "policy gradient", "q-learning"]),
    ("Object Detection", ["object detection", "yolo", "faster r-cnn", "ssd"]),
    ("Segmentation", ["segmentation", "semantic segmentation", "instance segmentation", "panoptic segmentation"]),
    ("Multiple Instance Learning", ["multiple instance learning", "mil"]),
    ("Knowledge Distillation", ["knowledge distillation", "teacher-student", "teacher student"]),
    ("Diffusion Model", ["diffusion model", "diffusion", "ddpm", "score-based"]),
    ("Vision-Language Model", ["vision-language", "vision language", "clip", "blip"]),
    ("Large Language Model", ["large language model", "llm", "gpt", "bert", "llama"]),
    ("Vision Transformer", ["vision transformer", "vit", "swin transformer", "deit"]),
    ("Optical Flow", ["optical flow", "motion estimation"]),
    ("3D Convolution", ["3d convolution", "3d cnn", "c3d", "i3d"]),
    ("Temporal Modeling", ["temporal modeling", "temporal convolution", "temporal attention"]),
    ("Anomaly Scoring", ["anomaly scoring", "anomaly score", "abnormality score"]),
    ("Future Frame Prediction", ["future frame prediction", "frame prediction"]),
]

DATASETS = [
    ("UCF-Crime", ["ucf-crime", "ucf crime"]),
    ("ShanghaiTech", ["shanghaitech", "shanghai tech"]),
    ("Pedestrian", ["ped2", "avenue", "pedestrian"]),
    ("Kinetics", ["kinetics", "kinetics-400", "kinetics-600"]),
    ("ImageNet", ["imagenet", "ilsvrc"]),
    ("COCO", ["coco", "ms-coco", "ms coco"]),
    ("ActivityNet", ["activitynet", "activity net"]),
    ("THUMOS", ["thumos"]),
    ("MNIST", ["mnist"]),
    ("CIFAR", ["cifar", "cifar-10", "cifar-100"]),
    ("Cityscapes", ["cityscapes"]),
    ("ADE20K", ["ade20k"]),
    ("VOC", ["pascal voc", "voc2007", "voc2012"]),
    ("HMDB", ["hmdb", "hmdb51"]),
    ("UCF101", ["ucf101", "ucf-101"]),
    ("Something-Something", ["something-something", "something something"]),
    ("Sports-1M", ["sports-1m"]),
    ("YouTube-BoundingBoxes", ["youtube-bounding"]),
]

METRICS = [
    ("AUC-ROC", ["auc-roc", "auroc", "area under roc"]),
    ("Average Precision", ["mean average precision", "average precision"]),
    ("F1 Score", ["f1-score", "f1 score"]),
    ("PSNR", ["psnr", "peak signal-to-noise"]),
    ("SSIM", ["ssim", "structural similarity"]),
    ("FID", ["fid", "fréchet inception", "frechet inception"]),
    ("IoU", ["iou", "intersection over union"]),
    ("RMSE", ["rmse", "root mean square error"]),
    ("MAE", ["mean absolute error"]),
    ("BLEU", ["bleu score", "bleu metric"]),
    ("ROUGE", ["rouge score", "rouge metric", "rouge-l"]),
    ("CIDEr", ["cider score", "cider metric"]),
]

RESEARCH_ENTITY_TYPES = {"Method", "Dataset", "Task", "Metric", "Contribution", "Limitation", "Concept", "Unknown"}

RELATION_PRIORITY = {
    "extends": 100,
    "improves_metric_on": 90,
    "compares_with": 80,
    "cites": 70,
    "uses_same_dataset": 60,
    "solves_same_task": 55,
    "shares_method_family": 50,
    "related_to": 10,
}

TASKS = [
    ("Video Anomaly Detection", ["video anomaly detection", "video anomalous", "vad"]),
    ("Anomaly Detection", ["anomaly detection", "anomaly detection", "outlier detection", "abnormal event detection"]),
    ("Object Detection", ["object detection", "object recognition"]),
    ("Image Classification", ["image classification", "visual recognition"]),
    ("Action Recognition", ["action recognition", "activity recognition", "temporal action detection"]),
    ("Semantic Segmentation", ["semantic segmentation"]),
    ("Image Generation", ["image generation", "image synthesis"]),
    ("Text-to-Image", ["text-to-image", "text to image"]),
    ("Object Tracking", ["object tracking", "multi-object tracking", "mot"]),
    ("Pose Estimation", ["pose estimation", "human pose"]),
    ("Scene Understanding", ["scene understanding", "scene recognition"]),
    ("Representation Learning", ["representation learning", "feature learning"]),
    ("Domain Adaptation", ["domain adaptation", "domain generalization"]),
    ("Weakly Supervised Learning", ["weakly supervised", "weak supervision", "semi-supervised"]),
    ("Zero-Shot Learning", ["zero-shot learning", "zero shot"]),
    ("Video Understanding", ["video understanding", "video recognition"]),
    ("Person Re-Identification", ["person re-identification", "re-id", "reid"]),
    ("Change Detection", ["change detection"]),
    ("Medical Image Analysis", ["medical image", "medical imaging"]),
]

ENTITY_DICTIONARIES = {
    "Method": METHODS,
    "Dataset": DATASETS,
    "Task": TASKS,
    "Metric": METRICS,
}


def _extract_entities_from_text(text: str) -> list[dict]:
    """从文本中提取实体（方法/数据集/任务/指标），返回 [{name, type}]"""
    text_lower = text.lower()
    found = []
    seen = set()

    for category, dictionary in [("Method", METHODS), ("Dataset", DATASETS), ("Task", TASKS), ("Metric", METRICS)]:
        for canonical, aliases in dictionary:
            if canonical.lower() in seen:
                continue
            for alias in aliases:
                # 用 word boundary 匹配，避免子串误匹配
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    found.append({"name": canonical, "type": category})
                    seen.add(canonical.lower())
                    break

    return found


def _normalize_text(value: str) -> str:
    """用于实体匹配的宽松归一化。"""
    value = re.sub(r"\([^)]*\)", " ", value or "")
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def _slug(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value or "")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug:
        return slug
    digest = hashlib.sha1((value or "unknown").encode("utf-8")).hexdigest()[:12]
    return f"entity_{digest}"


def _strip_concept_prefix(value: str) -> str:
    value = (value or "").strip()
    while value.startswith("concept:"):
        value = value[len("concept:"):].strip()
    if re.fullmatch(r"concept_entity_[0-9a-f]{12}", value):
        value = value[len("concept_"):]
    return value


def _is_internal_id_label(value: str) -> bool:
    value = (value or "").strip()
    return value.startswith("concept:") or bool(re.fullmatch(r"(?:concept_)?entity_[0-9a-f]{12}", value))


def _aliases_for(canonical_name: str, entity_type: str) -> list[str]:
    aliases = {canonical_name}
    for canonical, known_aliases in ENTITY_DICTIONARIES.get(entity_type, []):
        if _normalize_text(canonical) == _normalize_text(canonical_name):
            aliases.update(known_aliases)
            break
    return sorted(a for a in aliases if a)


def _lookup_canonical_entity(name: str, preferred_type: str | None = None) -> tuple[str, str]:
    """返回 canonical_name 和 canonical_type。

    preferred_type 为 Unknown/Concept 时允许词典推断具体类型。
    """
    raw_name = _strip_concept_prefix(name)
    if not raw_name:
        return "Unknown", preferred_type or "Unknown"

    candidates = []
    if preferred_type in ENTITY_DICTIONARIES:
        candidates.append(preferred_type)
    candidates.extend(t for t in ("Method", "Dataset", "Task", "Metric") if t not in candidates)

    normalized = _normalize_text(raw_name)
    compact = normalized.replace(" ", "")
    for entity_type in candidates:
        for canonical, aliases in ENTITY_DICTIONARIES[entity_type]:
            known = {canonical, *aliases}
            normalized_known = {_normalize_text(v) for v in known}
            compact_known = {v.replace(" ", "") for v in normalized_known}
            if normalized in normalized_known or compact in compact_known:
                return canonical, entity_type

    if preferred_type in {None, "", "Unknown", "Concept"}:
        return raw_name, preferred_type or "Concept"
    return raw_name, preferred_type


def _canonical_entity(name: str, entity_type: str | None = None) -> dict:
    canonical_name, canonical_type = _lookup_canonical_entity(name, entity_type)
    raw_name = _strip_concept_prefix(name)
    aliases = set(_aliases_for(canonical_name, canonical_type))
    if raw_name and raw_name != canonical_name:
        aliases.add(raw_name)
    paren = re.search(r"\(([^)]+)\)", raw_name)
    if paren:
        aliases.add(paren.group(1).strip())

    return {
        "id": f"concept:{_slug(canonical_name)}",
        "name": canonical_name,
        "type": canonical_type,
        "aliases": sorted(a for a in aliases if a),
        "source_mention": raw_name,
    }


def _merge_unique(existing, additions) -> list:
    values = []
    for item in existing or []:
        if item not in values:
            values.append(item)
    for item in additions or []:
        if item not in values:
            values.append(item)
    return values


def _merge_node_attrs(existing: dict, incoming: dict) -> dict:
    merged = dict(existing or {})
    incoming = dict(incoming or {})

    if merged.get("type") in {None, "", "Unknown", "Concept"} and incoming.get("type") not in {None, "", "Unknown"}:
        merged["type"] = incoming["type"]
    elif not merged.get("type"):
        merged["type"] = incoming.get("type", "Concept")

    for key in ("name", "canonical_name"):
        incoming_value = incoming.get(key)
        existing_value = merged.get(key)
        if incoming_value and (not existing_value or _is_internal_id_label(str(existing_value))) and not _is_internal_id_label(str(incoming_value)):
            merged[key] = incoming_value

    if incoming.get("description") and not merged.get("description"):
        merged["description"] = incoming["description"]

    aliases = [a for a in incoming.get("aliases", []) if not _is_internal_id_label(str(a))]
    mentions = [m for m in incoming.get("source_mentions", []) if not _is_internal_id_label(str(m))]
    merged["aliases"] = _merge_unique(
        [a for a in merged.get("aliases", []) if not _is_internal_id_label(str(a))],
        aliases,
    )
    merged["source_mentions"] = _merge_unique(
        [m for m in merged.get("source_mentions", []) if not _is_internal_id_label(str(m))],
        mentions,
    )
    return merged


def _add_or_update_edge(G: nx.DiGraph, source: str, target: str, relation: str, **properties) -> bool:
    """添加边；若同一对节点已有边，按语义优先级合并。"""
    if not source or not target or source == target:
        return False

    properties = {k: v for k, v in properties.items() if v not in (None, [], {}, "")}
    if not G.has_edge(source, target):
        G.add_edge(source, target, type=relation, **properties)
        return True

    edge = G.edges[source, target]
    existing_relation = edge.get("type", "related_to")
    changed = False
    promoted = False

    if RELATION_PRIORITY.get(relation, 0) > RELATION_PRIORITY.get(existing_relation, 0):
        edge["type"] = relation
        edge["alternate_relations"] = _merge_unique(edge.get("alternate_relations"), [existing_relation])
        changed = True
        promoted = True
    elif relation != existing_relation:
        edge["alternate_relations"] = _merge_unique(edge.get("alternate_relations"), [relation])
        changed = True

    for key, value in properties.items():
        if isinstance(value, list):
            merged = _merge_unique(edge.get(key), value)
            if merged != edge.get(key):
                edge[key] = merged
                changed = True
        elif promoted or key not in edge:
            edge[key] = value
            changed = True
    return changed


def _merge_nodes(G: nx.DiGraph, source_id: str, target_id: str, target_attrs: dict) -> bool:
    if source_id == target_id or not G.has_node(source_id):
        return False

    if not G.has_node(target_id):
        G.add_node(target_id, **target_attrs)
    else:
        G.nodes[target_id].update(_merge_node_attrs(G.nodes[target_id], target_attrs))

    source_attrs = G.nodes[source_id]
    G.nodes[target_id].update(_merge_node_attrs(G.nodes[target_id], source_attrs))

    for pred, _, data in list(G.in_edges(source_id, data=True)):
        if pred != target_id:
            _add_or_update_edge(G, pred, target_id, data.get("type", "related_to"),
                                **{k: v for k, v in data.items() if k != "type"})
    for _, succ, data in list(G.out_edges(source_id, data=True)):
        if succ != target_id:
            _add_or_update_edge(G, target_id, succ, data.get("type", "related_to"),
                                **{k: v for k, v in data.items() if k != "type"})

    G.remove_node(source_id)
    return True


def _upsert_entity_node(G: nx.DiGraph, name: str, entity_type: str, paper_id: str | None = None) -> tuple[str, bool]:
    entity = _canonical_entity(name, entity_type)
    node_attrs = {
        "type": entity["type"],
        "name": entity["name"],
        "canonical_name": entity["name"],
        "aliases": [a for a in entity["aliases"] if not _is_internal_id_label(a)],
        "source_mentions": [entity["source_mention"]] if entity["source_mention"] and not _is_internal_id_label(entity["source_mention"]) else [],
    }
    if paper_id and entity["source_mention"] and not _is_internal_id_label(entity["source_mention"]):
        node_attrs["source_mentions"] = [f"{paper_id}: {entity['source_mention']}"]

    node_id = entity["id"]
    changed = False
    if G.has_node(node_id):
        merged = _merge_node_attrs(G.nodes[node_id], node_attrs)
        changed = merged != dict(G.nodes[node_id])
        G.nodes[node_id].update(merged)
    else:
        G.add_node(node_id, **node_attrs)
        changed = True

    aliases_norm = {_normalize_text(a) for a in entity["aliases"]}
    aliases_norm.add(_normalize_text(entity["name"]))
    raw_slug_ids = {
        name,
        f"concept:{_slug(name)}",
        f"concept:{_slug(entity['name'])}",
    }
    for other_id, data in list(G.nodes(data=True)):
        if other_id == node_id or data.get("type") in {"Paper", "Author"}:
            continue
        label = data.get("canonical_name") or data.get("name") or data.get("title") or other_id
        if other_id in raw_slug_ids or _normalize_text(label) in aliases_norm:
            changed = _merge_nodes(G, other_id, node_id, node_attrs) or changed

    return node_id, changed


def _normalize_graph_entities(G: nx.DiGraph) -> int:
    """合并已知研究实体的别名节点，返回合并/更新次数。"""
    changed = 0
    for node_id, data in list(G.nodes(data=True)):
        node_type = data.get("type", "Unknown")
        if node_type in {"Paper", "Author"}:
            continue
        label = data.get("canonical_name") or data.get("name") or data.get("title") or node_id
        if not label:
            continue
        _, did_change = _upsert_entity_node(G, label, node_type)
        if did_change:
            changed += 1
    return changed


def _entity_label(G: nx.DiGraph, node_id: str) -> str:
    data = G.nodes.get(node_id, {})
    return data.get("canonical_name") or data.get("name") or data.get("title") or node_id


def _paper_features(G: nx.DiGraph, paper_id: str) -> dict[str, set[str]]:
    features = {
        "tasks": set(),
        "datasets": set(),
        "methods": set(),
        "metrics": set(),
    }
    relation_map = {
        "addresses_task": "tasks",
        "evaluated_on": "datasets",
        "uses_method": "methods",
        "evaluated_with": "metrics",
    }
    for _, target, data in G.out_edges(paper_id, data=True):
        bucket = relation_map.get(data.get("type"))
        if bucket:
            features[bucket].add(target)
    return features


def _shared_labels(G: nx.DiGraph, values: set[str]) -> list[str]:
    return sorted(_entity_label(G, node_id) for node_id in values)


def _paper_year(G: nx.DiGraph, paper_id: str) -> int | None:
    year = G.nodes.get(paper_id, {}).get("year")
    return year if isinstance(year, int) else None


def _newer_first(G: nx.DiGraph, left: str, right: str) -> tuple[str, str]:
    left_year = _paper_year(G, left)
    right_year = _paper_year(G, right)
    if left_year is not None and right_year is not None and right_year > left_year:
        return right, left
    return left, right


def _infer_paper_relations(G: nx.DiGraph, paper_id: str | None = None) -> int:
    """根据共享实体和引用关系推断论文间语义边。"""
    paper_ids = [
        node_id for node_id, data in G.nodes(data=True)
        if data.get("type") == "Paper" and (paper_id is None or node_id == paper_id or G.has_node(paper_id))
    ]
    if paper_id:
        paper_ids = [paper_id] if G.has_node(paper_id) else []

    changed = 0
    all_papers = [node_id for node_id, data in G.nodes(data=True) if data.get("type") == "Paper"]
    for current in paper_ids:
        current_features = _paper_features(G, current)
        if not any(current_features.values()):
            continue

        for other in all_papers:
            if other == current:
                continue
            other_features = _paper_features(G, other)
            shared = {
                "shared_tasks": current_features["tasks"] & other_features["tasks"],
                "shared_datasets": current_features["datasets"] & other_features["datasets"],
                "shared_methods": current_features["methods"] & other_features["methods"],
                "shared_metrics": current_features["metrics"] & other_features["metrics"],
            }
            if not any(shared.values()):
                continue
            direct_citation = G.has_edge(current, other) and G.edges[current, other].get("type") == "cites"

            props = {
                "shared_tasks": _shared_labels(G, shared["shared_tasks"]),
                "shared_datasets": _shared_labels(G, shared["shared_datasets"]),
                "shared_methods": _shared_labels(G, shared["shared_methods"]),
                "shared_metrics": _shared_labels(G, shared["shared_metrics"]),
            }

            if shared["shared_datasets"]:
                changed += int(_add_or_update_edge(G, current, other, "uses_same_dataset",
                                                   inference="shared_entities", confidence=0.55, **props))
            if shared["shared_tasks"]:
                changed += int(_add_or_update_edge(G, current, other, "solves_same_task",
                                                   inference="shared_entities", confidence=0.55, **props))
            if shared["shared_methods"]:
                changed += int(_add_or_update_edge(G, current, other, "shares_method_family",
                                                   inference="shared_entities", confidence=0.55, **props))

            overlap_groups = sum(1 for values in shared.values() if values)
            if overlap_groups >= 2:
                changed += int(_add_or_update_edge(G, current, other, "compares_with",
                                                   inference="shared_entities", confidence=0.65, **props))

            if shared["shared_metrics"] and (shared["shared_datasets"] or shared["shared_tasks"]):
                source, target = _newer_first(G, current, other)
                changed += int(_add_or_update_edge(
                    G, source, target, "improves_metric_on",
                    inference="candidate_from_shared_metric",
                    confidence=0.35,
                    evidence_note="Shared metric context only; verify actual reported scores before treating as proven improvement.",
                    **props,
                ))

            if direct_citation:
                if shared["shared_tasks"] or shared["shared_methods"]:
                    changed += int(_add_or_update_edge(
                        G, current, other, "extends",
                        inference="citation_plus_shared_entities",
                        confidence=0.75,
                        **props,
                    ))

    return changed


def _safe_user_id(user_id: str | None) -> str:
    raw = str(user_id or DEFAULT_KG_USER).strip() or DEFAULT_KG_USER
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)
    return safe.strip("._") or DEFAULT_KG_USER


def _kg_path(user_id: str | None = None) -> str:
    """Return the per-user knowledge graph path under workspace/{user_id}."""
    safe_user_id = _safe_user_id(user_id)
    try:
        from novare.config import get_user_workspace
        workspace = get_user_workspace(safe_user_id)
    except Exception:
        base = os.getenv("NOVARE_WORKSPACE", "workspace")
        workspace = os.path.join(base, safe_user_id)
        os.makedirs(workspace, exist_ok=True)
    return os.path.join(workspace, "knowledge_graph.json")


# ── LLM 实体抽取 ──────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """你是一个学术论文实体抽取助手。请从以下论文摘要中抽取研究实体，输出 JSON 数组。

抽取类型和说明：
- Method: 模型架构、算法、技术方法（如 Transformer, CNN, Contrastive Learning 等）
- Task: 研究任务或问题（如 Object Detection, Anomaly Detection 等）
- Dataset: 数据集名称（如 UCF-Crime, ImageNet, COCO 等）
- Metric: 评估指标（如 AUC-ROC, PSNR, mAP, F1 Score 等）
- Contribution: 论文的主要贡献或创新点，用简洁短语描述（如 "提出时序建模框架", "首次将对比学习用于异常检测"）
- Limitation: 论文明确提到的局限性，用简洁短语描述（如 "依赖光流输入", "无法处理实时场景"）

规则：
1. 每个实体用 canonical name（英文），不要用缩写作主名
2. Contribution 和 Limitation 必须是论文摘要中明确提及的，不要推测
3. 每种类型最多提取 5 个
4. 只输出 JSON 数组，不要输出其他内容

输出格式：
[{"name": "实体名称", "type": "Method"}, ...]

论文摘要：
{text}"""

# 全局 httpx 客户端（惰性初始化）
_llm_http: httpx.Client | None = None


def _get_llm_client() -> httpx.Client | None:
    """惰性初始化 LLM httpx 同步客户端"""
    global _llm_http
    if _llm_http is not None:
        return _llm_http

    enabled = os.environ.get("NOVARE_KG_LLM_EXTRACT", "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        logger.debug("NOVARE_KG_LLM_EXTRACT 未启用，跳过 LLM 实体抽取")
        return None

    api_key = os.environ.get("NOVARE_API_KEY", "")
    base_url = os.environ.get("NOVARE_BASE_URL", "https://api.minimax.chat/v1")
    if not api_key:
        logger.warning("NOVARE_API_KEY 未配置，LLM 实体抽取不可用")
        return None

    _llm_http = httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )
    return _llm_http


def _llm_extract_entities(text: str) -> list[dict]:
    """使用 LLM 从论文摘要中抽取研究实体，返回 [{name, type}]。

    抽取失败时返回空列表，调用方应 fallback 到字典匹配。
    """
    client = _get_llm_client()
    if client is None:
        return []

    model = os.environ.get("NOVARE_MODEL", "MiniMax-Text-01")
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": _EXTRACTION_PROMPT.format(text=text)},
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
        "stream": False,
    }

    try:
        resp = client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # 尝试从响应中提取 JSON 数组（可能被 ```json ``` 包裹）
        if "```" in content:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        elif not content.startswith("["):
            # 尝试找到第一个 [ 到最后一个 ]
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                content = content[start:end + 1]

        entities = json.loads(content)
        if not isinstance(entities, list):
            logger.warning("LLM 返回非数组: %s", content[:200])
            return []

        # 校验实体结构
        valid_types = {"Method", "Task", "Dataset", "Metric", "Contribution", "Limitation"}
        result = []
        for e in entities:
            if not isinstance(e, dict) or "name" not in e or "type" not in e:
                continue
            if e["type"] not in valid_types:
                logger.debug("跳过未知实体类型: %s", e["type"])
                continue
            result.append({"name": str(e["name"]).strip(), "type": e["type"]})

        logger.info("LLM 实体抽取成功: %d 个实体", len(result))
        return result

    except httpx.HTTPError as e:
        logger.warning("LLM 实体抽取 HTTP 错误: %s", e)
        return []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("LLM 实体抽取解析错误: %s", e)
        return []
    except Exception as e:
        logger.warning("LLM 实体抽取未知错误: %s", e)
        return []


def _load_graph(user_id: str | None = None) -> nx.DiGraph:
    """加载知识图谱"""
    path = _kg_path(user_id)
    legacy_fallback = not user_id and not os.path.exists(path) and os.path.exists(KG_PATH)
    load_path = KG_PATH if legacy_fallback else path
    if os.path.exists(load_path):
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            edge_key = "edges" if "edges" in data else "links"
            try:
                G = nx.node_link_graph(data, directed=True, edges=edge_key)
            except TypeError:
                G = nx.node_link_graph(data, directed=True, link=edge_key)
            G.graph["_kg_user_id"] = _safe_user_id(user_id)
            G.graph["_kg_path"] = path
            return G
        except Exception as e:
            logger.warning("Failed to load knowledge graph from %s: %s", load_path, e)
    G = nx.DiGraph()
    G.graph["_kg_user_id"] = _safe_user_id(user_id)
    G.graph["_kg_path"] = path
    return G


def _save_graph(G: nx.DiGraph, user_id: str | None = None) -> None:
    """保存知识图谱"""
    path = _kg_path(user_id or G.graph.get("_kg_user_id"))
    G.graph["_kg_user_id"] = _safe_user_id(user_id or G.graph.get("_kg_user_id"))
    G.graph["_kg_path"] = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _normalize_graph_entities(G)
    graph_meta = dict(G.graph)
    G.graph.pop("_kg_user_id", None)
    G.graph.pop("_kg_path", None)
    try:
        data = nx.node_link_data(G, edges="edges")
    except TypeError:
        data = nx.node_link_data(G)
    G.graph.update(graph_meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Actions that mutate the graph (need PG sync)
_MUTATING_ACTIONS = {"add_paper", "add_concept", "add_relation", "extract_from_abstract", "normalize"}


async def save_to_pg(user_id: str, graph: nx.DiGraph):
    """Save NetworkX graph nodes/edges to PostgreSQL (secondary storage)."""
    if not user_id:
        return
    try:
        from web.backend.db.base import get_session_factory
        from web.backend.db.models import KnowledgeNode, KnowledgeEdge
        from uuid import UUID
        from sqlalchemy import select
        async with get_session_factory()() as db:
            uid = UUID(user_id)
            label_to_id = {}

            for node_id, data in graph.nodes(data=True):
                label = data.get("label") or data.get("name") or data.get("title") or node_id
                node_type = data.get("type", "concept")
                result = await db.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.user_id == uid,
                        KnowledgeNode.label == label,
                    )
                )
                existing = result.scalar_one_or_none()
                if not existing:
                    pg_node = KnowledgeNode(
                        user_id=uid,
                        label=label,
                        type=node_type,
                        properties={k: v for k, v in data.items() if k not in ("label", "type", "name", "title")},
                    )
                    db.add(pg_node)
                    await db.flush()
                    label_to_id[node_id] = pg_node.id
                else:
                    existing.type = node_type
                    existing.properties = {k: v for k, v in data.items() if k not in ("label", "type", "name", "title")}
                    label_to_id[node_id] = existing.id

            await db.commit()

            for u, v, edata in graph.edges(data=True):
                source_label = graph.nodes.get(u, {}).get("label") or graph.nodes.get(u, {}).get("name") or graph.nodes.get(u, {}).get("title") or u
                target_label = graph.nodes.get(v, {}).get("label") or graph.nodes.get(v, {}).get("name") or graph.nodes.get(v, {}).get("title") or v
                rel_type = edata.get("type", "related_to")

                src_id = label_to_id.get(u)
                tgt_id = label_to_id.get(v)
                if not src_id:
                    result = await db.execute(
                        select(KnowledgeNode).where(KnowledgeNode.user_id == uid, KnowledgeNode.label == source_label)
                    )
                    src_node = result.scalar_one_or_none()
                    src_id = src_node.id if src_node else None
                if not tgt_id:
                    result = await db.execute(
                        select(KnowledgeNode).where(KnowledgeNode.user_id == uid, KnowledgeNode.label == target_label)
                    )
                    tgt_node = result.scalar_one_or_none()
                    tgt_id = tgt_node.id if tgt_node else None

                if not src_id or not tgt_id:
                    continue

                result = await db.execute(
                    select(KnowledgeEdge).where(
                        KnowledgeEdge.user_id == uid,
                        KnowledgeEdge.source_node_id == src_id,
                        KnowledgeEdge.target_node_id == tgt_id,
                        KnowledgeEdge.relation_type == rel_type,
                    )
                )
                existing_edge = result.scalar_one_or_none()
                edge_props = {k: v for k, v in edata.items() if k != "type"}
                if not existing_edge:
                    db.add(KnowledgeEdge(
                        user_id=uid,
                        source_node_id=src_id,
                        target_node_id=tgt_id,
                        relation_type=rel_type,
                        properties=edge_props,
                    ))
                else:
                    existing_edge.properties = edge_props

            await db.commit()
    except Exception as e:
        logger.warning("Failed to save KG to PostgreSQL: %s", e)


async def _action_add_paper(G: nx.DiGraph, args: dict) -> str:
    """添加论文节点及其关系"""
    paper_id = args.get("paper_id")
    if not paper_id:
        return fail("knowledge_graph", "请提供 paper_id。")

    async with get_connection() as conn:
        paper = await get_paper(conn, paper_id)
        if not paper:
            return fail("knowledge_graph", f"论文 {paper_id} 不存在于数据库中。请先使用 paper_search 搜索。")

        from core.database import get_citations
        citations = await get_citations(conn, paper_id)

    # 创建 Paper 节点
    G.add_node(paper_id, type="Paper", title=paper["title"],
               year=paper.get("year"), citation_count=paper.get("citation_count", 0))

    # 解析作者，创建 Author 节点 + authored_by 边
    authors = json.loads(paper.get("authors", "[]")) if isinstance(paper.get("authors"), str) else paper.get("authors", [])
    for author_name in authors:
        if not author_name:
            continue
        author_id = f"author:{author_name.strip().lower().replace(' ', '_')}"
        if not G.has_node(author_id):
            G.add_node(author_id, type="Author", name=author_name.strip())
        _add_or_update_edge(G, paper_id, author_id, "authored_by")

    # 引用关系
    for cited_id in citations.get("citing", []):
        if not G.has_node(cited_id):
            G.add_node(cited_id, type="Paper", title=cited_id)
        _add_or_update_edge(G, paper_id, cited_id, "cites")

    _normalize_graph_entities(G)
    semantic_count = _infer_paper_relations(G, paper_id)

    _save_graph(G)

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    return ok("knowledge_graph", {
        "action": "add_paper",
        "title": paper["title"],
        "author_count": len(authors),
        "citation_count": len(citations.get("citing", [])),
        "semantic_relations_added": semantic_count,
        "graph_stats": {"nodes": node_count, "edges": edge_count},
    }, summary=f"已添加论文到知识图谱: {paper['title']}")


def _action_add_concept(G: nx.DiGraph, args: dict) -> str:
    """添加概念节点"""
    name = args.get("name")
    if not name:
        return fail("knowledge_graph", "请提供概念名称 (name)。")

    concept_id, _ = _upsert_entity_node(G, name.strip(), args.get("type", "Concept"))
    if args.get("description"):
        G.nodes[concept_id]["description"] = args["description"]
    _save_graph(G)
    return ok("knowledge_graph", {
        "action": "add_concept",
        "name": G.nodes[concept_id].get("name", name),
        "canonical_id": concept_id,
        "aliases": G.nodes[concept_id].get("aliases", []),
    }, summary=f"已添加概念: {G.nodes[concept_id].get('name', name)}")


def _action_add_relation(G: nx.DiGraph, args: dict) -> str:
    """添加关系边"""
    subject = args.get("subject")
    predicate = args.get("predicate")
    obj = args.get("object")

    if not all([subject, predicate, obj]):
        return fail("knowledge_graph", "请提供 subject、predicate、object。")

    subject_id = _find_node(G, subject) or subject
    object_id = _find_node(G, obj) or obj

    # 确保节点存在；已知研究实体会被归一化为 canonical concept 节点。
    if not G.has_node(subject_id):
        subject_id, _ = _upsert_entity_node(G, subject, "Unknown")
    if not G.has_node(object_id):
        object_id, _ = _upsert_entity_node(G, obj, "Unknown")

    _add_or_update_edge(G, subject_id, object_id, predicate)
    _normalize_graph_entities(G)
    semantic_count = _infer_paper_relations(G)
    _save_graph(G)
    return ok("knowledge_graph", {
        "action": "add_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "semantic_relations_added": semantic_count,
    }, summary=f"已添加关系: {subject} --[{predicate}]--> {obj}")


def _action_query(G: nx.DiGraph, args: dict) -> str:
    """查询子图"""
    subject = args.get("subject")
    predicate = args.get("predicate")
    obj = args.get("object")

    if not any([subject, predicate, obj]):
        # 返回图谱概况 — 委托给 _action_stats 格式
        return _action_stats(G, args)

    matching_edges = []
    for u, v, data in G.edges(data=True):
        match = True
        if subject and u != subject:
            # 也检查节点名称
            u_data = G.nodes.get(u, {})
            if u_data.get("name", "").lower() != subject.lower():
                match = False
        if predicate and data.get("type") != predicate:
            match = False
        if obj and v != obj:
            v_data = G.nodes.get(v, {})
            if v_data.get("name", "").lower() != obj.lower():
                match = False
        if match:
            matching_edges.append((u, v, data))

    if not matching_edges:
        return fail("knowledge_graph", "未找到匹配的关系。")

    relations = []
    for u, v, data in matching_edges[:20]:
        u_label = G.nodes.get(u, {}).get("name", G.nodes.get(u, {}).get("title", u))
        v_label = G.nodes.get(v, {}).get("name", G.nodes.get(v, {}).get("title", v))
        relations.append({"subject": u_label, "predicate": data.get("type", "?"), "object": v_label})

    return ok("knowledge_graph", {
        "action": "query",
        "relations": relations,
        "graph_stats": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
    }, summary=f"查询到 {len(matching_edges)} 条关系")


def _action_find_path(G: nx.DiGraph, args: dict) -> str:
    """查找两个实体之间的路径"""
    source = args.get("subject")
    target = args.get("target")

    if not source or not target:
        return fail("knowledge_graph", "请提供 subject 和 target。")

    # 尝试直接匹配或按名称匹配
    source_id = _find_node(G, source)
    target_id = _find_node(G, target)

    if not source_id:
        return fail("knowledge_graph", f"未找到节点 {source}。")
    if not target_id:
        return fail("knowledge_graph", f"未找到节点 {target}。")

    try:
        path = nx.shortest_path(G, source_id, target_id)
        path_labels = [_node_label(G, n) for n in path]
        edges_list = []
        for i in range(len(path) - 1):
            edge_data = G.edges[path[i], path[i + 1]]
            edges_list.append({
                "subject": _node_label(G, path[i]),
                "predicate": edge_data.get("type", "?"),
                "object": _node_label(G, path[i + 1]),
            })
        return ok("knowledge_graph", {
            "action": "find_path",
            "path": path_labels,
            "edges": edges_list,
        }, summary=f"找到路径: {' -> '.join(path_labels)}")
    except nx.NetworkXNoPath:
        return fail("knowledge_graph", f"未找到从 {source} 到 {target} 的路径。")
    except Exception as e:
        return fail("knowledge_graph", str(e))


def _action_stats(G: nx.DiGraph, args: dict) -> str:
    """图谱统计"""
    if G.number_of_nodes() == 0:
        return ok("knowledge_graph", {
            "action": "stats",
            "node_count": 0,
            "edge_count": 0,
            "node_types": {},
            "edge_types": {},
        }, summary="知识图谱为空")

    # 节点类型统计
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # 边类型统计
    edge_counts = {}
    for _, _, data in G.edges(data=True):
        t = data.get("type", "Unknown")
        edge_counts[t] = edge_counts.get(t, 0) + 1

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    return ok("knowledge_graph", {
        "action": "stats",
        "node_count": node_count,
        "edge_count": edge_count,
        "node_types": type_counts,
        "edge_types": edge_counts,
    }, summary=f"知识图谱: {node_count} 节点, {edge_count} 边")


def _action_normalize(G: nx.DiGraph, args: dict) -> str:
    """归一化已有实体并重建论文间语义关系。"""
    normalized_count = _normalize_graph_entities(G)
    semantic_count = _infer_paper_relations(G, args.get("paper_id"))
    _save_graph(G)
    return ok("knowledge_graph", {
        "action": "normalize",
        "paper_id": args.get("paper_id"),
        "normalized_entities": normalized_count,
        "semantic_relations_added": semantic_count,
        "graph_stats": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
    }, summary=f"图谱归一化完成: 归一化/合并 {normalized_count} 个实体，补充 {semantic_count} 条论文语义关系")


async def _action_extract_from_abstract(G: nx.DiGraph, args: dict) -> str:
    """从论文摘要中自动提取实体并构建知识图谱

    支持两种模式：
    1. 自动模式：提供 paper_id，优先用 LLM 从摘要抽取，失败则回退词典匹配
    2. 手动模式：提供 paper_id + entities 列表，由调用方提供实体
    """
    paper_id = args.get("paper_id")
    if not paper_id:
        return fail("knowledge_graph", "请提供 paper_id。")

    async with get_connection() as conn:
        paper = await get_paper(conn, paper_id)

    if not paper:
        return fail("knowledge_graph", f"论文 {paper_id} 不存在于数据库中。请先使用 paper_search 搜索。")

    # 确保 Paper 节点在图谱中
    if not G.has_node(paper_id):
        G.add_node(paper_id, type="Paper", title=paper.get("title", ""),
                   year=paper.get("year"))

    # 作者节点
    authors = json.loads(paper.get("authors", "[]")) if isinstance(paper.get("authors"), str) else paper.get("authors", [])
    for author_name in authors:
        if not author_name:
            continue
        author_id = f"author:{author_name.strip().lower().replace(' ', '_')}"
        if not G.has_node(author_id):
            G.add_node(author_id, type="Author", name=author_name.strip())
        if not G.has_edge(paper_id, author_id):
            _add_or_update_edge(G, paper_id, author_id, "authored_by")

    # 获取实体列表：优先使用手动提供的 entities，否则从摘要自动提取
    manual_entities = args.get("entities", [])
    if manual_entities:
        entities = [
            {"name": e["name"], "type": e.get("type", "Concept")}
            for e in manual_entities if e.get("name")
        ]
        source = "手动提供"
    else:
        abstract = paper.get("abstract", "")
        if not abstract:
            # 尝试从已解析的分块中获取摘要
            async with get_connection() as conn:
                from web.backend.db.models import Chunk
                from sqlalchemy import select
                result = await conn.execute(
                    select(Chunk.text)
                    .where(Chunk.paper_id == paper_id, Chunk.section.ilike("%abstract%"))
                    .order_by(Chunk.ordinal)
                    .limit(1)
                )
                rows = result.all()
            if rows:
                abstract = rows[0][0]
        if not abstract:
            _save_graph(G)
            return fail("knowledge_graph", f"论文 {paper_id} 没有摘要，无法自动提取实体。请手动提供 entities 参数。")

        # 优先使用 LLM 抽取，失败则回退词典匹配
        entities = _llm_extract_entities(abstract)
        if entities:
            source = "LLM 抽取"
        else:
            entities = _extract_entities_from_text(abstract)
            source = "词典匹配（LLM 未启用或不可用）"

    if not entities:
        _save_graph(G)
        return fail("knowledge_graph", f"未从论文 {paper_id} 中提取到已知实体。",
                     data={"action": "extract_from_abstract", "paper_id": paper_id,
                           "entities_added": 0, "relations_added": 0,
                           "graph_stats": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}})

    # 关系映射：实体类型 → 论文到实体的关系名
    RELATION_MAP = {
        "Method": "uses_method",
        "Dataset": "evaluated_on",
        "Task": "addresses_task",
        "Metric": "evaluated_with",
        "Contribution": "contributes",
        "Limitation": "limited_by",
    }

    added = []
    entity_ids = []
    for ent in entities:
        name = ent["name"]
        etype = ent["type"]
        concept_id, _ = _upsert_entity_node(G, name, etype, paper_id=paper_id)
        entity_ids.append((concept_id, G.nodes[concept_id].get("type", etype)))

        # 创建关系
        relation = RELATION_MAP.get(G.nodes[concept_id].get("type", etype), "related_to")
        if _add_or_update_edge(G, paper_id, concept_id, relation):
            added.append(f"  {paper.get('title', paper_id)[:40]} --[{relation}]--> {G.nodes[concept_id].get('name', name)}")

    # 自动建立 Method 之间的 "related_to" 关系
    method_ids = [
        entity_id
        for entity_id, entity_type in entity_ids if entity_type == "Method"
    ]
    for i, m1 in enumerate(method_ids):
        for m2 in method_ids[i+1:]:
            if not G.has_edge(m1, m2) and not G.has_edge(m2, m1):
                _add_or_update_edge(G, m1, m2, "related_to")

    normalized_count = _normalize_graph_entities(G)
    semantic_count = _infer_paper_relations(G, paper_id)

    _save_graph(G)

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    return ok("knowledge_graph", {
        "action": "extract_from_abstract",
        "paper_id": paper_id,
        "source": source,
        "entities_added": len(entities),
        "relations_added": len(added),
        "normalized_entities": normalized_count,
        "semantic_relations_added": semantic_count,
        "graph_stats": {"nodes": node_count, "edges": edge_count},
    }, summary=f"实体提取完成({source}): {paper.get('title', paper_id)}")


def _find_node(G: nx.DiGraph, name: str) -> Optional[str]:
    """按 ID 或名称查找节点"""
    if G.has_node(name):
        return name
    name_lower = name.lower()
    name_norm = _normalize_text(name)
    for node, data in G.nodes(data=True):
        if data.get("name", "").lower() == name_lower:
            return node
        if data.get("title", "").lower() == name_lower:
            return node
        if _normalize_text(data.get("canonical_name", "")) == name_norm:
            return node
        aliases = data.get("aliases") or []
        if any(_normalize_text(alias) == name_norm for alias in aliases):
            return node
    return None


def _node_label(G: nx.DiGraph, node: str) -> str:
    """获取节点显示标签"""
    data = G.nodes.get(node, {})
    return data.get("name") or data.get("title") or node


def _format_graph_stats(G: nx.DiGraph) -> str:
    """格式化图谱统计信息"""
    if G.number_of_nodes() == 0:
        return "知识图谱为空。使用 add_paper 或 add_concept 添加节点。"

    # 节点类型统计
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # 边类型统计
    edge_counts = {}
    for _, _, data in G.edges(data=True):
        t = data.get("type", "Unknown")
        edge_counts[t] = edge_counts.get(t, 0) + 1

    lines = [
        f"## 知识图谱统计",
        f"",
        f"节点总数: {G.number_of_nodes()}",
        f"边总数: {G.number_of_edges()}",
        f"",
        f"### 节点类型分布",
    ]
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {count}")

    lines.append(f"\n### 边类型分布")
    for t, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {count}")

    return "\n".join(lines)


# ── 路由 ──────────────────────────────────────────────────────────────────

ACTION_MAP = {
    "add_paper": _action_add_paper,
    "add_concept": _action_add_concept,
    "add_relation": _action_add_relation,
    "extract_from_abstract": _action_extract_from_abstract,
    "normalize": _action_normalize,
    "query": _action_query,
    "find_path": _action_find_path,
    "stats": _action_stats,
}


async def handle_knowledge_graph(args: dict, user_id: str = None) -> str:
    """知识图谱工具入口"""
    action = args.get("action")
    if not action or action not in ACTION_MAP:
        return fail("knowledge_graph", f"未知的 action '{action}'。支持: {', '.join(ACTION_MAP.keys())}")

    G = _load_graph(user_id)
    handler = ACTION_MAP[action]
    result = handler(G, args)

    # Persist to PostgreSQL after mutating actions
    if action in _MUTATING_ACTIONS:
        await save_to_pg(user_id, G)

    return result


async def extract_from_abstract_sync(paper_id: str, entities: list[dict] | None = None, user_id: str = None) -> str:
    """异步版本，供 paper_parse 直接调用。

    返回统一 JSON 字符串 (ok/fail)。调用方可以 json.loads() 获取结构化数据。
    """
    G = _load_graph(user_id)
    args = {"paper_id": paper_id}
    if entities:
        args["entities"] = entities
    result = await _action_extract_from_abstract(G, args)
    await save_to_pg(user_id, G)
    return result
