#***********************************************
#      Filename: search_factory.py
#   Description: 搜索函数工厂类定义
#***********************************************

"""支持自定义搜索后端，默认采用Tavily"""


from __future__ import annotations

import os
import importlib
from typing import Any, Dict, Protocol
from tavily import TavilyClient

from deep_research import logging as dr_logging
from deep_research.utils import load_config

DEFAULT_STAGE = "prod"
logger = dr_logging.get_logger(__name__)


class SearchConfigError(ValueError):
    """自定义搜索Exception class"""


class SearchProvider(Protocol):
    """search providers基类"""

    def build_client(self, provider_cfg: Dict[str, Any]) -> Any:
        ...

    def search(
        self,
        client: Any,
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
        topic: str,
        timeout_seconds: int | None,
    ) -> Any:
        ...

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        ...


# 缓存搜索clients和providers，避免重复构建
_SEARCH_CLIENT_CACHE: Dict[tuple[str, str, str], Any] = {}
_PROVIDER_CACHE: Dict[tuple[str, str], SearchProvider] = {}

# providers注册中心
_PROVIDER_REGISTRY: Dict[str, SearchProvider] = {}


def register_provider(name: str, provider: SearchProvider) -> None:
    """ 注册provider
    """
    _PROVIDER_REGISTRY[name.lower()] = provider


def override_provider(name: str, provider: SearchProvider, *, clear_cache: bool = True) -> None:
    """注册/替换provider，可选清空provider缓存
    """
    normalized = name.lower()
    _PROVIDER_REGISTRY[normalized] = provider

    if not clear_cache:
        return

    # 清空cache
    for key in [k for k in list(_PROVIDER_CACHE) if k[1] == normalized]:
        _PROVIDER_CACHE.pop(key, None)
    for key in [k for k in list(_SEARCH_CLIENT_CACHE) if k[2] == normalized]:
        _SEARCH_CLIENT_CACHE.pop(key, None)


def _resolve_stage(stage: str | None) -> str:
    """解析stage"""
    return stage or os.environ.get("STAGE") or DEFAULT_STAGE


def _load_stage_config(stage: str | None) -> Dict[str, Any]:
    """加载stage默认参数"""
    stage_name = _resolve_stage(stage)
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    cfg = load_config(stage_name=stage_name, config_path=config_path)
    if cfg is None:
        raise SearchConfigError(f"No config found for stage '{stage_name}'")
    return cfg


def _get_search_cfg(stage: str | None) -> Dict[str, Any]:
    """获取搜索配置"""
    cfg = _load_stage_config(stage)
    search_cfg = cfg.get("search") or {}
    if not search_cfg:
        raise SearchConfigError("Missing 'search' configuration block for this stage")
    return search_cfg


class TavilyProvider:
    """TavilyClient作为provider，也是默认值"""

    def build_client(self, provider_cfg: Dict[str, Any]) -> TavilyClient:
        kwargs: Dict[str, Any] = {}
        if provider_cfg.get("api_key"):
            kwargs["api_key"] = provider_cfg["api_key"]

        # 获取base_url
        base_url = provider_cfg.get("api_base_url") or provider_cfg.get("base_url")
        if base_url:
            kwargs["api_base_url"] = base_url

        # 获取timeout
        timeout_seconds = provider_cfg.get("timeout_seconds")
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds

        try:
            return TavilyClient(**kwargs)
        except TypeError:
            kwargs.pop("timeout", None)
            return TavilyClient(**kwargs)

    def search(
        self,
        client: TavilyClient,
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
        topic: str,
        timeout_seconds: int | None,
    ) -> Any:
        """搜索函数核心逻辑"""

        search_kwargs: Dict[str, Any] = {
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "topic": topic,
            "country": "china", # 地区可修改
        }
        if timeout_seconds is not None:
            search_kwargs["timeout"] = timeout_seconds
        return client.search(query, **search_kwargs)

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """搜索client的默认参数"""

        defaults = {
            "max_results": 3,
            "topic": "general",
            "include_raw_content": True,
            "timeout_seconds": None,
        }
        defaults.update({k: provider_cfg.get(k, defaults[k]) for k in defaults})
        return defaults


# 注册tavily client
register_provider("tavily", TavilyProvider())


def _maybe_import_provider(backend: str) -> None:
    """ 实现动态导入机制, 真正需要时才加载, 应用快速启动，用户不用等
    """

    module_candidates = [
        f"deep_research.providers.{backend}",
        f"deep_research_search_{backend}",
        backend,
    ]

    for mod_name in module_candidates:
        try:
            module = importlib.import_module(mod_name)
            logger.debug("Imported module '%s' for backend '%s'", mod_name, backend)
        except Exception as exc:  # pragma: no cover - import guards
            logger.debug("Module import failed for '%s' (backend='%s'): %s", mod_name, backend, exc)
            continue

        # 实现provider自注册
        provider_obj = getattr(module, "PROVIDER", None)
        if provider_obj is not None:
            logger.debug("Registering provider via PROVIDER attribute for backend '%s'", backend)
            register_provider(backend, provider_obj)

        # 或者实现一个显式的注册hook
        register_fn = getattr(module, "register_search_provider", None) or getattr(module, "register_provider", None)
        if callable(register_fn):
            logger.debug("Invoking registration hook in module '%s' for backend '%s'", mod_name, backend)
            register_fn(register_provider)

        if backend.lower() in _PROVIDER_REGISTRY:
            logger.debug("Provider resolved for backend '%s' after importing '%s'", backend, mod_name)
            return

    logger.warning("No provider registered after attempting imports for backend '%s'", backend)


def _get_provider(search_cfg: Dict[str, Any]) -> tuple[str, SearchProvider]:
    """根据config.yml获取search provider"""

    backend = (search_cfg.get("backend") or "tavily").lower()

    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), backend)
    if cache_key in _PROVIDER_CACHE:
        logger.debug("Using cached search provider for backend='%s'", backend)
        return backend, _PROVIDER_CACHE[cache_key]

    provider = _PROVIDER_REGISTRY.get(backend)
    if provider is None:
        logger.info("No registered provider for backend='%s'; attempting dynamic import", backend)
        _maybe_import_provider(backend)
        provider = _PROVIDER_REGISTRY.get(backend)

    if provider is None:
        raise SearchConfigError(
            f"Unsupported search backend '{backend}'. "
            "Ensure the provider module register a provider via override_provider/register_provider."
        )

    _PROVIDER_CACHE[cache_key] = provider
    logger.debug("Search provider resolved and cached for backend='%s'", backend)
    return backend, provider


def get_search_client(*, stage: str | None = None):
    """根据配置的search backend 获取search cleint
    """

    stage_name = _resolve_stage(stage)
    search_cfg = _get_search_cfg(stage_name)
    backend, provider = _get_provider(search_cfg)

    cache_key = (os.environ.get("CONFIG_PATH", "config.yml"), stage_name, backend)
    if cache_key in _SEARCH_CLIENT_CACHE:
        logger.debug("Using cached search client for backend='%s' stage='%s'", backend, stage_name)
        return _SEARCH_CLIENT_CACHE[cache_key]

    backend_cfg = search_cfg.get(backend, {}) if isinstance(search_cfg, dict) else {}
    if not isinstance(backend_cfg, dict):
        raise SearchConfigError(
            f"Search config for backend '{backend}' must be a mapping, got {type(backend_cfg).__name__}"
        )

    logger.info("Building search client for backend='%s' stage='%s'", backend, stage_name)
    try:
        client = provider.build_client(backend_cfg)
    except Exception as exc:
        logger.error("Failed to build search client for backend='%s': %s", backend, exc)
        raise

    _SEARCH_CLIENT_CACHE[cache_key] = client
    return client


def get_search_provider(*, stage: str | None = None) -> SearchProvider:
    """根据config.yml配置的search backend获取providers"""
    search_cfg = _get_search_cfg(stage)
    _, provider = _get_provider(search_cfg)
    return provider


def get_search_defaults(*, stage: str | None = None) -> Dict[str, Any]:
    """根据config.yml配置的search backend获取默认参数"""
    search_cfg = _get_search_cfg(stage)
    backend, provider = _get_provider(search_cfg)
    backend_cfg = search_cfg.get(backend, {}) if isinstance(search_cfg, dict) else {}
    if not isinstance(backend_cfg, dict):
        raise SearchConfigError(
            f"Search config for backend '{backend}' must be a mapping, got {type(backend_cfg).__name__}"
        )
    return provider.defaults(backend_cfg)


def clear_cache() -> None:
    """清空search clients/providers缓存"""
    _SEARCH_CLIENT_CACHE.clear()
    _PROVIDER_CACHE.clear()
