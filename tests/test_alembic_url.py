"""tests/test_alembic_url.py — Alembic URL 验证测试

验证 validate_database_url_for_alembic 函数的行为。
不需要连接真实数据库。
"""

import pytest
from web.backend.db.base import validate_database_url_for_alembic


class TestValidateDatabaseUrlForAlembic:
    """validate_database_url_for_alembic 的 URL 验证逻辑。"""

    def test_postgresql_url_accepted_and_converted(self):
        """postgresql:// 被接受并转换成 postgresql+asyncpg://。"""
        result = validate_database_url_for_alembic(
            "postgresql://user:pass@localhost:5432/testdb"
        )
        assert "postgresql+asyncpg://" in result
        assert "user" in result
        assert "testdb" in result

    def test_postgresql_asyncpg_url_accepted(self):
        """postgresql+asyncpg:// 被直接接受。"""
        result = validate_database_url_for_alembic(
            "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        )
        assert "postgresql+asyncpg://" in result

    def test_postgresql_asyncpg_preserves_credentials(self):
        """postgresql+asyncpg:// URL 保留原始凭据。"""
        url = "postgresql+asyncpg://admin:secret123@db.example.com:5432/prod"
        result = validate_database_url_for_alembic(url)
        assert "admin" in result
        assert "secret123" in result
        assert "db.example.com" in result
        assert "prod" in result

    def test_sqlite_url_rejected(self):
        """SQLite URL 被拒绝。"""
        with pytest.raises(ValueError, match="仅支持 PostgreSQL"):
            validate_database_url_for_alembic("sqlite:///test.db")

    def test_sqlite_memory_url_rejected(self):
        """SQLite 内存 URL 被拒绝。"""
        with pytest.raises(ValueError, match="仅支持 PostgreSQL"):
            validate_database_url_for_alembic("sqlite+aiosqlite:///:memory:")

    def test_mysql_url_rejected(self):
        """MySQL URL 被拒绝。"""
        with pytest.raises(ValueError, match="仅支持 PostgreSQL"):
            validate_database_url_for_alembic("mysql://user:pass@localhost/test")

    def test_empty_url_rejected(self):
        """空 URL 被拒绝并给出明确错误。"""
        with pytest.raises(ValueError, match="DATABASE_URL 环境变量未设置"):
            validate_database_url_for_alembic("")

    def test_missing_database_url_rejected(self):
        """缺少 DATABASE_URL 被拒绝并给出明确错误。"""
        with pytest.raises(ValueError, match="DATABASE_URL 环境变量未设置"):
            validate_database_url_for_alembic("")

    def test_postgresql_with_ssl(self):
        """带 SSL 参数的 PostgreSQL URL 被接受。"""
        result = validate_database_url_for_alembic(
            "postgresql://user:pass@localhost:5432/testdb?sslmode=require"
        )
        assert "postgresql+asyncpg://" in result
