"""tests/test_dialect_types.py — JSONB 和 GUID 方言测试

验证 JSON_TYPE 和 GUID 在不同方言下的编译行为。
不需要连接真实数据库。
"""

import uuid
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

from web.backend.db.models import JSON_TYPE, GUID


class TestJsonTypeDialect:
    """JSON_TYPE 在不同方言下的编译行为。"""

    def test_postgresql_compiles_to_jsonb(self):
        """PostgreSQL dialect 下 JSON_TYPE 编译为 JSONB。"""
        from sqlalchemy.dialects.postgresql import dialect as pg_dialect
        compiled = JSON_TYPE.compile(dialect=pg_dialect())
        assert "JSONB" in compiled.upper() or "jsonb" in compiled.lower()

    def test_sqlite_compiles_to_json(self):
        """SQLite dialect 下 JSON_TYPE 编译为 JSON。"""
        compiled = JSON_TYPE.compile(dialect=sqlite_dialect())
        assert "JSON" in compiled.upper()

    def test_json_type_is_compilable(self):
        """JSON_TYPE 可以被编译。"""
        from sqlalchemy.dialects.postgresql import dialect as pg_dialect
        from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
        # 验证可以编译（不检查具体属性）
        pg_compiled = JSON_TYPE.compile(dialect=pg_dialect())
        sqlite_compiled = JSON_TYPE.compile(dialect=sqlite_dialect())
        assert pg_compiled is not None
        assert sqlite_compiled is not None


class TestGuidTypeDialect:
    """GUID 在不同方言下的绑定和结果处理。"""

    def test_postgresql_bind_returns_uuid(self):
        """PostgreSQL bind processor 返回 uuid.UUID。"""
        guid = GUID()
        test_uuid = uuid.uuid4()
        pg_dialect = PGDialect_asyncpg()
        result = guid.process_bind_param(test_uuid, pg_dialect)
        assert isinstance(result, uuid.UUID)
        assert result == test_uuid

    def test_postgresql_bind_converts_string_to_uuid(self):
        """PostgreSQL bind processor 将字符串转换为 uuid.UUID。"""
        guid = GUID()
        test_uuid = uuid.uuid4()
        pg_dialect = PGDialect_asyncpg()
        result = guid.process_bind_param(str(test_uuid), pg_dialect)
        assert isinstance(result, uuid.UUID)
        assert result == test_uuid

    def test_sqlite_bind_returns_string(self):
        """SQLite bind processor 返回字符串。"""
        guid = GUID()
        test_uuid = uuid.uuid4()
        result = guid.process_bind_param(test_uuid, sqlite_dialect())
        assert isinstance(result, str)
        assert result == str(test_uuid)

    def test_sqlite_bind_converts_uuid_to_string(self):
        """SQLite bind processor 将 uuid.UUID 转换为字符串。"""
        guid = GUID()
        test_uuid = uuid.uuid4()
        result = guid.process_bind_param(test_uuid, sqlite_dialect())
        assert isinstance(result, str)
        assert result == str(test_uuid)

    def test_bind_none_returns_none(self):
        """None 输入返回 None。"""
        guid = GUID()
        assert guid.process_bind_param(None, sqlite_dialect()) is None
        assert guid.process_bind_param(None, PGDialect_asyncpg()) is None

    def test_result_value_returns_uuid(self):
        """process_result_value 始终返回 uuid.UUID。"""
        guid = GUID()
        test_uuid = uuid.uuid4()

        # 从字符串恢复
        result = guid.process_result_value(str(test_uuid), sqlite_dialect())
        assert isinstance(result, uuid.UUID)
        assert result == test_uuid

        # 从 uuid.UUID 输入
        result = guid.process_result_value(test_uuid, sqlite_dialect())
        assert isinstance(result, uuid.UUID)
        assert result == test_uuid

    def test_sqlite_crud_roundtrip(self):
        """SQLite 写入并读取后返回 uuid.UUID。"""
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session

        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            # 创建测试表
            conn.execute(text("""
                CREATE TABLE test_guid (
                    id TEXT PRIMARY KEY,
                    value TEXT
                )
            """))
            conn.commit()

            # 写入
            guid = GUID()
            test_uuid = uuid.uuid4()
            bound_value = guid.process_bind_param(test_uuid, sqlite_dialect())
            conn.execute(
                text("INSERT INTO test_guid (id, value) VALUES (:id, :value)"),
                {"id": bound_value, "value": "test"}
            )
            conn.commit()

            # 读取
            result = conn.execute(text("SELECT id FROM test_guid WHERE id = :id"), {"id": bound_value}).fetchone()
            restored = guid.process_result_value(result[0], sqlite_dialect())
            assert isinstance(restored, uuid.UUID)
            assert restored == test_uuid
