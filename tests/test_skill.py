"""tests/test_skill.py — Skill 发现、解析、加载的单元测试"""

import textwrap
from pathlib import Path

import pytest

from novare.skill import Skill, _parse_frontmatter, discover_skills


# ── Frontmatter 解析 ────────────────────────────────────────────────────────


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = textwrap.dedent("""\
            ---
            name: my-skill
            description: A test skill
            ---
            Hello $ARGUMENTS
        """)
        meta, body = _parse_frontmatter(text)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "A test skill"
        assert "Hello $ARGUMENTS" in body

    def test_no_frontmatter(self):
        text = "Just some plain text\nNo frontmatter here"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self):
        text = "---\n---\nBody content"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert "Body content" in body

    def test_frontmatter_with_quotes(self):
        text = '---\nname: test\ndescription: "Quoted value"\n---\nBody'
        meta, body = _parse_frontmatter(text)
        assert meta["description"] == "Quoted value"


# ── Skill 模型 ──────────────────────────────────────────────────────────────


class TestSkill:
    def test_render_replaces_arguments(self):
        skill = Skill(
            name="test",
            description="desc",
            prompt_template="Search for: $ARGUMENTS\nThen summarize.",
            source=Path("/fake"),
        )
        result = skill.render("Transformer models")
        assert result == "Search for: Transformer models\nThen summarize."

    def test_render_no_arguments(self):
        skill = Skill(
            name="test",
            description="desc",
            prompt_template="Do something useful.",
            source=Path("/fake"),
        )
        result = skill.render("")
        assert result == "Do something useful."


# ── Skill 发现 ──────────────────────────────────────────────────────────────


class TestDiscoverSkills:
    def test_discover_flat_md(self, tmp_path):
        """扁平式：.novare/skills/research.md"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "research.md").write_text(textwrap.dedent("""\
            ---
            name: research
            description: Search papers
            ---
            Search for $ARGUMENTS
        """), encoding="utf-8")

        skills = discover_skills([skills_dir])
        assert len(skills) == 1
        assert skills[0].name == "research"
        assert skills[0].description == "Search papers"
        assert "Search for $ARGUMENTS" in skills[0].prompt_template

    def test_discover_directory_skill(self, tmp_path):
        """目录式：.novare/skills/my-skill/SKILL.md"""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            description: My custom skill
            ---
            Do $ARGUMENTS now
        """), encoding="utf-8")

        skills = discover_skills([skills_dir])
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "My custom skill"

    def test_shadowing_first_root_wins(self, tmp_path):
        """同名 skill，先发现的优先"""
        dir1 = tmp_path / "project"
        dir2 = tmp_path / "user"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "research.md").write_text("---\ndescription: Project version\n---\nProject body", encoding="utf-8")
        (dir2 / "research.md").write_text("---\ndescription: User version\n---\nUser body", encoding="utf-8")

        skills = discover_skills([dir1, dir2])
        assert len(skills) == 1
        assert skills[0].description == "Project version"

    def test_empty_body_skipped(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "empty.md").write_text("---\nname: empty\ndescription: Nothing\n---\n", encoding="utf-8")

        skills = discover_skills([skills_dir])
        assert len(skills) == 0

    def test_nonexistent_root_skipped(self, tmp_path):
        skills = discover_skills([tmp_path / "nonexistent"])
        assert skills == []

    def test_multiple_skills_sorted(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "beta.md").write_text("---\ndescription: B\n---\nBody B", encoding="utf-8")
        (skills_dir / "alpha.md").write_text("---\ndescription: A\n---\nBody A", encoding="utf-8")

        skills = discover_skills([skills_dir])
        names = [s.name for s in skills]
        assert names == ["alpha", "beta"]
