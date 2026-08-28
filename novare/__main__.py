"""novare/__main__.py — python -m novare 入口"""

import asyncio
import sys


def main():
    args = sys.argv[1:]

    # 子命令路由
    if args and args[0] == "channels":
        _handle_channels(args[1:])
        return

    # 默认：启动交互式 REPL
    from novare.cli import main as repl_main
    asyncio.run(repl_main())


def _handle_channels(args: list[str]):
    """Handle `python -m novare channels ...` subcommands."""
    if not args:
        print("Usage: python -m novare channels <command>")
        print()
        print("Commands:")
        print("  login <channel>  Login to a channel (QR code scan, etc.)")
        print("  status           Show channel status")
        return

    cmd = args[0]

    if cmd == "login":
        _channels_login(args[1:])
    elif cmd == "status":
        _channels_status()
    else:
        print(f"Unknown channels command: {cmd}")
        print("Available: login, status")


def _channels_login(args: list[str]):
    """`python -m novare channels login weixin [--force]`"""
    if not args:
        print("Usage: python -m novare channels login <channel> [--force]")
        print()
        print("Available channels: weixin")
        return

    channel_name = args[0]
    force = "--force" in args

    if channel_name == "weixin":
        asyncio.run(_login_weixin(force))
    else:
        print(f"Unknown channel: {channel_name}")
        print("Available channels: weixin")


async def _login_weixin(force: bool):
    """Execute WeChat QR code login."""
    from novare.channels.bus import MessageBus
    from novare.channels.weixin import WeixinChannel, WeixinConfig

    config = WeixinConfig(enabled=True, allow_from=["*"])
    bus = MessageBus()
    channel = WeixinChannel(config, bus)

    if force:
        print("Clearing existing credentials, re-login...")
    else:
        print("Checking existing credentials...")

    # login() 内部会：
    # 1. 检查 account.json 是否有 token → 有则直接返回
    # 2. 没有则打印 QR 码，等待扫码
    # 3. 扫码成功后保存 token 到 account.json
    success = await channel.login(force=force)

    if success:
        state_dir = channel._get_state_dir()
        print(f"\n[OK] WeChat login successful! Token saved to: {state_dir / 'account.json'}")
        print("You can now start the web server; the channel will auto-use the saved token.")
    else:
        print("\n[FAIL] WeChat login failed. Please retry.")
        sys.exit(1)


def _channels_status():
    """`python -m novare channels status`"""
    from novare.channels.weixin import WeixinConfig
    from novare.channels.weixin import _get_runtime_subdir
    from pathlib import Path
    import json

    print("Channel Status:")
    print()

    # Check WeChat token
    state_dir = _get_runtime_subdir("weixin")
    account_file = state_dir / "account.json"
    if account_file.exists():
        try:
            data = json.loads(account_file.read_text())
            token = data.get("token", "")
            if token:
                masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
                print(f"  weixin: [OK] logged in (token={masked})")
                print(f"         state file: {account_file}")
            else:
                print(f"  weixin: [WARN] state file exists but no token")
        except Exception:
            print(f"  weixin: [WARN] state file corrupted")
    else:
        print(f"  weixin: [NOT LOGGED IN]")
        print(f"         Run: python -m novare channels login weixin")


if __name__ == "__main__":
    main()
