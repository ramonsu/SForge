import argparse
import json
import signal

from runtime import AgentApplication


def main(argv=None):
    parser = argparse.ArgumentParser(description="SForge Agent Runtime")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用终端界面，而不是桌面窗口",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="在终端模式中输出开发者 RuntimeSnapshot",
    )
    options = parser.parse_args(argv)
    if not options.cli and not options.inspect:
        try:
            from ui.desktop import launch_desktop

            return launch_desktop()
        except RuntimeError as exc:
            print(f"启动失败：{exc}")
            return 1
    return run_cli(inspect=options.inspect)


def run_cli(*, inspect=False):
    print("Agent 启动中...")
    try:
        application = AgentApplication()
    except RuntimeError as exc:
        print(f"启动失败：{exc}")
        return 1
    print("Agent 已就绪。输入 exit、quit 或 退出 可结束。\n")
    if inspect:
        print("Runtime Inspector 已启用；每次运行后将输出只读快照。\n")

    def show_inspection():
        if inspect:
            snapshot = application.inspect().as_dict()
            print("Runtime Inspector：")
            print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            print()

    def close(*_):
        application.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, close)
    while True:
        try:
            user_input = input("你：")
        except EOFError:
            close()
        if user_input.lower() in {"exit", "quit", "退出"}:
            close()
        try:
            reply = application.handle(user_input)
        except Exception as exc:
            print(f"运行失败：{exc}\n")
            show_inspection()
            continue
        print(f"Agent：{reply}\n")
        show_inspection()


if __name__ == "__main__":
    raise SystemExit(main())
