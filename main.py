import signal

from runtime import AgentApplication


def main():
    print("Agent 启动中...")
    try:
        application = AgentApplication()
    except RuntimeError as exc:
        print(f"启动失败：{exc}")
        return 1
    print("Agent 已就绪。输入 exit、quit 或 退出 可结束。\n")

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
            continue
        print(f"Agent：{reply}\n")


if __name__ == "__main__":
    main()
