"""Tk desktop adapter for the frontend-neutral SForge RunService."""

from __future__ import annotations

from queue import Empty, Queue
import tkinter as tk
from tkinter import ttk

from ui.contracts import ProgressEvent, ProgressKind, RunStatus
from ui.service import RunService


class DesktopApp:
    BG = "#F4F6FA"
    SURFACE = "#FFFFFF"
    PANEL = "#F8FAFC"
    TEXT = "#172033"
    MUTED = "#667085"
    BORDER = "#DDE3EC"
    PRIMARY = "#4F46E5"
    PRIMARY_ACTIVE = "#4338CA"
    SUCCESS = "#16815D"
    ERROR = "#C24156"

    def __init__(self, root: tk.Tk, service: RunService | None = None):
        self.root = root
        self.service = service or RunService()
        self._progress: Queue[ProgressEvent] = Queue()
        self._unsubscribe = self.service.subscribe(self._progress.put)
        self._active_run_id: str | None = None
        self._closed = False
        self._workflow_ids: dict[str, str | None] = {}

        self.status_text = tk.StringVar(value="已就绪")
        self.state_text = tk.StringVar(value="idle")
        self.workflow_state_text = tk.StringVar(value="尚未进入 Workflow")
        self.profile_text = tk.StringVar(value="未选择")
        self.tool_text = tk.StringVar(value="—")
        self.token_text = tk.StringVar(value="0 输入 · 0 输出 · 0 总计")
        self.workflow_text = tk.StringVar(value="Agent 自主选择")

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._load_workflows()
        self._append_message(
            "SForge",
            "你好。我可以通过 Workflow、Memory 和受控能力帮助你完成任务。",
            "assistant",
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self._drain_progress)

    def _configure_window(self) -> None:
        self.root.title("SForge")
        self.root.geometry("1120x800")
        self.root.minsize(900, 620)
        self.root.configure(bg=self.BG)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=self.BG)
        style.configure("Surface.TFrame", background=self.SURFACE)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=self.SURFACE,
            foreground=self.TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Meta.TLabel",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Value.TLabel",
            background=self.SURFACE,
            foreground=self.TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Primary.TButton",
            background=self.PRIMARY,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.PRIMARY_ACTIVE), ("disabled", "#A7A3D9")],
        )
        style.configure(
            "Workflow.TCombobox",
            fieldbackground=self.SURFACE,
            background=self.SURFACE,
            foreground=self.TEXT,
            padding=6,
        )

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=20)
        shell.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="SForge", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Agent runtime workspace",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w")

        workflow_box = ttk.Frame(header, style="App.TFrame")
        workflow_box.grid(row=0, column=2, rowspan=2, sticky="e")
        ttk.Label(
            workflow_box, text="Workflow 建议", style="Subtitle.TLabel"
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.workflow_combo = ttk.Combobox(
            workflow_box,
            textvariable=self.workflow_text,
            state="readonly",
            width=23,
            style="Workflow.TCombobox",
        )
        self.workflow_combo.grid(row=1, column=0, sticky="e", padx=(0, 14))
        self.status_badge = tk.Label(
            workflow_box,
            textvariable=self.status_text,
            bg="#E8F5EF",
            fg=self.SUCCESS,
            padx=12,
            pady=7,
            font=("Segoe UI", 9, "bold"),
        )
        self.status_badge.grid(row=1, column=1, sticky="e")

        body = ttk.Frame(shell, style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        conversation = ttk.Frame(
            body, style="Surface.TFrame", padding=(16, 14)
        )
        conversation.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        conversation.rowconfigure(1, weight=1)
        conversation.columnconfigure(0, weight=1)
        ttk.Label(
            conversation, text="对话", style="PanelTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        chat_frame = ttk.Frame(conversation, style="Surface.TFrame")
        chat_frame.grid(row=1, column=0, sticky="nsew")
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        self.chat = tk.Text(
            chat_frame,
            width=1,
            wrap="word",
            relief="flat",
            borderwidth=0,
            bg=self.SURFACE,
            fg=self.TEXT,
            padx=8,
            pady=8,
            font=("Segoe UI", 11),
            state="disabled",
            cursor="arrow",
        )
        self.chat.grid(row=0, column=0, sticky="nsew")
        chat_scroll = ttk.Scrollbar(
            chat_frame, orient="vertical", command=self.chat.yview
        )
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat.configure(yscrollcommand=chat_scroll.set)
        self.chat.tag_configure(
            "user_name", foreground=self.PRIMARY, font=("Segoe UI", 9, "bold")
        )
        self.chat.tag_configure(
            "assistant_name", foreground=self.SUCCESS, font=("Segoe UI", 9, "bold")
        )
        self.chat.tag_configure(
            "error_name", foreground=self.ERROR, font=("Segoe UI", 9, "bold")
        )
        self.chat.tag_configure(
            "message", foreground=self.TEXT, spacing1=4, spacing3=14
        )

        compose = ttk.Frame(conversation, style="Panel.TFrame", padding=10)
        compose.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        compose.columnconfigure(0, weight=1)
        self.input = tk.Text(
            compose,
            width=1,
            height=3,
            wrap="word",
            relief="flat",
            borderwidth=0,
            bg=self.PANEL,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            font=("Segoe UI", 10),
            padx=4,
            pady=4,
        )
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input.bind("<Control-Return>", self._submit_from_key)
        self.send_button = ttk.Button(
            compose,
            text="发送",
            command=self.submit,
            style="Primary.TButton",
        )
        self.send_button.grid(row=0, column=1, sticky="se")
        ttk.Label(
            compose,
            text="Ctrl + Enter 发送",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        activity = ttk.Frame(body, style="Surface.TFrame", padding=(16, 14))
        activity.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        activity.rowconfigure(7, weight=1)
        activity.columnconfigure(0, weight=1)
        ttk.Label(
            activity, text="运行状态", style="PanelTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))
        self._metric(activity, 1, "运行阶段", self.state_text)
        self._metric(activity, 2, "Workflow 状态", self.workflow_state_text)
        self._metric(activity, 3, "工作角色", self.profile_text)
        self._metric(activity, 4, "当前能力", self.tool_text)
        self._metric(activity, 5, "Token 用量", self.token_text)
        ttk.Separator(activity).grid(
            row=6, column=0, sticky="ew", pady=(14, 10)
        )

        log_frame = ttk.Frame(activity, style="Surface.TFrame")
        log_frame.grid(row=7, column=0, sticky="nsew")
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        ttk.Label(
            log_frame, text="活动", style="PanelTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.activity = tk.Text(
            log_frame,
            width=1,
            wrap="word",
            relief="flat",
            borderwidth=0,
            bg=self.SURFACE,
            fg=self.MUTED,
            font=("Segoe UI", 9),
            state="disabled",
            padx=2,
        )
        self.activity.grid(row=1, column=0, sticky="nsew")
        activity_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.activity.yview
        )
        activity_scroll.grid(row=1, column=1, sticky="ns")
        self.activity.configure(yscrollcommand=activity_scroll.set)

    def _metric(
        self, parent: ttk.Frame, row: int, label: str, value: tk.StringVar
    ) -> None:
        frame = ttk.Frame(parent, style="Surface.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=4)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label, style="Meta.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(frame, textvariable=value, style="Value.TLabel").grid(
            row=0, column=1, sticky="e", padx=(12, 0)
        )

    def _load_workflows(self) -> None:
        self._workflow_ids = {"Agent 自主选择": None}
        friendly = {
            "general_task": "通用任务",
            "novel_writing": "小说写作",
        }
        for workflow in self.service.available_workflows():
            workflow_id = str(workflow["id"])
            label = friendly.get(workflow_id, workflow_id)
            if label in self._workflow_ids:
                label = f"{label} · {workflow_id}"
            self._workflow_ids[label] = workflow_id
        values = tuple(self._workflow_ids)
        self.workflow_combo.configure(values=values)
        self.workflow_text.set(values[0])

    def submit(self) -> None:
        if self._active_run_id is not None:
            return
        request = self.input.get("1.0", "end").strip()
        if not request:
            self.status_text.set("请输入任务")
            return
        self.input.delete("1.0", "end")
        self._append_message("你", request, "user")
        self._clear_activity()
        self.state_text.set("queued")
        selected_workflow = self._workflow_ids[self.workflow_text.get()]
        self.workflow_state_text.set(
            "等待 Agent 申请" if selected_workflow else "尚未进入 Workflow"
        )
        self.profile_text.set("未选择")
        self.tool_text.set("—")
        self.token_text.set("0 输入 · 0 输出 · 0 总计")
        self.status_text.set("正在启动")
        self.status_badge.configure(bg="#EEF2FF", fg=self.PRIMARY)
        self.send_button.state(["disabled"])
        self.workflow_combo.configure(state="disabled")
        try:
            self._active_run_id = self.service.start(
                request, selected_workflow
            )
        except Exception as exc:
            self._append_message("系统", f"无法启动：{exc}", "error")
            self._finish_ui(False)

    def _submit_from_key(self, _event):
        self.submit()
        return "break"

    def _drain_progress(self) -> None:
        if self._closed:
            return
        try:
            while True:
                event = self._progress.get_nowait()
                if event.run_id == self._active_run_id:
                    self._handle_progress(event)
        except Empty:
            pass
        self.root.after(80, self._drain_progress)

    def _handle_progress(self, event: ProgressEvent) -> None:
        self.status_text.set(event.message)
        self._append_activity(event)
        snapshot = self.service.snapshot(event.run_id)
        stage_labels = {
            "queued": "排队中",
            "preparing": "准备中",
            "workflow_admission": "申请认知环境",
            "work_assignment_admission": "建立工作关系",
            "resource_binding_admission": "装载认知资源",
            "context": "装载上下文",
            "reasoning": "分析决策",
            "capability": "执行能力",
            "presentation": "整理回答",
            "completed": "已完成",
            "failed": "失败",
        }
        self.state_text.set(stage_labels.get(snapshot.stage, snapshot.stage))
        self.workflow_state_text.set(
            (
                f"{snapshot.workflow_id} / {snapshot.workflow_state_id or '—'}"
                if snapshot.workflow_id
                else "尚未进入 Workflow"
            )
        )
        self.tool_text.set(snapshot.current_capability or "—")
        resource_labels = []
        if snapshot.work_role_id:
            resource_labels.append(f"职责 {snapshot.work_role_id}")
        if snapshot.cognitive_policy_id:
            resource_labels.append(f"策略 {snapshot.cognitive_policy_id}")
        if snapshot.profession_ids:
            resource_labels.append(
                "专业 " + ", ".join(snapshot.profession_ids)
            )
        self.profile_text.set(" · ".join(resource_labels) or "未选择")
        usage = snapshot.token_usage
        self.token_text.set(
            f"{usage.input_tokens} 输入 · {usage.output_tokens} 输出 · "
            f"{usage.total_tokens} 总计"
        )
        if event.kind is ProgressKind.RUN_COMPLETED:
            self._append_message(
                "SForge", snapshot.answer or "", "assistant"
            )
            self._finish_ui(True)
        elif event.kind is ProgressKind.RUN_FAILED:
            self._append_message(
                "系统", snapshot.error or event.message, "error"
            )
            self._finish_ui(False)

    def _finish_ui(self, success: bool) -> None:
        self.status_text.set("已完成" if success else "运行失败")
        self.status_badge.configure(
            bg="#E8F5EF" if success else "#FDECEF",
            fg=self.SUCCESS if success else self.ERROR,
        )
        self.send_button.state(["!disabled"])
        self.workflow_combo.configure(state="readonly")
        self._active_run_id = None
        self.input.focus_set()

    def _append_message(self, name: str, content: str, role: str) -> None:
        self.chat.configure(state="normal")
        name_tag = f"{role}_name" if role != "assistant" else "assistant_name"
        self.chat.insert("end", f"{name}\n", name_tag)
        self.chat.insert("end", f"{content}\n", "message")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _append_activity(self, event: ProgressEvent) -> None:
        marker = "✓" if event.kind in {
            ProgressKind.CAPABILITY_COMPLETED,
            ProgressKind.ACTION_COMPLETED,
            ProgressKind.RUN_COMPLETED,
        } else "●"
        if event.kind is ProgressKind.RUN_FAILED:
            marker = "!"
        time_text = event.timestamp.astimezone().strftime("%H:%M:%S")
        self.activity.configure(state="normal")
        self.activity.insert("end", f"{marker} {time_text}  {event.message}\n")
        self.activity.configure(state="disabled")
        self.activity.see("end")

    def _clear_activity(self) -> None:
        self.activity.configure(state="normal")
        self.activity.delete("1.0", "end")
        self.activity.configure(state="disabled")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        try:
            self.service.close()
        finally:
            self.root.destroy()


def launch_desktop(service: RunService | None = None) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(f"无法启动桌面界面: {exc}") from exc
    DesktopApp(root, service)
    root.mainloop()
    return 0
