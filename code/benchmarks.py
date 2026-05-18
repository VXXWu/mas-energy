"""Benchmark loaders, tool executors, and evaluators for agentic benchmarks.

Each benchmark provides:
    - load_tasks(n) -> list of task dicts
    - get_tools() -> list of OpenAI-format tool schemas
    - make_executor(task) -> (callable(name, args) -> result, cleanup_fn)
    - evaluate(task, executor) -> bool

WorkBench is the primary benchmark for toy/initial experiments.
"""

import ast
import os
import random
import re
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────
# WorkBench
# ─────────────────────────────────────────────────────────

class WorkBenchBenchmark:
    """Adapter for the WorkBench benchmark (olly-styles/WorkBench).

    WorkBench has 690 tasks across 6 domains with 27 tools.
    Data lives in CSVs loaded as pandas DataFrames.
    Evaluation is state-based: compare DataFrame state after agent actions
    vs ground truth actions.

    Expected repo layout at `repo_path`:
        WorkBench/
        ├── data/
        │   ├── processed/
        │   │   ├── calendar_events.csv
        │   │   ├── emails.csv
        │   │   ├── analytics_data.csv
        │   │   ├── project_tasks.csv
        │   │   ├── customer_relationship_manager_data.csv
        │   │   └── queries_and_answers/
        │   │       ├── calendar_queries_and_answers.csv
        │   │       ├── email_queries_and_answers.csv
        │   │       ├── analytics_queries_and_answers.csv
        │   │       ├── project_management_queries_and_answers.csv
        │   │       ├── customer_relationship_manager_queries_and_answers.csv
        │   │       └── multi_domain_queries_and_answers.csv
        │   └── raw/
        │       └── email_addresses.csv
    """

    CURRENT_TIME = "2023-11-30 00:00:00"
    CURRENT_DATE = "2023-11-30"
    TIME_CONTEXT = (
        "Today's date is Thursday, 2023-11-30 and the current time is 00:00:00. "
        "Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm."
    )

    TASK_CSV_FILES = [
        "calendar_queries_and_answers.csv",
        "email_queries_and_answers.csv",
        "analytics_queries_and_answers.csv",
        "project_management_queries_and_answers.csv",
        "customer_relationship_manager_queries_and_answers.csv",
        "multi_domain_queries_and_answers.csv",
    ]

    DOMAIN_PREFIXES = {
        "calendar_queries_and_answers.csv": "calendar",
        "email_queries_and_answers.csv": "email",
        "analytics_queries_and_answers.csv": "analytics",
        "project_management_queries_and_answers.csv": "project_management",
        "customer_relationship_manager_queries_and_answers.csv": "customer_relationship_manager",
        "multi_domain_queries_and_answers.csv": "multi_domain",
    }

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.data_dir = self.repo_path / "data" / "processed"
        self.raw_dir = self.repo_path / "data" / "raw"
        self.qa_dir = self.data_dir / "queries_and_answers"
        self._validate_repo()

    def _validate_repo(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"WorkBench repo not found at {self.repo_path}. "
                f"Clone it: git clone https://github.com/olly-styles/WorkBench.git"
            )
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"WorkBench data/processed/ not found at {self.data_dir}."
            )
        required_csvs = [
            "calendar_events.csv", "emails.csv", "analytics_data.csv",
            "project_tasks.csv", "customer_relationship_manager_data.csv",
        ]
        for csv_name in required_csvs:
            if not (self.data_dir / csv_name).exists():
                raise FileNotFoundError(
                    f"Required data file missing: {self.data_dir / csv_name}"
                )

    def load_tasks(self, n_tasks=None, seed=42):
        """Load WorkBench tasks from CSV files.

        Each row becomes a task dict with id, question, ground_truth_actions,
        domains, and base_template.
        """
        all_tasks = []
        for csv_name in self.TASK_CSV_FILES:
            csv_path = self.qa_dir / csv_name
            if not csv_path.exists():
                continue
            domain_prefix = self.DOMAIN_PREFIXES[csv_name]
            df = pd.read_csv(csv_path, dtype=str)
            for row_idx, row in df.iterrows():
                all_tasks.append({
                    "id": f"{domain_prefix}_{row_idx}",
                    "question": row["query"],
                    "ground_truth_actions": ast.literal_eval(row["answer"]),
                    "domains": ast.literal_eval(row["domains"]),
                    "base_template": row["base_template"],
                })

        if n_tasks and n_tasks < len(all_tasks):
            rng = random.Random(seed)
            rng.shuffle(all_tasks)
            all_tasks = all_tasks[:n_tasks]

        return all_tasks

    def get_tools(self):
        """Return 27 WorkBench tool schemas in OpenAI function calling format."""
        return WORKBENCH_TOOLS

    def make_executor(self, task=None):
        """Create a fresh WorkBenchExecutor with independent DataFrame copies.

        Returns (executor, cleanup_fn). cleanup is a no-op since state is in-memory.
        """
        executor = WorkBenchExecutor(self.data_dir, self.raw_dir)
        return executor, lambda: None

    def evaluate(self, task, executor):
        """State-based evaluation matching WorkBench's is_correct().

        1. Get executor's current DataFrames (predicted state from agent actions).
        2. Create a fresh executor, run ground truth actions, capture state.
        3. Lowercase all string columns except status, list_name, board.
        4. Compare with DataFrame.equals().
        """
        gt_actions = task.get("ground_truth_actions", [])

        # Get predicted state
        pred_state = executor.get_state()

        # Get ground truth state: fresh executor + ground truth actions
        gt_executor = WorkBenchExecutor(self.data_dir, self.raw_dir)
        for action_str in gt_actions:
            try:
                tool_name, args = parse_workbench_action(action_str)
                gt_executor(tool_name, args)
            except Exception:
                continue
        gt_state = gt_executor.get_state()

        # Compare states with case-insensitive strings (except certain fields)
        fields_not_to_convert = {"status", "list_name", "board"}
        for key in pred_state:
            if key not in gt_state:
                return False
            pred_df = _lowercase_except(pred_state[key], fields_not_to_convert)
            gt_df = _lowercase_except(gt_state[key], fields_not_to_convert)
            if not pred_df.equals(gt_df):
                return False
        return True


def _lowercase_except(df, keep_fields):
    """Lowercase all string columns except those in keep_fields."""
    df = df.copy()
    for col in df.columns:
        if col not in keep_fields and df[col].dtype == object:
            df[col] = df[col].str.lower()
    return df


class ToolCallRecorder:
    """Wraps an executor to record all tool calls for sequence-matching evaluation."""

    def __init__(self, executor):
        self.executor = executor
        self.calls = []

    def __call__(self, tool_name, args):
        self.calls.append((tool_name, dict(args)))
        return self.executor(tool_name, args)

    def get_state(self):
        """Proxy to underlying executor for state-based eval."""
        return self.executor.get_state()


def _normalize_tool_name(name):
    """Normalize tool names: underscore → dot for consistent comparison."""
    # LLMs may emit 'calendar_delete_event' instead of 'calendar.delete_event'
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0] in (
        "calendar", "email", "analytics", "project", "customer", "company",
    ):
        # Handle 'project_management' and 'customer_relationship_manager' prefixes
        if parts[0] == "project" and parts[1].startswith("management_"):
            return "project_management." + parts[1][len("management_"):]
        if parts[0] == "customer" and parts[1].startswith("relationship_manager_"):
            return "customer_relationship_manager." + parts[1][len("relationship_manager_"):]
        if parts[0] == "company" and parts[1].startswith("directory_"):
            return "company_directory." + parts[1][len("directory_"):]
        return parts[0] + "." + parts[1]
    return name


def _normalize_arg_value(val):
    """Normalize argument values for comparison: lowercase, strip whitespace,
    normalize date formats."""
    if not isinstance(val, str):
        val = str(val)
    val = val.strip().lower()
    # Normalize ISO datetime separator: '2023-11-30t00:00:00z' → '2023-11-30 00:00:00'
    val = re.sub(r'(\d{4}-\d{2}-\d{2})[t ](\d{2}:\d{2}:\d{2})z?$', r'\1 \2', val)
    # Strip midnight timestamp: '2023-11-30 00:00:00' → '2023-11-30'
    if re.match(r'^\d{4}-\d{2}-\d{2} 00:00:00$', val):
        val = val[:10]
    return val


def _args_match(gold_args, pred_args):
    """Check if predicted args contain all gold args with tolerance."""
    for key, gold_val in gold_args.items():
        if key not in pred_args:
            return False
        if _normalize_arg_value(gold_val) != _normalize_arg_value(pred_args[key]):
            return False
    return True


def evaluate_sequence(task, recorder):
    """Sequence-matching evaluation per Kim et al.

    Checks whether all gold actions appear somewhere in the recorded tool calls.
    Gold ⊆ predicted (as a set). Extra exploratory calls are ignored.
    Argument matching uses case/date normalization.
    """
    gt_actions = task.get("ground_truth_actions", [])
    recorded = recorder.calls

    for action_str in gt_actions:
        try:
            gt_name, gt_args = parse_workbench_action(action_str)
        except ValueError:
            return False

        gt_name_norm = _normalize_tool_name(gt_name)

        found = False
        for pred_name, pred_args in recorded:
            pred_name_norm = _normalize_tool_name(pred_name)
            if gt_name_norm == pred_name_norm and _args_match(gt_args, pred_args):
                found = True
                break

        if not found:
            return False

    return True


def parse_workbench_action(action_str):
    """Parse a WorkBench ground truth action string into (tool_name, kwargs).

    Examples:
        'calendar.delete_event.func(event_id="00000256")'
        -> ("calendar.delete_event", {"event_id": "00000256"})

        'email.send_email.func(recipient="a@b.com", subject="Hi", body="Hello")'
        -> ("email.send_email", {"recipient": "a@b.com", "subject": "Hi", "body": "Hello"})
    """
    # Format: domain.tool_name.func(key="val", ...)
    match = re.match(r'^([\w.]+)\.func\((.*)\)$', action_str, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot parse action: {action_str}")
    tool_name = match.group(1)
    args_str = match.group(2)

    # Parse keyword arguments
    kwargs = {}
    if args_str.strip():
        # Use a mini Python eval for safety: construct a dict from the kwargs
        # The args are always key="value" format
        try:
            # Wrap in dict() call and eval
            kwargs = eval(f"dict({args_str})")
        except Exception:
            # Fallback: regex-based parsing
            for m in re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', args_str):
                kwargs[m.group(1)] = m.group(2)

    return tool_name, kwargs


# ─────────────────────────────────────────────────────────
# WorkBench Executor
# ─────────────────────────────────────────────────────────

class WorkBenchExecutor:
    """Executes WorkBench tool calls against in-memory pandas DataFrames.

    Each instance holds independent copies of the 5 domain DataFrames plus
    a plots DataFrame. Tool calls mutate the instance state.
    """

    HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T00:00:00")

    def __init__(self, data_dir, raw_dir=None):
        self.data_dir = Path(data_dir)
        if raw_dir is None:
            raw_dir = self.data_dir.parent / "raw"
        self.raw_dir = Path(raw_dir)

        self.calendar = pd.read_csv(self.data_dir / "calendar_events.csv", dtype=str)
        self.emails = pd.read_csv(self.data_dir / "emails.csv", dtype=str)
        self.analytics = pd.read_csv(self.data_dir / "analytics_data.csv", dtype=str)
        self.analytics["user_engaged"] = self.analytics["user_engaged"] == "True"
        self.project_tasks = pd.read_csv(self.data_dir / "project_tasks.csv", dtype=str)
        self.crm = pd.read_csv(
            self.data_dir / "customer_relationship_manager_data.csv", dtype=str
        )
        self.email_addresses = pd.read_csv(
            self.raw_dir / "email_addresses.csv",
            header=None, names=["email_address"],
        )
        self.plots = pd.DataFrame(columns=["file_path"])

        self._tool_handlers = {
            # Calendar (5)
            "calendar.get_event_information_by_id": self._calendar_get_event_by_id,
            "calendar.search_events": self._calendar_search_events,
            "calendar.create_event": self._calendar_create_event,
            "calendar.delete_event": self._calendar_delete_event,
            "calendar.update_event": self._calendar_update_event,
            # Email (6)
            "email.get_email_information_by_id": self._email_get_by_id,
            "email.search_emails": self._email_search,
            "email.send_email": self._email_send,
            "email.delete_email": self._email_delete,
            "email.forward_email": self._email_forward,
            "email.reply_email": self._email_reply,
            # Analytics (6)
            "analytics.get_visitor_information_by_id": self._analytics_get_visitor,
            "analytics.create_plot": self._analytics_create_plot,
            "analytics.total_visits_count": self._analytics_total_visits,
            "analytics.engaged_users_count": self._analytics_engaged_users,
            "analytics.traffic_source_count": self._analytics_traffic_source,
            "analytics.get_average_session_duration": self._analytics_avg_session,
            # Project Management (5)
            "project_management.get_task_information_by_id": self._pm_get_task_by_id,
            "project_management.search_tasks": self._pm_search_tasks,
            "project_management.create_task": self._pm_create_task,
            "project_management.delete_task": self._pm_delete_task,
            "project_management.update_task": self._pm_update_task,
            # CRM (4)
            "customer_relationship_manager.search_customers": self._crm_search,
            "customer_relationship_manager.update_customer": self._crm_update,
            "customer_relationship_manager.add_customer": self._crm_add,
            "customer_relationship_manager.delete_customer": self._crm_delete,
            # Company Directory (1)
            "company_directory.find_email_address": self._directory_find_email,
        }

        # Underscore-based aliases for OpenAI tool name compatibility
        self._name_map = {}
        for dotted in self._tool_handlers:
            underscored = dotted.replace(".", "_", 1)
            self._name_map[underscored] = dotted

    def __call__(self, tool_name, args):
        """Dispatch a tool call by name."""
        # Try direct match first, then underscore->dot translation
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            dotted = self._name_map.get(tool_name)
            if dotted:
                handler = self._tool_handlers[dotted]
        if handler is None:
            return f"Unknown tool: {tool_name}"
        return handler(args)

    def get_state(self):
        """Return copies of all 5 domain DataFrames + plots."""
        return {
            "calendar": self.calendar.copy(),
            "emails": self.emails.copy(),
            "analytics": self.plots.copy(),
            "project_tasks": self.project_tasks.copy(),
            "crm": self.crm.copy(),
        }

    # ── Calendar tools ──────────────────────────────────

    def _calendar_get_event_by_id(self, args):
        event_id = args.get("event_id")
        field = args.get("field")
        if not event_id:
            return "Event ID not provided."
        if not field:
            return "Field not provided."
        event = self.calendar[self.calendar["event_id"] == event_id].to_dict(orient="records")
        if event:
            if field in event[0]:
                return {field: event[0][field]}
            else:
                return "Field not found."
        else:
            return "Event not found."

    def _calendar_search_events(self, args):
        query = args.get("query", "")
        time_min = args.get("time_min")
        time_max = args.get("time_max")

        events = self.calendar[
            (self.calendar["event_name"].str.contains(query, case=False))
            | (self.calendar["participant_email"].str.contains(query, case=False))
        ].to_dict(orient="records")

        if time_min:
            events = [e for e in events if pd.Timestamp(e["event_start"]) >= pd.Timestamp(time_min)]
        if time_max:
            events = [e for e in events if pd.Timestamp(e["event_start"]) <= pd.Timestamp(time_max)]
        if events:
            return events[:5]
        else:
            return "No events found."

    def _calendar_create_event(self, args):
        event_name = args.get("event_name")
        participant_email = args.get("participant_email")
        event_start = args.get("event_start")
        duration = args.get("duration")

        if not event_name:
            return "Event name not provided."
        if not participant_email:
            return "Participant email not provided."
        if not event_start:
            return "Event start not provided."
        if not duration:
            return "Event duration not provided."

        participant_email = participant_email.lower()
        event_id = str(int(self.calendar["event_id"].max()) + 1).zfill(8)
        new_event = pd.DataFrame({
            "event_id": [event_id],
            "event_name": [event_name],
            "participant_email": [participant_email],
            "event_start": [event_start],
            "duration": [str(duration)],
        })
        self.calendar = pd.concat([self.calendar, new_event])
        return event_id

    def _calendar_delete_event(self, args):
        event_id = args.get("event_id")
        if not event_id:
            return "Event ID not provided."
        if event_id in self.calendar["event_id"].values:
            self.calendar = self.calendar[self.calendar["event_id"] != event_id]
            return "Event deleted successfully."
        else:
            return "Event not found."

    def _calendar_update_event(self, args):
        event_id = args.get("event_id")
        field = args.get("field")
        new_value = args.get("new_value")

        if not event_id or not field or not new_value:
            return "Event ID, field, or new value not provided."
        if event_id in self.calendar["event_id"].values:
            if field == "participant_email":
                new_value = new_value.lower()
            self.calendar.loc[self.calendar["event_id"] == event_id, field] = new_value
            return "Event updated successfully."
        else:
            return "Event not found."

    # ── Email tools ─────────────────────────────────────

    def _email_get_by_id(self, args):
        email_id = args.get("email_id")
        field = args.get("field")
        if not email_id:
            return "Email ID not provided."
        if not field:
            return "Field not provided."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")
        if email:
            if field in email[0]:
                return {field: email[0][field]}
            else:
                return "Field not found."
        else:
            return "Email not found."

    def _email_search(self, args):
        query = args.get("query", "")
        date_min = args.get("date_min")
        date_max = args.get("date_max")

        query_words = query.lower().split()

        def filter_fn(row):
            combined = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
            return all(word in combined for word in query_words)

        filtered = self.emails.apply(filter_fn, axis=1)
        emails = self.emails[filtered].sort_values("sent_datetime", ascending=False).to_dict(orient="records")

        if date_min:
            emails = [e for e in emails if pd.Timestamp(e["sent_datetime"]).date() >= pd.Timestamp(date_min).date()]
        if date_max:
            emails = [e for e in emails if pd.Timestamp(e["sent_datetime"]).date() <= pd.Timestamp(date_max).date()]
        if len(emails):
            return emails[:5]
        else:
            return "No emails found."

    def _email_send(self, args):
        recipient = args.get("recipient")
        subject = args.get("subject")
        body = args.get("body")
        if not recipient or not subject or not body:
            return "Recipient, subject, or body not provided."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."
        recipient = recipient.lower()

        email_id = str(int(self.emails["email_id"].max()) + 1)
        sent_datetime = self.HARDCODED_CURRENT_TIME
        self.emails.loc[len(self.emails)] = [
            email_id,
            "outbox",
            recipient,
            subject,
            str(sent_datetime),
            body,
        ]
        return "Email sent successfully."

    def _email_delete(self, args):
        email_id = args.get("email_id")
        if not email_id:
            return "Email ID not provided."
        if email_id in self.emails["email_id"].values:
            self.emails = self.emails[self.emails["email_id"] != email_id]
            return "Email deleted successfully."
        else:
            return "Email not found."

    def _email_forward(self, args):
        email_id = args.get("email_id")
        recipient = args.get("recipient")
        if not email_id or not recipient:
            return "Email ID or recipient not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        if "@" not in recipient or "." not in recipient:
            return "Invalid recipient email address."
        recipient = recipient.lower()
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")[0]
        result = self._email_send({
            "recipient": recipient,
            "subject": f"FW: {email['subject']}",
            "body": email["body"],
        })
        return "Email forwarded successfully." if result == "Email sent successfully." else result

    def _email_reply(self, args):
        email_id = args.get("email_id")
        body = args.get("body")
        if not email_id or not body:
            return "Email ID or body not provided."
        if email_id not in self.emails["email_id"].values:
            return "Email not found."
        email = self.emails[self.emails["email_id"] == email_id].to_dict(orient="records")[0]
        result = self._email_send({
            "recipient": email["sender/recipient"],
            "subject": email["subject"],
            "body": body,
        })
        return "Email replied successfully." if result == "Email sent successfully." else result

    # ── Analytics tools ─────────────────────────────────

    def _analytics_get_visitor(self, args):
        visitor_id = args.get("visitor_id")
        if not visitor_id:
            return "Visitor ID not provided."
        visitor_data = self.analytics[self.analytics["visitor_id"] == visitor_id].to_dict(orient="records")
        if visitor_data:
            return visitor_data
        else:
            return "Visitor not found."

    def _analytics_create_plot(self, args):
        time_min = args.get("time_min")
        time_max = args.get("time_max")
        value_to_plot = args.get("value_to_plot")
        plot_type = args.get("plot_type")

        if not time_min:
            return "Start date not provided."
        if not time_max:
            return "End date not provided."
        valid_values = [
            "total_visits", "session_duration_seconds", "user_engaged",
            "visits_direct", "visits_referral", "visits_search_engine", "visits_social_media",
        ]
        if value_to_plot not in valid_values:
            return "Value to plot must be one of 'total_visits', 'session_duration_seconds', 'user_engaged', 'direct', 'referral', 'search engine', 'social media'"
        if plot_type not in ["bar", "line", "scatter", "histogram"]:
            return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"

        file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
        self.plots.loc[len(self.plots)] = [file_path]
        return file_path

    def _analytics_total_visits(self, args):
        time_min = args.get("time_min")
        time_max = args.get("time_max")
        data = self.analytics
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        return data.groupby("date_of_visit").size().to_dict()

    def _analytics_engaged_users(self, args):
        time_min = args.get("time_min")
        time_max = args.get("time_max")
        data = self.analytics.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)
        return data.groupby("date_of_visit").sum()["user_engaged"].to_dict()

    def _analytics_traffic_source(self, args):
        time_min = args.get("time_min")
        time_max = args.get("time_max")
        traffic_source = args.get("traffic_source")
        data = self.analytics.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        if traffic_source:
            data["visits_from_source"] = (data["traffic_source"] == traffic_source).astype(int)
            return data.groupby("date_of_visit").sum()["visits_from_source"].to_dict()
        else:
            return data.groupby("date_of_visit").size().to_dict()

    def _analytics_avg_session(self, args):
        time_min = args.get("time_min")
        time_max = args.get("time_max")
        data = self.analytics.copy()
        if time_min:
            data = data[data["date_of_visit"] >= time_min]
        if time_max:
            data = data[data["date_of_visit"] <= time_max]
        data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
        return (
            data[["date_of_visit", "session_duration_seconds"]]
            .groupby("date_of_visit")
            .mean()["session_duration_seconds"]
            .to_dict()
        )

    # ── Project Management tools ────────────────────────

    def _pm_get_task_by_id(self, args):
        task_id = args.get("task_id")
        field = args.get("field")
        if not task_id:
            return "Task ID not provided."
        if not field:
            return "Field not provided."
        task = self.project_tasks[self.project_tasks["task_id"] == task_id].to_dict(orient="records")
        if task:
            if field in task[0]:
                return {field: task[0][field]}
            else:
                return "Field not found."
        else:
            return "Task not found."

    def _pm_search_tasks(self, args):
        task_name = args.get("task_name")
        assigned_to_email = args.get("assigned_to_email")
        list_name = args.get("list_name")
        due_date = args.get("due_date")
        board = args.get("board")

        if not any([task_name, assigned_to_email, list_name, due_date, board]):
            return "No search parameters provided."
        tasks = self.project_tasks.copy()
        if task_name:
            tasks = tasks[tasks["task_name"].str.contains(task_name, case=False)]
        if assigned_to_email:
            tasks = tasks[tasks["assigned_to_email"].str.contains(assigned_to_email, case=False)]
        if list_name:
            tasks = tasks[tasks["list_name"].str.contains(list_name, case=False)]
        if due_date:
            tasks = tasks[tasks["due_date"].str.contains(due_date, case=False)]
        if board:
            tasks = tasks[tasks["board"].str.contains(board, case=False)]
        return tasks.to_dict(orient="records")

    def _pm_create_task(self, args):
        task_name = args.get("task_name")
        assigned_to_email = args.get("assigned_to_email")
        list_name = args.get("list_name")
        due_date = args.get("due_date")
        board = args.get("board")

        if not all([task_name, assigned_to_email, list_name, due_date, board]):
            return "Missing task details."

        assigned_to_email = assigned_to_email.lower()
        if assigned_to_email not in self.project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."
        if list_name not in ["Backlog", "In Progress", "In Review", "Completed"]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if board not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."

        task_id = str(int(self.project_tasks["task_id"].max()) + 1).zfill(8)
        new_task = pd.DataFrame({
            "task_id": [task_id],
            "task_name": [task_name],
            "assigned_to_email": [assigned_to_email],
            "list_name": [list_name],
            "due_date": [due_date],
            "board": [board],
        })
        self.project_tasks = pd.concat([self.project_tasks, new_task], ignore_index=True)
        return task_id

    def _pm_delete_task(self, args):
        task_id = args.get("task_id")
        if not task_id:
            return "Task ID not provided."
        if task_id in self.project_tasks["task_id"].values:
            self.project_tasks = self.project_tasks[self.project_tasks["task_id"] != task_id]
            return "Task deleted successfully."
        else:
            return "Task not found."

    def _pm_update_task(self, args):
        task_id = args.get("task_id")
        field = args.get("field")
        new_value = args.get("new_value")

        if not task_id or not field or not new_value:
            return "Task ID, field, or new value not provided."
        if field == "assigned_to_email":
            new_value = new_value.lower()
        if field == "board" and new_value not in ["Back end", "Front end", "Design"]:
            return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
        if field == "list_name" and new_value not in ["Backlog", "In Progress", "In Review", "Completed"]:
            return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
        if field == "assigned_to_email" and new_value not in self.project_tasks["assigned_to_email"].str.lower().values:
            return "Assignee email not valid. Please choose from the list of team members."
        if task_id in self.project_tasks["task_id"].values:
            if field in self.project_tasks.columns:
                self.project_tasks.loc[self.project_tasks["task_id"] == task_id, field] = new_value
                return "Task updated successfully."
            else:
                return "Field not valid."
        else:
            return "Task not found."

    # ── CRM tools ───────────────────────────────────────

    def _crm_search(self, args):
        customer_name = args.get("customer_name")
        customer_email = args.get("customer_email")
        product_interest = args.get("product_interest")
        status = args.get("status")
        assigned_to_email = args.get("assigned_to_email")
        last_contact_date_min = args.get("last_contact_date_min")
        last_contact_date_max = args.get("last_contact_date_max")
        follow_up_by_min = args.get("follow_up_by_min")
        follow_up_by_max = args.get("follow_up_by_max")

        if not any([
            customer_name, customer_email, product_interest, status,
            assigned_to_email, last_contact_date_min, last_contact_date_max,
            follow_up_by_min, follow_up_by_max,
        ]):
            return "No search parameters provided. Please provide at least one parameter."

        customers = self.crm.copy()
        if customer_name:
            customers = customers[customers["customer_name"].str.contains(customer_name, case=False)]
        if customer_email:
            customers = customers[customers["customer_email"].str.contains(customer_email, case=False)]
        if product_interest:
            customers = customers[customers["product_interest"].str.contains(product_interest, case=False)]
        if status:
            customers = customers[customers["status"].str.contains(status, case=False)]
        if assigned_to_email:
            customers = customers[customers["assigned_to_email"].str.contains(assigned_to_email, case=False)]
        if last_contact_date_min:
            customers = customers[customers["last_contact_date"] >= last_contact_date_min]
        if last_contact_date_max:
            customers = customers[customers["last_contact_date"] <= last_contact_date_max]
        if follow_up_by_min:
            customers = customers[customers["follow_up_by"] >= follow_up_by_min]
        if follow_up_by_max:
            customers = customers[customers["follow_up_by"] <= follow_up_by_max]
        return customers.to_dict(orient="records")[:5]

    def _crm_update(self, args):
        customer_id = args.get("customer_id")
        field = args.get("field")
        new_value = args.get("new_value")

        if not customer_id or not field or not new_value:
            return "Customer ID, field, or new value not provided."
        if field == "status" and new_value not in ["Qualified", "Won", "Lost", "Lead", "Proposal"]:
            return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"
        if field == "product_interest" and new_value not in ["Software", "Hardware", "Services", "Consulting", "Training"]:
            return "Product interest not valid. Please choose from: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'"
        if field in ("customer_email", "assigned_to_email"):
            new_value = new_value.lower()

        if customer_id in self.crm["customer_id"].values:
            if field in self.crm.columns:
                self.crm.loc[self.crm["customer_id"] == customer_id, field] = new_value
                return "Customer updated successfully."
            else:
                return "Field not valid. Please choose from: 'customer_name', 'assigned_to_email', 'customer_email', 'customer_phone', 'last_contact_date', 'product_interest', 'status', 'notes', 'follow_up_by'"
        else:
            return "Customer not found."

    def _crm_add(self, args):
        customer_name = args.get("customer_name")
        assigned_to_email = args.get("assigned_to_email")
        status = args.get("status")
        customer_email = args.get("customer_email")
        customer_phone = args.get("customer_phone")
        last_contact_date = args.get("last_contact_date")
        product_interest = args.get("product_interest")
        notes = args.get("notes", "")
        follow_up_by = args.get("follow_up_by")

        if not all([customer_name, assigned_to_email, status]):
            return "Please provide all required fields: customer_name, assigned_to_email, status."

        assigned_to_email = assigned_to_email.lower()
        if customer_email:
            customer_email = customer_email.lower()

        new_id = str(int(self.crm["customer_id"].max()) + 1).zfill(8)
        new_customer = pd.DataFrame({
            "customer_id": [new_id],
            "customer_name": [customer_name],
            "customer_email": [customer_email],
            "customer_phone": [customer_phone],
            "last_contact_date": [last_contact_date],
            "product_interest": [product_interest],
            "status": [status],
            "assigned_to_email": [assigned_to_email],
            "notes": [notes],
            "follow_up_by": [follow_up_by],
        })
        self.crm = pd.concat([self.crm, new_customer], ignore_index=True)
        return new_id

    def _crm_delete(self, args):
        customer_id = args.get("customer_id")
        if not customer_id:
            return "Customer ID not provided."
        if customer_id not in self.crm["customer_id"].values:
            return "Customer not found."
        self.crm = self.crm[self.crm["customer_id"] != customer_id]
        return "Customer deleted successfully."

    # ── Company Directory tools ─────────────────────────

    def _directory_find_email(self, args):
        name = args.get("name", "")
        if name == "":
            return "Name not provided."
        name = name.lower()
        matches = self.email_addresses[self.email_addresses["email_address"].str.contains(name)]
        return matches["email_address"].values.tolist()


# ─────────────────────────────────────────────────────────
# WorkBench tool schemas (OpenAI function calling format)
# 27 tools across 6 domains
# ─────────────────────────────────────────────────────────

WORKBENCH_TOOLS = [
    # ── Calendar (5) ────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "calendar.get_event_information_by_id",
            "description": (
                'Returns the event for a given ID.\n\n'
                'Parameters:\n'
                '- event_id: str - 8-digit ID of the event.\n'
                '- field: str - Field to return. Available fields: "event_id", "event_name", "participant_email", "event_start", "duration"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "8-digit ID of the event"},
                    "field": {"type": "string", "description": 'Field to return. Available fields: "event_id", "event_name", "participant_email", "event_start", "duration"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar.search_events",
            "description": (
                "Returns events matching a query. Terms are matched in event_name and participant_email fields. "
                "Returns at most 5 events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to search for in event_name and participant_email fields"},
                    "time_min": {"type": "string", "description": 'Lower bound (inclusive) for event end time. Format: "YYYY-MM-DD HH:MM:SS"'},
                    "time_max": {"type": "string", "description": 'Upper bound (inclusive) for event start time. Format: "YYYY-MM-DD HH:MM:SS"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar.create_event",
            "description": "Creates a new calendar event. Returns the new event ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_name": {"type": "string", "description": "Name of the event"},
                    "participant_email": {"type": "string", "description": "Email of the participant"},
                    "event_start": {"type": "string", "description": 'Start time. Format: "YYYY-MM-DD HH:MM:SS"'},
                    "duration": {"type": "string", "description": "Duration in minutes"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar.delete_event",
            "description": "Deletes a calendar event by its 8-digit ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "8-digit ID of the event to delete"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar.update_event",
            "description": "Updates a field of a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "8-digit ID of the event"},
                    "field": {"type": "string", "description": "Field to update"},
                    "new_value": {"type": "string", "description": "New value for the field"},
                },
            },
        },
    },
    # ── Email (6) ───────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "email.get_email_information_by_id",
            "description": (
                'Retrieves specific details of an email by its ID.\n\n'
                'Parameters:\n'
                '- email_id: str - Unique ID of the email.\n'
                '- field: str - Specific field to return. Available fields: "email_id", "sender", "subject", "sent_date", "body", "inbox/outbox".'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "Unique ID of the email"},
                    "field": {"type": "string", "description": 'Field to return. Available fields: "email_id", "sender", "subject", "sent_date", "body", "inbox/outbox"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email.search_emails",
            "description": (
                "Searches for emails matching the given query across subject, body, or sender fields. "
                "All words in the query must appear in any of these fields. Returns at most 5 emails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query matching terms in subject, body, or sender/recipient fields"},
                    "date_min": {"type": "string", "description": 'Lower date limit (inclusive). Format: "YYYY-MM-DD"'},
                    "date_max": {"type": "string", "description": 'Upper date limit (inclusive). Format: "YYYY-MM-DD"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email.send_email",
            "description": "Sends an email to the specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Email address of the recipient"},
                    "subject": {"type": "string", "description": "Subject line of the email"},
                    "body": {"type": "string", "description": "Body content of the email"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email.delete_email",
            "description": "Deletes an email by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "Unique ID of the email to delete"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email.forward_email",
            "description": "Forwards an email to the specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "Unique ID of the email to forward"},
                    "recipient": {"type": "string", "description": "Email address of the recipient"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email.reply_email",
            "description": "Replies to an email by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "Unique ID of the email to reply to"},
                    "body": {"type": "string", "description": "Body content of the reply"},
                },
            },
        },
    },
    # ── Analytics (6) ───────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analytics.get_visitor_information_by_id",
            "description": "Returns the analytics data for a given visitor ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visitor_id": {"type": "string", "description": "ID of the visitor"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics.create_plot",
            "description": (
                "Plots analytics data for a given time range and value. "
                'Returns the file path of the plot.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string", "description": 'Start date. Format: "YYYY-MM-DD"'},
                    "time_max": {"type": "string", "description": 'End date. Format: "YYYY-MM-DD"'},
                    "value_to_plot": {"type": "string", "description": 'Value to plot. Available: "total_visits", "session_duration_seconds", "user_engaged", "visits_direct", "visits_referral", "visits_search_engine", "visits_social_media"'},
                    "plot_type": {"type": "string", "description": 'Type of plot: "bar", "line", "scatter", or "histogram"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics.total_visits_count",
            "description": "Returns the total number of visits per day within a specified time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string", "description": 'Start date. Format: "YYYY-MM-DD"'},
                    "time_max": {"type": "string", "description": 'End date. Format: "YYYY-MM-DD"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics.engaged_users_count",
            "description": "Returns the number of engaged users per day within a specified time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string", "description": 'Start date. Format: "YYYY-MM-DD"'},
                    "time_max": {"type": "string", "description": 'End date. Format: "YYYY-MM-DD"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics.traffic_source_count",
            "description": (
                "Returns the number of visits from a specific traffic source per day within a time range. "
                'Available traffic sources: "direct", "referral", "search engine", "social media"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string", "description": 'Start date. Format: "YYYY-MM-DD"'},
                    "time_max": {"type": "string", "description": 'End date. Format: "YYYY-MM-DD"'},
                    "traffic_source": {"type": "string", "description": 'Traffic source: "direct", "referral", "search engine", "social media"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analytics.get_average_session_duration",
            "description": "Returns the average session duration per day within a specified time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string", "description": 'Start date. Format: "YYYY-MM-DD"'},
                    "time_max": {"type": "string", "description": 'End date. Format: "YYYY-MM-DD"'},
                },
            },
        },
    },
    # ── Project Management (5) ──────────────────────────
    {
        "type": "function",
        "function": {
            "name": "project_management.get_task_information_by_id",
            "description": (
                'Returns the task information for a given ID.\n\n'
                'Parameters:\n'
                '- task_id: str - 8-digit ID of the task.\n'
                '- field: str - Field to return. Available fields: "task_id", "task_name", "assigned_to_email", "list_name", "due_date", "board"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "8-digit ID of the task"},
                    "field": {"type": "string", "description": 'Field to return. Available fields: "task_id", "task_name", "assigned_to_email", "list_name", "due_date", "board"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_management.search_tasks",
            "description": "Searches for tasks based on the given parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {"type": "string", "description": "Name of the task"},
                    "assigned_to_email": {"type": "string", "description": "Email of the person assigned"},
                    "list_name": {"type": "string", "description": "Name of the list (Backlog, In Progress, In Review, Completed)"},
                    "due_date": {"type": "string", "description": 'Due date in "YYYY-MM-DD" format'},
                    "board": {"type": "string", "description": "Name of the board (Back end, Front end, Design)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_management.create_task",
            "description": (
                "Creates a new project task. Returns the new task ID. "
                "All fields are required. "
                'Valid list_name: "Backlog", "In Progress", "In Review", "Completed". '
                'Valid board: "Back end", "Front end", "Design".'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {"type": "string", "description": "Name of the task"},
                    "assigned_to_email": {"type": "string", "description": "Email of the person assigned"},
                    "list_name": {"type": "string", "description": 'List name: "Backlog", "In Progress", "In Review", "Completed"'},
                    "due_date": {"type": "string", "description": 'Due date in "YYYY-MM-DD" format'},
                    "board": {"type": "string", "description": 'Board name: "Back end", "Front end", "Design"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_management.delete_task",
            "description": "Deletes a task by its 8-digit ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "8-digit ID of the task to delete"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_management.update_task",
            "description": (
                "Updates a task by its ID.\n\n"
                'Available fields: "task_name", "assigned_to_email", "list_name", "due_date", "board"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "8-digit ID of the task"},
                    "field": {"type": "string", "description": 'Field to update: "task_name", "assigned_to_email", "list_name", "due_date", "board"'},
                    "new_value": {"type": "string", "description": "New value for the field"},
                },
            },
        },
    },
    # ── CRM (4) ─────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "customer_relationship_manager.search_customers",
            "description": "Searches for customers based on the given parameters. Returns at most 5 records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "Name of the customer"},
                    "customer_email": {"type": "string", "description": "Email address of the customer"},
                    "product_interest": {"type": "string", "description": "Product interest of the customer"},
                    "status": {"type": "string", "description": "Current status of the customer"},
                    "assigned_to_email": {"type": "string", "description": "Email of the person assigned"},
                    "last_contact_date_min": {"type": "string", "description": 'Min last contact date. Format: "YYYY-MM-DD"'},
                    "last_contact_date_max": {"type": "string", "description": 'Max last contact date. Format: "YYYY-MM-DD"'},
                    "follow_up_by_min": {"type": "string", "description": 'Min follow up date. Format: "YYYY-MM-DD"'},
                    "follow_up_by_max": {"type": "string", "description": 'Max follow up date. Format: "YYYY-MM-DD"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "customer_relationship_manager.update_customer",
            "description": (
                "Updates a customer record by ID.\n\n"
                'Available fields: "customer_name", "assigned_to_email", "customer_email", "customer_phone", '
                '"last_contact_date", "product_interest", "status", "notes", "follow_up_by"\n'
                'Valid status: "Qualified", "Won", "Lost", "Lead", "Proposal"\n'
                'Valid product_interest: "Software", "Hardware", "Services", "Consulting", "Training"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "ID of the customer"},
                    "field": {"type": "string", "description": "Field to update"},
                    "new_value": {"type": "string", "description": "New value for the field"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "customer_relationship_manager.add_customer",
            "description": (
                "Adds a new customer record. Returns the new customer ID.\n\n"
                "Required: customer_name, assigned_to_email, status.\n"
                'Valid status: "Qualified", "Won", "Lost", "Lead", "Proposal"\n'
                'Valid product_interest: "Software", "Hardware", "Services", "Consulting", "Training"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "Name of the customer"},
                    "assigned_to_email": {"type": "string", "description": "Email of the person assigned"},
                    "status": {"type": "string", "description": 'Status: "Qualified", "Won", "Lost", "Lead", "Proposal"'},
                    "customer_email": {"type": "string", "description": "Email of the customer"},
                    "customer_phone": {"type": "string", "description": "Phone number of the customer"},
                    "last_contact_date": {"type": "string", "description": 'Last contact date. Format: "YYYY-MM-DD"'},
                    "product_interest": {"type": "string", "description": 'Product interest: "Software", "Hardware", "Services", "Consulting", "Training"'},
                    "notes": {"type": "string", "description": "Notes about the customer"},
                    "follow_up_by": {"type": "string", "description": 'Follow up date. Format: "YYYY-MM-DD"'},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "customer_relationship_manager.delete_customer",
            "description": "Deletes a customer record by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "ID of the customer to delete"},
                },
            },
        },
    },
    # ── Company Directory (1) ───────────────────────────
    {
        "type": "function",
        "function": {
            "name": "company_directory.find_email_address",
            "description": "Finds the email address of an employee by their name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the person"},
                },
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# Benchmark registry
# ─────────────────────────────────────────────────────────

def load_benchmark(name, repo_path=None, **kwargs):
    """Factory for benchmark instances."""
    if name == "fanoutqa":
        from benchmarks_fanoutqa import FanOutQABenchmark
        return FanOutQABenchmark()
    elif name == "workbench":
        if not repo_path:
            repo_path = os.environ.get(
                "WORKBENCH_PATH",
                os.path.expanduser("~/WorkBench"),
            )
        return WorkBenchBenchmark(repo_path)
    elif name == "qampari":
        from benchmarks_qampari import QampariBenchmark
        data_dir = kwargs.get("data_dir")
        return QampariBenchmark(data_dir=data_dir)
    elif name == "maslegalbench":
        from benchmarks_maslegalbench import MASLegalBenchmark
        data_dir = kwargs.get("data_dir")
        return MASLegalBenchmark(data_dir=data_dir)
    elif name == "browsecomp_plus":
        from benchmarks_browsecomp import BrowseCompBenchmark
        data_dir = kwargs.get("data_dir")
        cache_dir = kwargs.get("cache_dir")
        return BrowseCompBenchmark(data_dir=data_dir, cache_dir=cache_dir)
    elif name == "swebench":
        from benchmarks_swebench import SWEBenchBenchmark
        repos_dir = kwargs.get("repos_dir")
        return SWEBenchBenchmark(repos_dir=repos_dir)
    elif name == "swebench_batched":
        from benchmarks_swebench_batched import SWEBenchBatchedBenchmark
        repos_dir = kwargs.get("repos_dir")
        return SWEBenchBatchedBenchmark(repos_dir=repos_dir)
    elif name == "math":
        from benchmarks_math import MATHBenchmark
        levels = kwargs.get("math_levels") or ("Level 5",)
        return MATHBenchmark(levels=tuple(levels))
    elif name == "humaneval":
        from benchmarks_humaneval import HumanEvalBenchmark
        return HumanEvalBenchmark()
    elif name == "livecodebench":
        from benchmarks_livecodebench import LiveCodeBenchBenchmark
        return LiveCodeBenchBenchmark()
    elif name == "predictability":
        from benchmarks_predictability import PredictabilityBenchmark
        return PredictabilityBenchmark()
    elif name == "bigcodebench":
        from benchmarks_bigcodebench import BigCodeBenchBenchmark
        return BigCodeBenchBenchmark()
    elif name == "plancraft":
        raise NotImplementedError(
            "PlanCraft not yet implemented. "
            "Requires plancraft pip package + Gym wrapper."
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")
