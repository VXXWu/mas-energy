"""Thread/context-local tracking of which agent is currently executing.

Used to route tool calls to per-agent isolated workspaces (e.g., SWE-bench
isolated mode), so that the shared-workspace hidden communication channel
(agents implicitly seeing each other's file edits) can be ablated cleanly.

Set by react_loop() at the start of each agent's loop; read by benchmark
executors that need per-agent state routing.
"""

import contextvars
import re

current_agent_id = contextvars.ContextVar("mas_current_agent_id", default=None)


def extract_agent_idx(agent_id) -> int:
    """Extract the numeric agent index from an agent_id string.

    Examples:
      'worker_2_r3'      -> 2
      'debater_0_init'   -> 0
      'independent_1'    -> 1
      'sas_agent'        -> 0   (fallback)
      'synthesizer'      -> 0   (fallback)
      'orchestrator'     -> 0   (fallback)
      None / ''          -> 0
    """
    if not agent_id:
        return 0
    m = re.search(r'(?:^|_)(\d+)(?=_|$)', str(agent_id))
    return int(m.group(1)) if m else 0
