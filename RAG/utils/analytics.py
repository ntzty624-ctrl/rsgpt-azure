import json
import logging

logger = logging.getLogger("rag.analytics")

EVENT_PREFIX = "rag_event "


def log_event(event: str, **fields) -> dict:
    """Emit one structured analytics event as a single log line.

    Lines look like: rag_event {"event": "rag_query", ...}
    In Application Insights (traces table) they can be parsed with:
        traces | where message startswith "rag_event "
               | extend e = parse_json(substring(message, 10))
    """
    payload = {"event": event, **fields}
    logger.info(EVENT_PREFIX + json.dumps(payload, default=str))
    return payload
