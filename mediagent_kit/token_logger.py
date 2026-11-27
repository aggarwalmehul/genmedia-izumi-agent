import datetime, logging, contextvars
log = logging.getLogger("token_logger")
current_user_email = contextvars.ContextVar("current_user_email", default="unknown")
try:
    from google.cloud import bigquery
    _bq = bigquery.Client(project="ltm-craftstudio-poc")
except Exception as e:
    _bq = None; log.warning("TOKENLOG bq init: %s", e)
_TABLE = "ltm-craftstudio-poc.token_usage.usage"

def log_tokens(platform, model, resp, user_email="unknown"):
    if _bq is None: return
    # IZUMI_TOKEN_ATTRIBUTION_V1: callers that pass no email (the Gemini text/image
    # helpers in media_generation_service) were always logged as "unknown";
    # fall back to the request's user captured by the middleware in main.py.
    if not user_email or user_email == "unknown":
        user_email = current_user_email.get()
    u = getattr(resp, "usage_metadata", None)
    if u is None: return
    try:
        errs = _bq.insert_rows_json(_TABLE, [{
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform, "user_email": user_email, "model": str(model),
            "tokens_in": int(getattr(u,"prompt_token_count",0) or 0),
            "tokens_out": int(getattr(u,"candidates_token_count",0) or 0) + int(getattr(u,"thoughts_token_count",0) or 0),  # IZUMI_THINKING_TOKENS_V1
            "total": int(getattr(u,"total_token_count",0) or 0),
        }])
        if errs: log.warning("TOKENLOG errors: %s", errs)
        else: log.info("TOKENLOG: inserted %s for %s", getattr(u,'total_token_count',0), model)
    except Exception as e:
        log.warning("TOKENLOG exc: %s", e)


_BILLING_TABLE = "ltm-craftstudio-poc.token_usage.billing_units"


def log_billing_units(platform, model, unit_type, quantity, user_email="unknown"):
    """IZUMI_BILLING_UNITS_V1: non-token media billing (video_seconds, music_generations, ...).

    Same row shape Genflow and Creative Studio write to the shared
    token_usage.billing_units table. Never raises: logging must not break a generation."""
    if _bq is None: return
    if not user_email or user_email == "unknown":
        user_email = current_user_email.get()
    try:
        row = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform, "user_email": user_email, "model": str(model),
            "unit_type": unit_type, "quantity": float(quantity),
        }
        errs = _bq.insert_rows_json(_BILLING_TABLE, [row])
        if errs: log.warning("BILLINGUNITS errors: %s", errs)
        else: log.info("BILLINGUNITS: inserted %s %s for %s", quantity, unit_type, model)
    except Exception as e:
        log.warning("BILLINGUNITS exc: %s", e)


def adk_token_callback(callback_context=None, llm_response=None):
    """ADK after_model_callback — logs agent token usage (model-aware, dedup partials)."""
    try:
        # Skip streaming partials to avoid double-counting
        if llm_response is not None and getattr(llm_response, "partial", False):
            return None
        u = getattr(llm_response, "usage_metadata", None)
        if u is None:
            return None
        # Extract model name from response, then agent
        model = "unknown"
        for attr in ("model_version", "model"):
            v = getattr(llm_response, attr, None)
            if v:
                model = str(v); break
        if model == "unknown" and callback_context is not None:
            for getter in (
                lambda c: c._invocation_context.agent.model,
                lambda c: c.agent.model,
            ):
                try:
                    v = getter(callback_context)
                    if v:
                        model = str(v); break
                except Exception:
                    pass
        # Normalize (agent.model may be like "gemini-2.5-flash" already)
        if "/" in model:
            model = model.rsplit("/", 1)[-1]
        log_tokens("izumi-agent", model, llm_response, user_email=current_user_email.get())
    except Exception as e:
        log.warning("TOKENLOG adk callback: %s", e)
    return None
