"""模型适配器确认的 SDK 失败类型；程序异常不属于提供商失败。"""

import litellm

LLM_PROVIDER_ERRORS = (
    litellm.APIError,
    litellm.OpenAIError,
    litellm.BadGatewayError,
    litellm.APIConnectionError,
    litellm.APIResponseValidationError,
    litellm.BadRequestError,
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.NotFoundError,
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.UnprocessableEntityError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.BudgetExceededError,
)

__all__ = ["LLM_PROVIDER_ERRORS"]
