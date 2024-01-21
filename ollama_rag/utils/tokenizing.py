from aleph_alpha_client import Client


def get_tokenizer(aleph_alpha_token: str):
    """Initialize the tokenizer."""
    client = Client(token=aleph_alpha_token)
    tokenizer = client.tokenizer("luminous-base-control")
    return tokenizer


def count_tokens(tokenizer: str, text: str):
    """Count the number of tokens in the text.

    Args:
        text (str): The text to count the tokens for.

    Returns:
        int: Number of tokens.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)
