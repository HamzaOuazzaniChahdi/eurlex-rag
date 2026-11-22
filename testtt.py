import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:26b"


def ask_gemma(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


if __name__ == "__main__":
    answer = ask_gemma(
        "Résume en 3 points clés le rôle des considérants dans un règlement européen.",
        system="Tu es un assistant juridique précis, concis et en français.",
    )
    print(answer)